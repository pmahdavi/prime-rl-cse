import os
import warnings
from pathlib import Path
from typing import List

import torch
from huggingface_hub import HfApi
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from prime_rl.trainer.rl.config import HuggingFaceConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import WandbMonitor


class HuggingFaceUploader:
    """A utility class to upload model checkpoints to the HuggingFace Hub."""

    def __init__(
        self,
        config: HuggingFaceConfig,
        monitor: WandbMonitor,
    ):
        self.config = config
        self.logger = get_logger()
        self.monitor = monitor
        self.api = HfApi()

    def upload(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
        optimizers: List[Optimizer] | None = None,
    ) -> None:
        """Uploads the model, tokenizer, and optionally optimizer states to the HuggingFace Hub."""
        if not self.config:
            return

        repo_id = f"{self.config.organization}/{self.config.repo_id}" if self.config.organization else self.config.repo_id

        self.logger.info(f"Attempting to upload model to Hugging Face Hub at {repo_id}...")

        try:
            # Create the repository if it doesn't exist
            self.api.create_repo(
                repo_id=repo_id,
                private=self.config.private,
                exist_ok=True,
            )

            # Save the model and tokenizer to a temporary directory
            temp_dir = Path(f"/tmp/{repo_id}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Gather distributed weights using the same approach as WeightCheckpointManager
            self.logger.debug("Gathering sharded weights for HuggingFace upload")
            
            # Suppress torch.distributed warnings during checkpoint saving
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
                
                cpu_state = {}
                for key, value in model.state_dict().items():
                    if isinstance(value, DTensor):
                        # Convert DTensor to regular tensor by gathering all shards
                        value = value.to(torch.bfloat16)
                        value = value.full_tensor()
                    
                    if torch.distributed.get_rank() == 0:
                        # Clean up the key to remove FSDP prefixes
                        clean_key = key.replace("_fsdp_wrapped_module.", "")
                        cpu_state[clean_key] = value.to("cpu", non_blocking=False)
                
                torch.distributed.barrier()
                
                # Save on rank 0 only
                if torch.distributed.get_rank() == 0:
                    # Save model weights
                    model_path = temp_dir / "pytorch_model.bin"
                    torch.save(cpu_state, model_path)
                    
                    # Save model config and tokenizer
                    model.config.save_pretrained(temp_dir)
                    if hasattr(model, 'generation_config') and model.generation_config:
                        model.generation_config.save_pretrained(temp_dir)
                    tokenizer.save_pretrained(temp_dir)
            
            # Save optimizer states if requested
            if optimizers and self.config.optimizer_save_mode:
                self.logger.info(f"Saving optimizer states using mode: {self.config.optimizer_save_mode}")
                if not self._save_optimizer_states(model, optimizers, temp_dir):
                    self.logger.warning("Failed to save optimizer states, continuing with model upload only")
            
            # Ensure all ranks wait for rank 0 to finish saving
            torch.distributed.barrier()

            # Only rank 0 uploads to HuggingFace Hub
            if torch.distributed.get_rank() == 0:
                # Upload the contents of the temporary directory to the Hub
                self.api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    commit_message=f"Upload model from step {step}",
                )

                repo_url = f"https://huggingface.co/{repo_id}"
                self.logger.success(f"Successfully uploaded model to {repo_url}")
                self.monitor.log({"hf_upload_success": 1, "hf_repo_url": repo_url})

        except Exception as e:
            self.logger.error(f"Failed to upload model to Hugging Face Hub: {e}")
            self.monitor.log({"hf_upload_success": 0}) 
    
    def _gather_optimizer_full(
        self,
        model: PreTrainedModel,
        optimizers: List[Optimizer],
    ) -> dict[str, torch.Tensor] | None:
        """Gather optimizer states using FSDP's full state dict approach."""
        try:
            self.logger.debug("Gathering optimizer states using FSDP full state dict")
            
            # Use StateDictOptions to request full (non-sharded) state dict
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,  # Offload to CPU to save GPU memory
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
                
                # Get full state dict (model and optimizer)
                model_state_dict, optimizer_state_dict = get_state_dict(
                    model, 
                    optimizers,
                    options=options,
                )
                
                # We only need the optimizer state dict
                return optimizer_state_dict
                
        except Exception as e:
            self.logger.error(f"Failed to gather optimizer states using full state dict: {e}")
            return None
    
    def _gather_optimizer_staged(
        self,
        model: PreTrainedModel,
        optimizers: List[Optimizer],
        chunk_size_mb: int = 1024,
    ) -> dict[str, torch.Tensor] | None:
        """Gather optimizer states in stages to handle large models."""
        try:
            self.logger.debug("Gathering optimizer states using staged approach")
            
            # For staged gathering, we'll gather param groups one at a time
            full_optim_state = {"state": {}, "param_groups": []}
            
            for opt_idx, optimizer in enumerate(optimizers):
                optim_state_dict = optimizer.state_dict()
                
                # Handle param groups (these are small, so we can gather them normally)
                if opt_idx == 0:
                    full_optim_state["param_groups"] = optim_state_dict["param_groups"]
                else:
                    full_optim_state["param_groups"].extend(optim_state_dict["param_groups"])
                
                # For the state, we need to gather each parameter's optimizer state
                # This is more complex and would require iterating through each parameter
                # For now, we'll fall back to the full approach
                self.logger.warning("Staged gathering not fully implemented, falling back to full state dict")
                return self._gather_optimizer_full(model, optimizers)
                
        except Exception as e:
            self.logger.error(f"Failed to gather optimizer states using staged approach: {e}")
            return None
    
    def _save_optimizer_states(
        self,
        model: PreTrainedModel,
        optimizers: List[Optimizer],
        temp_dir: Path,
    ) -> bool:
        """Save optimizer states to the temporary directory."""
        if not self.config.optimizer_save_mode or not optimizers:
            return True
            
        try:
            # Gather optimizer states based on the configured mode
            if self.config.optimizer_save_mode == "full":
                optimizer_state_dict = self._gather_optimizer_full(model, optimizers)
            elif self.config.optimizer_save_mode == "staged":
                optimizer_state_dict = self._gather_optimizer_staged(model, optimizers)
            else:
                self.logger.warning(f"Unknown optimizer_save_mode: {self.config.optimizer_save_mode}")
                return False
            
            if optimizer_state_dict is None:
                return False
            
            # Save on rank 0 only
            if torch.distributed.get_rank() == 0:
                optimizer_path = temp_dir / "optimizer.pt"
                self.logger.debug(f"Saving optimizer state to {optimizer_path}")
                torch.save(optimizer_state_dict, optimizer_path)
                
            torch.distributed.barrier()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save optimizer states: {e}")
            return False