import os
import warnings
from pathlib import Path

import torch
from huggingface_hub import HfApi
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module
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
    ) -> None:
        """Uploads the model and tokenizer to the HuggingFace Hub."""
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