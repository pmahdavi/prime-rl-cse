import time

import torch
from torch import nn
from transformers import PretrainedConfig

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class PerfCounter:
    """
    A class to count throughput (tokens/s) with a rolling window to obtain
    precise throughput and MFU estimates.

    Inspired from https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119
    """

    def __init__(self, model: nn.Module, seq_len: int, window_size: int):
        self.window_size = window_size
        self.tokens = []
        self.times = []
        self.model = model

        self._world = get_world()
        self._logger = get_logger()

        if torch.cuda.is_available():
            self.gpu_peak_flops = self._get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
        else:
            self.gpu_peak_flops = 0
        # If not tie_word_embeddings, we exclude the embedding parameters from the total number of parameters
        # If tie_word_embeddings, the embedding parameters are already excluded (shared with the LM head)
        self.num_params = self._get_num_params(model, exclude_embedding=not model.config.tie_word_embeddings)
        self.num_flop_per_token = self._get_num_flop_per_token(self.num_params, model.config, seq_len=seq_len)

    def count_tokens(self, tokens: int):
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])

    def get_mfu(self) -> float | None:
        tokens_per_second = self.get_tokens_per_second()
        if tokens_per_second is None:
            return None
        return 100 * self.num_flop_per_token * tokens_per_second / self.gpu_peak_flops / self._world.world_size

    def _get_peak_flops(self, device_name: str) -> float:
        """
        Peak BF16 FLOPs (without sparsity)

        From: https://github.com/pytorch/torchtitan/blob/05e47c38d99fdb1dd39aeba76f080e529a425c5c/torchtitan/tools/utils.py#L69
        """
        if "A100" in device_name:
            # https://www.nvidia.com/en-us/data-center/a100/
            return 312e12
        if "H100" in device_name or "H200" in device_name:
            # https://www.nvidia.com/en-us/data-center/h100/
            # https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200
            if "NVL" in device_name:
                return 835e12
            elif "PCIe" in device_name:
                return 756e12
            else:  # For H100 SXM and other variants
                return 989e12
        if "B200" in device_name:
            # https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
            return 2.25e15  # This is half of the FLOPS reported in torchtitan
        else:
            self._logger.warning(f"Peak FLOPS undefined for `{device_name}`. Falling back to A100 (312 TFLOPS)")
            return 312e12

    @staticmethod
    def get_active_mm_params(config: PretrainedConfig) -> float:
        """Get number of active parameters per token involved in matmuls"""
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        head_dim = config.head_dim
        num_attention_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers

        ## Attention
        if hasattr(config, "q_lora_rank") and hasattr(config, "kv_lora_rank"):
            # MLA
            q_params = num_hidden_layers * (
                hidden_size * config.q_lora_rank + config.q_lora_rank * num_attention_heads * config.qk_head_dim
            )
            kv_params = num_hidden_layers * (
                hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
                + config.kv_lora_rank * num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)
            )
            o_params = num_hidden_layers * (num_attention_heads * config.v_head_dim * hidden_size)
        else:
            # GQA
            num_key_value_heads = config.num_key_value_heads
            q_params = num_hidden_layers * hidden_size * num_attention_heads * head_dim
            kv_params = 2 * num_hidden_layers * hidden_size * num_key_value_heads * head_dim
            o_params = num_hidden_layers * hidden_size * num_attention_heads * head_dim

        ## MLP
        if hasattr(config, "first_k_dense_replace"):
            num_dense_layers = config.first_k_dense_replace
            num_sparse_layers = config.num_hidden_layers - num_dense_layers
        elif hasattr(config, "num_experts_per_tok"):
            num_dense_layers = 0
            num_sparse_layers = config.num_hidden_layers
        else:
            num_dense_layers = config.num_hidden_layers
            num_sparse_layers = 0

        dense_mlp_params = num_dense_layers * 3 * intermediate_size * hidden_size
        sparse_mlp_params = 0
        if hasattr(config, "num_shared_experts"):  # Shared experts
            sparse_mlp_params += (
                num_sparse_layers * config.num_shared_experts * 3 * config.moe_intermediate_size * hidden_size
            )
        if hasattr(config, "num_experts_per_tok"):  # Routed experts
            sparse_mlp_params += (
                num_sparse_layers * config.num_experts_per_tok * 3 * config.moe_intermediate_size * hidden_size
            )
        if hasattr(config, "n_routed_experts"):  # DeepSeek Router
            sparse_mlp_params += num_sparse_layers * config.n_routed_experts * hidden_size
        elif hasattr(config, "num_experts"):  # Qwen Router
            sparse_mlp_params += num_sparse_layers * config.num_experts * hidden_size
        else:
            sparse_mlp_params = 0

        ## LM Head
        lm_head_params = vocab_size * hidden_size
        ## Total
        return q_params + kv_params + o_params + dense_mlp_params + sparse_mlp_params + lm_head_params

    def _get_num_flop_per_token(self, num_params: int, model_config: PretrainedConfig, seq_len: int) -> int:
        l, h, q, t = (  # noqa: E741
            model_config.num_hidden_layers,
            model_config.num_attention_heads,
            model_config.hidden_size // model_config.num_attention_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        try:
            flop_per_token = 6 * self.get_active_mm_params(model_config) + 12 * l * h * q * t
        except Exception as e:
            self._logger.warning(f"Error calculating flop_per_token using get_active_mm_params: {e}")
            flop_per_token = 6 * num_params + 12 * l * h * q * t

        return flop_per_token

    def _get_num_params(self, model: nn.Module, exclude_embedding: bool = False) -> int:
        num_params = sum(p.numel() for p in model.parameters())
        if exclude_embedding:
            if hasattr(model.lm_head, 'weight'):
                num_params -= model.lm_head.weight.numel()
            elif hasattr(model.lm_head, 'base_layer'):  # LoRALinear
                num_params -= model.lm_head.base_layer.weight.numel()
        return num_params


_PERF_COUNTER: PerfCounter | None = None


def get_perf_counter(model: nn.Module, seq_len: int, window_size: int = 10) -> PerfCounter:
    global _PERF_COUNTER
    if _PERF_COUNTER is None:
        _PERF_COUNTER = PerfCounter(model, seq_len, window_size)

    return _PERF_COUNTER
