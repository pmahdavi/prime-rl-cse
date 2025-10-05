# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PRIME-RL is a framework for large-scale asynchronous reinforcement learning training. It uses a disaggregated architecture with three main components that can be scaled independently:

1. **Trainer** (`src/prime_rl/trainer/`): Handles SFT and RL training using PyTorch FSDP2 for distributed training
2. **Inference Server** (`src/prime_rl/inference/`): vLLM-based inference backend with custom weight update endpoints
3. **Orchestrator** (`src/prime_rl/orchestrator/`): Lightweight CPU process coordinating data flow between trainer and inference

The framework supports asynchronous off-policy training where inference can run k steps ahead of the trainer (default k=2), enabling continuous operation without idle time.

## Key Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync && uv sync --all-extras

# Install pre-commit hooks for development
uv run pre-commit install

# Validate environment
uv run python -c "import flash_attn"
```

### Running Training

All commands use the format: `uv run <entrypoint> @ path/to/config.toml`

**SFT Training:**
```bash
# Single GPU
uv run sft @ configs/debug/sft/train.toml

# Multi-GPU (8 GPUs example)
uv run torchrun --nproc-per-node 8 src/prime_rl/trainer/sft/train.py @ configs/debug/sft/train.toml

# Multi-node (2 nodes example)
# On head node (rank 0):
uv run torchrun --nnodes 2 --node-rank 0 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --nproc-per-node 8 src/prime_rl/trainer/sft/train.py @ config.toml

# On worker node (rank 1):
uv run torchrun --nnodes 2 --node-rank 1 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --nproc-per-node 8 src/prime_rl/trainer/sft/train.py @ config.toml
```

**RL Training:**
```bash
# Single-node (starts all 3 components)
uv run rl \
  --trainer @ configs/debug/rl/train.toml \
  --orchestrator @ configs/debug/orch.toml \
  --inference @ configs/debug/infer.toml

# Manual component startup (for multi-node)
uv run inference @ configs/debug/infer.toml
uv run orchestrator @ configs/debug/orch.toml
uv run trainer @ configs/debug/rl/train.toml

# Specify GPU allocation
uv run rl \
  --trainer @ config.toml \
  --orchestrator @ config.toml \
  --inference @ config.toml \
  --inference-gpu-ids 0,1,2,3,4,5 \
  --trainer-gpu-ids 6,7
```

**Evaluation:**
```bash
# Evaluate checkpoint
uv run eval \
  --model.name <model-name> \
  --environment-ids math500,aime2025 \
  --weights-dir outputs/weights \
  --steps 10,20,30
```

### Testing
```bash
# Run all tests
uv run pytest -v

# Unit tests only
uv run pytest tests/unit -v

# Integration tests only
uv run pytest tests/integration -v

# CPU-only tests (exclude GPU tests)
uv run pytest -v -m "not gpu"

# Run specific test
uv run pytest tests/unit/test_file.py::test_function -v
```

### Code Quality
```bash
# Run linting (ruff handles both formatting and linting)
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Architecture

### Configuration System
Uses `pydantic-settings` with TOML configs. Configuration precedence (highest to lowest):
1. CLI arguments: `--key.subkey value`
2. Config files: `@ path/to/config.toml`
3. Environment variables: `PRIME_KEY__SUBKEY=value`
4. Default values

Example: `uv run trainer @ base.toml @ override.toml --model.name Qwen/Qwen3-8B`

### Async Training Architecture
- Global training step n tags all artifacts
- Trainer produces policy π_n from rollouts (x_n, y_n)
- Inference generates rollouts from policy π_{max(0, n-k)} where k=async_level (default 2)
- Uses AIPO loss objective with importance sampling for off-policy corrections

### Module Structure
```
src/prime_rl/
├── trainer/          # FSDP2-based training
│   ├── rl/          # RL training implementation
│   ├── sft/         # SFT training implementation
│   ├── models/      # Custom model implementations
│   ├── model.py     # Model loading/setup
│   ├── parallel_dims.py  # Parallelism configuration
│   └── weights.py   # Weight management
├── orchestrator/    # Data flow coordination
│   ├── buffer.py    # Rollout buffering
│   ├── batch.py     # Batch assembly
│   └── advantage.py # Advantage computation
├── inference/       # vLLM inference server
├── eval/           # Evaluation harness
└── utils/          # Shared utilities
```

### Checkpointing
- Trainer checkpoints: FSDP shards (DCP), optimizer/scheduler state, progress
- Orchestrator checkpoints: progress counters (step, tokens, samples, problems)
- Inference: stateless, orchestrator reloads weights on resume
- Default location: `checkpoints/step_{step}/`
- Configure with `--ckpt.interval`, `--ckpt.keep`, `--ckpt.resume-step`

### Environments
Uses `verifiers` package for RL tasks. Install environments from the [Environment Hub](https://app.primeintellect.ai/dashboard/environments):
```bash
prime env install <owner>/<name>
# Or local: uv pip install -e path/to/env
```

### Multi-Node Networking
For multi-node setups, configure network interfaces:
```bash
export GLOO_SOCKET_IFNAME=...
export NCCL_SOCKET_IFNAME=...
export MASTER_ADDR=...  # Head node IP
export MASTER_PORT=...  # Typically 29500
```

## Project-Specific Patterns

- **Entry points** are defined in `pyproject.toml` under `[project.scripts]`
- **Config precedence**: Always check if a setting exists in multiple places (top-level RLConfig vs submodule configs)
- **Shared file system required** for multi-node RL training
- **Python 3.12** is required (`requires-python = "~=3.12.0"`)
- **flash-attn** must be built without isolation: `no-build-isolation-package = ["flash-attn"]`
- Uses `uv` as package manager instead of pip/conda
- Ruff is configured to ignore F722/F821 for jaxtyping compatibility

## Development Workflow

1. Make code changes
2. Run `uv run ruff check --fix .` to fix linting issues
3. Run `uv run pytest tests/unit -v` for quick validation
4. For GPU-dependent changes, run `uv run pytest -v -m gpu`
5. Pre-commit hooks will automatically check formatting on commit

## Common Patterns

**Adding a new model:**
- Add model class to `src/prime_rl/trainer/models/`
- Register in model loading logic in `src/prime_rl/trainer/model.py`
- Ensure FSDP compatibility

**Adding a new training objective:**
- Modify loss computation in `src/prime_rl/trainer/rl/`
- Update config in `src/prime_rl/trainer/rl/config.py`

**Debugging multi-node issues:**
- Check `--local-rank-filter 0` is set to see master rank logs only
- Verify network interfaces with `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME`
- Ensure shared file system is accessible from all nodes
