# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRIME-RL is a framework for large-scale asynchronous reinforcement learning (RL) training designed to scale from single GPUs to 1000+ GPU clusters. It's built for training language models using RL post-training techniques like PPO.

## Essential Commands

### Development Setup
```bash
# Install dependencies using uv package manager
uv sync && uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Validate environment
uv run python -c "import flash_attn"  # Check flash-attn installed
```

### Running Training

```bash
# Full RL training (requires 2+ GPUs)
uv run rl \
  --trainer @ configs/reverse_text/rl/train.toml \
  --orchestrator @ configs/reverse_text/rl/orch.toml \
  --inference @ configs/reverse_text/rl/infer.toml

# SFT training (requires 1 GPU)
uv run sft @ configs/debug/sft/train.toml

# RL training only (requires 1 GPU, assumes inference server running separately)
uv run trainer @ configs/debug/rl/train.toml

# Inference server (requires 1 GPU)
uv run inference @ configs/debug/infer.toml

# Orchestrator (no GPU required, connects to running inference server)
uv run orchestrator @ configs/debug/orch.toml

# Evaluation
uv run eval @ configs/debug/eval.toml
```

### Testing

```bash
# Full test suite
uv run pytest -v

# Unit tests only
uv run pytest tests/unit -v

# Integration tests only
uv run pytest tests/integration -v

# CPU-only tests (skip GPU tests)
uv run pytest -v -m "not gpu"

# Fast tests only (skip slow tests)
uv run pytest -v -m "not slow"
```

### Development Utilities

```bash
# View logs in tmux (creates 3-pane layout for trainer/orchestrator/inference)
bash scripts/tmux.sh

# Clean outputs directory
bash scripts/clean.sh

# Lint and format code
uv run ruff --fix .
uv run ruff format .
```

### Modal Deployment

```bash
# Run on Modal serverless infrastructure
modal run modal/deploy.py

# Custom command on Modal
modal run modal/deploy.py --command "uv run rl --trainer @ configs/..."

# Different GPU configuration (via environment variable)
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py
```

## Architecture

### Core Components

PRIME-RL uses a **decoupled multi-process architecture** with three main components:

1. **Trainer** (`src/prime_rl/trainer/`)
   - Handles model weight updates using PyTorch FSDP (Fully Sharded Data Parallelism)
   - Supports both SFT (`sft/`) and RL (`rl/`) training
   - Consumes training batches from disk (written by orchestrator)
   - Saves checkpoints and weight-only snapshots for inference server
   - Entry points: `sft/train.py`, `rl/train.py`

2. **Inference Server** (`src/prime_rl/inference/`)
   - High-performance vLLM-based server for generating model completions
   - Exposes OpenAI-compatible HTTP API
   - Supports dynamic weight loading without server restart
   - Can use tensor parallelism across multiple GPUs
   - Entry point: `server.py`

3. **Orchestrator** (`src/prime_rl/orchestrator/`)
   - Coordinates the RL training loop
   - Samples prompts, requests completions from inference server
   - Calculates rewards using pluggable environment modules
   - Computes advantages (GAE) and prepares training batches
   - Manages on-policy synchronization via async_level
   - Entry point: `orchestrator.py`

### Data Flow

```
Orchestrator → (HTTP) → Inference Server → Completions
     ↓
  Rewards & Advantages
     ↓
Training Batches (saved to disk at outputs/rollouts/step_N/)
     ↓
  Trainer reads batches → Updates weights → Saves to outputs/weights/step_N/
     ↓
Orchestrator detects new weights → (HTTP) → Tells Inference Server to reload
```

### Key Design Patterns

- **Process isolation**: Components run as independent processes, enabling independent scaling
- **File-based data passing**: Large data payloads (training batches, weights) transferred via filesystem
- **HTTP-based control**: Lightweight commands (generate text, load weights) via HTTP API
- **Pydantic configuration**: All config is strongly-typed with `BaseSettings` classes using TOML files
- **Async orchestration**: Orchestrator uses `asyncio` for concurrent I/O operations

### Directory Structure

```
src/prime_rl/
├── rl.py                    # Main entrypoint that launches all components
├── trainer/
│   ├── sft/                 # Supervised fine-tuning
│   ├── rl/                  # RL training (PPO)
│   │   ├── train.py         # Main RL training script
│   │   ├── loss.py          # PPO loss computation
│   │   ├── config.py        # RL trainer configuration
│   │   └── data.py          # Data loading from orchestrator
│   ├── model.py             # Model setup (FSDP, flash-attn)
│   ├── ckpt.py              # Checkpoint saving/loading
│   └── weights.py           # Weight-only checkpoint for inference
├── inference/
│   ├── server.py            # vLLM server with dynamic weight loading
│   └── vllm/                # vLLM integration
├── orchestrator/
│   ├── orchestrator.py      # Main orchestration loop
│   ├── buffer.py            # Prompt buffer management
│   ├── batch.py             # Batch preparation for trainer
│   ├── advantage.py         # GAE computation
│   └── client.py            # HTTP client for inference server
├── eval/                    # Evaluation scripts
└── utils/                   # Shared utilities (logging, config, monitoring)

configs/
├── debug/                   # Fast debug configs (small models, few steps)
├── reverse_text/            # Example: text reversal task
└── [other_tasks]/           # Various RL task configurations
```

## Important Implementation Details

### Configuration System

- All components use Pydantic `BaseSettings` with TOML config files
- CLI argument pattern: `uv run <command> @ <config.toml> [--overrides]`
- Shared configs (model name, output dir, wandb, async_level) propagated from `rl.py` to subprocesses
- See `src/prime_rl/utils/pydantic_config.py` for config parsing utilities

### GPU Allocation

- The `rl.py` launcher uses `CUDA_VISIBLE_DEVICES` to partition GPUs between components
- Flags: `--trainer-gpu-ids`, `--inference-gpu-ids`, `--orchestrator-gpu-ids`
- Example: `--trainer-gpu-ids 0 1 --inference-gpu-ids 2 3` assigns first 2 GPUs to trainer, last 2 to inference

### Checkpointing

- **Full checkpoints** (trainer state, optimizer, scheduler): `outputs/ckpt/step_N/`
  - Used for resuming training runs
  - Controlled by `[checkpoint.interval]` and `[checkpoint.resume_step]`
- **Weight-only checkpoints**: `outputs/weights/step_N/`
  - Lightweight, used by inference server to update policy
  - Required for on-policy RL training

### Asynchronous Training

- `async_level` parameter controls staleness of inference policy vs trainer policy
- `async_level = N` means inference server lags trainer by N steps
- Higher async_level = faster throughput but less on-policy
- Orchestrator blocks when it gets too far ahead (`progress.step - ckpt_step > async_level`)

### Environments

- RL tasks are defined in separate packages based on the `verifiers` framework
- Installed via `uv run prime env install <env-name>`
- Examples: `reverse_text`, `wordle`, math reasoning tasks
- Environment provides: prompt sampling, reward calculation, evaluation metrics
- Configured in TOML via `[environment]` section

### Monitoring

- W&B integration for metrics, sample rollouts, and training curves
- Log files written to `outputs/logs/` for each component
- Use `scripts/tmux.sh` to view all logs simultaneously
- Key metrics: loss, reward, advantages, throughput (tokens/sec, TFLOPs)

## Common Workflows

### Adding a New RL Environment

1. Create environment package following `verifiers` framework
2. Install: `uv run prime env install <env-name>`
3. Create new config TOML in `configs/<env-name>/`
4. Set `[environment] id = "<env-name>"` in config
5. Run: `uv run rl --trainer @ configs/<env-name>/train.toml ...`

### Debugging with Persistent Inference Server

To avoid restarting slow-loading inference server during development:

```bash
# Terminal 1: Start inference server (runs continuously)
uv run inference @ configs/reverse_text/rl/infer.toml

# Terminal 2: Run trainer + orchestrator only (omit --inference)
uv run rl \
  --trainer @ configs/reverse_text/rl/train.toml \
  --orchestrator @ configs/reverse_text/rl/orch.toml
```

### Modifying Loss Functions

1. Edit loss computation in `src/prime_rl/trainer/rl/loss.py`
2. Update config schema in `src/prime_rl/trainer/rl/config.py` (e.g., `LossConfig`)
3. Add new config option to TOML file
4. Run tests: `uv run pytest tests/unit/trainer/ -v`

### Multi-Node Training

For trainer (using torchrun):
```bash
# On all nodes, set master node address
export RDZV_ENDPOINT="master-node-ip:29500"

# On each node, run with appropriate node_rank
uv run torchrun \
  --nnodes=4 \
  --node-rank=$NODE_RANK \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$RDZV_ENDPOINT \
  -m prime_rl.trainer.rl.train @ configs/...
```

For inference (using vLLM data parallelism):
- See detailed instructions in README.md
- Requires setting `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `DATA_PARALLEL_ADDRESS`

## Code Quality Standards

- **Linting**: `ruff` with F (pyflakes) and I (isort) rules
- **Line length**: 120 characters
- **Type hints**: Encouraged, especially for public APIs
- **Testing**: All new features should have unit tests in `tests/unit/`
- **Pre-commit hooks**: Automatically run ruff before commits

## Package Management

- Uses `uv` as package manager (faster than pip)
- Dependencies defined in `pyproject.toml`
- Lock file: `uv.lock` (commit this)
- Custom indices: PyTorch (test/cu128), PrimeIntellect environments
- No build isolation for `flash-attn` (set in `[tool.uv]`)

## Performance Considerations

- vLLM uses PagedAttention for high-throughput inference
- Trainer uses FlashAttention-2 for memory-efficient attention
- FSDP enables training models larger than single-GPU memory
- Liger Kernel for optimized ops
- Benchmark modes available: `--bench` flag on `rl`, `trainer`, or `orchestrator`

## File Naming Conventions

- Config files: `<component>.toml` (e.g., `train.toml`, `infer.toml`, `orch.toml`)
- Python modules: snake_case
- Classes: PascalCase
- Config classes: Suffix with `Config` (e.g., `TrainerConfig`, `LossConfig`)

## Entry Points

Defined in `pyproject.toml [project.scripts]`:
- `rl`: Main launcher for full RL run
- `trainer`: RL trainer only
- `sft`: SFT trainer only
- `orchestrator`: Orchestrator only
- `inference`: Inference server only
- `eval`: Evaluation script
