# Modal Deployment for PRIME-RL

Run PRIME-RL training on [Modal](https://modal.com) serverless GPU infrastructure with a single command.

## Quick Start

```bash
# Run with default settings (reverse_text RL training with W&B tracking)
modal run modal/deploy.py

# Run custom command
modal run modal/deploy.py --command "uv run rl --trainer @ configs/..."
```

## Prerequisites

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Set up Modal account**:
   ```bash
   modal setup
   ```

3. **Create Modal secrets** (required for W&B and HuggingFace):
   ```bash
   # Create wandb secret
   modal secret create wandb WANDB_API_KEY=your_wandb_key_here

   # Create huggingface secret
   modal secret create huggingface HF_TOKEN=your_hf_token_here
   ```

   Or create them via the web UI:
   - https://modal.com/secrets

## Usage Examples

### Basic RL Training

```bash
# Run with defaults (reverse_text, 2 GPUs: 1 trainer + 1 inference)
modal run modal/deploy.py
```

### Custom Command

```bash
# Run any command you want
modal run modal/deploy.py --command "uv run rl \
  --trainer @ configs/reverse_text/rl/train.toml \
  --orchestrator @ configs/reverse_text/rl/orch.toml \
  --inference @ configs/reverse_text/rl/infer.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 1"
```

### SFT Training

```bash
modal run modal/deploy.py --command "uv run sft @ configs/debug/sft/train.toml"
```

### Evaluation

```bash
modal run modal/deploy.py --command "uv run eval \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT \
  --environment-ids reverse_text"
```

### Custom Experiment Name

```bash
modal run modal/deploy.py \
  --command "uv run rl --trainer @ configs/..." \
  --experiment-name "my-custom-experiment"
```

### Different GPU Types

Set the `MODAL_GPU_CONFIG` environment variable to change GPU configuration:

```bash
# Use 8 H100 GPUs
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py

# Use 4 A100-80GB GPUs
MODAL_GPU_CONFIG="A100-80GB:4" modal run modal/deploy.py

# Default: H100:4
modal run modal/deploy.py
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--command` | str | (reverse_text RL) | Command to run on Modal |
| `--experiment-name` | str | auto-generated | Name for this experiment |
| `--download-results` | bool | True | Show download instructions |

## GPU Configuration

Set the `MODAL_GPU_CONFIG` environment variable to change GPU allocation:

```bash
# Examples
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py
MODAL_GPU_CONFIG="A100-80GB:4" modal run modal/deploy.py
MODAL_GPU_CONFIG="L4:2" modal run modal/deploy.py

# Default if not set: H100:4
modal run modal/deploy.py
```

## Outputs

All outputs are saved to Modal volumes:

- **Logs**: Training, orchestrator, and inference logs
- **Rollouts**: Generated rollout data for each step
- **Weights**: Model checkpoints at specified intervals
- **Checkpoints**: Full training state for resumption

### Downloading Results

```bash
# Download entire experiment
modal volume get prime-rl-outputs <experiment-name> ./outputs/<experiment-name>

# List experiments
modal volume ls prime-rl-outputs

# List files in an experiment
modal volume ls prime-rl-outputs/<experiment-name>
```

## Monitoring

```bash
# View real-time logs
modal app logs

# Monitor GPU usage
modal app stats

# View on W&B (if enabled)
# Go to: https://wandb.ai/<your-username>/prime-rl
```

## Architecture

The deployment uses Modal 1.0 API with:

- **Base Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **Package Manager**: `uv` for fast dependency installation
- **Secrets**: Automatic injection of W&B and HuggingFace tokens
- **Volumes**: Persistent storage for outputs, checkpoints, and cache
- **Code Mounting**: Your local `src/` and `configs/` are mounted at runtime for fast iteration

### Image Build Process

1. Install system dependencies (git, curl, build tools)
2. Install `uv` package manager
3. Copy and install Python dependencies from `pyproject.toml` and `uv.lock`
4. Set up virtual environment at `/app/.venv`
5. Mount local source code and configs at runtime (for fast iteration)

## W&B Integration

By default, the script enables W&B tracking:

- **Project**: `prime-rl`
- **Run name**: Based on experiment type (e.g., `reverse-text-modal`)

To disable W&B, remove the `--wandb.*` flags from the default command in `deploy.py`.

## Cost Estimation

Modal GPU pricing (as of 2025):

| GPU Type | Cost/GPU/Hour | Example: 4 GPUs | Example: 8 GPUs |
|----------|---------------|-----------------|-----------------|
| T4 | $0.59 | $2.36/hour | $4.72/hour |
| L4 | $0.80 | $3.20/hour | $6.40/hour |
| A10 | $1.10 | $4.40/hour | $8.80/hour |
| L40S | $1.95 | $7.80/hour | $15.60/hour |
| A100-40GB | $2.10 | $8.40/hour | $16.80/hour |
| A100-80GB | $2.50 | $10.00/hour | $20.00/hour |
| H100 | $3.95 | $15.80/hour | $31.60/hour |
| H200 | $4.54 | $18.16/hour | $36.32/hour |
| B200 | $6.25 | $25.00/hour | $50.00/hour |

## Troubleshooting

### Build Issues

If you get dependency errors:
```bash
# Update lockfile locally first
uv lock

# Then deploy
modal run modal/deploy.py
```

### Secret Issues

If secrets are missing:
```bash
# List existing secrets
modal secret list

# Recreate if needed
modal secret create wandb WANDB_API_KEY=$WANDB_API_KEY
modal secret create huggingface HF_TOKEN=$HF_TOKEN
```

### Out of Memory

If you hit OOM errors, reduce batch size in your config TOML or allocate more GPUs.

## Advanced Configuration

### Multi-Node Training

Multi-node distributed training is not currently supported in this simple deployment script. For multi-node training, use the manual setup described in the main PRIME-RL README.

### Custom Docker Image

To use a custom base image, modify the `prime_rl_image` definition in `deploy.py`:

```python
prime_rl_image = (
    modal.Image.from_registry("your/custom:image")
    # ... rest of setup
)
```

## Development Workflow

1. Make changes to your code locally in `src/`
2. Run deployment - your changes are automatically synced
3. No need to rebuild the image unless you change dependencies

**Fast iteration**: Source code is mounted at runtime, so changes to `src/` and `configs/` don't require image rebuilds!

## Resources

- [Modal Documentation](https://modal.com/docs)
- [PRIME-RL Documentation](../README.md)
- [Modal Pricing](https://modal.com/pricing)
