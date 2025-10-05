# Modal Deployment for prime-rl

This directory contains scripts and documentation for deploying prime-rl training on [Modal](https://modal.com), a serverless GPU platform.

## 🚀 Quick Start (5 minutes)

### 1. Install Modal
```bash
# In your conda environment
pip install modal

# Authenticate (one-time setup)
modal setup
```

### 2. Set API Keys
Make sure these are in your `~/.zshrc` or `~/.bashrc`:
```bash
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_huggingface_token"
```

### 3. Run Training
```bash
# Simple reverse text experiment (2 GPUs)
modal run modal/deploy.py

# Hendrycks math with 8 GPUs (2 training, 6 inference)
modal run modal/deploy.py \
  --trainer-config configs/hendrycks_math/1b/train.toml \
  --orchestrator-config configs/hendrycks_math/1b/orch.toml \
  --inference-config configs/hendrycks_math/1b/infer.toml \
  --gpu-count 8 \
  --trainer-gpu-ratio 0.25

# LOCAL DEVELOPMENT: Test local changes without pushing to GitHub
modal run modal/deploy.py --use-local-code

# Using the example script
modal run modal/examples/hendrycks_math.py
```

## 📁 Directory Structure

```
modal/
├── README.md              # This file - Getting started guide
├── deploy.py              # Main deployment script
├── mcp_server_v2.py       # 🤖 MCP server for chat-based Modal control
├── mcp_config.json        # Example MCP configuration
├── test_mcp.py            # Test script for MCP server
├── docs/                  # 📚 Documentation
│   ├── SUMMARY.md         # Overview of all improvements
│   ├── DEEP_DIVE.md       # Comprehensive Modal feature guide
│   ├── QUICK_WINS.md      # 30-minute improvements
│   ├── QUICK_REFERENCE.md # Cheat sheet & commands
│   ├── OPTIMIZATIONS.md   # Long-term optimization roadmap
│   └── CHANGELOG.md       # What changed and why
└── examples/
    └── hendrycks_math.py  # Example for specific experiment
```

### 📖 Documentation Guide

**New to Modal?** Start here! ↓
1. **This README** - Basic usage and quick start
2. [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md) - Commands and patterns

**Want to understand what's under the hood?** ↓
3. [`docs/SUMMARY.md`](docs/SUMMARY.md) - Overview of improvements and features
4. [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) - Comprehensive guide to Modal features

**Ready to optimize further?** ↓
5. [`docs/QUICK_WINS.md`](docs/QUICK_WINS.md) - Easy improvements (30 min each)
6. [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) - Advanced optimization roadmap

**Curious what changed?** ↓
7. [`docs/CHANGELOG.md`](docs/CHANGELOG.md) - Complete change history

## ✅ Key Features

- **Serverless GPU**: Pay only for what you use
- **Automatic Setup**: Dependencies, CUDA tools, and models pre-installed
- **Flexible GPU Allocation**: Split GPUs between training and inference
- **Persistent Storage**: Results saved to Modal volumes
- **Cost Effective**: ~$3.70/hour per A100
- **Local Dev Mode**: Test changes instantly without pushing to GitHub
- **Fast Cold Starts**: Memory snapshots reduce startup from 2-5 min → 10-30 sec

## 🛠️ Technical Details

### Image Configuration
We use a PyTorch CUDA development image to ensure compatibility with flash-attn:
```python
modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
```

The build process matches `Dockerfile.cuda` and `scripts/install.sh` for consistency:
- Uses official `uv` installer (not pip) for latest version
- Sets proper environment variables for `uv` caching and bytecode compilation
- Includes monitoring tools (`nvtop`, `htop`, `tmux`)
- Clones repo at build time, pulls latest at runtime for flexibility

### GPU Configuration
Modal's new syntax for GPU specification:
```python
gpu="A100-40GB:8"  # 8x A100 40GB GPUs
```

### Inference Parallelism
When using multiple GPUs for inference, we automatically configure:
```python
--inference.parallel.dp 6  # Data parallel
--inference.parallel.tp 1  # Tensor parallel
```

## 📊 Monitoring

```bash
# View logs
modal app logs

# Monitor GPU usage
modal app stats

# List experiments
modal volume list prime-rl-outputs

# Download results
modal volume get prime-rl-outputs experiment-name ./outputs/experiment-name
```

## 🔧 Advanced Usage

### Local Development Mode

For rapid iteration, use `--use-local-code` to test changes without pushing to GitHub:

```bash
# 1. Make changes to your local code
vim src/prime_rl/trainer/rl/loss.py

# 2. Run immediately on Modal with your changes
modal run modal/deploy.py --use-local-code

# No git commit/push needed!
```

**How it works:**
- Mounts your local `src/` and `configs/` directories to Modal
- Changes sync automatically (no git operations)
- Perfect for debugging and experimentation

**When to use:**
- ✅ Rapid prototyping and testing
- ✅ Debugging specific issues
- ✅ Iterating on algorithms

**When NOT to use:**
- ❌ Production runs (use GitHub for reproducibility)
- ❌ Sharing results with team (code must be in git)
- ❌ Long-running experiments (local machine must stay on)

### Custom GPU Types
```bash
modal run modal/deploy.py --gpu-type H100 --gpu-count 4
```

### Distributed Training (Coming Soon)
```bash
modal run modal/deploy.py --distributed --num-nodes 4
```

## ⚡ Performance Optimizations

### Memory Snapshots (Enabled by Default)
We use Modal's memory snapshot feature to dramatically improve cold start times:

- **First run**: Container initialization takes 2-5 minutes (imports, model loading)
- **Subsequent runs**: Restored from snapshot in 10-30 seconds
- **How it works**: Modal captures container state after initialization and restores it instantly

This is especially valuable when:
- Running multiple experiments back-to-back
- Iterating on training hyperparameters
- Using the same configuration repeatedly

### Additional Optimizations
For more advanced optimizations, see `modal/OPTIMIZATIONS.md`:
- Class-based patterns with `@modal.enter()`
- Keep-warm containers for zero cold starts
- Multi-node training with `@clustered` (beta)

## 🐛 Troubleshooting

1. **Build Errors**: The container build is cached, first run takes ~5-10 minutes
2. **GPU Allocation**: Ensure trainer-gpu-ratio leaves at least 1 GPU for inference
3. **API Keys**: Check that WANDB_API_KEY and HF_TOKEN are set
4. **Results**: Use `modal volume get` to download outputs
5. **Slow Cold Starts**: First run after image rebuild will be slow; subsequent runs use memory snapshots

## 📚 References

- [Modal Documentation](https://modal.com/docs)
- [prime-rl Documentation](../README.md)
- [Modal Pricing](https://modal.com/pricing) 
