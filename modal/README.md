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

# Using the example script
modal run modal/examples/hendrycks_math.py
```

## 📁 Directory Structure

```
modal/
├── README.md          # This file
├── deploy.py          # Main deployment script
└── examples/
    └── hendrycks_math.py  # Example for specific experiment
```

## ✅ Key Features

- **Serverless GPU**: Pay only for what you use
- **Automatic Setup**: Dependencies, CUDA tools, and models pre-installed
- **Flexible GPU Allocation**: Split GPUs between training and inference
- **Persistent Storage**: Results saved to Modal volumes
- **Cost Effective**: ~$3.70/hour per A100

## 🛠️ Technical Details

### Image Configuration
We use a PyTorch CUDA development image to ensure compatibility with flash-attn:
```python
modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
```

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

### Custom GPU Types
```bash
modal run modal/deploy.py --gpu-type H100 --gpu-count 4
```

### Distributed Training (Coming Soon)
```bash
modal run modal/deploy.py --distributed --num-nodes 4
```

## 🐛 Troubleshooting

1. **Build Errors**: The container build is cached, first run takes ~5-10 minutes
2. **GPU Allocation**: Ensure trainer-gpu-ratio leaves at least 1 GPU for inference
3. **API Keys**: Check that WANDB_API_KEY and HF_TOKEN are set
4. **Results**: Use `modal volume get` to download outputs

## 📚 References

- [Modal Documentation](https://modal.com/docs)
- [prime-rl Documentation](../README.md)
- [Modal Pricing](https://modal.com/pricing) 
