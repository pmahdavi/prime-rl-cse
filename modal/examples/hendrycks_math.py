#!/usr/bin/env python3
"""
Modal deployment for hendrycks_math 1b experiment.
Configured for 8 GPUs total: 2 for training, 6 for inference.
"""

import modal
import os
import sys
from pathlib import Path
from typing import Optional

# Create the Modal app
app = modal.App("prime-rl-hendrycks-test")

# Define persistent volumes for outputs and caches
outputs_volume = modal.Volume.from_name("prime-rl-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("prime-rl-cache", create_if_missing=True)

# Build the container image with all dependencies
# Using PyTorch CUDA development image for flash-attn compilation
prime_rl_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    # Set CUDA environment
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
    })
    # Install system dependencies
    .apt_install([
        "git", 
        "curl", 
        "build-essential",
        "vim",  # For debugging
    ])
    # Install uv package manager
    .run_commands(
        "pip install uv",
    )
    # Clone the repository
    .run_commands(
        "cd /root && git clone https://github.com/pmahdavi/prime-rl-cse.git",
    )
    # Install dependencies with all extras (including flash-attn)
    .run_commands(
        "cd /root/prime-rl-cse && uv sync --all-extras",
    )
    # Pre-download the model to speed up startup (after dependencies are installed)
    .run_commands(
        "cd /root/prime-rl-cse && uv run python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; print('Pre-downloading model: willcb/DeepSeek-R1-Distill-Qwen-1.5B'); AutoTokenizer.from_pretrained('willcb/DeepSeek-R1-Distill-Qwen-1.5B'); AutoModelForCausalLM.from_pretrained('willcb/DeepSeek-R1-Distill-Qwen-1.5B'); print('Model downloaded successfully')\"",
        gpu="any",  # Use any available GPU for model download
    )
    # Set environment variables for runtime
    .env({
        "HF_HUB_CACHE": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "WANDB_DIR": "/outputs/wandb",
        # Pass through API keys if they exist
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    })
)

@app.function(
    image=prime_rl_image,
    gpu="A100-40GB:8",  # 8 total GPUs using new syntax
    cpu=16.0,  # 16 CPU cores for better performance
    memory=65536,  # 64GB RAM
    volumes={
        "/outputs": outputs_volume,
        "/cache": cache_volume,
    },
    timeout=86400,  # 24 hour timeout
)
def train_hendrycks_math(
    experiment_name: str,
    output_subdir: str = "hendrycks-math-1b",
):
    """
    Run hendrycks_math 1b training on Modal with fixed GPU allocation:
    - 2 GPUs for training
    - 6 GPUs for inference
    """
    import subprocess
    import os
    
    # Change to the repository directory
    os.chdir("/root/prime-rl-cse")
    
    # Setup output directory
    output_dir = f"/outputs/{experiment_name}/{output_subdir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command with fixed configuration
    cmd = [
        "uv", "run", "rl",
        "--trainer", "@", "configs/hendrycks_math/1b/train.toml",
        "--orchestrator", "@", "configs/hendrycks_math/1b/orch.toml",
        "--inference", "@", "configs/hendrycks_math/1b/infer.toml",
        "--trainer-gpus", "2",  # Fixed at 2
        "--inference-gpus", "6",  # Fixed at 6
        "--inference.parallel.dp", "6",  # Use 6 data parallel replicas
        "--inference.parallel.tp", "1",  # Keep tensor parallel at 1
        "--output-dir", output_dir,
    ]
    
    # Add W&B tracking
    cmd.extend(["--wandb.project", "prime-rl-hendrycks"])
    cmd.extend(["--wandb.name", f"{experiment_name}-hendrycks-1b"])
    
    print("="*60)
    print("Starting hendrycks_math 1b training on Modal")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Model: willcb/DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"  - Total GPUs: 8 (2 training, 6 inference)")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Command: {' '.join(cmd)}")
    print("="*60)
    
    # Run the training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    # List output files
    print("\nOutput files:")
    for root, dirs, files in os.walk(output_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.startswith('.'):
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                print(f"  {rel_path}")
    
    return f"Training completed! Results saved to {output_dir}"


@app.local_entrypoint()
def main(
    experiment_name: Optional[str] = None,
    download_results: bool = True,
    dry_run: bool = False,
):
    """
    Deploy hendrycks_math 1b training on Modal.
    
    Args:
        experiment_name: Name for this experiment (auto-generated if not provided)
        download_results: Whether to download results after training
        dry_run: If True, just print what would be run without executing
    """
    import time
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"hendrycks-{int(time.time())}"
    
    print(f"\n🚀 Modal Deployment: hendrycks_math 1b")
    print(f"="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration:")
    print(f"  - Model: willcb/DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"  - Dataset: hendrycks_math")
    print(f"  - Total GPUs: 8x A100-40GB")
    print(f"  - GPU allocation: 2 for training, 6 for inference")
    print(f"  - Estimated time: 30-60 minutes")
    print(f"  - Estimated cost: ~$4-8 on Modal")
    print(f"="*60)
    
    if dry_run:
        print("\n[DRY RUN] Would execute the training with above configuration")
        print("\nTo actually run, remove the --dry-run flag")
        return
    
    # Run training
    print("\n📦 Starting training on Modal...")
    print("(This may take a few minutes to build the container on first run)")
    
    result = train_hendrycks_math.remote(
        experiment_name=experiment_name,
    )
    
    print(result)
    
    # Download results
    if download_results:
        print(f"\n📥 Downloading results...")
        local_output_dir = f"./outputs/{experiment_name}"
        os.makedirs(local_output_dir, exist_ok=True)
        
        # Download from Modal volume
        outputs_volume.get(f"{experiment_name}/hendrycks-math-1b", local_output_dir)
        print(f"✅ Results downloaded to: {local_output_dir}")
        
        # List downloaded files
        print("\n📁 Downloaded files:")
        for root, dirs, files in os.walk(local_output_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if not file.startswith('.'):
                    rel_path = os.path.relpath(os.path.join(root, file), local_output_dir)
                    print(f"  {rel_path}")
    
    print("\n✅ Training session complete!")
    
    # Print useful commands
    print("\n📝 Useful commands:")
    print(f"  # View logs")
    print(f"  modal app logs")
    print(f"  ")
    print(f"  # Monitor GPU usage")
    print(f"  modal app stats")
    print(f"  ")
    print(f"  # Download results later")
    print(f"  modal volume get prime-rl-outputs {experiment_name}/hendrycks-math-1b ./outputs/{experiment_name}")
    print(f"  ")
    print(f"  # List all experiments")
    print(f"  modal volume list prime-rl-outputs")


if __name__ == "__main__":
    main() 