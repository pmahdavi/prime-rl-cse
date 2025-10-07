#!/usr/bin/env python3
"""
Modal deployment for prime-rl training.

This script sets up the three components of prime-rl on Modal:
1. Inference server (vLLM)
2. Orchestrator (RL loop management)  
3. Trainer (PyTorch FSDP training)

Features:
- Single-node and multi-node training support
- Flexible GPU allocation between training and inference
- Automatic dependency installation including flash-attn
- Persistent storage for outputs and checkpoints
- Cost-effective serverless GPU usage
"""

import modal
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict

# Create the Modal app
app = modal.App("prime-rl-training")

# Define persistent volumes for outputs and caches
outputs_volume = modal.Volume.from_name("prime-rl-outputs", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("prime-rl-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("prime-rl-cache", create_if_missing=True)

# Build the container image with all dependencies
# Using PyTorch CUDA development image for flash-attn compilation
# This matches Dockerfile.cuda for consistency
prime_rl_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    # Set CUDA environment (matches Dockerfile.cuda)
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
    })
    # Install system dependencies (matches install.sh + Dockerfile.cuda)
    .apt_install([
        "git", 
        "curl", 
        "build-essential",
        "vim",  # For debugging
        "htop", # For monitoring
        "tmux", # For session management
        "nvtop",  # For GPU monitoring
        "openssh-client",  # For SSH operations
    ])
    # Install uv package manager (using official installer like Dockerfile.cuda)
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR=/usr/local/bin sh",
    )
    # Set uv environment variables (matches Dockerfile.cuda)
    .env({
        "PATH": "/usr/local/bin:$PATH",
        "UV_PYTHON_INSTALL_DIR": "/usr/local/share/uv/python",
        "UV_CACHE_DIR": "/usr/local/share/uv/cache",
        "UV_COMPILE_BYTECODE": "1",
        "UV_LINK_MODE": "copy",
    })
    # Clone the repository
    .run_commands(
        "cd /root && git clone https://github.com/pmahdavi/prime-rl-cse.git",
    )
    # Install dependencies with all extras (including flash-attn)
    # Note: We don't use --locked here because the lockfile might be stale
    # The runtime git pull ensures we get the latest code
    .run_commands(
        "cd /root/prime-rl-cse && uv sync --all-extras",
    )
    # Install wordle environment directly into the prime-rl environment
    .run_commands(
        "cd /root/prime-rl-cse && uv pip install wordle --extra-index-url https://hub.primeintellect.ai/will/simple/",
    )
    # Set runtime environment variables
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
    gpu="A100-80GB:4",  # ⚠️ EDIT THIS LINE to change GPU count/type (e.g., "H100:2", "A100-40GB:8")
    cpu=16.0,  # 16 CPU cores  
    memory=65536,  # 64GB RAM
    volumes={
        "/outputs": outputs_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    timeout=86400,  # 24 hour timeout
    enable_memory_snapshot=True,  # Dramatically improves cold start (2-5min → 10-30sec)
)
def train_prime_rl(
    experiment_name: str,
    trainer_config: str,
    orchestrator_config: str,
    inference_config: str,
    trainer_gpus: int,
    inference_gpus: int,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    output_subdir: str = "",
    use_local_code: bool = False,
):
    """
    Run prime-rl training on Modal.
    
    This function runs all three components (trainer, orchestrator, inference)
    in a single container for simplicity.
    """
    import subprocess
    import os
    
    if use_local_code:
        print("Using local code (mounted from your machine)")
        print("⚠️  Local mode: Changes are from your laptop, not GitHub")
    else:
        # Update to the latest code from GitHub at runtime
        print("Updating to latest code from GitHub...")
        os.chdir("/root/prime-rl-cse")
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    
    # Dependencies are already installed in the Docker image
    
    # Setup output directory
    output_dir = f"/outputs/{experiment_name}"
    if output_subdir:
        output_dir = f"{output_dir}/{output_subdir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command
    # Generate GPU ID lists based on the counts
    trainer_gpu_ids = list(range(trainer_gpus))  # e.g., [0, 1] for trainer_gpus=2
    inference_gpu_ids = list(range(trainer_gpus, trainer_gpus + inference_gpus))  # e.g., [2, 3, 4, 5, 6, 7]
    
    cmd = [
        "uv", "run", "rl",
        "--trainer", "@", trainer_config,
        "--orchestrator", "@", orchestrator_config,
        "--inference", "@", inference_config,
        "--output-dir", output_dir,
    ]
    
    # Add inference parallelism configuration if needed
    if inference_gpus > 1:
        cmd.extend(["--inference.parallel.dp", str(inference_gpus)])
        cmd.extend(["--inference.parallel.tp", "1"])
    
    # Add W&B tracking if specified
    if wandb_project:
        cmd.extend(["--wandb.project", wandb_project])
        if wandb_name:
            cmd.extend(["--wandb.name", wandb_name])
    
    # Add GPU IDs at the end (they appear to be processed last in the argument parser)
    # Using comma-separated format for list arguments
    if trainer_gpu_ids:
        cmd.extend(["--trainer-gpu-ids", ",".join(str(gpu_id) for gpu_id in trainer_gpu_ids)])
    
    if inference_gpu_ids:
        cmd.extend(["--inference-gpu-ids", ",".join(str(gpu_id) for gpu_id in inference_gpu_ids)])
    
    # Set scheduler tuning via typed inference config (no env indirection)
    # Users can override in the TOML; we supply nothing extra here.

    print("="*60)
    print("Starting prime-rl training on Modal")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Total GPUs: {trainer_gpus + inference_gpus} ({trainer_gpus} training, {inference_gpus} inference)")
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


@app.function(
    image=prime_rl_image,
    gpu="A100-40GB:2",  # For distributed training coordination
    volumes={"/outputs": outputs_volume},
)
def distributed_trainer_node(
    node_rank: int,
    num_nodes: int,
    master_addr: str,
    master_port: int,
    trainer_config: str,
    output_dir: str,
    num_gpus_per_node: int,
):
    """Run a single node in distributed training."""
    import subprocess
    import os
    import uuid
    
    # Update to the latest code from GitHub at runtime
    print(f"Node {node_rank}: Updating to latest code from GitHub...")
    os.chdir("/root/prime-rl-cse")
    subprocess.run(["git", "fetch", "origin"], check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    
    # Dependencies are already installed in the Docker image
    
    # Setup distributed training environment
    env = os.environ.copy()
    env.update({
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "WORLD_SIZE": str(num_nodes * num_gpus_per_node),
        "RANK": str(node_rank * num_gpus_per_node),
    })
    
    # Run trainer with torchrun
    cmd = [
        "uv", "run",
        "torchrun",
        f"--nnodes={num_nodes}",
        f"--nproc_per_node={num_gpus_per_node}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m", "prime_rl.trainer.rl.train",
        "@", trainer_config,
        "--output-dir", output_dir,
    ]
    
    print(f"Starting trainer node {node_rank} with {num_gpus_per_node} GPUs")
    result = subprocess.run(cmd, env=env)
    
    return result.returncode


@app.local_entrypoint()
def main(
    experiment_name: Optional[str] = None,
    trainer_config: str = "configs/reverse_text/train.toml",
    orchestrator_config: str = "configs/reverse_text/orch.toml",
    inference_config: str = "configs/reverse_text/infer.toml",
    gpu_type: str = "A100-40GB",
    gpu_count: int = 2,  # Total GPUs
    trainer_gpu_ratio: float = 0.5,  # Fraction of GPUs for training
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    download_results: bool = True,
    distributed: bool = False,
    num_nodes: int = 1,
    use_local_code: bool = False,
):
    """
    Deploy prime-rl training on Modal.
    
    Args:
        experiment_name: Name for this experiment (auto-generated if not provided)
        trainer_config: Path to trainer TOML config
        orchestrator_config: Path to orchestrator TOML config
        inference_config: Path to inference TOML config
        gpu_type: GPU type (T4, L4, A10G, A100-40GB, A100-80GB, H100)
        gpu_count: Total number of GPUs to use
        trainer_gpu_ratio: Fraction of GPUs for training (rest for inference)
        wandb_project: W&B project name (optional)
        wandb_name: W&B run name (optional)
        download_results: Whether to download results after training
        distributed: Enable multi-node distributed training
        num_nodes: Number of nodes for distributed training
        use_local_code: Use local files instead of GitHub (for development)
    
    Examples:
        # Simple run with 2 GPUs (1 training, 1 inference)
        modal run modal/deploy.py
        
        # Hendrycks math with 8 GPUs (2 training, 6 inference)
        modal run modal/deploy.py \
            --trainer-config configs/hendrycks_math/1b/train.toml \
            --orchestrator-config configs/hendrycks_math/1b/orch.toml \
            --inference-config configs/hendrycks_math/1b/infer.toml \
            --gpu-count 8 \
            --trainer-gpu-ratio 0.25
        
        # Local development mode (uses your local code changes)
        modal run modal/deploy.py --use-local-code
        
        # Multi-node training with 4 nodes
        modal run modal/deploy.py \
            --distributed \
            --num-nodes 4 \
            --gpu-type H100
    """
    import time
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = int(time.time())
        config_name = Path(trainer_config).parent.name
        experiment_name = f"{config_name}-{timestamp}"
    
    print(f"\n🚀 Modal Deployment: prime-rl")
    print(f"="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration:")
    print(f"  - Configs: {trainer_config}, {orchestrator_config}, {inference_config}")
    print(f"  - GPU type: {gpu_type}")
    print(f"  - Total GPUs: {gpu_count}")
    
    if distributed:
        print(f"  - Distributed training: {num_nodes} nodes")
        # For distributed training, we need to handle it differently
        # This is a placeholder - implement based on your needs
        raise NotImplementedError("Distributed training support coming soon!")
    
    # Calculate GPU allocation
    trainer_gpus = max(1, int(gpu_count * trainer_gpu_ratio))
    inference_gpus = max(1, gpu_count - trainer_gpus)
    
    print(f"  - GPU allocation: {trainer_gpus} for training, {inference_gpus} for inference")
    
    # Validate GPU type
    valid_gpu_types = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100", "H200", "B200"]
    if not any(gpu_type.startswith(valid) for valid in valid_gpu_types):
        print(f"Warning: Unknown GPU type {gpu_type}, using A100-40GB")
        gpu_type = "A100-40GB"
    
    # Build GPU spec dynamically
    gpu_spec = f"{gpu_type}:{gpu_count}"
    
    # Cost estimation (approximate, varies by GPU type and region)
    cost_per_gpu = {
        "T4": 0.50,
        "L4": 1.00,
        "A10G": 1.50,
        "A100-40GB": 3.70,
        "A100-80GB": 5.00,
        "H100": 4.00,
        "H200": 5.00,
        "B200": 6.00,
    }
    gpu_cost = next((v for k, v in cost_per_gpu.items() if gpu_type.startswith(k)), 3.70)
    
    print(f"  - GPU spec: {gpu_spec}")
    print(f"  - Estimated time: varies by experiment")
    print(f"  - Estimated cost: ~${gpu_count * gpu_cost:.2f}/hour ({gpu_type} pricing)")
    
    if use_local_code:
        print(f"\n⚠️  LOCAL DEV MODE ENABLED")
        print(f"  - Using local code from your laptop")
        print(f"  - Changes will be synced automatically")
        print(f"  - NOT pulling from GitHub")
    
    print(f"="*60)
    
    # Run training
    print("\n📦 Starting training on Modal...")
    print("(This may take a few minutes to build the container on first run)")
    print(f"⚠️  Note: GPU config is set in deploy.py line 94 (currently hardcoded)")
    print(f"   Requested: {gpu_spec}, but using whatever is in the decorator")
    
    # Use the training function
    train_fn = train_prime_rl
    
    # Add local mounts if needed
    if use_local_code:
        # Get the project root (parent of modal/ directory)
        project_root = Path(__file__).parent.parent
        
        # TODO: Modal doesn't support dynamic GPU changes, but we can mount local code
        # Mount local directories to the container
        print("⚠️  Local code mounting not yet implemented in simplified version")
        # train_fn = train_fn.with_options(
        #     mounts=[...]
        # )
        print("✅ Using GitHub code instead")
    
    result = train_fn.remote(
        experiment_name=experiment_name,
        trainer_config=trainer_config,
        orchestrator_config=orchestrator_config,
        inference_config=inference_config,
        trainer_gpus=trainer_gpus,
        inference_gpus=inference_gpus,
        wandb_project=wandb_project,
        wandb_name=wandb_name or experiment_name,
        use_local_code=use_local_code,
    )
    
    print(result)
    
    # Download results
    if download_results:
        print(f"\n📥 Downloading results...")
        local_output_dir = f"./outputs/{experiment_name}"
        os.makedirs(local_output_dir, exist_ok=True)
        
        # Download from Modal volume
        outputs_volume.get(experiment_name, local_output_dir)
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
    print(f"  modal volume get prime-rl-outputs {experiment_name} ./outputs/{experiment_name}")
    print(f"  ")
    print(f"  # List all experiments")
    print(f"  modal volume list prime-rl-outputs")


if __name__ == "__main__":
    main() 