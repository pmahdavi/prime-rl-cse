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

# Define project root to access local files for the image build
project_root = Path(__file__).parent.parent

# Create the Modal app
app = modal.App("prime-rl-training")

def build_dependencies():
    """This function is run once during the image build process."""
    import subprocess
    subprocess.run(["uv", "sync", "--locked", "--no-dev"], check=True)
    subprocess.run(["uv", "sync", "--all-extras", "--locked", "--no-dev"], check=True)

# Define persistent volumes for outputs and caches
outputs_volume = modal.Volume.from_name("prime-rl-outputs", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("prime-rl-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("prime-rl-cache", create_if_missing=True)

# Build the container image with all dependencies
# This now mimics Dockerfile.cuda by using local files and locked dependencies.
prime_rl_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    # Set CUDA environment (matches Dockerfile.cuda)
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
    })
    # Install system dependencies (aligned with Dockerfile.cuda)
    .apt_install([
        "git",
        "curl",
        "build-essential",
        "sudo",
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
    # Set up the app directory and install dependencies using a dedicated function.
    # Add local files for dependency installation (Modal 1.0 API)
    # Use copy=True to allow running build steps after adding files
    # Matches Dockerfile.cuda structure
    .workdir("/app")
    .add_local_file(project_root / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(project_root / "uv.lock", "/app/uv.lock", copy=True)
    .add_local_file(project_root / "README.md", "/app/README.md", copy=True)
    .add_local_dir(project_root / "src", "/app/src", copy=True)
    .add_local_dir(project_root / "configs", "/app/configs", copy=True)
    .run_function(build_dependencies)
    # Set the virtual environment's Python as the default (must come before other env vars)
    .env({"PATH": "/app/.venv/bin:$PATH"})
    # Set runtime environment variables (secrets are injected via Modal Secret)
    .env({
        "HF_HUB_CACHE": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "WANDB_DIR": "/outputs/wandb",
    })
    # Mount source code and configs at runtime for fast iteration
    # This must be LAST to avoid triggering rebuilds on code changes
    .add_local_dir(project_root / "src", "/app/src")
    .add_local_dir(project_root / "configs", "/app/configs")
)


# Get GPU configuration from environment variable (set before modal loads)
# Default to H100:4 if not specified
DEFAULT_GPU_CONFIG = os.environ.get("MODAL_GPU_CONFIG", "H100:4")

@app.function(
    image=prime_rl_image,
    gpu=DEFAULT_GPU_CONFIG,  # This is read at module import time
    cpu=16.0,
    memory=65536,
    volumes={
        "/outputs": outputs_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    timeout=86400,
    enable_memory_snapshot=True,
)
def run_command(
    command: str,
    experiment_name: str,
):
    """
    Run an arbitrary command on Modal.

    This function runs any command in the container with all dependencies installed.
    The code is mounted from your local machine to ensure it's always up-to-date.

    Args:
        command: The command to run (e.g., "uv run rl --trainer @ configs/...")
        experiment_name: Name for this experiment (used for output directory)
    """
    import subprocess
    import os
    import shlex

    # The working directory is set to /app in the image definition.
    # The code is mounted from your local machine.
    print("Running with local code mounted from your machine.")

    # Setup output directory
    output_dir = f"/outputs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Parse the command string into a list
    # Use shlex to properly handle quoted strings
    cmd = shlex.split(command)

    # If the command doesn't already have --output-dir, add it
    if "--output-dir" not in command:
        cmd.extend(["--output-dir", output_dir])

    print("="*60)
    print("Starting command on Modal")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Command: {' '.join(cmd)}")
    print("="*60)

    # Run the command
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n" + "="*60)
    print("Command completed successfully!")
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

    return f"Command completed! Results saved to {output_dir}"


@app.local_entrypoint()
def main(
    command: Optional[str] = None,
    experiment_name: Optional[str] = None,
    download_results: bool = True,
):
    """
    Deploy a command on Modal.

    Args:
        command: Command to run (e.g., "uv run rl --trainer @ configs/...")
                 If not provided, uses default reverse_text example
        experiment_name: Name for this experiment (auto-generated if not provided)
        download_results: Whether to show download instructions after completion

    GPU Configuration:
        Set the MODAL_GPU_CONFIG environment variable to change GPU allocation.
        Default: H100:4

        MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py

    Examples:
        # Run with default command (reverse_text example)
        modal run modal/deploy.py

        # Run with different GPU configuration
        MODAL_GPU_CONFIG="A100-80GB:4" modal run modal/deploy.py

        # Run custom RL training command
        modal run modal/deploy.py --command "uv run rl --trainer @ configs/reverse_text/rl/train.toml --orchestrator @ configs/reverse_text/rl/orch.toml --inference @ configs/reverse_text/rl/infer.toml --trainer-gpu-ids 0 --inference-gpu-ids 1"

        # Run SFT training
        modal run modal/deploy.py --command "uv run sft @ configs/debug/sft/train.toml"

        # Run evaluation
        modal run modal/deploy.py --command "uv run eval --model.name my-model --environment-ids math500"

        # Custom experiment name with GPU config
        MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py \
            --command "uv run rl --trainer @ configs/hendrycks_math/1b/train.toml --orchestrator @ configs/hendrycks_math/1b/orch.toml --inference @ configs/hendrycks_math/1b/infer.toml" \
            --experiment-name "math-experiment-1"
    """
    import time

    # Default command if none provided
    if command is None:
        command = (
            "uv run rl "
            "--trainer @ configs/reverse_text/rl/train.toml "
            "--orchestrator @ configs/reverse_text/rl/orch.toml "
            "--inference @ configs/reverse_text/rl/infer.toml "
            "--trainer-gpu-ids 0 "
            "--inference-gpu-ids 1 "
            "--wandb.project prime-rl "
            "--wandb.name reverse-text-modal"
        )

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = int(time.time())
        # Try to extract a meaningful name from the command
        if "reverse_text" in command:
            prefix = "reverse-text"
        elif "hendrycks_math" in command:
            prefix = "math"
        elif "sft" in command:
            prefix = "sft"
        elif "eval" in command:
            prefix = "eval"
        else:
            prefix = "experiment"
        experiment_name = f"{prefix}-{timestamp}"

    # Parse the actual GPU config being used
    actual_gpu_config = DEFAULT_GPU_CONFIG

    # Try to parse GPU type and count from config (e.g., "A100-80GB:4" -> "A100-80GB", 4)
    if ":" in actual_gpu_config:
        gpu_type_actual, gpu_count_str = actual_gpu_config.rsplit(":", 1)
        try:
            gpu_count_actual = int(gpu_count_str)
        except ValueError:
            gpu_type_actual = actual_gpu_config
            gpu_count_actual = 1
    else:
        gpu_type_actual = actual_gpu_config
        gpu_count_actual = 1

    print(f"\n🚀 Modal Deployment: prime-rl")
    print(f"="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration:")
    print(f"  - Command: {command}")
    print(f"  - GPU config: {actual_gpu_config}")

    # Cost estimation (Modal pricing as of 2025)
    cost_per_gpu = {
        "T4": 0.59,
        "L4": 0.80,
        "A10": 1.10,
        "L40S": 1.95,
        "A100-40GB": 2.10,
        "A100-80GB": 2.50,
        "H100": 3.95,
        "H200": 4.54,
        "B200": 6.25,
    }
    gpu_cost = next((v for k, v in cost_per_gpu.items() if gpu_type_actual.startswith(k)), 2.10)

    print(f"  - Estimated time: varies by experiment")
    print(f"  - Estimated cost: ~${gpu_count_actual * gpu_cost:.2f}/hour ({gpu_type_actual} pricing)")

    print(f"\n✅  This script runs your local code on Modal.")
    print(f"  - Your project directory is mounted into the container.")
    print(f"  - Changes to your code will be reflected on the next run.")

    print(f"="*60)

    # Run training
    print("\n📦 Starting command on Modal...")
    print("(This may take a few minutes to build the container on first run)")
    if os.environ.get("MODAL_GPU_CONFIG"):
        print(f"✅ Using custom GPU config from MODAL_GPU_CONFIG: {actual_gpu_config}")
    else:
        print(f"ℹ️  Using default GPU config: {actual_gpu_config}")
        print(f"   To change, set: MODAL_GPU_CONFIG='A100-80GB:4' modal run modal/deploy.py")

    # Run the command
    result = run_command.remote(
        command=command,
        experiment_name=experiment_name,
    )

    print(result)
    
    # Download results
    if download_results:
        print(f"\n📥 To download results, use the Modal CLI:")
        print(f"  modal volume get prime-rl-outputs {experiment_name} ./outputs/{experiment_name}")
        print(f"\nOr download individual files programmatically using volume.read_file_into_fileobj()")
    
    print("\n✅ Training session complete!")
    
    # Print useful commands
    print("\n📝 Useful commands:")
    print(f"  # View logs")
    print(f"  modal app logs")
    print(f"  ")
    print(f"  # Monitor GPU usage")
    print(f"  modal app stats")
    print(f"  ")
    print(f"  # Download results")
    print(f"  modal volume get prime-rl-outputs {experiment_name} ./outputs/{experiment_name}")
    print(f"  ")
    print(f"  # List all experiments in volume")
    print(f"  modal volume ls prime-rl-outputs")
    print(f"  ")
    print(f"  # List files in this experiment")
    print(f"  modal volume ls prime-rl-outputs/{experiment_name}")


if __name__ == "__main__":
    main() 