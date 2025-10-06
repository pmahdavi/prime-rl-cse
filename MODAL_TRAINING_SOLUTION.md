# Modal Training Solution for PRIME-RL

## The Core Issue

Your Modal app disconnected after ~12 minutes because you're running in attached mode. When your local client disconnects (laptop sleeps, network issues, etc.), Modal terminates the function.

## Solution: Use Detached Mode

### Option 1: Run with --detach flag

```bash
modal run --detach your_script.py
```

This will:
- Keep the function running even if your client disconnects
- Return a call ID you can use to check logs later
- Allow long-running training jobs to complete

### Option 2: Use modal.Function.spawn()

Instead of running directly, use the spawn pattern:

```python
import modal

app = modal.App("prime-rl-training")

@app.function(
    gpu="A100-80GB:4",
    timeout=86400,  # 24 hours
    volumes={"/outputs": modal.Volume.from_name("prime-rl-outputs")},
)
def train_model(config):
    # Your training code here
    pass

@app.local_entrypoint()
def main():
    # Spawn the training function
    train_model.spawn(config)
    print("Training job spawned! Check Modal dashboard for logs.")
```

### Option 3: Use Modal's Persistent Storage

Ensure your outputs are saved to persistent storage:

```python
@app.function(
    gpu="A100-80GB:4",
    timeout=86400,
    volumes={
        "/outputs": modal.Volume.from_name("prime-rl-outputs", create_if_not_exists=True),
        "/checkpoints": modal.Volume.from_name("prime-rl-checkpoints", create_if_not_exists=True),
    },
    _allow_background_volume_commits=True,  # Important for long-running jobs
)
def train_model():
    # Training code
    pass
```

## Enhanced Modal Script for PRIME-RL

Here's a production-ready Modal script that handles disconnections properly:

```python
import modal
import subprocess
import os
import time
from pathlib import Path
from datetime import datetime

app = modal.App("prime-rl-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
    )
    .run_commands(
        "cd /root && git clone https://github.com/pmahdavi/prime-rl-cse.git",
        "cd /root/prime-rl-cse && /root/.cargo/bin/uv sync --all-extras",
        gpu="A100",  # Need GPU for flash-attn build
    )
)

@app.function(
    image=image,
    gpu="A100-80GB:4",
    timeout=86400,  # 24 hours
    volumes={
        "/outputs": modal.Volume.from_name("prime-rl-outputs"),
        "/checkpoints": modal.Volume.from_name("prime-rl-checkpoints"),
    },
    _allow_background_volume_commits=True,
    secret=modal.Secret.from_name("wandb-secret"),  # Add your W&B API key
)
def train_prime_rl(
    experiment_name: str,
    trainer_config: str,
    orchestrator_config: str, 
    inference_config: str,
    wandb_project: str = "prime-rl",
    resume_step: int = None,
):
    """
    Run PRIME-RL training with proper error handling and monitoring.
    """
    
    os.chdir("/root/prime-rl-cse")
    
    # Setup output directory
    output_dir = f"/outputs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file for this run
    log_file = f"{output_dir}/modal_run_{datetime.now().isoformat()}.log"
    
    def log(message):
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")
    
    log(f"Starting PRIME-RL training: {experiment_name}")
    
    # Build the command
    cmd = [
        "/root/.cargo/bin/uv", "run", "rl",
        "--trainer", "@", trainer_config,
        "--orchestrator", "@", orchestrator_config,
        "--inference", "@", inference_config,
        "--output-dir", output_dir,
        "--wandb.project", wandb_project,
        "--wandb.name", experiment_name,
        "--trainer-gpu-ids", "0,1",
        "--inference-gpu-ids", "2,3",
    ]
    
    if resume_step:
        cmd.extend(["--trainer.ckpt.resume-step", str(resume_step)])
        cmd.extend(["--orchestrator.ckpt.resume-step", str(resume_step)])
    
    log(f"Command: {' '.join(cmd)}")
    
    # Run training with proper process management
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            log("Training completed successfully!")
        else:
            log(f"Training failed with return code: {return_code}")
            
    except Exception as e:
        log(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Ensure volumes are committed
        log("Committing volume changes...")
        modal.Volume.commit()
        
    return {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "log_file": log_file,
        "return_code": return_code,
    }

@app.function(image=image, gpu="A100")
def debug_environment():
    """Debug function to check environment setup"""
    import torch
    
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "prime_rl_exists": Path("/root/prime-rl-cse").exists(),
        "uv_exists": Path("/root/.cargo/bin/uv").exists(),
    }
    
    # Check if we can import prime_rl
    try:
        os.chdir("/root/prime-rl-cse")
        subprocess.run(["/root/.cargo/bin/uv", "run", "python", "-c", "import prime_rl"], check=True)
        checks["prime_rl_importable"] = True
    except:
        checks["prime_rl_importable"] = False
    
    # Check verifiers
    try:
        subprocess.run(
            ["/root/.cargo/bin/uv", "run", "python", "-c", 
             "from verifiers import get_verifier; get_verifier('wordle')"],
            check=True
        )
        checks["wordle_env_available"] = True
    except:
        checks["wordle_env_available"] = False
    
    return checks

@app.local_entrypoint()
def main(
    experiment: str = "wordle-rl-test",
    detach: bool = True,
    debug: bool = False,
):
    """
    Main entry point for Modal training.
    
    Args:
        experiment: Name of the experiment
        detach: Whether to run in detached mode (recommended)
        debug: Run debug checks before training
    """
    
    if debug:
        print("Running environment checks...")
        checks = debug_environment.remote()
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check}: {result}")
        
        if not all(checks.values()):
            print("Some checks failed! Fix these before training.")
            return
    
    # Configuration paths
    trainer_config = "examples/wordle/rl/train.toml"
    orchestrator_config = "examples/wordle/rl/orch.toml"
    inference_config = "examples/wordle/rl/infer.toml"
    
    if detach:
        # Spawn the training job
        handle = train_prime_rl.spawn(
            experiment_name=experiment,
            trainer_config=trainer_config,
            orchestrator_config=orchestrator_config,
            inference_config=inference_config,
            wandb_project="prime-rl-wordle",
        )
        print(f"Training job spawned with ID: {handle.object_id}")
        print(f"Check progress at: https://modal.com/apps/{app.name}/deployed/{handle.object_id}")
        print(f"Get logs with: modal app logs {handle.object_id}")
    else:
        # Run attached (not recommended for long training)
        result = train_prime_rl.remote(
            experiment_name=experiment,
            trainer_config=trainer_config,
            orchestrator_config=orchestrator_config,
            inference_config=inference_config,
            wandb_project="prime-rl-wordle",
        )
        print(f"Training completed: {result}")

if __name__ == "__main__":
    main()
```

## Usage Instructions

1. **First time setup**:
   ```bash
   # Install Modal CLI
   pip install modal
   
   # Authenticate
   modal setup
   
   # Create volumes
   modal volume create prime-rl-outputs
   modal volume create prime-rl-checkpoints
   
   # Add W&B secret
   modal secret create wandb-secret WANDB_API_KEY=your_key_here
   ```

2. **Run training (detached)**:
   ```bash
   python modal_train.py --experiment "wordle-experiment-1" --detach
   ```

3. **Check logs**:
   ```bash
   # Get the call ID from the output
   modal app logs <call-id>
   
   # Or check in the Modal dashboard
   ```

4. **Resume training**:
   ```python
   # In the script, add resume_step parameter
   train_prime_rl.remote(
       experiment_name="wordle-experiment-1",
       resume_step=50,  # Resume from step 50
       ...
   )
   ```

## Key Points

1. **Always use detached mode** for long-running training jobs
2. **Use persistent volumes** for outputs and checkpoints
3. **Add proper error handling** and logging
4. **Monitor via Modal dashboard** or CLI
5. **Set appropriate timeouts** (24 hours in example)
6. **Commit volumes** to ensure data persistence

## Debugging Tips

If training still hangs:

1. **Check Modal dashboard** for real-time logs
2. **SSH into running container**:
   ```bash
   modal shell <function-name>
   ```
3. **Monitor GPU usage**:
   ```bash
   modal app exec <call-id> -- nvidia-smi
   ```
4. **Check volume contents**:
   ```bash
   modal volume ls prime-rl-outputs
   ```

This approach ensures your training continues even if your local machine disconnects, and provides better observability for debugging issues.