# Complete Modal Debug Guide for PRIME-RL Training

## Problem Identification

Your training job failed due to **Modal client disconnection** after ~12 minutes. The actual training never started because:

1. Modal terminated the job when your local client disconnected
2. The trainer was waiting for data that the orchestrator hadn't generated yet
3. This waiting period exceeded the time before disconnection

## Immediate Solutions

### Solution 1: Use Detached Mode (Recommended)

```bash
modal run --detach your_training_script.py
```

This prevents disconnection issues entirely.

### Solution 2: Fix the Actual Training Hang

Even with detached mode, you need to fix why training isn't starting. The log shows:
- Inference server started ✓
- Orchestrator started ✓  
- Trainer started and waiting for batch ✓
- **But no batch arrived for 12+ minutes** ❌

This suggests the orchestrator isn't successfully generating rollouts.

## Complete Debugging Checklist

### 1. Verify Wordle Environment Installation

```python
# Add this check to your Modal script
def check_wordle_env():
    try:
        from verifiers import get_verifier
        verifier = get_verifier('wordle')
        print(f"✓ Wordle environment loaded: {verifier}")
        return True
    except Exception as e:
        print(f"✗ Wordle environment error: {e}")
        # Try to install it
        subprocess.run(["prime", "env", "install", "wordle"], check=False)
        return False
```

### 2. Monitor All Component Outputs

```python
import modal
import subprocess
import threading
import queue
import time

app = modal.App("prime-rl-debug")

def stream_output(proc, name, output_queue):
    """Stream process output to queue"""
    for line in iter(proc.stdout.readline, ''):
        if line:
            output_queue.put((name, line.rstrip()))

@app.function(
    gpu="A100-80GB:4",
    timeout=3600,
    volumes={"/outputs": modal.Volume.from_name("prime-rl-outputs")},
)
def debug_training():
    output_queue = queue.Queue()
    processes = {}
    threads = []
    
    # Commands for each component
    commands = {
        "inference": [
            "uv", "run", "inference",
            "@", "examples/wordle/rl/infer.toml",
            "--log.level", "DEBUG",
            "--port", "8000"
        ],
        "orchestrator": [
            "uv", "run", "orchestrator",
            "@", "examples/wordle/rl/orch.toml", 
            "--log.level", "DEBUG",
            "--client.base_url", "http://localhost:8000/v1",
            "--output-dir", "/outputs/debug-run"
        ],
        "trainer": [
            "uv", "run", "torchrun",
            "--nproc-per-node", "2",
            "src/prime_rl/trainer/rl/train.py",
            "@", "examples/wordle/rl/train.toml",
            "--log.level", "DEBUG",
            "--output-dir", "/outputs/debug-run"
        ]
    }
    
    # Start components with delays
    startup_delays = {"inference": 0, "orchestrator": 20, "trainer": 30}
    
    for name, cmd in commands.items():
        print(f"\n{'='*50}")
        print(f"Starting {name} (waiting {startup_delays[name]}s)...")
        print(f"Command: {' '.join(cmd)}")
        print('='*50)
        
        time.sleep(startup_delays[name])
        
        env = os.environ.copy()
        if name == "inference":
            env["CUDA_VISIBLE_DEVICES"] = "2,3"
        elif name == "trainer":
            env["CUDA_VISIBLE_DEVICES"] = "0,1"
            
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        processes[name] = proc
        
        # Start output streaming thread
        thread = threading.Thread(
            target=stream_output,
            args=(proc, name, output_queue),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Monitor outputs and system state
    start_time = time.time()
    rollout_check_interval = 10
    last_rollout_check = 0
    
    while True:
        # Get output from queue
        try:
            name, line = output_queue.get(timeout=1)
            print(f"[{name}] {line}")
            
            # Look for specific error patterns
            if "error" in line.lower() or "exception" in line.lower():
                print(f"\n⚠️  ERROR DETECTED IN {name}: {line}\n")
            
        except queue.Empty:
            pass
        
        # Check process health
        for name, proc in processes.items():
            if proc.poll() is not None:
                print(f"\n❌ {name} CRASHED with exit code {proc.returncode}\n")
                # Get any remaining output
                remaining = proc.stdout.read()
                if remaining:
                    print(f"Final output from {name}:\n{remaining}")
                return
        
        # Periodically check rollout directory
        if time.time() - last_rollout_check > rollout_check_interval:
            rollout_dir = Path("/outputs/debug-run/rollouts")
            if rollout_dir.exists():
                files = list(rollout_dir.rglob("*.pt"))
                print(f"\n📁 Rollout check: {len(files)} files found")
                if files:
                    for f in files[:3]:
                        size = f.stat().st_size / 1024 / 1024  # MB
                        print(f"   - {f.name}: {size:.2f} MB")
            else:
                print(f"\n📁 Rollout directory not created yet")
            last_rollout_check = time.time()
        
        # Timeout after 5 minutes
        if time.time() - start_time > 300:
            print("\n⏱️  Timeout reached (5 minutes)")
            break
    
    # Cleanup
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
            
    return "Debug session completed"

@app.local_entrypoint()
def main():
    result = debug_training.spawn()
    print(f"Debug job spawned: {result.object_id}")
    print(f"View logs: modal app logs {result.object_id}")
```

### 3. Common Issues and Fixes

#### Issue: "Wordle environment not found"
```bash
# Install in Modal function
subprocess.run(["prime", "env", "install", "wordle"], check=True)
# OR
subprocess.run(["uv", "pip", "install", "git+https://github.com/PrimeIntellect/wordle-env"], check=True)
```

#### Issue: "Connection refused" from orchestrator
```python
# Ensure inference server is ready before starting orchestrator
def wait_for_inference_server(url="http://localhost:8000", timeout=60):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health")
            if resp.status_code == 200:
                print("✓ Inference server is ready")
                return True
        except:
            pass
        time.sleep(1)
    raise TimeoutError("Inference server did not start")
```

#### Issue: "CUDA out of memory"
```toml
# Reduce batch sizes in configs
# orch.toml
batch_size = 256  # Reduce from 1024
micro_batch_size = 1

# train.toml  
micro_batch_size = 1
```

### 4. Minimal Test Configuration

Create test configs with minimal resource usage:

```toml
# test_orch.toml
max_steps = 5
seq_len = 512  # Reduce from 4096
batch_size = 16  # Reduce from 1024
micro_batch_size = 1
rollouts_per_example = 2  # Reduce from 16
async_level = 0  # Disable async for testing

[model]
name = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"

[environment]
id = "wordle"

[sampling]
max_tokens = 128  # Reduce from 1024
```

### 5. Production Modal Script

```python
import modal
from pathlib import Path
import subprocess
import os

app = modal.App("prime-rl-production")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "wget")
    .run_commands(
        # Install uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # Clone repo
        "cd /root && git clone https://github.com/pmahdavi/prime-rl-cse.git",
    )
    .run_commands(
        # Install dependencies with GPU for flash-attn
        "cd /root/prime-rl-cse && /root/.cargo/bin/uv sync --all-extras",
        gpu="A100",
    )
    .run_commands(
        # Pre-install Wordle environment
        "cd /root/prime-rl-cse && /root/.cargo/bin/uv run python -c 'from verifiers import get_verifier; get_verifier(\"wordle\")'",
    )
)

@app.function(
    image=image,
    gpu="A100-80GB:4",
    timeout=86400,
    volumes={
        "/outputs": modal.Volume.from_name("prime-rl-outputs"),
    },
    secret=modal.Secret.from_name("wandb-secret"),
)
def train_with_monitoring(experiment_name: str):
    os.chdir("/root/prime-rl-cse")
    
    # Create output directory
    output_dir = f"/outputs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write a monitoring script
    monitor_script = f"""
import subprocess
import time
import sys
from pathlib import Path

# Start training
proc = subprocess.Popen([
    "/root/.cargo/bin/uv", "run", "rl",
    "--trainer", "@", "examples/wordle/rl/train.toml",
    "--orchestrator", "@", "examples/wordle/rl/orch.toml",
    "--inference", "@", "examples/wordle/rl/infer.toml",
    "--output-dir", "{output_dir}",
    "--trainer-gpu-ids", "0,1",
    "--inference-gpu-ids", "2,3",
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)

# Monitor output
for line in iter(proc.stdout.readline, ''):
    print(line.rstrip())
    sys.stdout.flush()
    
    # Check for rollout creation
    rollout_dir = Path("{output_dir}/rollouts")
    if rollout_dir.exists():
        steps = list(rollout_dir.glob("step_*"))
        if steps and "Waiting for training batch" in line:
            print(f"\\n⚠️  ISSUE: Trainer waiting but {len(steps)} rollout steps exist!")
            for step in sorted(steps)[-3:]:
                files = list(step.glob("*.pt"))
                print(f"   {step.name}: {len(files)} files")

proc.wait()
"""
    
    # Run the monitoring script
    with open("/tmp/monitor.py", "w") as f:
        f.write(monitor_script)
    
    subprocess.run(["/root/.cargo/bin/uv", "run", "python", "/tmp/monitor.py"])

@app.local_entrypoint()
def main():
    # Always use spawn for long-running jobs
    handle = train_with_monitoring.spawn("wordle-debug-run")
    print(f"Training started with ID: {handle.object_id}")
    print(f"Monitor at: https://modal.com/apps/{handle.app_id}/deployed/{handle.object_id}")
    print(f"Get logs: modal app logs {handle.object_id}")
```

## Summary

The core issue is Modal disconnection, but even after fixing that, you need to ensure:

1. **Use detached mode** (`modal run --detach` or `.spawn()`)
2. **All components start successfully** (check logs from all three)
3. **Wordle environment is installed** 
4. **File paths match** between trainer and orchestrator configs
5. **Resources are sufficient** (reduce batch sizes if needed)

The debugging scripts above will help identify exactly where the pipeline breaks down.