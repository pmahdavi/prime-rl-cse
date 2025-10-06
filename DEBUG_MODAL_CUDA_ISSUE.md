# Modal CUDA Training Interruption Debug Guide

## Issue Summary
The PRIME-RL training hangs at "Waiting for training batch to arrive" because the orchestrator isn't writing rollout data files that the trainer is waiting for.

## Root Cause Analysis

The PRIME-RL architecture uses a file-based communication pattern:

1. **Orchestrator** collects rollouts from inference server and writes them to disk:
   - Location: `/outputs/{experiment}/rollouts/step_{N}/rank_{R}.pt`
   
2. **Trainer** waits for these files to appear:
   - Uses `wait_for_path()` to poll for file existence
   - Reads batch data from these `.pt` files

3. **Inference Server** generates rollouts via vLLM API

The hang occurs because one of these components isn't functioning properly.

## Immediate Debugging Steps

### 1. Check Component Logs Separately

The current Modal script only shows trainer logs. Modify your Modal function to capture ALL component logs:

```python
import subprocess
import threading
import queue

def capture_output(process, name, output_queue):
    """Capture and forward process output"""
    for line in iter(process.stdout.readline, b''):
        output_queue.put(f"[{name}] {line.decode().rstrip()}")
    
def run_training():
    output_queue = queue.Queue()
    processes = {}
    
    # Start inference
    inf_cmd = ["uv", "run", "inference", "@", "examples/wordle/rl/infer.toml"]
    processes['inference'] = subprocess.Popen(
        inf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "2,3"}
    )
    
    # Start orchestrator
    orch_cmd = ["uv", "run", "orchestrator", "@", "examples/wordle/rl/orch.toml"]
    processes['orchestrator'] = subprocess.Popen(
        orch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Start output capture threads
    for name, proc in processes.items():
        thread = threading.Thread(
            target=capture_output,
            args=(proc, name, output_queue)
        )
        thread.daemon = True
        thread.start()
    
    # Monitor and print all outputs
    while True:
        try:
            line = output_queue.get(timeout=0.1)
            print(line)
        except queue.Empty:
            # Check if any process died
            for name, proc in processes.items():
                if proc.poll() is not None:
                    print(f"ERROR: {name} exited with code {proc.returncode}")
                    return
```

### 2. Check File System Permissions

The orchestrator needs to write to the shared file system:

```python
# Add this check before starting training
import os
import tempfile

def check_filesystem():
    test_dir = "/outputs/test_write"
    try:
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        print(f"✓ File system write test passed")
        os.remove(test_file)
        os.rmdir(test_dir)
    except Exception as e:
        print(f"✗ File system write test failed: {e}")
        raise
```

### 3. Verify Rollout Directory Structure

Check if the orchestrator is creating the expected directory structure:

```bash
# Add monitoring to your Modal script
def monitor_rollout_dir():
    rollout_dir = "/outputs/wordle-rl-4xa100-80gb/rollouts"
    print(f"Monitoring {rollout_dir}")
    
    import time
    for _ in range(60):  # Monitor for 1 minute
        if os.path.exists(rollout_dir):
            files = list(Path(rollout_dir).rglob("*.pt"))
            print(f"Found {len(files)} rollout files")
            for f in files[:5]:  # Show first 5
                print(f"  - {f}")
        time.sleep(1)
```

### 4. Check Inference Server Health

The orchestrator depends on the inference server being healthy:

```python
def check_inference_health():
    import requests
    import time
    
    # Wait for inference server to start
    time.sleep(10)
    
    try:
        # Check health endpoint
        resp = requests.get("http://localhost:8000/health")
        print(f"Inference health check: {resp.status_code}")
        
        # Check models endpoint
        resp = requests.get("http://localhost:8000/v1/models")
        print(f"Available models: {resp.json()}")
        
        return True
    except Exception as e:
        print(f"Inference server not responding: {e}")
        return False
```

### 5. Debug Orchestrator Connection

The orchestrator needs to connect to the inference server. Check its configuration:

```python
# In your orch.toml or command line args
client.base_url = "http://localhost:8000/v1"  # Must match inference server

# Verify in logs
print(f"Orchestrator connecting to: {config.client.base_url}")
```

## Common Issues and Solutions

### Issue 1: Inference Server Crash
**Symptoms**: No inference logs after startup
**Solution**: 
- Check GPU memory: Reduce `--inference.parallel.dp` or use smaller model
- Check vLLM initialization errors in inference logs

### Issue 2: Orchestrator Can't Connect
**Symptoms**: Orchestrator logs show connection errors
**Solution**:
- Ensure inference server is fully started before orchestrator
- Check `client.base_url` matches inference server address
- For Modal, all components run on same container so use `localhost`

### Issue 3: Environment Not Found
**Symptoms**: Orchestrator crashes with "environment not found"
**Solution**:
```bash
# Install Wordle environment before training
uv run python -c "from verifiers import get_verifier; get_verifier('wordle')"
```

### Issue 4: File System Issues
**Symptoms**: Permission errors or missing directories
**Solution**:
- Ensure `/outputs` is writable
- Create directories with proper permissions
- For Modal, use persistent volume mounts

## Enhanced Modal Script

Here's a complete debugging-friendly Modal script:

```python
import modal
import subprocess
import time
import os
from pathlib import Path

app = modal.App("prime-rl-debug")

@app.function(
    gpu="A100-80GB:4",
    timeout=3600,
    volumes={"/outputs": modal.Volume.from_name("prime-rl-outputs")},
)
def run_training_debug():
    # Pre-flight checks
    print("=== PRE-FLIGHT CHECKS ===")
    
    # 1. Check file system
    test_write = Path("/outputs/test.txt")
    test_write.write_text("test")
    assert test_write.exists(), "File system not writable"
    test_write.unlink()
    print("✓ File system check passed")
    
    # 2. Check GPU allocation
    import torch
    print(f"✓ GPUs available: {torch.cuda.device_count()}")
    
    # 3. Check environment
    try:
        from verifiers import get_verifier
        get_verifier('wordle')
        print("✓ Wordle environment available")
    except Exception as e:
        print(f"✗ Wordle environment error: {e}")
        raise
    
    # Start components with full logging
    print("\n=== STARTING COMPONENTS ===")
    
    # Component commands
    commands = {
        "inference": [
            "uv", "run", "inference",
            "@", "examples/wordle/rl/infer.toml",
            "--log.level", "DEBUG"
        ],
        "orchestrator": [
            "uv", "run", "orchestrator", 
            "@", "examples/wordle/rl/orch.toml",
            "--log.level", "DEBUG"
        ],
        "trainer": [
            "uv", "run", "torchrun",
            "--nproc-per-node", "2",
            "src/prime_rl/trainer/rl/train.py",
            "@", "examples/wordle/rl/train.toml",
            "--log.level", "DEBUG"
        ]
    }
    
    # Environment variables
    envs = {
        "inference": {"CUDA_VISIBLE_DEVICES": "2,3"},
        "orchestrator": {},
        "trainer": {"CUDA_VISIBLE_DEVICES": "0,1"}
    }
    
    # Start processes with output capture
    processes = {}
    for name, cmd in commands.items():
        if name == "orchestrator":
            # Wait for inference to start
            time.sleep(15)
        elif name == "trainer":
            # Wait for orchestrator to start
            time.sleep(10)
            
        print(f"Starting {name}...")
        processes[name] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, **envs[name]},
            universal_newlines=True,
            bufsize=1
        )
    
    # Monitor all outputs
    import select
    fds = {p.stdout: name for name, p in processes.items()}
    
    print("\n=== COMPONENT LOGS ===")
    while fds:
        readable, _, _ = select.select(list(fds.keys()), [], [], 0.1)
        
        for fd in readable:
            line = fd.readline()
            if line:
                name = fds[fd]
                print(f"[{name}] {line.rstrip()}")
            else:
                # EOF, process might have died
                name = fds.pop(fd)
                proc = processes[name]
                if proc.poll() is not None:
                    print(f"\n[ERROR] {name} exited with code {proc.returncode}")
                    
                    # Kill other processes
                    for p in processes.values():
                        if p.poll() is None:
                            p.terminate()
                    return
        
        # Check rollout directory periodically
        rollout_dir = Path("/outputs/wordle-rl-4xa100-80gb/rollouts")
        if rollout_dir.exists():
            steps = list(rollout_dir.glob("step_*"))
            if steps:
                latest = max(steps, key=lambda p: int(p.name.split("_")[1]))
                files = list(latest.glob("*.pt"))
                if files:
                    print(f"\n[INFO] Found rollout files in {latest}: {len(files)} files")

if __name__ == "__main__":
    with app.run():
        run_training_debug()
```

## Quick Fixes to Try

### 1. Synchronous Mode
Remove async training complexity:
```toml
# In orch.toml
async_level = 0  # Instead of 2
```

### 2. Simplified Configuration
Start with minimal setup:
```bash
uv run rl \
  --trainer @ examples/wordle/rl/train.toml \
  --orchestrator @ examples/wordle/rl/orch.toml \
  --inference @ examples/wordle/rl/infer.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 1 \
  --inference.parallel.dp 1 \
  --orchestrator.batch_size 8 \
  --orchestrator.micro_batch_size 8
```

### 3. Debug with Fake Data
Test trainer in isolation:
```toml
# In train.toml
[fake_data_loader]
batch_size = 8
micro_batch_size = 8
seq_len = 512
```

This will help identify if the issue is with the trainer or the data pipeline.

## Summary

The hang is almost certainly due to the orchestrator not successfully writing rollout files. The debugging steps above will help identify which component is failing and why. Focus on:

1. Capturing logs from ALL components
2. Checking file system operations
3. Verifying component health
4. Monitoring rollout directory creation

Once you identify which component is failing, the logs should provide specific error messages to resolve the issue.