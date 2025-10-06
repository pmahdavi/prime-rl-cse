# Debugging Modal Training Hang Issue

## Problem Summary
The training process hangs at "Waiting for training batch to arrive" - this indicates a communication issue between the three PRIME-RL components (trainer, orchestrator, inference).

## Root Causes Analysis

### 1. Component Communication Breakdown
The PRIME-RL architecture has three components that must communicate:
- **Trainer**: Waiting for rollout data (stuck here)
- **Orchestrator**: Should be collecting data from inference and sending to trainer
- **Inference**: Should be generating rollouts

### 2. Common Causes

#### A. Inference Server Not Running Properly
- The inference server might have crashed or failed to start
- vLLM initialization issues (GPU memory, model loading)

#### B. Orchestrator Connection Issues
- Network connectivity between components
- Incorrect endpoints in configuration
- File system permissions for shared rollout directory

#### C. Environment/Verifier Issues
- The Wordle environment might not be installed
- Verifier initialization failure

## Debugging Steps

### Step 1: Check All Component Logs
The current log only shows trainer output. You need to check all three components:

```bash
# In your Modal setup, ensure you're capturing logs from all processes
# Modify the rl.py to show logs from all components, not just trainer
```

### Step 2: Verify Component Status
Add health checks to your Modal script:

```python
# Add after starting each component
def check_component_health(process_name, process):
    time.sleep(5)  # Give it time to start
    if process.poll() is not None:
        print(f"ERROR: {process_name} crashed with code {process.returncode}")
        # Capture stderr/stdout
```

### Step 3: Enable Debug Logging
Modify your command to include debug flags:

```bash
uv run rl \
  --trainer @ examples/wordle/rl/train.toml \
  --orchestrator @ examples/wordle/rl/orch.toml \
  --inference @ examples/wordle/rl/infer.toml \
  --output-dir /outputs/wordle-rl-4xa100-80gb \
  --inference.parallel.dp 2 \
  --inference.parallel.tp 1 \
  --wandb.project prime-rl-wordle \
  --wandb.name wordle-rl-4xa100-80gb \
  --trainer-gpu-ids 0,1 \
  --inference-gpu-ids 2,3 \
  --log-level DEBUG  # Add this
```

### Step 4: Check Network Configuration
For Modal, ensure proper network setup:

```python
# In your Modal app configuration
@modal.app(
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        # Add any other required secrets
    ],
    network_file_systems={
        "/outputs": modal.NetworkFileSystem.from_name("prime-rl-outputs"),
    }
)
```

### Step 5: Verify Environment Installation
Check if the Wordle environment is properly installed:

```bash
# Add this check before starting training
uv run python -c "from verifiers import get_verifier; v = get_verifier('wordle'); print('Wordle env OK')"
```

### Step 6: Add Component Communication Test
Before starting the full training, test component communication:

```python
def test_component_communication():
    """Test that all components can communicate"""
    # Start inference server
    inference_proc = start_inference(...)
    time.sleep(10)
    
    # Test inference endpoint
    import requests
    try:
        resp = requests.get("http://localhost:8000/health")
        print(f"Inference health: {resp.status_code}")
    except Exception as e:
        print(f"Inference not responding: {e}")
    
    # Similar tests for orchestrator
```

### Step 7: Monitor Resource Usage
The hang might be due to resource constraints:

```python
# Add resource monitoring
import psutil
import GPUtil

def log_resources():
    # CPU and Memory
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")
    
    # GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal}MB")
```

## Quick Fixes to Try

### 1. Reduce Batch Size
In your config files, try reducing batch sizes to ensure it's not a memory issue:
```toml
# In train.toml
batch_size = 8  # Reduce from default
micro_batch_size = 2
```

### 2. Disable Async Training
Try synchronous mode first to isolate the issue:
```toml
# In orch.toml
async_level = 0  # Instead of default 2
```

### 3. Use Single GPU Mode
Test with simplified setup:
```bash
uv run rl \
  --trainer @ examples/wordle/rl/train.toml \
  --orchestrator @ examples/wordle/rl/orch.toml \
  --inference @ examples/wordle/rl/infer.toml \
  --inference-gpu-ids 0 \
  --trainer-gpu-ids 1 \
  --inference.parallel.dp 1 \
  --inference.parallel.tp 1
```

### 4. Add Timeout and Retry Logic
Modify the trainer to timeout and provide more info:
```python
# In trainer code where it waits for batch
timeout = 300  # 5 minutes
start_time = time.time()
while not batch_available():
    if time.time() - start_time > timeout:
        raise TimeoutError("No batch received in 5 minutes. Check orchestrator/inference logs.")
    time.sleep(1)
```

## Modal-Specific Considerations

### 1. Shared File System
Ensure Modal's network file system is properly mounted and accessible:
```python
# Test file system access
test_file = "/outputs/test_write.txt"
with open(test_file, "w") as f:
    f.write("test")
print(f"File system write test: {'OK' if os.path.exists(test_file) else 'FAILED'}")
```

### 2. GPU Allocation
Modal might be having issues with GPU allocation:
```python
# Verify GPU visibility
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
```

### 3. Process Management
Modal's process management might need adjustment:
```python
# Use Modal's process management instead of subprocess
@modal.function(gpu="A100", cpu=8)
def run_component(component_type, config_path, gpu_ids):
    # Run component with proper error handling
    pass
```

## Recommended Debugging Approach

1. **First**: Modify your Modal script to capture and display logs from ALL components, not just the trainer
2. **Second**: Add health checks and timeouts to identify which component is failing
3. **Third**: Test components individually before running the full pipeline
4. **Fourth**: Enable debug logging and add instrumentation

## Example Enhanced Modal Script Structure

```python
import modal
import subprocess
import time
import sys

@modal.function(gpu="A100", cpu=16, timeout=3600)
def run_training_with_debugging():
    # Pre-flight checks
    check_environment()
    check_file_system()
    check_gpu_allocation()
    
    # Start components with monitoring
    processes = {}
    
    # Start inference with logging
    print("Starting inference server...")
    processes['inference'] = start_component_with_logging(
        "inference", 
        ["uv", "run", "inference", "@", "configs/debug/infer.toml"],
        log_prefix="[INFERENCE]"
    )
    
    # Wait and check health
    time.sleep(10)
    if not check_inference_health():
        print("ERROR: Inference server failed to start")
        print_component_logs(processes['inference'])
        return
    
    # Similar for orchestrator and trainer...
    
def start_component_with_logging(name, cmd, log_prefix):
    """Start a component and capture its output"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Start thread to capture and print output
    import threading
    def log_output():
        for line in process.stdout:
            print(f"{log_prefix} {line.rstrip()}")
    
    thread = threading.Thread(target=log_output)
    thread.daemon = True
    thread.start()
    
    return process
```

This debugging guide should help identify where the training pipeline is breaking down.