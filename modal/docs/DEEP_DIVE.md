# Modal Deep Dive: Advanced Features for RL Training

## 🧠 Memory Snapshots Explained (The Secret Weapon)

### What Are Memory Snapshots?

Memory snapshots are Modal's way of **freezing the state of your container after initialization** and restoring it instantly on subsequent runs. Think of it like "save state" in video games!

### The Problem They Solve

**Without memory snapshots:**
```python
Container starts
↓ 30 seconds: Import PyTorch (20,000+ file operations!)
↓ 45 seconds: Import transformers, flash-attn, etc.
↓ 60 seconds: Load model to memory
↓ 10 seconds: CUDA initialization
--- Total: ~2-5 minutes ---
↓ 5 seconds: Actually run your training code
```

**With memory snapshots:**
```python
Container starts
↓ 10 seconds: Restore from snapshot (everything already loaded!)
↓ 5 seconds: Actually run your training code
--- Total: ~15 seconds ---
```

### How It Works Internally

1. **First run** (snapshot creation):
   - Container boots normally
   - All imports execute
   - Models load
   - `@modal.enter()` hooks run
   - **Modal takes a snapshot of RAM**
   - Snapshot cached for future use

2. **Subsequent runs** (snapshot restore):
   - Container boots with minimal environment
   - **Entire RAM state restored from snapshot** (10-30 sec)
   - Code picks up as if initialization just finished
   - No imports, no model loading, no CUDA init!

### CPU vs GPU Memory Snapshots

#### CPU Memory Snapshots (What We're Using)
```python
@app.function(enable_memory_snapshot=True, gpu="H100:8")
```

**What gets snapshotted:**
- ✅ Python interpreter state
- ✅ Imported modules (PyTorch, transformers, etc.)
- ✅ CPU-side data structures
- ✅ Models loaded in CPU RAM
- ❌ GPU memory (VRAM)

**GPU availability:**
- ❌ Not available during snapshot creation (`@modal.enter(snap=True)`)
- ✅ Available after snapshot restore

**For our use case:** Perfect! We don't preload models onto GPU during init anyway.

#### GPU Memory Snapshots (Alpha Feature)
```python
@app.function(enable_gpu_snapshot=True, gpu="A10")
```

**What gets snapshotted:**
- ✅ Everything from CPU snapshots
- ✅ GPU memory (VRAM)
- ✅ CUDA contexts
- ✅ Models already on GPU

**GPU availability:**
- ✅ Available during snapshot creation (`@modal.enter(snap=True)`)
- ✅ Can preload models onto GPU before snapshot

**Status:** Alpha (opt-in), best for inference workloads

### Two-Stage Initialization Pattern (for GPU workloads)

From the docs, you can split initialization into two phases:

```python
@app.cls(enable_memory_snapshot=True, gpu="H100:8")
class PrimeRLTrainer:
    @modal.enter(snap=True)  # Runs BEFORE snapshot, NO GPU
    def setup_no_gpu(self):
        import torch  # Heavy imports here
        import transformers
        self.config = load_config()
        # No GPU operations!
    
    @modal.enter(snap=False)  # Runs AFTER snapshot restore, GPU available
    def setup_with_gpu(self):
        import torch
        self.model = load_model()
        self.model = self.model.cuda()  # Now GPU is available!
    
    @modal.method()
    def train(self):
        # Training code with full GPU access
        pass
```

**Why this pattern?**
- Snapshot captures expensive CPU-side work (imports)
- GPU operations run after restore (when GPU is available)
- Best of both worlds!

---

## 🎯 Container Lifecycle Hooks (Advanced Patterns)

### The Full Lifecycle

```python
@app.cls(
    image=image,
    gpu="H100:8",
    enable_memory_snapshot=True,
)
class AdvancedTrainer:
    # ┌─── Container Starts ───┐
    
    @modal.build()  # [1] Runs during IMAGE BUILD (once per image version)
    def download_weights():
        # Downloads models into image
        # Only runs when you rebuild the image
        pass
    
    @modal.enter(snap=True)  # [2] Runs BEFORE snapshot (no GPU for CPU snapshots)
    def heavy_initialization(self):
        # Expensive imports
        import torch
        import transformers
        # Load configs, setup logging, etc.
        self.logger = setup_logger()
    
    # ┌─── SNAPSHOT TAKEN HERE ───┐
    
    @modal.enter(snap=False)  # [3] Runs AFTER snapshot restore (GPU available)
    def gpu_initialization(self):
        # GPU-dependent setup
        self.model = load_model().cuda()
    
    @modal.enter()  # [4] Runs AFTER all enter hooks (default: snap=True)
    def final_setup(self):
        # Any final setup before serving requests
        self.ready = True
    
    # ┌─── Container Ready for Requests ───┐
    
    @modal.method()
    def train(self, ...):
        # Your actual training code
        # Container is reused for multiple calls
        pass
    
    @modal.exit()  # [5] Runs when container is shutting down
    def cleanup(self):
        # Save checkpoints, close connections, etc.
        # Called even during preemption (30 sec grace period)
        if hasattr(self, 'checkpoint'):
            self.checkpoint.save()
```

### Why Class-Based > Function-Based

**Function-based (current):**
```python
@app.function(...)
def train_prime_rl(...):
    # Setup runs on EVERY container start
    subprocess.run(["git", "pull"])  # Slow!
    # Training logic
```

**Class-based (better):**
```python
@app.cls(...)
class PrimeRLTrainer:
    @modal.enter()
    def setup(self):
        # Setup runs ONCE per container
        subprocess.run(["git", "pull"])
    
    @modal.method()
    def train(self, ...):
        # Training logic
        # Container reused for multiple train() calls!
```

**Benefits:**
- Setup captured in snapshot
- Container reuse across training runs
- Cleaner code organization
- Better resource utilization

---

## 🔁 Retries & Fault Tolerance (Production-Ready)

### Smart Retry Configuration

```python
import modal

@app.function(
    retries=modal.Retries(
        max_retries=5,
        initial_delay=1.0,        # Start with 1 second
        backoff_coefficient=2.0,  # Double each time
        max_delay=60.0,           # Cap at 60 seconds
    )
)
def resilient_training(...):
    # If this fails, Modal will:
    # - Retry 1: wait 1s, retry
    # - Retry 2: wait 2s, retry
    # - Retry 3: wait 4s, retry
    # - Retry 4: wait 8s, retry
    # - Retry 5: wait 16s, retry
    # After 5 failures: propagate exception
    pass
```

### What Gets Retried Automatically

| Failure Type | Behavior | Use Retries? |
|--------------|----------|--------------|
| **Function exception** | Caught, logged | ✅ Yes |
| **Container crash** | Auto-rescheduled | ✅ Yes (infinite for deployed) |
| **Out of memory** | Container killed, rescheduled | ✅ Yes |
| **GPU failure** | Auto-detected, rescheduled | ✅ Yes |
| **Preemption** | Graceful shutdown, rescheduled | ⚠️  No (not a failure) |

### Handling Partial Failures in Map Operations

```python
@app.function(retries=3)
def train_on_batch(batch):
    if should_fail():
        raise ValueError("Batch failed!")
    return process(batch)

# Option 1: Stop on first failure (default)
results = list(train_on_batch.map([1, 2, 3, 4, 5]))
# Raises ValueError on first failure

# Option 2: Collect all results + exceptions
results = list(train_on_batch.map(
    [1, 2, 3, 4, 5],
    return_exceptions=True
))
# Returns: [result1, ValueError(...), result3, ...]
# You can filter/handle exceptions after
```

---

## 💾 Volumes & Data Persistence (Critical for RL)

### Volume Types & Use Cases

#### 1. **Persistent Volumes** (For Checkpoints & Model Weights)
```python
ckpt_vol = modal.Volume.from_name("checkpoints", create_if_missing=True)
model_vol = modal.Volume.from_name("models", create_if_missing=True)

@app.function(
    volumes={
        "/checkpoints": ckpt_vol,
        "/models": model_vol,
    }
)
def train():
    # Writes here persist across runs!
    torch.save(model.state_dict(), "/checkpoints/step_100.pt")
    # Volume automatically synced to cloud storage
```

**Features:**
- Persists across all runs (even image rebuilds!)
- Shared across all containers in workspace
- Automatically synced to cloud storage
- Versioned (can rollback)

**Best for:**
- Model checkpoints
- Training state
- Cached model weights
- Any data you need to survive container restarts

#### 2. **NetworkFileSystem (NFS) Volumes** (For High-Performance Shared State)
```python
nfs_vol = modal.NetworkFileSystem.from_name("training-cache", create_if_missing=True)

@app.function(
    network_file_systems={"/cache": nfs_vol}
)
def distributed_training():
    # All containers see the SAME file system in real-time!
    # Perfect for distributed training coordination
    pass
```

**Features:**
- Real-time synchronization across containers
- Lower latency than regular Volumes
- Shared state for multi-GPU/multi-node training

**Best for:**
- Distributed training coordination
- Shared cache across workers
- Real-time data sharing

#### 3. **Cloud Bucket Mounts** (For Massive Datasets)
```python
# Mount S3 bucket directly
@app.function(
    cloud_bucket_mounts={
        "/data": modal.CloudBucketMount(
            bucket_name="my-training-data",
            bucket_endpoint_url="https://s3.amazonaws.com",
            read_only=True,  # Optimize for read performance
        )
    }
)
def train_on_s3_data():
    # Read directly from S3 (no copy needed!)
    data = load_from_path("/data/train.parquet")
```

**Features:**
- No data copying (direct streaming)
- Works with S3, GCS, Azure Blob
- Read-only mode optimizes performance

**Best for:**
- Huge datasets (TBs+)
- Data that's already in cloud storage
- Read-heavy workloads

### Volume Commit Strategy

```python
ckpt_vol = modal.Volume.from_name("checkpoints", create_if_missing=True)

@app.function(volumes={"/checkpoints": ckpt_vol})
def train_with_checkpointing(num_steps):
    for step in range(num_steps):
        train_one_step()
        
        if step % 100 == 0:
            # Save checkpoint locally (fast)
            torch.save(model, f"/checkpoints/step_{step}.pt")
            
            # Explicitly sync to cloud (slower but durable)
            ckpt_vol.commit()  # Blocks until synced
            # Now safe even if container crashes!
```

**Strategies:**
- **Frequent local writes**: Fast, in-memory
- **Periodic commits**: Durable, synchronized
- **Auto-commit on exit**: Happens automatically

---

## 🚀 Advanced Performance Features

### 1. Keep-Warm Containers (Zero Cold Starts)

```python
@app.function(
    gpu="H100:8",
    keep_warm=2,  # Always keep 2 containers warm
    # OR
    keep_warm=modal.warm_pool(
        size=2,         # 2 warm containers
        idle_timeout=300,  # Kill after 5min idle
    )
)
def instant_inference():
    # First request hits instantly (no cold start!)
    pass
```

**Cost:** Pays for idle containers
**Use when:** Response time > cost (production APIs)

### 2. Concurrent Inputs (Higher Throughput)

```python
@app.function(
    concurrency_limit=100,  # Max 100 parallel executions
    allow_concurrent_inputs=10,  # Each container handles 10 requests
)
def high_throughput_inference(x):
    # Container processes 10 requests concurrently
    # Total capacity: 100 concurrent executions
    return model(x)
```

**For training:** Probably not useful (training is sequential)
**For eval:** Very useful! Run many eval jobs in parallel

### 3. Batch Processing with Dynamic Batching

```python
@app.function(
    max_batch_size=32,
    batch_wait_time_ms=100,
)
def batched_inference(inputs):
    # Modal automatically batches requests
    # Waits up to 100ms to fill batch of 32
    return model(torch.stack(inputs))
```

**Use for:** Inference endpoints (not training)

### 4. Global Variables (Shared State Pattern)

```python
# BAD: Don't do this
@app.function()
def train():
    model = load_model()  # Reloads on EVERY call!
    train_one_step(model)

# GOOD: Use class with @modal.enter
@app.cls()
class Trainer:
    @modal.enter()
    def load_once(self):
        self.model = load_model()  # Loads ONCE
    
    @modal.method()
    def train(self):
        train_one_step(self.model)  # Reuses loaded model
```

### 5. Preemption Handling (Spot Instances)

```python
@app.cls()
class ResilientTrainer:
    @modal.exit()
    def handle_preemption(self):
        # Called when container is preempted
        # You have 30 seconds to clean up!
        
        print("⚠️  Preemption detected! Saving checkpoint...")
        self.save_checkpoint()
        print("✅ Checkpoint saved. Safe to terminate.")
```

**When it happens:**
- Spot instance reclaimed
- Container crashes
- Function timeout
- Manual termination

---

## 🎨 Recommended Architecture for Prime-RL

Based on all the above, here's the optimal Modal architecture for prime-rl:

```python
import modal

app = modal.App("prime-rl-optimized")

# Volumes for persistent data
checkpoints = modal.Volume.from_name("prime-rl-checkpoints", create_if_missing=True)
model_cache = modal.Volume.from_name("prime-rl-models", create_if_missing=True)
wandb_cache = modal.Volume.from_name("prime-rl-wandb", create_if_missing=True)

# NFS for real-time sharing (multi-node coordination)
shared_state = modal.NetworkFileSystem.from_name("prime-rl-shared", create_if_missing=True)

# Image with all dependencies
prime_rl_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    # ... (build steps as before)
)

@app.cls(
    image=prime_rl_image,
    gpu="H100:8",
    cpu=16.0,
    memory=65536,
    volumes={
        "/checkpoints": checkpoints,
        "/models": model_cache,
        "/wandb": wandb_cache,
    },
    network_file_systems={"/shared": shared_state},
    enable_memory_snapshot=True,  # Fast cold starts
    timeout=86400,  # 24 hours
    retries=modal.Retries(
        max_retries=3,
        initial_delay=60,  # Training failures need longer delays
        backoff_coefficient=2.0,
    ),
)
class PrimeRLTrainer:
    
    @modal.enter(snap=True)  # Before snapshot, no GPU
    def setup_cpu_heavy(self):
        """Load heavy Python modules (captured in snapshot)"""
        import torch
        import transformers
        from prime_rl import trainer
        
        # Git pull (captured in snapshot if not using local code)
        if not self.use_local_code:
            import subprocess, os
            os.chdir("/root/prime-rl-cse")
            subprocess.run(["git", "fetch", "origin"], check=True)
            subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
        
        self.logger = self.setup_logging()
    
    @modal.enter(snap=False)  # After snapshot, GPU available
    def setup_gpu_dependent(self):
        """Initialize GPU-dependent components"""
        import torch
        # CUDA is now available!
        self.device_count = torch.cuda.device_count()
        print(f"✅ {self.device_count} GPUs available")
    
    @modal.method()
    def train(self, config_dict: dict, step: int) -> dict:
        """Run one training step"""
        # Training logic here
        # Container is reused across multiple train() calls!
        
        # Periodic checkpoint commit
        if step % 100 == 0:
            checkpoints.commit()  # Sync to cloud storage
        
        return {"step": step, "loss": 0.5}
    
    @modal.exit()
    def cleanup(self):
        """Graceful shutdown (30 sec grace period)"""
        print("🛑 Container shutting down...")
        checkpoints.commit()  # Final sync
        print("✅ Checkpoints saved")

@app.local_entrypoint()
def main(use_local_code: bool = False):
    """Local entry point"""
    trainer = PrimeRLTrainer(use_local_code=use_local_code)
    
    # Run multiple training steps (container reused!)
    for step in range(1000):
        result = trainer.train.remote(config={}, step=step)
        print(f"Step {step}: loss={result['loss']}")
```

### Why This Architecture Wins

1. **Memory snapshots**: 10-20x faster cold starts
2. **Class-based**: Container reuse across training steps
3. **Two-stage init**: CPU work before snapshot, GPU after
4. **Smart retries**: Exponential backoff for transient failures
5. **Persistent volumes**: Checkpoints survive everything
6. **NFS for coordination**: Multi-node can share state
7. **Graceful shutdown**: Saves checkpoints even on preemption

---

## 📊 Feature Comparison & Recommendations

| Feature | Current Implementation | Recommended | Priority | Effort |
|---------|------------------------|-------------|----------|--------|
| **Memory snapshots** | ✅ Enabled | ✅ Keep | ⭐⭐⭐ | Done |
| **Class-based pattern** | ❌ Function-based | ✅ Refactor | ⭐⭐⭐ | 3 hours |
| **Two-stage init** | ❌ No | ✅ Add | ⭐⭐ | 1 hour |
| **Smart retries** | ❌ No | ✅ Add | ⭐⭐ | 30 min |
| **Volume commits** | ❌ Auto only | ✅ Explicit | ⭐⭐ | 30 min |
| **Exit hooks** | ❌ No | ✅ Add | ⭐⭐ | 30 min |
| **NFS volumes** | ❌ No | ⚠️  If multi-node | ⭐ | 1 hour |
| **Keep-warm** | ❌ No | ❌ Skip (cost) | ⭐ | 5 min |
| **GPU snapshots** | ❌ No | ⚠️  Future (alpha) | 💤 | N/A |
| **@clustered** | ❌ No | ⚠️  Need beta access | 💤 | N/A |

---

## 🎯 Quick Wins Summary

### Already Implemented ✅
1. Memory snapshots (10-20x faster cold starts)
2. Local dev mode (`--use-local-code`)
3. Build process alignment

### Next 30 Minutes 🔥
1. Add smart retries with exponential backoff
2. Add explicit volume commits after checkpoints
3. Add `@modal.exit()` hook for graceful shutdown

### Next 3 Hours 💪
1. Refactor to class-based pattern
2. Implement two-stage initialization
3. Test container reuse benefits

### Future (When Needed) 🔮
1. Request multi-node beta access from Modal
2. Implement `@clustered` for >8 GPUs
3. Try GPU memory snapshots (when out of alpha)

---

## 📚 References

- [Memory Snapshots](https://modal.com/docs/guide/memory-snapshot)
- [Container Lifecycle](https://modal.com/docs/guide/lifecycle-functions)
- [Retries](https://modal.com/docs/guide/retries)
- [Volumes](https://modal.com/docs/guide/volumes)
- [Network File Systems](https://modal.com/docs/guide/volumes#network-file-systems)
- [Multi-node Clusters](https://modal.com/docs/guide/multi-node-training)
- [Cold Start Performance](https://modal.com/docs/guide/cold-start)
- [Preemption](https://modal.com/docs/guide/preemption)

