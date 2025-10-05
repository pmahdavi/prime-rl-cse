# Modal Deployment Optimizations

Based on Modal documentation deep dive (October 2025)

## Summary of Potential Improvements

### 1. ⚡ Memory Snapshots (IMPLEMENT NOW - High Impact)

**Impact:** Reduce cold starts from 2-5 minutes → 10-30 seconds

**Change:** Add `enable_memory_snapshot=True` to function decorator

```python
@app.function(
    image=prime_rl_image,
    gpu="H100:8",
    enable_memory_snapshot=True,  # ← ADD THIS
    ...
)
```

**Why it works:** 
- PyTorch imports take 20,000+ file operations
- Memory snapshot captures post-import state
- Subsequent containers restore from snapshot instantly

**Docs:** https://modal.com/docs/guide/memory-snapshot

---

### 2. 🏗️ Refactor to Class-Based Pattern (RECOMMENDED)

**Impact:** Better code organization + enables memory snapshots for setup

**Current pattern (function-based):**
```python
@app.function(...)
def train_prime_rl(...):
    # Setup runs on every container start
    subprocess.run(["git", "pull"])  # Slow!
    # Training logic
```

**Better pattern (class-based with @modal.enter):**
```python
@app.cls(
    image=prime_rl_image,
    gpu="H100:8",
    enable_memory_snapshot=True,
)
class PrimeRLTrainer:
    @modal.enter()  # Runs once, included in snapshot
    def setup(self):
        # Do expensive setup ONCE
        if not self.use_local_code:
            os.chdir("/root/prime-rl-cse")
            subprocess.run(["git", "fetch", "origin"], check=True)
            subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    
    @modal.method()
    def train(self, experiment_name, ...):
        # Training logic - reuses warm container
        pass
```

**Benefits:**
- Setup captured in memory snapshot
- Container reuse across multiple training runs
- Cleaner separation of concerns

**Docs:** https://modal.com/docs/guide/lifecycle-functions

---

### 3. 🔥 Keep Containers Warm (OPTIONAL - Cost vs Speed)

**Impact:** Zero cold starts (but costs money for idle containers)

```python
@app.function(
    ...
    keep_warm=1,  # Always keep 1 container ready
)
```

**When to use:**
- Production services needing instant response
- When cold start latency is critical
- When idle cost < business value

**Cost:** ~$3.70/hour per A100 (even when idle)

**Docs:** https://modal.com/docs/guide/cold-start#keep-containers-warm

---

### 4. 🌐 Multi-Node Training with @clustered (FUTURE)

**Status:** Private beta - requires contacting Modal

**Benefit:** Native multi-node support with:
- 3200 Gbps RDMA networking (RoCE)
- Auto-configured rank/world_size
- No manual torchrun coordination

```python
import modal.experimental

@app.function(gpu="H100:8")
@modal.experimental.clustered(size=4)  # 32 H100s total
def train_distributed():
    cluster_info = modal.experimental.get_cluster_info()
    # cluster_info.rank, cluster_info.world_size, etc.
```

**Docs:** https://modal.com/docs/guide/multi-node-training

**Action:** Contact Modal via Slack if you need >8 GPUs

---

### 5. 📦 Optimize Model Caching

**Current:** Models downloaded in image build or at runtime

**Better:** Use Volumes for persistent model cache

```python
model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(
    volumes={
        "/cache": model_cache,
        ...
    }
)
def train():
    # HF_HUB_CACHE="/cache/huggingface" already set
    # Models cached across all runs
```

**Why:** 
- Volume persists across image rebuilds
- Shared across all functions
- No re-download on code changes

**Docs:** https://modal.com/docs/guide/model-weights

---

### 6. 🎛️ Concurrent Inputs for Throughput

**Current:** Single training job per function call

**Future use case:** If you want to run parallel experiments

```python
@app.function(
    concurrency_limit=10,  # Allow 10 parallel executions
    allow_concurrent_inputs=5,  # Each container handles 5 jobs
)
```

**Not applicable now** (training is single-threaded), but useful if you add:
- Parallel hyperparameter search
- Multiple eval jobs
- Batch inference endpoints

**Docs:** https://modal.com/docs/guide/concurrent-inputs

---

## Priority Implementation Plan

### Phase 1: Quick Wins (30 minutes)
1. ✅ Add `enable_memory_snapshot=True` to `train_prime_rl`
2. ✅ Test cold start improvement
3. ✅ Document in README

### Phase 2: Refactoring (2-3 hours)
1. Convert function-based to class-based pattern
2. Move git pull to `@modal.enter()` 
3. Test with multiple training runs
4. Compare cold start times

### Phase 3: Advanced (When needed)
1. Contact Modal for multi-node beta access
2. Implement `@clustered` for >8 GPU training
3. Add `keep_warm` for production deployments

---

## Quick Implementation: Memory Snapshots

**File:** `modal/deploy.py`

**Change:** Line 88-99

```python
@app.function(
    image=prime_rl_image,
    gpu="H100:8",
    cpu=16.0,
    memory=65536,
    volumes={
        "/outputs": outputs_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    timeout=86400,
    enable_memory_snapshot=True,  # ← ADD THIS LINE
)
def train_prime_rl(...):
```

**Expected improvement:**
- First run: ~same (2-5 min cold start)
- Subsequent runs: ~10-30 sec cold start
- Especially valuable when running multiple experiments back-to-back

---

## Testing the Improvements

```bash
# Test 1: Measure baseline cold start
time modal run modal/deploy.py --gpu-count 2

# Test 2: Add memory snapshot, measure improvement
# (After adding enable_memory_snapshot=True)
time modal run modal/deploy.py --gpu-count 2

# Test 3: Run twice in succession
modal run modal/deploy.py --gpu-count 2
modal run modal/deploy.py --gpu-count 2  # Should be much faster

# Check logs for snapshot creation
modal app logs | grep -i snapshot
```

---

## Cost-Benefit Analysis

| Optimization | Dev Time | Cost Impact | Speedup | Priority |
|--------------|----------|-------------|---------|----------|
| Memory Snapshots | 5 min | None | 10-20x | ⭐⭐⭐ DO NOW |
| Class-based pattern | 2 hrs | None | 2-3x | ⭐⭐ RECOMMENDED |
| keep_warm | 1 min | +100% | ∞ | ⭐ SITUATIONAL |
| @clustered | N/A | Requires beta | N/A | 💤 FUTURE |

---

## References

- [Memory Snapshot](https://modal.com/docs/guide/memory-snapshot)
- [Cold Start Performance](https://modal.com/docs/guide/cold-start)
- [Multi-node Training](https://modal.com/docs/guide/multi-node-training)
- [Model Weights Storage](https://modal.com/docs/guide/model-weights)
- [Container Lifecycle](https://modal.com/docs/guide/lifecycle-functions)
- [Concurrent Inputs](https://modal.com/docs/guide/concurrent-inputs)
