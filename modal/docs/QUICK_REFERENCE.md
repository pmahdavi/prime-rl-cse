# Modal Quick Reference for Prime-RL

## 🎯 Cheat Sheet

### Current Setup
```bash
# Production (uses GitHub)
modal run modal/deploy.py

# Development (uses local code)
modal run modal/deploy.py --use-local-code

# Example with configs
modal run modal/deploy.py \
  --trainer-config configs/hendrycks_math/1b/train.toml \
  --orchestrator-config configs/hendrycks_math/1b/orch.toml \
  --inference-config configs/hendrycks_math/1b/infer.toml \
  --gpu-count 8 \
  --trainer-gpu-ratio 0.25
```

### Performance Specs (After Optimizations)

| Scenario | Time | Notes |
|----------|------|-------|
| First run (new image) | 5-10 min | Build + snapshot creation |
| Second run (same image) | 10-30 sec | **Snapshot restore** ⚡ |
| Local dev iteration | 10-30 sec | No git push needed |
| Training (8 H100s) | ~$30/hour | Modal pricing |

---

## 🔧 Common Tasks

### Check Container Logs
```bash
modal app logs
modal app logs --follow  # Tail logs
```

### List Running Containers
```bash
modal app list
```

### Check Volume Contents
```bash
modal volume list prime-rl-checkpoints
modal volume get prime-rl-checkpoints ./local-dir
```

### Kill Running Job
```bash
# Find app ID
modal app list

# Stop it
modal app stop app-xyz123
```

### Download Results
```bash
modal volume get prime-rl-outputs experiment-name ./outputs/
```

---

## 🐛 Debugging

### Problem: Slow Cold Starts
```bash
# Check if snapshots are enabled
grep "enable_memory_snapshot" modal/deploy.py
# Should see: enable_memory_snapshot=True

# Check snapshot creation in logs
modal app logs | grep -i snapshot
```

### Problem: Checkpoints Not Saved
```python
# Make sure you're committing explicitly
checkpoint_volume.commit()  # Add after saving

# Or check auto-commit happened
modal volume list prime-rl-checkpoints
```

### Problem: GPU Not Available
```python
# In @modal.enter(), check snap parameter:

@modal.enter(snap=True)   # NO GPU (before snapshot)
def setup(): 
    # Don't use CUDA here!

@modal.enter(snap=False)  # GPU available (after snapshot)
def setup_gpu():
    # CUDA OK here!
```

### Problem: Out of Memory
```python
# Increase memory allocation
@app.function(
    memory=131072,  # 128GB instead of 64GB
    ...
)

# Or reduce batch size in configs
```

### Problem: Container Crashes
```bash
# Check crash logs
modal app logs | grep -i "error\|crash\|killed"

# Check retries are configured
# Should see in deploy.py:
retries=modal.Retries(...)
```

---

## 📝 Code Patterns

### Pattern: Basic Function
```python
@app.function(
    image=image,
    gpu="H100:8",
    enable_memory_snapshot=True,
)
def my_function(x):
    return process(x)
```

### Pattern: Class with Lifecycle
```python
@app.cls(
    image=image,
    gpu="H100:8",
    enable_memory_snapshot=True,
)
class MyClass:
    @modal.enter()
    def setup(self):
        # Runs once per container
        self.model = load_model()
    
    @modal.method()
    def process(self, x):
        # Can be called multiple times
        return self.model(x)
    
    @modal.exit()
    def cleanup(self):
        # Runs on shutdown
        self.save_state()
```

### Pattern: With Retries
```python
@app.function(
    retries=modal.Retries(
        max_retries=3,
        initial_delay=60.0,
        backoff_coefficient=2.0,
    )
)
def resilient_function():
    # Automatically retried on failure
    pass
```

### Pattern: With Volumes
```python
vol = modal.Volume.from_name("my-vol", create_if_missing=True)

@app.function(volumes={"/data": vol})
def with_volume():
    # Write to volume
    save_file("/data/output.txt")
    
    # Explicit commit
    vol.commit()  # Sync to cloud
```

---

## 📊 Cost Optimization

### GPU Selection
```python
# Development: Cheapest
gpu="T4"       # ~$0.50/hr

# Production: Fast
gpu="A100-40GB:8"  # ~$30/hr
gpu="H100:8"       # ~$40/hr
```

### Keep-Warm (Only for Production)
```python
# Don't use for training (too expensive)
keep_warm=0  # Default, no idle cost

# Use for APIs that need instant response
keep_warm=2  # Keeps 2 containers warm ($$)
```

### Timeout Management
```python
# Set realistic timeouts
timeout=3600  # 1 hour for short jobs
timeout=86400  # 24 hours for long training
```

---

## 🚀 Performance Checklist

- [x] Memory snapshots enabled (`enable_memory_snapshot=True`)
- [ ] Class-based pattern for container reuse
- [ ] Two-stage initialization (CPU before snapshot, GPU after)
- [ ] Smart retries configured
- [ ] Explicit volume commits after checkpoints
- [ ] Exit hooks for graceful shutdown
- [ ] Proper logging and monitoring
- [ ] Config validation

---

## 📚 Documentation Map

```
modal/
├── README.md           ← Start here (usage guide)
├── SUMMARY.md          ← Overview of everything
├── DEEP_DIVE.md        ← Comprehensive feature guide
├── QUICK_WINS.md       ← 30-min improvements
├── QUICK_REFERENCE.md  ← This file (cheat sheet)
├── OPTIMIZATIONS.md    ← Long-term roadmap
└── CHANGELOG.md        ← What changed
```

**Reading Order:**
1. **New to Modal?** → `README.md`
2. **Want details?** → `SUMMARY.md`
3. **Want to optimize?** → `QUICK_WINS.md`
4. **Going deep?** → `DEEP_DIVE.md`
5. **Need command?** → `QUICK_REFERENCE.md` (this file)

---

## 🎯 Common Workflows

### Workflow 1: Quick Experiment
```bash
# 1. Edit code locally
vim src/prime_rl/trainer/rl/loss.py

# 2. Test on Modal with local code
modal run modal/deploy.py --use-local-code

# 3. If good, commit and push
git commit -am "Improved loss function"
git push

# 4. Run production version
modal run modal/deploy.py
```

### Workflow 2: Production Run
```bash
# 1. Deploy to Modal
modal deploy modal/deploy.py

# 2. Check it's running
modal app list

# 3. Monitor logs
modal app logs --follow

# 4. Download results when done
modal volume get prime-rl-outputs exp-name ./outputs/
```

### Workflow 3: Debug Failed Run
```bash
# 1. Check logs
modal app logs > logs.txt
grep -i "error" logs.txt

# 2. Check GPU memory
grep -i "cuda\|memory" logs.txt

# 3. Check checkpoints were saved
modal volume list prime-rl-checkpoints

# 4. Download checkpoint to inspect locally
modal volume get prime-rl-checkpoints step_100 ./
```

---

## 🆘 Emergency Commands

### Container Stuck/Frozen
```bash
# List apps
modal app list

# Force stop
modal app stop <app-id>
```

### Need to Free Up GPU Credits
```bash
# Stop all running apps
modal app list | grep Running | awk '{print $1}' | xargs -n1 modal app stop
```

### Volume Full
```bash
# Check volume size
modal volume list

# Delete old checkpoints (careful!)
# Download first!
modal volume get prime-rl-checkpoints ./backup/
# Then delete via Modal dashboard
```

### Lost Connection During Training
```bash
# Don't panic! Training continues on Modal
# Just reconnect to logs:
modal app logs --follow
```

---

## 📞 Getting Help

1. **Check logs first**: `modal app logs`
2. **Check Modal status**: https://status.modal.com
3. **Search docs**: https://modal.com/docs
4. **Ask on Slack**: https://modal.com/slack
5. **Check our docs**: See `DEEP_DIVE.md` and `SUMMARY.md`

---

## 🎓 Pro Tips

1. **Always use `--use-local-code` for development** - Much faster iteration
2. **Commit volumes explicitly** - Don't rely on auto-commit
3. **Set realistic timeouts** - Training can take hours
4. **Monitor GPU memory** - Add logging in your code
5. **Use class-based pattern** - Better resource utilization
6. **Test locally first** - Catch errors before expensive GPU time
7. **Download important results** - Volumes can be deleted

---

## 🔗 Quick Links

- [Modal Dashboard](https://modal.com/apps)
- [Modal Docs](https://modal.com/docs)
- [Modal Pricing](https://modal.com/pricing)
- [Modal Slack](https://modal.com/slack)
- [Prime-RL Repo](https://github.com/pmahdavi/prime-rl-cse)

---

**Last Updated**: October 2025
**Modal Version**: Latest
**Prime-RL Version**: With memory snapshot optimizations
