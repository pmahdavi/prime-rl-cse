#!/usr/bin/env python3
"""
Test script to verify PRIME-RL setup and run a minimal training test.
This helps identify issues before running on Modal.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_command(cmd, description, check=True):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"📝 Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=isinstance(cmd, str),
            capture_output=True, 
            text=True,
            check=check
        )
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[:500])  # First 500 chars
        return True, result
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print("Error:", e.stderr)
        return False, e

def test_environment():
    """Test that the environment is properly set up."""
    
    print("\n" + "="*60)
    print("🚀 PRIME-RL Local Setup Test")
    print("="*60)
    
    # 1. Check Python version
    print("\n1️⃣ Checking Python version...")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 12:
        print(f"✅ Python {python_version.major}.{python_version.minor} - OK")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} - Need Python 3.12+")
        return False
    
    # 2. Check if uv is installed
    print("\n2️⃣ Checking uv installation...")
    success, _ = run_command(["uv", "--version"], "Check uv version", check=False)
    if not success:
        print("📦 Installing uv...")
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh", "Install uv")
        print("⚠️  Please restart your shell or run: source $HOME/.cargo/env")
        return False
    
    # 3. Check CUDA availability
    print("\n3️⃣ Checking CUDA...")
    success, result = run_command(
        ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"],
        "Check CUDA availability",
        check=False
    )
    
    # 4. Install dependencies
    print("\n4️⃣ Installing dependencies...")
    os.chdir("/workspace")
    success, _ = run_command(["uv", "sync", "--all-extras"], "Install dependencies")
    if not success:
        print("💡 Tip: Make sure you have CUDA toolkit installed for flash-attn")
        return False
    
    # 5. Check imports
    print("\n5️⃣ Checking PRIME-RL imports...")
    test_imports = """
import prime_rl
from prime_rl.trainer.rl.train import main as train_main
from prime_rl.orchestrator.orchestrator import main as orch_main
from prime_rl.inference.server import main as inference_main
print("✅ All imports successful!")
"""
    success, _ = run_command(
        ["uv", "run", "python", "-c", test_imports],
        "Test PRIME-RL imports"
    )
    
    # 6. Check Wordle environment
    print("\n6️⃣ Checking Wordle environment...")
    test_wordle = """
try:
    from verifiers import get_verifier
    verifier = get_verifier('wordle')
    print("✅ Wordle environment available!")
except Exception as e:
    print(f"❌ Wordle environment error: {e}")
    print("📦 Try: prime env install wordle")
"""
    run_command(["uv", "run", "python", "-c", test_wordle], "Test Wordle environment", check=False)
    
    return True

def create_test_configs():
    """Create minimal test configurations."""
    
    print("\n7️⃣ Creating test configurations...")
    
    test_dir = Path("/workspace/test_configs")
    test_dir.mkdir(exist_ok=True)
    
    # Minimal trainer config
    trainer_config = """
max_steps = 3
micro_batch_size = 1

[model]
name = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"

[optimizer]
lr = 1e-6

[loss]
type = "grpo"
"""
    (test_dir / "train.toml").write_text(trainer_config)
    
    # Minimal orchestrator config
    orch_config = """
max_steps = 3
seq_len = 512
batch_size = 4
micro_batch_size = 1
rollouts_per_example = 1
async_level = 0

[model]
name = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"

[environment]
id = "wordle"

[sampling]
max_tokens = 128
temperature = 1.0

[client]
base_url = "http://localhost:8000/v1"
"""
    (test_dir / "orch.toml").write_text(orch_config)
    
    # Minimal inference config
    infer_config = """
[model]
name = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"

[parallel]
tp = 1
dp = 1
"""
    (test_dir / "infer.toml").write_text(infer_config)
    
    print("✅ Test configs created in test_configs/")
    return test_dir

def test_components_separately():
    """Test each component can at least start."""
    
    print("\n8️⃣ Testing components individually...")
    
    # Test inference server
    print("\n📡 Testing inference server startup...")
    inference_proc = subprocess.Popen(
        ["uv", "run", "inference", "@", "test_configs/infer.toml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(5)
    if inference_proc.poll() is None:
        print("✅ Inference server started successfully")
        inference_proc.terminate()
        inference_proc.wait()
    else:
        print("❌ Inference server failed to start")
        stderr = inference_proc.stderr.read()
        print("Error:", stderr[:500])
    
    # Test orchestrator dry run
    print("\n🎯 Testing orchestrator...")
    # We can't fully test orchestrator without inference running
    print("⏭️  Skipping full orchestrator test (needs inference server)")
    
    # Test trainer initialization
    print("\n🏋️ Testing trainer initialization...")
    test_trainer = """
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.world import init_world
from prime_rl.utils.logger import setup_logger

# Just test that we can initialize
config = RLTrainerConfig()
print("✅ Trainer config initialized")
"""
    run_command(["uv", "run", "python", "-c", test_trainer], "Test trainer initialization")

def run_minimal_test():
    """Run a minimal test with fake data."""
    
    print("\n9️⃣ Running minimal training test with fake data...")
    
    fake_data_config = """
max_steps = 2
micro_batch_size = 1

[model]
name = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"

[optimizer]
lr = 1e-6

[loss]
type = "grpo"

[fake_data_loader]
batch_size = 2
micro_batch_size = 1
seq_len = 128
"""
    
    Path("/workspace/test_configs/train_fake.toml").write_text(fake_data_config)
    
    # Run trainer with fake data
    cmd = [
        "uv", "run", "torchrun",
        "--nproc-per-node", "1",
        "src/prime_rl/trainer/rl/train.py",
        "@", "test_configs/train_fake.toml",
        "--output-dir", "/workspace/test_output"
    ]
    
    print(f"📝 Command: {' '.join(cmd)}")
    print("\n🏃 Running trainer with fake data...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Trainer ran successfully with fake data!")
        else:
            print("❌ Trainer failed")
            print("Error:", result.stderr[-1000:])  # Last 1000 chars
    except subprocess.TimeoutExpired:
        print("⏱️  Test timed out (this might be normal)")

def main():
    """Run all tests."""
    
    # Test environment setup
    if not test_environment():
        print("\n❌ Environment setup incomplete. Please fix the issues above.")
        return
    
    # Create test configs
    test_dir = create_test_configs()
    
    # Test components
    test_components_separately()
    
    # Run minimal test
    run_minimal_test()
    
    print("\n" + "="*60)
    print("🎉 Setup test complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Fix any ❌ errors above")
    print("2. For full test: uv run rl --trainer @ examples/wordle/rl/train.toml --orchestrator @ examples/wordle/rl/orch.toml --inference @ examples/wordle/rl/infer.toml")
    print("3. For Modal: modal run --detach your_modal_script.py")

if __name__ == "__main__":
    main()