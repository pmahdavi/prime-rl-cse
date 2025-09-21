#!/usr/bin/env python3
"""
Simple Modal deployment example for reverse text experiment.
This is the quickest way to test prime-rl on Modal with minimal configuration.
"""

import subprocess
import sys
from pathlib import Path

# Get the project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    """Run reverse text experiment with 2 GPUs (1 training, 1 inference)."""
    
    cmd = [
        "modal", "run",
        str(PROJECT_ROOT / "modal" / "deploy.py"),
        "--trainer-config", "configs/reverse_text/train.toml",
        "--orchestrator-config", "configs/reverse_text/orch.toml", 
        "--inference-config", "configs/reverse_text/infer.toml",
        "--gpu-count", "2",
        "--trainer-gpu-ratio", "0.5",
        "--gpu-type", "A10G",  # Cost-effective for small experiments
        "--wandb-project", "prime-rl-reverse-text",
    ]
    
    print("🚀 Starting reverse text experiment on Modal")
    print("Configuration:")
    print("  - Model: Qwen3-0.6B")
    print("  - GPUs: 2x A10G (1 training, 1 inference)")
    print("  - Estimated time: 10-20 minutes")
    print("  - Estimated cost: ~$0.50")
    print("="*60)
    
    # Run the command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main() 