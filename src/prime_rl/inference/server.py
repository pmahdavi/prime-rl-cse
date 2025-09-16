import os
import subprocess
from typing import Optional

from prime_rl.inference.config import InferenceConfig
from prime_rl.inference.vllm.server import server
from prime_rl.utils.pydantic_config import parse_argv


def _map_gpu_uuids_to_indices(uuid_string: str) -> Optional[str]:
    """Map GPU UUIDs to integer indices using nvidia-smi."""
    try:
        # Get GPU info from nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_mapping = {}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                index, uuid = line.split(",")
                gpu_mapping[uuid.strip()] = index.strip()

        # Map provided UUIDs to indices
        mapped_indices = []
        for uuid in uuid_string.split(","):
            uuid = uuid.strip()
            if uuid in gpu_mapping:
                mapped_indices.append(gpu_mapping[uuid])
            else:
                return None

        return ",".join(mapped_indices)

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _setup_gpu_environment():
    """Configure GPU environment for CUDA applications."""
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_devices:
        return

    # Check if we need to map UUIDs to indices
    if any(char in cuda_devices for char in ["-", ":"]):
        mapped_indices = _map_gpu_uuids_to_indices(cuda_devices)
        if mapped_indices:
            os.environ["CUDA_VISIBLE_DEVICES"] = mapped_indices


def main():
    _setup_gpu_environment()
    config = parse_argv(InferenceConfig, allow_extras=True)
    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
