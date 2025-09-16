from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    WeightCheckpointConfig,
)
from prime_rl.utils.config import LogConfig, WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings


class BaseDataConfig(BaseModel):
    """Base config for SFT data."""

    micro_batch_size: Annotated[int, Field(ge=1)] = 8
    batch_size: Annotated[int, Field(ge=1)] = 128
    seq_len: Annotated[int, Field(ge=1)] = 128
    num_examples: Annotated[
        int | None, Field(description="Number of examples to use from the dataset. If None, will use all examples.")
    ] = None
    pack_function: Literal["cat", "stack"] = "cat"

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class FakeDataConfig(BaseDataConfig):
    """Configures fake data used for debugging."""

    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    input_ids: Literal["increasing", "random"] = "increasing"


class SFTDataConfig(BaseDataConfig):
    """Configures the data used for training."""

    type: Literal["sft"] = "sft"

    name: Annotated[str, Field(description="Name or path of the HF dataset to use.")] = (
        "PrimeIntellect/Reverse-Text-SFT"
    )
    splits: Annotated[list[str], Field(description="Splits to use from the HF dataset.")] = ["train"]
    shuffle: Annotated[bool, Field(description="Whether to shuffle the dataset at the beginning of each epoch.")] = True


DataConfigType: TypeAlias = FakeDataConfig | SFTDataConfig


class SFTTrainerConfig(BaseSettings):
    """Configures the SFT trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The data configuration
    data: Annotated[DataConfigType, Field(discriminator="type")] = SFTDataConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # The weight checkpoint configuration
    weights: WeightCheckpointConfig = WeightCheckpointConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbMonitorConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(description="Maximum number of steps to run training for. If None, will run indefinitely."),
    ] = None

    memory_profiler_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5 and use fake data.",
        ),
    ] = False

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if self.wandb:  # Do not log extras
                self.wandb.log_extras = None
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def validate_scheduler(self):
        # Constant scheduler does not require any validation/ setup
        if self.scheduler.type == "constant":
            return self

        # Must specify max_steps when using a scheduler other than `constant`
        if self.max_steps is None:
            raise ValueError("Must specify max_steps when using a scheduler other than `constant`")

        # If decay_steps is not specified, use remaining steps after warmup
        if self.scheduler.decay_steps is None:
            if not (self.scheduler.warmup_steps <= self.max_steps):
                raise ValueError("config.scheduler.warmup_steps must be less than or equal to config.max_steps")

            self.scheduler.decay_steps = self.max_steps - self.scheduler.warmup_steps
            assert self.scheduler.decay_steps >= 0, "config.scheduler.decay_steps must be positive"

        # If decay_steps is specified, validate it
        else:
            if not (self.scheduler.warmup_steps + self.scheduler.decay_steps <= self.max_steps):
                raise ValueError(
                    "config.scheduler.warmup_steps + config.scheduler.decay_steps must be less than or equal to config.max_steps"
                )

        return self

    @model_validator(mode="after")
    def disable_logging_wandb_samples(self):
        if self.wandb and self.wandb.log_extras:
            self.wandb.log_extras.samples = False
        return self

    @model_validator(mode="after")
    def validate_pack_function(self):
        if self.model.cp > 1 and self.data.pack_function != "stack":
            raise ValueError("Packing function must be 'stack' when CP is enabled")
        return self
