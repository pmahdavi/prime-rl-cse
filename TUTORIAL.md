# Prime-RL Development Tutorial

## Table of Contents
*Estimated Time to Complete: 4-6 hours*

- **Module 1: Foundation** (45 mins)
  - 1.1 Project Overview
  - 1.2 Environment Setup
  - 1.3 First Run
  - 1.4 Project Navigation
- **Module 2: Architecture** (1 hour)
  - 2.1 High-Level Overview
  - 2.2 Module Breakdown
  - 2.3 Data Flow
  - 2.4 Design Patterns
- **Module 3: Development Workflows** (1.5 hours)
  - 3.1 Setup Workflows
  - 3.2 Development Workflows
  - 3.3 Testing Workflows
  - 3.4 Build & Deployment Workflows
- **Module 4: Hands-On Development** (2 hours)
  - 4.1 Code Reading Exercises
  - 4.2 Feature Modification Exercises
  - 4.3 New Feature Development
  - 4.4 Testing & Debugging
- **Module 5: Advanced Topics** (1 hour)
  - 5.1 Performance Optimization
  - 5.2 Multi-Node Training
  - 5.3 Monitoring & Maintenance

---

## Module 1: Foundation

### 1.1 Project Overview

**Purpose**: `prime-rl` is a framework for decentralized Reinforcement Learning (RL) training at scale. It is designed for training large language models on high-performance GPU hardware. The system is architected to be modular, allowing different components to be run and scaled independently.

**Business Context**: This project is aimed at researchers and engineers in the field of AI and Reinforcement Learning who need to train models on large, distributed computing environments. The focus is on performance, scalability, and reproducibility of RL experiments.

**Core Components**:
- **Trainer**: The core training module that handles both Supervised Fine-Tuning (SFT) and Reinforcement Learning. It leverages PyTorch FSDP for distributed training.
- **Inference Server**: A high-performance inference server using `vLLM` to generate text from the current model policy.
- **Orchestrator**: The "brain" of the system that coordinates the training loop, managing data flow between the inference server and the trainer.
- **Environments**: Pluggable modules that define the RL task, including data generation, reward functions, and evaluation metrics.

### 1.2 Environment Setup

**Prerequisites**:
- A Linux environment with NVIDIA GPUs (Ampere, Hopper, or Blackwell architecture recommended).
- NVIDIA drivers and CUDA toolkit installed.
- `git` installed.

**Steps**:

1.  **Clone the Repository**:
    ```bash
    git clone git@github.com:PrimeIntellect-ai/prime-rl.git
    cd prime-rl
    ```

2.  **Run the Quick Installation Script**: This script handles the installation of `uv` (the package manager) and all required Python dependencies.
    ```bash
    curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
    ```
    This will create a virtual environment in a `.venv` directory.

3.  **Activate the Virtual Environment**: The installer will provide instructions on how to activate the environment. It is typically:
    ```bash
    source .venv/bin/activate
    ```
    Or if you installed `uv` manually:
    ```bash
    source $HOME/.local/bin/env
    ```

4.  **Install Pre-commit Hooks (for developers)**: This will ensure your code contributions adhere to the project's linting and formatting standards.
    ```bash
    uv run pre-commit install
    ```

**Validation**:
To verify your setup, run the validation checks described in the `README.md`. For example, check that you can run the SFT trainer in debug mode (requires 1 GPU):
```bash
uv run sft @ configs/debug/sft.toml
```
You should see the trainer initialize and start running without errors.

### 1.3 First Run

Let's execute a simple, end-to-end RL training job. This example trains a small model to reverse a short piece of text. It's a great way to see all the components working together.

**Prerequisites**:
- At least 2 GPUs available.
- Environment successfully set up from the previous step.

**Steps**:

1.  **Execute the `rl` script**: This is the main entry point that starts and coordinates the trainer, orchestrator, and inference server for a complete RL run.
    ```bash
    uv run rl \
      --trainer @ configs/reverse_text/train.toml \
      --orchestrator @ configs/reverse_text/orch.toml \
      --inference @ configs/reverse_text/infer.toml
    ```

**Expected Outputs**:
- You will see logs from the main `rl` process, which primarily shows the trainer's output.
- The script creates an `outputs/` directory (by default). Inside, you will find log files for each component: `trainer.log`, `orchestrator.log`, and `inference.log`.
- After a few minutes, the run should complete successfully.

**Troubleshooting**:
- **`Too many open files` error**: You may need to increase the open file limit.
  ```bash
  ulimit -n 32000
  ```
- **GPU Memory Issues**: Ensure you have at least two GPUs with sufficient VRAM. The debug example is small, but larger experiments will require more resources.

### 1.4 Project Navigation

Here is a map of the key files and directories in the `prime-rl` repository:

- **`/` (Root)**
  - `pyproject.toml`: Defines project metadata, dependencies, and entry points.
  - `README.md`: The main project documentation.
  - `TUTORIAL.md`: This tutorial.
- **`/src/prime_rl/`**: The core Python source code.
  - `rl.py`: Main entry point for orchestrating a full RL run.
  - `/trainer/`: Contains the SFT and RL training logic.
    - `sft/train.py`: SFT training script.
    - `rl/train.py`: RL training script.
  - `/inference/`: Contains the inference server code.
    - `server.py`: The `vLLM` server implementation.
  - `/orchestrator/`: Contains the orchestration logic.
    - `orchestrator.py`: The main orchestrator script.
  - `/eval/`: Code for running evaluations on trained models.
- **`/configs/`**: Contains all the configuration files (`.toml`) for different experiments. This is where you define parameters for models, datasets, training runs, etc.
- **`/scripts/`**: Helper scripts for development and execution.
  - `install.sh`: The installation script.
  - `tmux.sh`: A useful script for setting up a `tmux` session to view logs from all components simultaneously.
- **`/tests/`**: Unit and integration tests.
- **`/outputs/`**: The default directory for logs, checkpoints, and other artifacts from training runs.

---
*End of Module 1*

---

## Module 2: Architecture

### 2.1 High-Level Overview

`prime-rl` is designed as a decoupled, multi-process system to enable scalable Reinforcement Learning. The architecture separates the three main tasks of an RL loop into independent, communicating components:

1.  **Inference Server**: A stateless, high-performance service responsible for generating text (actions) from the current policy (the model).
2.  **Orchestrator**: The stateful coordinator that drives the RL loop. It manages data generation, reward calculation, and communication between the other components.
3.  **Trainer**: The component responsible for updating the model's weights based on the data collected by the orchestrator.

These components communicate primarily through the file system and an HTTP API, which allows them to be scaled independently, even across different machines.

```
+------------------+         +--------------------+         +-----------------+
|                  |---------|                    |---------|                 |
|  Inference       | (HTTP)  |   Orchestrator     | (Files) |   Trainer       |
|  Server (vLLM)   |         |                    |         |   (PyTorch FSDP)|
|                  |---------|                    |---------|                 |
+------------------+         +--------------------+         +-----------------+
        ^                                                            |
        |                  (File System)                             |
        +------------------------------------------------------------+
                           (Model Weight Updates)
```

The entire system is launched and managed by a top-level `rl.py` script, which acts as a process manager.

### 2.2 Module Breakdown

#### `rl.py` (The Launcher)

-   **File**: `src/prime_rl/rl.py`
-   **Responsibilities**:
    -   Parses command-line arguments and configuration files.
    -   Consolidates shared configurations and propagates them to the individual modules.
    -   Launches the `Inference Server`, `Orchestrator`, and `Trainer` as separate subprocesses.
    -   Manages GPU allocation for each subprocess using `CUDA_VISIBLE_DEVICES`.
    -   Monitors the health of the subprocesses and ensures clean shutdown.
    -   Streams the trainer's logs to the main console for user visibility.

#### The Orchestrator

-   **File**: `src/prime_rl/orchestrator/orchestrator.py`
-   **Responsibilities**:
    -   **Main Loop**: Drives the RL training loop step-by-step.
    -   **Data Generation**: Samples prompts from a buffer, sends them to the Inference Server for completion to generate "rollouts".
    -   **Reward Calculation**: Uses a pluggable `Environment` module to calculate rewards for the generated rollouts.
    -   **Advantage Estimation**: Computes advantages (e.g., GAE) based on the rewards.
    -   **Batch Creation**: Prepares and saves training batches (rollout data) to the disk for the Trainer to consume.
    -   **Weight Management**: Monitors for new model weight checkpoints from the Trainer and instructs the Inference Server to load them. This is the key to the on-policy training cycle.
    -   **State Management**: Manages its own state through checkpointing, allowing runs to be resumed.

#### The Inference Server

-   **File**: `src/prime_rl/inference/server.py`
-   **Responsibilities**:
    -   **Model Serving**: Wraps `vLLM` to create a high-throughput, OpenAI-compatible API for text generation.
    -   **Dynamic Weight Loading**: Exposes endpoints that allow the Orchestrator to command it to load new model weights from a specified file path without restarting the server.
    -   **Hardware Abstraction**: Handles the complexities of running inference on single or multiple GPUs (Tensor Parallelism).

#### The Trainer

-   **File**: `src/prime_rl/trainer/rl/train.py`
-   **Responsibilities**:
    -   **Distributed Training**: Sets up and manages the PyTorch distributed environment, using Fully Sharded Data Parallelism (FSDP) to train large models.
    -   **Data Consumption**: Polls the file system for new batches of training data produced by the Orchestrator.
    -   **Model Optimization**: Executes the training step: forward pass, loss calculation (e.g., PPO), backward pass, and optimizer step.
    -   **Checkpointing**:
        -   Saves full checkpoints of the training state (model, optimizer, scheduler) for resuming runs.
        -   Saves lightweight "weight-only" checkpoints that are used by the Orchestrator to update the Inference Server.

### 2.3 Data Flow

The RL training loop follows this data flow:

1.  **Initiation**: The `Orchestrator` starts the loop at `step_n`. It knows the `Trainer` is at `step_n` and the `Inference Server` is using weights from a previous step (e.g., `step_n - async_level`).

2.  **Rollout Generation**:
    -   The `Orchestrator` samples prompts from its buffer.
    -   It makes an **HTTP request** to the `Inference Server` with the prompts.
    -   The `Inference Server` responds with the generated completions (the "rollouts").

3.  **Processing & Batching**:
    -   The `Orchestrator` calculates rewards and advantages for the rollouts.
    -   It packages this data into a structured training batch.
    -   It saves this batch to disk at a specific path: `outputs/rollouts/step_n/rank_i.pt`.

4.  **Training**:
    -   The `Trainer` is waiting for the file `outputs/rollouts/step_n/rank_i.pt` to appear.
    -   Once it appears, the `Trainer` loads the data and performs a training step, updating its model weights. The `Trainer` is now at `step_n+1`.

5.  **Weight Update**:
    -   After the training step, the `Trainer` saves its new weights to disk at `outputs/weights/step_n+1/`.
    -   The `Orchestrator`, which has moved on to `step_n+1`, may now be waiting for this new weight checkpoint (depending on the `async_level`).
    -   Once the `Orchestrator` detects the new weights, it sends an **HTTP request** to the `Inference Server`, telling it to load the weights from `outputs/weights/step_n+1/`.

6.  **Cycle Repeats**: The `Inference Server` now has the updated policy, and the `Orchestrator` can begin generating rollouts for the next step.

### 2.4 Design Patterns

-   **Process-Based Decoupling**: The core components are independent processes. This is a powerful pattern for scalability and fault isolation. The Inference Server can be scaled with more GPUs without affecting the Trainer, and vice-versa.
-   **Configuration as Code (Pydantic)**: The entire system is configured through strongly-typed Pydantic classes. This provides excellent validation, auto-completion, and a clear hierarchy for settings. The system cleverly uses `model_validator` decorators (e.g., in `rl.py`) to propagate shared settings, reducing configuration duplication.
-   **File-System as a Message Queue**: The communication of large data payloads (training batches) from the Orchestrator to the Trainer is done via the file system. This is a simple and effective pattern for handling large data that would be inefficient to send over a typical message bus.
-   **Asynchronous and Non-Blocking I/O**: The `Orchestrator` uses `asyncio` to handle communication with the inference server and other tasks concurrently, making it highly efficient. Both the Trainer and Orchestrator use non-blocking file polling to wait for data and checkpoints.

---
*End of Module 2*

---

## Module 3: Development Workflows

This module documents the standard workflows a developer will use when working with `prime-rl`. Each workflow is a series of steps to accomplish a common development task.

### 3.1 Setup Workflows

#### Workflow: Initial Project Setup

**Purpose**: To set up the development environment from scratch.

**Prerequisites**:
- `git`
- `curl`
- NVIDIA GPU with appropriate drivers.

**Steps**:
1.  **Clone the Repository**:
    ```bash
    git clone git@github.com:PrimeIntellect-ai/prime-rl.git
    cd prime-rl
    ```

2.  **Run the Installation Script**: This is the recommended one-step process to install dependencies.
    ```bash
    curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
    ```

3.  **Activate Virtual Environment**: The script will create a `.venv` folder. Activate it using:
    ```bash
    source .venv/bin/activate
    ```

4.  **Install Pre-commit Hooks**: This enables automatic code quality checks before each commit.
    ```bash
    uv run pre-commit install
    ```

**Validation**: Run a debug-mode SFT training job.
```bash
uv run sft @ configs/debug/sft.toml
```
A successful run indicates your environment is correctly configured.

### 3.2 Development Workflows

#### Workflow: Running a Local RL Experiment

**Purpose**: To run a full, local RL experiment using the three core components. This is the most common workflow for testing changes.

**Prerequisites**:
- Completed initial project setup.
- At least 2 available GPUs.

**Steps**:
1.  **Use the `rl` Entrypoint**: This script handles the launching and coordination of all components. The following command runs a small, fast experiment defined in the `reverse_text` config files.
    ```bash
    uv run rl \
      --trainer @ configs/reverse_text/train.toml \
      --orchestrator @ configs/reverse_text/orch.toml \
      --inference @ configs/reverse_text/infer.toml
    ```

**Expected Outputs**:
- Logs from the trainer will be streamed to your console.
- Detailed logs for each component will be created in `outputs/`:
  - `outputs/trainer.log`
  - `outputs/orchestrator.log`
  - `outputs/inference.log`

#### Workflow: Debugging with a Persistent Inference Server

**Purpose**: To avoid the long startup time of the `vLLM` inference server during iterative development. You can start the server once and run the trainer/orchestrator multiple times.

**Prerequisites**:
- Completed initial project setup.
- A terminal multiplexer like `tmux` is highly recommended.

**Steps**:
1.  **Start a `tmux` Session**: The provided script creates a convenient layout with panes for each component.
    ```bash
    bash scripts/tmux.sh
    ```
    This creates a session named `prime-rl` with panes for Trainer, Orchestrator, and Inference.

2.  **Start the Inference Server**: In the "Inference" `tmux` pane, start the server.
    ```bash
    uv run inference @ configs/reverse_text/infer.toml
    ```
    The server will start and wait for requests.

3.  **Start the Training Run**: In the "Trainer" `tmux` pane, run the `rl` command but *omit* the `--inference` argument. This tells the script not to start a new inference server. The orchestrator will automatically connect to the one you started manually (as long as the ports in the config files match).
    ```bash
    uv run rl \
      --trainer @ configs/reverse_text/train.toml \
      --orchestrator @ configs/reverse_text/orch.toml
    ```

4.  **Iterate**: You can now stop and restart the command in the "Trainer" pane to test your changes without restarting the inference server.

**Validation**: The orchestrator will log that it has successfully connected to the inference server, and the training will proceed as normal.

### 3.3 Testing Workflows

#### Workflow: Running the Full Test Suite

**Purpose**: To run all unit and integration tests and ensure the entire codebase is functioning correctly.

**Prerequisites**:
- Completed initial project setup.

**Steps**:
1.  **Execute Pytest**: Use the `uv run` command to execute `pytest`.
    ```bash
    uv run pytest -v
    ```

**Expected Outputs**: `pytest` will discover and run all tests in the `tests/` directory. All tests should pass.

#### Workflow: Running Specific Test Subsets

**Purpose**: To run a smaller, faster subset of tests relevant to your current work.

**Prerequisites**:
- Completed initial project setup.

**Customization**: `prime-rl` uses `pytest` markers (`gpu`, `slow`) to categorize tests.

**Steps**:
- **Run Unit Tests**:
  ```bash
  uv run pytest tests/unit -v
  ```
- **Run Integration Tests**:
  ```bash
  uv run pytest tests/integration -v
  ```
- **Run CPU-only tests** (exclude tests marked as `gpu`):
  ```bash
  uv run pytest -v -m "not gpu"
  ```
- **Run fast tests** (exclude tests marked as `slow`):
  ```bash
  uv run pytest -v -m "not slow"
  ```

### 3.4 Code Quality Workflows

#### Workflow: Linting and Formatting

**Purpose**: To ensure code adheres to the project's style and quality standards.

**Prerequisites**:
- Pre-commit hooks installed (`uv run pre-commit install`).

**Steps**:
1.  **Automatic Checks**: When you run `git commit`, the pre-commit hooks will automatically run `ruff` to lint and format your staged files. If any issues are found and fixed, the commit will be aborted, and you will need to `git add` the changes and commit again.

2.  **Manual Checks**: You can run the checks manually on all files at any time.
    - **To lint and fix errors**:
      ```bash
      uv run ruff --fix .
      ```
    - **To format code**:
      ```bash
      uv run ruff format .
      ```

**Configuration**: The rules for `ruff` are defined in `pyproject.toml` under the `[tool.ruff]` section. The pre-commit hooks are defined in `.pre-commit-config.yaml`.

---
*End of Module 3*

---

## Module 4: Hands-On Development

This module provides practical, hands-on exercises to solidify your understanding of the `prime-rl` codebase. Each exercise builds on the concepts from the previous modules.

### 4.1 Code Reading Exercises

#### Exercise: Tracing a Rollout in the Orchestrator

**Goal**: Understand how the `Orchestrator` generates data by tracing the main loop.

**File**: `src/prime_rl/orchestrator/orchestrator.py`

**Instructions**:
1.  Open the file and locate the `orchestrate` async function. This is the main entry point for the orchestrator.
2.  Navigate to the main `while True:` loop. This loop represents one training step.
3.  **Follow the logic inside the loop**:
    *   **Checkpointing**: Notice the logic at the top of the loop that saves a checkpoint (`ckpt_manager.save(...)`).
    *   **Weight Updates**: Find the block that checks `if progress.step - ckpt_step > config.async_level:`. This is the "async barrier". Inside this block, the orchestrator waits for a new weight checkpoint from the trainer (`wait_for_weight_checkpoint`) and then tells the inference server to load it (`await update_weights(...)`).
    *   **Data Sampling**: Look for the inner `while True:` loop. This loop continues until a full batch of rollouts is collected. It starts by sampling problems from the buffer: `buffer.sample_problems(...)`.
    *   **Inference Request**: The code then prepares an inference request. The line `generate_outputs: GenerateOutputs = await vf_env.a_generate(...)` is where the orchestrator makes an HTTP call to the inference server to get completions.
    *   **Processing Results**: After getting the response, the code processes the results to get rewards and advantages (`vf_env.process_env_results_vllm(...)`, `compute_advantages(...)`).
    *   **Saving the Batch**: Finally, locate the code that saves the prepared batch to disk for the trainer. It starts around `all_data_ranks_batches = prepare_batch(...)` and ends with `tmp_path.rename(batch_path)`.
    *   **Logging**: At the end of the loop, observe the various `monitor.log(...)` calls. This is how all the metrics are recorded.

### 4.2 Feature Modification Exercises

#### Exercise: Add a New Logged Metric

**Goal**: Add a new metric to the orchestrator's logs to track the average number of rollouts that were truncated in a batch.

**File**: `src/prime_rl/orchestrator/orchestrator.py`

**Instructions**:
1.  **Locate the data**: In the `orchestrate` function's main loop, after the rollouts are generated, the `is_truncated` tensor is created (around line 272). This tensor contains a boolean flag for each rollout indicating if it was truncated.
    ```python
    // ... existing code ...
    is_truncated = parse_truncated_completions(states=generate_outputs.state)
    // ... existing code ...
    ```
    It's later converted to a tensor:
    ```python
    // ... existing code ...
        is_truncated = (
            torch.tensor([rollout.is_truncated for rollout in accepted_rollouts])
            .reshape(-1, config.rollouts_per_example)
            .float()
        )
    // ... existing code ...
    ```

2.  **Calculate the metric**: The `is_truncated` tensor has a shape of `(problems_per_batch, rollouts_per_example)`. We want the mean value. You can calculate it like this: `is_truncated.mean().item()`.

3.  **Log the metric**: Find the section where metrics are logged to the `monitor` (around line 413). Add a new block to log your truncation metric.
    ```python
    // ... existing code ...
        truncated_metrics = {
            "is_truncated/mean": is_truncated.mean(-1).mean().item(),
            "is_truncated/max": is_truncated.mean(-1).max().item(),
            "is_truncated/min": is_truncated.mean(-1).min().item(),
            "step": progress.step,
        }
        monitor.log(truncated_metrics)

        # ADD YOUR NEW METRIC HERE
        # For example:
        #
        # new_metrics = {
        #     "my_metric/mean_truncation": is_truncated.mean().item(),
        #     "step": progress.step
        # }
        # monitor.log(new_metrics)
    // ... existing code ...
    ```
    The existing `is_truncated/mean` is already logging this, but this exercise demonstrates the process. You could add a different aggregation, like `sum`.

4.  **Run and Validate**: Execute a run (e.g., the `reverse_text` example). If you have `wandb` configured, you should see your new metric appear in the dashboard. Otherwise, it will be in the monitor history logged to the console/file.

### 4.3 New Feature Development

#### Exercise: Add a New PPO Loss Clipping Type

**Goal**: Extend the PPO loss function in the trainer to support a new type of clipping.

**Files**:
- `src/prime_rl/trainer/rl/config.py`
- `src/prime_rl/trainer/rl/loss.py`

**Instructions**:
1.  **Update the Config**: Open `src/prime_rl/trainer/rl/config.py`. Find the `LossConfig` class. The `clip_type` field is a `Literal` that only allows `"ratio"` or `"value"`. Add a new option, for example, `"ratio_squared"`.
    ```python
    // In src/prime_rl/trainer/rl/config.py, class LossConfig
    // ... existing code ...
    clip_type: Annotated[
        Literal["ratio", "value", "ratio_squared"],  # <-- ADD YOUR NEW TYPE HERE
        Field(description="The type of clipping to use for the PPO loss."),
    ] = "ratio"
    // ... existing code ...
    ```

2.  **Implement the Logic**: Open `src/prime_rl/trainer/rl/loss.py`. Find the `compute_loss` function. Inside this function, there's a section that calculates the `clipped_surrogate_loss` based on `loss_config.clip_type`. Add a new `elif` block to handle your `"ratio_squared"` case.
    ```python
    // In src/prime_rl/trainer/rl/loss.py, function compute_loss
    # ... existing code ...
    if loss_config.clip_type == "ratio":
        clipped_ratio = torch.clamp(ratio, 1.0 - loss_config.clip_eps, 1.0 + loss_config.clip_eps)
        clipped_surrogate_loss = clipped_ratio * advantages
    elif loss_config.clip_type == "value":
        clipped_surrogate_loss = torch.where(
            advantages > 0, (1 + loss_config.clip_eps) * advantages, (1 - loss_config.clip_eps) * advantages
        )
    # ADD YOUR LOGIC HERE
    elif loss_config.clip_type == "ratio_squared":
        # Example implementation: square the clipping range
        clip_eps_squared = loss_config.clip_eps ** 2
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps_squared, 1.0 + clip_eps_squared)
        clipped_surrogate_loss = clipped_ratio * advantages
    # ... existing code ...
    ```

3.  **Test your change**: Create a new `.toml` config file in the `configs/` directory. Copy an existing config, and modify it to use your new loss clipping type:
    ```toml
    # in configs/my_test_config.toml
    [loss]
    type = "ppo"
    clip_type = "ratio_squared"
    ```
    Now, run the trainer with this config file and observe the behavior.

### 4.4 Debugging Exercises

#### Scenario: A Tensor Shape Mismatch

**The Problem**: You've made a change in the `Orchestrator`, but now the `Trainer` is crashing with a cryptic CUDA error related to tensor shapes. The error message in `trainer.log` says something like `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension Z`.

**How to Debug**:
1.  **Identify the Location**: The error occurs in the `Trainer`. The stack trace will likely point to the `forward_backward_pass` inside `src/prime_rl/trainer/rl/train.py`, probably during the `compute_loss` call or the `loss.backward()` call.

2.  **Isolate the Data**: The `Trainer` gets its data from the `Orchestrator`. The bug is likely in how the `Orchestrator` is creating the data batches. The data for each step is saved in `outputs/rollouts/step_N/`.

3.  **Inspect the Data**: Write a small, separate Python script to load one of the saved batch files (`.pt` file) from the rollout directory.
    ```python
    import torch
    
    # Load the batch data for a specific rank
    batch_data = torch.load("outputs/rollouts/step_0/rank_0.pt")
    
    # It's a list of micro-batches
    first_micro_batch = batch_data[0]
    
    # Print the shapes of all tensors in the micro-batch
    for key, tensor in first_micro_batch.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{key}: {tensor.shape}")
    ```

4.  **Find the Mismatch**: Run this script and look at the shapes. You will likely find that two tensors that are supposed to have the same shape (e.g., `logprobs` and `advantages`) have a mismatch. This tells you that the error is in the `Orchestrator`'s `prepare_batch` function or the logic that creates these tensors.

5.  **Fix the Root Cause**: Go back to `src/prime_rl/orchestrator/orchestrator.py` and inspect the logic where the mismatched tensors are created. Correct the logic to ensure their shapes are consistent before they are saved.

---
*End of Module 4*

---

## Module 5: Advanced Topics

### 5.1 Performance Optimization

`prime-rl` is built for high-performance training. Here are some of the key performance features and how to use them:

-   **`vLLM` for Inference**: The inference server uses `vLLM`, a state-of-the-art serving engine that provides high throughput via techniques like PagedAttention. The performance of `vLLM` is critical to the overall speed of the RL loop.
-   **`FlashAttention`**: The trainer model uses `flash-attn` where available, which is a highly optimized attention implementation that is much faster and more memory-efficient than the standard PyTorch implementation. This is enabled by default.
-   **FSDP for Training**: The trainer uses PyTorch's Fully Sharded Data Parallelism (FSDP) to distribute the model and optimizer states across multiple GPUs. This allows for the training of very large models that would not fit on a single GPU.

#### Workflow: Benchmarking

**Purpose**: To measure the performance of the trainer and inference components in isolation. This is useful for identifying bottlenecks and tuning configurations for maximum throughput.

**Prerequisites**:
- Completed initial project setup.

**Steps**:
1.  **Benchmark Inference**: This workflow measures the throughput of the inference server.
    -   First, start the inference server normally.
        ```bash
        uv run inference @ configs/reverse_text/infer.toml
        ```
    -   Then, run the orchestrator in benchmark mode. It will send a barrage of requests to the server and report performance statistics.
        ```bash
        uv run orchestrator @ configs/reverse_text/orch.toml --bench
        ```

2.  **Benchmark the Trainer**: This workflow measures the trainer's TFLOPs and throughput using a synthetic, on-the-fly dataset, removing any data-loading bottlenecks.
    ```bash
    uv run trainer @ configs/reverse_text/train.toml --bench
    ```

3.  **Benchmark Everything**: You can also use the main `rl` entrypoint with the `--bench` flag, which will run both of the above benchmarks.
    ```bash
    uv run rl \
      --trainer @ configs/reverse_text/train.toml \
      --orchestrator @ configs/reverse_text/orch.toml \
      --inference @ configs/reverse_text/infer.toml \
      --bench
    ```

### 5.2 Security Considerations

In its default configuration, `prime-rl` is designed for use in a trusted research environment. The inference server's API endpoint is open and unauthenticated.

**Production Environments**: If you were to adapt this for a production system, you would need to add security layers.
-   **Network Security**: The inference server port should be firewalled and only accessible from the machine(s) running the orchestrator.
-   **Authentication**: An authentication layer (e.g., API keys, mTLS) should be added to the inference server's HTTP endpoints.

### 5.3 Monitoring & Maintenance

#### W&B Integration

**Purpose**: To log detailed metrics, sample rollouts, and system performance to Weights & Biases for experiment tracking and visualization.

**Prerequisites**:
- A W&B account (`wandb.ai`).
- Logged in via the CLI (`uv run wandb login`) or `WANDB_API_KEY` environment variable set.

**Steps**:
1.  **Enable W&B in your run**: The easiest way is to use the shared `--wandb` config flag on the `rl` entrypoint. This will automatically configure both the trainer and orchestrator to log to the same project, with linked names.
    ```bash
    uv run rl \
      --trainer @ configs/reverse_text/train.toml \
      --orchestrator @ configs/reverse_text/orch.toml \
      --inference @ configs/reverse_text/infer.toml \
      --wandb.project "my-prime-rl-experiments" \
      --wandb.name "experiment-v1"
    ```

2.  **View the Dashboard**: Go to your W&B project page. You will see two runs, e.g., `experiment-v1-trainer` and `experiment-v1-orchestrator`. You can view detailed graphs of loss, reward, throughput, and more. You can also see tables of sample rollouts logged by the orchestrator.

#### Log Files

All components produce detailed log files in the `outputs/` directory by default. The `tmux.sh` script is the recommended way to view all of these logs live in a tiled layout.

### 5.4 Multi-Node Training

`prime-rl` supports scaling both the trainer and the inference server across multiple nodes to handle very large models and datasets.

#### Multi-Node Trainer

The trainer uses `torchrun` for multi-node distributed training.
-   **Concept**: You designate one node as the "master" and provide its IP address and a port to all nodes. `torchrun` then establishes the distributed communication group.
-   **Workflow**: As detailed in the main `README.md`, you need to set the `RDZV_ENDPOINT` environment variable on all nodes and then launch the `torchrun` command with the appropriate `--nnodes` and `--node-rank` on each machine.

#### Multi-Node Inference

The inference server uses `vLLM`'s built-in support for data parallelism.
-   **Concept**: This allows you to run multiple inference engines on different nodes that all serve the same model, with a head node distributing requests among them.
-   **Workflow**: The `README.md` provides a detailed example. You must configure network interfaces and set environment variables (`GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `DATA_PARALLEL_ADDRESS`) to allow the nodes to communicate.

### 5.5 Custom Environments

The RL tasks themselves are defined in separate, installable packages called "Environments", which are built on the `verifiers` framework.
-   **Purpose**: This decouples the core RL training logic from the specifics of any one task (e.g., math problems, code generation).
-   **Workflow**:
    1.  **Create a New Environment**: Follow the instructions in the [`prime-environments`](https://github.com/PrimeIntellect-ai/prime-environments) repository to create a new environment. This typically involves defining how to generate prompts and how to calculate a reward from a completion.
    2.  **Install the Environment**: You can install your custom environment into the `prime-rl` virtual environment using `uv run prime env install ...`.
    3.  **Use in a Run**: Create new `.toml` config files for your experiment and set the `[environment]` `id` to the name of your new environment.

---
*End of Tutorial*
