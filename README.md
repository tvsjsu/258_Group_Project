# Function Calling Fine-tuning

## Setup

Use Python 3.10 or newer. Run all commands from the repository root: the folder containing `pyproject.toml`, `data/`, `model/`, and your test folder.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then install the project dependencies:

```bash
python -m pip install uv
uv sync
```

`uv sync` creates a local `.venv` and installs the dependencies from `pyproject.toml`, including pytest. Commands run with `uv run` use this environment automatically.

## Hugging Face login

The dataset requires accepting its access conditions before downloading it.

1. Sign in to [Hugging Face](https://huggingface.co/).
2. Open the [dataset page](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and accept the access conditions.
3. Create a **Read** token in [Settings → Access Tokens](https://huggingface.co/settings/tokens) using the same account.
4. [Log in](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli#hf-auth-login) from the terminal:

```bash
uv run hf auth login
```

Paste your token when prompted. If asked to add it as a Git credential, choose **n**; the saved Hugging Face login is enough for these downloads.

## Run tests

```cmd
uv run pytest -v
```

Optional CUDA smoke test (Windows CMD):

```cmd
set RUN_QLORA_CUDA_TEST=1
uv run pytest test/qlora_cuda_test.py -v -s -rs
set RUN_QLORA_CUDA_TEST=
```

Optional 8-bit AdamW training/resume tests for all three methods (Windows CMD):

```cmd
set RUN_8BIT_ADAMW_CUDA_TEST=1
uv run pytest test/optimizer_cuda_test.py -v -s -rs
set RUN_8BIT_ADAMW_CUDA_TEST=
```

## Shared data

Training and evaluation recreate the same splits using `seed=42` and `unseen_fraction=0.2`. Expected test sizes: **3,333 seen-tool** and **3,937 unseen-tool** examples.

Check the evaluation data:

```cmd
uv run python -m eval.eval --split both --check-data-only
```

Each training run saves split IDs, filtering details, and generation-subset IDs in `data_splits.json`.

## Train

All methods use the shared `train/trainer.py`.

| Method | Learning rate | Microbatch × accumulation |
| --- | --- | --- |
| Full fine-tuning | `5e-6` | 1 × 16 |
| LoRA | `1e-4` | 1 × 16 |
| QLoRA | `1e-4` | 2 × 8 |

Shared defaults: **four epochs**, effective batch size 16, **bitsandbytes 8-bit AdamW** (`--optim adamw8bit`), cosine scheduling, 3% warmup, a 2,048-token cutoff, and gradient checkpointing. LoRA/QLoRA use rank 16, alpha 32, dropout 0.05, and all linear target layers.

Full fine-tuning:

```cmd
uv run python -m train.finetune --output-dir checkpoints/full_final
```

LoRA:

```cmd
uv run python -m train.LoRA --output-dir checkpoints/lora_final
```

QLoRA:

QLoRA training requires an NVIDIA CUDA GPU and uses NF4 with double quantization.
```cmd
uv run python -m train.QLoRA --output-dir checkpoints/qlora_final
```

### Validation and outputs

- **Validation loss:** the full retained validation split, approximately ten checks per epoch.
- **Generated metrics:** the same seeded 256-example validation subset before training and once per epoch; generation batch 4, greedy decoding, and up to 512 new tokens.
- **Best checkpoint:** highest subset complete-call accuracy, then lower full-validation loss, then the earlier checkpoint. All four epochs finish; there is no early stopping.

Results include `train_metrics.json`, `training_history.jsonl`, `validation_history.jsonl` / `.csv`, and `data_splits.json`. These contain metrics, plot data, timing, memory, parameters, storage, and reproducibility details.

**`best/` contains the selected model or adapter; root weights are the last model.** The latest two resumable checkpoints plus the best are retained.

To resume training, use the checkpoint path recorded in your run’s `latest_checkpoint.json`:

```cmd
uv run python -m train.<METHOD> --output-dir <RUN_DIR> --resume-from-checkpoint <CHECKPOINT_PATH>
```

Replace the placeholders using your training method:

| Method | `<METHOD>` |
| --- | --- |
| Full fine-tuning | `finetune` |
| LoRA | `LoRA` |
| QLoRA | `QLoRA` |

Use your original output folder for `<RUN_DIR>` and the actual checkpoint directory for `<CHECKPOINT_PATH>`.

Include any custom arguments from the original training command. Resume requires the same training settings, total training budget, and source files.

## Evaluate

These commands evaluate the **complete seen-tool and unseen-tool test sets** with the same BF16 inference settings. Predictions are saved as JSONL, with aggregate results in the matching `.metrics.json` file.

Original baseline:

```cmd
uv run python -m eval.eval --model deepseek-ai/deepseek-coder-1.3b-instruct --split both --batch-size 8 --dtype bfloat16 --device cuda --max-new-tokens 512 --output results/baseline_full_one_shot_rerun.jsonl
```

Full fine-tuning:

```cmd
uv run python -m eval.eval --model checkpoints/full_final/best --split both --batch-size 8 --dtype bfloat16 --device cuda --max-new-tokens 512 --output results/full_finetuned_predictions.jsonl
```

LoRA:

```cmd
uv run python -m eval.eval --adapter checkpoints/lora_final/best --split both --batch-size 8 --dtype bfloat16 --device cuda --max-new-tokens 512 --output results/lora_predictions.jsonl
```

QLoRA:

```cmd
uv run python -m eval.eval --adapter checkpoints/qlora_final/best --split both --batch-size 8 --dtype bfloat16 --device cuda --max-new-tokens 512 --output results/qlora_predictions.jsonl
```

Final evaluation uses an unquantized BF16 base and BF16 adapters for all methods. QLoRA's training-time validation uses its NF4 training model.

| Metric | Meaning |
| --- | --- |
| Valid-call rate | Output parses as a strict JSON array with the required call structure. |
| Function accuracy | Function names and call counts match the reference. |
| Argument accuracy | Correct argument slots after call matching; missing/extra arguments are penalized. |
| Schema compliance | Calls name available functions and satisfy their parameter schemas. |
| Complete-call accuracy | Every function and argument matches, including call counts. |

Call order is ignored. Argument accuracy is aggregated over argument slots; the other metrics are aggregated over examples. Test scores are not used for checkpoint selection.
