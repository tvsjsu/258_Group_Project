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

After setup and Hugging Face login, run all tests:

```bash
uv run pytest -v
```

## Prepare the baseline data splits

Use these shared settings for all training methods: `seed=42`, `unseen_fraction=0.2`.

```python
from data.data import FunctionCallingDataset

splits = FunctionCallingDataset.load_splits(seed=42, unseen_fraction=0.2)
for name, dataset in splits.items():
    print(f"{name}: {len(dataset)}")
```

Expected test sizes: `test_seen: 3333`, `test_unseen: 3937`. Evaluation recreate the splits automatically.

## Run evaluation

Check the evaluation data without loading the model:

```bash
uv run python -m eval.eval --split both --check-data-only
```

Run the full baseline evaluation on a GPU:

```bash
uv run python -m eval.eval \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --split both \
  --batch-size 8 \
  --dtype bfloat16 \
  --device cuda \
  --max-new-tokens 512 \
  --output results/baseline_full_one_shot_rerun.jsonl
```

Outputs: `results/baseline_full_one_shot_rerun.jsonl` and `results/baseline_full_one_shot_rerun.metrics.json`.

Run the same evaluation on a fine-tuned full-model checkpoint:

```bash
uv run python -m eval.eval \
  --model checkpoints/full \
  --split both \
  --batch-size 8 \
  --dtype bfloat16 \
  --device cuda \
  --max-new-tokens 512 \
  --output results/full_finetuned_predictions.jsonl
```

`--model` accepts a Hugging Face model ID or local full-model checkpoint directory, including its tokenizer files. It defaults to the original DeepSeek model when omitted. Metrics are saved alongside predictions with the suffix `.metrics.json`.

### Evaluation metrics

The script reports these five metrics for each test split and the combined results:

| Metric | What it measures |
| --- | --- |
| **Valid-call rate** | Fraction of outputs that parse as strict JSON arrays with the required `name` and `arguments` call structure. |
| **Function accuracy** | Fraction of examples where the predicted function names and call counts match the reference. |
| **Argument accuracy** | Fraction of argument slots with correct values after matching calls by function name; missing and extra arguments count against the score. |
| **Schema compliance** | Fraction of examples where every generated call names an available function and satisfies its parameter schema. |
| **Complete-call accuracy** | Fraction of examples where all function names and arguments match the reference, including call counts. |

Call order is ignored when scoring. Argument accuracy is aggregated over argument slots; the other four metrics are aggregated over examples.
