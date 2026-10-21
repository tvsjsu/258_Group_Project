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

Run all tests:

```bash
uv run pytest -v
```

Run only the data tests:

```bash
uv run pytest -v -k data_test
```

Run only the model tests:

```bash
uv run pytest -v -k model_test
```
Run only the eval tests (no model or dataset download):

```bash
uv run pytest -v -k eval_test
```

## Run baseline evaluation

Evaluate the base model (no fine-tuned adapter) on the held-out `test_seen` split. This requires the [Hugging Face login](#hugging-face-login) above. The first run downloads the dataset and the ~1.3B model (~3 GB). A GPU is used automatically if available; on Apple Silicon add `--device mps`, since CPU-only runs are slow.

```bash
uv run python -m eval.baseline --limit 100
```

`--limit` sets how many examples to evaluate (default 100). Use `--split test_unseen` to evaluate on unseen APIs instead.

The script prints three metrics:

- **Valid output rate**: the output parses as a JSON array of `{"name", "arguments"}` calls.
- **Function-name accuracy**: the predicted function names match the expected ones.
- **Exact-match accuracy**: both the function names and the arguments match exactly.

Call order is ignored. Predictions (expected vs. predicted, plus the raw output) are saved to `results/baseline_predictions.jsonl`, which is ignored by Git.
