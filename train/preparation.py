"""Shared prompt encoding, filtering, data identity, and validation-subset selection."""
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

import torch
from torch.utils.data import DataLoader

from model.prompt import build_messages

IGNORE_INDEX = -100
DATA_SEED = 42
UNSEEN_FRACTION = 0.2


def response_text(tokenizer, query, tools, answers):
    """Return (prompt, target) strings.

    The prompt is rendered exactly as at inference (generation prompt added).
    The target is whatever the chat template places after it for an assistant
    turn, so training matches the model's native format, and it always ends
    with the EOS token that stops generation.
    """
    messages = build_messages(query, tools)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    answer = json.dumps(answers, ensure_ascii=False)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}], tokenize=False
    )
    if full.startswith(prompt):
        target = full[len(prompt):].rstrip()
    else:
        target = answer
    eos = tokenizer.eos_token
    if eos and not target.endswith(eos):
        target += eos
    return prompt, target


def encode_example(tokenizer, example):
    prompt, target = response_text(
        tokenizer, example["query"], example["tools"], example["answers"]
    )
    # Same call as FunctionCallingModel.generate_batch: the template already
    # contains any BOS token, so do not add special tokens again.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def update_fingerprint(digest, example):
    content = json.dumps(example, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    digest.update(content.encode("utf-8") + b"\n")


def prepare_split(tokenizer, examples, max_length):
    """Encode one split and record exactly which examples survive filtering."""
    features, retained_ids, dropped_ids = [], [], []
    digest = hashlib.sha256()
    sequence_tokens = response_tokens = 0
    for example in examples:
        feature = encode_example(tokenizer, example)
        if len(feature["input_ids"]) > max_length:
            dropped_ids.append(example["id"])
            continue
        supervised = sum(label != IGNORE_INDEX for label in feature["labels"][1:])
        if not supervised:
            raise ValueError(f"Example {example['id']!r} has no supervised response tokens")
        features.append(feature)
        retained_ids.append(example["id"])
        update_fingerprint(digest, example)
        sequence_tokens += len(feature["input_ids"])
        response_tokens += supervised
    return features, {
        "selected_count": len(examples),
        "retained_count": len(features),
        "dropped_count": len(dropped_ids),
        "retained_ids": retained_ids,
        "dropped_ids": dropped_ids,
        "retained_sha256": digest.hexdigest(),
        "sequence_tokens": sequence_tokens,
        "response_tokens": response_tokens,
    }


def encode_split(tokenizer, examples, max_length):
    """Keep the original (features, dropped_count) interface for other callers."""
    features, details = prepare_split(tokenizer, examples, max_length)
    return features, details["dropped_count"]


def describe_splits(splits):
    """Fingerprint split contents and order, preserving original dataset IDs."""
    result = {}
    for name, examples in splits.items():
        digest = hashlib.sha256()
        ids = []
        for example in examples:
            ids.append(example["id"])
            update_fingerprint(digest, example)
        result[name] = {"count": len(ids), "ids": ids, "sha256": digest.hexdigest()}
    return result


def make_collator(pad_id):
    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [IGNORE_INDEX] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }
    return collate


def select_generation_subset(examples, limit=256, seed=42):
    """Sample without replacement using a private RNG; keep source order for batching.

    Selection is independent of the training seed and is reused at every epoch.
    The caller supplies only retained validation examples, never training/test rows.
    """
    if limit is None or len(examples) <= limit:
        return list(examples)
    indices = sorted(random.Random(seed).sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


@dataclass
class PreparedData:
    tokenizer: Any
    train_examples: list
    val_examples: list
    train_features: list
    val_features: list
    generation_examples: list
    manifest: dict


def prepare_data(args, *, splits=None, revision=None, code_hashes=None):
    """Prepare features and manifests before loading model weights."""
    from transformers import AutoTokenizer

    supplied_splits = splits is not None
    dataset_name = "provided_splits"
    if splits is None:
        from data.data import FunctionCallingDataset
        dataset_name = FunctionCallingDataset.DATASET_NAME
        splits = FunctionCallingDataset.load_splits(seed=DATA_SEED, unseen_fraction=UNSEEN_FRACTION)
    train_examples = list(splits["train"])[:args.limit_train]
    val_examples = list(splits["validation"])[:args.limit_val]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features, train_data = prepare_split(tokenizer, train_examples, args.max_length)
    val_features, validation_data = prepare_split(tokenizer, val_examples, args.max_length)
    retained_validation_ids = set(validation_data["retained_ids"])
    generation_pool = [row for row in val_examples if row["id"] in retained_validation_ids]
    generation_examples = select_generation_subset(generation_pool, args.generation_limit, args.generation_subset_seed)
    generation_subset = {
        **describe_splits({"generation": generation_examples})["generation"],
        "eligible_count": len(generation_pool), "requested_count": args.generation_limit,
        "seed": args.generation_subset_seed,
        "selection": "seeded sample without replacement; source order retained",
    }
    if not train_features:
        raise SystemExit("No training examples fit within --max-length.")
    print(f"Train examples: {len(train_features)} (dropped {train_data['dropped_count']} over {args.max_length} tokens)")
    print(f"Validation examples: {len(val_features)} (dropped {validation_data['dropped_count']})")
    print(f"Generation validation subset: {len(generation_examples)} / {len(generation_pool)} retained rows (seed {args.generation_subset_seed})")
    data_settings = {
        "dataset": dataset_name,
        "source_split": None if supplied_splits else "train",
        "split_seed": None if supplied_splits else DATA_SEED,
        "unseen_fraction": None if supplied_splits else UNSEEN_FRACTION,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "max_length": args.max_length,
        "length_rule": "drop if prompt plus response (including ending token) exceeds max_length",
    }
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    data_manifest = {
        "settings": data_settings,
        "raw_splits": describe_splits(splits),
        "used_splits": {"train": train_data, "validation": validation_data},
        "generation_validation": generation_subset,
        "tokenizer": args.model,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tokenizer.chat_template, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "code_sha256": code_hashes or {},
    }
    return PreparedData(tokenizer, train_examples, val_examples, train_features, val_features,
                        generation_examples, data_manifest)


def make_loaders(prepared, args):
    collate = make_collator(prepared.tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(prepared.train_features, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator)
    val_loader = DataLoader(prepared.val_features, batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collate)
    return train_loader, val_loader
