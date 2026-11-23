"""Measure training resources and save reproducible metrics and histories."""

import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

try:
    import resource
except ImportError:
    resource = None


def write_json(path, value):
    """Save a UTF-8 JSON artifact while rejecting non-finite numeric values."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_history(output_dir, records):
    """Rewrite from recovered state so resumed curves have no duplicate future rows."""
    with (output_dir / "validation_history.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    if records:
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output_dir / "validation_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def environment_info(device):
    """Record hardware and software so measurements can be interpreted later."""
    packages = {}
    for name in ("torch", "transformers", "datasets", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": packages,
        "device": str(device),
        "device_type": device.type,
        "cpu_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
    }
    try:
        info["system_physical_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        info["system_physical_memory_bytes"] = None
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if chip.returncode == 0:
                info["processor"] = chip.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu_name"] = properties.name
        info["gpu_total_memory_bytes"] = properties.total_memory
    return info


def checkpoint_sizes(output_dir, tokenizer_files):
    """Count model/tokenizer artifacts without counting logs or split records."""
    output_dir = Path(output_dir)
    weights, indexes = [], []
    for filename in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"):
        path = output_dir / filename
        index = output_dir / (filename + ".index.json")
        if path.is_file():
            weights = [path]
            break
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            weights = [output_dir / name for name in sorted(set(mapping.values()))]
            indexes = [index]
            break
    files = set(weights + indexes + [Path(name) for name in tokenizer_files])
    files.update(output_dir / name for name in ("config.json", "generation_config.json", "adapter_config.json"))
    return {
        "model_weights_bytes": sum(path.stat().st_size for path in weights),
        "checkpoint_bytes": sum(path.stat().st_size for path in files if path.is_file()),
    }


def optimizer_report(optimizer, name):
    """Describe optimizer settings, actual state dtypes, and logical tensor bytes."""
    state_bytes = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                dtype = str(value.dtype)
                state_bytes[dtype] = state_bytes.get(dtype, 0) + value.numel() * value.element_size()
    report = {
        "name": name,
        "implementation": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        **{key: list(value) if isinstance(value, tuple) else value
           for key, value in optimizer.defaults.items() if key in ("lr", "betas", "eps", "weight_decay")},
        "state_tensor_bytes_by_dtype": state_bytes,
        "state_tensor_bytes": sum(state_bytes.values()),
        "state_tensor_bytes_scope": "logical tensor bytes; shared tensors may be counted more than once; excludes allocator overhead",
    }
    if name == "adamw8bit":
        report.update({
            "configured_state_bits": 8, "small_tensor_state_bits": 32,
            "min_8bit_size": optimizer.args.min_8bit_size,
            "block_wise": optimizer.args.block_wise,
            "percentile_clipping": optimizer.args.percentile_clipping,
            "is_paged": optimizer.is_paged,
        })
    return report


class MemoryTracker:
    """Track training intervals; keep platform-specific measurement labels."""

    def __init__(self, device):
        """Initialize resource peaks and begin the first training measurement interval."""
        self.device = device
        self.peak = 0
        self.reserved_peak = 0
        self.restart_interval()

    def restart_interval(self):
        """Exclude a validation interval without losing earlier training peaks."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()

    def sample(self):
        """Update memory peaks using the counters available on the active platform."""
        if self.device.type == "cuda":
            self.peak = max(self.peak, torch.cuda.max_memory_allocated(self.device))
            self.reserved_peak = max(self.reserved_peak, torch.cuda.max_memory_reserved(self.device))
        elif self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
        elif resource is not None:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak = max(self.peak, rss if sys.platform == "darwin" else rss * 1024)

    def peak_bytes(self):
        """Return the tracked memory peak or None when unavailable."""
        return None if self.device.type == "cpu" and resource is None else self.peak

    def reserved_bytes(self):
        """Return the tracked CUDA allocator reservation peak when applicable."""
        return self.reserved_peak if self.device.type == "cuda" else None

    def label(self):
        """Describe the platform-specific meaning of the recorded memory peak."""
        return {
            "cuda": "Peak GPU memory allocated during training",
            "mps": "Sampled peak MPS driver memory during training",
            "cpu": "Peak process RSS (process lifetime)",
        }[self.device.type]


def source_hashes(method_source=None):
    """Fingerprint shared source files and the selected method entry point."""
    root = Path(__file__).resolve().parents[1]
    names = ["data/data.py", "model/prompt.py", "model/model.py", "eval/metrics.py", "eval/eval.py",
             "train/trainer.py", "train/config.py", "train/preparation.py", "train/reporting.py",
             "train/runtime.py", "train/validation.py", "train/checkpointing.py"]
    if method_source is not None:
        names.append(Path(method_source).resolve().relative_to(root).as_posix())
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for name in dict.fromkeys(names)}


def parameter_counts(model):
    """Count logical parameters, including packed quantized base weights."""
    if hasattr(model, "get_nb_trainable_parameters"):
        return model.get_nb_trainable_parameters()
    return (sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            sum(parameter.numel() for parameter in model.parameters()))


def build_report(*, args, method, model, device, precision, weight_dtype, autocast_dtype,
                 environment, prepared, training_metrics, sizes, model_load_seconds,
                 checkpoint_save_seconds, preprocessing_seconds, run_wall_seconds, checkpoints):
    """Assemble training metrics, data identities, settings, and resource measurements."""
    data_manifest = prepared.manifest
    data_settings = data_manifest["settings"]
    code_hashes = data_manifest["code_sha256"]
    tokenizer_revision = data_manifest["tokenizer_revision"]
    generation_subset = data_manifest["generation_validation"]
    train_examples, val_examples = prepared.train_examples, prepared.val_examples
    train_features, val_features = prepared.train_features, prepared.val_features
    train_data, validation_data = data_manifest["used_splits"]["train"], data_manifest["used_splits"]["validation"]
    generation_examples = prepared.generation_examples
    trainable_params, total_params = parameter_counts(model)
    report = {
        "method": method,
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer_revision,
        "device": str(device),
        "precision": precision,
        "weight_dtype": str(weight_dtype),
        "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        "autocast_weight_cache_enabled": False,
        "settings": {key: value for key, value in vars(args).items() if key not in ("model", "device")},
        "environment": environment,
        "code_sha256": code_hashes,
        "data_settings": data_settings,
        "split_counts": {name: details["count"] for name, details in data_manifest["raw_splits"].items()},
        "train_examples_selected": len(train_examples),
        "train_examples": len(train_features),
        "train_examples_dropped": train_data["dropped_count"],
        "validation_examples_selected": len(val_examples),
        "validation_examples": len(val_features),
        "validation_examples_dropped": validation_data["dropped_count"],
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_parameter_percentage": 100 * trainable_params / total_params,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "loss_objective": "mean over supervised response tokens in each accumulation window",
        "training_time_scope": "training loop including logging, excluding validation, preprocessing, model loading, and checkpoint saving",
        "preprocessing_seconds": preprocessing_seconds,
        "model_load_seconds": model_load_seconds,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "run_wall_seconds": run_wall_seconds,
        "data_splits_file": "data_splits.json",
        "training_history_file": "training_history.jsonl",
        "validation_history_file": "validation_history.jsonl",
        "validation_history_csv": "validation_history.csv",
        "validation_protocol": {
            "metric_protocol": "strict-json-v1", "split": "validation",
            "examples": len(generation_examples), "uses_retained_after_length_filter": True,
            "loss_examples": len(val_features), "generation_limit": args.generation_limit,
            "generation_subset_seed": args.generation_subset_seed,
            "generation_subset_sha256": generation_subset["sha256"],
            "generation_subset_ids": generation_subset["ids"],
            "selection_population": "fixed subset of retained validation examples; full retained validation for loss",
            "val_checks_per_epoch": args.val_checks_per_epoch, "eval_every_override": args.eval_every,
            "generation_checks_per_epoch": args.generation_checks_per_epoch,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.generation_max_new_tokens, "do_sample": False, "num_beams": 1,
            "generation_weights": "in-memory training weights; QLoRA retains NF4 base and FP32 adapters",
            "precision": precision, "warmup_rounding": "ceil",
            "selection_includes_initial_baseline": False,
        },
        "periodic_checkpoints_storage_bytes": checkpoints.storage_bytes() if checkpoints is not None else 0,
        "checkpoint_storage_scope": "logical file bytes; hard links may reduce occupied disk space",
        "memory_measurement_note": "Training tensor peaks exclude validation operations; reserved memory includes cached allocations retained from earlier phases. CUDA counters do not measure Windows shared-memory residency.",
        **training_metrics,
        **sizes,
    }
    report.update({
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "training_sessions": checkpoints.sessions() if checkpoints is not None else [],
        "run_wall_time_scope": "current invocation only",
        "resume_accounting": "training totals include recovered checkpoint progress plus this invocation; discarded work since the last checkpoint is excluded",
    })
    return report


def adapter_report(model, args, report):
    """Common LoRA/QLoRA metadata; the method supplies quantization-specific fields."""
    return {
        "lora": {"rank": args.rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": args.target_modules},
        "adapter_parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()
                                            if parameter.requires_grad}),
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "checkpoint_artifact": "adapter and tokenizer; base model required separately",
        "adapter_weights_bytes": report["model_weights_bytes"],
        "base_model_required": args.model,
        "loaded_model_memory_footprint_bytes": model.get_memory_footprint(),
    }


def print_report(report, args):
    """Print the completed run metrics and artifact locations."""
    method = report["method"]
    trainable_params = report["trainable_parameters"]
    output_dir = Path(args.output_dir)
    gib = 1024 ** 3
    print(f"\n=== {method} fine-tuning summary ===")
    print(f"Optimizer steps:      {report['optimizer_steps']}")
    print(f"Epochs completed:     {report['epochs_completed']:.3f}")
    print(f"Training time:        {report['train_seconds']:.1f}s ({report['seconds_per_step']:.2f}s/step)")
    peak = report["peak_memory_bytes"]
    memory_text = f"{peak / gib:.2f} GiB" if peak is not None else "unavailable"
    print(f"{report['peak_memory_kind']}: {memory_text}")
    print(f"Trainable parameters: {trainable_params:,}")
    if report["val_loss_before"] is not None:
        print(f"Validation loss:      {report['val_loss_before']:.4f} -> {report['val_loss_after']:.4f}")
    if report["checkpoint_bytes"] is not None:
        print(f"Checkpoint:           {output_dir} ({report['checkpoint_bytes'] / gib:.2f} GiB)")
    print(f"Metrics saved to {output_dir / 'train_metrics.json'}")
