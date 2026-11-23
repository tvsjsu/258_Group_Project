"""Full-model or adapter checkpoints, best-model snapshots, and deterministic recovery."""

import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import torch


def rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class AdapterCheckpoints:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.resume = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        self.state = None
        self.best = None
        if self.resume:
            self.state = torch.load(self.resume / "training_state.pt", map_location="cpu", weights_only=True)
            self.best = self.state.get("best")

    def prepare(self, manifest, args, tokenizer):
        self.tokenizer = tokenizer
        ignored = {"output_dir", "resume_from_checkpoint", "save_steps", "log_every", "reference_data"}
        self.contract = {
            "settings": {key: value for key, value in vars(args).items() if key not in ignored},
            "raw_splits": manifest["raw_splits"],
            "used_splits": manifest["used_splits"],
            "chat_template_sha256": manifest["chat_template_sha256"],
            "tokenizer_revision": manifest["tokenizer_revision"],
            "code_sha256": manifest["code_sha256"],
        }
        if self.state and self.contract != self.state["contract"]:
            raise ValueError("Resume configuration, data, tokenizer, or training code differs from the checkpoint. Keep the original training budget/settings.")
        if args.reference_data:
            reference = json.loads(Path(args.reference_data).read_text(encoding="utf-8"))
            for key in ("raw_splits", "used_splits", "chat_template_sha256", "tokenizer_revision"):
                if manifest[key] != reference[key]:
                    raise ValueError(f"Training data differs from reference data_splits.json: {key}")
            if manifest.get("tokenizer") != reference.get("tokenizer"):
                raise ValueError("Tokenizer differs from reference data_splits.json")

    def bind_model(self, model):
        self.base_revision = getattr(model.config, "_commit_hash", None)
        if self.state and self.base_revision != self.state["base_revision"]:
            raise ValueError("The resolved base model revision differs from the checkpoint")

    def record_environment(self, environment):
        session_id = time.time_ns()
        path = self.output_dir / f"run_session-{session_id}.json"
        path.write_text(json.dumps({
            "session_id": session_id, "environment": environment,
            "start_step": self.state["progress"]["step"] if self.state else 0,
            "resume_from": str(self.resume) if self.resume else None,
        }, indent=2), encoding="utf-8")

    def sessions(self):
        records = list(self.state.get("sessions", [])) if self.state else []
        records.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.output_dir.glob("run_session-*.json")))
        return list({record["session_id"]: record for record in records}.values())

    def restore(self, optimizer, scheduler):
        if self.state:
            predictions = self.resume / "validation_predictions"
            if predictions.is_dir():
                self._replace_directory(predictions, self.output_dir / "validation_predictions")
            if self.best:
                snapshot = self.resume / "best_model"
                if not snapshot.is_dir():
                    raise ValueError("Checkpoint is missing its best-model snapshot")
                self._replace_directory(snapshot, self.output_dir / "best")
                self._write_best()
            optimizer.load_state_dict(self.state["optimizer"])
            scheduler.load_state_dict(self.state["scheduler"])
            progress = dict(self.state["progress"])
            info = json.loads((self.resume / "checkpoint_info.json").read_text(encoding="utf-8"))
            progress["checkpoint_seconds"] += info["checkpoint_save_seconds"]
            return progress
        return {}

    def restore_random_state(self):
        if self.state:
            restore_rng(self.state["rng"])

    def should_save(self, step, epoch_end, final):
        return step % self.args.save_steps == 0 or epoch_end or final

    @staticmethod
    def _link_or_copy(source, destination):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def _replace_directory(self, source, destination):
        temporary = self.output_dir / ".best-recovery"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(source, temporary, copy_function=self._link_or_copy)
        shutil.rmtree(destination, ignore_errors=True)
        temporary.rename(destination)

    def _write_best(self):
        path = self.output_dir / "best_checkpoint.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({**self.best, "model_path": "best",
                                        "selection_rule": "validation complete-call accuracy; tie: lower validation loss; tie: earlier step"},
                                       indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def prune(self):
        checkpoints = sorted((path for path in self.output_dir.glob("checkpoint-*")
                              if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
                             key=lambda path: int(path.name.removeprefix("checkpoint-")))
        keep = {path.name for path in checkpoints[-2:]}
        if self.best:
            keep.add(f"checkpoint-{self.best['step']}")
        for path in checkpoints:
            if path.name not in keep:
                shutil.rmtree(path)

    def save(self, model, optimizer, scheduler, progress, best_candidate=None):
        started = time.perf_counter()
        destination = self.output_dir / f"checkpoint-{progress['step']}"
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}; resume into a new output directory to branch a run")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.output_dir))
        saved_rng = rng_state()
        try:
            model.save_pretrained(temporary, safe_serialization=True)
            self.tokenizer.save_pretrained(temporary)
            next_best = best_candidate or self.best
            if next_best:
                snapshot = temporary / "best_model"
                if best_candidate:
                    snapshot.mkdir()
                    for path in temporary.iterdir():
                        if path.is_file():
                            self._link_or_copy(path, snapshot / path.name)
                else:
                    shutil.copytree(self.output_dir / "best", snapshot, copy_function=self._link_or_copy)
            predictions = self.output_dir / "validation_predictions"
            if predictions.is_dir():
                shutil.copytree(predictions, temporary / "validation_predictions", copy_function=self._link_or_copy)
            torch.save({
                "contract": self.contract,
                "base_revision": self.base_revision,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "progress": progress,
                "rng": saved_rng,
                "sessions": self.sessions(),
                "best": next_best,
            }, temporary / "training_state.pt")
            elapsed = time.perf_counter() - started
            (temporary / "checkpoint_info.json").write_text(json.dumps({
                "step": progress["step"], "epochs_completed": progress["history"][-1]["epochs_completed"],
                "checkpoint_save_seconds": elapsed,
            }, indent=2), encoding="utf-8")
            temporary.rename(destination)
            if best_candidate:
                self.best = best_candidate
                self._replace_directory(destination / "best_model", self.output_dir / "best")
                self._write_best()
            latest = self.output_dir / "latest_checkpoint.json"
            latest_temporary = latest.with_suffix(".tmp")
            latest_temporary.write_text(json.dumps({
                "path": destination.name, "step": progress["step"],
            }, indent=2), encoding="utf-8")
            latest_temporary.replace(latest)
            self.prune()
            # Recovery accounting includes export/retention work, not just serialization.
            info_path = destination / "checkpoint_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["checkpoint_save_seconds"] = time.perf_counter() - started
            info_temporary = info_path.with_suffix(".tmp")
            info_temporary.write_text(json.dumps(info, indent=2), encoding="utf-8")
            info_temporary.replace(info_path)
            return time.perf_counter() - started
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            # Serialization must not advance the training RNG stream.
            restore_rng(saved_rng)

    def storage_bytes(self):
        # Logical file sizes; filesystem hard links can reduce actual occupied space.
        return sum(path.stat().st_size for directory in self.output_dir.glob("checkpoint-*")
                   for path in directory.rglob("*") if path.is_file())
