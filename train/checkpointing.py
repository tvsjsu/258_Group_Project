"""Adapter checkpoints at optimizer boundaries, with deterministic data-position recovery."""

import json
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
        if self.resume:
            self.state = torch.load(self.resume / "training_state.pt", map_location="cpu", weights_only=True)

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

    def save(self, model, optimizer, scheduler, progress):
        started = time.perf_counter()
        destination = self.output_dir / f"checkpoint-{progress['step']}"
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}; resume into a new output directory to branch a run")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.output_dir))
        saved_rng = rng_state()
        try:
            model.save_pretrained(temporary, safe_serialization=True)
            self.tokenizer.save_pretrained(temporary)
            torch.save({
                "contract": self.contract,
                "base_revision": self.base_revision,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "progress": progress,
                "rng": saved_rng,
                "sessions": self.sessions(),
            }, temporary / "training_state.pt")
            elapsed = time.perf_counter() - started
            (temporary / "checkpoint_info.json").write_text(json.dumps({
                "step": progress["step"], "epochs_completed": progress["history"][-1]["epochs_completed"],
                "checkpoint_save_seconds": elapsed,
            }, indent=2), encoding="utf-8")
            temporary.rename(destination)
            (self.output_dir / "latest_checkpoint.json").write_text(json.dumps({
                "path": destination.name, "step": progress["step"],
            }, indent=2), encoding="utf-8")
            return time.perf_counter() - started
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            # Serialization must not advance the training RNG stream.
            restore_rng(saved_rng)

    def storage_bytes(self):
        return sum(path.stat().st_size for path in self.output_dir.glob("checkpoint-*/*") if path.is_file())
