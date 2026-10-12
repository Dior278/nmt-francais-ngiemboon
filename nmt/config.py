"""Training size follows the machine: GPU if present, otherwise a small CPU profile."""

from __future__ import annotations

import argparse

import torch


CPU_PROFILE = {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "d_ff": 256,
    "dropout": 0.15,
    "max_len": 96,
    "vocab_size": 4000,
    "batch_size": 16,
    "epochs": 8,
    "lr": 3e-4,
    "amp": False,
    "num_workers": 0,
    "label_smoothing": 0.0,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "tone_augment": False,
    "use_domain_tags": False,
}

# Friend's GPU: a bit deeper, longer health sentences, label smoothing,
# warmup, and a tone-stripped NG→FR copy so typed-without-tones input
# degrades instead of collapsing.
GPU_PROFILE = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.2,
    "max_len": 160,
    "vocab_size": 8000,
    "batch_size": 48,
    "epochs": 50,
    "lr": 4e-4,
    "amp": True,
    "num_workers": 2,
    "label_smoothing": 0.1,
    "warmup_steps": 400,
    "weight_decay": 0.01,
    "tone_augment": True,
    "use_domain_tags": True,
}


def get_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is visible.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def profile_for(device: torch.device) -> dict:
    return dict(GPU_PROFILE if device.type == "cuda" else CPU_PROFILE)


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the French ↔ Ngiemboon Transformer.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--rebuild-corpus", action="store_true", help="Re-extract PDFs and retrain BPE.")
    p.add_argument("--no-resume", action="store_true", help="Ignore an existing checkpoint.")
    return p.parse_args(argv)
