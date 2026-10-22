"""Training size follows the machine: a smaller network on CPU, a wider one on GPU."""

from __future__ import annotations

import argparse

import torch


# CPU.
CPU_PROFILE = {
    "d_model": 192,
    "num_heads": 6,
    "num_layers": 3,
    "d_ff": 768,
    "dropout": 0.1,
    "max_len": 160,
    "vocab_size": 5000,
    "batch_size": 64,
    "epochs": 12,
    "lr_factor": 1.0,
    "amp": False,
    "num_workers": 0,
    "label_smoothing": 0.0,
    "warmup_steps": 400,
    "weight_decay": 0.0,
    "tone_augment": True,
    "use_domain_tags": True,
    "patience": 6,
    "beam_size": 4,
}

# GPU.
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
    "lr_factor": 0.5,
    "amp": True,
    "num_workers": 2,
    "label_smoothing": 0.1,
    "warmup_steps": 2000,
    "weight_decay": 0.01,
    "tone_augment": True,
    "use_domain_tags": True,
    "patience": 8,
    "beam_size": 4,
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
    p = argparse.ArgumentParser(
        description="Train the French ↔ Ngiemboon Transformer. English is translated later, through French."
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--rebuild-corpus", action="store_true", help="Re-extract PDFs and retrain BPE.")
    p.add_argument("--no-resume", action="store_true", help="Ignore an existing checkpoint and train from scratch.")
    return p.parse_args(argv)
