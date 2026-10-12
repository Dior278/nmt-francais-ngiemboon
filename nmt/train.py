"""Train on GPU when one is visible. Safe to resume. Do not run this on a CPU-only laptop if you can avoid it."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .config import get_device, parse_train_args, profile_for
from .corpus import build_all
from .dataset import PairDataset, split_pairs
from .model import Seq2SeqTransformer
from .paths import CHECKPOINTS, PROCESSED_DIR
from .tokenizer import BPETokenizer, train_bpe


def _log(message: str) -> None:
    print(message, flush=True)


def _tokenizer_matches_profile(
    tokenizer: BPETokenizer, vocab_size: int, require_domain_tags: bool
) -> bool:
    if require_domain_tags and not tokenizer.domain_ids:
        return False
    return tokenizer.vocab_size >= max(500, int(vocab_size * 0.55))


def prepare(
    force_extract: bool = False,
    vocab_size: int = 4000,
    require_domain_tags: bool = False,
):
    pairs = build_all(force=force_extract, include_adapted_ortho=False)
    tok_path = PROCESSED_DIR / "tokenizer.json"
    tokenizer = None
    if tok_path.exists() and not force_extract:
        tokenizer = BPETokenizer.load(tok_path)
        if not _tokenizer_matches_profile(tokenizer, vocab_size, require_domain_tags):
            _log(
                f"tokenizer.json trop petit ou incomplet "
                f"(vocab={tokenizer.vocab_size}, attendu ≈{vocab_size}) — BPE recréé"
            )
            tokenizer = None
    if tokenizer is None:
        texts = [p["fr"] for p in pairs] + [p["ng"] for p in pairs]
        tokenizer = train_bpe(texts, vocab_size=vocab_size)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)
    train_pairs, val_pairs = split_pairs(pairs)
    return pairs, train_pairs, val_pairs, tokenizer


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    train: bool,
    use_amp: bool,
    scheduler=None,
):
    model.train(train)
    total, ntok = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(src, tgt[:, :-1])
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1),
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            tokens = int((tgt[:, 1:] != model.pad_id).sum().item())
            total += float(loss.item()) * tokens
            ntok += tokens
    return total / max(ntok, 1)


def train(
    epochs: int | None = None,
    batch_size: int | None = None,
    force_extract: bool = False,
    resume: bool = True,
    device_name: str = "auto",
    log=_log,
):
    device = get_device(device_name)
    cfg = profile_for(device)
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    log(f"device={device}  cuda={torch.cuda.is_available()}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")
    else:
        log(
            "ATTENTION: pas de GPU. Profil CPU (petit, lent). "
            "Installez PyTorch CUDA depuis https://pytorch.org puis relancez."
        )
    log(
        f"profile: layers={cfg['num_layers']} d_model={cfg['d_model']} "
        f"batch={cfg['batch_size']} epochs={cfg['epochs']} amp={cfg['amp']} "
        f"ls={cfg.get('label_smoothing')} warmup={cfg.get('warmup_steps')} "
        f"max_len={cfg['max_len']}"
    )

    pairs, train_pairs, val_pairs, tokenizer = prepare(
        force_extract=force_extract,
        vocab_size=cfg["vocab_size"],
        require_domain_tags=bool(cfg.get("use_domain_tags")),
    )
    log(f"{len(pairs)} pairs  train={len(train_pairs)} val={len(val_pairs)}  vocab={tokenizer.vocab_size}")

    train_ds = PairDataset(
        train_pairs,
        tokenizer,
        max_len=cfg["max_len"],
        tone_augment=bool(cfg.get("tone_augment")),
        use_domain_tags=bool(cfg.get("use_domain_tags")),
    )
    val_ds = PairDataset(
        val_pairs,
        tokenizer,
        max_len=cfg["max_len"],
        tone_augment=False,
        use_domain_tags=bool(cfg.get("use_domain_tags")),
    )
    log(f"bidirectional examples  train={len(train_ds)} val={len(val_ds)}")

    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
        pad_id=tokenizer.pad_id,
    ).to(device)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS / "fr_ngiemboon.pt"
    if resume and ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        same_shape = (
            payload.get("vocab_size") == tokenizer.vocab_size
            and payload.get("config", {}).get("d_model") == cfg["d_model"]
            and payload.get("config", {}).get("num_layers") == cfg["num_layers"]
        )
        if same_shape:
            model.load_state_dict(payload["model"])
            log(f"resumed {ckpt}")
        else:
            log("checkpoint architecture differs from this profile — training from scratch")

    optimizer = Adam(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=float(cfg.get("weight_decay") or 0.0),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=float(cfg.get("label_smoothing") or 0.0),
    )
    warmup = int(cfg.get("warmup_steps") or 0)

    def _lr_lambda(step: int) -> float:
        if warmup <= 0:
            return 1.0
        if step < warmup:
            return float(step + 1) / float(warmup)
        return 1.0

    scheduler = LambdaLR(optimizer, _lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    best_val = float("inf")
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(
            model,
            train_ds.batches(cfg["batch_size"], tokenizer.pad_id, shuffle=True),
            optimizer,
            criterion,
            device,
            scaler,
            train=True,
            use_amp=cfg["amp"] and device.type == "cuda",
            scheduler=scheduler,
        )
        val_loss = run_epoch(
            model,
            val_ds.batches(cfg["batch_size"], tokenizer.pad_id, shuffle=False),
            optimizer,
            criterion,
            device,
            scaler,
            train=False,
            use_amp=False,
        )
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        log(f"epoch {epoch}/{cfg['epochs']}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "vocab_size": tokenizer.vocab_size,
                    "pad_id": tokenizer.pad_id,
                },
                ckpt,
            )
            log(f"  saved {ckpt}")

    (CHECKPOINTS / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, tokenizer, history


def load_model(device_name: str = "auto"):
    device = get_device(device_name)
    ckpt_path = CHECKPOINTS / "fr_ngiemboon.pt"
    tok_path = PROCESSED_DIR / "tokenizer.json"
    if not ckpt_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            "Pas de checkpoint. Sur la machine GPU : python -m nmt.train --device auto --no-resume"
        )
    tokenizer = BPETokenizer.load(tok_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = Seq2SeqTransformer(
        vocab_size=payload["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
        pad_id=payload["pad_id"],
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, device


if __name__ == "__main__":
    args = parse_train_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_extract=args.rebuild_corpus,
        resume=not args.no_resume,
        device_name=args.device,
    )
