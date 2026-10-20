"""Train the French ↔ Ngiemboon Transformer. CPU is supported; CUDA is used when it exists."""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .config import get_device, parse_train_args, profile_for
from .corpus import build_all
from .normalize import normalize_fr, normalize_ng
from .dataset import PairDataset, split_pairs
from .decode import greedy_translate
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
    # High and low tones must be their own AGLC symbols, not hidden inside á / à.
    if "\u0300" not in tokenizer.vocab or "\u0301" not in tokenizer.vocab:
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
                f"tokenizer.json incomplet "
                f"(vocab={tokenizer.vocab_size}, tons AGLC ou domaine) — BPE recréé"
            )
            tokenizer = None
    for row in pairs:
        row["fr"] = normalize_fr(row.get("fr") or "")
        row["ng"] = normalize_ng(row.get("ng") or "")
    if tokenizer is None:
        texts = [p["fr"] for p in pairs] + [p["ng"] for p in pairs]
        tokenizer = train_bpe(texts, vocab_size=vocab_size)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)
    train_pairs, val_pairs = split_pairs(pairs)
    return pairs, train_pairs, val_pairs, tokenizer


def _chrf(hyp: str, ref: str, n: int = 4) -> float:
    def grams(text: str, k: int) -> Counter:
        chars = text.replace(" ", "")
        return Counter(chars[i : i + k] for i in range(max(0, len(chars) - k + 1)))

    scores = []
    for k in range(1, n + 1):
        h, r = grams(hyp, k), grams(ref, k)
        if not h or not r:
            scores.append(0.0)
            continue
        overlap = sum(min(h[g], r[g]) for g in h)
        prec = overlap / sum(h.values())
        rec = overlap / sum(r.values())
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores)


def _sample_chrf(model, tokenizer, val_pairs, device, n: int = 24) -> tuple[float, float]:
    """chrF on both directions. The budget stays `n` decodes (half each way)."""
    sample = val_pairs[: max(1, n // 2)]
    if not val_pairs:
        return 0.0, 0.0
    fr_scores = [
        _chrf(
            greedy_translate(model, tokenizer, row["fr"], "ng", device, max_len=96, warn=False),
            row["ng"],
        )
        for row in sample
    ]
    ng_scores = [
        _chrf(
            greedy_translate(model, tokenizer, row["ng"], "fr", device, max_len=96, warn=False),
            row["fr"],
        )
        for row in sample
    ]
    return sum(fr_scores) / len(fr_scores), sum(ng_scores) / len(ng_scores)


def _checkpoint_matches(payload: dict, cfg: dict, vocab_size: int, pad_id: int) -> bool:
    saved = payload.get("config") or {}
    return (
        payload.get("vocab_size") == vocab_size
        and payload.get("pad_id") == pad_id
        and saved.get("d_model") == cfg["d_model"]
        and saved.get("num_layers") == cfg["num_layers"]
        and saved.get("num_heads") == cfg["num_heads"]
        and saved.get("d_ff") == cfg["d_ff"]
        and saved.get("max_len") == cfg["max_len"]
    )


def _atomic_save(payload: dict, path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validation_loss(model, val_ds, criterion, device, cfg, tokenizer) -> float:
    return run_epoch(
        model,
        val_ds.batches(cfg["batch_size"], tokenizer.pad_id, shuffle=False),
        None,
        criterion,
        device,
        None,
        train=False,
        use_amp=False,
    )


def _load_resume(resume_path, ckpt, cfg, tokenizer, device, log):
    """Prefer the latest training state, then the best weights."""
    for path in (resume_path, ckpt):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:
            log(f"{path.name} illisible ({exc}) — ignoré")
            continue
        if not _checkpoint_matches(payload, cfg, tokenizer.vocab_size, tokenizer.pad_id):
            log(f"{path.name} a une autre taille — ignoré")
            continue
        return path, payload
    return None, None


def _restore_optim(payload, optimizer, scheduler, scaler, log) -> None:
    if "optimizer" not in payload:
        log("pas d'état d'optimiseur : le pas d'apprentissage repart au début.")
        return
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except Exception as exc:
        log(f"optimiseur non repris ({exc})")
        return
    if "scheduler" in payload:
        try:
            scheduler.load_state_dict(payload["scheduler"])
        except Exception as exc:
            log(f"ordonnance de pas non reprise ({exc})")
    if scaler.is_enabled() and payload.get("scaler"):
        try:
            scaler.load_state_dict(payload["scaler"])
        except Exception as exc:
            log(f"scaler non repris ({exc})")


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
    log=None,
):
    model.train(train)
    total, ntok = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, (src, tgt) in enumerate(loader, start=1):
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
            if train and log is not None and step % 100 == 0:
                log(f"  lot {step}  loss {loss.item():.3f}")
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

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log(f"device={device}  cuda={torch.cuda.is_available()}  threads={torch.get_num_threads()}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")
    else:
        log("Profil CPU : 3 couches, d_model=192, assez petit pour ~8 000 paires.")
    log(
        f"profile: layers={cfg['num_layers']} d_model={cfg['d_model']} "
        f"batch={cfg['batch_size']} epochs={cfg['epochs']} amp={cfg['amp']} "
        f"ls={cfg.get('label_smoothing')} warmup={cfg.get('warmup_steps')} "
        f"max_len={cfg['max_len']} patience={cfg.get('patience')}"
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
    n_params = sum(p.numel() for p in model.parameters())
    log(f"paramètres={n_params / 1e6:.2f} M")
    if len(train_ds) == 0:
        raise RuntimeError("Aucune phrase d'entraînement (corpus vide ou phrases trop longues).")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS / "fr_ngiemboon.pt"
    resume_path = CHECKPOINTS / "fr_ngiemboon.resume.pt"

    optimizer = Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=float(cfg.get("weight_decay") or 0.0),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=float(cfg.get("label_smoothing") or 0.0),
    )
    warmup = max(int(cfg.get("warmup_steps") or 1), 1)
    lr_factor = float(cfg.get("lr_factor") or 0.5)
    d_model = int(cfg["d_model"])

    def _lr_lambda(step: int) -> float:
        # LambdaLR starts at 0. Noam: peak around step == warmup, then 1/sqrt.
        step = max(int(step) + 1, 1)
        scale = (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
        return lr_factor * scale

    scheduler = LambdaLR(optimizer, _lr_lambda)
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    stale = 0
    patience = int(cfg.get("patience") or 0)
    history: list[dict] = []
    start_epoch = 1
    if resume:
        origin, payload = _load_resume(resume_path, ckpt, cfg, tokenizer, device, log)
        if payload is not None:
            try:
                model.load_state_dict(payload["model"])
            except RuntimeError as exc:
                log(f"poids incompatibles ({exc}) — entraînement depuis zéro")
                payload = None
        if payload is not None:
            _restore_optim(payload, optimizer, scheduler, scaler, log)
            start_epoch = int(payload.get("epoch", 0)) + 1
            stale = int(payload.get("stale", 0))
            history = list(payload.get("history") or [])
            if "best_val" in payload:
                best_val = float(payload["best_val"])
            else:
                best_val = _validation_loss(model, val_ds, criterion, device, cfg, tokenizer)
                log(f"validation du modèle repris : {best_val:.4f}")
            log(
                f"reprise depuis {origin.name} à l'epoch {start_epoch}  "
                f"best_val {best_val:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    def _snapshot(epoch: int, include_optim: bool) -> dict:
        payload = {
            "model": model.state_dict(),
            "config": cfg,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "epoch": epoch,
            "best_val": best_val,
            "stale": stale,
            "history": history,
        }
        if include_optim:
            payload["optimizer"] = optimizer.state_dict()
            payload["scheduler"] = scheduler.state_dict()
            if scaler.is_enabled():
                payload["scaler"] = scaler.state_dict()
        return payload

    def _write_history() -> None:
        if history:
            (CHECKPOINTS / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

    try:
        if start_epoch > cfg["epochs"]:
            log(
                f"déjà entraîné jusqu'à l'epoch {start_epoch - 1} "
                f"(maximum {cfg['epochs']}). Relancez avec --epochs plus grand pour continuer."
            )
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            started = time.perf_counter()
            torch.manual_seed(epoch)
            train_loss = run_epoch(
                model,
                train_ds.batches(cfg["batch_size"], tokenizer.pad_id, shuffle=True, seed=epoch),
                optimizer,
                criterion,
                device,
                scaler,
                train=True,
                use_amp=use_amp,
                scheduler=scheduler,
                log=log,
            )
            val_loss = _validation_loss(model, val_ds, criterion, device, cfg, tokenizer)
            chrf_to_ng, chrf_to_fr = _sample_chrf(model, tokenizer, val_pairs, device)
            minutes = (time.perf_counter() - started) / 60
            history.append(
                {
                    "epoch": epoch,
                    "train": train_loss,
                    "val": val_loss,
                    "chrf": (chrf_to_ng + chrf_to_fr) / 2,
                    "chrf_fr2ng": chrf_to_ng,
                    "chrf_ng2fr": chrf_to_fr,
                    "minutes": minutes,
                }
            )
            log(
                f"epoch {epoch}/{cfg['epochs']}  train {train_loss:.4f}  "
                f"val {val_loss:.4f}  chrF fr→ng {chrf_to_ng:.3f} ng→fr {chrf_to_fr:.3f}  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}  {minutes:.1f} min"
            )
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                stale = 0
                _atomic_save(_snapshot(epoch, include_optim=False), ckpt)
                log(f"  meilleur modèle → {ckpt}")
            else:
                stale += 1
            _atomic_save(_snapshot(epoch, include_optim=True), resume_path)
            _write_history()
            if patience and stale >= patience:
                log(f"validation stable depuis {patience} epochs — arrêt.")
                break
    except KeyboardInterrupt:
        log("interrompu — le meilleur checkpoint déjà écrit est conservé.")
    finally:
        _write_history()
    return model, tokenizer, history


def load_model(device_name: str = "auto"):
    device = get_device(device_name)
    ckpt_path = CHECKPOINTS / "fr_ngiemboon.pt"
    tok_path = PROCESSED_DIR / "tokenizer.json"
    if not ckpt_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            "Pas de modèle. Lancez : python -m nmt train"
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
