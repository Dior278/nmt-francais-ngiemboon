"""English ↔ Ngiemboon goes through French.

The trained network only sees French ↔ Ngiemboon. Opus-MT (Helsinki-NLP)
turns English into French, and French back into English.
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch

from .decode import prepare_source, tone_notes, translate_pair

DIRECTIONS = ("fr2ng", "en2ng", "ng2fr", "ng2en")
_MODELS = {
    "en2fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr2en": "Helsinki-NLP/opus-mt-fr-en",
}
_WEIGHT_FILE = {
    "en2fr": "pytorch_model.bin",
    "fr2en": "model.safetensors",
}
_LOADED: dict[str, tuple] = {}

_TEXT = {
    "en": {
        "en2ng": "English is translated into French first, then into Ngiemboon.",
        "ng2en": "Ngiemboon is translated into French first, then into English.",
        "long": "The sentence is past the model length, so the end is dropped.",
    },
    "fr": {
        "en2ng": "L’anglais est d’abord traduit en français, puis en ngiemboon.",
        "ng2en": "Le ngiemboon est d’abord traduit en français, puis en anglais.",
        "long": "La phrase dépasse la longueur du modèle, la fin est ignorée.",
    },
}


class PivotError(RuntimeError):
    """The English–French step could not run."""


def resolve_direction(source: str | None, target: str | None) -> str:
    """Pair a source and a target. French ↔ English alone is not a direction."""
    if source is None and target in {"fr", "en"}:
        source = "ng"
    if source is None:
        source = "fr"
    if target is None:
        target = "fr" if source == "ng" else "ng"
    found = f"{source}2{target}"
    if found not in DIRECTIONS:
        raise ValueError(
            "Translate between Ngiemboon and French, or Ngiemboon and English. "
            f"Got {source} → {target}."
        )
    return found


def pivot(text: str, direction: str) -> str:
    """Translate `en2fr` or `fr2en`. The first call downloads the model."""
    if direction not in _MODELS:
        raise ValueError("direction must be 'en2fr' or 'fr2en'")
    text = text.strip()
    if not text:
        return ""
    tokenizer, model = _load(direction)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        tokens = model.generate(**encoded, max_new_tokens=256)
    return tokenizer.decode(tokens[0], skip_special_tokens=True).strip()


def translate_between(
    model,
    tokenizer,
    text: str,
    direction: str,
    device: torch.device,
    *,
    lang: str = "en",
    pivot_fn=None,
    beam_size: int = 4,
) -> tuple[str, list[str]]:
    """Run one of fr2ng, ng2fr, en2ng, ng2en. English uses `pivot_fn`."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    pack = _TEXT["fr" if lang == "fr" else "en"]
    pivot_fn = pivot if pivot_fn is None else pivot_fn
    notes: list[str] = []
    model_dir = {"en2ng": "fr2ng", "ng2en": "ng2fr"}.get(direction, direction)
    source = text
    if direction == "en2ng":
        source = pivot_fn(text, "en2fr")
        notes.append(pack["en2ng"])
    prepared, missing = prepare_source(source, model_dir)
    target = "ng" if model_dir == "fr2ng" else "fr"
    if len(tokenizer.encode(prepared, lang_tag=target)) > model.max_positions():
        notes.append(pack["long"])
    result = translate_pair(
        model,
        tokenizer,
        prepared,
        model_dir,
        device,
        beam_size=beam_size,
        warn=False,
    )
    if direction == "ng2en":
        french = result
        result = pivot_fn(french, "fr2en")
        notes.append(pack["ng2en"])
        notes.extend(tone_notes(model_dir, missing, french, lang=lang))
    else:
        notes.extend(tone_notes(model_dir, missing, result, lang=lang))
    return result, notes


def _load(direction: str):
    cached = _LOADED.get(direction)
    if cached is not None:
        return cached
    name = _MODELS[direction]
    try:
        from transformers import MarianConfig, MarianMTModel, MarianTokenizer
    except ImportError as exc:
        raise PivotError(
            "English needs transformers and sentencepiece. "
            "Run: pip install transformers sentencepiece"
        ) from exc
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Recommended: pip install sacremoses.")
            tokenizer = MarianTokenizer.from_pretrained(name)
        config = MarianConfig.from_pretrained(name)
        model = MarianMTModel(config)
        state = _read_weights(direction)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.tie_weights()
        tied = {
            "lm_head.weight",
            "model.shared.weight",
            "model.encoder.embed_tokens.weight",
            "model.decoder.embed_tokens.weight",
        }
        loaded_embed = tied.intersection(state) - {"lm_head.weight"}
        real_missing = [key for key in missing if key not in tied]
        if not loaded_embed or real_missing or unexpected:
            raise PivotError(
                f"Unexpected weights for {name}: missing {real_missing or missing}, extra {unexpected}."
            )
    except PivotError:
        raise
    except Exception as exc:
        raise PivotError(
            f"Could not load the English–French model ({name}). "
            f"The first English sentence downloads it. ({exc})"
        ) from exc
    model.eval()
    _LOADED[direction] = (tokenizer, model)
    return tokenizer, model


def _read_weights(direction: str) -> dict:
    """Read the Marian weights. The first English sentence downloads them."""
    path = _weight_path(direction)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def _weight_path(direction: str) -> Path:
    repo = _MODELS[direction]
    filename = _WEIGHT_FILE[direction]
    dest = Path.home() / ".cache" / "nmt-francais-ngiemboon" / repo.replace("/", "--") / filename
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        _download(url, dest)
    except Exception as exc:
        raise PivotError(
            f"Could not download the English–French model ({repo}). ({exc})"
        ) from exc
    return dest


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "nmt-francais-ngiemboon"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        while True:
            chunk = response.read(256 * 1024)
            if not chunk:
                break
            out.write(chunk)
    partial.replace(dest)
