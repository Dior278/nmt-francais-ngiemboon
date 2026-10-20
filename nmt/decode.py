from __future__ import annotations

import sys

import torch

from .model import Seq2SeqTransformer, causal_mask
from .normalize import has_tone_marks, normalize_fr, normalize_ng, present
from .tokenizer import BPETokenizer


def prepare_source(text: str, direction: str) -> tuple[str, bool]:
    """French in, French accents. Ngiemboon in, AGLC tones as their own signs.

    NG→FR with no tone at all matches the toneless training copy.
    The meaning is still ambiguous (má ≠ mà).
    """
    if direction == "ng2fr":
        text = normalize_ng(text)
        return text, bool(text) and not has_tone_marks(text)
    return normalize_fr(text), False


def tone_notes(
    direction: str,
    source_missing: bool,
    output: str,
    *,
    keyboard: bool = False,
) -> list[str]:
    notes: list[str] = []
    if direction == "ng2fr" and source_missing:
        extra = " Utilisez le clavier AGLC." if keyboard else ""
        notes.append(
            "Aucun ton en entrée : le français proposé est une approximation "
            "(má « mère » ≠ mà « boubou »)." + extra
        )
    if direction == "fr2ng" and output and not has_tone_marks(output):
        notes.append(
            "Sortie sans ton AGLC : le sens peut changer (má « mère » ≠ mà « boubou »)."
        )
    return notes


def _generation_limit(model: Seq2SeqTransformer, max_len: int | None) -> int:
    if max_len is not None:
        return max_len
    try:
        return min(model.max_positions() - 1, 128)
    except Exception:
        return 96


def _banned_ids(tokenizer: BPETokenizer) -> list[int]:
    ids = [tokenizer.pad_id, tokenizer.unk_id, tokenizer.to_fr, tokenizer.to_ng]
    ids.extend(tokenizer.domain_ids.values())
    return list(dict.fromkeys(ids))


def _source_tensor(model, tokenizer, text, target_lang, device, domain, warn: bool = True):
    limit = model.max_positions()
    ids = tokenizer.encode(text, lang_tag=target_lang, domain=domain)
    if len(ids) > limit:
        if warn:
            print(
                "Note : la phrase dépasse la longueur du modèle, la fin est ignorée.",
                file=sys.stderr,
            )
        ids = ids[:limit]
        ids[-1] = tokenizer.eos_id
    return torch.tensor([ids], dtype=torch.long, device=device)


def _step_logits(model, tokenizer, ys, memory, src_mask):
    logits = model.decode(ys, memory, src_mask, causal_mask(ys, tokenizer.pad_id))[:, -1, :]
    logits[:, _banned_ids(tokenizer)] = float("-inf")
    return logits


@torch.no_grad()
def greedy_translate(
    model: Seq2SeqTransformer,
    tokenizer: BPETokenizer,
    text: str,
    target_lang: str,
    device: torch.device,
    max_len: int | None = None,
    domain: str | None = None,
    warn: bool = True,
) -> str:
    model.eval()
    max_len = _generation_limit(model, max_len)
    src = _source_tensor(model, tokenizer, text, target_lang, device, domain, warn=warn)
    src_mask = (src != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    for _ in range(max_len):
        next_id = int(_step_logits(model, tokenizer, ys, memory, src_mask).argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.eos_id:
            break
    return present(tokenizer.decode(ys[0].tolist()), target_lang)


def _length_penalty(score: float, length: int, alpha: float = 0.6) -> float:
    return score / ((5 + max(length, 1)) / 6) ** alpha


@torch.no_grad()
def beam_translate(
    model: Seq2SeqTransformer,
    tokenizer: BPETokenizer,
    text: str,
    target_lang: str,
    device: torch.device,
    beam_size: int = 4,
    max_len: int | None = None,
    domain: str | None = None,
    warn: bool = True,
) -> str:
    if beam_size <= 1:
        return greedy_translate(
            model, tokenizer, text, target_lang, device, max_len, domain, warn=warn
        )
    model.eval()
    max_len = _generation_limit(model, max_len)
    src = _source_tensor(model, tokenizer, text, target_lang, device, domain, warn=warn)
    src_mask = (src != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    beams: list[tuple[float, list[int]]] = [(0.0, [tokenizer.bos_id])]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(max_len):
        expanded: list[tuple[float, list[int]]] = []
        for score, ids in beams:
            if ids[-1] == tokenizer.eos_id:
                finished.append((score, ids))
                continue
            ys = torch.tensor([ids], dtype=torch.long, device=device)
            logp = torch.log_softmax(_step_logits(model, tokenizer, ys, memory, src_mask), dim=-1)
            topv, topi = logp.topk(beam_size)
            for value, index in zip(topv[0].tolist(), topi[0].tolist()):
                expanded.append((score + value, ids + [int(index)]))
        if not expanded:
            break
        expanded.sort(key=lambda item: _length_penalty(item[0], len(item[1]) - 1), reverse=True)
        beams = expanded[:beam_size]
        if all(ids[-1] == tokenizer.eos_id for _, ids in beams):
            finished.extend(beams)
            break
    finished.extend(beams)
    best = max(finished, key=lambda item: _length_penalty(item[0], len(item[1]) - 1))
    return present(tokenizer.decode(best[1]), target_lang)


def translate_pair(
    model: Seq2SeqTransformer,
    tokenizer: BPETokenizer,
    text: str,
    direction: str,
    device: torch.device,
    domain: str | None = None,
    beam_size: int = 4,
) -> str:
    if direction not in {"fr2ng", "ng2fr"}:
        raise ValueError("direction must be 'fr2ng' or 'ng2fr'")
    text, _ = prepare_source(text, direction)
    target = "ng" if direction == "fr2ng" else "fr"
    return beam_translate(
        model, tokenizer, text, target, device, beam_size=beam_size, domain=domain
    )
