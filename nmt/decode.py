from __future__ import annotations

import torch

from .model import Seq2SeqTransformer
from .normalize import has_tone_marks, nfc, strip_tones
from .tokenizer import BPETokenizer


def prepare_source(text: str, direction: str) -> tuple[str, bool]:
    """Normalize input. For NG→FR without tones, strip marks so the
    tone-augmented training examples can match. Warn the caller: a missing
    tone is still ambiguous (má ≠ mà).
    """
    text = nfc(text.strip())
    tone_missing = False
    if direction == "ng2fr" and text and not has_tone_marks(text):
        text = strip_tones(text)
        tone_missing = True
    return text, tone_missing


@torch.no_grad()
def greedy_translate(
    model: Seq2SeqTransformer,
    tokenizer: BPETokenizer,
    text: str,
    target_lang: str,
    device: torch.device,
    max_len: int | None = None,
    domain: str | None = None,
) -> str:
    model.eval()
    if max_len is None:
        try:
            max_len = min(int(model.pe.pe.size(1)) - 1, 160)
        except Exception:
            max_len = 96
    src = torch.tensor(
        [tokenizer.encode(text, lang_tag=target_lang, domain=domain)],
        dtype=torch.long,
        device=device,
    )
    src_mask = (src != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_pad = (ys != tokenizer.pad_id).unsqueeze(1).unsqueeze(3)
        causal = torch.tril(torch.ones(1, ys.size(1), ys.size(1), device=device)).bool()
        tgt_mask = tgt_pad & causal
        logits = model.decode(ys, memory, src_mask, tgt_mask)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ys[0].tolist())


def translate_pair(
    model: Seq2SeqTransformer,
    tokenizer: BPETokenizer,
    text: str,
    direction: str,
    device: torch.device,
    domain: str | None = None,
) -> str:
    if direction not in {"fr2ng", "ng2fr"}:
        raise ValueError("direction must be 'fr2ng' or 'ng2fr'")
    text, _ = prepare_source(text, direction)
    target = "ng" if direction == "fr2ng" else "fr"
    return greedy_translate(model, tokenizer, text, target, device, domain=domain)
