from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.nn.utils.rnn import pad_sequence

from .normalize import domain_of, has_tone_marks, normalize_fr, normalize_ng, strip_tones
from .tokenizer import BPETokenizer


class PairDataset:
    def __init__(
        self,
        pairs: list[dict],
        tokenizer: BPETokenizer,
        max_len: int = 128,
        tone_augment: bool = False,
        use_domain_tags: bool = False,
    ):
        self.examples: list[tuple[list[int], list[int]]] = []
        # Half the rows keep a domain tag, half do not, so a plain
        # `python -m nmt "phrase"` (no tag) still matches training.
        rng = random.Random(0)
        for row in pairs:
            domain = domain_of(row) if use_domain_tags else None
            if domain not in {"bible", "health", "lexicon", "grammar"}:
                domain = None
            if domain is not None and rng.random() >= 0.5:
                domain = None
            fr = normalize_fr(row["fr"])
            ng = normalize_ng(row["ng"])
            fr_to_ng = (
                tokenizer.encode(fr, lang_tag="ng", domain=domain),
                tokenizer.encode(ng, lang_tag=None),
            )
            ng_to_fr = (
                tokenizer.encode(ng, lang_tag="fr", domain=domain),
                tokenizer.encode(fr, lang_tag=None),
            )
            candidates = [fr_to_ng, ng_to_fr]
            # Toneless copy is only NG→FR. FR→NG must keep the AGLC tones.
            if tone_augment and has_tone_marks(ng):
                stripped = strip_tones(ng)
                if stripped != ng and stripped.strip():
                    candidates.append(
                        (
                            tokenizer.encode(stripped, lang_tag="fr", domain=domain),
                            tokenizer.encode(fr, lang_tag=None),
                        )
                    )
            for src, tgt in candidates:
                if 3 < len(src) <= max_len and 3 < len(tgt) <= max_len:
                    self.examples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.examples)

    def batches(
        self, batch_size: int, pad_id: int, shuffle: bool = True, seed: int | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = list(range(len(self.examples)))
        order.sort(key=lambda i: len(self.examples[i][0]))
        groups = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        if shuffle:
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(groups)
        for group in groups:
            chunk = [self.examples[i] for i in group]
            src = pad_sequence(
                [torch.tensor(s, dtype=torch.long) for s, _ in chunk],
                batch_first=True,
                padding_value=pad_id,
            )
            tgt = pad_sequence(
                [torch.tensor(t, dtype=torch.long) for _, t in chunk],
                batch_first=True,
                padding_value=pad_id,
            )
            yield src, tgt


def split_pairs(pairs: list[dict], seed: int = 0, val_ratio: float = 0.05):
    """Hold out about 5%, and at least 50 pairs once the corpus is large enough.

    A smaller corpus keeps a train set: the old floor of 50 could swallow it.
    """
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)
    if len(items) < 2:
        return items, list(items)
    n_val = max(1, int(len(items) * val_ratio))
    if len(items) >= 1000:
        n_val = max(50, n_val)
    n_val = min(n_val, len(items) - 1)
    return items[n_val:], items[:n_val]
