from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.nn.utils.rnn import pad_sequence

from .normalize import domain_of, has_tone_marks, strip_tones
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
        for row in pairs:
            domain = domain_of(row) if use_domain_tags else None
            if domain not in {"bible", "health", "lexicon", "grammar"}:
                domain = None
            fr_to_ng = (
                tokenizer.encode(row["fr"], lang_tag="ng", domain=domain),
                tokenizer.encode(row["ng"], lang_tag=None),
            )
            ng_to_fr = (
                tokenizer.encode(row["ng"], lang_tag="fr", domain=domain),
                tokenizer.encode(row["fr"], lang_tag=None),
            )
            candidates = [fr_to_ng, ng_to_fr]
            if tone_augment and has_tone_marks(row["ng"]):
                stripped = strip_tones(row["ng"])
                if stripped != row["ng"] and stripped.strip():
                    candidates.append(
                        (
                            tokenizer.encode(stripped, lang_tag="fr", domain=domain),
                            tokenizer.encode(row["fr"], lang_tag=None),
                        )
                    )
            for src, tgt in candidates:
                if 3 < len(src) <= max_len and 3 < len(tgt) <= max_len:
                    self.examples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.examples)

    def batches(
        self, batch_size: int, pad_id: int, shuffle: bool = True
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = list(range(len(self.examples)))
        if shuffle:
            random.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = [self.examples[i] for i in order[start : start + batch_size]]
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
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)
    n_val = max(50, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]
