"""Joint BPE tokenizer. AGLC letters and tone marks stay atomic."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIAL = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<2fr>",
    "<2ng>",
    "<bible>",
    "<health>",
    "<lexicon>",
    "<grammar>",
]
TOKEN_RE = re.compile(r"\S+")


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.merges = merges
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.pad_id = vocab["<pad>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.unk_id = vocab["<unk>"]
        self.to_fr = vocab["<2fr>"]
        self.to_ng = vocab["<2ng>"]
        self.domain_ids = {
            name: vocab[tok]
            for name, tok in (
                ("bible", "<bible>"),
                ("health", "<health>"),
                ("lexicon", "<lexicon>"),
                ("grammar", "<grammar>"),
            )
            if tok in vocab
        }
        self._word_cache: dict[str, list[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_word(self, word: str) -> list[str]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        chars = list(word) + ["</w>"]
        while len(chars) >= 2:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            ranked = [(self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks]
            if not ranked:
                break
            _, idx = min(ranked)
            chars = chars[:idx] + [chars[idx] + chars[idx + 1]] + chars[idx + 2 :]
        self._word_cache[word] = chars
        return chars

    def encode(
        self,
        text: str,
        lang_tag: str | None = None,
        add_special: bool = True,
        domain: str | None = None,
    ) -> list[int]:
        pieces: list[str] = []
        for word in TOKEN_RE.findall(text.strip()):
            pieces.extend(self._encode_word(word))
        ids = [self.vocab.get(p, self.unk_id) for p in pieces]
        if add_special:
            prefix = []
            if lang_tag == "fr":
                prefix.append(self.to_fr)
            elif lang_tag == "ng":
                prefix.append(self.to_ng)
            if domain and domain in self.domain_ids:
                prefix.append(self.domain_ids[domain])
            ids = prefix + [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id, self.to_fr, self.to_ng, self.unk_id}
        skip.update(self.domain_ids.values())
        tokens = [self.id_to_token.get(i, "") for i in ids if i not in skip]
        return re.sub(r"\s+", " ", "".join(tokens).replace("</w>", " ")).strip()

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"vocab": self.vocab, "merges": self.merges}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(payload["vocab"], merges)


def train_bpe(texts: list[str], vocab_size: int = 4000, min_freq: int = 3) -> BPETokenizer:
    freqs: Counter[str] = Counter()
    for text in texts:
        freqs.update(TOKEN_RE.findall(text))

    splits: dict[str, list[str]] = {word: list(word) + ["</w>"] for word in freqs}
    alphabet = {sym for pieces in splits.values() for sym in pieces}
    merges: list[tuple[str, str]] = []
    vocab_tokens = set(SPECIAL) | alphabet
    target_merges = max(0, vocab_size - len(vocab_tokens))

    print(f"BPE: {len(freqs)} word types, {len(alphabet)} chars, up to {target_merges} merges", flush=True)

    for step in range(target_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, pieces in splits.items():
            if len(pieces) < 2:
                continue
            w = freqs[word]
            for a, b in zip(pieces, pieces[1:]):
                pair_counts[(a, b)] += w
        if not pair_counts:
            break
        best, freq = pair_counts.most_common(1)[0]
        if freq < min_freq:
            break
        merges.append(best)
        new = best[0] + best[1]
        vocab_tokens.add(new)
        for word, pieces in splits.items():
            i = 0
            merged: list[str] = []
            while i < len(pieces):
                if i < len(pieces) - 1 and pieces[i] == best[0] and pieces[i + 1] == best[1]:
                    merged.append(new)
                    i += 2
                else:
                    merged.append(pieces[i])
                    i += 1
            splits[word] = merged
        if (step + 1) % 250 == 0:
            print(f"  merge {step + 1}/{target_merges}  pair={best!r}  freq={freq}", flush=True)

    vocab = {tok: i for i, tok in enumerate(SPECIAL)}
    for tok in sorted(t for t in vocab_tokens if t not in vocab):
        vocab[tok] = len(vocab)
    print(f"BPE done: vocab={len(vocab)} merges={len(merges)}", flush=True)
    return BPETokenizer(vocab, merges)
