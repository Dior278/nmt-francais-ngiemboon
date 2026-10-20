"""Joint BPE tokenizer.

AGLC letters stay atomic. Ngiemboon tone marks stay their own symbols:
they are not merged into the vowel, so á and ɛ́ share the high-tone sign.
French é è ê stay whole letters.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .normalize import TONE_COMBINING

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


def _holds_tone(piece: str) -> bool:
    return any(ch in TONE_COMBINING for ch in piece)


def _mergable(left: str, right: str) -> bool:
    return not _holds_tone(left) and not _holds_tone(right)


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
            ranked = [
                (self.merge_ranks[p], i)
                for i, p in enumerate(pairs)
                if p in self.merge_ranks and _mergable(*p)
            ]
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


def _apply_merge(pieces: list[str], left: str, right: str, merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(pieces):
        if i < len(pieces) - 1 and pieces[i] == left and pieces[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def train_bpe(texts: list[str], vocab_size: int = 4000, min_freq: int = 3) -> BPETokenizer:
    freqs: Counter[str] = Counter()
    for text in texts:
        freqs.update(TOKEN_RE.findall(text))

    splits: dict[str, list[str]] = {word: list(word) + ["</w>"] for word in freqs}
    alphabet = {sym for pieces in splits.values() for sym in pieces}
    merges: list[tuple[str, str]] = []
    vocab_tokens = set(SPECIAL) | alphabet
    target_merges = max(0, vocab_size - len(vocab_tokens))

    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_words: dict[tuple[str, str], set[str]] = defaultdict(set)
    for word, pieces in splits.items():
        weight = freqs[word]
        for pair in zip(pieces, pieces[1:]):
            if not _mergable(*pair):
                continue
            pair_counts[pair] += weight
            pair_words[pair].add(word)
    heap = [(-freq, pair) for pair, freq in pair_counts.items()]
    heapq.heapify(heap)

    print(f"BPE: {len(freqs)} word types, {len(alphabet)} chars, up to {target_merges} merges", flush=True)

    while len(merges) < target_merges:
        best: tuple[str, str] | None = None
        freq = 0
        while heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current == -neg and current >= min_freq and _mergable(*pair):
                best, freq = pair, current
                break
        if best is None:
            break
        merges.append(best)
        symbol = best[0] + best[1]
        vocab_tokens.add(symbol)
        changed: set[tuple[str, str]] = set()
        for word in list(pair_words.get(best, ())):
            pieces = splits[word]
            weight = freqs[word]
            old_pairs = {pair for pair in zip(pieces, pieces[1:]) if _mergable(*pair)}
            for pair in zip(pieces, pieces[1:]):
                if not _mergable(*pair):
                    continue
                pair_counts[pair] -= weight
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            updated = _apply_merge(pieces, best[0], best[1], symbol)
            splits[word] = updated
            new_pairs = {pair for pair in zip(updated, updated[1:]) if _mergable(*pair)}
            for pair in zip(updated, updated[1:]):
                if not _mergable(*pair):
                    continue
                pair_counts[pair] += weight
            for pair in old_pairs - new_pairs:
                pair_words[pair].discard(word)
            for pair in new_pairs - old_pairs:
                pair_words[pair].add(word)
            changed |= old_pairs | new_pairs
        for pair in changed:
            current = pair_counts.get(pair, 0)
            if current >= min_freq:
                heapq.heappush(heap, (-current, pair))
        if len(merges) % 250 == 0:
            print(f"  merge {len(merges)}/{target_merges}  pair={best!r}  freq={freq}", flush=True)

    vocab = {tok: i for i, tok in enumerate(SPECIAL)}
    for tok in sorted(t for t in vocab_tokens if t not in vocab):
        vocab[tok] = len(vocab)
    print(f"BPE done: vocab={len(vocab)} merges={len(merges)}", flush=True)
    return BPETokenizer(vocab, merges)
