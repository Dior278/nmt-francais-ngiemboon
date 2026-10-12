"""Extract verse-aligned French / Ngiemboon pairs from the two Bible PDFs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import fitz

from .books import BOOK_LABEL_FR, detect_chapter
from .normalize import make_pair
from .paths import DATA_DIR, PROCESSED_DIR

FR_PDF = DATA_DIR / "Francais-Bible-Segond-1910-All-Bible.pdf"
NG_PDF = DATA_DIR / "Ngiemboon-Bible-New-Testasment.pdf"

FR_NOISE = (
    "La Sainte Bible par Louis Segond 1910",
    "Français - All Bible",
    "Francais - All Bible",
    "(Bible Segond - 1910)",
    "Français Language",
    "Francais Language",
)
NG_NOISE = (
    "Nouveau Testament en Ngienboum.",
    "Ngiemboon - Bible - New Testasment",
    "Ngiemboon  Language",
    "Alliance Biblique du Cameroun",
    "Bible Society of Cameroon",
)

CROSS_REF_LINE = re.compile(
    r"^V\.\s*\d+",
)
CROSS_REF_PAREN = re.compile(r"\([^)]*\d[^)]*\)")
VERSE_SPLIT = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=\s))(\d{1,3})(?=\s*\w)", re.UNICODE)
MULTISPACE = re.compile(r"\s+")


def extract_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def _is_noise(line: str, extra: tuple[str, ...]) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.isdigit() and len(s) <= 5:
        return True
    if s.startswith("©"):
        return True
    return any(token in s for token in extra)


def clean_pages(raw: str, extra_noise: tuple[str, ...]) -> list[str]:
    kept: list[str] = []
    for line in raw.splitlines():
        if _is_noise(line, extra_noise):
            continue
        kept.append(line.rstrip())
    return kept


def parse_verses(lines: list[str], lang: str) -> dict[str, str]:
    """Return {MAT.1.1: verse text, ...}."""
    verses: dict[str, str] = {}
    book: str | None = None
    chapter: int | None = None
    buf: list[str] = []

    def flush_chapter() -> None:
        nonlocal buf
        if book is None or chapter is None or not buf:
            buf = []
            return
        body = " ".join(buf)
        body = CROSS_REF_PAREN.sub(" ", body)
        body = MULTISPACE.sub(" ", body).strip()
        parts = VERSE_SPLIT.split(body)
        # parts = [preamble, num, text, num, text, ...]
        for i in range(1, len(parts), 2):
            num = int(parts[i])
            text = MULTISPACE.sub(" ", parts[i + 1]).strip(" .;:")
            text = text.strip()
            if len(text) < 8:
                continue
            verses[f"{book}.{chapter}.{num}"] = text
        buf = []

    for line in lines:
        stripped = line.strip()
        if CROSS_REF_LINE.match(stripped):
            continue
        header = detect_chapter(stripped, lang)
        if header:
            flush_chapter()
            book, chapter = header
            continue
        if book is None:
            continue
        buf.append(stripped)
    flush_chapter()
    return verses


def align_verses(fr: dict[str, str], ng: dict[str, str]) -> list[dict]:
    keys = sorted(set(fr) & set(ng), key=_verse_key)
    pairs = []
    for key in keys:
        pairs.append(
            make_pair(key, key.split(".")[0], _pretty_ref(key), fr[key], ng[key])
        )
    return pairs


def _verse_key(key: str) -> tuple[str, int, int]:
    book, ch, vs = key.split(".")
    return book, int(ch), int(vs)


def _pretty_ref(key: str) -> str:
    book, ch, vs = key.split(".")
    return f"{BOOK_LABEL_FR[book]} {ch}.{vs}"


def build_corpus(force: bool = False) -> list[dict]:
    """Extract NT verse pairs. Cached in bible_pairs.jsonl (not the merged corpus)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "bible_pairs.jsonl"
    stats_path = PROCESSED_DIR / "bible_stats.json"
    if out_path.exists() and not force:
        return [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]

    if not FR_PDF.exists() or not NG_PDF.exists():
        raise FileNotFoundError(
            f"PDFs missing. Expected:\n  {FR_PDF}\n  {NG_PDF}"
        )

    fr_raw = extract_pdf_text(FR_PDF)
    ng_raw = extract_pdf_text(NG_PDF)
    fr_verses = parse_verses(clean_pages(fr_raw, FR_NOISE), "fr")
    ng_verses = parse_verses(clean_pages(ng_raw, NG_NOISE), "ng")
    pairs = align_verses(fr_verses, ng_verses)

    stats = {
        "fr_verses": len(fr_verses),
        "ng_verses": len(ng_verses),
        "aligned": len(pairs),
        "fr_only": len(set(fr_verses) - set(ng_verses)),
        "ng_only": len(set(ng_verses) - set(fr_verses)),
        "by_book": dict(Counter(p["book"] for p in pairs)),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return pairs


if __name__ == "__main__":
    data = build_corpus(force=True)
    print(f"aligned pairs: {len(data)}")
    print("sample MAT.1.1:")
    for row in data[:2]:
        print(row["ref"])
        print(" FR:", row["fr"][:160])
        print(" NG:", row["ng"][:160])
