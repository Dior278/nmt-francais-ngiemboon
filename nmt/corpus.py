"""Build the merged French ↔ Ngiemboon corpus from every usable PDF."""

from __future__ import annotations

import json
from collections import Counter

from .normalize import PAIR_FIELDS, only_pair_fields
from .parse_bibles import build_corpus as build_bible
from .parse_extra import extra_sentence_pairs, lexicon_pairs
from .paths import PROCESSED_DIR


def _bucket(row: dict) -> str:
    book = row.get("book", "")
    pid = row.get("id", "")
    if book == "HEALTH" or pid.startswith("HEALTH."):
        return "health"
    if pid.startswith("SIL.S."):
        return "sil_sentence"
    if book == "SIL" or pid.startswith("SIL."):
        return "sil"
    if book == "LONFO" or pid.startswith("LONFO."):
        return "lonfo"
    if book == "NDIOLA" or pid.startswith("NDIOLA."):
        return "ndiola"
    return "bible"


def _cache_includes_adapted() -> bool:
    stats_path = PROCESSED_DIR / "stats.json"
    if not stats_path.exists():
        return False
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(stats.get("include_adapted_ortho"))


def _load_cached_pairs(include_adapted_ortho: bool) -> list[dict] | None:
    cache_path = PROCESSED_DIR / "pairs.jsonl"
    if not cache_path.exists():
        return None
    rows = [
        only_pair_fields(json.loads(line))
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if include_adapted_ortho and not _cache_includes_adapted():
        if not any(_bucket(row) == "ndiola" for row in rows):
            return None
    if not include_adapted_ortho:
        rows = [row for row in rows if _bucket(row) != "ndiola"]
    return rows


def _write_corpus(
    unique: list[dict],
    lexicon: list[dict] | None = None,
    include_adapted_ortho: bool = False,
) -> None:
    stats = {
        "total": len(unique),
        "by_book_group": dict(Counter(_bucket(p) for p in unique)),
        "columns": list(PAIR_FIELDS),
        "include_adapted_ortho": include_adapted_ortho,
        "note": (
            "Ndiola (id prefix NDIOLA. / book NDIOLA) is excluded unless "
            "--include-adapted-ortho. Wrong tones change meaning; do not mix "
            "adapted spelling into AGLC training."
        ),
    }
    (PROCESSED_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (PROCESSED_DIR / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in unique:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if lexicon is not None:
        (PROCESSED_DIR / "lexicon.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in lexicon) + "\n",
            encoding="utf-8",
        )


def build_all(force: bool = False, include_adapted_ortho: bool = False) -> list[dict]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force:
        cached = _load_cached_pairs(include_adapted_ortho)
        if cached:
            return cached

    bible = [only_pair_fields(p) for p in build_bible(force=force)]
    extra = [only_pair_fields(p) for p in extra_sentence_pairs(include_adapted_ortho=include_adapted_ortho)]
    lexicon = [only_pair_fields(p) for p in lexicon_pairs()]

    merged = bible + extra + lexicon
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in merged:
        key = (row["fr"].casefold(), row["ng"].casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append({field: row[field] for field in PAIR_FIELDS})

    _write_corpus(unique, lexicon, include_adapted_ortho=include_adapted_ortho)
    return unique


def _preview(rows: list[dict], book: str, n: int = 2) -> list[dict]:
    picked = [r for r in rows if r.get("book") == book][:n]
    return [{"id": r["id"], "fr": r["fr"][:140], "ng": r["ng"][:140], "ref": r.get("ref")} for r in picked]


if __name__ == "__main__":
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Extraire le corpus FR ↔ ngiemboon (pas d'entraînement).")
    parser.add_argument("--force", action="store_true", help="Relire les PDF (Bible + extras).")
    parser.add_argument(
        "--include-adapted-ortho",
        action="store_true",
        help="Inclure Ndiola 2011 (book=NDIOLA). Désactivé par défaut.",
    )
    args = parser.parse_args()
    data = build_all(force=args.force, include_adapted_ortho=args.include_adapted_ortho)
    stats = json.loads((PROCESSED_DIR / "stats.json").read_text(encoding="utf-8"))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for book in ("MAT", "HEALTH", "SIL", "LONFO", "NDIOLA"):
        sample = _preview(data, book)
        if sample:
            print(f"\n--- {book} ---")
            print(json.dumps(sample, ensure_ascii=False, indent=2))
