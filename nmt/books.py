"""New Testament book codes and chapter-header patterns.

Ngiemboon names follow the 2007 Bible Society of Cameroon New Testament
(Ngienboum / Ngiemboon). French names follow Louis Segond 1910.
"""

from __future__ import annotations

import re

# Longer / numbered titles first so "1 Jean" is not read as "Jean".
FR_BOOKS: list[tuple[str, str]] = [
    ("1CO", r"1\s+Corinthiens"),
    ("2CO", r"2\s+Corinthiens"),
    ("1TH", r"1\s+Thessaloniciens"),
    ("2TH", r"2\s+Thessaloniciens"),
    ("1TI", r"1\s+Timothée"),
    ("2TI", r"2\s+Timothée"),
    ("1PE", r"1\s+Pierre"),
    ("2PE", r"2\s+Pierre"),
    ("1JN", r"1\s+Jean"),
    ("2JN", r"2\s+Jean"),
    ("3JN", r"3\s+Jean"),
    ("MAT", r"Matthieu"),
    ("MRK", r"Marc"),
    ("LUK", r"Luc"),
    ("JHN", r"Jean"),
    ("ACT", r"Actes"),
    ("ROM", r"Romains"),
    ("GAL", r"Galates"),
    ("EPH", r"Éphésiens"),
    ("PHP", r"Philippiens"),
    ("COL", r"Colossiens"),
    ("TIT", r"Tite"),
    ("PHM", r"Philémon"),
    ("HEB", r"Hébreux"),
    ("JAS", r"Jacques"),
    ("JUD", r"Jude"),
    ("REV", r"Apocalypse"),
]

NG_BOOKS: list[tuple[str, str]] = [
    ("1CO", r"1\s+Korento"),
    ("2CO", r"2\s+Korento"),
    ("1TH", r"1\s+Tesalonika"),
    ("2TH", r"2\s+Tesalonika"),
    ("1TI", r"1\s+Timote"),
    ("2TI", r"2\s+Timote"),
    ("1PE", r"1\s+Piɛ"),
    ("2PE", r"2\s+Piɛ"),
    ("1JN", r"1\s+Jâŋ"),
    ("2JN", r"2\s+Jâŋ"),
    ("3JN", r"3\s+Jâŋ"),
    ("MAT", r"Matio"),
    ("MRK", r"Malékuse"),
    ("LUK", r"Lukase"),
    ("JHN", r"Jâŋ"),
    ("ACT", r"Megàŋ\s+ntúm"),
    ("ROM", r"Roma"),
    ("GAL", r"Galatia"),
    ("EPH", r"Efeso"),
    ("PHP", r"Filipo"),
    ("COL", r"Kolosia"),
    ("TIT", r"Titus"),
    ("PHM", r"Filemo"),
    ("HEB", r"Heber"),
    ("JAS", r"Jâg"),
    ("JUD", r"Jẅide"),
    ("REV", r"Ŋwà['’]ne\s+Ngyà\s+jʉ['’]"),
]

BOOK_LABEL_FR = {
    "MAT": "Matthieu",
    "MRK": "Marc",
    "LUK": "Luc",
    "JHN": "Jean",
    "ACT": "Actes",
    "ROM": "Romains",
    "1CO": "1 Corinthiens",
    "2CO": "2 Corinthiens",
    "GAL": "Galates",
    "EPH": "Éphésiens",
    "PHP": "Philippiens",
    "COL": "Colossiens",
    "1TH": "1 Thessaloniciens",
    "2TH": "2 Thessaloniciens",
    "1TI": "1 Timothée",
    "2TI": "2 Timothée",
    "TIT": "Tite",
    "PHM": "Philémon",
    "HEB": "Hébreux",
    "JAS": "Jacques",
    "1PE": "1 Pierre",
    "2PE": "2 Pierre",
    "1JN": "1 Jean",
    "2JN": "2 Jean",
    "3JN": "3 Jean",
    "JUD": "Jude",
    "REV": "Apocalypse",
}


def chapter_header_re(books: list[tuple[str, str]]) -> re.Pattern[str]:
    parts = [rf"(?P<{code}>{pat})\s+(?P<{code}_ch>\d{{1,3}})" for code, pat in books]
    return re.compile(rf"^(?:{'|'.join(parts)})$")


def match_chapter_header(line: str, pattern: re.Pattern[str]) -> tuple[str, int] | None:
    m = pattern.match(line.strip())
    if not m:
        return None
    for code, _ in (FR_BOOKS if "Matthieu" in pattern.pattern else NG_BOOKS):
        if m.group(code):
            return code, int(m.group(f"{code}_ch"))
    return None


def detect_chapter(line: str, lang: str) -> tuple[str, int] | None:
    books = FR_BOOKS if lang == "fr" else NG_BOOKS
    text = line.strip()
    for code, pat in books:
        m = re.fullmatch(rf"{pat}\s+(\d{{1,3}})", text)
        if m:
            return code, int(m.group(1))
    return None
