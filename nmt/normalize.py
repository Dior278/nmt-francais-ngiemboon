"""AGLC cleaning, Timene PUA repair, tone folding, quality gates.

The Timene 2024 PDF embeds Liberation Serif without a ToUnicode map for
letters the font subset never named (ń ḿ ì ò ú í ě Ŋ and spacing tones).
PyMuPDF exposes those CIDs as U+E001..U+E010. The mapping below is
recovered from context against the 2007 NT (manzwě, nyìŋ, ńdɔg, ḿbiŋ,
zɔ́g, ngwòŋ, tsítsà) and Anderson's tables (láŋ, kàŋsê).
"""

from __future__ import annotations

import re
import unicodedata

# Letters the AGLC / Anderson 2007 orthography actually uses.
# ẅ ÿ are letters (diaeresis), not tones. Capital ÿ is Ÿ.
AGLC_LETTERS = "ŋŊɛƐɔƆʉɄẅẄÿŸʼ'"
# Anderson: low `, high ´, falling ˆ, rising ˇ, plus mid ¯ and extra-low ̏.
TONE_COMBINING = "\u0300\u0301\u0302\u030c\u0304\u030f"  # ̀ ́ ̂ ̌ ̄ ̏
# Keyboards often emit a spacing accent after the vowel. Same tones, AGLC form.
_SPACING_TONE = {
    "\u0060": "\u0300",  # `
    "\u02cb": "\u0300",  # ˋ
    "\u00b4": "\u0301",  # ´
    "\u02ca": "\u0301",  # ˊ
    "\u02c6": "\u0302",  # ˆ
    "\u02c7": "\u030c",  # ˇ
    "\u00af": "\u0304",  # ¯
    "\u02c9": "\u0304",  # ˉ
}
_NG_APOSTROPHE = str.maketrans({
    "\u0027": "\u02bc",  # ' → ʼ
    "\u2019": "\u02bc",  # ’ → ʼ
    "\u2018": "\u02bc",  # ‘ → ʼ
})

# CID leftovers in Timene 2024 → AGLC. Combining marks attach to the
# previous vowel (they appear *after* it in the PDF stream).
TIMENE_PUA = {
    "\ue001": "\u0301",  # high tone (ntentʉ́, tsɔ́, gwɔ́)
    "\ue002": "\u0301",  # high tone on ʉ / ɔ (nkʉ́a, pɔ́gɔ)
    "\ue003": "ń",
    "\ue004": "ḿ",
    "\ue005": "ú",
    "\ue006": "á",  # l[á]ŋ « col / cou » ; letá
    "\ue007": "í",  # tsítsà
    "\ue008": "\u0301",  # high tone (zɔ́g ntentʉ́)
    "\ue009": "ě",  # manzwě / panzwě
    "\ue00a": "ì",  # nyìŋ
    "\ue00b": "ò",  # ngwòŋ, nò
    "\ue00c": "\u0302",  # kàŋsɛ̂ ≈ kàŋsê
    "\ue00d": "Ŋ",
    "\ue00e": "ǹ",
    "\ue00f": "ó",
    "\ue010": "m̀",
}

PRIVATE = re.compile(r"[\ue000-\uf8ff]")
SPACE = re.compile(r"[ \t\u00a0\u202f]+")
MULTI_NL = re.compile(r"\n+")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

THEORY_NOISE = (
    "cinque",
    "i.e.",
    "i.e ",
    "cf.",
    "op. cit",
    "ibid",
    "et al",
    "thue vold",
    "eva thue",
)

HEADER_NOISE = (
    "texte source transcrit",
    "texte cible",
    "time code",
    "tableau 13",
    "phonèmes",
    "allophones",
    "graphèmes",
    "position initiale",
    "nouveau testament en ngienboum",
    "alliance biblique",
    "la sainte bible",
)

PAIR_FIELDS = ("id", "book", "ref", "fr", "ng")


def make_pair(id_: str, book: str, ref: str, fr: str, ng: str) -> dict:
    """Every corpus line uses the Bible columns only."""
    return {
        "id": id_,
        "book": book,
        "ref": ref,
        "fr": normalize_fr(fr),
        "ng": normalize_ng(ng),
    }


def domain_of(row: dict) -> str:
    """Infer training domain from id / book — never from extra JSON keys."""
    book = str(row.get("book") or "")
    pid = str(row.get("id") or "")
    if book == "HEALTH" or pid.startswith("HEALTH."):
        return "health"
    if pid.startswith("SIL.S.") or book == "NDIOLA" or pid.startswith("NDIOLA."):
        return "grammar"
    if book in {"SIL", "LONFO"} or pid.startswith(("SIL.", "LONFO.")):
        return "lexicon"
    return "bible"


def only_pair_fields(row: dict) -> dict:
    return {key: row[key] for key in PAIR_FIELDS if key in row}

FR_LIKE = re.compile(r"[A-Za-zÀ-ÿ]")
NG_LIKE = re.compile(r"[A-Za-zÀ-ÿŋɛɔʉẅÿŊƐƆ]")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _attach_spacing_tones(text: str) -> str:
    out: list[str] = []
    for ch in text:
        tone = _SPACING_TONE.get(ch)
        if tone and out and unicodedata.category(out[-1])[0] == "L":
            out.append(tone)
        else:
            out.append(ch)
    return "".join(out)


def normalize_fr(text: str) -> str:
    """French accents stay letters: é è ê à are not Ngiemboon tones."""
    return nfc(text.strip())


def normalize_ng(text: str) -> str:
    """AGLC for Ngiemboon.

    ŋ ɛ ɔ ʉ ẅ ÿ stay letters. A tone is always the same combining sign
    after the vowel or the syllabic nasal, whether the source wrote á or
    a + ´ or ɛ́. The diaeresis of ẅ and ÿ is not a tone. The apostrophe
    of the orthography is ʼ.
    """
    text = _attach_spacing_tones(text.strip().translate(_NG_APOSTROPHE))
    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []
    index = 0
    while index < len(decomposed):
        char = decomposed[index]
        index += 1
        if unicodedata.category(char).startswith("M"):
            out.append(char)
            continue
        tones: list[str] = []
        others: list[str] = []
        while index < len(decomposed) and unicodedata.category(decomposed[index]).startswith("M"):
            mark = decomposed[index]
            index += 1
            if mark in TONE_COMBINING:
                tones.append(mark)
            else:
                others.append(mark)
        out.append(nfc(char + "".join(others)))
        for tone in TONE_COMBINING:
            if tone in tones:
                out.append(tone)
    return "".join(out)


def present(text: str, lang: str) -> str:
    """Show Ngiemboon in AGLC and French with its own accents."""
    return normalize_ng(text) if lang == "ng" else normalize_fr(text)


def repair_timene_pua(text: str) -> str:
    """Map Timene private-use CIDs, then drop anything still in the PUA."""
    text = text.translate(str.maketrans(TIMENE_PUA))
    text = PRIVATE.sub("", text)
    return nfc(text)


def strip_controls(text: str) -> str:
    return CONTROL.sub("", text)


def normalize_ws(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = SPACE.sub(" ", text)
    return text.strip()


def clean_text(text: str, *, timene: bool = False) -> str:
    if timene:
        text = repair_timene_pua(text)
    else:
        text = PRIVATE.sub("", text)
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )
    text = normalize_ws(text.replace("\n", " "))
    return nfc(text)


def clean_gloss(text: str) -> str:
    """Shorter lexical items: trim leftover punctuation and quote marks."""
    return clean_text(text).strip(" \t.;:\u00a0()[]«»,")


def strip_tones(text: str) -> str:
    """Remove tone marks only. ŋ ɛ ɔ ʉ ẅ ÿ stay: the diaeresis is not a tone.

    Used for tone-tolerant NG→FR input and training augmentation.
    Ambiguity remains (má ≠ mà).
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if ch not in TONE_COMBINING)
    return nfc(stripped)


def has_tone_marks(text: str) -> bool:
    decomposed = unicodedata.normalize("NFD", text)
    return any(ch in TONE_COMBINING for ch in decomposed)


def looks_like_french(text: str) -> bool:
    if not FR_LIKE.search(text):
        return False
    # AGLC-only letters are rare in French running text.
    special = sum(text.count(ch) for ch in "ŋɛɔʉẅ")
    return special <= 1


def looks_like_ngiemboon(text: str) -> bool:
    if not NG_LIKE.search(text):
        return False
    return True


def is_page_number(text: str) -> bool:
    s = text.strip()
    return s.isdigit() and len(s) <= 4


def is_noise_sentence(text: str) -> bool:
    low = text.casefold()
    if any(tok in low for tok in THEORY_NOISE):
        return True
    if any(tok in low for tok in HEADER_NOISE):
        return True
    if is_page_number(text):
        return True
    return False


def passes_pair_quality(
    fr: str,
    ng: str,
    *,
    min_fr: int = 8,
    min_ng: int = 3,
    require_space: bool = False,
) -> bool:
    if not fr or not ng:
        return False
    if len(fr) < min_fr or len(ng) < min_ng:
        return False
    if is_noise_sentence(fr) or is_noise_sentence(ng):
        return False
    if not looks_like_french(fr):
        return False
    if not looks_like_ngiemboon(ng):
        return False
    if require_space and " " not in ng.strip():
        return False
    if fr.casefold() == ng.casefold():
        return False
    return True
