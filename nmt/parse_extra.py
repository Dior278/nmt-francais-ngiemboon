"""Extra sources: health voice-over, SIL lexicon, Lonfo grammar, Ndiola examples.

Every written pair uses the Bible columns only: id, book, ref, fr, ng.
Ndiola is excluded from default training by id prefix / book, not extra keys.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import fitz

from .normalize import (
    clean_gloss,
    clean_text,
    has_tone_marks,
    is_page_number,
    looks_like_french,
    make_pair,
    passes_pair_quality,
)
from .paths import DATA_DIR

HEALTH_PDF = DATA_DIR / "Timene-2024-voiceover-sante-fr-ngiemboon.pdf"
HEALTH_PDF_FALLBACK = DATA_DIR / "a32c14b012ace2a5.pdf"
ORTHO_PDF = DATA_DIR / "precisdOrthographepourlaLangueNgiemboom.pdf"
LONFO_PDF = DATA_DIR / "Lonfo-apercu-grammatical-ngiemboon.pdf"
LONFO_PDF_FALLBACK = DATA_DIR / "279869e7750a4b49.pdf"
NDIOLA_PDF = DATA_DIR / "MemoireNdiolaTsuata.pdf"

TIMECODE = re.compile(r"^\d{2}:\d{2}")
SPEAKER_FR = re.compile(
    r"^(Journaliste|Docteur|Nadine|Professeur|TCHENTE)\b", re.I
)
SPEAKER_NG = re.compile(
    r"^(Lɔgtà|Lɔ́gtà|Mbìte|Mbite)\b", re.I
)
SPEAKER_STRIP = re.compile(
    r"^(Journaliste|Docteur|Nadine|Professeur|TCHENTE|Lɔgtà|Lɔ́gtà|"
    r"Mbìte mésáʼakɔ|Mbìte mésá'akɔ|Mbìte)\s*:?\s*",
    re.I,
)
GUILLEMET = re.compile(r"«\s*(.+?)\s*»", re.S)
EXAMPLE_NUM = re.compile(r"^\s*\*?\s*(\d+[a-z]?)(?:\s+|\s*$)")
IPA_TOKEN = re.compile(r"^[/\[].+|___+$")
FR_GLOSS = re.compile(r"^[A-Za-zÀ-ÿœŒ''’\- !?,;:«»()/]+$")
DANGLING_FR = re.compile(
    r"\b(d['’]un|d['’]une|dans|souvent|une|les|des|de|qui|par|historiquement)$",
    re.I,
)
NG_WORD = re.compile(
    r"^[A-Za-zÀ-ÿŋŊɛƐɔƆʉɄẅÿʼ'ḿńǹ̌̂̀́]+$",
)
AGLC_CORE = re.compile(r"[ŋŊɛƐɔƆʉɄẅÿʼ]")
SYLLABIC_NASAL = re.compile(r"[ńǹḿ]")
IPA_LEFTOVER = re.compile(r"[̩ː̚ʔ̃˺ɨɯəʁβɖʈɳ]")
INTERLINEAR = re.compile(r"\b(1s|1p|2s|2p|3s|3p|P1|P3|P4|F1|NEG|COMP|PROG|CONS|EX)\b")
PLACE_OR_BIBLIO = re.compile(
    r"\b(yaound[eé]|dschang|bati[eé]|bafoussam|cameroun|"
    r"bamileke|bamiléké|dallas|texas|indiana|k[oö]ln|california|"
    r"angeles|ethnologue|propelca|cabtal|non-publi|"
    r"manuscrit|catholiques|protestants)\b",
    re.I,
)
SHORT_OK_FR = {
    "là",
    "il",
    "tu",
    "oui",
    "peu",
    "nom",
    "vin",
    "lit",
    "pus",
    "sol",
    "art",
    "bal",
    "clé",
    "cou",
    "foi",
    "pas",
    "eau",
    "sel",
    "os",
}

FR_STOP = {
    "le",
    "la",
    "les",
    "de",
    "du",
    "des",
    "et",
    "un",
    "une",
    "à",
    "au",
    "aux",
    "en",
    "par",
    "pour",
    "dans",
    "que",
    "qui",
    "est",
    "sont",
    "il",
    "elle",
    "on",
    "ce",
    "cet",
    "cette",
    "ces",
    "ou",
    "où",
    "ne",
    "pas",
    "plus",
    "très",
    "aussi",
    "après",
    "avant",
    "avec",
    "sans",
    "son",
    "sa",
    "ses",
    "leur",
    "leurs",
    "nous",
    "vous",
    "ils",
    "elles",
    "je",
    "tu",
    "a",
    "ont",
    "été",
    "être",
    "comme",
    "sur",
    "sous",
    "dont",
    "mais",
    "donc",
    "car",
    "si",
    "entre",
    "chez",
    "vers",
    "après",
    "anderson",
    "usage",
    "calendrier",
    "orthographe",
    "acceptable",
    "essayé",
    "début",
    "année",
    "langue",
    "noms",
    "autres",
}

AGLC_HINT = re.compile(r"[ŋŊɛƐɔƆʉɄẅÿʼńǹḿáàâǎéèêěíìóòôǒúù]")

SKIP_GLOSS = {
    "initiale",
    "médiane",
    "mediane",
    "finale",
    "position",
    "graphèmes",
    "graphemes",
    "allophones",
    "phonèmes",
    "phonemes",
    "beaucoup",
    "certains",
    "plusieurs",
    "exemple",
    "exemples",
    "noms",
    "verbes",
    "transcription",
    "phonétique",
    "phonetique",
    "fréquence",
    "frequence",
    "tableau",
    "ci-après",
    "ci-apres",
    "graphème",
    "signe",
    "diacritique",
    "haut",
    "bas",
    "montant",
    "descendant",
    "préfixes",
    "prefixes",
    "grammaticaux",
    "grammaticales",
    "écrites",
    "ecrites",
    "inhérent",
    "inherent",
    "espagnol",
    "portugais",
    "traduction",
    "références",
    "references",
    "volume",
    "journal",
    "doctoral",
    "dissertation",
    "yaoundé",
    "yaounde",
    "dschang",
    "bamileke",
    "bamiléké",
    "cameroun",
    "catholiques",
    "protestants",
    "manuscrit",
    "publié",
    "publie",
    "traits",
    "beaucoup",
    "certains",
    "plusieurs",
    "voyelles",
    "nasalisées",
    "nasalisees",
    "consonnes",
    "labiodentales",
    "palatalisées",
    "palatalisees",
    "ci-après",
    "ci-apres",
    "allophone",
    "phonème",
    "phoneme",
    "graphème",
    "grapheme",
}

# Glosses attested in Lonfo (and SIL) — never invented.
BATCHAM_ATTESTED: list[tuple[str, str]] = [
    ("pɛ̀ Tsɔ́ɔn", "les habitants de Batcham"),
    ("pɛ̀ Leshʉ́ŋ", "les habitants de Balessing"),
]


def _resolve(primary: Path, fallback: Path | None = None) -> Path | None:
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return None


def _broken_eng(text: str) -> bool:
    """Lonfo PUA often deletes ɛ/ɔ next to ŋ (mbŋte, mekŋ)."""
    vowels = set("aeiouɛɔʉAEIOUƐƆáàâǎéèêěíìóòôǒúùẃẅÿʼ'")
    for i, ch in enumerate(text):
        if ch not in "ŋŊ":
            continue
        j = i - 1
        while j >= 0 and unicodedata.category(text[j]) == "Mn":
            j -= 1
        prev = text[j] if j >= 0 else ""
        if prev and prev not in vowels and prev.lower() not in "nʼ'":
            return True
    return False


FR_ACCENTED = {
    "père",
    "mère",
    "après",
    "très",
    "être",
    "été",
    "côté",
    "déjà",
    "là",
    "dès",
    "près",
    "même",
    "frère",
    "forêt",
    "étrange",
    "étranger",
    "étrangers",
    "écureuil",
    "espèce",
    "éléphant",
    "évier",
    "également",
    "matériels",
    "premièrement",
    "deuxièmement",
    "troisièmement",
    "médiane",
    "qualité",
    "profondeur",
    "nombreux",
    "sauterelle",
    "hérnie",
    "hernie",
    "âme",
    "ame",
    "cœur",
    "coeur",
    "voilà",
    "voila",
    "côté",
    "cote",
    "déjà",
    "deja",
    "après",
    "apres",
    "nouvellement",
    "mariée",
    "mariee",
}

GLOSS_HEAD = {
    "très",
    "tres",
    "être",
    "etre",
    "petit",
    "petits",
    "quand",
    "faire",
    "mettre",
    "avoir",
    "nouvellement",
    "petits",
    "petit",
}
GLOSS_TAIL = {
    "fort",
    "fixe",
    "malade",
    "même",
    "meme",
    "dessous",
    "chose",
    "habit",
    "mariée",
    "mariee",
    "sauvage",
    "pierre",
    "tracée",
    "tracee",
    "amer",
}

HAS_TONE_CHAR = re.compile(r"[áàâǎéèêěíìóòôǒúùńǹḿ]")
AGLC_SPECIAL = re.compile(r"[ŋŊɛƐɔƆʉɄẅÿʼńǹḿ]")
# Marks that almost never appear in a French gloss token.
AGLC_ONLY_MARK = re.compile(r"[ŋŊɛƐɔƆʉɄẅÿʼǎǐǔǒěńǹḿ]")
TABLE_HEADERS = {
    "position",
    "initiale",
    "médiane",
    "mediane",
    "finale",
    "graphèmes",
    "graphemes",
    "médiane7",
    "mediane7",
}
TABLE_COL_W = 52.0


def _ng_cell(text: str) -> bool:
    text = clean_gloss(text)
    low = text.casefold()
    if not text or low in FR_STOP or low in SKIP_GLOSS or low in FR_ACCENTED:
        return False
    if IPA_TOKEN.match(text) or text.startswith("___"):
        return False
    if IPA_LEFTOVER.search(text):
        return False
    if PLACE_OR_BIBLIO.search(text):
        return False
    if AGLC_CORE.search(text) or SYLLABIC_NASAL.search(text):
        return True
    if has_tone_marks(text) and 1 < len(text) <= 16:
        if looks_like_french(text) and not AGLC_CORE.search(text) and not SYLLABIC_NASAL.search(text):
            if len(text) > 3 or "'" in text or "’" in text or "-" in text:
                return False
        return True
    return False


def _fr_cell(text: str) -> bool:
    text = clean_gloss(text)
    if not text or text.casefold() in SKIP_GLOSS:
        return False
    if AGLC_CORE.search(text) or SYLLABIC_NASAL.search(text) or AGLC_ONLY_MARK.search(text):
        return False
    if PLACE_OR_BIBLIO.search(text):
        return False
    if INTERLINEAR.search(text):
        return False
    if has_tone_marks(text) and text.casefold() not in FR_ACCENTED:
        if AGLC_ONLY_MARK.search(text) or re.search(r"[ǐǔǒěńǹḿúìò]", text):
            return False
    if len(text) < 3 and text.casefold() not in SHORT_OK_FR:
        return False
    if 2 <= len(text) <= 80 and FR_GLOSS.match(text) and looks_like_french(text):
        return True
    return False


def _is_real_ng(text: str) -> bool:
    if PLACE_OR_BIBLIO.search(text):
        return False
    if not re.search(r"[aeiouɛɔʉáàâéèíìóòúùAEIOUƐƆ]", text):
        return False
    if text.casefold() in FR_STOP or text.casefold() in SKIP_GLOSS:
        return False
    if AGLC_CORE.search(text) or SYLLABIC_NASAL.search(text):
        return True
    if has_tone_marks(text) and not looks_like_french(text):
        return True
    return False


def _strip_speaker(text: str) -> str:
    text = re.sub(r"Lɔ\s+gtà", "Lɔgtà", text)
    text = re.sub(r"Lɔ́\s+gtà", "Lɔ́gtà", text)
    return SPEAKER_STRIP.sub("", text).strip(" :")


def _group_words_by_y(words: list[tuple[float, float, str]], tol: float = 8.0) -> list[str]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (w[1], w[0]))
    lines: list[list[tuple[float, float, str]]] = []
    for item in words:
        if lines and abs(item[1] - lines[-1][0][1]) <= tol:
            lines[-1].append(item)
        else:
            lines.append([item])
    out = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        out.append(" ".join(w[2] for w in line))
    return out


# ---------------------------------------------------------------------------
# Timene 2024 — two-column ACTU SANTÉ transcript
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def extract_health_pairs() -> list[dict]:
    """French left (x≈223–442), Ngiemboon right (x≥440). Pair by timecode bands."""
    path = _resolve(HEALTH_PDF, HEALTH_PDF_FALLBACK)
    if path is None:
        return []

    doc = fitz.open(path)
    # Tableau 13 starts on printed p.66 ≈ index 76; two-col layout to ~123.
    start, end = 76, min(124, doc.page_count)

    # (page, y) timecode anchors, then words tagged with (page, y).
    anchors: list[tuple[int, float]] = []
    fr_words: list[tuple[int, float, float, str]] = []
    ng_words: list[tuple[int, float, float, str]] = []

    for index in range(start, end):
        page = doc[index]
        for x0, y0, _x1, _y1, raw, *_ in page.get_text("words"):
            blob = raw.strip()
            if not blob or is_page_number(blob):
                continue
            if x0 < 215:
                if TIMECODE.match(blob):
                    anchors.append((index, y0))
                continue
            text = clean_text(blob, timene=True)
            if not text:
                continue
            if 215 <= x0 < 442:
                fr_words.append((index, y0, x0, text))
            elif x0 >= 440:
                ng_words.append((index, y0, x0, text))
    doc.close()

    if not fr_words or not ng_words:
        return []

    # Cluster near-duplicate in/out timecodes on the same row.
    clustered: list[tuple[int, float]] = []
    for page, y in sorted(anchors):
        if clustered and clustered[-1][0] == page and abs(y - clustered[-1][1]) < 55:
            continue
        clustered.append((page, y))
    if not clustered:
        clustered = [(start, 0.0)]
    clustered.append((end, 10_000.0))

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for i in range(len(clustered) - 1):
        p0, y0 = clustered[i]
        p1, y1 = clustered[i + 1]
        fr = _words_in_band(fr_words, p0, y0, p1, y1)
        ng = _words_in_band(ng_words, p0, y0, p1, y1)
        fr = _fix_health_wraps(_strip_speaker(fr))
        ng = _fix_health_wraps(_strip_speaker(ng))
        if not passes_pair_quality(fr, ng, min_fr=22, min_ng=16, require_space=True):
            continue
        if not _health_fr_ok(fr):
            continue
        key = (fr.casefold(), ng.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            make_pair(
                f"HEALTH.{p0 + 1}.{i + 1}",
                "HEALTH",
                f"ACTU SANTÉ p.{p0 + 1} #{i + 1}",
                fr,
                ng,
            )
        )
    return pairs


def _words_in_band(
    words: list[tuple[int, float, float, str]],
    p0: int,
    y0: float,
    p1: int,
    y1: float,
) -> str:
    picked: list[tuple[float, float, str]] = []
    for page, y, x, text in words:
        if page < p0 or page > p1:
            continue
        if page == p0 and y < y0 - 4:
            continue
        if page == p1 and y >= y1 - 4:
            continue
        picked.append((x, y + page * 2000.0, text))
    return clean_text(" ".join(_group_words_by_y(picked)), timene=False)


def _fix_health_wraps(text: str) -> str:
    """Column wraps that split a single AGLC word across two lines."""
    text = re.sub(r"zɔ́\s+g(?=\s|$|[.,?])", "zɔ́g", text)
    text = re.sub(r"\bg\s+zɔ́(?=\s|$|[.,?])", "zɔ́g", text)
    text = re.sub(r"\s+:\s+", " ", text)
    text = re.sub(r"nkʉ\s+a\b", "nkʉa", text)
    text = re.sub(r"Lɔ\s+gtà", "Lɔgtà", text)
    text = re.sub(r"Lɔ́\s+gtà", "Lɔ́gtà", text)
    text = re.sub(r"wɔ́\s+ɔn\b", "wɔ́ɔn", text)
    text = re.sub(r"pɔ́\s+gɔ\b", "pɔ́gɔ", text)
    text = re.sub(r"mashʉ́\s+ŋ\b", "mashʉ́ŋ", text)
    return text


def _health_fr_ok(fr: str) -> bool:
    low = fr.casefold()
    if "texte source" in low or "texte cible" in low:
        return False
    # Real turns mention the show, the body, or a speaker leftover already stripped.
    if SPEAKER_FR.search(fr):
        return True
    keys = (
        "cancer",
        "utérus",
        "uterus",
        "vaccin",
        "virus",
        "femme",
        "hôpital",
        "hopital",
        "docteur",
        "bonsoir",
        "mesdames",
        "equinoxe",
        "santé",
        "sante",
        "dépistage",
        "depistage",
        "col de",
        "papillome",
        "invitation",
        "téléspect",
        "telespect",
    )
    return any(k in low for k in keys) or len(fr) >= 40


# ---------------------------------------------------------------------------
# Anderson 2007 SIL — official AGLC word/gloss tables + attested sentences
# Pages 1–3 / 16 / 25–28 are title, abbreviations, prose, bibliography, TOC.
# ---------------------------------------------------------------------------

# Skip bibliography / TOC / title: place names there are not translations.
SIL_SKIP_PAGES = {0, 1, 2, 15, 24, 25, 26, 27}
SIL_TABLE_PAGES = range(3, 15)  # printed 4–15
SIL_EXAMPLE_PAGES = range(16, 24)  # printed 17–24
TABLE_X_MIN = 265.0  # left of this: phonemes / allophones / graphemes

# Full-sentence examples copied from Anderson 2007 (pages 20, 23, 24).
SIL_SENTENCES: list[tuple[str, str, int]] = [
    ("Pi zsě ngie: Mèŋ n ge tá gẅi.", "Vous savez que je suis votre père.", 20),
    ("!Pi zsé ngie: Mèŋ n ge tá gẅi!", "Sachez que je suis votre père !", 20),
    ("A ka pí pùʼ.", "Il a pris un couteau (hier).", 20),
    ("A ka maʼa pùʼ.", "Il a lancé un couteau (hier).", 20),
    ("A pǐ pùʼ.", "Il a pris un couteau (tout à l'heure).", 20),
    ("A maʼá pùʼ.", "Il a lancé un couteau (tout à l'heure).", 20),
    ("Pege ka sẅiŋe kúnɔ̀ɔn wège.", "Nous avons vendu notre cochon (hier).", 23),
    ("Pi ka sẅiŋe kúnɔ̀ɔn gẅi?", "Avez-vous vendu votre cochon (hier) ?", 23),
    ("Náa jʉʼɔɔn!", "Donne-la moi !", 23),
    ("!Pi pfɛ́ shúm sí!", "Que vous mangez votre nourriture !", 23),
    (
        "Múɔ kaa túʼ ntse wɔ́, ńtí nue à ne ńguɔ.",
        "L'enfant n'a pas puisé de l'eau, parce qu'il est malade.",
        23,
    ),
    (
        "Mèŋ ge metʉ̌a, ńjú shúm páʼ sésɔg, ngwáŋ, mevɛ́, messó.",
        "Moi, j'irai au marché acheter des choses comme le piment, le sel, l'huile, les poissons.",
        23,
    ),
    (
        "À gɔ̌ɔn ne mèŋ ngie: «Tóo ḿpfɛ́ɛ péli.»",
        "Il m'a dit : « Viens manger de l'arachide. »",
        24,
    ),
    ("À la pfɛ́ɛ mmó.", "Il a mangé la nourriture.", 24),
    (
        "Tâm tumó Soŋnkwɛ̌ ngie à gʉa Mbùʼndɔɔn.",
        "Tam a demandé à Sonkwe qu'il parte lui faire une commission à Mbouda.",
        24,
    ),
    ("Á zɛ́te wɔ́ɔn ngie: Ɔ gʉa nzɔ̌?", "Il me demanda : Où vas-tu ?", 24),
]


def _is_placeholder(text: str) -> bool:
    t = text.strip()
    return t in {"___", "—", "–", "…"} or (len(t) >= 3 and set(t) <= {"_"})


def _is_clear_ng(text: str) -> bool:
    """Vernacular token in a SIL example column (not a French gloss)."""
    text = clean_gloss(text)
    if not text or _is_placeholder(text):
        return False
    if IPA_TOKEN.match(text) or text.startswith("___"):
        return False
    low = text.casefold()
    if low in TABLE_HEADERS or low in FR_STOP or low in SKIP_GLOSS or low in FR_ACCENTED:
        return False
    if AGLC_CORE.search(text) or SYLLABIC_NASAL.search(text) or AGLC_ONLY_MARK.search(text):
        return True
    if has_tone_marks(text) and re.search(r"[ǐǔǒěńǹḿúìò]", text):
        return True
    return False


def _is_inferred_ng(text: str) -> bool:
    """Short Latin-only vernacular sitting on an NG / ___ row (mboo, ha, gem)."""
    text = clean_gloss(text)
    if not (2 <= len(text) <= 12):
        return False
    if _is_clear_ng(text) or _is_placeholder(text):
        return True
    low = text.casefold()
    if low in TABLE_HEADERS or low in FR_STOP or low in SKIP_GLOSS or low in FR_ACCENTED:
        return False
    if _fr_cell(text):
        return False
    if IPA_TOKEN.match(text) or PLACE_OR_BIBLIO.search(text) or IPA_LEFTOVER.search(text):
        return False
    if not re.search(r"[aeiouɛɔʉAEIOUƐƆáàâǎéèêěíìóòôǒúùǔǐ]", text):
        return False
    return bool(NG_WORD.match(text))


def _ng_table_cell(text: str) -> bool:
    """Table example columns: allow short vernacular without ŋ/ɛ if not French."""
    if _is_clear_ng(text) or _ng_cell(text):
        return True
    return _is_inferred_ng(text) and not _fr_cell(clean_gloss(text))


def _same_table_col(x_a: float, x_b: float) -> bool:
    return abs(x_a - x_b) < TABLE_COL_W


def _first_gloss(fr: str) -> str:
    """Keep the first French gloss; stop if a vernacular token leaked in."""
    fr = fr.strip()
    if fr.startswith("!") and _fr_cell(fr.lstrip("! ").strip()):
        fr = fr.lstrip("! ").strip()
    toks = fr.split()
    if not toks:
        return fr
    kept: list[str] = [toks[0]]
    for tok in toks[1:]:
        bare = tok.strip("()[]«»!,.?")
        if tok in {"!", "?"} or tok.startswith("(") or tok.endswith(")"):
            kept.append(tok)
            continue
        if re.match(r"^(nz|nk|ng|mb|nt|ŋ|ḿ|ń)", bare.casefold()) and (
            has_tone_marks(bare) or AGLC_ONLY_MARK.search(bare)
        ):
            break
        if _fr_cell(bare) or bare.casefold() in FR_ACCENTED:
            kept.append(tok)
            continue
        if _is_clear_ng(bare) or AGLC_ONLY_MARK.search(bare):
            break
        if "'" in bare or "’" in bare or "ʼ" in bare:
            folded = bare.casefold().replace("’", "'").replace("ʼ", "'")
            if not re.match(r"^[ldncdjtsm]'", folded):
                break
        if has_tone_marks(bare) and bare.casefold() not in FR_ACCENTED:
            break
        if NG_WORD.match(bare) and not _fr_cell(bare) and not looks_like_french(bare):
            break
        kept.append(tok)
    return " ".join(kept)


def _connected_gloss(fr: str) -> bool:
    """True if a multi-word French string is one gloss, not two table cells."""
    if any(ch in fr for ch in ",!?("):
        return True
    low = f" {fr.casefold()} "
    if any(
        cue in low
        for cue in (
            " d'",
            " d’",
            " de ",
            " du ",
            " des ",
            " un ",
            " une ",
            " qui ",
            " à ",
            " au ",
            " aux ",
            " en ",
            " pour ",
            " comme ",
        )
    ):
        return True
    toks = fr.split()
    if len(toks) <= 1:
        return True
    if toks[0].casefold() in GLOSS_HEAD or toks[-1].casefold() in GLOSS_TAIL:
        return True
    return False


def _collect_words(page) -> list[tuple[float, float, str]]:
    words = []
    for item in page.get_text("words"):
        x0, y0, _x1, _y1, raw = item[:5]
        raw = raw.strip()
        if not raw or is_page_number(raw) or y0 > 782:
            continue
        words.append((x0, y0, raw))
    return words


@lru_cache(maxsize=1)
def extract_sil_pairs() -> list[dict]:
    if not ORTHO_PDF.exists():
        return []
    doc = fitz.open(ORTHO_PDF)
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    idx = 0
    sent_idx = 0

    def add(ng: str, fr: str, page: int, kind: str, *, sentence: bool = False) -> None:
        nonlocal idx, sent_idx
        if sentence:
            ng, fr = clean_text(ng), clean_text(fr)
        else:
            if ng.startswith("[") or ng.startswith("/") or fr.startswith("["):
                return
            ng, fr = clean_gloss(ng), clean_gloss(fr)
            fr = fr.strip(" []«»")
            if fr.count("(") == fr.count(")") + 1:
                fr = fr + ")"
            fr = fr.strip()
        fr = re.sub(r"\s+", " ", fr)
        if not sentence:
            fr = _first_gloss(fr)
            if not _connected_gloss(fr):
                return
        if not ng or not fr:
            return
        if PLACE_OR_BIBLIO.search(ng):
            return
        if not sentence and PLACE_OR_BIBLIO.search(fr):
            return
        if IPA_TOKEN.match(ng) or IPA_TOKEN.match(fr) or IPA_LEFTOVER.search(ng):
            return
        if "/" in ng or ng.endswith("="):
            return
        if not sentence and DANGLING_FR.search(fr):
            return
        if INTERLINEAR.search(fr) or INTERLINEAR.search(ng):
            return
        if fr.casefold() in SKIP_GLOSS or fr.casefold() in FR_STOP:
            return
        if ng.casefold() in FR_STOP or ng.casefold() in SKIP_GLOSS:
            return
        if not sentence:
            if fr.count(" ") > 6:
                return
            if not _ng_table_cell(ng) and not _is_real_ng(ng):
                return
            if not _fr_cell(fr) and not (3 <= len(fr) <= 80 and looks_like_french(fr) and not AGLC_CORE.search(fr)):
                return
            if looks_like_french(ng) and not AGLC_CORE.search(ng) and not SYLLABIC_NASAL.search(ng):
                if not HAS_TONE_CHAR.search(ng) or len(ng) > 4 or "'" in ng or "’" in ng:
                    return
            if not passes_pair_quality(fr, ng, min_fr=2, min_ng=2):
                return
        else:
            if not passes_pair_quality(fr, ng, min_fr=10, min_ng=6, require_space=True):
                return
        key = (fr.casefold(), ng.casefold())
        if key in seen:
            return
        seen.add(key)
        if sentence:
            sent_idx += 1
            pairs.append(
                make_pair(
                    f"SIL.S.{sent_idx}",
                    "SIL",
                    f"Anderson 2007 p.{page} phrase",
                    fr,
                    ng,
                )
            )
            return
        idx += 1
        pairs.append(
            make_pair(
                f"SIL.{idx}",
                "SIL",
                f"Anderson 2007 p.{page} {kind}",
                fr,
                ng,
            )
        )

    for page_i, page in enumerate(doc):
        if page_i in SIL_SKIP_PAGES:
            continue
        words = _collect_words(page)
        if page_i in SIL_TABLE_PAGES:
            _extract_sil_table(words, page_i, add)
        if page_i in SIL_EXAMPLE_PAGES:
            _extract_sil_minpairs(words, page_i, add)
            # Pages 20, 23, 24: running text + interlinear. Skip row pairing.
            if page_i not in {19, 22, 23}:
                _extract_sil_row_glosses(words, page_i, add)

    doc.close()
    for ng, fr, page in SIL_SENTENCES:
        add(ng, fr, page, "phrase", sentence=True)
    for ng, fr, page in (
        ("nzwěngʉ̀", "femme nouvellement mariée", 21),
        ("kéyàʼ", "chant d'introduction aux funérailles", 5),
        ("fwɔ̀ʼɔ", "celui qui retire quelque chose d'un trou", 8),
        ("lèŋ", "espèce d'écureuil", 17),
        ("sǒ", "ami", 11),
        ("éfó", "venir de", 11),
        ("ssó", "poisson", 11),
        ("ffó", "pus", 11),
        ("nzwě ngʉ̀", "femme d'un visiteur", 21),
        ("ngʉ̀", "visiteur", 21),
    ):
        add(ng, fr, page, "table")
    kept = [p for p in pairs if _keep_sil_pair(p)]
    sentences = [p for p in kept if p["id"].startswith("SIL.S.")]
    lexicon_rows = [p for p in kept if not p["id"].startswith("SIL.S.")]
    best: dict[str, dict] = {}
    for p in lexicon_rows:
        ng = re.sub(r"^\(?[ae]\)", "", p["ng"]).strip("() ")
        if not ng:
            continue
        p = make_pair(p["id"], p["book"], p["ref"], p["fr"], ng)
        key = ng.casefold()
        if key not in best:
            best[key] = p
            continue
        old_fr = best[key]["fr"].casefold()
        new_fr = p["fr"].casefold()
        if new_fr.startswith(old_fr) and len(new_fr) > len(old_fr):
            continue
        if old_fr.startswith(new_fr) and len(old_fr) > len(new_fr):
            best[key] = p
            continue
        if len(p["fr"]) > len(best[key]["fr"]):
            best[key] = p
    lexicon_rows = list(best.values())
    by_fr: dict[str, list[dict]] = {}
    for p in lexicon_rows:
        by_fr.setdefault(p["fr"].casefold(), []).append(p)
    drop: set[str] = set()
    for group in by_fr.values():
        ngs = [p["ng"] for p in group]
        for p in group:
            if any(other != p["ng"] and other.endswith(p["ng"]) and len(other) > len(p["ng"]) for other in ngs):
                drop.add(p["id"])
    lexicon_rows = [p for p in lexicon_rows if p["id"] not in drop]
    lexicon_rows = [
        p
        for p in lexicon_rows
        if not (p["fr"].casefold().startswith("entre dans") and "kú" not in p["ng"])
    ]
    return sentences + lexicon_rows


def _keep_sil_pair(row: dict) -> bool:
    if row["id"].startswith("SIL.S."):
        return True
    ng = row["ng"]
    if looks_like_french(ng) and not AGLC_CORE.search(ng) and not SYLLABIC_NASAL.search(ng):
        if len(ng) > 4 or "'" in ng or "’" in ng:
            return False
    if INTERLINEAR.search(row["fr"]) or PLACE_OR_BIBLIO.search(ng):
        return False
    if re.search(r"historiquement|palato-labial", row["fr"], re.I):
        return False
    if "visiteur" in row["fr"].casefold() and "femme" in row["fr"].casefold() and "nzwě" not in row["ng"]:
        return False
    if any(AGLC_ONLY_MARK.search(tok) for tok in row["fr"].split()):
        return False
    if any(tok.casefold() in SKIP_GLOSS or tok.casefold() in TABLE_HEADERS for tok in row["fr"].split()):
        return False
    return True


def _extract_sil_table(words, page_i, add) -> None:
    """One stacked example column at a time: NG, then its French, stop at next NG."""
    header_y = max(
        (y for _x, y, t in words if t.casefold() in TABLE_HEADERS),
        default=None,
    )
    if header_y is None:
        return
    example = [
        (x, y, t)
        for x, y, t in words
        if x >= TABLE_X_MIN and y > header_y + 4
    ]
    if not example:
        return

    bands: dict[int, list[tuple[float, float, str]]] = {}
    for x, y, t in example:
        bands.setdefault(int(round(y / 6.0) * 6), []).append((x, y, t))

    ngs: list[tuple[float, float, str]] = []
    for row in bands.values():
        has_mark = any(_is_placeholder(t) or _is_clear_ng(t) for _x, _y, t in row)
        for x, y, t in row:
            if t.casefold() in TABLE_HEADERS or t in {"!", "?"}:
                continue
            if _is_placeholder(t):
                continue
            # A ___ / clear-NG row is vernacular only — never read those tokens as French.
            if _is_clear_ng(t) or (has_mark and NG_WORD.match(clean_gloss(t))):
                ngs.append((x, y, t))

    placeholders = [(x, y) for x, y, t in example if _is_placeholder(t)]
    for x0, y0, raw in ngs:
        next_ys = [y for x, y, _t in ngs if _same_table_col(x, x0) and y > y0 + 6]
        next_ys += [y for x, y in placeholders if _same_table_col(x, x0) and y > y0 + 6]
        limit = min(next_ys) if next_ys else y0 + 48
        bits = []
        for x, y, t in example:
            if y <= y0 + 5 or y >= limit:
                continue
            if x < x0 - 10:
                continue
            if not _same_table_col(x, x0):
                continue
            if _is_placeholder(t) or _is_clear_ng(t) or (x, y, t) in ngs:
                continue
            if _fr_cell(t) or t in {"!", "?"} or t.startswith("(") or t.endswith(")"):
                bits.append((y, x, t))
        if not bits:
            continue
        bits.sort()
        fr = _first_gloss(" ".join(t for _y, _x, t in bits))
        add(raw, fr, page_i + 1, "table")


def _extract_sil_minpairs(words, page_i, add) -> None:
    """má [má] mère — and the right-hand pair on the same row."""
    lines: dict[int, list[tuple[float, str]]] = {}
    for x0, y0, raw in words:
        lines.setdefault(int(round(y0 / 7.0) * 7), []).append((x0, raw))
    for _y, row in lines.items():
        row.sort()
        toks = [t for _, t in row]
        i = 0
        while i < len(toks) - 2:
            tok = toks[i]
            nxt = toks[i + 1]
            if nxt.startswith("[") and (_ng_table_cell(tok) or _ng_cell(tok)):
                if i > 0 and (_ng_table_cell(toks[i - 1]) or _ng_cell(toks[i - 1])):
                    prev = toks[i - 1]
                    if not looks_like_french(prev) and len(prev) <= 12:
                        tok = f"{prev} {tok}"
                j = i + 1
                while j < len(toks) and not toks[j].endswith("]") and toks[j].startswith("["):
                    j += 1
                if j < len(toks) and "]" in toks[j]:
                    j += 1
                fr_bits = []
                while j < len(toks) and not toks[j].startswith("["):
                    if _ng_table_cell(toks[j]) or _ng_cell(toks[j]):
                        break
                    if _fr_cell(toks[j]) or toks[j] in {"!", "?"}:
                        fr_bits.append(toks[j])
                    j += 1
                if fr_bits:
                    add(tok, _first_gloss(" ".join(fr_bits)), page_i + 1, "minpair")
                i = j
                continue
            i += 1


def _extract_sil_row_glosses(words, page_i, add) -> None:
    """Same-row NG + French (class table, compounds, small contrast boxes)."""
    lines: dict[int, list[tuple[float, str]]] = {}
    for x0, y0, raw in words:
        if y0 > 770:
            continue
        lines.setdefault(int(round(y0 / 7.0) * 7), []).append((x0, raw))
    for _y, row in lines.items():
        row.sort()
        if len(row) > 8:
            continue
        joined = " ".join(t for _, t in row).casefold()
        if any(
            tok in joined
            for tok in (
                "ngiemboon",
                "consonne",
                "voyelle",
                "phonème",
                "allophone",
                "orthographe",
                "préfixe",
                "décision",
                "écriture",
                "c’est",
                "c'est",
            )
        ):
            continue
        i = 0
        while i < len(row) - 1:
            x0, tok = row[i]
            if not (_ng_table_cell(tok) or _ng_cell(tok)):
                i += 1
                continue
            # Skip minpair rows (handled elsewhere).
            if i + 1 < len(row) and row[i + 1][1].startswith("["):
                i += 1
                continue
            fr_bits = []
            j = i + 1
            while j < len(row):
                x1, t1 = row[j]
                if t1.startswith("["):
                    break
                if _ng_table_cell(t1) or _ng_cell(t1):
                    if fr_bits:
                        break
                    # compound like nzwě ngʉ̀
                    if x1 - x0 < 50:
                        tok = f"{tok} {t1}"
                        j += 1
                        continue
                    break
                if (_fr_cell(t1) or t1 in {"!", "?"}) and x1 - x0 < TABLE_COL_W + 10:
                    fr_bits.append(t1)
                    j += 1
                    continue
                break
            if fr_bits:
                add(tok, _first_gloss(" ".join(fr_bits)), page_i + 1, "row")
            i = max(j, i + 1)


# ---------------------------------------------------------------------------
# Lonfo grammatical sketch — glosses and short examples
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def extract_lonfo_pairs() -> list[dict]:
    path = _resolve(LONFO_PDF, LONFO_PDF_FALLBACK)
    if path is None:
        return []
    doc = fitz.open(path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    text = PRIVATE_DROP.sub("", text)

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    idx = 0

    def add(ng: str, fr: str, domain: str) -> None:
        nonlocal idx
        ng, fr = clean_text(ng), clean_text(fr)
        if looks_like_french(ng) and not _is_real_ng(ng):
            return
        if not _is_real_ng(ng) or len(ng) < 3:
            return
        if fr.casefold() in FR_STOP or fr.casefold() in SKIP_GLOSS:
            return
        if "(cl" in fr.casefold() or fr.endswith("("):
            return
        if domain == "lexicon" and looks_like_french(ng) and not re.search(r"[ŋɛɔʉẅÿńḿʼ]", ng):
            return
        if _broken_eng(ng):
            return
        if not passes_pair_quality(
            fr,
            ng,
            min_fr=3 if domain == "lexicon" else 12,
            min_ng=2,
            require_space=domain != "lexicon",
        ):
            return
        if domain == "lexicon" and not FR_GLOSS.match(fr):
            return
        key = (fr.casefold(), ng.casefold())
        if key in seen:
            return
        seen.add(key)
        idx += 1
        pairs.append(make_pair(f"LONFO.{idx}", "LONFO", f"Lonfo #{idx}", fr, ng))

    # Word = gloss, often joined without a colon.
    gloss_pat = re.compile(
        r"([A-Za-zÀ-ÿŋŊɛƐɔƆʉɄẅÿʼ'ḿńǹ]+(?:\s+[A-Za-zÀ-ÿŋŊɛƐɔƆʉɄẅÿʼ'ḿńǹ]+)?)"
        r"[\s:]+"
        r"(les habitants de [A-Za-zÀ-ÿ\- ]+|mère|père|chefferie|oreilles|"
        r"écureuil|ecureuil|étranger|etranger|étrangers|homme|friandise|"
        r"dernier|rhume|sacrificateur|sacrificateurs|coupeurs?|panthère|"
        r"livre|lune|enfants|maison|village|couscous)",
        re.I,
    )
    for m in gloss_pat.finditer(text):
        add(m.group(1), m.group(2), "lexicon")

    for ng, fr in BATCHAM_ATTESTED:
        add(ng, fr, "lexicon")

    return pairs


PRIVATE_DROP = re.compile(r"[\ue000-\uf8ff]")


# ---------------------------------------------------------------------------
# Ndiola 2011 — numbered examples, adapted orthography
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def extract_ndiola_pairs() -> list[dict]:
    """Examples with French in « guillemets ». Orthography is often adapted."""
    if not NDIOLA_PDF.exists():
        return []
    doc = fitz.open(NDIOLA_PDF)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for i, m in enumerate(GUILLEMET.finditer(text), start=1):
        fr = clean_text(m.group(1))
        if len(fr) < 18:
            continue
        if "Cinque" in fr or fr.casefold().startswith("i.e"):
            continue
        if re.search(r"\b(mode|évidentiel|évaluatif|épistémique)\b", fr, re.I):
            continue
        start = max(0, m.start() - 420)
        window = text[start : m.start()]
        if re.search(r"Cinque\s*\(\s*1999", window):
            continue
        if "«" in window[-80:] and window.count("«") > window.count("»"):
            continue
        lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
        if not lines:
            continue
        # Starred line nearby → ungrammatical, drop.
        if any(ln.lstrip().startswith("*") for ln in lines[-6:]):
            continue
        ng = _ndiola_source_line(lines)
        ng = clean_text(ng)
        if ng.lstrip().startswith("*") or re.search(r"\*\s", ng):
            continue
        if "«" in ng or "»" in ng:
            continue
        if re.search(r"\b(mode|évidentiel|évaluatif|épistémique|morphème)\b", ng, re.I):
            continue
        if not passes_pair_quality(fr, ng, min_fr=18, min_ng=10, require_space=True):
            continue
        # Adapted theses often mix French gloss words into the "source" line.
        if _mostly_french_tokens(ng):
            continue
        if not re.search(
            r"matuwo|djú|tèé|ngwe|nduè|róò|tokò|meessegne|dué|igné|kuno",
            ng,
            re.I,
        ) and not re.search(r"\b(á|a)\b.*\b(ro|róò|djú)\b", ng, re.I):
            continue
        key = (fr.casefold(), ng.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(make_pair(f"NDIOLA.{i}", "NDIOLA", f"Ndiola 2011 ex.{i}", fr, ng))
    return pairs


def _ndiola_source_line(lines: list[str]) -> str:
    """Prefer the numbered vernacular line over the word-by-word gloss."""
    for ln in reversed(lines):
        if EXAMPLE_NUM.match(ln):
            return EXAMPLE_NUM.sub("", ln).strip()
    # Fall back: last line that does not look like a French gloss row.
    for ln in reversed(lines):
        low = ln.casefold()
        if any(tok in low for tok in ("ms ", "fut ", "acheter", "voiture", "jean")):
            continue
        if looks_like_french(ln) and not re.search(r"[ŋɛɔʉẅńḿ]|tèé|djú|matuwo|ngwe|nduè", ln):
            continue
        return ln
    return lines[-1] if lines else ""


def _mostly_french_tokens(text: str) -> bool:
    toks = [t for t in re.split(r"\s+", text) if t]
    if len(toks) < 3:
        return True
    fr_hits = sum(1 for t in toks if FR_GLOSS.match(t) and not re.search(r"[ŋɛɔʉẅńḿʼ']", t))
    return fr_hits / len(toks) > 0.65


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lexicon_pairs() -> list[dict]:
    rows = [p for p in extract_sil_pairs() if not p["id"].startswith("SIL.S.")]
    rows += extract_lonfo_pairs()
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in rows:
        key = (row["fr"].casefold(), row["ng"].casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def extra_sentence_pairs(include_adapted_ortho: bool = False) -> list[dict]:
    health = extract_health_pairs()
    sil_sent = [p for p in extract_sil_pairs() if p["id"].startswith("SIL.S.")]
    extra = health + sil_sent
    if include_adapted_ortho:
        extra = extra + extract_ndiola_pairs()
    return extra


def all_extra_pairs(include_adapted_ortho: bool = False) -> list[dict]:
    return extra_sentence_pairs(include_adapted_ortho) + lexicon_pairs()
