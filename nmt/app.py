"""French / English ↔ Ngiemboon window. Batcham, Bamboutos.

    python -m nmt app

Never trains. If there is no checkpoint, the window says to run training.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk

from .paths import CHECKPOINTS, PROCESSED_DIR, ROOT

try:
    import customtkinter as ctk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "customtkinter is missing. From the project folder:  pip install -r nmt/requirements.txt"
    ) from exc

from .normalize import domain_of, normalize_fr, normalize_ng
from .pivot import DIRECTIONS, PivotError, translate_between

AGLC_LETTERS = [
    ("ŋ", "ŋ"),
    ("Ŋ", "Ŋ"),
    ("ɛ", "ɛ"),
    ("Ɛ", "Ɛ"),
    ("ɔ", "ɔ"),
    ("Ɔ", "Ɔ"),
    ("ʉ", "ʉ"),
    ("Ʉ", "Ʉ"),
    ("ẅ", "ẅ"),
    ("Ẅ", "Ẅ"),
    ("ÿ", "ÿ"),
    ("Ÿ", "Ÿ"),
    ("ʼ", "ʼ"),
]
_TONE_CHARS = ["\u0301", "\u0300", "\u030c", "\u0302"]
_SWAP = {"fr2ng": "ng2fr", "ng2fr": "fr2ng", "en2ng": "ng2en", "ng2en": "en2ng"}
_SOURCE = {"fr2ng": "fr", "en2ng": "en", "ng2fr": "ng", "ng2en": "ng"}
_TARGET = {"fr2ng": "ng", "en2ng": "ng", "ng2fr": "fr", "ng2en": "en"}
FALLBACK_EXAMPLES = [
    ("fr2ng", "Bonsoir, comment ça va ?", ""),
    ("en2ng", "Good evening, how are you?", ""),
    ("fr2ng", "Le cancer du col de l’utérus.", ""),
    ("ng2fr", "má", "high tone: mother"),
    ("ng2fr", "mà", "low tone: robe"),
]
COPY = {
    "en": {
        "window": "French, English ↔ Ngiemboon  ·  Batcham",
        "title": "French and English  ↔  Ngiemboon",
        "sub": "Batcham · Bamboutos · West Cameroon   ·   AGLC spelling (Anderson / SIL 2007)",
        "translate": "Translate    Ctrl+Enter",
        "clear": "Clear",
        "copy": "Copy",
        "swap": "Swap ⇄",
        "examples": "Examples:",
        "none": "(no examples)",
        "keyboard": "AGLC keyboard  ·  a wrong tone changes the meaning (má “mother” ≠ mà “robe”)",
        "tones": "AGLC tones, after the vowel:",
        "tone_btn": ["´ high", "` low", "ˇ rising", "ˆ falling"],
        "ui": "Français",
        "copied": "Translation copied.",
        "empty": "Write a sentence on the left.",
        "loading": "Loading…",
        "working": "Translating…",
        "draft": "Draft translation — check it with a Ngiemboon speaker.",
        "no_model": "No model yet. From this folder:  python -m nmt train    Then run  python -m nmt app.",
        "need_train": "No model. Train first:  python -m nmt train",
        "load_fail": "Could not load the model: {exc}",
        "loaded": "Model loaded ({device}). This is not a finished translator: the corpus is still small. Check the tones.",
        "error": "Error: {exc}",
        "langs": {"fr": "French", "en": "English", "ng": "Ngiemboon"},
        "dirs": {
            "fr2ng": "French  →  Ngiemboon",
            "en2ng": "English  →  Ngiemboon",
            "ng2fr": "Ngiemboon  →  French",
            "ng2en": "Ngiemboon  →  English",
        },
    },
    "fr": {
        "window": "Français, anglais ↔ ngiemboon  ·  Batcham",
        "title": "Français et anglais  ↔  ngiemboon",
        "sub": "Batcham · Bamboutos · Ouest Cameroun   ·   orthographe AGLC (Anderson / SIL 2007)",
        "translate": "Traduire    Ctrl+Entrée",
        "clear": "Effacer",
        "copy": "Copier",
        "swap": "Inverser ⇄",
        "examples": "Exemples :",
        "none": "(pas d’exemples)",
        "keyboard": "Clavier AGLC  ·  un mauvais ton change le sens (má « mère » ≠ mà « boubou »)",
        "tones": "Tons AGLC, après la voyelle :",
        "tone_btn": ["´ haut", "` bas", "ˇ montant", "ˆ descendant"],
        "ui": "English",
        "copied": "Traduction copiée.",
        "empty": "Écrivez une phrase à gauche.",
        "loading": "Chargement…",
        "working": "Traduction…",
        "draft": "Traduction brouillon — à relire avec un locuteur de Batcham.",
        "no_model": "Aucun modèle pour l’instant. Depuis ce dossier :  python -m nmt train    Puis relancez  python -m nmt app.",
        "need_train": "Pas de modèle. Entraînez d’abord :  python -m nmt train",
        "load_fail": "Impossible de charger le modèle : {exc}",
        "loaded": "Modèle chargé ({device}). Ce n’est pas un traducteur fini : le corpus est encore petit. Vérifiez les tons.",
        "error": "Erreur : {exc}",
        "langs": {"fr": "Français", "en": "Anglais", "ng": "Ngiemboon"},
        "dirs": {
            "fr2ng": "Français  →  ngiemboon",
            "en2ng": "Anglais  →  ngiemboon",
            "ng2fr": "Ngiemboon  →  français",
            "ng2en": "Ngiemboon  →  anglais",
        },
    },
}


def _example_label(index: int, direction: str, src: str) -> str:
    arrow = {"fr2ng": "fr → ng", "en2ng": "en → ng", "ng2fr": "ng → fr", "ng2en": "ng → en"}[direction]
    return f"{index + 1}. {arrow}  {src[:48]}"


def _load_examples() -> list[tuple[str, str, str]]:
    path = PROCESSED_DIR / "pairs.jsonl"
    if not path.exists():
        return FALLBACK_EXAMPLES
    health, bible, lexicon = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        kind = domain_of(row)
        item = (kind, normalize_fr(row["fr"]), normalize_ng(row["ng"]))
        if kind == "health" and len(health) < 4:
            health.append(item)
        elif kind == "bible" and len(bible) < 3:
            bible.append(item)
        elif kind == "lexicon" and len(lexicon) < 3:
            lexicon.append(item)
        if len(health) >= 4 and len(bible) >= 3 and len(lexicon) >= 3:
            break
    out: list[tuple[str, str, str]] = []
    for _dom, fr, ng in health + bible:
        out.append(("fr2ng", fr, ng))
    for _dom, fr, ng in lexicon:
        out.append(("ng2fr", ng, fr))
    out.append(("en2ng", "Good evening, how are you?", ""))
    return out or FALLBACK_EXAMPLES


class TranslatorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.geometry("980x740")
        self.minsize(820, 660)
        self.configure(fg_color="#13241c")

        self.ui_lang = "en"
        self.direction = tk.StringVar(value="fr2ng")
        self.status = tk.StringVar(value=COPY["en"]["loading"])
        self.model = None
        self.tokenizer = None
        self.device = None
        self.examples = _load_examples()

        self._build()
        self._apply_copy()
        self.after(80, self._try_load_model)

    def _t(self) -> dict:
        return COPY[self.ui_lang]

    def _build(self) -> None:
        pad = {"padx": 18, "pady": 8}

        header = ctk.CTkFrame(self, fg_color="#1c3328", corner_radius=16)
        header.pack(fill="x", padx=16, pady=(16, 8))
        self.header_title = ctk.CTkLabel(header, text="", font=ctk.CTkFont(size=22, weight="bold"))
        self.header_title.pack(anchor="w", padx=18, pady=(14, 0))
        self.header_sub = ctk.CTkLabel(
            header, text="", font=ctk.CTkFont(size=13), text_color="#b8d4c4"
        )
        self.header_sub.pack(anchor="w", padx=18, pady=(2, 14))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", **pad)
        self.dir_menu = ctk.CTkOptionMenu(bar, values=list(DIRECTIONS), command=self._on_direction, width=240)
        self.dir_menu.pack(side="left")
        self.swap_btn = ctk.CTkButton(bar, text="", width=120, command=self._swap)
        self.swap_btn.pack(side="left", padx=8)
        self.ui_btn = ctk.CTkButton(bar, text="", width=110, command=self._toggle_ui)
        self.ui_btn.pack(side="right")
        self.copy_btn = ctk.CTkButton(bar, text="", width=100, command=self._copy_out)
        self.copy_btn.pack(side="right", padx=6)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=4)
        body.grid_columnconfigure((0, 1), weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.src_title = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=14, weight="bold"))
        self.src_title.grid(row=0, column=0, sticky="w", padx=6)
        self.tgt_title = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=14, weight="bold"))
        self.tgt_title.grid(row=0, column=1, sticky="w", padx=6)

        self.src = ctk.CTkTextbox(body, height=180, font=ctk.CTkFont(size=16), wrap="word")
        self.src.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=4)
        self.src.bind("<Control-Return>", self._on_ctrl_enter)
        self.out = ctk.CTkTextbox(body, height=180, font=ctk.CTkFont(size=16), wrap="word")
        self.out.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=4)
        self.out.configure(state="disabled")

        self.palette = ctk.CTkFrame(body, fg_color="#1c3328", corner_radius=12)
        self.palette.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.keyboard_label = ctk.CTkLabel(
            self.palette, text="", font=ctk.CTkFont(size=12), text_color="#d7c27a"
        )
        self.keyboard_label.pack(anchor="w", padx=12, pady=(8, 4))
        row1 = ctk.CTkFrame(self.palette, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=2)
        for label, ch in AGLC_LETTERS:
            ctk.CTkButton(
                row1, text=label, width=44, height=32, command=lambda c=ch: self._insert(c)
            ).pack(side="left", padx=2, pady=2)
        row2 = ctk.CTkFrame(self.palette, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 8))
        self.tones_label = ctk.CTkLabel(row2, text="", font=ctk.CTkFont(size=12))
        self.tones_label.pack(side="left", padx=(4, 8))
        self.tone_buttons = []
        for char in _TONE_CHARS:
            button = ctk.CTkButton(
                row2, text="", width=110, height=32, command=lambda c=char: self._insert(c)
            )
            button.pack(side="left", padx=2)
            self.tone_buttons.append(button)

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=8)
        self.translate_btn = ctk.CTkButton(
            actions,
            text="",
            height=42,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._translate,
        )
        self.translate_btn.pack(side="left")
        self.clear_btn = ctk.CTkButton(actions, text="", height=42, width=110, command=self._clear)
        self.clear_btn.pack(side="left", padx=8)
        ex_values = [_example_label(i, d, src) for i, (d, src, _) in enumerate(self.examples)]
        self.ex_menu = ctk.CTkOptionMenu(
            actions, values=ex_values or ["(no examples)"], command=self._use_example, width=360
        )
        self.ex_menu.pack(side="right")
        self.examples_label = ctk.CTkLabel(actions, text="")
        self.examples_label.pack(side="right", padx=8)

        ctk.CTkLabel(
            self,
            textvariable=self.status,
            font=ctk.CTkFont(size=13),
            text_color="#9ec9b3",
            wraplength=920,
            justify="left",
        ).pack(fill="x", padx=20, pady=(0, 14))

    def _apply_copy(self) -> None:
        text = self._t()
        self.title(text["window"])
        self.header_title.configure(text=text["title"])
        self.header_sub.configure(text=text["sub"])
        self.dir_menu.configure(values=[text["dirs"][code] for code in DIRECTIONS])
        self.dir_menu.set(text["dirs"][self.direction.get()])
        self.swap_btn.configure(text=text["swap"])
        self.ui_btn.configure(text=text["ui"])
        self.copy_btn.configure(text=text["copy"])
        self.translate_btn.configure(text=text["translate"])
        self.clear_btn.configure(text=text["clear"])
        self.examples_label.configure(text=text["examples"])
        self.keyboard_label.configure(text=text["keyboard"])
        self.tones_label.configure(text=text["tones"])
        for button, label in zip(self.tone_buttons, text["tone_btn"]):
            button.configure(text=label)
        self._sync_direction_ui()

    def _sync_direction_ui(self) -> None:
        text = self._t()
        code = self.direction.get()
        self.src_title.configure(text=text["langs"][_SOURCE[code]])
        self.tgt_title.configure(text=text["langs"][_TARGET[code]])

    def _toggle_ui(self) -> None:
        self.ui_lang = "fr" if self.ui_lang == "en" else "en"
        self._apply_copy()

    def _on_direction(self, choice: str) -> None:
        for code, label in self._t()["dirs"].items():
            if label == choice:
                self.direction.set(code)
                self._sync_direction_ui()
                return

    def _on_ctrl_enter(self, _event):
        self._translate()
        return "break"

    def _insert(self, char: str) -> None:
        self.src.insert("insert", char)
        self.src.focus()

    def _swap(self) -> None:
        current = self.src.get("1.0", "end").strip()
        self.out.configure(state="normal")
        other = self.out.get("1.0", "end").strip()
        self.out.configure(state="disabled")
        self.direction.set(_SWAP[self.direction.get()])
        self.dir_menu.set(self._t()["dirs"][self.direction.get()])
        self.src.delete("1.0", "end")
        if other:
            self.src.insert("1.0", other)
        elif current:
            self.src.insert("1.0", current)
        self._set_out("")
        self._sync_direction_ui()

    def _clear(self) -> None:
        self.src.delete("1.0", "end")
        self._set_out("")

    def _set_out(self, text: str) -> None:
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        if text:
            self.out.insert("1.0", text)
        self.out.configure(state="disabled")

    def _copy_out(self) -> None:
        self.out.configure(state="normal")
        text = self.out.get("1.0", "end").strip()
        self.out.configure(state="disabled")
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status.set(self._t()["copied"])

    def _use_example(self, choice: str) -> None:
        for index, (direction, src, _tgt) in enumerate(self.examples):
            if _example_label(index, direction, src) == choice:
                self.direction.set(direction)
                self.dir_menu.set(self._t()["dirs"][direction])
                self._sync_direction_ui()
                self.src.delete("1.0", "end")
                self.src.insert("1.0", src)
                self._set_out("")
                return

    def _try_load_model(self) -> None:
        ckpt = CHECKPOINTS / "fr_ngiemboon.pt"
        tok = PROCESSED_DIR / "tokenizer.json"
        text = self._t()
        if not ckpt.exists() or not tok.exists():
            self.status.set(text["no_model"])
            return
        try:
            from .train import load_model

            self.model, self.tokenizer, self.device = load_model()
            self.status.set(text["loaded"].format(device=self.device))
        except Exception as exc:
            self.status.set(text["load_fail"].format(exc=exc))

    def _translate(self) -> None:
        text = self.src.get("1.0", "end").strip()
        copy = self._t()
        if not text:
            self.status.set(copy["empty"])
            return
        if self.model is None:
            self.status.set(copy["need_train"])
            return
        self.status.set(copy["working"])
        self.update_idletasks()
        try:
            result, notes = translate_between(
                self.model,
                self.tokenizer,
                text,
                self.direction.get(),
                self.device,
                lang=self.ui_lang,
            )
        except (PivotError, Exception) as exc:
            self.status.set(copy["error"].format(exc=exc))
            return
        self._set_out(result)
        self.status.set(" ".join(notes) or copy["draft"])


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    app = TranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
