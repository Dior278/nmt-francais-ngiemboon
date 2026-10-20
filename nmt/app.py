"""Interface de traduction français ↔ ngiemboon (Batcham, Bamboutos).

Lancement :  python -m nmt.app
N’entraîne jamais le modèle. S’il n’y a pas de checkpoint, un message
indique de lancer l’entraînement sur GPU.
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
        "customtkinter manque. Dans Transformer/ :  pip install -r nmt/requirements.txt"
    ) from exc

from .decode import prepare_source, tone_notes, translate_pair
from .normalize import domain_of, normalize_fr, normalize_ng

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
TONE_MARKS = [
    ("´ haut", "\u0301"),
    ("` bas", "\u0300"),
    ("ˇ montant", "\u030c"),
    ("ˆ descendant", "\u0302"),
]
FALLBACK_EXAMPLES = [
    ("fr2ng", "Bonsoir, comment ça va ?", ""),
    ("fr2ng", "Le cancer du col de l’utérus.", ""),
    ("ng2fr", "má", "ton haut : mère"),
    ("ng2fr", "mà", "ton bas : boubou"),
]


def _example_label(index: int, direction: str, src: str) -> str:
    arrow = "fr → ng" if direction == "fr2ng" else "ng → fr"
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
    return out or FALLBACK_EXAMPLES


class TranslatorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Français ↔ Ngiemboon  ·  Batcham")
        self.geometry("980x720")
        self.minsize(820, 640)
        self.configure(fg_color="#13241c")

        self.direction = tk.StringVar(value="fr2ng")
        self.status = tk.StringVar(value="Chargement…")
        self.model = None
        self.tokenizer = None
        self.device = None
        self.examples = _load_examples()

        self._build()
        self.after(80, self._try_load_model)

    def _build(self) -> None:
        pad = {"padx": 18, "pady": 8}

        header = ctk.CTkFrame(self, fg_color="#1c3328", corner_radius=16)
        header.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(
            header,
            text="Traducteur français ↔ ngiemboon",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(14, 0))
        ctk.CTkLabel(
            header,
            text="Batcham · Bamboutos · Ouest Cameroun   ·   orthographe AGLC (Anderson / SIL 2007)",
            font=ctk.CTkFont(size=13),
            text_color="#b8d4c4",
        ).pack(anchor="w", padx=18, pady=(2, 14))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", **pad)
        self.dir_label = ctk.CTkLabel(
            bar, text="Français  →  Ngiemboon", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.dir_label.pack(side="left")
        ctk.CTkButton(bar, text="Inverser ⇄", width=130, command=self._swap).pack(side="right", padx=6)
        ctk.CTkButton(bar, text="Copier", width=100, command=self._copy_out).pack(side="right")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=4)
        body.grid_columnconfigure((0, 1), weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.src_title = ctk.CTkLabel(body, text="Français", font=ctk.CTkFont(size=14, weight="bold"))
        self.src_title.grid(row=0, column=0, sticky="w", padx=6)
        self.tgt_title = ctk.CTkLabel(body, text="Ngiemboon", font=ctk.CTkFont(size=14, weight="bold"))
        self.tgt_title.grid(row=0, column=1, sticky="w", padx=6)

        self.src = ctk.CTkTextbox(body, height=180, font=ctk.CTkFont(size=16), wrap="word")
        self.src.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=4)
        self.src.bind("<Control-Return>", self._on_ctrl_enter)
        self.out = ctk.CTkTextbox(body, height=180, font=ctk.CTkFont(size=16), wrap="word")
        self.out.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=4)
        self.out.configure(state="disabled")

        self.palette = ctk.CTkFrame(body, fg_color="#1c3328", corner_radius=12)
        self.palette.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ctk.CTkLabel(
            self.palette,
            text="Clavier AGLC  ·  un mauvais ton change le sens (má « mère » ≠ mà « boubou »)",
            font=ctk.CTkFont(size=12),
            text_color="#d7c27a",
        ).pack(anchor="w", padx=12, pady=(8, 4))
        row1 = ctk.CTkFrame(self.palette, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=2)
        for label, ch in AGLC_LETTERS:
            ctk.CTkButton(
                row1, text=label, width=44, height=32, command=lambda c=ch: self._insert(c)
            ).pack(side="left", padx=2, pady=2)
        row2 = ctk.CTkFrame(self.palette, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(row2, text="Tons AGLC, après la voyelle :", font=ctk.CTkFont(size=12)).pack(
            side="left", padx=(4, 8)
        )
        for label, ch in TONE_MARKS:
            ctk.CTkButton(
                row2, text=label, width=110, height=32, command=lambda c=ch: self._insert(c)
            ).pack(side="left", padx=2)

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=8)
        ctk.CTkButton(
            actions,
            text="Traduire    Ctrl+Entrée",
            height=42,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._translate,
        ).pack(side="left")
        ctk.CTkButton(actions, text="Effacer", height=42, width=110, command=self._clear).pack(
            side="left", padx=8
        )
        ex_values = [_example_label(i, d, src) for i, (d, src, _) in enumerate(self.examples)]
        self.ex_menu = ctk.CTkOptionMenu(
            actions, values=ex_values or ["(pas d’exemples)"], command=self._use_example, width=360
        )
        self.ex_menu.pack(side="right")
        ctk.CTkLabel(actions, text="Exemples :").pack(side="right", padx=8)

        ctk.CTkLabel(
            self,
            textvariable=self.status,
            font=ctk.CTkFont(size=13),
            text_color="#9ec9b3",
            wraplength=920,
            justify="left",
        ).pack(fill="x", padx=20, pady=(0, 14))

        self._sync_direction_ui()

    def _sync_direction_ui(self) -> None:
        if self.direction.get() == "fr2ng":
            self.dir_label.configure(text="Français  →  Ngiemboon")
            self.src_title.configure(text="Français")
            self.tgt_title.configure(text="Ngiemboon")
        else:
            self.dir_label.configure(text="Ngiemboon  →  Français")
            self.src_title.configure(text="Ngiemboon")
            self.tgt_title.configure(text="Français")

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
        self.direction.set("ng2fr" if self.direction.get() == "fr2ng" else "fr2ng")
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
        self.status.set("Traduction copiée.")

    def _use_example(self, choice: str) -> None:
        for index, (direction, src, _tgt) in enumerate(self.examples):
            if _example_label(index, direction, src) == choice:
                self.direction.set(direction)
                self._sync_direction_ui()
                self.src.delete("1.0", "end")
                self.src.insert("1.0", src)
                self._set_out("")
                return

    def _try_load_model(self) -> None:
        ckpt = CHECKPOINTS / "fr_ngiemboon.pt"
        tok = PROCESSED_DIR / "tokenizer.json"
        if not ckpt.exists() or not tok.exists():
            self.status.set(
                "Aucun modèle pour l’instant. Depuis ce dossier :  python -m nmt train    "
                "Puis relancez  python -m nmt app."
            )
            return
        try:
            from .train import load_model

            self.model, self.tokenizer, self.device = load_model()
            self.status.set(
                f"Modèle chargé ({self.device}). Ce n’est pas un traducteur parfait : "
                "Bible + santé + lexique, corpus encore petit. Vérifiez surtout les tons."
            )
        except Exception as exc:
            self.status.set(f"Impossible de charger le modèle : {exc}")

    def _translate(self) -> None:
        text = self.src.get("1.0", "end").strip()
        if not text:
            self.status.set("Écrivez une phrase à gauche.")
            return
        if self.model is None:
            self.status.set("Pas de modèle. Entraînez d’abord :  python -m nmt train")
            return
        direction = self.direction.get()
        prepared, tone_missing = prepare_source(text, direction)
        target = "ng" if direction == "fr2ng" else "fr"
        notes: list[str] = []
        if len(self.tokenizer.encode(prepared, lang_tag=target)) > self.model.max_positions():
            notes.append("Phrase trop longue : la fin n'est pas lue.")
        self.status.set("Traduction…")
        self.update_idletasks()
        try:
            result = translate_pair(self.model, self.tokenizer, prepared, direction, self.device)
        except Exception as exc:
            self.status.set(f"Erreur : {exc}")
            return
        notes.extend(tone_notes(direction, tone_missing, result, keyboard=True))
        self._set_out(result)
        self.status.set(" ".join(notes) or "Traduction brouillon — à relire avec un locuteur de Batcham.")


def main() -> None:
    # Make `import nmt` work when launched as a script from Transformer/.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    app = TranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
