"""Command-line translation: French ↔ Ngiemboon."""

from __future__ import annotations

import argparse
import sys

from .decode import prepare_source, translate_pair
from .train import load_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Translate between French and Ngiemboon (Guembon).")
    parser.add_argument("text", nargs="?", help="Text to translate. If omitted, reads stdin.")
    parser.add_argument(
        "--dir",
        choices=["fr2ng", "ng2fr"],
        default="fr2ng",
        help="fr2ng = French → Ngiemboon, ng2fr = Ngiemboon → French",
    )
    args = parser.parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("Empty input.")
    model, tokenizer, device = load_model()
    prepared, tone_missing = prepare_source(text, args.dir)
    if tone_missing:
        print(
            "Note: aucun ton détecté. Sans tons le ngiemboon est ambigu "
            "(má « mère » ≠ mà « boubou »). Le modèle tente quand même.",
            file=sys.stderr,
        )
    print(translate_pair(model, tokenizer, prepared, args.dir, device))


if __name__ == "__main__":
    main()
