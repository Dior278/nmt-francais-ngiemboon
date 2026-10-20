"""Command-line translation: French or English ↔ Ngiemboon."""

from __future__ import annotations

import argparse
import sys

from .pivot import PivotError, translate_between
from .train import load_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Translate between Ngiemboon and French or English. English goes through French."
    )
    parser.add_argument("text", nargs="?", help="Text to translate. If omitted, reads stdin.")
    parser.add_argument(
        "--dir",
        choices=["fr2ng", "en2ng", "ng2fr", "ng2en"],
        default="fr2ng",
        help="fr2ng or en2ng into Ngiemboon; ng2fr or ng2en out of Ngiemboon.",
    )
    args = parser.parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("Empty input.")
    model, tokenizer, device = load_model()
    try:
        result, notes = translate_between(model, tokenizer, text, args.dir, device)
    except PivotError as exc:
        raise SystemExit(str(exc)) from exc
    for note in notes:
        print("Note: " + note, file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
