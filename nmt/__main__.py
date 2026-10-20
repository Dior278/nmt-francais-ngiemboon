"""One command.

    python -m nmt                         French → Ngiemboon, then a prompt
    python -m nmt "Bonsoir, comment ça va ?"
    python -m nmt --from en "Good evening"
    python -m nmt --to fr "má"
    python -m nmt --to en "má"
    python -m nmt train
    python -m nmt app
"""

from __future__ import annotations

import argparse
import sys

from .pivot import PivotError, resolve_direction, translate_between
from .train import load_model

_LABELS = {
    "fr2ng": "French → Ngiemboon",
    "en2ng": "English → Ngiemboon",
    "ng2fr": "Ngiemboon → French",
    "ng2en": "Ngiemboon → English",
}
_COMMANDS = {
    "/fr-ng": "fr2ng",
    "/en-ng": "en2ng",
    "/ng-fr": "ng2fr",
    "/ng-en": "ng2en",
    "/ng": "fr2ng",
    "/fr": "ng2fr",
    "/en": "ng2en",
}


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _translate_once(text: str, direction: str, bundle) -> str:
    model, tokenizer, device = bundle
    try:
        result, notes = translate_between(model, tokenizer, text, direction, device)
    except PivotError as exc:
        raise SystemExit(str(exc)) from exc
    for note in notes:
        print("Note: " + note, file=sys.stderr)
    return result


def _interactive(direction: str, bundle) -> None:
    print("French, English ↔ Ngiemboon    Batcham, Bamboutos")
    print("Enter translates.   /fr-ng  /en-ng  /ng-fr  /ng-en   /q quits.")
    print("direction:", _LABELS[direction])
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in {"/q", "/quit", "quit", "exit"}:
            return
        if text in _COMMANDS:
            direction = _COMMANDS[text]
            print("direction:", _LABELS[direction])
            continue
        print(_translate_once(text, direction, bundle))


def _translate_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m nmt",
        description=(
            "Translate between Ngiemboon and French or English. "
            "Traduction ngiemboon, français ou anglais. "
            "English goes through French."
        ),
    )
    parser.add_argument("text", nargs="*", help="Sentence to translate. With no sentence, prompt.")
    parser.add_argument(
        "--from",
        dest="source",
        choices=["fr", "en", "ng"],
        help="Source language. Default: fr, or ng when the target is fr or en.",
    )
    parser.add_argument(
        "--to",
        dest="target",
        choices=["ng", "fr", "en"],
        help="Target language. Default: ng, or fr when the source is ng.",
    )
    parser.add_argument(
        "--vers",
        choices=["ng", "fr", "en"],
        help="Same as --to. Older commands used --vers fr.",
    )
    args = parser.parse_args(argv)
    target = args.target or args.vers
    try:
        direction = resolve_direction(args.source, target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bundle = load_model()
    text = " ".join(args.text).strip()
    if not text:
        _interactive(direction, bundle)
        return
    print(_translate_once(text, direction, bundle))


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        from .config import parse_train_args
        from .train import train

        args = parse_train_args(argv[1:])
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            force_extract=args.rebuild_corpus,
            resume=not args.no_resume,
            device_name=args.device,
        )
        return
    if argv and argv[0] == "app":
        from .app import main as app_main

        app_main()
        return
    _translate_cli(argv)


if __name__ == "__main__":
    main()
