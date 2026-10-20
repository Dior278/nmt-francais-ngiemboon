"""Commande unique.

    python -m nmt                         boucle français → ngiemboon
    python -m nmt "Bonsoir, comment ça va ?"
    python -m nmt --vers fr "má"
    python -m nmt train
    python -m nmt app
"""

from __future__ import annotations

import argparse
import sys

from .decode import prepare_source, tone_notes, translate_pair
from .train import load_model


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _translate_once(text: str, direction: str, bundle) -> str:
    model, tokenizer, device = bundle
    prepared, tone_missing = prepare_source(text, direction)
    result = translate_pair(model, tokenizer, prepared, direction, device)
    for note in tone_notes(direction, tone_missing, result):
        print("Note : " + note, file=sys.stderr)
    return result


def _interactive(direction: str, bundle) -> None:
    print("français ↔ ngiemboon    Batcham · Bamboutos")
    print("Entrée pour traduire.   /fr ou /ng pour changer de sens.   /q pour quitter.")
    print(f"sens : {'français → ngiemboon' if direction == 'fr2ng' else 'ngiemboon → français'}")
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
        if text == "/fr":
            direction = "ng2fr"
            print("sens : ngiemboon → français")
            continue
        if text == "/ng":
            direction = "fr2ng"
            print("sens : français → ngiemboon")
            continue
        print(_translate_once(text, direction, bundle))


def _translate_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m nmt",
        description="Traduire entre le français et le ngiemboon.",
    )
    parser.add_argument("text", nargs="*", help="Phrase à traduire. Sans phrase : mode interactif.")
    parser.add_argument(
        "--vers",
        choices=["ng", "fr"],
        default="ng",
        help="Langue d'arrivée. ng = français → ngiemboon (défaut), fr = l'inverse.",
    )
    args = parser.parse_args(argv)
    direction = "fr2ng" if args.vers == "ng" else "ng2fr"
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
