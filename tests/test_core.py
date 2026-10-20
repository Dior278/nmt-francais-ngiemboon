"""Regression tests for tone handling, masks, and checkpoint resume."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nmt.dataset import PairDataset, split_pairs
from nmt.decode import beam_translate, greedy_translate, prepare_source, tone_notes
from nmt.model import MultiHeadAttention, Seq2SeqTransformer
from nmt.normalize import has_tone_marks, make_pair, normalize_fr, normalize_ng, present, strip_tones
from nmt.tokenizer import train_bpe
from nmt.train import _atomic_save, _checkpoint_matches, _restore_optim


class ToneTests(unittest.TestCase):
    def test_diaeresis_is_not_a_tone(self) -> None:
        self.assertEqual(strip_tones("pɛ\u0300ẅÿ"), "pɛẅÿ")
        self.assertEqual(strip_tones("má"), "ma")
        self.assertEqual(strip_tones("mà"), "ma")
        self.assertFalse(has_tone_marks("ẅÿ"))
        self.assertTrue(has_tone_marks("má"))
        self.assertTrue(has_tone_marks("pɛ\u0300"))
        self.assertFalse(has_tone_marks("ça"))

    def test_same_tone_on_every_vowel(self) -> None:
        high = "\u0301"
        self.assertEqual(normalize_ng("má"), "ma" + high)
        self.assertEqual(normalize_ng("ma\u0301"), "ma" + high)
        self.assertEqual(normalize_ng("ma´"), "ma" + high)
        self.assertEqual(normalize_ng("mé"), "me" + high)
        self.assertEqual(normalize_ng("mɛ\u0301"), "mɛ" + high)
        self.assertEqual(normalize_ng("ń"), "n" + high)
        self.assertEqual(normalize_ng("má")[-1], normalize_ng("mɛ\u0301")[-1])
        self.assertEqual(normalize_ng(normalize_ng("má")), normalize_ng("má"))
        self.assertEqual(strip_tones(normalize_ng("ẅ\u0301")), "ẅ")
        self.assertEqual(normalize_ng("sa'a"), "sa\u02bca")

    def test_french_accents_stay_letters(self) -> None:
        self.assertEqual(normalize_fr("La mère est à café."), "La mère est à café.")
        self.assertNotIn("\u0301", normalize_fr("café"))
        self.assertNotIn("\u0300", normalize_fr("mère"))
        self.assertEqual(normalize_fr("l'homme"), "l'homme")
        self.assertEqual(present("e\u0301té", "fr"), "été")
        self.assertEqual(present("má", "ng"), "ma\u0301")

    def test_directions_keep_their_orthography(self) -> None:
        french, missing = prepare_source("café", "fr2ng")
        ng, ambiguous = prepare_source("má", "ng2fr")
        bare, bare_missing = prepare_source("ma", "ng2fr")
        self.assertEqual(french, "café")
        self.assertFalse(missing)
        self.assertEqual(ng, "ma\u0301")
        self.assertFalse(ambiguous)
        self.assertTrue(bare_missing)
        self.assertEqual(prepare_source(ng, "ng2fr")[0], ng)
        row = make_pair("T.1", "MOI", "1", "La mère.", "má")
        self.assertEqual(row["ng"], "ma\u0301")
        self.assertIn("è", row["fr"])
        self.assertNotIn("\u0300", row["fr"])
        self.assertTrue(tone_notes("fr2ng", False, "ma"))
        self.assertFalse(tone_notes("fr2ng", False, "ma\u0301"))
        self.assertTrue(tone_notes("ng2fr", True, "mère"))

    def test_training_pairs_separate_tone_from_french_accent(self) -> None:
        french = "La mère."
        ng = "má"
        with contextlib.redirect_stdout(io.StringIO()):
            tokenizer = train_bpe(
                [normalize_fr(french), normalize_ng(ng), normalize_ng("mɛ\u0301")],
                vocab_size=80,
                min_freq=1,
            )
        pieces = tokenizer._encode_word(normalize_ng(ng))
        self.assertIn("\u0301", pieces)
        self.assertTrue(any(piece == "\u0301" for piece in pieces))
        mother = tokenizer._encode_word("mère")
        self.assertTrue(any("è" in piece for piece in mother))
        self.assertNotIn("\u0300", "".join(mother))
        dataset = PairDataset(
            [{"id": "T.1", "book": "MOI", "ref": "1", "fr": french, "ng": ng}],
            tokenizer,
            tone_augment=True,
        )
        surfaces = [(tokenizer.decode(src), tokenizer.decode(tgt)) for src, tgt in dataset.examples]
        self.assertIn((normalize_fr(french), normalize_ng(ng)), surfaces)
        self.assertIn((normalize_ng(ng), normalize_fr(french)), surfaces)
        self.assertIn(("ma", normalize_fr(french)), surfaces)
        toned_targets = [tgt for _src, tgt in surfaces if "\u0301" in tgt]
        self.assertTrue(toned_targets)
        self.assertTrue(all("\u0300" not in tgt for _src, tgt in surfaces if "mère" in tgt))

    def test_bpe_roundtrip_keeps_aglc(self) -> None:
        text = "pɛ\u0300ẅÿ má"
        with contextlib.redirect_stdout(io.StringIO()):
            tokenizer = train_bpe([text, "bonjour mère"], vocab_size=80, min_freq=1)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text, add_special=False)), text)


class MaskTests(unittest.TestCase):
    def test_padding_is_on_keys(self) -> None:
        model = Seq2SeqTransformer(
            vocab_size=40,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            dropout=0.0,
            max_len=16,
            pad_id=0,
        )
        src = torch.tensor([[1, 2, 3, 0, 0]])
        tgt = torch.tensor([[1, 4, 0, 0]])
        src_mask, tgt_mask = model.make_masks(src, tgt)
        self.assertTrue(bool(src_mask[0, 0, 0, 1]))
        self.assertFalse(bool(src_mask[0, 0, 0, 3]))
        self.assertTrue(bool(tgt_mask[0, 0, 0, 0]))
        self.assertFalse(bool(tgt_mask[0, 0, 0, 1]))
        # Pad query can read earlier real keys, and nobody reads the pad key.
        self.assertTrue(bool(tgt_mask[0, 0, 2, 0]))
        self.assertFalse(bool(tgt_mask[0, 0, :, 2].any()))

    def test_fp16_attention_accepts_a_full_mask(self) -> None:
        attention = MultiHeadAttention(32, 4, 0.0).half()
        hidden = torch.randn(1, 4, 32, dtype=torch.float16)
        blocked = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        output = attention(hidden, hidden, hidden, blocked)
        self.assertTrue(torch.isfinite(output).all())
        partial = torch.zeros(1, 1, 1, 4, dtype=torch.bool)
        partial[..., :2] = True
        output = attention(hidden, hidden, hidden, partial)
        self.assertTrue(torch.isfinite(output).all())

    def test_padded_backward_stays_finite(self) -> None:
        torch.manual_seed(0)
        model = Seq2SeqTransformer(
            vocab_size=50,
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            dropout=0.0,
            max_len=16,
            pad_id=0,
        )
        src = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 3, 0, 0, 0]])
        tgt = torch.tensor([[1, 5, 6, 7, 2, 0], [1, 5, 6, 0, 0, 0]])
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())


class DataTests(unittest.TestCase):
    def test_small_split_keeps_a_train_set(self) -> None:
        pairs = [{"id": str(i)} for i in range(10)]
        train, val = split_pairs(pairs, seed=0)
        self.assertEqual(len(train) + len(val), 10)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)

    def test_large_split_holds_out_at_least_50(self) -> None:
        pairs = [{"id": str(i)} for i in range(1000)]
        _train, val = split_pairs(pairs, seed=0)
        self.assertGreaterEqual(len(val), 50)
        self.assertLess(len(val), 1000)


class ResumeTests(unittest.TestCase):
    def test_checkpoint_match_requires_the_same_width(self) -> None:
        cfg = {"d_model": 32, "num_heads": 4, "num_layers": 1, "d_ff": 64, "max_len": 16}
        payload = {"vocab_size": 40, "pad_id": 0, "config": dict(cfg)}
        self.assertTrue(_checkpoint_matches(payload, cfg, 40, 0))
        other = dict(cfg)
        other["d_model"] = 64
        self.assertFalse(_checkpoint_matches(payload, other, 40, 0))

    def test_restore_keeps_the_learning_rate(self) -> None:
        torch.manual_seed(0)
        model = Seq2SeqTransformer(
            vocab_size=40,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            dropout=0.0,
            max_len=16,
            pad_id=0,
        )

        def lr_lambda(step: int) -> float:
            step = max(int(step) + 1, 1)
            return (32 ** -0.5) * min(step ** -0.5, step * (8 ** -1.5))

        optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = LambdaLR(optimizer, lr_lambda)
        for _ in range(6):
            model.embed.weight.grad = torch.ones_like(model.embed.weight)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        expected = optimizer.param_groups[0]["lr"]
        payload = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}

        restarted = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        restarted_sched = LambdaLR(restarted, lr_lambda)
        self.assertNotAlmostEqual(restarted.param_groups[0]["lr"], expected)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        _restore_optim(payload, restarted, restarted_sched, scaler, lambda _message: None)
        self.assertAlmostEqual(restarted.param_groups[0]["lr"], expected)

    def test_atomic_save(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            _atomic_save({"value": 3}, path)
            self.assertEqual(torch.load(path, weights_only=True)["value"], 3)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


class DecodeTests(unittest.TestCase):
    def _tokenizer(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return train_bpe(["bonjour ẅɛ", "má mère"], vocab_size=60, min_freq=1)

    def test_greedy_and_beam_return_text(self) -> None:
        tokenizer = self._tokenizer()
        model = Seq2SeqTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            dropout=0.0,
            max_len=32,
            pad_id=tokenizer.pad_id,
        ).eval()
        device = torch.device("cpu")
        greedy = greedy_translate(model, tokenizer, "bonjour", "ng", device, max_len=6)
        beam = beam_translate(model, tokenizer, "bonjour", "ng", device, beam_size=3, max_len=6)
        self.assertIsInstance(greedy, str)
        self.assertIsInstance(beam, str)

    def test_long_source_is_clipped(self) -> None:
        tokenizer = self._tokenizer()
        model = Seq2SeqTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            dropout=0.0,
            max_len=16,
            pad_id=tokenizer.pad_id,
        ).eval()
        note = io.StringIO()
        with contextlib.redirect_stderr(note):
            text = greedy_translate(
                model, tokenizer, "mot " * 80, "ng", torch.device("cpu"), max_len=4
            )
        self.assertIsInstance(text, str)
        self.assertIn("length", note.getvalue())


class EnglishRouteTests(unittest.TestCase):
    def _bundle(self):
        with contextlib.redirect_stdout(io.StringIO()):
            tokenizer = train_bpe(["bonjour ẅɛ", "má mère"], vocab_size=60, min_freq=1)
        model = Seq2SeqTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            dropout=0.0,
            max_len=32,
            pad_id=tokenizer.pad_id,
        ).eval()
        return model, tokenizer, torch.device("cpu")

    def test_direction_pairs(self) -> None:
        from nmt.pivot import resolve_direction

        self.assertEqual(resolve_direction(None, None), "fr2ng")
        self.assertEqual(resolve_direction("en", None), "en2ng")
        self.assertEqual(resolve_direction(None, "fr"), "ng2fr")
        self.assertEqual(resolve_direction(None, "en"), "ng2en")
        self.assertEqual(resolve_direction("ng", "en"), "ng2en")
        with self.assertRaises(ValueError):
            resolve_direction("fr", "en")

    def test_english_goes_through_french(self) -> None:
        from nmt.pivot import translate_between

        model, tokenizer, device = self._bundle()
        calls = []

        def fake_pivot(text, direction):
            calls.append((text, direction))
            return "Bonsoir" if direction == "en2fr" else "Good evening"

        result, notes = translate_between(
            model, tokenizer, "Good evening", "en2ng", device, pivot_fn=fake_pivot, beam_size=1
        )
        self.assertIsInstance(result, str)
        self.assertEqual(calls, [("Good evening", "en2fr")])
        self.assertTrue(any("French" in note for note in notes))

        calls.clear()
        result, notes = translate_between(
            model, tokenizer, "má", "ng2en", device, pivot_fn=fake_pivot, beam_size=1
        )
        self.assertEqual(result, "Good evening")
        self.assertEqual(calls[0][1], "fr2en")
        self.assertTrue(any("French" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
