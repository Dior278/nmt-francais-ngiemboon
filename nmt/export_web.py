"""Write the trained network as two ONNX files for the public page.

The page runs them in the browser. Nothing here is a server.
"""

from __future__ import annotations

import shutil

import torch
import torch.nn as nn

from .decode import greedy_translate
from .model import Seq2SeqTransformer, causal_mask
from .paths import PROCESSED_DIR, ROOT
from .train import load_model

OUT = ROOT / "docs" / "model"


class _Encode(nn.Module):
    def __init__(self, model: Seq2SeqTransformer):
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = src.ne(self.model.pad_id).unsqueeze(1).unsqueeze(2)
        return self.model.encode(src, mask)


class _Decode(nn.Module):
    def __init__(self, model: Seq2SeqTransformer):
        super().__init__()
        self.model = model

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        src_mask = src.ne(self.model.pad_id).unsqueeze(1).unsqueeze(2)
        logits = self.model.decode(tgt, memory, src_mask, causal_mask(tgt, self.model.pad_id))
        return logits[:, -1, :]


def export() -> None:
    model, tokenizer, device = load_model("cpu")
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROCESSED_DIR / "tokenizer.json", OUT / "tokenizer.json")

    src = torch.tensor([[tokenizer.to_ng, tokenizer.bos_id, 12, tokenizer.eos_id]], dtype=torch.long)
    tgt = torch.tensor([[tokenizer.bos_id, 12]], dtype=torch.long)
    encode = _Encode(model).eval()
    decode = _Decode(model).eval()
    memory = encode(src)

    torch.onnx.export(
        encode,
        (src,),
        OUT / "encode.onnx",
        input_names=["src"],
        output_names=["memory"],
        dynamic_axes={"src": {1: "src_len"}, "memory": {1: "src_len"}},
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        decode,
        (tgt, memory, src),
        OUT / "decode.onnx",
        input_names=["tgt", "memory", "src"],
        output_names=["logits"],
        dynamic_axes={
            "tgt": {1: "tgt_len"},
            "memory": {1: "src_len"},
            "src": {1: "src_len"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    enc = ort.InferenceSession(str(OUT / "encode.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(OUT / "decode.onnx"), providers=["CPUExecutionProvider"])
    text = "Bonsoir, comment ça va ?"
    expected = greedy_translate(model, tokenizer, text, "ng", device, max_len=32)
    ids = tokenizer.encode(text, lang_tag="ng")
    src_np = torch.tensor([ids], dtype=torch.long).numpy()
    mem = enc.run(None, {"src": src_np})[0]
    ys = [tokenizer.bos_id]
    for _ in range(32):
        logits = dec.run(
            None,
            {
                "tgt": torch.tensor([ys], dtype=torch.long).numpy(),
                "memory": mem,
                "src": src_np,
            },
        )[0]
        nxt = int(logits[0].argmax())
        ys.append(nxt)
        if nxt == tokenizer.eos_id:
            break
    from .normalize import present

    got = present(tokenizer.decode(ys), "ng")
    report = OUT / "check.txt"
    report.write_text(f"expected\n{expected}\nonnx\n{got}\n", encoding="utf-8")
    if got != expected:
        raise SystemExit("ONNX decode does not match the checkpoint")
    print("match", OUT)


if __name__ == "__main__":
    export()
