/* French / English ↔ Ngiemboon in the browser. The network runs here. */

const TONE = "\u0300\u0301\u0302\u030c\u0304\u030f";
const SPACING = {
  "\u0060": "\u0300",
  "\u02cb": "\u0300",
  "\u00b4": "\u0301",
  "\u02ca": "\u0301",
  "\u02c6": "\u0302",
  "\u02c7": "\u030c",
  "\u00af": "\u0304",
  "\u02c9": "\u0304",
};
const TOKEN_RE = /\S+/g;

function holdsTone(piece) {
  for (const ch of piece) if (TONE.includes(ch)) return true;
  return false;
}

function normalizeFr(text) {
  return text.trim().normalize("NFC");
}

function normalizeNg(text) {
  let s = text.trim().replace(/['\u2018\u2019]/g, "\u02bc");
  let out = [];
  for (const ch of s) {
    const tone = SPACING[ch];
    if (tone && out.length && /\p{L}/u.test(out[out.length - 1])) out.push(tone);
    else out.push(ch);
  }
  const decomposed = out.join("").normalize("NFD");
  const result = [];
  let i = 0;
  while (i < decomposed.length) {
    const char = decomposed[i++];
    if (/^\p{M}$/u.test(char)) {
      result.push(char);
      continue;
    }
    const tones = [];
    const others = [];
    while (i < decomposed.length && /^\p{M}$/u.test(decomposed[i])) {
      const mark = decomposed[i++];
      if (TONE.includes(mark)) tones.push(mark);
      else others.push(mark);
    }
    result.push((char + others.join("")).normalize("NFC"));
    for (const tone of TONE) if (tones.includes(tone)) result.push(tone);
  }
  return result.join("");
}

function present(text, lang) {
  return lang === "ng" ? normalizeNg(text) : normalizeFr(text);
}

class BPETokenizer {
  constructor(payload) {
    this.vocab = payload.vocab;
    this.merges = payload.merges.map((p) => [p[0], p[1]]);
    this.idToToken = Object.fromEntries(Object.entries(this.vocab).map(([t, i]) => [i, t]));
    this.mergeRanks = new Map(this.merges.map((p, i) => [p[0] + "\0" + p[1], i]));
    this.padId = this.vocab["<pad>"];
    this.bosId = this.vocab["<bos>"];
    this.eosId = this.vocab["<eos>"];
    this.unkId = this.vocab["<unk>"];
    this.toFr = this.vocab["<2fr>"];
    this.toNg = this.vocab["<2ng>"];
    this.domainIds = ["<bible>", "<health>", "<lexicon>", "<grammar>"]
      .map((t) => this.vocab[t])
      .filter((v) => v !== undefined);
    this.wordCache = new Map();
  }

  encodeWord(word) {
    if (this.wordCache.has(word)) return this.wordCache.get(word);
    let chars = [...word, "</w>"];
    while (chars.length >= 2) {
      let best = null;
      for (let i = 0; i < chars.length - 1; i++) {
        if (holdsTone(chars[i]) || holdsTone(chars[i + 1])) continue;
        const key = chars[i] + "\0" + chars[i + 1];
        if (!this.mergeRanks.has(key)) continue;
        const rank = this.mergeRanks.get(key);
        if (best === null || rank < best.rank) best = { rank, i };
      }
      if (!best) break;
      const i = best.i;
      chars = [...chars.slice(0, i), chars[i] + chars[i + 1], ...chars.slice(i + 2)];
    }
    this.wordCache.set(word, chars);
    return chars;
  }

  encode(text, langTag) {
    const pieces = [];
    for (const word of text.trim().match(TOKEN_RE) || []) pieces.push(...this.encodeWord(word));
    let ids = pieces.map((p) => (p in this.vocab ? this.vocab[p] : this.unkId));
    const prefix = [];
    if (langTag === "fr") prefix.push(this.toFr);
    else if (langTag === "ng") prefix.push(this.toNg);
    return [...prefix, this.bosId, ...ids, this.eosId];
  }

  decode(ids) {
    const skip = new Set([
      this.padId,
      this.bosId,
      this.eosId,
      this.toFr,
      this.toNg,
      this.unkId,
      ...this.domainIds,
    ]);
    const tokens = ids.filter((i) => !skip.has(i)).map((i) => this.idToToken[i] || "");
    return tokens.join("").replace(/<\/w>/g, " ").replace(/\s+/g, " ").trim();
  }
}

const state = {
  ort: null,
  tokenizer: null,
  encode: null,
  decode: null,
  ready: false,
};

function bannedIds(tok) {
  return new Set([tok.padId, tok.unkId, tok.toFr, tok.toNg, ...tok.domainIds]);
}

async function loadModel(onStatus) {
  onStatus("loading");
  const mod = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.wasm.min.mjs");
  const ort = mod.default || mod;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
  state.ort = ort;
  const [tokJson, encode, decode] = await Promise.all([
    fetch("model/tokenizer.json").then((r) => r.json()),
    ort.InferenceSession.create("model/encode.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("model/decode.onnx", { executionProviders: ["wasm"] }),
  ]);
  state.tokenizer = new BPETokenizer(tokJson);
  state.encode = encode;
  state.decode = decode;
  state.ready = true;
  onStatus("ready");
}

async function pivot(text, pair) {
  const url =
    "https://api.mymemory.translated.net/get?q=" +
    encodeURIComponent(text) +
    "&langpair=" +
    encodeURIComponent(pair);
  const res = await fetch(url);
  if (!res.ok) throw new Error("pivot");
  const data = await res.json();
  const out = data?.responseData?.translatedText;
  if (!out) throw new Error("pivot");
  return out;
}

async function translateCore(text, targetLang, maxLen = 96) {
  const tok = state.tokenizer;
  const ban = bannedIds(tok);
  const prepared = targetLang === "ng" ? normalizeFr(text) : normalizeNg(text);
  let ids = tok.encode(prepared, targetLang);
  if (ids.length > 160) {
    ids = ids.slice(0, 160);
    ids[ids.length - 1] = tok.eosId;
  }
  const { Tensor } = state.ort;
  const src = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const mem = (await state.encode.run({ src })).memory;
  let ys = [tok.bosId];
  for (let step = 0; step < maxLen; step++) {
    const tgt = new Tensor("int64", BigInt64Array.from(ys.map(BigInt)), [1, ys.length]);
    const logits = (await state.decode.run({ tgt, memory: mem, src })).logits;
    const data = logits.data;
    let best = -Infinity;
    let next = tok.eosId;
    for (let i = 0; i < data.length; i++) {
      if (ban.has(i)) continue;
      const v = data[i];
      if (v > best) {
        best = v;
        next = i;
      }
    }
    ys.push(next);
    if (next === tok.eosId) break;
  }
  return present(tok.decode(ys), targetLang);
}

async function translate(text, direction) {
  text = text.trim();
  if (!text) return { text: "", notes: [] };
  if (!state.ready) throw new Error("not ready");
  const notes = [];
  let source = text;
  if (direction === "en2ng") {
    source = await pivot(text, "en|fr");
    notes.push("en2ng");
  }
  const modelDir = direction === "ng2en" ? "ng2fr" : direction === "en2ng" ? "fr2ng" : direction;
  const target = modelDir === "fr2ng" ? "ng" : "fr";
  let result = await translateCore(source, target);
  if (direction === "ng2en") {
    result = await pivot(result, "fr|en");
    notes.push("ng2en");
  }
  return { text: result, notes };
}

export { loadModel, translate };
