# French, English ↔ Ngiemboon

**Français.** Traduction automatique entre le français, l’anglais et le ngiemboon (aussi écrit *ngienboum*, *guembon*), langue bamiléké parlée à Batcham (Bamboutos, Ouest, Cameroun). Le mode d’emploi en français est dans la section [Français](#français). L’anglais passe par le français.

**Français.** Si la préservation du ngiemboon vous intéresse, ou si vous avez des phrases, des dialogues ou d’autres données qui pourraient entrer dans l’entraînement du modèle, [Me contacter](https://www.linkedin.com/in/pavel-wadoh-tchinda-b9357a328).

**English.** A small Transformer for French ↔ Ngiemboon, plus English ↔ Ngiemboon through French. Ngiemboon is a Grassfields language spoken in Batcham (Bamboutos, West Region, Cameroon) and written in the General Alphabet of Cameroon Languages. The code builds a parallel corpus, trains, and translates. Source PDFs and the extracted sentence pairs are **not** in this repository: several texts are copyrighted and are not redistributed here.

The aim is a tool for **keeping the language in use**, and a base other people can pick up. The New Testament gives quantity. The health voice-over and the SIL lexicon give everyday language. More pairs (stories, dialogue, messages) make the model useful outside church language.

This is **not** a production translator. Even with the Bible, health text, and lexicon, a speaker should check the output. A wrong tone changes the meaning (`má` “mother” ≠ `mà` “robe”, Anderson 2007).

**English.** If you care about keeping Ngiemboon in use, or if you have sentences, dialogue, or other data that could be used to train the model, [Contact me](https://www.linkedin.com/in/pavel-wadoh-tchinda-b9357a328).

## English

### What is in the repository

| Path | Role |
| --- | --- |
| `nmt/` | Extraction, training, decoding, window |
| `traduction_fr_ngiemboon.ipynb` | Walkthrough. The file name stays French so it is easy to find. |
| `data/README.md` | Expected file names for the **local** PDFs |

PDFs, the extracted corpus (`processed/`), and the weights (`checkpoints/`) are gitignored. The code is MIT. The source texts are not: see [Rights](#rights).

### Install

Python 3.10+.

```powershell
pip install -r nmt/requirements.txt
```

`torch`, `pymupdf`, `customtkinter`, `transformers`, `sentencepiece`.

On a GPU machine, install **CUDA PyTorch** from [pytorch.org](https://pytorch.org) **before** the requirements file, or `pip` may install a CPU wheel.

Check: `python -c "import torch; print(torch.cuda.is_available())"`.

### Extract the corpus

Put the PDFs in `data/` as listed in [`data/README.md`](data/README.md), then:

```powershell
python -m nmt.corpus --force
```

That writes `processed/pairs.jsonl` and `processed/stats.json`. Each line has `id`, `book`, `ref`, `fr`, `ng`. **Ndiola 2011 is left out by default** (`book` = `NDIOLA`). To include it anyway: `python -m nmt.corpus --force --include-adapted-ortho`.

### Train

```powershell
python -m nmt train
```

Training is French ↔ Ngiemboon only. English is not a third trained language. At translation time, English is turned into French with Opus-MT, then the French ↔ Ngiemboon model runs. The other way comes back through French.

Without CUDA, the profile is the one for this laptop: **3 layers**, `d_model=192`, 6 heads, vocabulary around 5000, dropout 0.1, a toneless copy of the Ngiemboon sentences, and the Noam learning-rate schedule. Input and output embeddings are tied. That size is on purpose: with about 8,000 pairs, a wider network memorizes the New Testament instead of generalizing.

Training stops when validation no longer improves (`patience`). The best point is `checkpoints/fr_ngiemboon.pt`. The tokenizer is `processed/tokenizer.json`. The same command continues from the next epoch and keeps the learning-rate schedule. `--no-resume` starts over.

On a GPU, the same script uses the wider profile (4 layers, `d_model=256`, dropout 0.2, label smoothing 0.1) and mixed precision. If CUDA runs out of memory: `python -m nmt train --batch-size 16`.

The first English sentence downloads two Marian models, about 300 MB each (`Helsinki-NLP/opus-mt-en-fr` and `opus-mt-fr-en`).

### Translate

One sentence, French to Ngiemboon:

```powershell
python -m nmt "Bonsoir, comment ça va ?"
```

English to Ngiemboon, and Ngiemboon back to English or French:

```powershell
python -m nmt --from en "Good evening, how are you?"
python -m nmt --to en "má"
python -m nmt --to fr "má"
```

`má`, with the high tone, means “mother”. It is not the translation of the French sentence above.

With no argument, a prompt waits for sentences (`/fr-ng`, `/en-ng`, `/ng-fr`, `/ng-en`, `/q` to quit):

```powershell
python -m nmt
```

On the Ngiemboon side, AGLC (Anderson 2007) treats tone as its own sign, the same one on every vowel: ´ high, ` low, ˇ rising, ˆ falling. `á` and `ɛ́` carry the same high tone. ŋ ɛ ɔ ʉ stay letters, and the diaeresis on ẅ and ÿ is not a tone. On the French side, é è ê à stay letters. If the Ngiemboon input has no tone, or the Ngiemboon output has none, the program says so.

Window, with an AGLC keyboard and an English / French switch:

```powershell
python -m nmt app
```

Ctrl+Enter translates.

### What the pipeline reads

| Source | Role |
| --- | --- |
| Segond 1910 + Ngiemboon NT 2007 | Aligned New Testament verses, AGLC spelling. `book` = NT code (`MAT`, …) |
| Timene 2024 (ACTU SANTÉ voice-over) | Modern turns (hospital, virus, invitation). `book` = `HEALTH` |
| Anderson 2007 (SIL orthography) | Word glosses and example sentences. `book` = `SIL` |
| Lonfo (grammar sketch) | Glossed examples, including `pɛ̀ Tsɔ́ɔn` = people of Batcham. `book` = `LONFO` |
| Ndiola Tsuata 2011 (syntax) | Clause examples. **Out** of the default training mix (`book` = `NDIOLA`) |

Once the PDFs are in place, the default mix is on the order of **8,000** pairs.

One Transformer learns both French directions. The source is prefixed with `<2ng>` or `<2fr>`.

### Add your own sentences

One JSON line per pair, the same shape as `processed/pairs.jsonl`:

```json
{"id": "MOI.1", "book": "MOI", "ref": "sentence 1", "fr": "On se retrouve demain au marché.", "ng": "..."}
```

Then: `python -m nmt train --rebuild-corpus`.

There is no English field. An English sentence is translated through French at use time. A direct English–Ngiemboon model would need English sentences aligned with Ngiemboon, which this corpus does not have.

### Rights

The **code** in this repository is [MIT](LICENSE).

The texts are not redistributed here:

- Segond 1910: public domain.
- Ngiemboon NT: © Alliance biblique du Cameroun, 2007.
- Anderson orthography sketch: SIL, 2007.
- Theses: the authors’ rights (Ndiola Tsuata; Timene Djouda; Lonfo).

Intended use: study and language keeping, from copies each person obtains on their own. Not a republication of the PDFs or of the aligned corpus.

## Français

Le ngiemboon (guembon, ngienboum) se parle à **Batcham**, dans les **Bamboutos**, à l’ouest du Cameroun. L’écriture suit l’**Alphabet général des langues camerounaises** : ŋ, ɛ, ɔ, ʉ, ẅ, ÿ, et les tons. `má` veut dire « mère ». `mà` veut dire « boubou ».

Ce n’est pas un traducteur fini. Les sorties se relisent avec un locuteur.

Installer, puis extraire et entraîner (le réseau apprend le français et le ngiemboon, pas l’anglais) :

```powershell
pip install -r nmt/requirements.txt
python -m nmt.corpus --force
python -m nmt train
```

Traduire :

```powershell
python -m nmt "Bonsoir, comment ça va ?"
python -m nmt --from en "Good evening"
python -m nmt --to fr "má"
python -m nmt --to en "má"
python -m nmt app
```

L’anglais est d’abord passé en français, puis le modèle français ↔ ngiemboon fait le reste. La première phrase anglaise télécharge deux modèles d’environ 300 Mo. Ctrl+Entrée traduit dans la fenêtre. Le bouton **Français / English** change la langue de l’interface.

Les PDF restent sur la machine. Leurs noms sont dans [`data/README.md`](data/README.md). Ndiola 2011 n’entre pas dans l’entraînement par défaut.
