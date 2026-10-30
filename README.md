# French, English ↔ Ngiemboon

**Français.** Traduction automatique entre le français, l’anglais et le ngiemboon (aussi écrit *ngienboum*, *guembon*), langue bamiléké parlée à Batcham (Bamboutos, Ouest, Cameroun). Le mode d’emploi en français est dans la section [Français](#français). L’anglais passe par le français.

**Français.** Si la préservation du ngiemboon vous intéresse, ou si vous avez des phrases, des dialogues ou d’autres données qui pourraient entrer dans l’entraînement du modèle, [Me contacter](https://www.linkedin.com/in/pavel-wadoh-tchinda-b9357a328).

**English.** A small Transformer for French ↔ Ngiemboon, and for English ↔ Ngiemboon through French. Ngiemboon is a Grassfields language spoken in Batcham (Bamboutos, West Region, Cameroon).

This is **not** a production translator. A speaker should check the output. A wrong tone changes the meaning (`má` “mother” ≠ `mà` “robe”).

**English.** If you care about keeping Ngiemboon in use, or if you have sentences, dialogue, or other data that could be used to train the model, [Contact me](https://www.linkedin.com/in/pavel-wadoh-tchinda-b9357a328).

## English

### Install

Python 3.10+.

```powershell
pip install -r nmt/requirements.txt
```

With a GPU, install PyTorch for CUDA from [pytorch.org](https://pytorch.org) before that file.

### Translate

On the [public page](https://dior278.github.io/nmt-francais-ngiemboon/), type a sentence. The model runs in the browser.

On a computer:

```powershell
python -m nmt "Bonsoir, comment ça va ?"
python -m nmt --from en "Good evening, how are you?"
python -m nmt --to en "má"
python -m nmt --to fr "má"
python -m nmt app
```

`má` means “mother”. It is not the translation of the French sentence above.

### Train

The source texts are not in this repository. Several are under copyright. File names for a local copy are in [`data/README.md`](data/README.md).

```powershell
python -m nmt.corpus --force
python -m nmt train
```

Training is French ↔ Ngiemboon. English is translated through French.

### Rights

The code is [MIT](LICENSE). The source texts are not included. Segond 1910 is in the public domain. The other texts stay with their rights holders.

## Français

Le ngiemboon (guembon, ngienboum) se parle à **Batcham**, dans les **Bamboutos**, à l’ouest du Cameroun. `má` veut dire « mère ». `mà` veut dire « boubou ».

Ce n’est pas un traducteur fini. Une sortie se relit avec un locuteur.

Les textes sources ne sont pas dans le dépôt. Leurs noms de fichiers sont dans [`data/README.md`](data/README.md).

```powershell
pip install -r nmt/requirements.txt
python -m nmt.corpus --force
python -m nmt train
```

Sur la [page publique](https://dior278.github.io/nmt-francais-ngiemboon/), écrire une phrase. Le modèle tourne dans le navigateur.

Sur un ordinateur :

```powershell
python -m nmt "Bonsoir, comment ça va ?"
python -m nmt --from en "Good evening"
python -m nmt --to fr "má"
python -m nmt --to en "má"
python -m nmt app
```

L’anglais passe par le français.
