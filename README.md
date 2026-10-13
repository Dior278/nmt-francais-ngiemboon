# Français ↔ ngiemboon

**English.** A small Transformer for French ↔ Ngiemboon (also spelled *ngienboum*, *guembon*), a Grassfields language spoken in Batcham (Bamboutos, West Region, Cameroon) and written in the General Alphabet of Cameroon Languages. The code builds a parallel corpus, trains on GPU, and translates in both directions. Source PDFs and the extracted sentence pairs are **not** in this repository: several texts are copyrighted and are not redistributed here.

Le ngiemboon est une langue bamiléké des Grassfields, parlée à l’ouest du Cameroun, notamment à **Batcham** (département des **Bamboutos**, région de l’**Ouest**), ainsi qu’à Balessing, Bangang et d’autres groupements de la Menoua. L’écriture suit l’**Alphabet général des langues camerounaises** (AGLC) : ŋ, ɛ, ɔ, ʉ, ẅ, ÿ, et les tons.

L’objectif est un outil de **préservation** et d’**échange**, et une base que d’autres peuvent reprendre. Le Nouveau Testament donne de la quantité ; le voice-over santé et le lexique SIL donnent de la langue de tous les jours. Plus on ajoute de paires (contes, dialogues, messages, homélies), plus le modèle sert hors de l’église.

Ce n’est **pas** un traducteur de production. Même avec Bible + santé + lexique, les sorties restent à relire avec un locuteur. Un mauvais ton change le sens (`má` « mère » ≠ `mà` « boubou », Anderson 2007).

## Ce qui est dans le dépôt

| Chemin | Rôle |
| --- | --- |
| `nmt/` | Extraction, entraînement, décodage, interface |
| `traduction_fr_ngiemboon.ipynb` | Démo et explications |
| `data/README.md` | Noms de fichiers attendus pour les PDF **locaux** |

Les PDF, le corpus extrait (`processed/`) et les poids (`checkpoints/`) sont ignorés par git. Le code est sous licence MIT. Les textes sources ne le sont pas : voir [Droits](#droits).

## Dépendances

Python 3.10+.

```powershell
pip install -r nmt/requirements.txt
```

`torch`, `pymupdf`, `customtkinter`.

Sur une machine avec GPU, installer **PyTorch CUDA** depuis [pytorch.org](https://pytorch.org) **avant** le fichier de dépendances, sinon `pip` peut poser un wheel CPU.

Vérifier : `python -c "import torch; print(torch.cuda.is_available())"`.

## Extraire le corpus

Mettre les PDF dans `data/` comme indiqué dans [`data/README.md`](data/README.md), puis :

```powershell
python -m nmt.corpus --force
```

Cela écrit `processed/pairs.jsonl` et `processed/stats.json`. Chaque ligne a les champs `id`, `book`, `ref`, `fr`, `ng`. **Ndiola 2011 est exclu par défaut** (`book` = `NDIOLA`). Pour l’inclure à part : `python -m nmt.corpus --force --include-adapted-ortho`.

## Entraîner

```powershell
python -m nmt.train --device auto --no-resume
```

`--no-resume` part de zéro. Avec CUDA, le script prend le profil GPU : 4 couches, `d_model=256`, vocabulaire ~8000, `max_len=160`, label smoothing, warmup, précision mixte, et une copie NG **sans tons** pour le sens ngiemboon → français.

Si CUDA manque de mémoire : `python -m nmt.train --device auto --no-resume --batch-size 16`.

Les poids vont dans `checkpoints/fr_ngiemboon.pt`, le tokenizer dans `processed/tokenizer.json`.

## Traduire

```powershell
python -m nmt.app
```

Français → ngiemboon, ngiemboon → français, clavier AGLC (ŋ ɛ ɔ ʉ ẅ ÿ et tons), exemples du corpus. L’interface n’entraîne pas. Sans checkpoint, elle affiche la commande d’entraînement.

En ligne de commande :

```powershell
python -m nmt.translate --dir fr2ng "Bonsoir, comment ça va ?"
```

Sens inverse. `má`, avec le ton haut, veut dire « mère ». Ce n’est pas la traduction de la phrase du dessus.

```powershell
python -m nmt.translate --dir ng2fr "má"
```

Sans tons à l’entrée ngiemboon, le programme normalise et prévient : l’ambiguïté reste.

## Ce que le pipeline extrait

| Source | Rôle pour le modèle |
| --- | --- |
| Bible Segond 1910 + NT ngiemboon 2007 | Versets alignés du Nouveau Testament, orthographe AGLC. `book` = code NT (`MAT`, …) |
| Timene 2024 (voice-over ACTU SANTÉ) | Tours de parole modernes (hôpital, virus, invitation). `book` = `HEALTH` |
| Anderson 2007 (précis d’orthographe SIL) | Glosses mot / français et phrases d’exemple. `book` = `SIL` |
| Lonfo (aperçu grammatical) | Glosses sûrs, dont `pɛ̀ Tsɔ́ɔn` = habitants de Batcham. `book` = `LONFO` |
| Ndiola Tsuata 2011 (syntaxe) | Exemples de clause. **Hors** entraînement par défaut (`book` = `NDIOLA`) |

Un ordre de grandeur, une fois les PDF en place : environ **8 000** paires par défaut. Ndiola reste hors du mélange.

Un seul Transformer apprend les deux sens : la source est préfixée par `<2ng>` ou `<2fr>`.

## Ajouter ses propres phrases

Une ligne JSON par paire, même format que `processed/pairs.jsonl` :

```json
{"id": "MOI.1", "book": "MOI", "ref": "phrase 1", "fr": "On se retrouve demain au marché.", "ng": "..."}
```

Puis : `python -m nmt.train --rebuild-corpus`.

## Droits

Le **code** de ce dépôt est sous [licence MIT](LICENSE).

Les textes ne sont pas redistribués ici :

- Segond 1910 : domaine public.
- NT ngiemboon : © Alliance biblique du Cameroun, 2007.
- Précis Anderson : SIL, 2007.
- Mémoires : droits des auteures (Ndiola Tsuata ; Timene Djouda ; Lonfo).

Usage prévu : étude et préservation, à partir de copies que chacun obtient de son côté. Pas une republication des PDF ni du corpus aligné.
