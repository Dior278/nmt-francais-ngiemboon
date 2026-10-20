# Local sources

This folder does **not** contain the PDFs. They stay on your machine, and they must not be committed: several texts are copyrighted.

**Français.** Les PDF ne sont pas dans le dépôt. Pose ici les fichiers que tu as le droit d’utiliser, avec exactement les noms du tableau.

Put the files you already have the right to use here, with exactly these names:

| File | Role | Status |
| --- | --- | --- |
| `Francais-Bible-Segond-1910-All-Bible.pdf` | French, Segond 1910 (whole Bible; only the New Testament is aligned) | Public domain |
| `Ngiemboon-Bible-New-Testasment.pdf` | Ngiemboon New Testament, Bible Society of Cameroon, 2007 | Copyright |
| `Timene-2024-voiceover-sante-fr-ngiemboon.pdf` | Health voice-over, French / Ngiemboon turns | Authors’ rights |
| `precisdOrthographepourlaLangueNgiemboom.pdf` | Anderson orthography sketch, SIL, 2007 | Copyright |
| `Lonfo-apercu-grammatical-ngiemboon.pdf` | Grammar sketch | Author’s rights |
| `MemoireNdiolaTsuata.pdf` | Syntax thesis, 2011 | Author’s rights; **left out** of default training |

Then, from the project root:

```powershell
python -m nmt.corpus --force
```

That writes `processed/pairs.jsonl`, which is not versioned either: it is an extraction of the texts above, not a republication.
