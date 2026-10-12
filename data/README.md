# Sources locales

Ce dossier ne contient **pas** les PDF. Ils restent sur ta machine, et ils ne doivent pas être commités : plusieurs textes sont sous droit d’auteur.

Place ici les fichiers que tu as déjà le droit d’utiliser, avec exactement ces noms :

| Fichier | Rôle | Statut |
| --- | --- | --- |
| `Francais-Bible-Segond-1910-All-Bible.pdf` | Français, Segond 1910 (Bible entière ; seul le Nouveau Testament est aligné) | Domaine public |
| `Ngiemboon-Bible-New-Testasment.pdf` | Nouveau Testament ngiemboon, Alliance biblique du Cameroun, 2007 | Droit d’auteur |
| `Timene-2024-voiceover-sante-fr-ngiemboon.pdf` | Voice-over santé, tours de parole FR / NG | Droit des auteures |
| `precisdOrthographepourlaLangueNgiemboom.pdf` | Précis d’orthographe Anderson, SIL, 2007 | Droit d’auteur |
| `Lonfo-apercu-grammatical-ngiemboon.pdf` | Aperçu grammatical | Droit de l’auteure |
| `MemoireNdiolaTsuata.pdf` | Mémoire de syntaxe, 2011 | Droit de l’auteure ; **exclu** de l’entraînement par défaut |

Ensuite, depuis la racine du projet :

```powershell
python -m nmt.corpus --force
```

Cela écrit `processed/pairs.jsonl`, qui n’est pas versionné non plus : c’est une extraction des textes ci-dessus, pas une republication.
