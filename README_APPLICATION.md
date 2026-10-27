# Classification d’intentions bancaires — Banking77

Le projet couvre la préparation des données, un dictionnaire de variantes, des embeddings de phrases, un classifieur SVM, une évaluation détaillée et une application Streamlit avec 77 réponses préenregistrées. Les textes descriptifs et les réponses adoptent un ton général et impersonnel. Les questions d’exemple conservent leur formulation d’origine.

## Installation

Les commandes suivantes s’exécutent depuis le dossier du projet, avec Python 3.12. Les dépendances de `requirements.txt` correspondent aux versions de l’exécution locale. Un environnement `.venv` est déjà présent dans le dossier de travail.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Application

```bash
.venv/bin/python -m streamlit run app.py
```

L’application charge `artifacts/modele_intention_banking77.joblib` et `reponses.json`. La normalisation, les embeddings et le SVM sont contenus dans le pipeline sauvegardé. Le modèle fourni utilise des embeddings LSA et fonctionne sans téléchargement. Le catalogue couvre exactement les 77 étiquettes Banking77 ; les réponses servent à la démonstration et nécessitent une adaptation aux procédures d’une banque avant un usage réel.

## Exécution complète et reproductibilité

```bash
.venv/bin/python executer_notebook.py
```

Cette commande exécute toutes les cellules du notebook dans un nouvel interpréteur, conserve les sorties et régénère le modèle, le rapport et les exports. Elle s’arrête à la première erreur. Le notebook est également exécutable dans Jupyter depuis le dossier du projet avec les dépendances indiquées. Aucune saisie interactive n’interrompt l’exécution complète.

Le protocole utilise une validation stratifiée de 20 %, une graine de 42 et une sélection du paramètre C sur la F1 macro de validation. Les lignes d’entraînement communes au test, les textes aux étiquettes contradictoires et les doublons internes sont exclus après normalisation. Le test fourni reste inchangé. La représentation et le SVM sont ensuite réentraînés sur tout l’entraînement nettoyé avant l’évaluation finale.

La comparaison porte sur TF-IDF seul, les embeddings seuls et une représentation hybride. Le modèle livré est sélectionné parmi les représentations incluant des embeddings ; TF-IDF seul sert de référence. Les scores mesurés et l’interprétation figurent dans [RAPPORT_PROJET.md](RAPPORT_PROJET.md).

## Option Sentence-BERT

```bash
.venv/bin/python -m pip install -r requirements-sentencebert.txt
.venv/bin/python executer_notebook.py --embeddings sentencebert
```

Cette option utilise `sentence-transformers/all-MiniLM-L6-v2`. La première exécution nécessite un accès réseau pour les dépendances et les poids. L’encodeur est sauvegardé dans `artifacts/sentence_bert/` pour les prédictions locales suivantes. Un déplacement de cette variante du projet nécessite également ce dossier et les dépendances optionnelles. La commande régénère le modèle, les résultats et le rapport avec le moteur Sentence-BERT ; la commande sans option régénère la variante LSA.

L’exécution locale fournie repose sur **LSA**, une représentation dense non contextuelle apprise sur le corpus, distincte d’un encodeur préentraîné. Sentence-BERT reste une option sans score annoncé tant qu’elle n’a pas été exécutée. Son état d’exécution est consigné dans `artifacts/metriques.json`.

## Fichiers et livrables

| Fichier | Rôle |
| --- | --- |
| `Projet_Classification_Intentions_Banking77.ipynb` | Exploration, entraînement, embeddings, évaluation, graphiques et démonstration |
| `banking77.py` | Prétraitement, représentations et classification partagés |
| `variantes.json` | Dictionnaire contrôlé de normalisation |
| `reponses.json` | Réponses de démonstration pour les 77 intentions |
| `app.py` | Interface Streamlit |
| `executer_notebook.py` | Exécution du notebook depuis un état vierge |
| `rapport.py` | Export des résultats et génération du rapport |
| `RAPPORT_PROJET.md` | Rapport final avec les résultats réellement mesurés |
| `artifacts/modele_intention_banking77.joblib` | Pipeline complet sauvegardé |
| `artifacts/embeddings_phrases.npz` | Vecteurs denses train/test et étiquettes correspondantes |
| `artifacts/rapport_classification.csv` | Précision, rappel, F1 et effectifs par intention |
| `artifacts/comparaison_validation.csv` | Comparaison des représentations et paramètres |
| `artifacts/predictions_test.csv` | Toutes les prédictions du test |
| `artifacts/erreurs_test.csv` | Prédictions erronées |
| `artifacts/confusions_frequentes.csv` | Paires d’intentions confondues |
| `artifacts/matrice_confusion.csv` | Matrice de confusion brute avec étiquettes |
| `artifacts/*.png` | Figures de distribution et de confusion |
| `artifacts/indices_*.csv` | Indices de la séparation train/validation |
| `artifacts/metriques.json` | Scores, paramètres, versions et empreintes des fichiers |

## Vérifications

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Les vérifications portent sur le prétraitement, l’absence de recouvrement des données normalisées, les embeddings, le rechargement du modèle et le parcours de classification Streamlit.

## Limites

La classification est limitée à 77 intentions bancaires en anglais. Aucun mécanisme de rejet hors domaine ni probabilité calibrée n’est implémenté. Les réponses sont préenregistrées et ne constituent pas une assistance bancaire opérationnelle. Les limites méthodologiques et les références sont détaillées dans le rapport.
