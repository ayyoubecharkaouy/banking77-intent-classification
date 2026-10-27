# Banking77 — classification d’intentions bancaires

Pipeline NLP de classification de questions bancaires en anglais parmi **77 intentions**, avec une application de démonstration Streamlit et des réponses préenregistrées.

Le modèle combine des caractéristiques TF-IDF, des embeddings LSA de 256 dimensions et un SVM linéaire. Le projet comprend le nettoyage des données, un dictionnaire de variantes, une comparaison sur validation, une analyse des erreurs et des tests automatisés.

| Mesure sur le test fourni | Résultat |
| --- | --- |
| Accuracy | 90,84 % |
| F1 macro | 90,84 % |
| Questions évaluées | 3 080 |
| Intentions | 77 |

Les résultats détaillés, les graphiques et les limites figurent dans le [rapport final](docs/RAPPORT_PROJET.md).

## Organisation

```text
banking77/             Package Python partagé
├── model.py           Prétraitement, embeddings et classification
├── reporting.py       Export des résultats et génération du rapport
├── paths.py           Chemins centralisés
└── resources/         Dictionnaire de variantes et réponses
app/                   Application Streamlit
scripts/               Exécution complète du notebook
data/raw/              CSV d’origine, conservés sans modification
notebooks/             Exploration, entraînement et évaluation
docs/                  Rapport, guide d’utilisation et énoncé du projet
artifacts/
├── models/            Pipeline sauvegardé
├── embeddings/        Vecteurs denses et étiquettes
├── metrics/           Scores, rapports et indices de validation
├── predictions/       Prédictions et erreurs du test
└── figures/           Graphiques exportés
tests/                 Vérifications automatisées
```

## Installation et application

Les commandes suivantes s’exécutent depuis la racine du dépôt, avec Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Le modèle LSA fourni permet une utilisation locale sans téléchargement de poids préentraînés.

## Reproduction des résultats

```bash
python scripts/executer_notebook.py
```

Cette commande exécute le notebook depuis un état vierge et régénère le modèle, les résultats, les graphiques et le rapport. La validation est stratifiée, la graine est fixée à 42 et le test fourni reste inchangé.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Documentation et limites

- [Notebook exécuté](notebooks/Projet_Classification_Intentions_Banking77.ipynb)
- [Rapport final et graphiques](docs/RAPPORT_PROJET.md)
- [Guide d’utilisation et option Sentence-BERT](docs/UTILISATION.md)
- [Énoncé du projet](docs/images/DESCRIPTION-PROJET.png)
- [Données Banking77 originales](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
- [Licence du dépôt](LICENSE)

L’application utilise une classification fermée en anglais : aucun rejet hors domaine ni probabilité calibrée n’est implémenté. Les réponses sont pédagogiques. L’option Sentence-BERT est disponible, mais les résultats fournis correspondent uniquement au moteur LSA.
