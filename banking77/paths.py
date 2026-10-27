"""Chemins du projet, indépendants du dossier courant d’exécution."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
DOCS_DIR = BASE_DIR / "docs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "modele_intention_banking77.joblib"
SENTENCE_BERT_DIR = MODELS_DIR / "sentence_bert"
VARIANTS_PATH = RESOURCES_DIR / "variantes.json"
RESPONSES_PATH = RESOURCES_DIR / "reponses.json"
REPORT_PATH = DOCS_DIR / "RAPPORT_PROJET.md"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "Projet_Classification_Intentions_Banking77.ipynb"


def preparer_dossiers_sortie():
    for dossier in (MODELS_DIR, EMBEDDINGS_DIR, METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR, DOCS_DIR):
        dossier.mkdir(parents=True, exist_ok=True)
