"""Pipeline de classification d’intentions bancaires.

Les classes restent accessibles depuis ``banking77`` afin de permettre le
chargement des modèles sauvegardés avant l’organisation en package.
"""

from .model import (
    RANDOM_STATE,
    NormaliseurTexte,
    RepresentationPhrases,
    charger_variantes,
    comparer_modeles,
    construire_modele,
    mesures,
    nettoyer_texte,
    predire_intention,
    preparer_donnees,
)
from .paths import BASE_DIR

__all__ = [
    "BASE_DIR", "RANDOM_STATE", "NormaliseurTexte", "RepresentationPhrases",
    "charger_variantes", "comparer_modeles", "construire_modele", "mesures",
    "nettoyer_texte", "predire_intention", "preparer_donnees",
]
