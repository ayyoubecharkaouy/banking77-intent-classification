"""Prétraitement, embeddings de phrases et classification Banking77."""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def charger_variantes():
    with (BASE_DIR / "variantes.json").open(encoding="utf-8") as fichier:
        return json.load(fichier)


def nettoyer_texte(texte, variantes=None):
    """Normalisation déterministe ; ponctuation et négations conservées."""
    if not isinstance(texte, str):
        raise TypeError("Le texte doit être une chaîne de caractères.")
    texte = unicodedata.normalize("NFKC", texte).casefold()
    texte = texte.translate(str.maketrans({"’": "'", "‘": "'", "`": "'"}))
    texte = re.sub(r"\s+", " ", texte).strip()
    if variantes:
        motif = r"(?<!\w)(?:" + "|".join(
            re.escape(cle) for cle in sorted(variantes, key=len, reverse=True)
        ) + r")(?!\w)"
        texte = re.sub(motif, lambda correspondance: variantes[correspondance[0]], texte)
    return texte


class NormaliseurTexte(TransformerMixin, BaseEstimator):
    """Le dictionnaire est stocké dans le pipeline sauvegardé."""

    def __init__(self, variantes=None):
        self.variantes = variantes

    def fit(self, X, y=None):
        self.variantes_ = dict(self.variantes if self.variantes is not None else charger_variantes())
        return self

    def transform(self, X):
        check_is_fitted(self, "variantes_")
        return np.asarray([nettoyer_texte(texte, self.variantes_) for texte in X], dtype=object)


class RepresentationPhrases(TransformerMixin, BaseEstimator):
    """Embeddings LSA locaux ou Sentence-BERT ; combinaison lexicale facultative.

    LSA apprend un espace latent sur les seules phrases d'entraînement.
    Sentence-BERT utilise un encodeur préentraîné sans ajustement sur Banking77.
    """

    def __init__(self, mode="hybride", n_components=256, moteur="lsa", poids_embeddings=0.5):
        self.mode = mode
        self.n_components = n_components
        self.moteur = moteur
        self.poids_embeddings = poids_embeddings

    def fit(self, X, y=None):
        if self.mode not in {"tfidf", "embeddings", "hybride"}:
            raise ValueError(f"Représentation inconnue : {self.mode}")
        if self.moteur not in {"lsa", "sentencebert"}:
            raise ValueError(f"Moteur d’embeddings inconnu : {self.moteur}")
        self.tfidf_ = FeatureUnion([
            ("mots", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=60_000,
                sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
            )),
            ("caracteres", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                max_features=90_000, sublinear_tf=True, dtype=np.float32,
            )),
        ])
        lexical = self.tfidf_.fit_transform(X)
        if self.mode != "tfidf":
            if self.moteur == "lsa":
                dimensions = min(self.n_components, lexical.shape[0] - 1, lexical.shape[1] - 1)
                if dimensions < 2:
                    raise ValueError("Au moins trois phrases et trois caractéristiques sont nécessaires.")
                self.svd_ = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=RANDOM_STATE)
                self.svd_.fit(lexical)
                self.dimension_ = dimensions
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as erreur:
                    raise ImportError(
                        "L’option Sentence-BERT nécessite requirements-sentencebert.txt "
                        "et un premier téléchargement du modèle."
                    ) from erreur
                dossier = BASE_DIR / "artifacts" / "sentence_bert"
                source = str(dossier) if (dossier / "modules.json").is_file() else (
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.encodeur_ = SentenceTransformer(
                    source, device="cpu", cache_folder=str(BASE_DIR / ".cache" / "huggingface")
                )
                self.encodeur_.save(str(dossier))
                self.dimension_ = self.encodeur_.get_sentence_embedding_dimension()
        return self

    def _embeddings(self, X, lexical):
        if self.moteur == "lsa":
            return normalize(self.svd_.transform(lexical)).astype(np.float32)
        if not hasattr(self, "encodeur_"):
            from sentence_transformers import SentenceTransformer
            dossier = BASE_DIR / "artifacts" / "sentence_bert"
            if not (dossier / "modules.json").is_file():
                raise FileNotFoundError("Encodeur Sentence-BERT absent du dossier artifacts/sentence_bert.")
            self.encodeur_ = SentenceTransformer(str(dossier), device="cpu", local_files_only=True)
        return self.encodeur_.encode(
            list(X), batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def transform_modes(self, X):
        """Les trois représentations partagent le même vocabulaire appris."""
        check_is_fitted(self, "tfidf_")
        lexical = self.tfidf_.transform(X)
        dense = self._embeddings(X, lexical)
        hybride = sparse.hstack([
            lexical, sparse.csr_matrix(dense * self.poids_embeddings)
        ], format="csr", dtype=np.float32)
        return {"tfidf": lexical, "embeddings": dense, "hybride": hybride}

    def embeddings_phrases(self, X):
        """Projection de nouvelles phrases normalisées dans l'espace dense appris."""
        check_is_fitted(self, "dimension_")
        return self._embeddings(X, self.tfidf_.transform(X))

    def transform(self, X):
        check_is_fitted(self, "tfidf_")
        if self.mode == "tfidf":
            return self.tfidf_.transform(X)
        return self.transform_modes(X)[self.mode]

    def __getstate__(self):
        etat = self.__dict__.copy()
        etat.pop("encodeur_", None)
        return etat


def construire_modele(mode="hybride", C=1.0, moteur="lsa", n_components=256):
    return Pipeline([
        ("normalisation", NormaliseurTexte()),
        ("representation", RepresentationPhrases(mode=mode, moteur=moteur, n_components=n_components)),
        ("classifieur", LinearSVC(C=C, random_state=RANDOM_STATE, max_iter=10_000)),
    ])


def mesures(y_vrai, y_predit):
    return {
        "accuracy": float(accuracy_score(y_vrai, y_predit)),
        "precision_macro": float(precision_score(y_vrai, y_predit, average="macro", zero_division=0)),
        "rappel_macro": float(recall_score(y_vrai, y_predit, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_vrai, y_predit, average="macro", zero_division=0)),
    }


def preparer_donnees(dossier=BASE_DIR):
    """Conservation du test fourni ; dédoublonnage effectué sur l'entraînement."""
    train = pd.read_csv(Path(dossier) / "banking77_train.csv")
    test = pd.read_csv(Path(dossier) / "banking77_test.csv")
    variantes = charger_variantes()
    bilan = {"train_initial": len(train), "test_initial": len(test)}
    sources = {"train": train, "test": test}
    for nom, df in sources.items():
        if not {"text", "intention"}.issubset(df.columns):
            raise ValueError(f"Colonnes text et intention requises dans {nom}.")
        invalides = df[["text", "intention"]].isna().any(axis=1)
        invalides |= df["text"].fillna("").str.strip().eq("")
        invalides |= df["intention"].fillna("").str.strip().eq("")
        bilan[f"lignes_invalides_{nom}"] = int(invalides.sum())
        if nom == "test" and invalides.any():
            raise ValueError("Le test contient des lignes invalides ; évaluation interrompue.")
        df = df.loc[~invalides].copy()
        df["cle"] = df["text"].map(lambda t: nettoyer_texte(t, variantes))
        sources[nom] = df
    train, test = sources["train"], sources["test"]
    communs = train["cle"].isin(set(test["cle"]))
    bilan["recouvrements_train_test"] = int(communs.sum())
    train = train.loc[~communs].copy()
    conflits = train.groupby("cle")["intention"].nunique()
    ambigus = train["cle"].isin(conflits.loc[conflits.gt(1)].index)
    bilan["conflits_etiquettes_train"] = int(ambigus.sum())
    train = train.loc[~ambigus].copy()
    bilan["doublons_internes_train"] = int(train["cle"].duplicated().sum())
    train = train.drop_duplicates("cle").reset_index(drop=True)
    test = test.reset_index(drop=True)
    if set(train["intention"]) != set(test["intention"]):
        raise ValueError("Les classes d’entraînement et de test diffèrent.")
    bilan.update(train_final=len(train), test_final=len(test), classes=int(train["intention"].nunique()))
    return train.drop(columns="cle"), test.drop(columns="cle"), bilan


def comparer_modeles(X_train, y_train, X_val, y_val, moteur="lsa"):
    """Sélection sur validation ; aucun accès au jeu de test."""
    normaliseur = NormaliseurTexte().fit(X_train)
    textes_train = normaliseur.transform(X_train)
    textes_val = normaliseur.transform(X_val)
    representation = RepresentationPhrases(moteur=moteur).fit(textes_train)
    matrices_train = representation.transform_modes(textes_train)
    matrices_val = representation.transform_modes(textes_val)
    lignes = []
    for mode in ("tfidf", "embeddings", "hybride"):
        for C in (0.5, 1.0, 3.0):
            classifieur = LinearSVC(C=C, random_state=RANDOM_STATE, max_iter=10_000)
            classifieur.fit(matrices_train[mode], y_train)
            prediction = classifieur.predict(matrices_val[mode])
            lignes.append({"mode": mode, "C": C, **mesures(y_val, prediction)})
            print(f"Validation — {mode}, C={C} : F1 macro={lignes[-1]['f1_macro']:.4f}", flush=True)
    tableau = pd.DataFrame(lignes).sort_values(
        ["f1_macro", "accuracy", "C"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    # Le modèle livré inclut des embeddings ; TF-IDF seul est la référence lexicale.
    meilleur = tableau.loc[tableau["mode"].ne("tfidf")].iloc[0]
    selection = {"mode": str(meilleur["mode"]), "C": float(meilleur["C"]), "moteur": moteur}
    return tableau, selection


def predire_intention(texte, modele):
    if not isinstance(texte, str) or not re.search(r"[a-zA-Z]", texte):
        raise ValueError("Une question bancaire non vide en anglais est nécessaire.")
    return str(modele.predict([texte])[0])
