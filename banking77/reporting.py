"""Export des résultats effectivement calculés et du rapport de synthèse."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .model import mesures
from .paths import (
    BASE_DIR, DATA_DIR, EMBEDDINGS_DIR, METRICS_DIR, MODEL_PATH, PREDICTIONS_DIR,
    REPORT_PATH, RESPONSES_PATH, VARIANTS_PATH, preparer_dossiers_sortie,
)


def tableau_markdown(df):
    entete = "| " + " | ".join(map(str, df.columns)) + " |"
    separation = "| " + " | ".join("---" for _ in df.columns) + " |"
    lignes = ["| " + " | ".join(map(str, ligne)) + " |" for ligne in df.itertuples(index=False, name=None)]
    return "\n".join([entete, separation, *lignes])


def exporter_resultats(modele, train, test, comparaison, selection, bilan, embeddings, indices_train, indices_val):
    preparer_dossiers_sortie()
    predictions = modele.predict(test["text"])
    scores = mesures(test["intention"], predictions)
    classes = list(modele.classes_)
    reponses = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    if set(classes) != set(reponses):
        raise ValueError("Le dictionnaire de réponses doit couvrir exactement les classes du modèle.")
    rapport = pd.DataFrame(classification_report(
        test["intention"], predictions, output_dict=True, zero_division=0
    )).T
    resultats = test[["text", "intention"]].rename(columns={"intention": "vrai"}).copy()
    resultats["predit"] = predictions
    erreurs = resultats.loc[resultats["vrai"].ne(resultats["predit"])].copy()
    paires = erreurs.groupby(["vrai", "predit"]).size().rename("nombre").reset_index()
    paires = paires.sort_values(["nombre", "vrai", "predit"], ascending=[False, True, True])
    representation = modele.named_steps["representation"]
    versions = {nom: importlib.metadata.version(nom) for nom in (
        "numpy", "scipy", "pandas", "scikit-learn", "joblib", "streamlit"
    )}
    if selection["moteur"] == "sentencebert":
        versions["sentence-transformers"] = importlib.metadata.version("sentence-transformers")
    metadonnees = {
        "date_utc": datetime.now(timezone.utc).isoformat(), "random_state": 42,
        "donnees": bilan, "selection": selection, "scores_test": scores,
        "dimension_embeddings": int(representation.dimension_),
        "nombre_reponses": len(reponses), "versions": versions,
        "comparaison_validation": comparaison.to_dict(orient="records"),
        "sentencebert_execute": selection["moteur"] == "sentencebert",
        "sha256": {chemin.relative_to(BASE_DIR).as_posix(): hashlib.sha256(chemin.read_bytes()).hexdigest() for chemin in (
            DATA_DIR / "banking77_train.csv", DATA_DIR / "banking77_test.csv", VARIANTS_PATH, RESPONSES_PATH
        )},
    }
    if hasattr(representation, "svd_"):
        metadonnees["variance_expliquee_lsa"] = float(representation.svd_.explained_variance_ratio_.sum())
    comparaison.to_csv(METRICS_DIR / "comparaison_validation.csv", index=False)
    rapport.to_csv(METRICS_DIR / "rapport_classification.csv", index_label="intention")
    resultats.to_csv(PREDICTIONS_DIR / "predictions_test.csv", index=False)
    erreurs.to_csv(PREDICTIONS_DIR / "erreurs_test.csv", index=False)
    paires.to_csv(METRICS_DIR / "confusions_frequentes.csv", index=False)
    pd.DataFrame(confusion_matrix(test["intention"], predictions, labels=classes),
                 index=classes, columns=classes).to_csv(METRICS_DIR / "matrice_confusion.csv", index_label="vrai")
    np.savez_compressed(
        EMBEDDINGS_DIR / "embeddings_phrases.npz", train=embeddings["train"], test=embeddings["test"],
        y_train=train["intention"].to_numpy(dtype=str), y_test=test["intention"].to_numpy(dtype=str),
    )
    pd.DataFrame({"index_train_nettoye": indices_train}).to_csv(METRICS_DIR / "indices_entrainement.csv", index=False)
    pd.DataFrame({"index_train_nettoye": indices_val}).to_csv(METRICS_DIR / "indices_validation.csv", index=False)
    # Remplacement atomique après validation de la couverture des réponses.
    temporaire = MODEL_PATH.with_suffix(".tmp.joblib")
    joblib.dump(modele, temporaire, compress=3)
    temporaire.replace(MODEL_PATH)
    (METRICS_DIR / "metriques.json").write_text(json.dumps(metadonnees, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meilleurs = comparaison.loc[comparaison["mode"].eq("tfidf")].iloc[0]
    denses = comparaison.loc[comparaison["mode"].eq("embeddings")].iloc[0]
    retenu = comparaison.loc[(comparaison["mode"] == selection["mode"]) & (comparaison["C"] == selection["C"])].iloc[0]
    difficiles = rapport.loc[classes].sort_values("f1-score").head(8).reset_index(names="intention")
    difficiles = difficiles[["intention", "precision", "recall", "f1-score", "support"]].round(3)
    lignes_scores = pd.DataFrame([scores]).round(4)
    sens = "supérieure" if retenu.f1_macro >= meilleurs.f1_macro else "inférieure"
    premiere = paires.iloc[0] if len(paires) else None
    analyse = (
        f"La confusion la plus fréquente relie `{premiere.vrai}` à `{premiere.predit}` "
        f"({int(premiere.nombre)} cas). " if premiere is not None else "Aucune confusion observée. "
    )
    texte = f"""# Classification d’intentions bancaires — rapport final

## Objectif et données

Le système associe une question bancaire en anglais à l’une des {len(classes)} intentions Banking77, puis à une réponse préenregistrée de démonstration. Aucun texte bancaire n’est généré par le classifieur.

Les fichiers fournis contiennent {bilan['train_initial']} exemples d’entraînement et {bilan['test_initial']} exemples de test. Après normalisation, {bilan['recouvrements_train_test']} lignes communes au test, {bilan['conflits_etiquettes_train']} lignes aux étiquettes contradictoires et {bilan['doublons_internes_train']} doublons internes ont été retirés de l’entraînement. Les variantes d’une même question associées à plusieurs intentions sont toutes exclues plutôt qu’arbitrées. L’entraînement final contient {len(train)} lignes ; le test fourni reste inchangé ({len(test)} lignes).

![Répartition des effectifs par intention dans l’entraînement et le test](../artifacts/figures/distribution_classes.png)

*Figure 1 — Distribution du nombre de phrases par intention.* Les effectifs varient selon l’intention dans l’entraînement, tandis que le test est équilibré avec 40 phrases par intention. La F1 macro complète l’accuracy en donnant le même poids à chaque classe.

## Prétraitement et protocole

La normalisation applique Unicode NFKC, les minuscules, l’unification des apostrophes, la réduction des espaces et le dictionnaire contrôlé `banking77/resources/variantes.json`. Les contractions négatives sont développées sans suppression de la négation. La ponctuation est conservée. Aucun dictionnaire n’est appris sur le test.

La validation est une séparation stratifiée de 20 % de l’entraînement nettoyé, avec une graine de 42 : {len(indices_train)} exemples de travail et {len(indices_val)} de validation. Le vocabulaire, l’IDF et la projection LSA sont appris exclusivement sur l’entraînement de travail pendant la comparaison. Le contrôle préalable des doublons utilise uniquement le texte du test, sans ses étiquettes. Le choix des paramètres repose sur la F1 macro de validation ; le test n’intervient pas dans ce choix.

## Représentations et classification

La référence lexicale combine TF-IDF de mots (1–2 grammes) et de caractères (3–5 grammes). Les embeddings denses utilisent le moteur **{selection['moteur']}**, avec **{representation.dimension_} dimensions**. En mode LSA, la SVD tronquée capture une structure latente de cooccurrence propre au corpus ; il s’agit d’embeddings non contextuels, distincts de Sentence-BERT. Le mode hybride concatène les caractéristiques lexicales et les embeddings normalisés, pondérés par 0,5. Le classifieur est un SVM linéaire.

Les valeurs de C comparées sont 0,5, 1 et 3. Le modèle livré inclut des embeddings, conformément à l’objectif de représentation dense ; TF-IDF seul sert de référence. Le modèle retenu est **{selection['mode']}**, avec **C = {selection['C']}**. Sa F1 macro de validation ({retenu.f1_macro:.4f}) est {sens} à la meilleure référence TF-IDF ({meilleurs.f1_macro:.4f}).

{tableau_markdown(comparaison.round(4))}

Les embeddings seuls obtiennent une F1 macro de validation de {denses.f1_macro:.4f}, contre {meilleurs.f1_macro:.4f} pour la référence lexicale. La projection dense compresse les informations disponibles ; les termes et séquences de caractères précis restent utiles pour distinguer des intentions proches. La représentation retenue présente un écart de {(retenu.f1_macro - meilleurs.f1_macro) * 100:+.2f} point de F1 macro par rapport à la référence. Cette mesure repose sur une seule séparation de validation et ne démontre pas à elle seule un gain statistiquement significatif.

## Évaluation finale

Après sélection, le pipeline est réentraîné sur tout l’entraînement nettoyé. Le test contient 40 exemples par intention. L’accuracy et le rappel macro coïncident donc. La F1 macro donne le même poids à chacune des 77 intentions ; elle permet de compléter l’accuracy par une mesure tenant compte de la précision et du rappel par classe.

{tableau_markdown(lignes_scores)}

Le modèle commet **{len(erreurs)} erreurs sur {len(test)} questions**, soit **{len(erreurs)/len(test):.2%}**. Le rapport complet par classe est disponible dans `artifacts/metrics/rapport_classification.csv`.

### Intentions les plus difficiles

{tableau_markdown(difficiles)}

### Matrice de confusion

![Matrice de confusion normalisée sur les 77 intentions du test](../artifacts/figures/matrice_confusion.png)

*Figure 2 — Matrice de confusion normalisée par intention réelle.* Chaque ligne correspond à la classe réelle et chaque colonne à la classe prédite. La diagonale concentre les classifications correctes ; les cellules hors diagonale montrent les confusions entre intentions.

### Confusions observées

{tableau_markdown(paires.head(10))}

![Les dix paires d’intentions les plus fréquemment confondues sur le test](../artifacts/figures/confusions_frequentes.png)

*Figure 3 — Principales confusions, en nombre de questions.* Chaque barre représente une confusion de l’intention réelle vers l’intention prédite. Ce graphique détaille les erreurs les plus fréquentes, moins faciles à repérer dans la matrice complète à 77 classes.

{analyse}Les catégories concernant les virements en attente, les délais et les fonds non reçus partagent un vocabulaire proche. De même, une demande de carte physique, sa commande et sa livraison décrivent des étapes voisines. Les prédictions erronées exportées dans `artifacts/predictions/erreurs_test.csv` permettent une inspection phrase par phrase. Ces observations décrivent les erreurs ; elles ne servent pas à modifier les paramètres après l’évaluation finale.

## Application et livrables

L’interface Streamlit charge le pipeline complet, applique le même nettoyage qu’à l’entraînement et présente l’intention prédite avec une réponse spécifique. Le fichier `banking77/resources/reponses.json` couvre exactement les {len(classes)} intentions, y compris leurs étiquettes originales. Les réponses sont pédagogiques et ne constituent pas les procédures d’une banque réelle.

Le notebook contient l’exploration, la comparaison sur validation, la génération des embeddings, l’évaluation, les visualisations et une démonstration. Le dossier `artifacts/` contient le pipeline, les matrices d’embeddings avec leurs étiquettes, les indices de validation, les prédictions, les métriques, les rapports CSV et les figures. Les versions des bibliothèques, la graine et les empreintes SHA-256 des données sont consignées dans `artifacts/metrics/metriques.json`.

## Limites et prolongements

Les résultats portent sur des questions en anglais issues de Banking77. Le système impose une intention bancaire même pour une requête hors domaine : aucun rejet hors domaine ni score de probabilité calibré n’est implémenté. Les marges du SVM ne sont pas présentées comme des probabilités. Les formulations ambiguës et les intentions voisines restent une source d’erreurs. Les résultats ne constituent pas une estimation de performance sur d’autres langues ou d’autres banques.

Le mode LSA fonctionne sans téléchargement. L’option Sentence-BERT utilise `sentence-transformers/all-MiniLM-L6-v2` et nécessite les dépendances optionnelles ainsi qu’un premier accès réseau ; les poids sont ensuite conservés dans `artifacts/models/sentence_bert/`. Son état d’exécution dans cette expérience est **{metadonnees['sentencebert_execute']}**. Aucun score Sentence-BERT n’est annoncé sans exécution correspondante. Si la consigne exige spécifiquement un encodeur préentraîné, l’exécution de cette option reste nécessaire ; la livraison locale utilise des embeddings LSA.

## Références

- [Banking77 — publication et données originales](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) ; les fichiers locaux fournis servent de référence pour cette expérience.
- [Scikit-learn — analyse sémantique latente et SVD tronquée](https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis).
- [Sentence Transformers — génération d’embeddings](https://sbert.net/docs/sentence_transformer/usage/usage.html).
- [Fiche du modèle all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
"""
    REPORT_PATH.write_text(texte, encoding="utf-8")
    return metadonnees, rapport, resultats, paires
