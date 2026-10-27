import json
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from banking77 import (
    BASE_DIR, NormaliseurTexte, charger_variantes, construire_modele,
    nettoyer_texte, predire_intention, preparer_donnees,
)
from banking77.paths import EMBEDDINGS_DIR, METRICS_DIR, MODEL_PATH, NOTEBOOK_PATH, RESPONSES_PATH


class PretraitementTests(unittest.TestCase):
    def test_negations_unicode_et_limites_des_mots(self):
        variantes = charger_variantes()
        self.assertEqual(nettoyer_texte('  I CAN’T find my creditcard.\n', variantes),
                         'i can not find my credit card.')
        self.assertEqual(nettoyer_texte('The cash machines are not working!', variantes),
                         'the atms are not working!')
        self.assertEqual(nettoyer_texte('topup top-up topupper', variantes),
                         'top up top up topupper')

    def test_normalisation_idempotente_et_dictionnaire_embarque(self):
        normaliseur = NormaliseurTexte().fit(['A sentence'])
        texte = normaliseur.transform(["My top-up hasn't arrived."])[0]
        self.assertEqual(normaliseur.transform([texte])[0], texte)
        self.assertIn('not', texte.split())
        self.assertTrue(normaliseur.variantes_)
        for valeur in ('', '  ', '???', None):
            with self.subTest(valeur=valeur), self.assertRaises(ValueError):
                predire_intention(valeur, None)

    def test_donnees_sans_fuite_et_test_inchange(self):
        train, test, bilan = preparer_donnees()
        variantes = charger_variantes()
        train_normalise = train.text.map(lambda t: nettoyer_texte(t, variantes))
        test_normalise = test.text.map(lambda t: nettoyer_texte(t, variantes))
        self.assertFalse(set(train_normalise) & set(test_normalise))
        self.assertFalse(train_normalise.duplicated().any())
        self.assertEqual(len(test), bilan['test_initial'])
        self.assertEqual(train.intention.nunique(), 77)
        self.assertEqual(set(train.intention), set(test.intention))


class PipelineTests(unittest.TestCase):
    def test_ajustement_nouvelles_phrases_et_serialisation(self):
        textes = [
            'my card has not arrived', 'where is the delivery of my card',
            'the new card delivery is late', 'waiting for the new card',
            'a payment was charged twice', 'duplicate charge for my payment',
            'the payment appears twice', 'two charges for one payment',
        ]
        etiquettes = ['card_arrival'] * 4 + ['transaction_charged_twice'] * 4
        with threadpool_limits(limits=2):
            modele = construire_modele(n_components=3)
            self.assertEqual(clone(modele).get_params()['representation__n_components'], 3)
            modele.fit(textes, etiquettes)
            representation = modele.named_steps['representation']
            vocabulaire_avant = dict(representation.tfidf_.transformer_list[0][1].vocabulary_)
            normalises = modele.named_steps['normalisation'].transform(['an unseen card delivery question'])
            embeddings = representation.embeddings_phrases(normalises)
            self.assertEqual(embeddings.shape, (1, 3))
            self.assertTrue(np.isfinite(embeddings).all())
            self.assertAlmostEqual(float(np.linalg.norm(embeddings)), 1.0, places=5)
            self.assertEqual(vocabulaire_avant, representation.tfidf_.transformer_list[0][1].vocabulary_)
            with tempfile.TemporaryDirectory() as temporaire:
                chemin = Path(temporaire) / 'modele.joblib'
                joblib.dump(modele, chemin)
                recharge = joblib.load(chemin)
                np.testing.assert_array_equal(modele.predict(textes), recharge.predict(textes))


class LivraisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modele = joblib.load(MODEL_PATH)
        cls.reponses = json.loads(RESPONSES_PATH.read_text(encoding='utf-8'))

    def test_couverture_des_77_intentions_et_scores_exportes(self):
        self.assertEqual(set(self.modele.classes_), set(self.reponses))
        self.assertEqual(len(self.reponses), 77)
        self.assertTrue(all(texte.strip() for texte in self.reponses.values()))
        metriques = json.loads((METRICS_DIR / 'metriques.json').read_text())
        self.assertEqual(metriques['nombre_reponses'], 77)
        self.assertGreater(metriques['scores_test']['f1_macro'], 0.80)
        self.assertNotEqual(metriques['selection']['mode'], 'tfidf')

    def test_matrices_embeddings_exportees(self):
        with np.load(EMBEDDINGS_DIR / 'embeddings_phrases.npz', allow_pickle=False) as archive:
            self.assertEqual(archive['train'].shape[0], len(archive['y_train']))
            self.assertEqual(archive['test'].shape[0], len(archive['y_test']))
            self.assertEqual(archive['train'].shape[1], archive['test'].shape[1])
            self.assertTrue(np.isfinite(archive['test']).all())

    def test_donnees_et_modele_depuis_un_autre_dossier(self):
        dossier_initial = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporaire:
                os.chdir(temporaire)
                train, test, bilan = preparer_donnees()
                self.assertEqual(len(train), bilan['train_final'])
                self.assertEqual(len(test), bilan['test_final'])
                self.assertIn(predire_intention('My card has not arrived', self.modele), self.reponses)
        finally:
            os.chdir(dossier_initial)

    def test_notebook_execute_avec_figures_et_sans_erreurs(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        ids = [cellule['id'] for cellule in notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        cellules = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertTrue(all(c['execution_count'] is not None for c in cellules))
        sorties = [sortie for cellule in cellules for sortie in cellule['outputs']]
        self.assertFalse(any(sortie['output_type'] == 'error' for sortie in sorties))
        self.assertGreaterEqual(sum('image/png' in sortie.get('data', {}) for sortie in sorties), 3)

    def test_interface_question_vide_exemple_et_classification(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(BASE_DIR / 'app' / 'streamlit_app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == 'Question classification').click().run(timeout=30)
        self.assertTrue(app.warning)
        next(b for b in app.button if b.label == 'Example insertion').click().run(timeout=30)
        self.assertEqual(app.text_area[0].value, 'I have not received my new card yet')
        next(b for b in app.button if b.label == 'Question classification').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.success), 1)
        intention = predire_intention(app.text_area[0].value, self.modele)
        self.assertEqual(app.success[0].value, self.reponses[intention])


if __name__ == '__main__':
    unittest.main()
