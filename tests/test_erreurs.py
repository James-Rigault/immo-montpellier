import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from erreurs import (  # noqa: E402
    NOMS_TRANCHES, bilan_hors_marche, erreur_par_groupe, hors_marche, predictions_test, tranche_surface,
)


def test_tranche_surface_bornes_incluses_a_gauche():
    tranches = tranche_surface(pd.Series([9, 29.9, 30, 50, 79, 80, 120, 400]))
    assert list(tranches) == [NOMS_TRANCHES[0], NOMS_TRANCHES[0], NOMS_TRANCHES[1], NOMS_TRANCHES[2],
                              NOMS_TRANCHES[2], NOMS_TRANCHES[3], NOMS_TRANCHES[4], NOMS_TRANCHES[4]]


def test_erreur_par_groupe_mediane_en_pourcent():
    test = pd.DataFrame({"erreur": [0.1, 0.2, 0.3, 0.5], "type_local": ["A", "A", "A", "M"]})
    t = erreur_par_groupe(test, "type_local").set_index("groupe")
    assert t.loc["A", "erreur_mediane"] == pytest.approx(20) and t.loc["A", "nb_ventes"] == 3
    assert t.loc["M", "erreur_mediane"] == pytest.approx(50)


def ventes_exemple():
    """Six ventes au même endroit : une à 600 €/m² au milieu de ventes à 4 000 €/m²."""
    return pd.DataFrame({
        "code_postal": ["34000"] * 6,
        "type_local": ["Appartement"] * 6,
        "date_mutation": pd.to_datetime(["2025-03-01"] * 6),
        "prix_m2": [4000, 4100, 3900, 4000, 600, 4050],
        "adresse_numero": [1, 2, 3, 4, 5, 6],
        "adresse_nom_voie": ["RUE A"] * 6,
    })


def test_hors_marche_meme_regle_que_le_nettoyage():
    assert list(hors_marche(ventes_exemple())) == [False, False, False, False, True, False]


def test_bilan_hors_marche():
    ventes = ventes_exemple()
    test = ventes.assign(erreur=[0.1, 0.05, 0.1, 0.2, 5.0, 0.1])
    bilan = bilan_hors_marche(ventes, test, nb_pires=2)
    assert bilan["nb"] == 1 and bilan["nb_test"] == 1 and bilan["prix_m2_median"] == 600
    assert bilan["erreur_mediane_hors_marche"] == pytest.approx(500)
    assert bilan["nb_parmi_pires"] == 1
    assert bilan["erreur_moyenne_sans"] < bilan["erreur_moyenne"]


def test_predictions_test_retrouvent_le_rapport_2025():
    # mêmes ventes, même modèle, même calcul que src/final.py : la même erreur médiane que le rapport
    modele = joblib.load(RACINE / "models" / "lightgbm_final.joblib")
    ventes = pd.read_parquet(RACINE / "data" / "processed" / "ventes_features.parquet")
    test = predictions_test(modele, ventes)
    rapport = pd.read_csv(RACINE / "reports" / "resultats_test_2025.csv", index_col=0).loc["LightGBM final"]
    assert 100 * test["erreur"].median() == pytest.approx(rapport["Erreur médiane (%)"], abs=0.05)
    assert 100 * test["erreur"].mean() == pytest.approx(rapport["MAPE (%)"], abs=0.05)
