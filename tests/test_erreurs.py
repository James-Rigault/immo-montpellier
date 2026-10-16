import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from erreurs import (  # noqa: E402
    NOMS_TRANCHES, erreur_par_groupe, predictions_test, tranche_surface,
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


def test_predictions_test_retrouvent_le_rapport_2025():
    # mêmes ventes, même modèle, même calcul que src/final.py : la même erreur médiane que le rapport
    modele = joblib.load(RACINE / "models" / "lightgbm_final.joblib")
    ventes = pd.read_parquet(RACINE / "data" / "processed" / "ventes_features.parquet")
    test = predictions_test(modele, ventes)
    rapport = pd.read_csv(RACINE / "reports" / "resultats_test_2025.csv", index_col=0).loc["LightGBM final"]
    assert 100 * test["erreur"].median() == pytest.approx(rapport["Erreur médiane (%)"], abs=0.05)
    assert 100 * test["erreur"].mean() == pytest.approx(rapport["MAPE (%)"], abs=0.05)
