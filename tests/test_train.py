import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import baseline, evaluer  # noqa: E402


def test_evaluer():
    mesures = evaluer([100, 200], [110, 180])
    assert mesures["MAE (€)"] == pytest.approx(15)
    assert mesures["MAPE (%)"] == pytest.approx(10)
    assert mesures["Part à ±10 % (%)"] == pytest.approx(100)


def test_baseline_utilise_la_mediane_du_train():
    train = pd.DataFrame({
        "code_postal": ["34000", "34000"],
        "type_local": ["Appartement", "Appartement"],
        "prix_m2": [3000, 5000],
    })
    autre = pd.DataFrame({
        "code_postal": ["34000", "99999"],
        "type_local": ["Appartement", "Appartement"],
        "surface_reelle_bati": [50, 10],
    })
    prix = baseline(train, autre)
    assert list(prix) == [200_000, 40_000]  # 4 000 €/m² × 50 m² ; code inconnu : médiane globale