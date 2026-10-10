import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
import predict  # noqa: E402
from test_predict import FausseReponse  # noqa: E402

APP = str(RACINE / "app" / "streamlit_app.py")


@pytest.mark.parametrize("quartier, adresse", [
    ("Écusson", "10 rue de la Loge"),
    ("Port Marianne", "15 rue de la Cavalade"),
    ("Beaux-Arts", "10 rue Proudhon"),
])
def test_bouton_exemple_remplit_adresse(monkeypatch, quartier, adresse):
    # Pas d'appel réseau : le service d'adresses répond toujours la même suggestion
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))

    app = AppTest.from_file(APP).run()
    next(b for b in app.button if b.label == quartier).click().run()

    assert app.text_input(key="texte").value == adresse
    assert app.selectbox[0].value == adresse
