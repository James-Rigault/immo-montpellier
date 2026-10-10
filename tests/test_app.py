import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
import predict  # noqa: E402
from test_predict import FausseReponse  # noqa: E402

APP = str(RACINE / "app" / "streamlit_app.py")


def lancer_app():
    # l'app charge le modèle et les ventes au démarrage : on laisse plus que les 3 s par défaut
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.mark.parametrize("quartier, adresse", [
    ("Écusson", "10 rue de la Loge"),
    ("Port Marianne", "15 rue de la Cavalade"),
    ("Beaux-Arts", "10 rue Proudhon"),
])
def test_bouton_exemple_remplit_adresse(monkeypatch, quartier, adresse):
    # Pas d'appel réseau : le service d'adresses répond toujours la même suggestion
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))

    app = lancer_app()
    next(b for b in app.button if b.label == quartier).click().run()

    assert app.text_input(key="texte").value == adresse
    assert app.selectbox[0].value == adresse


def test_badges_lus_dans_le_rapport_2025():
    final = pd.read_csv(RACINE / "reports" / "resultats_test_2025.csv", index_col=0).loc["LightGBM final"]
    app = lancer_app()
    badges = next(m.value for m in app.markdown if 'class="badges"' in m.value)
    assert f"{final['Erreur médiane (%)']:.1f} %".replace(".", ",") in badges
    assert f"{final['Part à ±10 % (%)']:.0f} %" in badges
    assert f"{final['Couverture fourchette (%)'] / 10:.0f} sur 10" in badges


def test_estimation_affiche_carte_et_confiance(monkeypatch):
    adresse = "10 Rue de la Loge 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))

    app = lancer_app()
    next(b for b in app.button if b.label == "Écusson").click().run()
    next(b for b in app.button if b.label == "Estimer le prix").click().run()

    assert not app.exception
    html = "".join(m.value for m in app.markdown)
    assert 'class="carte"' in html and 'class="fourchette"' in html
    assert 'class="confiance"' in html
    graphiques = [str(json.loads(g.proto.spec)) for g in app.get("vega_lite_chart")]
    assert any("Effet sur le prix (k€)" in spec for spec in graphiques)


def test_trois_onglets_et_pied_de_page():
    app = lancer_app()
    assert not app.exception
    assert [onglet.label for onglet in app.tabs] == ["Estimer", "Le modèle", "À propos"]
    assert "Limites" in "".join(m.value for m in app.tabs[1].markdown)
    assert "data.gouv.fr" in "".join(m.value for m in app.tabs[2].markdown)
    # liens pas encore renseignés : ils ne doivent pas apparaître vides
    assert "]()" not in app.caption[-1].value
