import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict  # noqa: E402


class FausseReponse:
    """Imite la réponse du service de géocodage."""

    def __init__(self, donnees):
        self.donnees = donnees

    def raise_for_status(self):
        pass

    def json(self):
        return self.donnees


def resultat(score=0.9, citycode="34172"):
    return {"features": [{
        "geometry": {"coordinates": [3.88, 43.61]},
        "properties": {"score": score, "citycode": citycode, "label": "1 Rue Test 34000 Montpellier"},
    }]}


def test_geocoder_renvoie_latitude_longitude(monkeypatch):
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(resultat()))
    assert predict.geocoder("1 rue test") == (43.61, 3.88, "1 Rue Test 34000 Montpellier")


def test_geocoder_adresse_introuvable(monkeypatch):
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse({"features": []}))
    with pytest.raises(predict.AdresseIntrouvable):
        predict.geocoder("adresse qui n'existe pas")


def test_geocoder_refuse_une_autre_commune(monkeypatch):
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(resultat(citycode="34129")))
    with pytest.raises(predict.AdresseIntrouvable):
        predict.geocoder("1 rue test, Lattes")

def test_estimateur_retrouve_le_batiment_et_ses_variables():
    estimateur = predict.Estimateur()
    adresses = estimateur.adresses
    batiments = estimateur.batiments.set_index("batiment_groupe_id")
    # une adresse dont le bâtiment a une année et un DPE connus
    connus = batiments.dropna(subset=["annee_construction", "dpe_classe"]).index
    cle = adresses[adresses.isin(connus)].index[0]
    infos = estimateur.batiment(43.61, 3.88, cle)
    ligne = batiments.loc[adresses[cle]]
    assert infos["trouve"] and infos["methode"] == "adresse"
    assert infos["annee_construction"] == ligne["annee_construction"] and infos["dpe_classe"] == ligne["dpe_classe"]
    # année et DPE inconnus : des valeurs manquantes, que LightGBM sait traiter
    X = estimateur.variables(43.61, 3.88, "Appartement", 60, 3, 0, 0)
    assert X[["annee_construction", "dpe_classe"]].isna().all(axis=None)
    assert list(X.columns) == predict.VARIABLES
