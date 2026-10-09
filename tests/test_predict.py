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