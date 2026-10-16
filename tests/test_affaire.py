import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from affaire import SEUILS_VERDICT, analyser_prix, niveau_verdict, position_comparables  # noqa: E402

# estimation 250 000 €, fourchette 200 000 – 300 000 € (largeur 100 000 €)
PRIX, BAS, HAUT = 250_000, 200_000, 300_000


@pytest.mark.parametrize("demande, attendu", [
    (150_000, "En dessous du marché"),
    (199_999, "En dessous du marché"),
    (200_000, "Dans la fourchette"),  # les bornes font partie de la fourchette
    (300_000, "Dans la fourchette"),
    (330_000, "Au-dessus du marché"),
    (350_000, "Au-dessus du marché"),  # exactement une demi-largeur au-dessus : pas encore « nettement »
    (350_001, "Nettement au-dessus"),
    (500_000, "Nettement au-dessus"),
])
def test_verdict_selon_le_prix_demande(demande, attendu):
    assert analyser_prix(demande, PRIX, BAS, HAUT, "moyenne")["verdict"] == attendu


def test_seuil_defini_dans_une_constante():
    assert niveau_verdict(340_000, BAS, HAUT, seuils={"nettement_au_dessus": 0.3}) == "nettement"
    assert SEUILS_VERDICT["nettement_au_dessus"] == 0.5


def test_ecart_en_euros_et_en_pourcent():
    analyse = analyser_prix(275_000, PRIX, BAS, HAUT, "élevée")
    assert analyse["ecart"] == 25_000 and analyse["ecart_pct"] == pytest.approx(10)
    assert analyser_prix(225_000, PRIX, BAS, HAUT, "élevée")["ecart_pct"] == pytest.approx(-10)


def test_confiance_faible_rend_le_verdict_prudent():
    analyse = analyser_prix(500_000, PRIX, BAS, HAUT, "faible")
    assert analyse["prudent"] and analyse["verdict"] == "Probablement au-dessus du marché"
    assert not analyser_prix(500_000, PRIX, BAS, HAUT, "élevée")["prudent"]


def test_position_parmi_les_comparables():
    assert position_comparables(4_000, [3_000, 3_500, 3_900, 4_100, 5_000]) == (3, 5)
    assert position_comparables(2_000, [3_000, 3_500]) == (0, 2)
