import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import CENTRE_MONTPELLIER  # noqa: E402
from marche import (  # noqa: E402
    contour_hexagone, en_degres, en_metres, evolution_prix, filtrer, grille_prix, hexagone, prix_selon_pieces,
)

LAT0, LON0 = CENTRE_MONTPELLIER


def ventes(prix, latitude=LAT0, longitude=LON0, type_local="Appartement", annee=2024, pieces=2):
    """Une vente par prix, toutes au même endroit sauf indication contraire."""
    n = len(prix)
    return pd.DataFrame({
        "prix_m2": prix,
        "latitude": [latitude] * n,
        "longitude": [longitude] * n,
        "type_local": [type_local] * n,
        "date_mutation": pd.to_datetime([f"{annee}-06-01"] * n),
        "nombre_pieces_principales": [pieces] * n,
    })


def test_en_metres_puis_en_degres_revient_au_point_de_depart():
    x, y = en_metres(43.62, 3.90)
    lat, lon = en_degres(x, y)
    assert lat == pytest.approx(43.62) and lon == pytest.approx(3.90)


def test_hexagone_points_proches_meme_zone_points_eloignes_zones_differentes():
    q, r = hexagone([LAT0, LAT0 + 0.0005, LAT0 + 0.02], [LON0, LON0, LON0])  # 0, 55 m et 2,2 km au nord
    assert (q[0], r[0]) == (q[1], r[1]) == (0, 0)
    assert (q[2], r[2]) != (0, 0)


def test_chaque_point_tombe_dans_l_hexagone_le_plus_proche():
    # tirage de points : le centre de leur hexagone doit être à moins d'un rayon
    rng = np.random.default_rng(0)
    lat = LAT0 + rng.uniform(-0.03, 0.03, 500)
    lon = LON0 + rng.uniform(-0.03, 0.03, 500)
    q, r = hexagone(lat, lon, rayon=250)
    x, y = en_metres(lat, lon)
    cx, cy = 250 * np.sqrt(3) * (q + r / 2), 250 * 1.5 * r
    assert np.all(np.hypot(x - cx, y - cy) <= 250 + 1e-6)


def test_contour_hexagone_six_sommets_a_un_rayon_du_centre():
    sommets = contour_hexagone(0, 0, rayon=250)
    assert len(sommets) == 6
    x, y = en_metres([s[1] for s in sommets], [s[0] for s in sommets])
    assert np.hypot(x, y) == pytest.approx([250] * 6)


def test_grille_prix_mediane_et_minimum_de_ventes():
    fournie = ventes([3000] * 6 + [5000] * 5)  # 11 ventes : zone gardée, médiane 3 000
    maigre = ventes([9000] * 3, latitude=LAT0 + 0.02)  # 3 ventes : zone écartée
    zones = grille_prix(pd.concat([fournie, maigre]), min_ventes=10)
    assert len(zones) == 1
    assert zones.loc[0, "prix_m2"] == 3000 and zones.loc[0, "nb_ventes"] == 11
    assert len(zones.loc[0, "contour"]) == 6


def test_filtrer_par_type_et_annee():
    v = pd.concat([ventes([1], annee=2021), ventes([2], type_local="Maison", annee=2021), ventes([3], annee=2025)])
    assert list(filtrer(v)["prix_m2"]) == [1, 2, 3]
    assert list(filtrer(v, "Appartement")["prix_m2"]) == [1, 3]
    assert list(filtrer(v, "Tous", 2021)["prix_m2"]) == [1, 2]
    assert list(filtrer(v, "Maison", 2025)["prix_m2"]) == []


def test_evolution_prix_mediane_par_annee_et_type():
    v = pd.concat([ventes([3000, 4000, 8000], annee=2021), ventes([5000], annee=2022),
                   ventes([2000, 2200], type_local="Maison", annee=2021)])
    evolution = evolution_prix(v).set_index(["annee", "type_local"])
    assert evolution.loc[(2021, "Appartement"), "prix_m2"] == 4000
    assert evolution.loc[(2021, "Appartement"), "nb_ventes"] == 3
    assert evolution.loc[(2022, "Appartement"), "prix_m2"] == 5000
    assert evolution.loc[(2021, "Maison"), "prix_m2"] == 2100


def test_prix_selon_pieces_regroupe_les_grands_logements():
    v = pd.concat([ventes([1000, 2000, 3000], pieces=1), ventes([4000], pieces=6), ventes([6000], pieces=9)])
    stats = prix_selon_pieces(v, min_ventes=1).set_index("Pièces")
    assert list(stats.index) == ["1", "6 et +"]
    assert stats.loc["1", "mediane"] == 2000 and stats.loc["1", "nb_ventes"] == 3
    assert stats.loc["6 et +", "mediane"] == 5000 and stats.loc["6 et +", "nb_ventes"] == 2
    assert (stats["p10"] <= stats["p25"]).all() and (stats["p75"] <= stats["p90"]).all()


def test_prix_selon_pieces_ecarte_les_classes_trop_maigres():
    v = pd.concat([ventes([3000] * 10, pieces=2), ventes([5000] * 3, pieces=4)])
    assert list(prix_selon_pieces(v, min_ventes=10)["Pièces"]) == ["2"]
