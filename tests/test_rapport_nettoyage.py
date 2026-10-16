import json
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from clean import filtrer_aberrations  # noqa: E402
from rapport_nettoyage import ENTONNOIR, HORS_MARCHE, entonnoir, ventes_hors_marche  # noqa: E402


def test_entonnoir_compte_les_lignes_retirees():
    tableau = entonnoir({"Lignes brutes": 100, "Ventes": 60, "Filtrées": 50})
    assert list(tableau["retirees"]) == [0, 40, 10]
    assert tableau["appliquee"].all()


def test_ventes_hors_marche_resume():
    v = pd.DataFrame({
        "code_postal": ["34000"] * 5, "type_local": ["Appartement"] * 5,
        "date_mutation": pd.to_datetime(["2024-01-01"] * 5),
        "prix_m2": [4000, 4100, 3900, 600, 600],
        "adresse_numero": [1, 2, 3, 9, 9], "adresse_nom_voie": ["RUE A"] * 5,
    })
    resume = ventes_hors_marche(v)
    assert resume["ventes_retirees"] == 2 and resume["prix_m2_median_retirees"] == 600
    assert resume["plus_grand_groupe"] == 2  # deux ventes à la même adresse, au même prix au m²


def test_filtrer_aberrations_sans_la_regle_garde_les_ventes_hors_marche():
    # 95 ventes à 4 000 €/m² et 5 à 600 €/m² : assez de ventes basses pour que le filtre du 1 % ne les retire pas
    n = 100
    ventes = pd.DataFrame({
        "valeur_fonciere": [200_000.0] * 95 + [30_000.0] * 5,
        "surface_reelle_bati": [50.0] * n,
        "nombre_pieces_principales": [2] * n,
        "latitude": [43.6] * n, "longitude": [3.88] * n,
        "code_postal": ["34000"] * n, "type_local": ["Appartement"] * n,
        "date_mutation": pd.to_datetime(["2024-01-01"] * n),
    })
    etapes = {}
    avec = filtrer_aberrations(ventes, etapes)
    sans = filtrer_aberrations(ventes, seuil=None)
    assert 600 not in set(avec["prix_m2"]) and 600 in set(sans["prix_m2"])
    assert etapes["Sans ventes hors marché"] == 95


def test_fichiers_du_rapport_coherents():
    # les fichiers lus par l'app : des nombres qui ne font que baisser, la fin = les données du modèle,
    # et la règle retire autant de ventes dans l'entonnoir que dans le résumé
    tableau = pd.read_csv(ENTONNOIR)
    assert tableau["lignes"].is_monotonic_decreasing and tableau["appliquee"].all()
    ventes = pd.read_parquet(RACINE / "data" / "processed" / "ventes_features.parquet")
    assert tableau["lignes"].iloc[-1] == len(ventes)
    resume = json.loads(HORS_MARCHE.read_text())
    assert tableau.set_index("etape").loc["Sans ventes hors marché", "retirees"] == resume["ventes_retirees"]
