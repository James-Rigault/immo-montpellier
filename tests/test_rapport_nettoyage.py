import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from clean import filtrer_aberrations  # noqa: E402
from rapport_nettoyage import ENTONNOIR, ETAPE_REGLE, entonnoir  # noqa: E402


def test_entonnoir_compte_les_lignes_retirees_et_ajoute_la_regle():
    tableau = entonnoir({"Lignes brutes": 100, "Ventes": 60, "Filtrées": 50}, lignes_avec_regle=45)
    assert list(tableau["retirees"]) == [0, 40, 10, 5]
    assert list(tableau["appliquee"]) == [True, True, True, False]
    assert tableau["etape"].iloc[-1] == ETAPE_REGLE


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


def test_fichier_entonnoir_coherent():
    # le fichier lu par l'app : des nombres qui ne font que baisser, et la fin = les données du modèle
    tableau = pd.read_csv(ENTONNOIR)
    assert tableau["lignes"].is_monotonic_decreasing
    appliquees = tableau[tableau["appliquee"]]
    ventes = pd.read_parquet(RACINE / "data" / "processed" / "ventes_features.parquet")
    assert appliquees["lignes"].iloc[-1] == len(ventes)
