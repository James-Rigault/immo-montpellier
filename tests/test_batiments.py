import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from batiments import (  # noqa: E402
    annee_par_batiment, classe_dpe_avant_vente, cle_ban, code_dpe, dpe_par_batiment, en_lambert, lettre_dpe,
    relier_ventes, taux_correspondance,
)


def test_cle_ban_depuis_les_colonnes_dvf():
    cles = cle_ban(["7", "12", None], ["0570", "A123", "0570"], [None, "B", None])
    assert list(cles[:2]) == ["34172_0570_00007", "34172_a123_00012_bis"]
    assert pd.isna(cles[2])  # pas de numéro : pas de clé


def test_lettre_et_code_dpe():
    assert [code_dpe(c) for c in "ADG"] == [1, 4, 7] and code_dpe("Inconnu") is None
    assert lettre_dpe(3) == "C" and lettre_dpe(3.5) == "D"  # entre deux classes : la moins bonne
    assert lettre_dpe(None) is None and lettre_dpe(np.nan) is None


def exemple():
    """Trois bâtiments, deux sur la même parcelle P1, et quatre ventes."""
    x, y = en_lambert([43.61, 43.61, 43.62], [3.87, 3.8702, 3.88])
    batiments = pd.DataFrame({
        "batiment_groupe_id": ["b1", "b2", "b3"], "x": x, "y": y,
        "annee_construction": [1900, 1975, np.nan], "nb_log": [4, 30, 1],
    })
    parcelles = pd.DataFrame({"batiment_groupe_id": ["b1", "b2"], "parcelle_id": ["P1", "P1"]})
    adresses = pd.DataFrame({"batiment_groupe_id": ["b1", "b3"],
                             "cle_interop_adr": ["34172_0570_00007", "34172_0001_00002"]})
    ventes = pd.DataFrame({
        "id_parcelle": ["P1", "P1", "P9", "P9"],
        "adresse_numero": ["7", "8", "2", None],
        "adresse_code_voie": ["0570", "0570", "0001", None],
        "adresse_suffixe": [None] * 4,
        "latitude": [43.61, 43.61, 43.62, 43.62],
        "longitude": [3.87, 3.87, 3.88, 3.88],
        "date_mutation": pd.to_datetime(["2023-06-01"] * 4),
    })
    return ventes, batiments, parcelles, adresses


def test_relier_ventes_parcelle_puis_adresse_puis_proximite():
    ventes, batiments, parcelles, adresses = exemple()
    lien = relier_ventes(ventes, batiments, parcelles, adresses)
    # parcelle à deux bâtiments : celui qui porte l'adresse (b1), sinon le plus grand (b2)
    assert list(lien["batiment_groupe_id"]) == ["b1", "b2", "b3", "b3"]
    assert list(lien["methode_batiment"]) == ["parcelle", "parcelle", "adresse", "proximité"]


def test_proximite_refuse_un_batiment_trop_loin():
    ventes, batiments, parcelles, adresses = exemple()
    loin = ventes.iloc[[3]].assign(latitude=43.70)  # environ 9 km : aucun bâtiment à moins de 50 m
    lien = relier_ventes(loin, batiments, parcelles, adresses)
    assert lien["batiment_groupe_id"].isna().all() and lien["methode_batiment"].isna().all()


def dpe_exemple():
    return pd.DataFrame({
        "numero_dpe": ["d1", "d2", "d3", "d4"],
        "date_etablissement_dpe": ["2022-01-10", "2023-02-01", "2023-06-01", "2024-01-01"],
        "etiquette_dpe": ["C", "E", "A", "A"],
        "identifiant_ban": ["x", "x", "x", "34172_0570_00007"],
        "annee_construction": [1890, None, None, None],
    })


def test_dpe_par_batiment_lien_officiel_puis_adresse():
    _, _, _, adresses = exemple()
    rel = pd.DataFrame({"batiment_groupe_id": ["b1", "b1", "b1"], "identifiant_dpe": ["d1", "d2", "d3"]})
    dpe = dpe_par_batiment(dpe_exemple(), rel, adresses)
    assert list(dpe["batiment_groupe_id"]) == ["b1"] * 4  # d4 : retrouvé par son adresse BAN
    assert list(dpe["classe"]) == [3, 5, 1, 1]


def test_classe_dpe_sans_fuite_de_donnees():
    # vente le 2023-06-01 : seuls d1 (C) et d2 (E) sont antérieurs ; d3 le jour même et d4 après sont ignorés
    ventes, *_, adresses = exemple()
    rel = pd.DataFrame({"batiment_groupe_id": ["b1", "b1", "b1"], "identifiant_dpe": ["d1", "d2", "d3"]})
    dpe = dpe_par_batiment(dpe_exemple(), rel, adresses)
    batiment = pd.Series(["b1", "b2", None, None], index=ventes.index)
    resultat = classe_dpe_avant_vente(ventes, batiment, dpe)
    assert resultat.loc[0, "dpe_classe"] == 4 and resultat.loc[0, "nb_dpe"] == 2  # médiane de C (3) et E (5)
    assert resultat.loc[1:, "dpe_classe"].isna().all()  # pas de DPE connu : la valeur reste manquante


def test_annee_fichiers_fonciers_puis_dpe():
    _, batiments, _, _ = exemple()
    dpe = pd.DataFrame({"batiment_groupe_id": ["b3", "b3", "b1"], "annee_construction": [1960, 1962, 2050]})
    annees = annee_par_batiment(batiments, dpe)
    assert annees["b1"] == 1900 and annees["b2"] == 1975
    assert annees["b3"] == pytest.approx(1961)  # inconnue des Fichiers fonciers : médiane des DPE


def test_taux_correspondance_par_annee():
    ventes = pd.DataFrame({
        "date_mutation": pd.to_datetime(["2022-01-01", "2022-06-01", "2023-01-01"]),
        "methode_batiment": ["parcelle", "adresse", None],
        "annee_construction": [1900, np.nan, 1950], "dpe_classe": [3, np.nan, np.nan],
    })
    taux = taux_correspondance(ventes)
    assert taux.loc[2022, "parcelle"] == 0.5 and taux.loc[2023, "sans_batiment"] == 1
    assert taux.loc["Toutes", "avec_annee"] == pytest.approx(0.667, abs=1e-3)


def test_fichiers_de_l_app_et_du_modele():
    # données versionnées : les ventes ont leurs nouvelles variables, la table de l'app est cohérente
    ventes = pd.read_parquet(RACINE / "data" / "processed" / "ventes_features.parquet")
    assert {"annee_construction", "dpe_classe", "methode_batiment"} <= set(ventes.columns)
    assert ventes["dpe_classe"].dropna().between(1, 7).all()
    batiments = pd.read_parquet(RACINE / "data" / "processed" / "batiments.parquet")
    assert batiments["batiment_groupe_id"].is_unique
    assert ventes["batiment_groupe_id"].dropna().isin(batiments["batiment_groupe_id"]).all()
