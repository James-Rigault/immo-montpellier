"""Nettoie les données DVF de Montpellier : une ligne = la vente d'un seul logement."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SORTIE = PROCESSED_DIR / "ventes_clean.parquet"

LOGEMENTS = ["Appartement", "Maison"]
SURFACE_MIN, SURFACE_MAX = 9, 400  # m² : en dessous ou au-dessus, ce sont presque toujours des erreurs
QUANTILE_BAS, QUANTILE_HAUT = 0.01, 0.99  # on retire le 1 % le moins cher et le 1 % le plus cher au m²
SEUIL_RELATIF = 0.3  # une vente à moins de 30 % du prix au m² médian local est jugée anormale

def charger_brut(raw_dir=RAW_DIR):
    """Charge et assemble tous les fichiers DVF téléchargés."""
    fichiers = sorted(Path(raw_dir).glob("dvf_*.csv"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier DVF dans {raw_dir}. Lance d'abord src/ingest.py")
    df = pd.concat(
        [pd.read_csv(f, low_memory=False, dtype={"code_postal": str, "code_commune": str, "adresse_code_voie": str, "id_parcelle": str}) for f in fichiers],
        ignore_index=True,
    )
    df["date_mutation"] = pd.to_datetime(df["date_mutation"])
    return df


def garder_ventes(df):
    """Garde uniquement les ventes (pas les échanges, adjudications, expropriations...)."""
    return df[df["nature_mutation"] == "Vente"].copy()


def dedoublonner_locaux(df):
    """Un même local peut apparaître sur plusieurs lignes (une par type de terrain) : on n'en garde qu'une."""
    locaux = df.dropna(subset=["type_local"])
    cles = ["id_mutation", "id_parcelle", "type_local", "surface_reelle_bati",
            "nombre_pieces_principales", "lot1_numero"]
    return locaux.drop_duplicates(subset=cles)


def resumer_mutations(df, locaux):
    """Pour chaque vente : nombre de logements, de dépendances, d'autres locaux, et surface de terrain."""
    comptes = (
        locaux.assign(
            nb_logements=locaux["type_local"].isin(LOGEMENTS),
            nb_dependances=locaux["type_local"].eq("Dépendance"),
            nb_autres=~locaux["type_local"].isin(LOGEMENTS + ["Dépendance"]),
        )
        .groupby("id_mutation")[["nb_logements", "nb_dependances", "nb_autres"]]
        .sum()
    )
    terrain = (
        df.dropna(subset=["surface_terrain"])
        .drop_duplicates(subset=["id_mutation", "id_parcelle", "nature_culture", "surface_terrain"])
        .groupby("id_mutation")["surface_terrain"]
        .sum()
        .rename("surface_terrain_totale")
    )
    return comptes.join(terrain, how="left").fillna({"surface_terrain_totale": 0})


def construire_ventes(locaux, resume):
    """Garde les ventes d'un seul logement (dépendances autorisées, pas de local commercial)."""
    simples = resume[(resume["nb_logements"] == 1) & (resume["nb_autres"] == 0)].index
    logements = locaux[locaux["type_local"].isin(LOGEMENTS) & locaux["id_mutation"].isin(simples)]
    # parcelle, code de voie et suffixe servent à retrouver le bâtiment de la vente (src/batiments.py)
    colonnes = ["id_mutation", "date_mutation", "valeur_fonciere", "type_local", "surface_reelle_bati",
                "nombre_pieces_principales", "lot1_surface_carrez", "adresse_numero", "adresse_suffixe",
                "adresse_nom_voie", "adresse_code_voie", "id_parcelle", "code_postal", "longitude", "latitude"]
    ventes = logements[colonnes].merge(
        resume[["nb_dependances", "surface_terrain_totale"]], left_on="id_mutation", right_index=True
    )
    return ventes.reset_index(drop=True)

def mediane_locale(v):
    """Prix au m² médian des ventes du même code postal, du même type de bien et de la même année."""
    annee = v["date_mutation"].dt.year
    return v.groupby([v["code_postal"], v["type_local"], annee])["prix_m2"].transform("median")


def retirer_prix_anormaux(v, seuil=SEUIL_RELATIF):
    """Retire les ventes très en dessous du prix local : viagers, ventes partielles, ventes familiales..."""
    return v[v["prix_m2"] >= seuil * mediane_locale(v)]


def compter(etapes, nom, v):
    """Note le nombre de lignes restantes après une étape, si on tient le compte."""
    if etapes is not None:
        etapes[nom] = len(v)


def preparer(ventes, etapes=None):
    """Retire les données manquantes et les surfaces impossibles, puis calcule le prix au m²."""
    v = ventes.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"])
    compter(etapes, "Données complètes", v)
    v = v[v["surface_reelle_bati"].between(SURFACE_MIN, SURFACE_MAX)]
    compter(etapes, f"Surfaces de {SURFACE_MIN} à {SURFACE_MAX} m²", v)
    v = v[v["nombre_pieces_principales"] >= 1]
    compter(etapes, "Au moins une pièce", v)
    return v.assign(prix_m2=v["valeur_fonciere"] / v["surface_reelle_bati"])


def filtrer_aberrations(ventes, etapes=None, seuil=SEUIL_RELATIF):
    """Retire les données manquantes, les surfaces impossibles et les prix au m² extrêmes.

    Si `etapes` est un dictionnaire, on y ajoute le nombre de ventes restantes après chaque filtre.
    `seuil=None` saute la règle des ventes hors marché (c'est ainsi qu'ont été produites les données du modèle actuel).
    """
    v = preparer(ventes, etapes)
    if seuil is not None:
        v = retirer_prix_anormaux(v, seuil)
        compter(etapes, "Sans ventes hors marché", v)
    # bornes calculées séparément pour les appartements et les maisons
    bornes = v.groupby("type_local")["prix_m2"].quantile([QUANTILE_BAS, QUANTILE_HAUT]).unstack()
    bas = v["type_local"].map(bornes[QUANTILE_BAS])
    haut = v["type_local"].map(bornes[QUANTILE_HAUT])
    v = v[v["prix_m2"].between(bas, haut)].reset_index(drop=True)
    compter(etapes, "Sans les 1 % de prix extrêmes", v)  # 1 % le moins cher et 1 % le plus cher
    return v


def nettoyer(df, seuil=SEUIL_RELATIF):
    """Enchaîne toutes les étapes et compte ce qui reste après chacune."""
    etapes = {"Lignes brutes": len(df)}
    ventes_brutes = garder_ventes(df)
    etapes["Ventes (mutations)"] = ventes_brutes["id_mutation"].nunique()
    locaux = dedoublonner_locaux(ventes_brutes)
    resume = resumer_mutations(ventes_brutes, locaux)
    ventes = construire_ventes(locaux, resume)
    etapes["Ventes d'un seul logement"] = len(ventes)
    propres = filtrer_aberrations(ventes, etapes, seuil)
    return propres, etapes

def test_retirer_prix_anormaux():
    """Médiane locale = 3 900 €/m², seuil = 30 % = 1 170 €/m² : la vente à 600 €/m² est retirée."""
    v = pd.DataFrame({
        "code_postal": ["34000"] * 4,
        "type_local": ["Appartement"] * 4,
        "date_mutation": pd.to_datetime(["2024-01-01"] * 4),
        "prix_m2": [4000, 4200, 3800, 600],
    })
    gardees = retirer_prix_anormaux(v)
    assert list(gardees["prix_m2"]) == [4000, 4200, 3800]

def main():
    df = charger_brut()
    propres, etapes = nettoyer(df)
    for nom, n in etapes.items():
        print(f"{nom:<28} {n:>8}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    propres.to_parquet(SORTIE, index=False)
    print(f"Fichier écrit : {SORTIE}")


if __name__ == "__main__":
    main()