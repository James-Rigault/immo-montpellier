"""Relie chaque vente à son bâtiment : année de construction (BDNB) et classe DPE (ADEME).

Sources, toutes deux ouvertes :
- BDNB, export open data du département 34 (millésime 2026-02.b, format v0.7.12) :
  batiment_groupe_ffo_bat.annee_construction (« Année de construction du bâtiment », Fichiers fonciers),
  rel_batiment_groupe_parcelle (bâtiment ↔ parcelle cadastrale), rel_batiment_groupe_adresse
  (bâtiment ↔ adresse BAN) et batiment_groupe (géométrie, en Lambert-93).
- ADEME, « DPE Logements existants (depuis juillet 2021) », jeu dpe03existant :
  date_etablissement_dpe, etiquette_dpe, identifiant_ban, coordonnées BAN en Lambert-93.

Les fichiers téléchargés sont lourds : ils restent dans data/external/ (ignoré par git).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RACINE = Path(__file__).resolve().parents[1]
BDNB_DIR = RACINE / "data" / "external" / "bdnb"
DPE_DIR = RACINE / "data" / "external" / "dpe"
BDNB_ZIP = BDNB_DIR / "open_data_millesime_2026-02-b_dep34_csv.zip"
URL_BDNB = ("https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2026-02-b/millesime_2026-02-b_dep34/"
            "open_data_millesime_2026-02-b_dep34_csv.zip")
URL_DPE = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
CODE_COMMUNE = "34172"  # Montpellier

COLONNES_DPE = [
    "numero_dpe", "date_etablissement_dpe", "etiquette_dpe", "type_batiment", "annee_construction",
    "identifiant_ban", "coordonnee_cartographique_x_ban", "coordonnee_cartographique_y_ban",
]
CLASSES_DPE = list("ABCDEFG")  # codées de 1 (A, le meilleur) à 7 (G)
DISTANCE_MAX = 50  # mètres : au-delà, on préfère « bâtiment inconnu » à un mauvais bâtiment


def lettre_dpe(classe):
    """Code DPE (1 à 7, parfois médiane comme 3,5) → lettre ; une médiane entre deux classes prend la moins bonne."""
    if classe is None or pd.isna(classe):
        return None
    return CLASSES_DPE[min(int(classe + 0.5), len(CLASSES_DPE)) - 1]


def code_dpe(lettre):
    """Lettre DPE (A à G) → code 1 à 7 ; toute autre valeur (« Inconnu ») → None."""
    return CLASSES_DPE.index(lettre) + 1 if lettre in CLASSES_DPE else None


def telecharger_bdnb():
    """Télécharge l'export BDNB du département une seule fois (environ 640 Mo)."""
    if BDNB_ZIP.exists():
        return BDNB_ZIP
    BDNB_DIR.mkdir(parents=True, exist_ok=True)
    with requests.get(URL_BDNB, stream=True, timeout=60) as reponse:
        reponse.raise_for_status()
        with open(BDNB_ZIP, "wb") as f:
            for morceau in reponse.iter_content(chunk_size=1 << 20):
                f.write(morceau)
    return BDNB_ZIP


def telecharger_dpe():
    """Tous les DPE de Montpellier depuis l'API de l'ADEME, page par page, gardés en cache."""
    fichier = DPE_DIR / f"dpe_{CODE_COMMUNE}.parquet"
    if fichier.exists():
        return pd.read_parquet(fichier)
    lignes, url = [], URL_DPE
    params = {"qs": f"code_insee_ban:{CODE_COMMUNE}", "size": 10_000, "select": ",".join(COLONNES_DPE)}
    while url:
        reponse = requests.get(url, params=params, timeout=120)
        reponse.raise_for_status()
        page = reponse.json()
        lignes += page["results"]
        url, params = page.get("next"), None  # le lien « next » contient déjà les paramètres
    dpe = pd.DataFrame(lignes).reindex(columns=COLONNES_DPE)
    DPE_DIR.mkdir(parents=True, exist_ok=True)
    dpe.to_parquet(fichier, index=False)
    return dpe


def lire_table(nom, colonnes, zip_bdnb=BDNB_ZIP):
    """Lit une table CSV de l'export BDNB (séparateur « ; »), en ne gardant que quelques colonnes."""
    with zipfile.ZipFile(zip_bdnb) as archive, archive.open(f"csv/{nom}.csv") as f:
        return pd.read_csv(f, sep=";", usecols=colonnes, dtype=str)


def extraire_bdnb():
    """Les tables utiles, réduites aux bâtiments de Montpellier, gardées en cache (quelques Mo)."""
    fichiers = {nom: BDNB_DIR / f"montpellier_{nom}.parquet" for nom in ["batiments", "parcelles", "adresses", "dpe"]}
    if all(f.exists() for f in fichiers.values()):
        return {nom: pd.read_parquet(f) for nom, f in fichiers.items()}
    telecharger_bdnb()
    import shapely  # seulement ici : l'app n'a pas besoin de relire les géométries

    groupes = lire_table("batiment_groupe", ["batiment_groupe_id", "code_commune_insee", "geom_groupe"])
    groupes = groupes[groupes["code_commune_insee"] == CODE_COMMUNE]
    centres = shapely.centroid(shapely.from_wkt(groupes["geom_groupe"].to_numpy()))  # Lambert-93, en mètres
    batiments = pd.DataFrame({
        "batiment_groupe_id": groupes["batiment_groupe_id"].to_numpy(),
        "x": shapely.get_x(centres), "y": shapely.get_y(centres),
    })
    ffo = lire_table("batiment_groupe_ffo_bat", ["batiment_groupe_id", "annee_construction", "nb_log"])
    batiments = batiments.merge(ffo, on="batiment_groupe_id", how="left")
    batiments["annee_construction"] = pd.to_numeric(batiments["annee_construction"], errors="coerce")
    batiments["nb_log"] = pd.to_numeric(batiments["nb_log"], errors="coerce")

    ids = set(batiments["batiment_groupe_id"])
    tables = {
        "batiments": batiments,
        "parcelles": lire_table("rel_batiment_groupe_parcelle", ["batiment_groupe_id", "parcelle_id"]),
        "adresses": lire_table("rel_batiment_groupe_adresse", ["batiment_groupe_id", "cle_interop_adr"]),
        "dpe": lire_table("rel_batiment_groupe_dpe_logement", ["batiment_groupe_id", "identifiant_dpe"]),
    }
    for nom in ["parcelles", "adresses", "dpe"]:
        tables[nom] = tables[nom][tables[nom]["batiment_groupe_id"].isin(ids)].reset_index(drop=True)
    for nom, table in tables.items():
        table.to_parquet(fichiers[nom], index=False)
    return tables


SUFFIXES = {"B": "bis", "T": "ter", "Q": "quater", "C": "quinquies"}  # suffixes DVF → suffixes BAN


def cle_ban(numero, code_voie, suffixe=None, commune=CODE_COMMUNE):
    """Clé d'interopérabilité BAN d'une adresse DVF : 34172_0570_00007 (_bis, _ter...)."""
    numero = pd.to_numeric(pd.Series(numero), errors="coerce")
    code_voie = pd.Series(code_voie).astype("string").str.lower()
    cle = commune + "_" + code_voie + "_" + numero.astype("Int64").astype("string").str.zfill(5)
    if suffixe is not None:
        suite = pd.Series(suffixe).map(SUFFIXES)
        cle = cle.where(suite.isna(), cle + "_" + suite)
    return cle.where(numero.notna() & code_voie.notna())


def en_lambert(latitude, longitude):
    """Coordonnées GPS → Lambert-93 (mètres), le système des données BDNB et ADEME."""
    from pyproj import Transformer
    transformeur = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    return transformeur.transform(np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float))


def plus_proche(x, y, batiments, distance_max=DISTANCE_MAX):
    """Bâtiment dont le centre est le plus proche de chaque point, s'il est à moins de distance_max mètres."""
    from scipy.spatial import cKDTree
    arbre = cKDTree(batiments[["x", "y"]].to_numpy())
    distances, rangs = arbre.query(np.column_stack([x, y]))
    ids = batiments["batiment_groupe_id"].to_numpy()[rangs]
    return pd.Series(np.where(distances <= distance_max, ids, None), dtype="object")


def relier_ventes(ventes, batiments, parcelles, adresses):
    """Bâtiment de chaque vente et méthode utilisée : parcelle, puis adresse, puis proximité (50 m au plus).

    Sur une parcelle à plusieurs bâtiments, on prend celui qui porte l'adresse de la vente,
    sinon celui qui compte le plus de logements.
    """
    v = pd.DataFrame({
        "vente": ventes.index,
        "parcelle_id": ventes["id_parcelle"].to_numpy(),
        "cle": cle_ban(ventes["adresse_numero"].to_numpy(), ventes["adresse_code_voie"].to_numpy(),
                       ventes["adresse_suffixe"].to_numpy()).to_numpy(),
    })
    taille = batiments[["batiment_groupe_id", "nb_log"]]
    porte_adresse = adresses.rename(columns={"cle_interop_adr": "cle"}).assign(meme_adresse=True)

    par_parcelle = (
        v.merge(parcelles, on="parcelle_id")
        .merge(porte_adresse, on=["batiment_groupe_id", "cle"], how="left")
        .merge(taille, on="batiment_groupe_id", how="left")
        .fillna({"meme_adresse": False, "nb_log": 0})
        .sort_values(["vente", "meme_adresse", "nb_log"], ascending=[True, False, False])
        .drop_duplicates("vente")
        .set_index("vente")["batiment_groupe_id"]
    )
    par_adresse = (
        v.merge(porte_adresse, on="cle").merge(taille, on="batiment_groupe_id", how="left")
        .sort_values(["vente", "nb_log"], ascending=[True, False])
        .drop_duplicates("vente").set_index("vente")["batiment_groupe_id"]
    )
    x, y = en_lambert(ventes["latitude"], ventes["longitude"])
    par_proximite = pd.Series(plus_proche(x, y, batiments).to_numpy(), index=ventes.index)

    resultat = pd.DataFrame(index=ventes.index)
    resultat["batiment_groupe_id"] = par_parcelle.reindex(ventes.index)
    resultat["methode_batiment"] = np.where(resultat["batiment_groupe_id"].notna(), "parcelle", None)
    for methode, serie in [("adresse", par_adresse.reindex(ventes.index)), ("proximité", par_proximite)]:
        manque = resultat["batiment_groupe_id"].isna() & serie.notna()
        resultat.loc[manque, "batiment_groupe_id"] = serie[manque]
        resultat.loc[manque, "methode_batiment"] = methode
    return resultat


def dpe_par_batiment(dpe_ademe, rel_dpe, adresses):
    """Chaque DPE de l'ADEME avec son bâtiment : lien officiel de la BDNB, sinon par l'adresse BAN."""
    dpe = dpe_ademe.assign(
        date_dpe=pd.to_datetime(dpe_ademe["date_etablissement_dpe"]),
        classe=dpe_ademe["etiquette_dpe"].map({c: i for i, c in enumerate(CLASSES_DPE, start=1)}),
    )
    officiel = rel_dpe.rename(columns={"identifiant_dpe": "numero_dpe"}).drop_duplicates("numero_dpe")
    par_adresse = (adresses.rename(columns={"cle_interop_adr": "identifiant_ban"})
                   .drop_duplicates("identifiant_ban"))
    dpe = dpe.merge(officiel, on="numero_dpe", how="left")
    secours = dpe["identifiant_ban"].map(par_adresse.set_index("identifiant_ban")["batiment_groupe_id"])
    dpe["batiment_groupe_id"] = dpe["batiment_groupe_id"].fillna(secours)
    return dpe.dropna(subset=["batiment_groupe_id", "classe"])


def classe_dpe_avant_vente(ventes, batiment, dpe):
    """Classe DPE médiane (1 = A ... 7 = G) des DPE du bâtiment établis AVANT la vente.

    Un DPE établi le jour de la vente ou après est ignoré : le modèle n'apprend que
    ce qu'un acheteur pouvait savoir au moment de la vente (pas de fuite de données).
    """
    paires = pd.DataFrame({
        "vente": ventes.index,
        "batiment_groupe_id": batiment.reindex(ventes.index).to_numpy(),
        "date_vente": ventes["date_mutation"].to_numpy(),
    }).merge(dpe[["batiment_groupe_id", "date_dpe", "classe"]], on="batiment_groupe_id")
    avant = paires[paires["date_dpe"] < paires["date_vente"]]
    groupes = avant.groupby("vente")["classe"]
    return pd.DataFrame({"dpe_classe": groupes.median(), "nb_dpe": groupes.size()}).reindex(ventes.index)


def annee_par_batiment(batiments, dpe):
    """Année de construction : Fichiers fonciers (BDNB), sinon l'année médiane déclarée dans les DPE du bâtiment.

    L'année de construction ne change pas avec le temps : la lire dans un DPE postérieur à la vente
    n'apprend rien au modèle sur le prix de la vente elle-même.
    """
    annee = batiments.set_index("batiment_groupe_id")["annee_construction"]
    annee = annee.where(annee.between(1600, pd.Timestamp.today().year))
    declaree = pd.to_numeric(dpe["annee_construction"], errors="coerce")
    declaree = dpe.assign(annee=declaree.where(declaree.between(1600, pd.Timestamp.today().year)))
    secours = declaree.groupby("batiment_groupe_id")["annee"].median()
    return annee.fillna(secours.reindex(annee.index))


TABLE_APP = RACINE / "data" / "processed" / "batiments.parquet"  # versionnée : l'app la lit en ligne
ADRESSES_APP = RACINE / "data" / "processed" / "adresses_batiments.parquet"
CORRESPONDANCE = RACINE / "reports" / "correspondance_batiments.csv"


def ajouter_batiments(ventes):
    """Ajoute à chaque vente son bâtiment, son année de construction et sa classe DPE (avant la vente)."""
    tables, dpe_ademe = extraire_bdnb(), telecharger_dpe()
    dpe = dpe_par_batiment(dpe_ademe, tables["dpe"], tables["adresses"])
    lien = relier_ventes(ventes, tables["batiments"], tables["parcelles"], tables["adresses"])
    annees = annee_par_batiment(tables["batiments"], dpe)
    resultat = ventes.join(lien)
    resultat["annee_construction"] = lien["batiment_groupe_id"].map(annees)
    resultat = resultat.join(classe_dpe_avant_vente(ventes, lien["batiment_groupe_id"], dpe))
    return resultat, tables, dpe


def taux_correspondance(ventes):
    """Part des ventes reliées par chaque méthode, et part des ventes avec année et DPE, par année."""
    annee = ventes["date_mutation"].dt.year.rename("annee")
    taux = pd.DataFrame({
        "ventes": ventes.groupby(annee).size(),
        "parcelle": ventes["methode_batiment"].eq("parcelle").groupby(annee).mean(),
        "adresse": ventes["methode_batiment"].eq("adresse").groupby(annee).mean(),
        "proximite": ventes["methode_batiment"].eq("proximité").groupby(annee).mean(),
        "sans_batiment": ventes["methode_batiment"].isna().groupby(annee).mean(),
        "avec_annee": ventes["annee_construction"].notna().groupby(annee).mean(),
        "avec_dpe": ventes["dpe_classe"].notna().groupby(annee).mean(),
    })
    total = taux.drop(columns="ventes").mul(taux["ventes"], axis=0).sum() / taux["ventes"].sum()
    taux.loc["Toutes"] = [taux["ventes"].sum(), *total]
    return taux.round(4)


def table_app(tables, dpe):
    """Ce que l'app lit pour pré-remplir le formulaire : année et classe DPE médiane actuelle de chaque bâtiment.

    Pour une estimation faite aujourd'hui, tous les DPE déjà établis sont connus : on les prend tous.
    """
    batiments = tables["batiments"][["batiment_groupe_id", "x", "y"]].copy()
    batiments["annee_construction"] = batiments["batiment_groupe_id"].map(annee_par_batiment(tables["batiments"], dpe))
    par_batiment = dpe.groupby("batiment_groupe_id")["classe"]
    batiments["dpe_classe"] = batiments["batiment_groupe_id"].map(par_batiment.median())
    batiments["nb_dpe"] = batiments["batiment_groupe_id"].map(par_batiment.size()).fillna(0).astype(int)
    adresses = tables["adresses"][tables["adresses"]["cle_interop_adr"].str.startswith(f"{CODE_COMMUNE}_")]
    return batiments, adresses.drop_duplicates("cle_interop_adr").reset_index(drop=True)
