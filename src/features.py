"""Ajoute des variables géographiques à chaque vente : distances et densités de lieux."""
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

ENTREE = Path("data/processed/ventes_clean.parquet")
SORTIE = Path("data/processed/ventes_features.parquet")
POI_DIR = Path("data/external/osm")

CRS_GPS = "EPSG:4326"     # latitude / longitude en degrés
CRS_METRES = "EPSG:2154"  # Lambert-93 : coordonnées en mètres
CENTRE_MONTPELLIER = (43.6086, 3.8797)  # place de la Comédie (latitude, longitude)
RAYON_TELECHARGEMENT = 10_000  # mètres autour du centre : couvre toute la commune et ses bords
RAYON_DENSITE = 500  # mètres

LIEUX = {
    "tram": {"railway": "tram_stop"},
    "ecole": {"amenity": "school"},
    "parc": {"leisure": "park"},
    "commerce": {"shop": True},
    "restaurant": {"amenity": ["restaurant", "cafe", "bar"]},
}
LIEUX_A_COMPTER = ["commerce", "restaurant", "ecole"]


def en_geodataframe(df):
    """Transforme le tableau de ventes en carte de points, en mètres."""
    points = gpd.points_from_xy(df["longitude"], df["latitude"])
    return gpd.GeoDataFrame(df[[]], geometry=points, crs=CRS_GPS).to_crs(CRS_METRES)


def telecharger_lieux(nom, tags):
    """Télécharge les lieux OSM une seule fois, puis les relit depuis le disque."""
    fichier = POI_DIR / f"{nom}.gpkg"
    if fichier.exists():
        return gpd.read_file(fichier)
    lieux = ox.features_from_point(CENTRE_MONTPELLIER, tags=tags, dist=RAYON_TELECHARGEMENT)
    lieux = lieux[["geometry"]].reset_index(drop=True).to_crs(CRS_METRES)
    lieux["geometry"] = lieux.geometry.centroid  # un parc (polygone) devient son point central
    POI_DIR.mkdir(parents=True, exist_ok=True)
    lieux.to_file(fichier, driver="GPKG")
    return lieux


def distance_plus_proche(ventes, lieux):
    """Distance en mètres entre chaque vente et le lieu le plus proche."""
    joint = gpd.sjoin_nearest(ventes, lieux, how="left", distance_col="distance")
    joint = joint[~joint.index.duplicated()]  # en cas d'égalité, on garde une seule ligne
    return joint["distance"].reindex(ventes.index)


def nombre_dans_rayon(ventes, lieux, rayon=RAYON_DENSITE):
    """Nombre de lieux à moins de `rayon` mètres de chaque vente."""
    cercles = ventes.copy()
    cercles["geometry"] = cercles.buffer(rayon)
    joint = gpd.sjoin(cercles, lieux, how="left", predicate="contains")
    compte = joint["index_right"].notna().groupby(level=0).sum()
    return compte.reindex(ventes.index, fill_value=0)


def ajouter_variables(df):
    """Ajoute toutes les variables géographiques et quelques variables simples."""
    resultat = df.reset_index(drop=True).copy()
    ventes = en_geodataframe(resultat)

    centre = gpd.GeoSeries([Point(CENTRE_MONTPELLIER[1], CENTRE_MONTPELLIER[0])], crs=CRS_GPS)
    resultat["dist_centre"] = ventes.distance(centre.to_crs(CRS_METRES).iloc[0])

    for nom, tags in LIEUX.items():
        lieux = telecharger_lieux(nom, tags)
        print(f"{nom:<12} {len(lieux):>6} lieux trouvés")
        resultat[f"dist_{nom}"] = distance_plus_proche(ventes, lieux)
        if nom in LIEUX_A_COMPTER:
            resultat[f"nb_{nom}_{RAYON_DENSITE}m"] = nombre_dans_rayon(ventes, lieux)

    resultat["est_maison"] = (resultat["type_local"] == "Maison").astype(int)
    resultat["surface_par_piece"] = resultat["surface_reelle_bati"] / resultat["nombre_pieces_principales"]
    resultat["annee"] = resultat["date_mutation"].dt.year
    resultat["mois"] = resultat["date_mutation"].dt.month
    return resultat


def main():
    df = pd.read_parquet(ENTREE)
    resultat = ajouter_variables(df)
    resultat.to_parquet(SORTIE, index=False)
    nouvelles = [c for c in resultat.columns if c not in df.columns]
    print(f"{len(nouvelles)} nouvelles variables : {', '.join(nouvelles)}")
    print(f"Fichier écrit : {SORTIE}")


if __name__ == "__main__":
    main()