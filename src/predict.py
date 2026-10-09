"""Estime le prix d'un logement à Montpellier à partir d'une adresse."""
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from features import (
    CENTRE_MONTPELLIER, CRS_GPS, CRS_METRES, LIEUX, LIEUX_A_COMPTER, RAYON_DENSITE,
    distance_plus_proche, nombre_dans_rayon,
)
from train import VARIABLES

RACINE = Path(__file__).resolve().parents[1]
MODELES_DIR = RACINE / "models"
POI_DIR = RACINE / "data" / "external" / "osm"
VENTES = RACINE / "data" / "processed" / "ventes_features.parquet"

URL_GEOCODAGE = "https://data.geopf.fr/geocodage/search"
CODE_COMMUNE = "34172"  # Montpellier
SCORE_MIN = 0.5  # en dessous, le géocodage n'est pas assez sûr


class AdresseIntrouvable(Exception):
    """L'adresse n'a pas été trouvée dans la commune de Montpellier."""


def geocoder(adresse):
    """Transforme une adresse en (latitude, longitude, adresse officielle)."""
    reponse = requests.get(
        URL_GEOCODAGE, params={"q": adresse, "citycode": CODE_COMMUNE, "limit": 1}, timeout=10
    )
    reponse.raise_for_status()
    resultats = reponse.json().get("features", [])
    if not resultats:
        raise AdresseIntrouvable(adresse)
    proprietes = resultats[0]["properties"]
    if proprietes.get("score", 0) < SCORE_MIN or proprietes.get("citycode") != CODE_COMMUNE:
        raise AdresseIntrouvable(adresse)
    longitude, latitude = resultats[0]["geometry"]["coordinates"]
    return latitude, longitude, proprietes["label"]


class Estimateur:
    """Charge une seule fois les modèles et les données, puis estime autant de biens que voulu."""

    def __init__(self):
        self.modele = joblib.load(MODELES_DIR / "lightgbm_final.joblib")
        self.modele_bas = joblib.load(MODELES_DIR / "lightgbm_q10.joblib")
        self.modele_haut = joblib.load(MODELES_DIR / "lightgbm_q90.joblib")
        self.lieux = {nom: gpd.read_file(POI_DIR / f"{nom}.gpkg") for nom in LIEUX}
        self.ventes = pd.read_parquet(VENTES)
        centre = gpd.GeoSeries([Point(CENTRE_MONTPELLIER[1], CENTRE_MONTPELLIER[0])], crs=CRS_GPS)
        self.centre = centre.to_crs(CRS_METRES).iloc[0]

    def variables(self, latitude, longitude, type_local, surface, pieces, dependances, terrain):
        """Calcule les mêmes variables qu'à l'entraînement, pour un seul bien."""
        point = gpd.GeoDataFrame(geometry=[Point(longitude, latitude)], crs=CRS_GPS).to_crs(CRS_METRES)
        ligne = {
            "surface_reelle_bati": surface,
            "nombre_pieces_principales": pieces,
            "surface_par_piece": surface / pieces,
            "est_maison": int(type_local == "Maison"),
            "nb_dependances": dependances,
            "surface_terrain_totale": terrain,
            "latitude": latitude,
            "longitude": longitude,
            "dist_centre": float(point.distance(self.centre).iloc[0]),
        }
        for nom, lieux in self.lieux.items():
            ligne[f"dist_{nom}"] = float(distance_plus_proche(point, lieux).iloc[0])
            if nom in LIEUX_A_COMPTER:
                ligne[f"nb_{nom}_{RAYON_DENSITE}m"] = int(nombre_dans_rayon(point, lieux).iloc[0])
        return pd.DataFrame([ligne])[VARIABLES]

    def comparables(self, latitude, longitude, type_local, surface, n=5):
        """Les n ventes du même type, de surface proche (±30 %), les plus proches de l'adresse."""
        v = self.ventes[
            (self.ventes["type_local"] == type_local)
            & self.ventes["surface_reelle_bati"].between(0.7 * surface, 1.3 * surface)
        ].copy()
        # distance approximative en mètres, largement suffisante à l'échelle d'une ville
        dy = (v["latitude"] - latitude) * 111_000
        dx = (v["longitude"] - longitude) * 111_000 * np.cos(np.radians(latitude))
        v["distance_m"] = np.hypot(dx, dy)
        colonnes = ["date_mutation", "adresse_numero", "adresse_nom_voie", "surface_reelle_bati",
                    "nombre_pieces_principales", "valeur_fonciere", "prix_m2", "distance_m",
                    "latitude", "longitude"]
        return v.nsmallest(n, "distance_m")[colonnes]

    def estimer(self, adresse, type_local, surface, pieces, dependances=0, terrain=0):
        """Estimation complète : prix, fourchette, explication et comparables."""
        latitude, longitude, label = geocoder(adresse)
        X = self.variables(latitude, longitude, type_local, surface, pieces, dependances, terrain)

        prix_m2 = float(self.modele.predict(X)[0])
        q10 = float(self.modele_bas.predict(X)[0])
        q90 = float(self.modele_haut.predict(X)[0])
        prix = prix_m2 * surface
        prix_bas = min(q10, q90) * surface
        prix_haut = max(q10, q90) * surface

        # valeurs SHAP calculées par LightGBM, en €/m² ; la dernière colonne est la valeur de base
        contributions = self.modele.predict(X, pred_contrib=True)[0]
        facteurs = pd.Series(contributions[:-1] * surface, index=VARIABLES)
        facteurs = facteurs.reindex(facteurs.abs().sort_values(ascending=False).index)

        return {
            "adresse": label,
            "latitude": latitude,
            "longitude": longitude,
            "prix": prix,
            "prix_m2": prix_m2,
            "prix_bas": min(prix_bas, prix),
            "prix_haut": max(prix_haut, prix),
            "base": contributions[-1] * surface,
            "facteurs": facteurs,
            "comparables": self.comparables(latitude, longitude, type_local, surface),
        }