"""Agrégations du marché : carte en hexagones, évolution des prix, prix selon le nombre de pièces."""
import numpy as np
import pandas as pd

from features import CENTRE_MONTPELLIER

RAYON_HEX = 250  # mètres du centre à un sommet : un hexagone fait environ 430 m de large
MIN_VENTES = 10  # en dessous, la médiane d'une zone dépend trop de quelques ventes
METRES_PAR_DEGRE = 111_000
PIECES_MAX = 6  # 6 pièces et plus sont regroupées : au-delà, trop peu de ventes


def filtrer(ventes, type_local="Tous", annee="Toutes"):
    """Garde un type de bien et une année ; « Tous » et « Toutes » ne filtrent pas."""
    if type_local != "Tous":
        ventes = ventes[ventes["type_local"] == type_local]
    if annee != "Toutes":
        ventes = ventes[ventes["date_mutation"].dt.year == annee]
    return ventes


def en_metres(latitude, longitude, origine=CENTRE_MONTPELLIER):
    """Coordonnées approximatives en mètres autour de l'origine, suffisantes à l'échelle d'une ville."""
    lat0, lon0 = origine
    x = (np.asarray(longitude) - lon0) * METRES_PAR_DEGRE * np.cos(np.radians(lat0))
    y = (np.asarray(latitude) - lat0) * METRES_PAR_DEGRE
    return x, y


def en_degres(x, y, origine=CENTRE_MONTPELLIER):
    """Opération inverse de en_metres : renvoie (latitude, longitude)."""
    lat0, lon0 = origine
    return lat0 + y / METRES_PAR_DEGRE, lon0 + x / (METRES_PAR_DEGRE * np.cos(np.radians(lat0)))


def hexagone(latitude, longitude, rayon=RAYON_HEX):
    """Numéro (q, r) de l'hexagone qui contient chaque point, sur une grille à sommet en haut.

    On passe en coordonnées « cubiques » (q, r, s avec q + r + s = 0) puis on arrondit
    à l'hexagone le plus proche : la coordonnée la plus mal arrondie est recalculée.
    """
    x, y = en_metres(latitude, longitude)
    q = (np.sqrt(3) / 3 * x - y / 3) / rayon
    r = (2 / 3 * y) / rayon
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    rq = np.where((dq > dr) & (dq > ds), -rr - rs, rq)
    rr = np.where(~((dq > dr) & (dq > ds)) & (dr > ds), -rq - rs, rr)
    return rq.astype(int), rr.astype(int)


def contour_hexagone(q, r, rayon=RAYON_HEX):
    """Les 6 sommets [longitude, latitude] de l'hexagone (q, r)."""
    cx = rayon * np.sqrt(3) * (q + r / 2)
    cy = rayon * 1.5 * r
    angles = np.radians(30 + 60 * np.arange(6))
    lat, lon = en_degres(cx + rayon * np.cos(angles), cy + rayon * np.sin(angles))
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def grille_prix(ventes, rayon=RAYON_HEX, min_ventes=MIN_VENTES):
    """Prix au m² médian et nombre de ventes par hexagone, pour les zones assez fournies."""
    q, r = hexagone(ventes["latitude"], ventes["longitude"], rayon)
    zones = (
        pd.DataFrame({"q": q, "r": r, "prix_m2": ventes["prix_m2"].to_numpy()})
        .groupby(["q", "r"])["prix_m2"]
        .agg(prix_m2="median", nb_ventes="size")
        .reset_index()
    )
    zones = zones[zones["nb_ventes"] >= min_ventes].reset_index(drop=True)
    zones["contour"] = [contour_hexagone(a, b, rayon) for a, b in zip(zones["q"], zones["r"])]
    return zones


def evolution_prix(ventes):
    """Prix au m² médian par année et par type de bien."""
    annee = ventes["date_mutation"].dt.year.rename("annee")
    return (
        ventes.groupby([annee, "type_local"])["prix_m2"]
        .agg(prix_m2="median", nb_ventes="size")
        .reset_index()
    )


def prix_selon_pieces(ventes, pieces_max=PIECES_MAX, min_ventes=MIN_VENTES):
    """Déciles 1 et 9, quartiles et médiane du prix au m² selon le nombre de pièces (dernière classe : « 6 et + »).

    Comme pour la carte, une classe de moins de min_ventes ventes n'est pas gardée.
    """
    pieces = ventes["nombre_pieces_principales"].clip(upper=pieces_max).astype(int)
    stats = ventes.groupby(pieces)["prix_m2"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    stats = stats.rename(columns={"count": "nb_ventes", "10%": "p10", "25%": "p25", "50%": "mediane",
                                  "75%": "p75", "90%": "p90"})
    stats = stats[["nb_ventes", "p10", "p25", "mediane", "p75", "p90"]].rename_axis("pieces").reset_index()
    stats["nb_ventes"] = stats["nb_ventes"].astype(int)
    stats = stats[stats["nb_ventes"] >= min_ventes].reset_index(drop=True)
    stats["Pièces"] = [f"{p} et +" if p == pieces_max else str(p) for p in stats["pieces"]]
    return stats
