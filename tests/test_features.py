import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import CRS_METRES, distance_plus_proche, nombre_dans_rayon  # noqa: E402


def exemple():
    """2 ventes et 3 lieux, en mètres, sur une ligne droite."""
    ventes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1000, 0)], crs=CRS_METRES)
    lieux = gpd.GeoDataFrame(geometry=[Point(100, 0), Point(300, 0), Point(1000, 450)], crs=CRS_METRES)
    return ventes, lieux


def test_distance_plus_proche():
    ventes, lieux = exemple()
    distances = distance_plus_proche(ventes, lieux)
    assert list(distances.round()) == [100, 450]


def test_nombre_dans_rayon():
    ventes, lieux = exemple()
    nombres = nombre_dans_rayon(ventes, lieux, rayon=500)
    assert list(nombres) == [2, 1]