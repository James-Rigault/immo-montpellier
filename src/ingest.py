"""Télécharge les données DVF géolocalisées de Montpellier."""
from pathlib import Path

import requests

CODE_DEPT = "34"
CODE_COMMUNE = "34172"  # Montpellier
ANNEES = range(2019, 2027)  # les années absentes sont ignorées
BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/communes/{dept}/{commune}.csv"
RAW_DIR = Path("data/raw")


def telecharger_dvf():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for annee in ANNEES:
        url = BASE_URL.format(annee=annee, dept=CODE_DEPT, commune=CODE_COMMUNE)
        dest = RAW_DIR / f"dvf_{CODE_COMMUNE}_{annee}.csv"
        if dest.exists():
            print(f"{annee} : déjà téléchargé")
            continue
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            print(f"{annee} : pas disponible")
            continue
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"{annee} : {len(r.content) / 1e6:.1f} Mo")


if __name__ == "__main__":
    telecharger_dvf()