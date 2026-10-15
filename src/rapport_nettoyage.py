"""Écrit dans reports/ l'entonnoir du nettoyage, que l'app affiche dans « À propos ».

Les données brutes ne sont pas versionnées : l'app en ligne ne peut pas refaire ce calcul,
elle lit donc le fichier produit ici (comme pour reports/resultats_test_2025.csv).

Les données du modèle actuel ont été produites SANS la règle des ventes hors marché (seuil de 30 %),
ajoutée à clean.py après coup. L'entonnoir montre donc les étapes réellement appliquées, puis
la règle comme étape à appliquer, avec le nombre de ventes qu'elle retirerait.
"""
from pathlib import Path

import pandas as pd

from clean import SORTIE, charger_brut, nettoyer

RACINE = Path(__file__).resolve().parents[1]
ENTONNOIR = RACINE / "reports" / "entonnoir_nettoyage.csv"
ETAPE_REGLE = "Règle des ventes hors marché"


def entonnoir(etapes_appliquees, lignes_avec_regle):
    """Tableau de l'entonnoir : lignes restantes et retirées à chaque étape, puis la règle à appliquer."""
    tableau = pd.DataFrame({"etape": list(etapes_appliquees), "lignes": list(etapes_appliquees.values())})
    tableau["retirees"] = (-tableau["lignes"].diff()).fillna(0).astype(int)
    tableau["appliquee"] = True
    regle = {"etape": ETAPE_REGLE, "lignes": lignes_avec_regle,
             "retirees": tableau["lignes"].iloc[-1] - lignes_avec_regle, "appliquee": False}
    return pd.concat([tableau, pd.DataFrame([regle])], ignore_index=True)


def main():
    df = charger_brut(RACINE / "data" / "raw")
    sans_regle, etapes = nettoyer(df, seuil=None)
    avec_regle, _ = nettoyer(df)

    # garde-fou : les étapes comptées sont bien celles qui ont produit le fichier utilisé par le modèle
    reference = pd.read_parquet(RACINE / SORTIE)
    pd.testing.assert_frame_equal(sans_regle, reference.reset_index(drop=True))

    tableau = entonnoir(etapes, len(avec_regle))
    ENTONNOIR.parent.mkdir(exist_ok=True)
    tableau.to_csv(ENTONNOIR, index=False)
    print(tableau.to_string(index=False))
    print(f"Fichier écrit : {ENTONNOIR}")


if __name__ == "__main__":
    main()
