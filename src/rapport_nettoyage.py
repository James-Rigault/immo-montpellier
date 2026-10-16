"""Écrit dans reports/ les chiffres du nettoyage que l'app affiche : l'entonnoir et les ventes hors marché.

Les données brutes ne sont pas versionnées : l'app en ligne ne peut pas refaire ces calculs,
elle lit donc les fichiers produits ici (comme pour reports/resultats_test_2025.csv).
"""
import json
from pathlib import Path

import pandas as pd

from clean import (
    SEUIL_RELATIF, SORTIE, charger_brut, construire_ventes, dedoublonner_locaux, garder_ventes, mediane_locale,
    nettoyer, preparer, resumer_mutations,
)

RACINE = Path(__file__).resolve().parents[1]
ENTONNOIR = RACINE / "reports" / "entonnoir_nettoyage.csv"
HORS_MARCHE = RACINE / "reports" / "ventes_hors_marche.json"


def entonnoir(etapes):
    """Tableau de l'entonnoir : lignes restantes et lignes retirées à chaque étape."""
    tableau = pd.DataFrame({"etape": list(etapes), "lignes": list(etapes.values())})
    tableau["retirees"] = (-tableau["lignes"].diff()).fillna(0).astype(int)
    tableau["appliquee"] = True
    return tableau


def avant_regle(df):
    """Les ventes d'un seul logement, nettoyées comme dans clean.py jusqu'à la règle des 30 % (exclue)."""
    ventes = garder_ventes(df)
    locaux = dedoublonner_locaux(ventes)
    return preparer(construire_ventes(locaux, resumer_mutations(ventes, locaux)))


def ventes_hors_marche(v, seuil=SEUIL_RELATIF):
    """Les ventes que la règle des 30 % retire, résumées en quelques chiffres.

    `v` : les ventes juste avant cette règle, avec leur prix au m².
    """
    locale = mediane_locale(v)
    # exactement le contraire de la condition de clean.py, pour retrouver le même nombre de ventes
    retirees = v[~(v["prix_m2"] >= seuil * locale)]
    # plusieurs ventes à la même adresse et au même prix au m² : souvent des lots vendus en bloc
    meme_immeuble = retirees.groupby(
        [retirees["adresse_numero"], retirees["adresse_nom_voie"], retirees["prix_m2"].round()]
    ).size()
    return {
        "seuil": seuil,
        "ventes_avant": int(len(v)),
        "ventes_retirees": int(len(retirees)),
        "prix_m2_median_retirees": float(retirees["prix_m2"].median()),
        "prix_m2_median_local": float(locale[retirees.index].median()),
        "plus_grand_groupe": int(meme_immeuble.max()) if len(meme_immeuble) else 0,
    }


def main():
    df = charger_brut(RACINE / "data" / "raw")
    propres, etapes = nettoyer(df)

    # garde-fou : les étapes comptées sont bien celles qui ont produit le fichier utilisé par le modèle
    reference = pd.read_parquet(RACINE / SORTIE)
    pd.testing.assert_frame_equal(propres, reference.reset_index(drop=True))

    tableau = entonnoir(etapes)
    ENTONNOIR.parent.mkdir(exist_ok=True)
    tableau.to_csv(ENTONNOIR, index=False)
    print(tableau.to_string(index=False))

    resume = ventes_hors_marche(avant_regle(df))
    HORS_MARCHE.write_text(json.dumps(resume, ensure_ascii=False, indent=2))
    print(json.dumps(resume, ensure_ascii=False, indent=2))
    print(f"Fichiers écrits : {ENTONNOIR}, {HORS_MARCHE}")


if __name__ == "__main__":
    main()
