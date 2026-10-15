"""Où le modèle se trompe sur les ventes de 2025, et ce que pèsent les ventes hors marché."""
import numpy as np
import pandas as pd

from clean import SEUIL_RELATIF, mediane_locale
from train import ANNEE_TEST, VARIABLES

TRANCHES_SURFACE = [0, 30, 50, 80, 120, np.inf]
NOMS_TRANCHES = ["Moins de 30 m²", "30 à 50 m²", "50 à 80 m²", "80 à 120 m²", "120 m² et plus"]


def predictions_test(modele, ventes):
    """Ventes de 2025 avec le prix estimé et l'erreur relative, calculés comme dans src/final.py."""
    test = ventes[ventes["date_mutation"].dt.year == ANNEE_TEST].copy()
    test["prix_estime"] = modele.predict(test[VARIABLES]) * test["surface_reelle_bati"]
    test["erreur"] = (test["prix_estime"] - test["valeur_fonciere"]).abs() / test["valeur_fonciere"]
    return test


def tranche_surface(surfaces):
    """Classe chaque surface dans une tranche (moins de 30 m², 30 à 50 m²…)."""
    return pd.cut(surfaces, TRANCHES_SURFACE, labels=NOMS_TRANCHES, right=False)


def erreur_par_groupe(test, groupes):
    """Erreur médiane (en %) et nombre de ventes pour chaque groupe, dans l'ordre des groupes."""
    return (
        test.groupby(groupes, observed=True)["erreur"]
        .agg(erreur_mediane="median", nb_ventes="size")
        .assign(erreur_mediane=lambda t: 100 * t["erreur_mediane"])
        .rename_axis("groupe")
        .reset_index()
        .astype({"groupe": str})
    )


def hors_marche(ventes, seuil=SEUIL_RELATIF):
    """Vrai pour les ventes que la règle de clean.py retirerait : moins de 30 % du prix au m² local."""
    return ventes["prix_m2"] < seuil * mediane_locale(ventes)


def bilan_hors_marche(ventes, test, nb_pires=20, seuil=SEUIL_RELATIF):
    """Les chiffres de l'encadré : combien de ventes hors marché, à quel prix, et quel poids dans les erreurs."""
    masque = hors_marche(ventes, seuil)
    dans_test = masque.loc[test.index]
    pires = test["erreur"].nlargest(nb_pires).index
    # plusieurs ventes à la même adresse au même prix au m² : souvent une vente en bloc
    meme_immeuble = ventes[masque].groupby(
        ["adresse_numero", "adresse_nom_voie", ventes.loc[masque, "prix_m2"].round()]
    ).size()
    return {
        "seuil": seuil,
        "nb": int(masque.sum()),
        "prix_m2_median": float(ventes.loc[masque, "prix_m2"].median()),
        "prix_m2_local": float(mediane_locale(ventes)[masque].median()),
        "plus_grand_groupe": int(meme_immeuble.max()) if len(meme_immeuble) else 0,
        "nb_test": int(dans_test.sum()),
        "erreur_mediane_hors_marche": 100 * float(test.loc[dans_test, "erreur"].median()),
        "nb_parmi_pires": int(dans_test.loc[pires].sum()),
        "nb_pires": nb_pires,
        "erreur_moyenne": 100 * float(test["erreur"].mean()),
        "erreur_moyenne_sans": 100 * float(test.loc[~dans_test, "erreur"].mean()),
    }
