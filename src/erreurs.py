"""Où le modèle se trompe sur les ventes de 2025."""
import numpy as np
import pandas as pd

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
