"""Compare, sur la validation 2024, le modèle sans et avec l'année de construction et le DPE.

Trois modèles, tous avec les réglages Optuna déjà trouvés (pas de nouvelle optimisation, qui
avantagerait le nouveau modèle sur cette même validation), tous évalués sur les mêmes ventes de 2024 :
1. l'ancien modèle : anciennes données (sans la règle des ventes hors marché), anciennes variables ;
2. les anciennes variables, sur les données nettoyées avec la règle ;
3. les mêmes, plus l'année de construction et la classe DPE.
L'écart 1 → 2 mesure l'effet du nettoyage, l'écart 2 → 3 celui des nouvelles variables.
Le test 2025 n'est pas utilisé ici.
"""
import json
from pathlib import Path

import pandas as pd

from final import PARAMETRES, entrainer
from train import VARIABLES, VARIABLES_BASE, charger, decouper, evaluer

RACINE = Path(__file__).resolve().parents[1]
ANCIENNES_DONNEES = RACINE / "data" / "external" / "archive_v1" / "ventes_features.parquet"
SORTIE = RACINE / "reports" / "comparaison_variables.csv"


GRAINES = range(5)  # les écarts sont petits : on répète l'entraînement pour les séparer du hasard


def scores(train, validation, variables, params):
    """Erreurs sur le prix total, moyennées sur plusieurs graines aléatoires (et leur écart-type)."""
    lignes = []
    for graine in GRAINES:
        modele = entrainer(train, params, variables, graine=graine)
        prix = modele.predict(validation[variables]) * validation["surface_reelle_bati"]
        lignes.append(evaluer(validation["valeur_fonciere"], prix))
    tableau = pd.DataFrame(lignes)
    return {**tableau.mean().to_dict(), "Écart-type MAE (€)": tableau["MAE (€)"].std()}


def main():
    params = json.loads((RACINE / PARAMETRES).read_text())
    train, validation, _ = decouper(charger(RACINE / "data" / "processed" / "ventes_features.parquet"))
    print(f"Validation 2024 : {len(validation)} ventes, {len(GRAINES)} entraînements par modèle")

    resultats = {}
    if ANCIENNES_DONNEES.exists():
        ancien_train, _, _ = decouper(charger(ANCIENNES_DONNEES))
        resultats["Ancien modèle (sans la règle)"] = scores(ancien_train, validation, VARIABLES_BASE, params)
    resultats["Anciennes variables, données nettoyées"] = scores(train, validation, VARIABLES_BASE, params)
    resultats["+ année de construction et DPE"] = scores(train, validation, VARIABLES, params)

    # un indicateur « DPE manquant » apporte-t-il quelque chose ? (LightGBM traite déjà les vides)
    marque = lambda d: d.assign(dpe_manquant=d["dpe_classe"].isna().astype(int))  # noqa: E731
    resultats["+ indicateur DPE manquant"] = scores(
        marque(train), marque(validation), VARIABLES + ["dpe_manquant"], params)

    tableau = pd.DataFrame(resultats).T.round(1)
    print(tableau.to_string())
    tableau.to_csv(SORTIE)
    print(f"Fichier écrit : {SORTIE}")


if __name__ == "__main__":
    main()
