"""Entraîne le modèle final sur 2021-2024 et l'évalue UNE SEULE FOIS sur 2025."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train import CIBLE, VARIABLES, baseline, charger, decouper, evaluer

PARAMETRES = Path("reports/meilleurs_parametres.json")
MODELES_DIR = Path("models")
RAPPORTS_DIR = Path("reports")


def couverture(reel, bas, haut):
    """Part des vrais prix qui tombent dans la fourchette [bas, haut]."""
    reel, bas, haut = (np.asarray(x, dtype=float) for x in (reel, bas, haut))
    return float(np.mean((reel >= bas) & (reel <= haut)))


def entrainer(donnees, params, **reglages):
    """Entraîne un LightGBM avec les réglages trouvés par Optuna."""
    modele = LGBMRegressor(**params, subsample_freq=1, random_state=42, verbose=-1, **reglages)
    return modele.fit(donnees[VARIABLES], donnees[CIBLE])


def main():
    train, validation, test = decouper(charger())
    complet = pd.concat([train, validation])  # toutes les ventes de 2021 à 2024
    params = json.loads(PARAMETRES.read_text())

    print("Entraînement du modèle final et des modèles de fourchette...")
    modele = entrainer(complet, params)
    modele_bas = entrainer(complet, params, objective="quantile", alpha=0.1)
    modele_haut = entrainer(complet, params, objective="quantile", alpha=0.9)

    surface = test["surface_reelle_bati"]
    prix = modele.predict(test[VARIABLES]) * surface
    q10 = modele_bas.predict(test[VARIABLES])
    q90 = modele_haut.predict(test[VARIABLES])
    prix_bas = np.minimum(q10, q90) * surface  # sécurité : le bas reste sous le haut
    prix_haut = np.maximum(q10, q90) * surface

    resultats = {
        "Baseline": evaluer(test["valeur_fonciere"], baseline(complet, test)),
        "LightGBM final": evaluer(test["valeur_fonciere"], prix),
    }
    tableau = pd.DataFrame(resultats).T.round(1)
    taux = couverture(test["valeur_fonciere"], prix_bas, prix_haut)

    print(f"\nTest 2025 : {len(test)} ventes jamais vues\n")
    print(tableau.to_string())
    print(f"\nFourchette 10 %-90 % : {100 * taux:.1f} % des vrais prix sont dedans (objectif : environ 80 %)")

    RAPPORTS_DIR.mkdir(exist_ok=True)
    MODELES_DIR.mkdir(exist_ok=True)
    tableau.assign(**{"Couverture fourchette (%)": [None, round(100 * taux, 1)]}).to_csv(
        RAPPORTS_DIR / "resultats_test_2025.csv"
    )
    joblib.dump(modele, MODELES_DIR / "lightgbm_final.joblib")
    joblib.dump(modele_bas, MODELES_DIR / "lightgbm_q10.joblib")
    joblib.dump(modele_haut, MODELES_DIR / "lightgbm_q90.joblib")
    print("\nModèles enregistrés dans models/")


if __name__ == "__main__":
    main()