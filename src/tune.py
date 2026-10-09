"""Cherche les meilleurs réglages de LightGBM sur la validation 2024."""
import json
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMRegressor

from train import CIBLE, VARIABLES, charger, decouper

SORTIE = Path("reports/meilleurs_parametres.json")
N_ESSAIS = 40


def erreur_mediane(prix_reel, prix_predit):
    """Erreur relative médiane : l'indicateur que l'on cherche à réduire."""
    prix_reel = np.asarray(prix_reel, dtype=float)
    prix_predit = np.asarray(prix_predit, dtype=float)
    return float(np.median(np.abs(prix_predit - prix_reel) / prix_reel))


def creer_objectif(train, validation):
    """Renvoie la fonction qu'Optuna appelle à chaque essai de réglages."""
    def objectif(essai):
        params = {
            "n_estimators": essai.suggest_int("n_estimators", 200, 1500),
            "learning_rate": essai.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": essai.suggest_int("num_leaves", 15, 127),
            "min_child_samples": essai.suggest_int("min_child_samples", 10, 100),
            "subsample": essai.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": essai.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": essai.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        modele = LGBMRegressor(**params, subsample_freq=1, random_state=42, verbose=-1)
        modele.fit(train[VARIABLES], train[CIBLE])
        prix = modele.predict(validation[VARIABLES]) * validation["surface_reelle_bati"]
        return erreur_mediane(validation["valeur_fonciere"], prix)
    return objectif


def main():
    train, validation, _ = decouper(charger())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    etude = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    etude.optimize(creer_objectif(train, validation), n_trials=N_ESSAIS, show_progress_bar=True)

    print(f"\nMeilleure erreur médiane sur 2024 : {100 * etude.best_value:.2f} %")
    print("Meilleurs réglages :", etude.best_params)
    SORTIE.parent.mkdir(exist_ok=True)
    SORTIE.write_text(json.dumps(etude.best_params, indent=2))
    print(f"Réglages enregistrés dans {SORTIE}")


if __name__ == "__main__":
    main()