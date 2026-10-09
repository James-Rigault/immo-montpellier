"""Entraîne et compare plusieurs modèles de prix avec une validation temporelle."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ENTREE = Path("data/processed/ventes_features.parquet")
MODELES_DIR = Path("models")
RAPPORTS_DIR = Path("reports")

CIBLE = "prix_m2"
VARIABLES = [
    "surface_reelle_bati", "nombre_pieces_principales", "surface_par_piece", "est_maison",
    "nb_dependances", "surface_terrain_totale", "latitude", "longitude",
    "dist_centre", "dist_tram", "dist_ecole", "dist_parc", "dist_commerce", "dist_restaurant",
    "nb_ecole_500m", "nb_commerce_500m", "nb_restaurant_500m",
]
ANNEES_TRAIN = [2021, 2022, 2023]
ANNEE_VALIDATION = 2024
ANNEE_TEST = 2025  # gardée de côté jusqu'à la semaine 5


def charger(chemin=ENTREE):
    """Charge les ventes avec leurs variables."""
    df = pd.read_parquet(chemin)
    df["annee"] = df["date_mutation"].dt.year
    return df


def decouper(df):
    """Découpage temporel : on apprend sur le passé, on évalue sur le futur."""
    train = df[df["annee"].isin(ANNEES_TRAIN)]
    validation = df[df["annee"] == ANNEE_VALIDATION]
    test = df[df["annee"] == ANNEE_TEST]
    return train, validation, test


def evaluer(prix_reel, prix_predit):
    """Mesures d'erreur sur le prix total, en euros et en pourcentage."""
    prix_reel = np.asarray(prix_reel, dtype=float)
    prix_predit = np.asarray(prix_predit, dtype=float)
    erreur_relative = np.abs(prix_predit - prix_reel) / prix_reel
    return {
        "MAE (€)": np.mean(np.abs(prix_predit - prix_reel)),
        "MAPE (%)": 100 * np.mean(erreur_relative),
        "Erreur médiane (%)": 100 * np.median(erreur_relative),
        "Part à ±10 % (%)": 100 * np.mean(erreur_relative <= 0.10),
    }


def baseline(train, autre):
    """Prix au m² médian du code postal et du type de bien (appris sur train) × surface."""
    medianes = train.groupby(["code_postal", "type_local"])[CIBLE].median()
    cles = pd.MultiIndex.from_frame(autre[["code_postal", "type_local"]])
    prix_m2 = pd.Series(medianes.reindex(cles).to_numpy(), index=autre.index)
    prix_m2 = prix_m2.fillna(train[CIBLE].median())  # code postal jamais vu : médiane globale
    return prix_m2 * autre["surface_reelle_bati"]


def creer_modeles():
    """Les modèles à comparer, du plus simple au plus puissant."""
    return {
        "Régression linéaire (Ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
        ),
    }


def main():
    df = charger()
    train, validation, _ = decouper(df)
    print(f"Entraînement : {len(train)} ventes ({ANNEES_TRAIN[0]}-{ANNEES_TRAIN[-1]})")
    print(f"Validation   : {len(validation)} ventes ({ANNEE_VALIDATION})")

    prix_reel = validation["valeur_fonciere"]
    predictions = validation[["type_local", "code_postal", "surface_reelle_bati", "dist_centre"]].copy()
    predictions["reel"] = prix_reel
    predictions["Baseline"] = baseline(train, validation)
    resultats = {"Baseline (médiane code postal)": evaluer(prix_reel, predictions["Baseline"])}

    entraines = {}
    for nom, modele in creer_modeles().items():
        print(f"Entraînement : {nom}...")
        modele.fit(train[VARIABLES], train[CIBLE])
        predictions[nom] = modele.predict(validation[VARIABLES]) * validation["surface_reelle_bati"]
        resultats[nom] = evaluer(prix_reel, predictions[nom])
        entraines[nom] = modele

    tableau = pd.DataFrame(resultats).T.round(1)
    print()
    print(tableau.to_string())

    RAPPORTS_DIR.mkdir(exist_ok=True)
    MODELES_DIR.mkdir(exist_ok=True)
    tableau.to_csv(RAPPORTS_DIR / "comparaison_modeles.csv")
    predictions.to_parquet(RAPPORTS_DIR / "predictions_validation.parquet")
    joblib.dump(entraines["LightGBM"], MODELES_DIR / "lightgbm_v1.joblib")
    print("\nRésultats et modèle enregistrés dans reports/ et models/")


if __name__ == "__main__":
    main()