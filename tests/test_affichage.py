import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pandas as pd  # noqa: E402
from affichage import (  # noqa: E402
    badge_confiance, carte_resultat, graduations, graphique_comparaison, graphique_facteurs, importance_variables,
    position_estimation, regrouper_facteurs,
)


def euros(x):
    return f"{x:.0f} €"


def test_position_estimation_dans_la_fourchette():
    assert position_estimation(250_000, 200_000, 300_000) == 50
    assert position_estimation(200_000, 200_000, 300_000) == 0
    assert position_estimation(300_000, 200_000, 300_000) == 100


def test_position_estimation_cas_limites():
    assert position_estimation(350_000, 200_000, 300_000) == 100
    assert position_estimation(250_000, 250_000, 250_000) == 50


def test_carte_resultat_affiche_les_trois_prix_et_protege_l_adresse():
    carte = carte_resultat("<b>1 Rue Test</b>", 250_000, 4_000, 200_000, 300_000, euros)
    for prix in ["250000 €", "4000 €/m²", "200000 €", "300000 €"]:
        assert prix in carte
    assert "<b>1 Rue Test</b>" not in carte
    assert 'style="left: 50.0%"' in carte


def test_badge_confiance_remplit_un_segment_par_niveau():
    for niveau, remplis in [("faible", 1), ("moyenne", 2), ("élevée", 3)]:
        badge = badge_confiance(niveau, ["une raison"])
        assert badge.count('class="plein"') == remplis
        assert f"Confiance {niveau}" in badge
        assert "<li>Une raison</li>" in badge


def test_regrouper_facteurs_additionne_latitude_et_longitude():
    facteurs = pd.Series({"surface_reelle_bati": -5_000.0, "latitude": 8_000.0, "longitude": 4_000.0, "dist_tram": 1_000.0})
    groupes = regrouper_facteurs(facteurs)
    assert "latitude" not in groupes and "longitude" not in groupes
    assert groupes["emplacement"] == 12_000
    assert groupes.sum() == facteurs.sum()  # le prix expliqué ne change pas
    assert list(groupes.index) == ["emplacement", "surface_reelle_bati", "dist_tram"]


def test_graphique_facteurs_axe_negatif_toujours_visible():
    facteurs = pd.Series({"Emplacement dans la ville": 20_000.0, "Surface": 6_000.0})
    spec = graphique_facteurs(facteurs).to_dict()
    domaine = spec["layer"][0]["encoding"]["x"]["scale"]["domain"]
    assert domaine[0] < 0 < domaine[1]
    assert "+20,0 k€" in str(spec)


def test_graduations_avec_une_valeur_negative():
    assert graduations(-7.7, 33.4) == [-5, 0, 10, 20, 30]
    assert graduations(-1.5, 4.0) == [-1, 0, 1, 2, 3, 4]
    assert graduations(-0.3, 1.2) == [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def test_importance_variables_regroupe_l_emplacement():
    import joblib
    modele = joblib.load(Path(__file__).resolve().parents[1] / "models" / "lightgbm_final.joblib")
    importance = importance_variables(modele)
    assert "latitude" not in importance and "longitude" not in importance
    assert "emplacement" in importance
    assert abs(importance.sum() - 100) < 1e-9
    assert importance.is_monotonic_decreasing


def test_graphique_comparaison_trois_scores():
    resultats = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "resultats_test_2025.csv", index_col=0)
    spec = str(graphique_comparaison(resultats).to_dict())
    for score in ["Erreur moyenne", "Erreur médiane", "Prix à ±10 %"]:
        assert score in spec
