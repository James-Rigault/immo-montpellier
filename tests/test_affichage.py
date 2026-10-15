import json
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
    for prix in ["250000 €", "4000 € le m²", "200000 €", "300000 €"]:
        assert prix in carte
    assert "<b>1 Rue Test</b>" not in carte
    assert 'style="left: 50.0%"' in carte


def test_badge_confiance_remplit_un_segment_par_niveau():
    for niveau, remplis in [("faible", 1), ("moyenne", 2), ("élevée", 3)]:
        badge = badge_confiance(niveau, ["une raison"])
        assert badge.count('class="plein"') == remplis
        assert "Confiance" in badge and niveau.capitalize() in badge
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
    # json garde les caractères tels quels ; str() écrirait l'espace fine sous la forme « \\u202f »
    spec = json.dumps(graphique_comparaison(resultats).to_dict(), ensure_ascii=False)
    for score in ["Erreur moyenne", "Erreur médiane", "Prix à ±10\u202f%"]:
        assert score in spec


def test_couleur_classe_suit_les_bornes():
    from affichage import RAMPE, bornes_classes, couleur_classe
    bornes = bornes_classes(pd.Series(range(1, 101)))
    assert len(bornes) == len(RAMPE) - 1 and bornes == sorted(bornes)
    assert couleur_classe(0, bornes) == [0xDC, 0xE3, 0xD9]  # la plus claire
    assert couleur_classe(1000, bornes) == [0x1F, 0x3B, 0x2D]  # le vert bouteille


def test_legende_carte_une_pastille_par_classe():
    from affichage import RAMPE, legende_carte
    legende = legende_carte([2000, 3000, 4000, 5000], euros)
    assert legende.count("<li>") == len(RAMPE)
    assert "moins de 2000 €" in legende and "5000 € et plus" in legende and "3000 € – 4000 €" in legende


def test_ecart_estimations():
    from affichage import ecart_estimations
    a = {"prix": 200_000, "prix_m2": 4_000, "prix_bas": 180_000, "prix_haut": 230_000}
    b = {"prix": 250_000, "prix_m2": 5_000, "prix_bas": 220_000, "prix_haut": 280_000}
    ecart = ecart_estimations(a, b)
    assert ecart["ecart"] == 50_000 and ecart["ecart_pct"] == 25 and ecart["ecart_m2"] == 1_000
    assert ecart["chevauchement"]  # 220 000 – 230 000 en commun
    loin = {**b, "prix_bas": 240_000}
    assert not ecart_estimations(a, loin)["chevauchement"]
    assert ecart_estimations(b, a)["ecart"] == -50_000


def test_graphique_erreurs_texte_et_repere():
    from affichage import graphique_erreurs
    erreurs = pd.DataFrame({"groupe": ["34000", "34070"], "erreur_mediane": [11.2, 10.8], "nb_ventes": [1480, 1465]})
    spec = json.dumps(graphique_erreurs(erreurs, 10.8).to_dict(), ensure_ascii=False)
    assert "11,2 %  ·  1 480 ventes" in spec
    assert '"x": 10.8' in spec  # le filet de l'erreur médiane globale


def test_graphique_entonnoir_trois_statuts():
    from affichage import graphique_entonnoir
    tableau = pd.DataFrame({
        "etape": ["Lignes brutes", "Filtrées", "Règle"], "lignes": [1000, 800, 750],
        "retirees": [0, 200, 50], "appliquee": [True, True, False],
    })
    spec = json.dumps(graphique_entonnoir(tableau).to_dict(), ensure_ascii=False)
    for statut in ["Étape appliquée", "Données du modèle", "Étape à appliquer"]:
        assert statut in spec
    assert "800   −200" in spec and '"Texte": "1 000"' in spec


def test_schema_parcours_une_case_par_etape_et_texte_protege():
    from affichage import schema_parcours
    etapes = [{"titre": "Collecte", "texte": "<b>DVF</b>", "script": "src/ingest.py", "sortie": "data/raw"},
              {"titre": "App", "texte": "Streamlit", "script": "app/streamlit_app.py", "sortie": "le site"}]
    html = schema_parcours(etapes)
    assert html.count("<li>") == 2 and "<b>DVF</b>" not in html and "src/ingest.py" in html
