import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
import predict  # noqa: E402
from batiments import lettre_dpe  # noqa: E402
from test_predict import FausseReponse  # noqa: E402

APP = str(RACINE / "app" / "streamlit_app.py")


def lancer_app():
    # l'app charge le modèle et les ventes au démarrage : on laisse plus que les 3 s par défaut
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.mark.parametrize("quartier, adresse", [
    ("Écusson", "10 rue de la Loge"),
    ("Port Marianne", "15 rue de la Cavalade"),
    ("Beaux-Arts", "10 rue Proudhon"),
])
def test_bouton_exemple_remplit_adresse(monkeypatch, quartier, adresse):
    # Pas d'appel réseau : le service d'adresses répond toujours la même suggestion
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))

    app = lancer_app()
    next(b for b in app.button if b.label == quartier).click().run()

    assert app.text_input(key="texte").value == adresse
    assert app.selectbox[0].value == adresse


def test_chiffres_cles_lus_dans_les_donnees_et_le_rapport_2025():
    final = pd.read_csv(RACINE / "reports" / "resultats_test_2025.csv", index_col=0).loc["LightGBM final"]
    ventes = pd.read_parquet(RACINE / "data" / "processed" / "ventes_features.parquet")
    app = lancer_app()
    badges = next(m.value for m in app.markdown if 'class="chiffres"' in m.value)
    # « \u202f » : espace fine insécable, entre milliers et avant « % »
    assert f"{len(ventes):,}".replace(",", "\u202f") in badges
    assert f"{final['Erreur médiane (%)']:.1f}\u202f%".replace(".", ",") in badges
    assert f"{final['Part à ±10 % (%)']:.0f}\u202f%" in badges


def test_estimation_affiche_carte_et_confiance(monkeypatch):
    adresse = "10 Rue de la Loge 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))

    app = lancer_app()
    next(b for b in app.button if b.label == "Écusson").click().run()
    next(b for b in app.button if b.label == "Estimer le prix").click().run()

    assert not app.exception
    html = "".join(m.value for m in app.markdown)
    assert 'class="resultat"' in html and 'class="fourchette"' in html
    assert 'class="confiance"' in html
    graphiques = [str(json.loads(g.proto.spec)) for g in app.get("vega_lite_chart")]
    assert any("Effet sur le prix, en k€" in spec for spec in graphiques)


def test_onglets_et_pied_de_page():
    app = lancer_app()
    assert not app.exception
    assert [onglet.label for onglet in app.tabs] == ["Estimer", "Explorer le marché", "Comparer", "Le modèle", "À propos"]
    assert "Limites" in "".join(m.value for m in app.tabs[3].markdown)
    assert "data.gouv.fr" in "".join(m.value for m in app.tabs[4].markdown)
    # le lien LinkedIn n'est pas renseigné : il ne doit pas apparaître vide
    pied = next(m.value for m in app.markdown if 'class="pied-grille"' in m.value)
    assert 'href=""' not in pied and "github.com/James-Rigault/immo-montpellier" in pied
    # bandeau : logo (le fichier tel quel, en ligne), nom de l'app, liens Méthode et GitHub
    bandeau = next(m.value for m in app.markdown if 'class="bandeau"' in m.value)
    assert "data:image/svg+xml;base64," in bandeau and "Valeur Montpellier" in bandeau
    assert ">Méthode<" in bandeau and ">GitHub<" in bandeau


@pytest.mark.parametrize("type_local, annee", [("Tous", "Toutes"), ("Appartement", 2023), ("Maison", 2025)])
def test_explorer_le_marche_suit_les_filtres(type_local, annee):
    app = lancer_app()
    app.radio(key="marche_type").set_value(type_local).run()
    app.selectbox(key="marche_annee").set_value(annee).run()
    assert not app.exception
    graphiques = [str(json.loads(g.proto.spec)) for g in app.tabs[1].get("vega_lite_chart")]
    assert len(graphiques) == 2 and any("Nombre de pièces" in spec for spec in graphiques)
    texte = "".join(m.value for m in app.tabs[1].markdown)
    assert "affichée" in texte or "Aucune zone" in "".join(i.value for i in app.tabs[1].info)


def test_comparer_deux_biens(monkeypatch):
    # le bien A en centre-ville, le bien B à La Paillade : B doit être estimé moins cher
    coordonnees = {"A": (43.6110, 3.8790), "B": (43.6290, 3.8250)}
    def suggestions_factices(*a, params, **k):
        nom = "A" if "Loge" in params["q"] else "B"
        return FausseReponse({"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": f"Bien {nom}"}}]})
    monkeypatch.setattr(predict.requests, "get", suggestions_factices)
    # les suggestions sont gardées en cache d'un test à l'autre : on reconnaît le bien A à « Loge »
    monkeypatch.setattr(predict, "geocoder", lambda texte: (*coordonnees["A" if "Loge" in texte else "B"], texte))

    app = lancer_app()
    app.text_input(key="cmp_texte_A").input("10 rue de la Loge").run()
    app.text_input(key="cmp_texte_B").input("rue de Tipasa").run()
    next(b for b in app.button if b.label == "Comparer").click().run()

    assert not app.exception
    onglet = app.tabs[2]
    html = "".join(m.value for m in onglet.markdown)
    assert html.count('class="resultat"') == 2
    assert "1er Bien" in html and "2ème Bien" in html
    assert "Écart entre les deux" in html and "Le 2ème bien est estimé" in html and "moins cher" in html
    graphiques = [json.loads(g.proto.spec) for g in onglet.get("vega_lite_chart")]
    assert len(graphiques) == 2
    # 5 facteurs par bien : le graphique des facteurs fait 38 px de haut par facteur
    assert [g["height"] for g in graphiques] == [5 * 38, 5 * 38]


def test_comparer_sans_adresse_previent():
    app = lancer_app()
    next(b for b in app.button if b.label == "Comparer").click().run()
    assert not app.exception
    assert "1er Bien : saisissez une adresse" in app.tabs[2].warning[0].value


def test_le_modele_erreurs_par_groupe_et_hors_marche():
    app = lancer_app()
    assert not app.exception
    onglet = app.tabs[3]
    texte = "".join(m.value for m in onglet.markdown)
    for titre in ["Par code postal", "Par surface", "Appartements et maisons", "Les ventes hors marché",
                  "L'année de construction et le DPE", "Limites"]:
        assert titre in texte
    # 1 graphique de comparaison, 1 d'importance, 3 d'erreurs
    assert len(onglet.get("vega_lite_chart")) == 5


def test_a_propos_parcours_et_entonnoir():
    tableau = pd.read_csv(RACINE / "reports" / "entonnoir_nettoyage.csv")
    app = lancer_app()
    assert not app.exception
    onglet = app.tabs[4]
    html = "".join(m.value for m in onglet.markdown)
    assert html.count("<li>") >= 5 and 'class="parcours"' in html
    brutes = f"{tableau['lignes'].iloc[0]:,}".replace(",", "\u202f")
    assert f"De {brutes} lignes" in html
    spec = str(json.loads(onglet.get("vega_lite_chart")[0].proto.spec))
    assert "Données du modèle" in spec and "Étape à appliquer" not in spec  # la règle est appliquée


def test_annee_pre_remplie_et_dpe_repris_du_batiment_sans_saisie(monkeypatch):
    adresse = "10 Rue de la Loge 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))
    # un bâtiment connu, avec année et DPE : celui de la première adresse de la table de l'app
    estimateur = predict.Estimateur()
    batiments = estimateur.batiments.set_index("batiment_groupe_id").dropna(subset=["annee_construction", "dpe_classe"])
    cle = estimateur.adresses[estimateur.adresses.isin(batiments.index)].index[0]
    attendu = batiments.loc[estimateur.adresses[cle]]
    monkeypatch.setattr(predict, "geocoder_complet",
                        lambda texte: {"latitude": 43.611, "longitude": 3.879, "label": adresse, "cle_ban": cle})

    app = lancer_app()
    app.text_input(key="texte").input("10 rue Pré-rempli").run()
    annee = next(n for n in app.number_input if n.label == "Année de construction")
    assert annee.value == int(attendu["annee_construction"])
    # pas de champ DPE : la classe du bâtiment est reprise automatiquement, et la note le dit
    assert not any(s.label == "Classe DPE" for s in app.selectbox)
    note = "".join(m.value for m in app.tabs[0].markdown)
    assert f"La classe DPE du bâtiment est reprise automatiquement : {lettre_dpe(attendu['dpe_classe'])}" in note

    next(b for b in app.button if b.label == "Estimer le prix").click().run()
    assert not app.exception
    r = app.session_state["resultat"]
    assert r["dpe_classe"] == attendu["dpe_classe"] and app.session_state["bien"]["dpe"] == lettre_dpe(attendu["dpe_classe"])
    assert {"annee_construction", "dpe_classe"} <= set(r["facteurs"].index)
    # simulateur : plus de curseur DPE
    assert not app.select_slider and "sim_dpe" not in app.session_state


def test_dpe_inconnu_reste_une_valeur_manquante(monkeypatch):
    # une adresse propre à ce test : le pré-remplissage est gardé en cache d'un test à l'autre
    adresse = "1 Rue Sans Bâtiment 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))
    # adresse introuvable pour le pré-remplissage : bâtiment inconnu
    monkeypatch.setattr(predict, "geocoder_complet", lambda texte: (_ for _ in ()).throw(predict.AdresseIntrouvable()))

    app = lancer_app()
    app.text_input(key="texte").input("10 rue Inconnue").run()
    assert "La classe DPE reste inconnue" in "".join(m.value for m in app.tabs[0].markdown)
    next(b for b in app.button if b.label == "Estimer le prix").click().run()
    assert not app.exception
    assert app.session_state["resultat"]["dpe_classe"] is None and app.session_state["bien"]["dpe"] is None



@pytest.mark.parametrize("demande, verdict", [
    (None, None), (50_000, "En dessous du marché"), (10_000_000, "Nettement au-dessus"),
])
def test_bonne_affaire_apres_estimation(monkeypatch, demande, verdict):
    adresse = "10 Rue de la Loge 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))

    app = lancer_app()
    next(b for b in app.button if b.label == "Écusson").click().run()
    next(b for b in app.button if b.label == "Estimer le prix").click().run()
    champ = app.number_input(key="prix_demande")
    assert champ.value is None and "affaire" not in app.session_state
    if demande is None:
        return
    champ.set_value(demande).run()
    assert not app.exception
    html = "".join(m.value for m in app.markdown)
    analyse = app.session_state["affaire"]
    # verdict ferme, ou sa version prudente si la confiance est faible
    assert analyse["verdict"] in html and verdict.split()[0].lower() in analyse["verdict"].lower()
    assert 'class="jauge-affaire"' in html and "ventes comparables les plus proches" in html
    assert analyse["nb_comparables"] == 10 and 0 <= analyse["moins_chers"] <= 10


def test_bouton_rapport_pdf_apres_estimation(monkeypatch):
    adresse = "10 Rue de la Loge 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))

    app = lancer_app()
    assert not app.get("download_button")  # pas de rapport avant une estimation
    next(b for b in app.button if b.label == "Écusson").click().run()
    next(b for b in app.button if b.label == "Estimer le prix").click().run()
    app.number_input(key="prix_demande").set_value(300_000).run()
    assert not app.exception
    boutons = app.get("download_button")
    assert len(boutons) == 1 and boutons[0].proto.label == "Télécharger le rapport (PDF)"


def test_textes_sans_titres_numerotes_ni_tirets_longs_et_une_seule_mise_en_garde(monkeypatch):
    adresse = "10 Rue de la Loge 34000 Montpellier"
    reponse = {"features": [{"properties": {"citycode": predict.CODE_COMMUNE, "label": adresse}}]}
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FausseReponse(reponse))
    monkeypatch.setattr(predict, "geocoder", lambda texte: (43.6110, 3.8790, adresse))

    app = lancer_app()
    next(b for b in app.button if b.label == "Écusson").click().run()
    next(b for b in app.button if b.label == "Estimer le prix").click().run()
    app.number_input(key="prix_demande").set_value(300_000).run()
    assert not app.exception
    textes = [m.value for m in app.markdown]
    assert not any("—" in t for t in textes)
    assert not any(f"0{i} —" in t or f"0{i} –" in t for t in textes for i in range(1, 10))
    # une seule mise en garde, à côté de l'estimation
    assert sum("ne remplace pas l'avis d'un professionnel" in t for t in textes) == 1
