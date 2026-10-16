"""Application Streamlit : estimer le prix d'un logement à Montpellier."""
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from pydeck.types import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from batiments import CLASSES_DPE, code_dpe, lettre_dpe  # noqa: E402
from affaire import analyser_prix, position_comparables  # noqa: E402
from erreurs import erreur_par_groupe, predictions_test, tranche_surface  # noqa: E402
from marche import MIN_VENTES, evolution_prix, filtrer, grille_prix, prix_selon_pieces  # noqa: E402
from train import ANNEE_TEST, ANNEE_VALIDATION, ANNEES_TRAIN, VARIABLES  # noqa: E402
from predict import AdresseIntrouvable, Estimateur, geocoder_complet, niveau_confiance, suggestions  # noqa: E402
from affichage import (  # noqa: E402
    badge_confiance, bornes_classes, jauge_affaire, carte_resultat, couleur_classe, ecart_estimations, graphique_comparaison,
    graphique_courbe,
    graphique_entonnoir, graphique_erreurs, graphique_evolution, graphique_facteurs, graphique_importance, graphique_pieces, importance_variables,
    legende_carte, regrouper_facteurs, schema_parcours,
)

# à remplir avec tes adresses ; un lien laissé vide n'est pas affiché
LIEN_GITHUB = ""
LIEN_LINKEDIN = ""

RESULTATS_2025 = Path(__file__).resolve().parents[1] / "reports" / "resultats_test_2025.csv"
RAPPORTS = Path(__file__).resolve().parents[1] / "reports"
ENTONNOIR = RAPPORTS / "entonnoir_nettoyage.csv"  # écrit par src/rapport_nettoyage.py
HORS_MARCHE = RAPPORTS / "ventes_hors_marche.json"  # écrit par src/rapport_nettoyage.py
COMPARAISON = RAPPORTS / "comparaison_variables.csv"  # écrit par src/comparer_variables.py
CORRESPONDANCE = RAPPORTS / "correspondance_batiments.csv"  # écrit par src/features.py
FEUILLE_DE_STYLE = Path(__file__).resolve().parent / "style.css"

NOMS = {
    "surface_reelle_bati": "Surface",
    "nombre_pieces_principales": "Nombre de pièces",
    "surface_par_piece": "Surface par pièce",
    "est_maison": "Maison ou appartement",
    "nb_dependances": "Caves et parkings",
    "surface_terrain_totale": "Terrain",
    "emplacement": "Emplacement dans la ville",
    "dist_centre": "Distance au centre",
    "dist_tram": "Distance au tram",
    "dist_ecole": "Distance à une école",
    "dist_parc": "Distance à un parc",
    "dist_commerce": "Distance aux commerces",
    "dist_restaurant": "Distance aux restaurants",
    "nb_ecole_500m": "Écoles à 500 m",
    "nb_commerce_500m": "Commerces à 500 m",
    "nb_restaurant_500m": "Restaurants et cafés à 500 m",
    "annee_construction": "Année de construction",
    "dpe_classe": "Classe DPE du bâtiment",
}
CHOIX_DPE = ["Inconnu", *CLASSES_DPE]

st.set_page_config(page_title="Estimation immobilière — Montpellier", layout="wide")


@st.cache_resource
def charger_estimateur():
    """Chargé une seule fois au démarrage de l'app."""
    return Estimateur()
@st.cache_data(ttl=3600, show_spinner=False)
def chercher_adresses(texte):
    """Suggestions d'adresses, gardées en mémoire une heure."""
    return suggestions(texte)

EXEMPLES = {
    "Écusson": "10 rue de la Loge",
    "Port Marianne": "15 rue de la Cavalade",
    "Beaux-Arts": "10 rue Proudhon",
}


@st.cache_data(ttl=3600, show_spinner=False)
def infos_batiment(adresse):
    """Année de construction et classe DPE du bâtiment de l'adresse, pour pré-remplir le formulaire.

    Renvoie None si l'adresse ne peut pas être localisée : les champs restent alors à remplir.
    """
    try:
        lieu = geocoder_complet(adresse)
    except (AdresseIntrouvable, requests.RequestException):
        return None
    return charger_estimateur().batiment(lieu["latitude"], lieu["longitude"], lieu["cle_ban"])


def note_batiment(infos):
    """Une phrase sous le formulaire : d'où viennent l'année et le DPE pré-remplis."""
    if not infos or not infos["trouve"]:
        return "Bâtiment non retrouvé : année et DPE à renseigner si vous les connaissez."
    dpe = (f"classe DPE médiane de {infos['nb_dpe']} DPE du bâtiment (ADEME)" if infos["nb_dpe"]
           else "aucun DPE connu dans le bâtiment")
    return f"Pré-rempli d'après le bâtiment : année des Fichiers fonciers (BDNB), {dpe}. Modifiable."


def remplir_adresse(exemple):
    """Appelé au clic sur un exemple, avant le réaffichage de la page."""
    st.session_state["texte"] = exemple

def euros(x):
    return f"{x:,.0f} €".replace(",", "\u202f")


def nombre(x):
    return f"{x:,.0f}".replace(",", "\u202f")


def pourcent(x):
    # espace fine insécable avant « % » : le signe ne part jamais seul à la ligne
    return f"{x:.1f}\u202f%".replace(".", ",")


@st.cache_data
def chiffres_cles():
    """Scores du modèle final sur les ventes de 2025, lus dans le rapport de test."""
    resultats = pd.read_csv(RESULTATS_2025, index_col=0)
    final = resultats.loc["LightGBM final"]
    return {
        "erreur_mediane": final["Erreur médiane (%)"],
        "part_10": final["Part à ±10 % (%)"],
        "couverture": final["Couverture fourchette (%)"],
    }


# toute la mise en forme est dans app/style.css
st.markdown(f"<style>{FEUILLE_DE_STYLE.read_text()}</style>", unsafe_allow_html=True)

ventes = charger_estimateur().ventes
nb_ventes = nombre(len(ventes))
annees = ventes["date_mutation"].dt.year
chiffres = chiffres_cles()


def champ_adresse(cle, exemple, libelle="Adresse"):
    """Champ d'adresse avec suggestions officielles. Renvoie l'adresse choisie, ou "" tant qu'il n'y en a pas."""
    texte = st.text_input(
        libelle,
        placeholder=exemple,
        help="Saisissez le début de l'adresse, puis Entrée pour afficher les suggestions.",
        key=cle,
    )
    if not texte.strip():
        return ""
    try:
        propositions = chercher_adresses(texte)
    except requests.RequestException:
        st.warning("Le service d'adresses ne répond pas. Réessayez dans un instant.")
        return ""
    if propositions:
        return st.selectbox("Adresse exacte", propositions, key=f"{cle}_choix")
    if len(texte.strip()) >= 3:
        st.info("Aucune adresse trouvée à Montpellier. Vérifiez l'orthographe.")
    return ""


def hero():
    """Ouverture plein écran : titre, champ d'adresse et chiffres clés. Renvoie l'adresse choisie."""
    with st.container(key="hero"):
        st.markdown(f"""
<div class="bandeau">
  <span class="etiquette">Montpellier</span>
  <span class="etiquette">Estimation immobilière · Ventes {annees.min()}–{annees.max()}</span>
</div>""", unsafe_allow_html=True)

        gauche, droite = st.columns([7, 5], gap="large", vertical_alignment="bottom")
        gauche.markdown(f"""
<h1 class="hero-titre">Ce que vaut<br>un logement<br><em>à Montpellier.</em></h1>
<p class="hero-texte">Une estimation fondée sur {nb_ventes} ventes réelles, avec sa fourchette
et les raisons de son prix.</p>""", unsafe_allow_html=True)

        with droite:
            adresse = champ_adresse("texte", "10 rue de la Loge")
            st.markdown('<span class="etiquette">Exemples</span>', unsafe_allow_html=True)
            with st.container(horizontal=True, key="exemples"):
                for quartier, exemple in EXEMPLES.items():
                    st.button(quartier, on_click=remplir_adresse, args=(exemple,))

        st.markdown(f"""
<div class="chiffres">
  <div class="chiffre"><b>{pourcent(chiffres["erreur_mediane"])}</b><span>Erreur médiane sur les ventes de 2025</span></div>
  <div class="chiffre"><b>{chiffres["part_10"]:.0f}\u202f%</b><span>Des prix estimés à ±10\u202f% près</span></div>
  <div class="chiffre"><b>{chiffres["couverture"] / 10:.0f} sur 10</b><span>Vrais prix compris dans la fourchette</span></div>
</div>""", unsafe_allow_html=True)
    return adresse


def entete(numero, titre, texte=""):
    """En-tête de section éditorial : numéro en petites capitales, titre serif, texte court."""
    paragraphe = f'<p class="texte-court">{texte}</p>' if texte else ""
    return f'<span class="etiquette">{numero}</span><h2 class="titre-section">{titre}</h2>{paragraphe}'


def section_bien(adresse):
    """01 — le formulaire. Enregistre le résultat dans la session quand l'estimation réussit."""
    with st.container(key="section-bien"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "01 — Le bien", "Décrire le logement",
            "L'adresse se saisit en haut de page. Les caves et parkings vendus avec le logement comptent dans le prix.",
        ), unsafe_allow_html=True)
        infos = infos_batiment(adresse) if adresse else None
        annee_connue = infos["annee_construction"] if infos else None
        dpe_connu = lettre_dpe(infos["dpe_classe"]) if infos else None
        with droite, st.form("formulaire", border=False):
            # une rangée de colonnes par ligne : sur téléphone, les champs s'empilent dans le bon ordre
            col1, col2 = st.columns(2, gap="large")
            type_local = col1.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
            surface = col2.number_input("Surface habitable (m²)", min_value=9, max_value=400, value=60)
            col1, col2 = st.columns(2, gap="large")
            pieces = col1.number_input("Nombre de pièces", min_value=1, max_value=15, value=3)
            dependances = col2.number_input("Caves et parkings", min_value=0, max_value=5, value=0)
            col1, col2 = st.columns(2, gap="large")
            terrain = col1.number_input("Terrain (m², pour une maison)", min_value=0, max_value=5000, value=0)
            # la clé dépend de l'adresse : une nouvelle adresse remet les valeurs du bâtiment
            annee = col2.number_input("Année de construction", min_value=1600, max_value=pd.Timestamp.today().year,
                                      value=annee_connue, placeholder="Inconnue", key=f"annee_{adresse}")
            col1, _ = st.columns(2, gap="large")
            dpe = col1.selectbox("Classe DPE", CHOIX_DPE, index=CHOIX_DPE.index(dpe_connu or "Inconnu"),
                                 key=f"dpe_{adresse}")
            if adresse:
                st.markdown(f'<p class="note">{note_batiment(infos)}</p>', unsafe_allow_html=True)
            valider = st.form_submit_button("Estimer le prix", type="primary")

        if not valider:
            return
        if not adresse:
            droite.warning("Saisissez une adresse en haut de page et choisissez-la dans les suggestions.")
            return
        try:
            with st.spinner("Calcul en cours…"):
                r = charger_estimateur().estimer(adresse, type_local, surface, pieces, dependances, terrain,
                                                 annee, code_dpe(dpe))
        except AdresseIntrouvable:
            droite.error("Adresse introuvable dans la commune de Montpellier. Vérifiez l'orthographe.")
            return
        except requests.RequestException:
            droite.error("Le service de géocodage ne répond pas. Réessayez dans un instant.")
            return

    # on garde le résultat en mémoire : il reste affiché quand on bouge les curseurs du simulateur
    st.session_state["resultat"] = r
    st.session_state["bien"] = {
        "type_local": type_local, "surface": surface, "pieces": pieces,
        "dependances": dependances, "terrain": terrain, "annee": annee, "dpe": dpe,
    }
    # nouvelle estimation : les curseurs du simulateur repartent des valeurs saisies
    for cle in ["sim_type", "sim_surface", "sim_pieces", "sim_dep", "sim_dpe", "prix_demande", "affaire"]:
        st.session_state.pop(cle, None)


def confiance(r, bien):
    """Niveau de confiance de l'estimation (élevée, moyenne, faible) et ses raisons."""
    estimateur = charger_estimateur()
    nb_voisins = estimateur.compter_voisins(r["latitude"], r["longitude"], bien["type_local"], bien["surface"])
    largeur = (r["prix_haut"] - r["prix_bas"]) / r["prix"]
    atypique = estimateur.surface_atypique(bien["type_local"], bien["surface"])
    return niveau_confiance(nb_voisins, largeur, atypique)


def section_resultat(r, bien, niveau, raisons):
    """Le prix, sa fourchette et la confiance, sur fond alterné."""

    with st.container(key="section-ombre-resultat"):
        gauche, droite = st.columns([7, 4], gap="large", vertical_alignment="bottom")
        gauche.markdown(
            carte_resultat(r["adresse"], r["prix"], r["prix_m2"], r["prix_bas"], r["prix_haut"], euros),
            unsafe_allow_html=True,
        )
        droite.markdown(badge_confiance(niveau, raisons), unsafe_allow_html=True)
        droite.markdown(
            f'<p class="note">Sur les ventes de 2025, {chiffres["couverture"] / 10:.0f} vrais prix sur 10 '
            "se trouvaient dans une fourchette de ce type.</p>",
            unsafe_allow_html=True,
        )


def section_affaire(r, bien, niveau):
    """« Bonne affaire ? » : le prix demandé face à l'estimation, à la fourchette et aux ventes comparables."""
    with st.container(key="section-affaire"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "02 — Bonne affaire ?", "Le prix demandé",
            "Saisissez le prix d'une annonce : il est comparé à l'estimation, à sa fourchette et aux ventes "
            "comparables du quartier.",
        ), unsafe_allow_html=True)
        with droite:
            demande = st.number_input("Prix demandé (€)", min_value=1_000, max_value=10_000_000, value=None,
                                      step=1_000, placeholder="Facultatif", key="prix_demande")
            if demande is None:
                st.session_state.pop("affaire", None)
                return
            analyse = analyser_prix(demande, r["prix"], r["prix_bas"], r["prix_haut"], niveau)
            comparables = charger_estimateur().comparables(
                r["latitude"], r["longitude"], bien["type_local"], bien["surface"], n=10)
            moins_chers, total = position_comparables(demande / bien["surface"], comparables["prix_m2"])
            analyse.update(prix_demande=demande, moins_chers=moins_chers, nb_comparables=total)
            st.session_state["affaire"] = analyse  # repris dans le rapport PDF

            sens = "au-dessus" if analyse["ecart"] > 0 else "en dessous"
            st.markdown(f"""
<span class="etiquette">Verdict</span>
<p class="verdict">{analyse["verdict"]}</p>
<div class="chiffres">
  <div class="chiffre"><b>{signe(analyse["ecart"], euros(abs(analyse["ecart"])))}</b><span>Écart à l'estimation</span></div>
  <div class="chiffre"><b>{signe(analyse["ecart_pct"], pourcent(abs(analyse["ecart_pct"])))}</b><span>{sens.capitalize()} de l'estimation</span></div>
  <div class="chiffre"><b>{moins_chers} sur {total}</b><span>Ventes comparables moins chères au m² ({euros(demande / bien["surface"])} le m² demandé)</span></div>
</div>""", unsafe_allow_html=True)
            st.markdown(jauge_affaire(r["prix"], r["prix_bas"], r["prix_haut"], demande, euros),
                        unsafe_allow_html=True)
            if analyse["prudent"]:
                st.warning("Confiance faible : peu de ventes comparables ou fourchette large. Ce verdict est une "
                           "indication, à vérifier avec d'autres références.")
            st.markdown(
                '<p class="note">Estimation statistique, pas une expertise : l\'état du logement, l\'étage, la vue, '
                "l'exposition ou les travaux à prévoir ne sont pas pris en compte et peuvent justifier un écart.</p>",
                unsafe_allow_html=True,
            )


def section_pourquoi(r):
    """03 — l'explication SHAP du prix."""
    with st.container(key="section-ombre-pourquoi"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "03 — Lecture", "Pourquoi ce prix",
            f"Le modèle part d'un prix moyen de {euros(r['base'])} pour cette surface. Chaque caractéristique "
            "l'augmente (en noir) ou le diminue (en gris). Méthode : valeurs SHAP.",
        ), unsafe_allow_html=True)
        principaux = regrouper_facteurs(r["facteurs"]).head(8).rename(index=NOMS)
        droite.altair_chart(graphique_facteurs(principaux), width="stretch")


def section_comparables(r):
    """04 — carte et tableau des ventes comparables, numérotées."""
    comp = r["comparables"].reset_index(drop=True)
    numero_rue = comp["adresse_numero"].fillna(0).astype(int).astype(str).replace("0", "")
    comp["adresse"] = (numero_rue + " " + comp["adresse_nom_voie"].str.title()).str.strip()
    comp["repere"] = [str(i + 1) for i in range(len(comp))]

    # une ligne de description par vente, pour l'infobulle
    comp["info"] = [
        f"{n}. {a} : {euros(p)} ({s:.0f} m²)"
        for n, a, p, s in zip(comp["repere"], comp["adresse"], comp["valeur_fonciere"], comp["surface_reelle_bati"])
    ]
    # les ventes au même endroit (même immeuble) sont regroupées sur un seul point
    comp["cle"] = comp["latitude"].round(5).astype(str) + "," + comp["longitude"].round(5).astype(str)
    groupes = comp.groupby("cle", sort=False).agg(
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        repere=("repere", ", ".join),
        info=("info", "\n".join),
    ).reset_index(drop=True)

    estime = pd.DataFrame({
        "lat": [r["latitude"]], "lon": [r["longitude"]],
        "repere": [""], "info": [f"Bien estimé : {r['adresse']}"],
    })
    points = pd.concat([estime, groupes], ignore_index=True)
    # le bien estimé en vert bouteille, les ventes en encre : deux couleurs, rien de plus
    points["couleur"] = [[31, 59, 45]] + [[20, 20, 20]] * len(groupes)

    cercles = pdk.Layer(
        "ScatterplotLayer", data=points, get_position="[lon, lat]", get_fill_color="couleur",
        get_radius=6, radius_min_pixels=9, radius_max_pixels=12, pickable=True,
        stroked=True, get_line_color=[244, 242, 238], line_width_min_pixels=1.5,
    )
    numeros = pdk.Layer(
        "TextLayer", data=points[points["repere"] != ""], get_position="[lon, lat]", get_text="repere",
        get_size=11, get_color=[244, 242, 238], font_family=String("Hanken Grotesk, Arial, sans-serif"),
        get_text_anchor=String("middle"), get_alignment_baseline=String("center"),
    )
    vue = pdk.ViewState(latitude=r["latitude"], longitude=r["longitude"], zoom=16)

    tableau = pd.DataFrame({
        "N°": comp["repere"],
        "Date": comp["date_mutation"].dt.strftime("%m/%Y"),
        "Adresse": comp["adresse"],
        "Surface (m²)": comp["surface_reelle_bati"].round(),
        "Pièces": comp["nombre_pieces_principales"],
        "Prix": comp["valeur_fonciere"].map(euros),
        "Prix au m²": comp["prix_m2"].map(euros),
        "Distance (m)": comp["distance_m"].round(),
    })

    with st.container(key="section-comparables"):
        st.markdown(entete(
            "04 — Références", "Ventes comparables",
            "Les ventes de même type et de surface proche les plus près de l'adresse. "
            "En vert, le bien estimé ; en noir, les ventes numérotées du tableau.",
        ), unsafe_allow_html=True)
        st.pydeck_chart(pdk.Deck(
            layers=[cercles, numeros], initial_view_state=vue, tooltip={"text": "{info}"},
            map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
        ))
        st.dataframe(tableau, hide_index=True, width="stretch")


def section_simulateur(r, bien):
    """05 — « Et si » : on change les caractéristiques, au même emplacement."""
    estimateur = charger_estimateur()
    with st.container(key="section-ombre-simulateur"):
        gauche, droite = st.columns([4, 8], gap="large")
        with gauche:
            st.markdown(entete(
                "05 — Simulation", "Et si…",
                "Modifiez les caractéristiques : le prix est recalculé au même emplacement. Le DPE est celui "
                "du bâtiment (le modèle a appris la classe médiane des logements de l'immeuble) : passer de F "
                "à C donne un ordre de grandeur, pas la valeur de travaux.",
            ), unsafe_allow_html=True)
            sim_type = st.radio(
                "Type de bien", ["Appartement", "Maison"],
                index=["Appartement", "Maison"].index(bien["type_local"]), horizontal=True, key="sim_type",
            )
            sim_surface = st.slider("Surface (m²)", 9, 250, int(bien["surface"]), key="sim_surface")
            sim_pieces = st.slider("Nombre de pièces", 1, 10, int(bien["pieces"]), key="sim_pieces")
            sim_dep = st.slider("Caves et parkings", 0, 3, int(bien["dependances"]), key="sim_dep")
            sim_dpe = st.select_slider("Classe DPE du bâtiment", CHOIX_DPE, value=bien["dpe"], key="sim_dpe")

        prix_sim = estimateur.prix(
            r["latitude"], r["longitude"], sim_type, sim_surface, sim_pieces, sim_dep, bien["terrain"],
            bien["annee"], code_dpe(sim_dpe),
        )
        ecart = prix_sim - r["prix"]
        comparaison = (
            f"{ecart:+,.0f} € par rapport à l'estimation".replace(",", " ").replace("-", "−")
            if round(ecart) else "Identique à l'estimation"
        )
        # courbe : comment évolue le prix avec la surface, à cet endroit
        surfaces = list(range(15, 201, 5))
        prix_courbe = estimateur.prix_selon_surface(
            r["latitude"], r["longitude"], sim_type, sim_pieces, sim_dep, bien["terrain"], surfaces,
            bien["annee"], code_dpe(sim_dpe),
        )
        with droite:
            st.markdown(f"""
<div class="simulation">
  <div><span class="etiquette">Prix simulé</span><b>{euros(prix_sim)}</b><span class="note">{comparaison}</span></div>
  <div><span class="etiquette">Prix au m²</span><b>{euros(prix_sim / sim_surface)}</b></div>
</div>""", unsafe_allow_html=True)
            st.altair_chart(graphique_courbe(surfaces, prix_courbe, sim_surface, prix_sim), width="stretch")
            st.markdown(
                f'<p class="note">Prix estimé selon la surface, pour {sim_pieces} pièce(s) à cette adresse. '
                "Le prix au m² baisse souvent quand la surface augmente.</p>",
                unsafe_allow_html=True,
            )


def onglet_estimer(adresse):
    """Formulaire, puis, après une estimation : résultat, lecture, comparables et simulateur."""
    section_bien(adresse)
    if "resultat" not in st.session_state:
        return
    r, bien = st.session_state["resultat"], st.session_state["bien"]
    niveau, raisons = confiance(r, bien)
    section_resultat(r, bien, niveau, raisons)
    section_affaire(r, bien, niveau)
    section_pourquoi(r)
    section_comparables(r)
    section_simulateur(r, bien)


# ---------- Explorer le marché ----------
# les agrégations sont gardées en mémoire pour chaque combinaison de filtres : l'app reste rapide

@st.cache_data(show_spinner=False)
def bornes_carte():
    """Limites des classes de couleur, calculées une fois sur toutes les zones : une couleur garde
    le même sens quels que soient les filtres."""
    return bornes_classes(grille_prix(charger_estimateur().ventes)["prix_m2"])


@st.cache_data(show_spinner=False)
def zones_marche(type_local, annee):
    """Hexagones d'au moins MIN_VENTES ventes pour ces filtres, et nombre de ventes filtrées."""
    ventes_filtrees = filtrer(charger_estimateur().ventes, type_local, annee)
    return grille_prix(ventes_filtrees), len(ventes_filtrees)


@st.cache_data(show_spinner=False)
def evolution_marche():
    return evolution_prix(charger_estimateur().ventes)


@st.cache_data(show_spinner=False)
def pieces_marche(type_local, annee):
    return prix_selon_pieces(filtrer(charger_estimateur().ventes, type_local, annee))


def description_filtres(type_local, annee):
    """« appartements, 2023 », « tous les biens, 2021–2025 »…"""
    biens = {"Tous": "tous les biens", "Appartement": "appartements", "Maison": "maisons"}[type_local]
    periode = f"{annees.min()}–{annees.max()}" if annee == "Toutes" else str(annee)
    return f"{biens}, {periode}"


def onglet_explorer():
    """Carte des prix par zone, évolution dans le temps et prix selon le nombre de pièces."""
    with st.container(key="section-explorer-carte"):
        gauche, droite = st.columns([4, 8], gap="large")
        with gauche:
            st.markdown(entete(
                "01 — Carte", "Les prix par quartier",
                f"Prix au m² médian des ventes, par zone d'environ 430 m de large. Une zone n'apparaît qu'à "
                f"partir de {MIN_VENTES} ventes : en dessous, sa médiane dépendrait de trop peu de cas.",
            ), unsafe_allow_html=True)
            type_local = st.radio("Type de bien", ["Tous", "Appartement", "Maison"], horizontal=True,
                                  key="marche_type")
            annee = st.selectbox("Année", ["Toutes", *sorted(annees.unique())], key="marche_annee")

        zones, nb = zones_marche(type_local, annee)
        with droite:
            if zones.empty:
                st.info("Aucune zone ne compte assez de ventes avec ces filtres.")
            else:
                bornes = bornes_carte()
                zones = zones.assign(
                    couleur=[couleur_classe(p, bornes) for p in zones["prix_m2"]],
                    info=[f"{euros(p)} le m² · {n} ventes" for p, n in zip(zones["prix_m2"], zones["nb_ventes"])],
                )
                hexagones = pdk.Layer(
                    "PolygonLayer", data=zones, get_polygon="contour", get_fill_color="couleur", opacity=0.85,
                    stroked=True, get_line_color=[244, 242, 238], line_width_min_pixels=1, pickable=True,
                )
                # vue centrée sur l'étendue des zones affichées, pas sur la Comédie : la ville s'étend au nord
                sommets = np.array([s for contour in zones["contour"] for s in contour])
                centre = (sommets.min(axis=0) + sommets.max(axis=0)) / 2
                vue = pdk.ViewState(latitude=centre[1], longitude=centre[0], zoom=11.6)
                st.pydeck_chart(pdk.Deck(
                    layers=[hexagones], initial_view_state=vue, tooltip={"text": "{info}"},
                    map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
                ))
                st.markdown(legende_carte(bornes, euros), unsafe_allow_html=True)
                dans_zones = nombre(zones["nb_ventes"].sum())
                st.markdown(
                    f'<p class="note">{len(zones)} zone{"s" * (len(zones) > 1)} affichée{"s" * (len(zones) > 1)} '
                    f"({description_filtres(type_local, annee)}). "
                    f"Elles regroupent {dans_zones} des {nombre(nb)} ventes ; les autres sont dans des zones trop "
                    "peu fournies.</p>",
                    unsafe_allow_html=True,
                )

    with st.container(key="section-ombre-explorer-evolution"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "02 — Évolution", "Cinq ans de prix",
            "Prix au m² médian de chaque année, pour les appartements et pour les maisons. "
            "Le filtre de type ne s'applique pas ici ; l'année choisie est marquée d'un pointillé.",
        ), unsafe_allow_html=True)
        droite.altair_chart(
            graphique_evolution(evolution_marche(), None if annee == "Toutes" else annee), width="stretch"
        )

    with st.container(key="section-explorer-pieces"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "03 — Taille", "Selon le nombre de pièces",
            f"Pour {description_filtres(type_local, annee)} : la boîte contient la moitié des ventes, le "
            "trait noir est la médiane, le filet va du 1er au 9e décile (8 ventes sur 10). "
            f"Une taille de logement n'est montrée qu'à partir de {MIN_VENTES} ventes.",
        ), unsafe_allow_html=True)
        droite.altair_chart(graphique_pieces(pieces_marche(type_local, annee)), width="stretch")


# ---------- Comparer ----------

BIENS = {"A": "4 boulevard du Jeu de Paume", "B": "20 avenue de Toulouse"}  # exemples affichés en gris
TITRES_BIENS = {"A": "1er Bien", "B": "2ème Bien"}  # ce que voit l'utilisateur ; A et B restent les clés internes


def signe(x, texte):
    """« +12 000 € » ou « −12 000 € » : vrai signe moins, jamais un trait d'union."""
    return ("+" if x > 0 else "−" if x < 0 else "") + texte


def formulaire_comparaison():
    """Deux biens côte à côte. Renvoie {"A": bien, "B": bien} au clic sur « Comparer », sinon None.

    Pas de st.form ici : dans un formulaire, la touche Entrée validerait tout au lieu d'afficher
    les suggestions d'adresse. Chaque colonne contient donc tout son bien, et rien n'est estimé avant le clic.
    """
    biens = {}
    for nom, colonne in zip(BIENS, st.columns(2, gap="large")):
        with colonne:
            st.markdown(f'<h3 class="titre-bien">{TITRES_BIENS[nom]}</h3>', unsafe_allow_html=True)
            biens[nom] = {
                "adresse": champ_adresse(f"cmp_texte_{nom}", BIENS[nom]),
                "type_local": st.radio("Type de bien", ["Appartement", "Maison"], horizontal=True,
                                       key=f"cmp_type_{nom}"),
                "surface": st.number_input("Surface habitable (m²)", 9, 400, 60, key=f"cmp_surface_{nom}"),
                "pieces": st.number_input("Nombre de pièces", 1, 15, 3, key=f"cmp_pieces_{nom}"),
            }
    return biens if st.button("Comparer", type="primary") else None


def estimer_biens(biens):
    """Estime les deux biens ; affiche l'erreur et renvoie None si l'un d'eux échoue."""
    resultats = {}
    for nom, bien in biens.items():
        if not bien["adresse"]:
            st.warning(f"{TITRES_BIENS[nom]} : saisissez une adresse et choisissez-la dans les suggestions.")
            return None
        infos = infos_batiment(bien["adresse"]) or {"annee_construction": None, "dpe_classe": None}
        try:
            resultats[nom] = charger_estimateur().estimer(
                bien["adresse"], bien["type_local"], bien["surface"], bien["pieces"],
                annee_construction=infos["annee_construction"], dpe_classe=infos["dpe_classe"],
            )
        except AdresseIntrouvable:
            st.error(f"{TITRES_BIENS[nom]} : adresse introuvable dans la commune de Montpellier.")
            return None
        except requests.RequestException:
            st.error("Le service de géocodage ne répond pas. Réessayez dans un instant.")
            return None
    return resultats


def section_ecart(resultats):
    """Le verdict : combien le 2ème bien coûte de plus (ou de moins) que le 1er, et si l'écart dépasse l'incertitude."""
    e = ecart_estimations(resultats["A"], resultats["B"])
    if round(e["ecart"]) == 0:
        phrase = "Les deux biens sont estimés au même prix."
    else:
        sens = "plus cher" if e["ecart"] > 0 else "moins cher"
        phrase = f"Le 2ème bien est estimé {euros(abs(e['ecart']))} {sens} que le 1er."
    prudence = (
        "Leurs fourchettes se recouvrent : l'écart reste dans la marge d'incertitude du modèle."
        if e["chevauchement"] else
        "Leurs fourchettes ne se recouvrent pas : l'écart dépasse la marge d'incertitude du modèle."
    )
    with st.container(key="section-comparer-ecart"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("03 — Écart", "Le 2ème face au 1er", phrase), unsafe_allow_html=True)
        droite.markdown(f"""
<div class="chiffres">
  <div class="chiffre"><b>{signe(e["ecart"], euros(abs(e["ecart"])))}</b><span>Écart de prix</span></div>
  <div class="chiffre"><b>{signe(e["ecart_pct"], pourcent(abs(e["ecart_pct"])))}</b><span>En pourcentage du prix du 1er bien</span></div>
  <div class="chiffre"><b>{signe(e["ecart_m2"], euros(abs(e["ecart_m2"])))}</b><span>Écart au m²</span></div>
</div>
<p class="note">{prudence}</p>""", unsafe_allow_html=True)


def onglet_comparer():
    """Deux biens, deux estimations, leurs 5 principaux facteurs, puis l'écart entre les deux."""
    with st.container(key="section-comparer-biens"):
        st.markdown(entete(
            "01 — Les biens", "Comparer deux logements",
            "Deux adresses, deux descriptions : chaque bien est estimé par le même modèle que l'onglet Estimer. "
            "L'année de construction et le DPE sont repris du bâtiment de chaque adresse.",
        ), unsafe_allow_html=True)
        biens = formulaire_comparaison()
        if biens:
            with st.spinner("Calcul en cours…"):
                resultats = estimer_biens(biens)
            if resultats:
                st.session_state["resultats_comparaison"] = resultats

    if "resultats_comparaison" not in st.session_state:
        return
    resultats = st.session_state["resultats_comparaison"]

    with st.container(key="section-ombre-comparer-resultats"):
        st.markdown(entete(
            "02 — Estimations", "Prix et raisons",
            "Pour chaque bien, le prix, sa fourchette, et les 5 caractéristiques qui pèsent le plus sur ce prix "
            "(valeurs SHAP, en milliers d'euros).",
        ), unsafe_allow_html=True)
        for nom, colonne in zip(resultats, st.columns(2, gap="large")):
            r = resultats[nom]
            colonne.markdown(
                f'<div class="comparaison"><h3 class="titre-bien">{TITRES_BIENS[nom]}</h3>'
                + carte_resultat(r["adresse"], r["prix"], r["prix_m2"], r["prix_bas"], r["prix_haut"], euros)
                + "</div>",
                unsafe_allow_html=True,
            )
            principaux = regrouper_facteurs(r["facteurs"]).head(5).rename(index=NOMS)
            colonne.altair_chart(graphique_facteurs(principaux), width="stretch")

    section_ecart(resultats)


def liste_filets(elements):
    """Liste numérotée éditoriale : chaque élément (titre, texte) entre deux filets."""
    lignes = "".join(
        f'<li><span class="liste-numero">{i:02d}</span><div><b>{titre}</b><p>{texte}</p></div></li>'
        for i, (titre, texte) in enumerate(elements, start=1)
    )
    return f'<ol class="liste-filets">{lignes}</ol>'


@st.cache_data(show_spinner=False)
def test_2025():
    """Les ventes de 2025 avec leur estimation : les mêmes prédictions que src/final.py, recalculées une fois."""
    return predictions_test(charger_estimateur().modele, charger_estimateur().ventes)


@st.cache_data(show_spinner=False)
def erreurs_2025():
    """Erreur médiane sur 2025 par code postal, par tranche de surface et par type de bien."""
    test = test_2025()
    return {
        "Par code postal": erreur_par_groupe(test, "code_postal"),
        "Par surface": erreur_par_groupe(test, tranche_surface(test["surface_reelle_bati"])),
        "Appartements et maisons": erreur_par_groupe(test, "type_local"),
    }


@st.cache_data
def rapports_batiment():
    """Ventes hors marché retirées, comparaison sur la validation 2024 et correspondance avec les bâtiments."""
    return (
        json.loads(HORS_MARCHE.read_text()),
        pd.read_csv(COMPARAISON, index_col=0),
        pd.read_csv(CORRESPONDANCE, index_col=0),
    )


def section_erreurs():
    """03 — où le modèle se trompe le plus, sur les ventes de 2025."""
    with st.container(key="section-modele-erreurs"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "03 — Détail", "Où il se trompe",
            f"Erreur médiane sur les ventes de 2025, groupe par groupe. Le pointillé vert marque l'erreur médiane "
            f"de l'ensemble ({pourcent(chiffres['erreur_mediane'])}).",
        ), unsafe_allow_html=True)
        with droite:
            for titre, erreurs in erreurs_2025().items():
                st.markdown(f'<span class="etiquette">{titre}</span>', unsafe_allow_html=True)
                st.altair_chart(graphique_erreurs(erreurs, chiffres["erreur_mediane"]), width="stretch")


def section_hors_marche():
    """04 — l'histoire des ventes hors marché et de la règle de nettoyage qui en a découlé."""
    b, comparaison, _ = rapports_batiment()
    avec = comparaison.loc["Ancien modèle (sans la règle)", "MAE (€)"]
    sans = comparaison.loc["Anciennes variables, données nettoyées", "MAE (€)"]
    immeuble = (
        f" Jusqu'à {b['plus_grand_groupe']} ventes partagent la même adresse et le même prix au m² : sans doute "
        "des lots vendus en bloc." if b["plus_grand_groupe"] > 1 else ""
    )
    with st.container(key="section-ombre-modele-hors-marche"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            # espace insécable : le nombre et « € » ne sont jamais séparés par un retour à la ligne
            "04 — Une enquête", f"Les ventes à {euros(round(b['prix_m2_median_retirees'], -2)).replace(' ', chr(160))} le m²",
            "En étudiant les pires erreurs, une famille de ventes revenait sans cesse. Le modèle n'y était pour rien.",
        ), unsafe_allow_html=True)
        droite.markdown(f"""
<div class="chiffres">
  <div class="chiffre"><b>{nombre(b["ventes_retirees"])}</b><span>Ventes retirées par la règle, sur {nombre(b["ventes_avant"])}</span></div>
  <div class="chiffre"><b>{euros(b["prix_m2_median_retirees"])}</b><span>Prix au m² médian de ces ventes</span></div>
  <div class="chiffre"><b>{euros(b["prix_m2_median_local"])}</b><span>Prix au m² médian local des mêmes biens</span></div>
</div>""", unsafe_allow_html=True)
        droite.markdown(liste_filets([
            ("Le constat", "Les pires erreurs concernaient des logements vendus à une fraction du prix du quartier, "
                           f"alors que le modèle prédisait un prix normal.{immeuble}"),
            ("L'explication", "Ce sont des ventes hors marché : viagers (seul le bouquet est déclaré), ventes "
                              "partielles (nue-propriété, quote-part), ventes entre proches, ventes en bloc. Le prix "
                              "déclaré ne dit pas ce que vaut le logement."),
            ("La règle", f"Retirer une vente si son prix au m² est inférieur à {b['seuil'] * 100:.0f}\u202f% de la "
                         "médiane de son code postal, de son type de bien et de son année. Une règle métier fixée à "
                         "l'avance et appliquée à toutes les années, et non la suppression des pires prédictions, "
                         "qui fausserait l'évaluation."),
            ("Ce qu'elle change", f"La règle est appliquée au modèle actuel. Sur les ventes de 2024 nettoyées, un "
                                  f"modèle appris sans ces ventes se trompe en moyenne de {euros(sans)}, contre "
                                  f"{euros(avec)} pour un modèle appris avec : les retirer "
                                  f"{'n’améliore pas' if sans >= avec else 'améliore à peine'} l'apprentissage. "
                                  "La règle compte surtout pour l'évaluation : les scores ne sont plus calculés sur "
                                  "des prix qui ne disent rien de la valeur d'un logement."),
        ]), unsafe_allow_html=True)


def section_batiment():
    """05 — ce qu'apportent l'année de construction et le DPE, mesuré sur la validation 2024."""
    _, comparaison, correspondance = rapports_batiment()
    toutes = correspondance.loc["Toutes"]
    premiere = str(ANNEES_TRAIN[0])  # l'index mêle années et « Toutes » : il est lu en texte
    base = comparaison.loc["Anciennes variables, données nettoyées"]
    nouveau = comparaison.loc["+ année de construction et DPE"]
    lignes = ""
    for nom, ligne in [("Sans année ni DPE", base), ("Avec année et DPE (modèle retenu)", nouveau)]:
        valeurs = [euros(ligne["MAE (€)"]), pourcent(ligne["MAPE (%)"]), pourcent(ligne["Erreur médiane (%)"]),
                   pourcent(ligne["Part à ±10 % (%)"])]
        lignes += f"<tr><th>{nom}</th>" + "".join(f"<td>{v}</td>" for v in valeurs) + "</tr>"
    entetes = "".join(f"<th>{c}</th>" for c in ["Erreur moyenne", "Erreur moyenne (%)", "Erreur médiane",
                                                 "Prix à ±10\u202f%"])
    with st.container(key="section-modele-batiment"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "05 — Le bâtiment", "Année et DPE",
            "Chaque vente est reliée à son bâtiment (BDNB) : son année de construction, et la classe DPE médiane "
            "des logements de l'immeuble, calculée avec les seuls DPE établis avant la vente (ADEME). Mêmes "
            "réglages, même validation 2024, moyenne de 5 entraînements.",
        ), unsafe_allow_html=True)
        droite.markdown(f"""
<div class="chiffres">
  <div class="chiffre"><b>{nombre(toutes["ventes"] * (1 - toutes["sans_batiment"]))}</b><span>Ventes reliées à leur bâtiment, sur {nombre(toutes["ventes"])}</span></div>
  <div class="chiffre"><b>{pourcent(100 * toutes["avec_annee"])}</b><span>Avec une année de construction</span></div>
  <div class="chiffre"><b>{pourcent(100 * toutes["avec_dpe"])}</b><span>Avec un DPE connu avant la vente</span></div>
</div>
<div class="tableau-defilant"><table class="tableau"><thead><tr><th></th>{entetes}</tr></thead>
<tbody>{lignes}</tbody></table></div>
<p class="note">Un gain modeste : l'erreur moyenne en euros baisse, l'erreur médiane et la part à ±10\u202f% bougent
à peine, l'erreur moyenne en pourcentage augmente un peu. Les DPE ne commencent qu'en juillet 2021 : les ventes de
{premiere} en ont rarement un ({pourcent(100 * correspondance.loc[premiere, "avec_dpe"])}).</p>""",
                        unsafe_allow_html=True)


def onglet_modele():
    """Résultats sur les ventes de 2025, importance des variables et limites."""
    resultats = pd.read_csv(RESULTATS_2025, index_col=0)
    final, baseline = resultats.loc["LightGBM final"], resultats.loc["Baseline"]
    gain = 1 - final["MAE (€)"] / baseline["MAE (€)"]

    with st.container(key="section-modele-resultats"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "01 — Évaluation", "Testé sur 2025",
            "Le modèle a appris sur les ventes de 2021 à 2024, puis a été évalué sur celles de 2025, qu'il "
            "n'avait jamais vues. Point de comparaison : le prix au m² médian du code postal et du type de bien.",
        ), unsafe_allow_html=True)
        with droite:
            st.markdown(f"""
<div class="chiffres">
  <div class="chiffre"><b>{euros(final["MAE (€)"])}</b><span>Erreur moyenne par logement</span></div>
  <div class="chiffre"><b>−{gain * 100:.0f}\u202f%</b><span>D'erreur par rapport au point de comparaison</span></div>
  <div class="chiffre"><b>{pourcent(final["Couverture fourchette (%)"])}</b><span>Des vrais prix dans la fourchette</span></div>
</div>""", unsafe_allow_html=True)
            st.altair_chart(graphique_comparaison(resultats), width="stretch")
            colonnes = ["Erreur moyenne", "Erreur moyenne (%)", "Erreur médiane", "Prix à ±10\u202f%", "Dans la fourchette"]
            lignes = ""
            for nom, ligne in [("Point de comparaison", baseline), ("Modèle final (LightGBM)", final)]:
                couverture = ligne["Couverture fourchette (%)"]
                valeurs = [euros(ligne["MAE (€)"]), pourcent(ligne["MAPE (%)"]), pourcent(ligne["Erreur médiane (%)"]),
                           pourcent(ligne["Part à ±10 % (%)"]), "—" if pd.isna(couverture) else pourcent(couverture)]
                lignes += f"<tr><th>{nom}</th>" + "".join(f"<td>{v}</td>" for v in valeurs) + "</tr>"
            entetes = "".join(f"<th>{c}</th>" for c in colonnes)
            st.markdown(
                f'<div class="tableau-defilant"><table class="tableau"><thead><tr><th></th>{entetes}</tr></thead>'
                f"<tbody>{lignes}</tbody></table></div>",
                unsafe_allow_html=True,
            )

    with st.container(key="section-ombre-modele-variables"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "02 — Variables", "Ce qui fait le prix",
            "Part de chaque variable dans les améliorations apportées par les arbres du modèle (importance "
            "« gain » de LightGBM). Latitude et longitude sont réunies dans « Emplacement dans la ville ».",
        ), unsafe_allow_html=True)
        importance = importance_variables(charger_estimateur().modele).rename(index=NOMS).head(10)
        droite.altair_chart(graphique_importance(importance), width="stretch")

    section_erreurs()
    section_hors_marche()
    section_batiment()

    with st.container(key="section-ombre-modele-limites"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("06 — Limites", "Ce qu'il ignore"), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            ("L'état du bien", "Ni l'étage, ni la vue, ni les travaux. Le DPE utilisé est la classe médiane "
                               "du bâtiment, pas celle du logement."),
            ("Le territoire", "La seule commune de Montpellier."),
            ("Les annexes", "Les prix incluent les caves et parkings vendus avec le logement."),
            ("Le temps", "Appris sur 2021–2024 : un changement récent du marché peut le décaler."),
            ("La fourchette", f"Elle vise 8 vrais prix sur 10 ; sur 2025, elle en contient "
                              f"{pourcent(chiffres['couverture'])}. Elle est un peu trop étroite."),
        ]), unsafe_allow_html=True)


@st.cache_data
def entonnoir():
    """Lignes restantes après chaque étape du nettoyage, lues dans le rapport."""
    return pd.read_csv(ENTONNOIR)


def section_parcours(tableau):
    """02 — le parcours des données, du téléchargement à cette page."""
    modele = tableau.loc[tableau["appliquee"], "lignes"].iloc[-1]
    etapes = [
        {"titre": "Collecte", "texte": f"Ventes DVF {annees.min()}–{annees.max()} et lieux OpenStreetMap.",
         "script": "src/ingest.py", "sortie": "data/raw/"},
        {"titre": "Nettoyage", "texte": f"Une ligne = la vente d'un seul logement : {nombre(modele)} ventes.",
         "script": "src/clean.py", "sortie": "ventes_clean.parquet"},
        {"titre": "Variables", "texte": f"{len(VARIABLES)} variables : surface, pièces, emplacement, distances "
                                        "et densités de lieux.",
         "script": "src/features.py", "sortie": "ventes_features.parquet"},
        {"titre": "Modèles", "texte": f"Appris sur {ANNEES_TRAIN[0]}–{ANNEE_VALIDATION}, testé une seule fois "
                                      f"sur {ANNEE_TEST}.",
         "script": "src/train.py · tune.py · final.py", "sortie": "models/ et reports/"},
        {"titre": "Application", "texte": "Géocodage IGN, estimation, explication SHAP et comparables.",
         "script": "app/streamlit_app.py", "sortie": "cette page"},
    ]
    with st.container(key="section-ombre-a-propos-parcours"):
        st.markdown(entete(
            "02 — Parcours", "Le chemin des données",
            "Chaque étape est un script qui lit le fichier de la précédente et écrit le sien.",
        ), unsafe_allow_html=True)
        st.markdown(schema_parcours(etapes), unsafe_allow_html=True)


def section_entonnoir(tableau):
    """03 — combien de lignes restent après chaque étape du nettoyage."""
    appliquees = tableau[tableau["appliquee"]]
    regle = tableau[~tableau["appliquee"]]
    pointille = (
        f" En pointillé, la règle des ventes hors marché (−{nombre(regle['retirees'].iloc[0])}) : écrite dans "
        "le nettoyage, pas encore appliquée aux données du modèle (voir l'onglet «\u00a0Le modèle\u00a0»)."
        if len(regle) else ""
    )
    with st.container(key="section-a-propos-entonnoir"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "03 — Nettoyage",
            f"De {nombre(appliquees['lignes'].iloc[0])} lignes à {nombre(appliquees['lignes'].iloc[-1])} ventes",
            "Une vente DVF occupe souvent plusieurs lignes (une par lot : logement, cave, parking). Chaque étape "
            "indique ce qui reste, et ce qu'elle a retiré." + pointille,
        ), unsafe_allow_html=True)
        droite.altair_chart(graphique_entonnoir(tableau), width="stretch")


def onglet_a_propos():
    """Démarche, parcours des données, entonnoir du nettoyage et sources."""
    with st.container(key="section-a-propos-demarche"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "01 — Démarche", "Cinq étapes",
            "Un projet de data science mené de la donnée brute à l'application.",
        ), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            ("Collecte", "Toutes les ventes de logements de Montpellier publiées dans DVF."),
            ("Nettoyage", f"Ventes d'un seul logement, sans prix aberrants : {nb_ventes} ventes conservées."),
            ("Variables", "Surface, pièces, type de bien, et l'environnement du logement calculé avec "
                          "OpenStreetMap : centre, tram, écoles, parcs, commerces, restaurants."),
            ("Modèles", "Une référence simple, une régression linéaire, une forêt aléatoire et LightGBM, "
                        "réglé avec Optuna. Deux modèles quantiles donnent la fourchette."),
            ("Explicabilité", "Les valeurs SHAP montrent ce qui fait monter ou baisser chaque estimation."),
        ]), unsafe_allow_html=True)

    tableau = entonnoir()
    section_parcours(tableau)
    section_entonnoir(tableau)

    with st.container(key="section-ombre-a-propos-sources"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("04 — Sources", "Les données"), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            ('<a href="https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/">'
             "Demandes de valeurs foncières</a>", "data.gouv.fr — les ventes immobilières."),
            ('<a href="https://www.openstreetmap.org">OpenStreetMap</a>',
             "Tram, écoles, parcs, commerces et restaurants."),
            ('<a href="https://geoservices.ign.fr/documentation/services/services-geoplateforme/geocodage">'
             "Géocodage de l'IGN</a>", "Suggestions d'adresses et coordonnées."),
        ]), unsafe_allow_html=True)


def liens_auteur():
    """Liens GitHub et LinkedIn, en ignorant ceux qui ne sont pas encore renseignés."""
    liens = {"GitHub": LIEN_GITHUB, "LinkedIn": LIEN_LINKEDIN}
    return " · ".join(f'<a href="{escape(url)}">{nom}</a>' for nom, url in liens.items() if url)


def pied_de_page():
    """Bande noire de fin de page : signature, sources, liens."""
    liens = liens_auteur() or "Liens à venir"
    with st.container(key="pied"):
        st.markdown(f"""
<div class="pied-grille">
  <p class="pied-titre">Ce que vaut un logement <em>à Montpellier.</em></p>
  <div><span class="etiquette">Données</span><p>DVF {annees.min()}–{annees.max()} · OpenStreetMap · IGN</p></div>
  <div><span class="etiquette">Auteur</span><p>{liens}</p></div>
</div>
<p class="pied-mention">Estimation indicative, établie par un modèle statistique. Elle ne remplace pas l'avis d'un professionnel.</p>
""", unsafe_allow_html=True)


# les onglets tout en haut ; l'ouverture (titre, adresse, chiffres clés) appartient à l'onglet Estimer
estimer, explorer, comparer, modele, a_propos = st.tabs(
    ["Estimer", "Explorer le marché", "Comparer", "Le modèle", "À propos"]
)
with estimer:
    onglet_estimer(hero())
with explorer:
    onglet_explorer()
with comparer:
    onglet_comparer()
with modele:
    onglet_modele()
with a_propos:
    onglet_a_propos()
pied_de_page()
