"""Application Streamlit : estimer le prix d'un logement à Montpellier."""
import base64
import hashlib
import importlib
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from pydeck.types import String

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Streamlit relance ce fichier à chaque interaction, mais garde en mémoire les modules du projet déjà importés :
# après une mise à jour (git push), l'app utiliserait encore leurs anciennes versions (ImportError sur une
# fonction nouvelle). On recharge donc chaque module dont le fichier a changé depuis l'exécution précédente,
# dans l'ordre de leurs dépendances. Les empreintes sont gardées dans un petit module qui survit aux exécutions.
MODULES_PROJET = ["clean", "train", "features", "batiments", "marche", "erreurs", "affaire", "predict",
                  "affichage", "rapport_pdf"]
_empreintes = sys.modules.setdefault("_empreintes_immo", types.ModuleType("_empreintes_immo"))
for _nom in MODULES_PROJET:
    _module = sys.modules.get(_nom)
    if _module is None:
        continue
    _empreinte = hashlib.sha1(Path(_module.__file__).read_bytes()).hexdigest()
    if getattr(_empreintes, _nom, _empreinte) != _empreinte:
        importlib.reload(_module)
    setattr(_empreintes, _nom, _empreinte)

from batiments import lettre_dpe  # noqa: E402
from affaire import analyser_prix, position_comparables  # noqa: E402
from erreurs import erreur_par_groupe, predictions_test, tranche_surface  # noqa: E402
from marche import MIN_VENTES, evolution_prix, filtrer, grille_prix, prix_selon_pieces, quartiers  # noqa: E402
from train import ANNEE_TEST, ANNEE_VALIDATION, ANNEES_TRAIN, VARIABLES  # noqa: E402
from predict import AdresseIntrouvable, Estimateur, geocoder_complet, niveau_confiance, suggestions  # noqa: E402
from rapport_pdf import rapport_pdf  # noqa: E402
from affichage import (  # noqa: E402
    adresses_comparables, badge_confiance, bornes_classes, jauge_affaire, carte_resultat, couleur_classe, ecart_estimations, graphique_comparaison,
    graphique_courbe,
    graphique_entonnoir, graphique_erreurs, graphique_evolution, graphique_facteurs, graphique_importance, graphique_pieces, importance_variables,
    legende_carte, rayon_cercle, regrouper_facteurs, regrouper_proches, schema_parcours,
)

AUTEUR = "James Rigault"
LIEN_GITHUB = "https://github.com/James-Rigault/immo-montpellier"
LIEN_METHODE = f"{LIEN_GITHUB}#démarche"

ASSETS = Path(__file__).resolve().parents[1] / "assets"
LOGO = ASSETS / "logo.svg"
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO.read_bytes()).decode()

RESULTATS_2025 = Path(__file__).resolve().parents[1] / "reports" / "resultats_test_2025.csv"
RAPPORTS = Path(__file__).resolve().parents[1] / "reports"
ENTONNOIR = RAPPORTS / "entonnoir_nettoyage.csv"  # écrit par src/rapport_nettoyage.py
HORS_MARCHE = RAPPORTS / "ventes_hors_marche.json"  # écrit par src/rapport_nettoyage.py
COMPARAISON = RAPPORTS / "comparaison_variables.csv"  # écrit par src/comparer_variables.py
CORRESPONDANCE = RAPPORTS / "correspondance_batiments.csv"  # écrit par src/features.py
FEUILLE_DE_STYLE = ASSETS / "style.css"

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


def noms_facteurs(r):
    """Noms lisibles des facteurs SHAP ; la classe DPE, reprise du bâtiment, est précisée entre parenthèses."""
    return {**NOMS, "dpe_classe": f"Classe DPE du bâtiment ({lettre_dpe(r.get('dpe_classe')) or 'inconnue'})"}

st.set_page_config(page_title="Estimation immobilière à Montpellier", page_icon=str(LOGO), layout="wide")


# empreinte du code de l'estimateur : si une mise à jour le modifie, l'estimateur gardé en cache est rechargé
VERSION_CODE = hashlib.sha1(b"".join(
    (SRC / f"{nom}.py").read_bytes() for nom in ["predict", "batiments", "features", "train"]
)).hexdigest()


@st.cache_resource(show_spinner=False)
def estimateur_en_cache(version):
    """Chargé une seule fois par version du code."""
    return Estimateur()


def charger_estimateur():
    return estimateur_en_cache(VERSION_CODE)
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
    """Une phrase sous le formulaire : d'où viennent l'année pré-remplie et la classe DPE du bâtiment."""
    if not infos or not infos["trouve"]:
        return "Bâtiment non retrouvé : indiquez l'année si vous la connaissez. La classe DPE reste inconnue."
    dpe = (f"La classe DPE du bâtiment est reprise automatiquement : {lettre_dpe(infos['dpe_classe'])} "
           f"(médiane de {infos['nb_dpe']} DPE, ADEME)." if infos["nb_dpe"]
           else "Aucun DPE n'est connu pour ce bâtiment.")
    return f"L'année vient des Fichiers fonciers (BDNB), vous pouvez la corriger. {dpe}"


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


@st.cache_data(show_spinner=False)
def chiffres_cles():
    """Scores du modèle final sur les ventes de 2025, lus dans le rapport de test."""
    resultats = pd.read_csv(RESULTATS_2025, index_col=0)
    final = resultats.loc["LightGBM final"]
    return {
        "erreur_mediane": final["Erreur médiane (%)"],
        "part_10": final["Part à ±10 % (%)"],
        "couverture": final["Couverture fourchette (%)"],
    }


@contextmanager
def chargement():
    """Indicateur aux couleurs de l'app, à la place de celui de Streamlit, le temps d'un calcul."""
    emplacement = st.empty()
    emplacement.markdown('<div class="chargement" role="status"><span></span><span></span><span></span>'
                         "<b>Loading...</b></div>", unsafe_allow_html=True)
    try:
        yield
    finally:
        emplacement.empty()


st.markdown(f"<style>{FEUILLE_DE_STYLE.read_text()}</style>", unsafe_allow_html=True)

with chargement():  # le premier lancement charge le modèle et les ventes : quelques secondes
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


def bandeau():
    """Le logo seul, en haut à gauche."""
    st.markdown(f"""
<header class="bandeau">
  <a class="marque" href="./" target="_self"><img src="{LOGO_URI}" alt="Logo"></a>
</header>""", unsafe_allow_html=True)


def entete(titre, texte=""):
    """Titre de section en serif, suivi d'un texte court facultatif."""
    paragraphe = f'<p class="texte-court">{texte}</p>' if texte else ""
    return f'<h2 class="titre-section">{titre}</h2>{paragraphe}'


def titre_carte(titre, texte=""):
    """Titre d'une carte de résultat, plus petit que celui d'une section."""
    paragraphe = f'<p class="note">{texte}</p>' if texte else ""
    return f'<h3 class="titre-carte">{titre}</h3>{paragraphe}'


def formulaire_bien(adresse):
    """Les caractéristiques du bien et le bouton d'estimation. Enregistre le résultat quand l'estimation réussit."""
    infos = infos_batiment(adresse) if adresse else None
    annee_connue = infos["annee_construction"] if infos else None
    # le DPE n'est pas saisi : c'est celui du bâtiment, ou une valeur manquante s'il est inconnu
    dpe_classe = infos["dpe_classe"] if infos else None
    with st.form("formulaire", border=False):
        # une rangée de colonnes par ligne : sur téléphone, les champs s'empilent dans le bon ordre
        col1, col2 = st.columns(2)
        type_local = col1.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
        surface = col2.number_input("Surface habitable (m²)", min_value=9, max_value=400, value=60)
        col1, col2 = st.columns(2)
        pieces = col1.number_input("Nombre de pièces", min_value=1, max_value=15, value=3)
        dependances = col2.number_input("Caves et parkings", min_value=0, max_value=5, value=0)
        col1, col2 = st.columns(2)
        terrain = col1.number_input("Terrain (m², pour une maison)", min_value=0, max_value=5000, value=0)
        # la clé dépend de l'adresse : une nouvelle adresse remet les valeurs du bâtiment
        annee = col2.number_input("Année de construction", min_value=1600, max_value=pd.Timestamp.today().year,
                                  value=annee_connue, placeholder="Inconnue", key=f"annee_{adresse}")
        if adresse:
            st.markdown(f'<p class="note">{note_batiment(infos)}</p>', unsafe_allow_html=True)
        valider = st.form_submit_button("Estimer le prix", type="primary")

    if not valider:
        return
    if not adresse:
        st.warning("Saisissez une adresse et choisissez-la dans les suggestions.")
        return
    try:
        with chargement():
            r = charger_estimateur().estimer(adresse, type_local, surface, pieces, dependances, terrain,
                                             annee, dpe_classe)
    except AdresseIntrouvable:
        st.error("Adresse introuvable dans la commune de Montpellier. Vérifiez l'orthographe.")
        return
    except requests.RequestException:
        st.error("Le service de géocodage ne répond pas. Réessayez dans un instant.")
        return

    # on garde le résultat en mémoire : il reste affiché quand on bouge les curseurs du simulateur
    r["dpe_classe"] = dpe_classe
    st.session_state["resultat"] = r
    st.session_state["bien"] = {
        "type_local": type_local, "surface": surface, "pieces": pieces,
        "dependances": dependances, "terrain": terrain, "annee": annee, "dpe": lettre_dpe(dpe_classe),
    }
    # nouvelle estimation : les curseurs du simulateur repartent des valeurs saisies
    for cle in ["sim_type", "sim_surface", "sim_pieces", "sim_dep", "prix_demande", "affaire"]:
        st.session_state.pop(cle, None)


def accueil():
    """Titre à gauche, carte de saisie à droite, puis les chiffres clés."""
    with st.container(key="hero"):
        gauche, droite = st.columns([5, 7], gap="large", vertical_alignment="center")
        gauche.markdown("""
<h1 class="hero-titre">Ce que vaut votre logement <em>à Montpellier.</em></h1>
<p class="hero-texte">Entrez l'adresse et les caractéristiques du logement pour obtenir son prix estimé et sa
fourchette.</p>""", unsafe_allow_html=True)

        with droite:
            with st.container(key="carte-saisie"):
                adresse = champ_adresse("texte", "10 rue de la Loge")
                with st.container(horizontal=True, key="exemples"):
                    st.markdown('<span class="etiquette">Exemples :</span>', unsafe_allow_html=True)
                    for quartier, exemple in EXEMPLES.items():
                        st.button(quartier, on_click=remplir_adresse, args=(exemple,))
                formulaire_bien(adresse)
            st.markdown(
                f'<p class="source-donnees">L\'estimation repose sur {nb_ventes} ventes réelles enregistrées à '
                f"Montpellier entre {annees.min()} et {annees.max()} (données DVF).</p>",
                unsafe_allow_html=True,
            )

        st.markdown(f"""
<div class="chiffres">
  <div class="chiffre"><b>{nb_ventes}</b><span>Ventes analysées</span></div>
  <div class="chiffre"><b>{pourcent(chiffres["erreur_mediane"])}</b><span>Erreur médiane sur les ventes de 2025</span></div>
  <div class="chiffre"><b>{chiffres["part_10"]:.0f}\u202f%</b><span>Des estimations à moins de 10\u202f% du vrai prix</span></div>
</div>""", unsafe_allow_html=True)


def confiance(r, bien):
    """Niveau de confiance de l'estimation (élevée, moyenne, faible) et ses raisons."""
    estimateur = charger_estimateur()
    nb_voisins = estimateur.compter_voisins(r["latitude"], r["longitude"], bien["type_local"], bien["surface"])
    largeur = (r["prix_haut"] - r["prix_bas"]) / r["prix"]
    atypique = estimateur.surface_atypique(bien["type_local"], bien["surface"])
    return niveau_confiance(nb_voisins, largeur, atypique)


def carte_prix(r, niveau, raisons):
    """La carte principale : le prix, sa fourchette et la confiance. Renvoie l'emplacement du bouton PDF."""
    with st.container(key="carte-prix"):
        gauche, droite = st.columns([7, 4], gap="large", vertical_alignment="bottom")
        gauche.markdown(
            carte_resultat(r["adresse"], r["prix"], r["prix_m2"], r["prix_bas"], r["prix_haut"], euros),
            unsafe_allow_html=True,
        )
        droite.markdown(badge_confiance(niveau, raisons), unsafe_allow_html=True)
        droite.markdown(
            '<p class="note">C\'est une estimation statistique : le modèle ne connaît ni l\'état du logement, ni '
            "l'étage, ni la vue. Elle ne remplace pas l'avis d'un professionnel.</p>",
            unsafe_allow_html=True,
        )
        # rempli plus bas, une fois « Bonne affaire ? » passé : le PDF reprend le prix demandé
        return droite.empty()


def bouton_rapport(emplacement, r, bien, niveau, raisons):
    """Bouton « Télécharger le rapport (PDF) » : le rapport est généré en mémoire, en une fraction de seconde."""
    facteurs = regrouper_facteurs(r["facteurs"]).head(8).rename(index=noms_facteurs(r))
    pdf = rapport_pdf(r, bien, niveau, raisons, facteurs, chiffres, st.session_state.get("affaire"))
    nom = "estimation-" + "-".join(r["adresse"].lower().split()[:4]) + ".pdf"
    emplacement.download_button("Télécharger le rapport (PDF)", pdf, file_name=nom, mime="application/pdf",
                                key="rapport_pdf")


def carte_affaire(r, bien, niveau):
    """« Bonne affaire ? » : le prix demandé face à l'estimation, à la fourchette et aux ventes comparables."""
    with st.container(key="carte-affaire"):
        st.markdown(titre_carte(
            "Bonne affaire ?",
            "Vous avez vu ce logement dans une annonce ? Entrez le prix demandé pour le comparer à l'estimation.",
        ), unsafe_allow_html=True)
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

        if round(analyse["ecart"]) == 0:
            ecart = "Le prix demandé est égal à l'estimation."
        else:
            sens = "au-dessus" if analyse["ecart"] > 0 else "en dessous"
            ecart = (f"Le prix demandé est {euros(abs(analyse['ecart']))} {sens} de l'estimation "
                     f"({signe(analyse['ecart_pct'], pourcent(abs(analyse['ecart_pct'])))}).")
        st.markdown(f"""
<p class="verdict">{analyse["verdict"]}</p>
<p class="texte-section">{ecart} Au m² ({euros(demande / bien["surface"])}), il est plus cher que {moins_chers} des
{total} ventes comparables les plus proches.</p>""", unsafe_allow_html=True)
        st.markdown(jauge_affaire(r["prix"], r["prix_bas"], r["prix_haut"], demande, euros), unsafe_allow_html=True)
        if analyse["prudent"]:
            st.warning("Confiance faible : peu de ventes comparables ou fourchette large. Ce verdict est une "
                       "indication, à vérifier avec d'autres références.")


def carte_pourquoi(r):
    """L'explication SHAP du prix."""
    with st.container(key="carte-pourquoi"):
        st.markdown(titre_carte(
            "Pourquoi ce prix",
            f"Le modèle part d'un prix moyen de {euros(r['base'])} pour cette surface. Chaque caractéristique "
            "l'augmente (en noir) ou le diminue (en gris). Ces effets sont calculés avec les valeurs SHAP.",
        ), unsafe_allow_html=True)
        principaux = regrouper_facteurs(r["facteurs"]).head(8).rename(index=noms_facteurs(r))
        st.altair_chart(graphique_facteurs(principaux), width="stretch")


def carte_comparables(r):
    """Carte et tableau des ventes comparables, numérotées."""
    comp = r["comparables"].reset_index(drop=True)
    comp["adresse"] = adresses_comparables(comp)
    comp["repere"] = [str(i + 1) for i in range(len(comp))]

    # une ligne de description par vente, pour l'infobulle
    comp["info"] = [
        f"{n}. {a} : {euros(p)} ({s:.0f} m²)"
        for n, a, p, s in zip(comp["repere"], comp["adresse"], comp["valeur_fonciere"], comp["surface_reelle_bati"])
    ]
    # les ventes dont les cercles se toucheraient à ce zoom partagent un seul cercle, agrandi selon
    # le nombre de ventes : les numéros ne se recouvrent jamais
    zoom = 16
    comp["cle"] = regrouper_proches(list(comp["latitude"]), list(comp["longitude"]), zoom)
    groupes = comp.groupby("cle", sort=False).agg(
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        repere=("repere", ", ".join),
        info=("info", "\n".join),
        nb=("repere", "size"),
    ).reset_index(drop=True)
    groupes["rayon"] = groupes["nb"].map(rayon_cercle)

    cercles = pdk.Layer(
        "ScatterplotLayer", data=groupes, get_position="[lon, lat]", get_fill_color=[42, 61, 47],
        get_radius="rayon", radius_units=String("pixels"), pickable=True,
        stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5,
    )
    numeros = pdk.Layer(
        "TextLayer", data=groupes, get_position="[lon, lat]", get_text="repere",
        get_size=11, get_color=[255, 255, 255], font_family=String("Hanken Grotesk, Arial, sans-serif"),
        get_text_anchor=String("middle"), get_alignment_baseline=String("center"),
    )
    # le logement estimé : un anneau doré, au-dessus des cercles mais sous les numéros, visible même s'il
    # tombe sur une vente
    estime = pd.DataFrame({"lat": [r["latitude"]], "lon": [r["longitude"]], "info": [f"Bien estimé : {r['adresse']}"],
                           "rayon": [groupes["rayon"].max() + 7]})
    anneau = pdk.Layer(
        "ScatterplotLayer", data=estime, get_position="[lon, lat]", get_radius="rayon", radius_units=String("pixels"),
        filled=False, stroked=True, get_line_color=[201, 168, 106], line_width_min_pixels=3, pickable=True,
    )
    vue = pdk.ViewState(latitude=r["latitude"], longitude=r["longitude"], zoom=zoom)

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

    with st.container(key="carte-comparables"):
        st.markdown(titre_carte(
            "Ventes comparables",
            "Les ventes du même type et de surface proche les plus près de l'adresse. L'anneau doré marque le "
            "logement estimé, les cercles verts renvoient aux numéros du tableau.",
        ), unsafe_allow_html=True)
        st.pydeck_chart(pdk.Deck(
            layers=[cercles, anneau, numeros], initial_view_state=vue, tooltip={"text": "{info}"},
            map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
        ))
        st.dataframe(tableau, hide_index=True, width="stretch")


def section_simulateur(r, bien):
    """« Et si » : on change les caractéristiques, au même emplacement."""
    estimateur = charger_estimateur()
    with st.container(key="section-simulateur"):
        gauche, droite = st.columns([4, 8], gap="large")
        with gauche:
            st.markdown(entete(
                "Et si…",
                "Modifiez les caractéristiques : le prix est recalculé au même emplacement.",
            ), unsafe_allow_html=True)
            sim_type = st.radio(
                "Type de bien", ["Appartement", "Maison"],
                index=["Appartement", "Maison"].index(bien["type_local"]), horizontal=True, key="sim_type",
            )
            sim_surface = st.slider("Surface (m²)", 9, 250, int(bien["surface"]), key="sim_surface")
            sim_pieces = st.slider("Nombre de pièces", 1, 10, int(bien["pieces"]), key="sim_pieces")
            sim_dep = st.slider("Caves et parkings", 0, 3, int(bien["dependances"]), key="sim_dep")

        prix_sim = estimateur.prix(
            r["latitude"], r["longitude"], sim_type, sim_surface, sim_pieces, sim_dep, bien["terrain"],
            bien["annee"], r["dpe_classe"],
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
            bien["annee"], r["dpe_classe"],
        )
        with droite:
            st.markdown(f"""
<div class="simulation">
  <div><span class="etiquette">Prix simulé</span><b>{euros(prix_sim)}</b><span class="note">{comparaison}</span></div>
  <div><span class="etiquette">Au m²</span><b>{euros(prix_sim / sim_surface)}</b></div>
</div>""", unsafe_allow_html=True)
            st.altair_chart(graphique_courbe(surfaces, prix_courbe, sim_surface, prix_sim), width="stretch")
            st.markdown(
                f'<p class="note">Prix estimé selon la surface, pour {sim_pieces} pièce(s) à cette adresse. '
                "Le prix au m² baisse souvent quand la surface augmente.</p>",
                unsafe_allow_html=True,
            )


def onglet_estimer():
    """La saisie, puis, après une estimation : le prix, l'explication, « Bonne affaire ? », les comparables
    et le simulateur."""
    accueil()
    if "resultat" not in st.session_state:
        return
    r, bien = st.session_state["resultat"], st.session_state["bien"]
    niveau, raisons = confiance(r, bien)
    with st.container(key="section-resultats"):
        emplacement = carte_prix(r, niveau, raisons)
        gauche, droite = st.columns([7, 5], gap="medium")
        with gauche:
            carte_pourquoi(r)
        with droite:
            carte_affaire(r, bien, niveau)
        bouton_rapport(emplacement, r, bien, niveau, raisons)
        carte_comparables(r)
    section_simulateur(r, bien)


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
def noms_quartiers():
    return quartiers()


@st.cache_data(show_spinner=False)
def evolution_marche():
    return evolution_prix(charger_estimateur().ventes)


@st.cache_data(show_spinner=False)
def pieces_marche(type_local, annee):
    return prix_selon_pieces(filtrer(charger_estimateur().ventes, type_local, annee))


def sur_telephone():
    """Vrai sur un téléphone, d'après le navigateur : la carte, plus étroite, est alors un peu dézoomée
    pour que toute la ville et les noms des quartiers y tiennent."""
    navigateur = st.context.headers.get("User-Agent", "")
    return any(mot in navigateur for mot in ("iPhone", "Android", "Mobile"))


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
                "Prix au m² par quartier",
                f"Prix au m² médian par zone d'environ 430 m de large. Une zone n'est affichée qu'à partir de "
                f"{MIN_VENTES} ventes, sinon sa médiane ne voudrait pas dire grand-chose.",
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
                    info=[f"{euros(p)} le m², {n} ventes" for p, n in zip(zones["prix_m2"], zones["nb_ventes"])],
                )
                hexagones = pdk.Layer(
                    "PolygonLayer", data=zones, get_polygon="contour", get_fill_color="couleur", opacity=0.85,
                    stroked=True, get_line_color=[244, 242, 238], line_width_min_pixels=1, pickable=True,
                )
                # vue centrée sur l'étendue des zones affichées, pas sur la Comédie : la ville s'étend au nord
                sommets = np.array([s for contour in zones["contour"] for s in contour])
                centre = (sommets.min(axis=0) + sommets.max(axis=0)) / 2
                vue = pdk.ViewState(latitude=centre[1], longitude=centre[0], zoom=10.9 if sur_telephone() else 11.6)
                # les noms des quartiers, au-dessus des hexagones qui cachent ceux du fond de carte
                etiquettes = noms_quartiers()
                noms = pdk.Layer(
                    "TextLayer", data=etiquettes, get_position="[longitude, latitude]", get_text="nom",
                    get_size=11 if sur_telephone() else 13, get_color=[20, 20, 20],
                    font_family=String("Hanken Grotesk, Arial, sans-serif"),
                    font_weight=600, get_text_anchor=String("middle"), get_alignment_baseline=String("center"),
                    background=True, get_background_color=[255, 255, 255, 235], background_padding=[7, 4, 7, 4],
                    # « auto » : deck.gl dessine aussi les accents (É, ô) ; String() le passe comme valeur, pas comme
                    # expression (une liste de caractères casserait sur l'apostrophe de « Croix d'Argent »)
                    character_set=String("auto"),
                )
                st.pydeck_chart(pdk.Deck(
                    layers=[hexagones, noms], initial_view_state=vue, tooltip={"text": "{info}"},
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
            "Évolution des prix",
            "Prix au m² médian de chaque année, pour les appartements et pour les maisons. Le filtre de type ne "
            "s'applique pas à ce graphique. L'année choisie est marquée d'un pointillé.",
        ), unsafe_allow_html=True)
        droite.altair_chart(
            graphique_evolution(evolution_marche(), None if annee == "Toutes" else annee), width="stretch"
        )

    with st.container(key="section-explorer-pieces"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "Prix au m² selon le nombre de pièces",
            f"Pour {description_filtres(type_local, annee)}. La boîte contient la moitié des ventes, le trait noir "
            "est la médiane et la ligne fine va du 1er au 9e décile. Une taille de logement n'est affichée qu'à "
            f"partir de {MIN_VENTES} ventes.",
        ), unsafe_allow_html=True)
        droite.altair_chart(graphique_pieces(pieces_marche(type_local, annee)), width="stretch")


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
            resultats[nom]["dpe_classe"] = infos["dpe_classe"]  # pour nommer le facteur DPE dans l'explication
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
    detail = "" if round(e["ecart"]) == 0 else (
        f" Cela fait {signe(e['ecart_pct'], pourcent(abs(e['ecart_pct'])))} par rapport au 1er bien, et "
        f"{euros(abs(e['ecart_m2']))} de {'plus' if e['ecart_m2'] > 0 else 'moins'} au m²."
    )
    with st.container(key="section-comparer-ecart"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("Écart entre les deux"), unsafe_allow_html=True)
        droite.markdown(f'<p class="texte-section">{phrase}{detail} {prudence}</p>', unsafe_allow_html=True)


def onglet_comparer():
    """Deux biens, deux estimations, leurs 5 principaux facteurs, puis l'écart entre les deux."""
    with st.container(key="section-comparer-biens"):
        st.markdown(entete(
            "Comparer deux logements",
            "Chaque logement est estimé avec le même modèle que dans l'onglet Estimer. L'année de construction "
            "et le DPE viennent du bâtiment de chaque adresse.",
        ), unsafe_allow_html=True)
        biens = formulaire_comparaison()
        if biens:
            with chargement():
                resultats = estimer_biens(biens)
            if resultats:
                st.session_state["resultats_comparaison"] = resultats

    if "resultats_comparaison" not in st.session_state:
        return
    resultats = st.session_state["resultats_comparaison"]

    with st.container(key="section-ombre-comparer-resultats"):
        st.markdown(entete(
            "Les deux estimations",
            "Sous chaque prix, les 5 caractéristiques qui pèsent le plus, en milliers d'euros.",
        ), unsafe_allow_html=True)
        for nom, colonne in zip(resultats, st.columns(2, gap="medium")):
            r = resultats[nom]
            with colonne.container(key=f"carte-bien-{nom}"):
                st.markdown(
                    f'<div class="comparaison"><h3 class="titre-bien">{TITRES_BIENS[nom]}</h3>'
                    + carte_resultat(r["adresse"], r["prix"], r["prix_m2"], r["prix_bas"], r["prix_haut"], euros)
                    + "</div>",
                    unsafe_allow_html=True,
                )
                principaux = regrouper_facteurs(r["facteurs"]).head(5).rename(index=noms_facteurs(r))
                st.altair_chart(graphique_facteurs(principaux), width="stretch")

    section_ecart(resultats)


def liste_filets(elements):
    """Liste entre deux filets. Un élément est une phrase, ou un couple (titre, texte)."""
    lignes = "".join(
        f'<li><span class="titre-liste">{e[0]}</span><p>{e[1]}</p></li>' if isinstance(e, tuple) else f"<li>{e}</li>"
        for e in elements
    )
    return f'<ul class="liste-filets">{lignes}</ul>'


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


@st.cache_data(show_spinner=False)
def rapports_batiment():
    """Ventes hors marché retirées, comparaison sur la validation 2024 et correspondance avec les bâtiments."""
    return (
        json.loads(HORS_MARCHE.read_text()),
        pd.read_csv(COMPARAISON, index_col=0),
        pd.read_csv(CORRESPONDANCE, index_col=0),
    )


def section_erreurs():
    """Où le modèle se trompe le plus, sur les ventes de 2025."""
    with st.container(key="section-modele-erreurs"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "Où il se trompe le plus",
            f"Erreur médiane sur les ventes de 2025, par code postal, par surface et par type de bien. Le "
            f"pointillé vert marque l'erreur médiane globale ({pourcent(chiffres['erreur_mediane'])}).",
        ), unsafe_allow_html=True)
        with droite:
            for titre, erreurs in erreurs_2025().items():
                st.markdown(f'<span class="etiquette">{titre}</span>', unsafe_allow_html=True)
                st.altair_chart(graphique_erreurs(erreurs, chiffres["erreur_mediane"]), width="stretch")


def section_hors_marche():
    """Les ventes hors marché et la règle de nettoyage qui en a découlé."""
    b, comparaison, _ = rapports_batiment()
    avec = comparaison.loc["Ancien modèle (sans la règle)", "MAE (€)"]
    sans = comparaison.loc["Anciennes variables, données nettoyées", "MAE (€)"]
    immeuble = (
        f" Certaines partageaient la même adresse et le même prix au m² (jusqu'à {b['plus_grand_groupe']} "
        "ventes) : sans doute des lots vendus ensemble." if b["plus_grand_groupe"] > 1 else ""
    )
    effet = "ne change presque rien" if abs(sans - avec) < 0.02 * avec else "change peu"
    with st.container(key="section-ombre-modele-hors-marche"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("Les ventes hors marché"), unsafe_allow_html=True)
        droite.markdown(f"""
<p class="texte-section">En regardant les plus grosses erreurs du modèle, j'ai trouvé des logements vendus autour de
{euros(b["prix_m2_median_retirees"])} le m² (médiane), alors que des biens du même type se vendaient environ
{euros(b["prix_m2_median_local"])} le m² dans le même secteur.{immeuble} Ce sont surtout des viagers (seul le
bouquet est déclaré), des ventes partielles comme la nue-propriété, ou des ventes entre proches. Le prix déclaré ne
correspond pas à la valeur du logement.</p>
<p class="texte-section">J'ai donc ajouté une règle au nettoyage : une vente est retirée si son prix au m² est
inférieur à {b["seuil"] * 100:.0f}\u202f% de la médiane de son code postal, de son type de bien et de son année.
Elle retire {nombre(b["ventes_retirees"])} ventes sur {nombre(b["ventes_avant"])}. La règle est fixée à l'avance et
s'applique à toutes les années : je n'ai pas retiré les ventes en fonction des erreurs du modèle, ce qui aurait faussé
l'évaluation.</p>
<p class="texte-section">Sur les ventes de 2024, apprendre sans ces ventes {effet} à l'erreur moyenne
({euros(sans)}, contre {euros(avec)} avec elles). La règle sert surtout à mesurer l'erreur sur de vrais prix de
marché.</p>""", unsafe_allow_html=True)


def section_batiment():
    """Ce qu'apportent l'année de construction et le DPE, mesuré sur la validation 2024."""
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
        gauche.markdown(entete("L'année de construction et le DPE"), unsafe_allow_html=True)
        droite.markdown(f"""
<p class="texte-section">Chaque vente est reliée à son bâtiment dans la BDNB, ce qui donne l'année de construction et
la classe DPE médiane des logements de l'immeuble. Seuls les DPE établis avant la vente sont pris en compte (données
ADEME). {nombre(toutes["ventes"] * (1 - toutes["sans_batiment"]))} ventes sur {nombre(toutes["ventes"])} ont été
reliées à un bâtiment, {pourcent(100 * toutes["avec_annee"])} ont une année de construction et
{pourcent(100 * toutes["avec_dpe"])} un DPE.</p>
<p class="texte-section">Le tableau compare le modèle avec et sans ces deux variables sur les ventes de 2024 (mêmes
réglages, moyenne de 5 entraînements).</p>
<div class="tableau-defilant"><table class="tableau"><thead><tr><th></th>{entetes}</tr></thead>
<tbody>{lignes}</tbody></table></div>
<p class="note">Le gain est faible : l'erreur moyenne en euros baisse un peu, l'erreur moyenne en pourcentage monte
un peu. Les DPE n'existent dans ces données que depuis juillet 2021, donc seulement
{pourcent(100 * correspondance.loc[premiere, "avec_dpe"])} des ventes de {premiere} en ont un.</p>""",
                        unsafe_allow_html=True)


def onglet_modele():
    """Résultats sur les ventes de 2025, importance des variables et limites."""
    resultats = pd.read_csv(RESULTATS_2025, index_col=0)
    final, baseline = resultats.loc["LightGBM final"], resultats.loc["Baseline"]
    gain = 1 - final["MAE (€)"] / baseline["MAE (€)"]

    with st.container(key="section-modele-resultats"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "Résultats sur 2025",
            "Le modèle a appris sur les ventes de 2021 à 2024, puis a été testé une seule fois sur celles de 2025, "
            "qu'il n'avait jamais vues.",
        ), unsafe_allow_html=True)
        with droite:
            st.markdown(f"""
<p class="texte-section">Pour avoir un point de comparaison, j'ai aussi estimé chaque vente avec le prix au m² médian
de son code postal et de son type de bien. Le modèle se trompe en moyenne de {euros(final["MAE (€)"])} par logement,
soit {gain * 100:.0f}\u202f% de moins que cette méthode simple.</p>""", unsafe_allow_html=True)
            st.altair_chart(graphique_comparaison(resultats), width="stretch")
            colonnes = ["Erreur moyenne", "Erreur moyenne (%)", "Erreur médiane", "Prix à ±10\u202f%", "Dans la fourchette"]
            lignes = ""
            for nom, ligne in [("Point de comparaison", baseline), ("Modèle final (LightGBM)", final)]:
                couverture = ligne["Couverture fourchette (%)"]
                valeurs = [euros(ligne["MAE (€)"]), pourcent(ligne["MAPE (%)"]), pourcent(ligne["Erreur médiane (%)"]),
                           pourcent(ligne["Part à ±10 % (%)"]), "aucune" if pd.isna(couverture) else pourcent(couverture)]
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
            "Ce qui compte le plus",
            "Part de chaque variable dans l'importance « gain » de LightGBM. Latitude et longitude sont "
            "regroupées dans « Emplacement dans la ville ».",
        ), unsafe_allow_html=True)
        importance = importance_variables(charger_estimateur().modele).rename(index=NOMS).head(10)
        droite.altair_chart(graphique_importance(importance), width="stretch")

    section_erreurs()
    section_hors_marche()
    section_batiment()

    with st.container(key="section-ombre-modele-limites"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("Limites"), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            "Le DPE utilisé est la classe médiane du bâtiment, pas celle du logement.",
            "Le modèle ne couvre que la commune de Montpellier.",
            "Les prix de vente incluent les caves et parkings vendus avec le logement.",
            f"Il a appris sur {ANNEES_TRAIN[0]}–{ANNEE_VALIDATION} : un changement récent du marché peut le décaler.",
            f"La fourchette devrait contenir 8 vrais prix sur 10. Sur 2025, elle en contient "
            f"{pourcent(chiffres['couverture'])} : elle est un peu trop étroite.",
        ]), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def entonnoir():
    """Lignes restantes après chaque étape du nettoyage, lues dans le rapport."""
    return pd.read_csv(ENTONNOIR)


def section_parcours(tableau):
    """Le parcours des données, du téléchargement à cette page."""
    modele = tableau.loc[tableau["appliquee"], "lignes"].iloc[-1]
    etapes = [
        {"titre": "Collecte", "texte": f"Ventes DVF {annees.min()}–{annees.max()} et lieux OpenStreetMap.",
         "script": "src/ingest.py", "sortie": "data/raw/"},
        {"titre": "Nettoyage", "texte": f"Une ligne par vente d'un seul logement, soit {nombre(modele)} ventes.",
         "script": "src/clean.py", "sortie": "ventes_clean.parquet"},
        {"titre": "Variables", "texte": f"{len(VARIABLES)} variables : surface, pièces, emplacement, lieux "
                                        "proches, année de construction et DPE du bâtiment.",
         "script": "src/features.py", "sortie": "ventes_features.parquet"},
        {"titre": "Modèles", "texte": f"Régression linéaire, forêt aléatoire et LightGBM comparés sur "
                                      f"{ANNEE_VALIDATION}. LightGBM, réglé avec Optuna, est testé une fois sur "
                                      f"{ANNEE_TEST}.",
         "script": "src/train.py, tune.py, final.py", "sortie": "models/ et reports/"},
        {"titre": "Application", "texte": "Géocodage de l'adresse, estimation et explication avec SHAP.",
         "script": "app/streamlit_app.py", "sortie": "cette page"},
    ]
    with st.container(key="section-ombre-a-propos-parcours"):
        st.markdown(entete(
            "Comment le projet est construit",
            "Chaque étape est un script qui lit le fichier de l'étape précédente et écrit le sien.",
        ), unsafe_allow_html=True)
        st.markdown(schema_parcours(etapes), unsafe_allow_html=True)


def section_entonnoir(tableau):
    """Combien de lignes restent après chaque étape du nettoyage."""
    appliquees = tableau[tableau["appliquee"]]
    regle = tableau[~tableau["appliquee"]]
    pointille = (
        f" En pointillé, la règle des ventes hors marché (−{nombre(regle['retirees'].iloc[0])}), pas encore "
        "appliquée aux données du modèle."
        if len(regle) else ""
    )
    with st.container(key="section-a-propos-entonnoir"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            f"De {nombre(appliquees['lignes'].iloc[0])} lignes à {nombre(appliquees['lignes'].iloc[-1])} ventes",
            "Une vente DVF occupe souvent plusieurs lignes, une par lot (logement, cave, parking). Le graphique "
            "montre ce qui reste après chaque étape du nettoyage et ce qu'elle retire." + pointille,
        ), unsafe_allow_html=True)
        droite.altair_chart(graphique_entonnoir(tableau), width="stretch")


def onglet_a_propos():
    """Parcours des données, entonnoir du nettoyage et sources."""
    tableau = entonnoir()
    section_parcours(tableau)
    section_entonnoir(tableau)

    with st.container(key="section-a-propos-code"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("Méthode et code"), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            (f'<a href="{LIEN_METHODE}" target="_blank">Méthode</a>',
             "La démarche complète, les choix de nettoyage et les résultats, dans le README du projet."),
            (f'<a href="{LIEN_GITHUB}" target="_blank">GitHub</a>',
             "Le code source, les notebooks et les tests."),
        ]), unsafe_allow_html=True)

    with st.container(key="section-ombre-a-propos-sources"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("Sources"), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            ('<a href="https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/">'
             "Demandes de valeurs foncières (DVF)</a>", "Les ventes immobilières, publiées sur data.gouv.fr."),
            ('<a href="https://www.openstreetmap.org">OpenStreetMap</a>',
             "Arrêts de tram, écoles, parcs, commerces et restaurants."),
            ('<a href="https://bdnb.io">Base de données nationale des bâtiments (BDNB)</a>',
             "L'année de construction et le lien entre ventes, bâtiments et DPE."),
            ('<a href="https://data.ademe.fr/datasets/dpe03existant">DPE des logements existants (ADEME)</a>',
             "Les diagnostics de performance énergétique depuis juillet 2021."),
            ('<a href="https://geoservices.ign.fr/documentation/services/services-geoplateforme/geocodage">'
             "Géocodage de l'IGN</a>", "Les suggestions d'adresses et leurs coordonnées."),
        ]), unsafe_allow_html=True)


def pied_de_page():
    """Bande verte de fin de page : le logo, les sources, l'auteur."""
    with st.container(key="pied"):
        st.markdown(f"""
<div class="pied-grille">
  <div class="pied-marque"><img src="{LOGO_URI}" alt="Logo"></div>
  <div><span class="etiquette">Données</span><p>DVF {annees.min()}–{annees.max()}, OpenStreetMap, BDNB, ADEME, IGN</p></div>
  <div><span class="etiquette">Auteur</span><p>{AUTEUR}</p></div>
</div>
""", unsafe_allow_html=True)


bandeau()
estimer, explorer, comparer, modele, a_propos = st.tabs(
    ["Estimer", "Explorer le marché", "Comparer", "Le modèle", "À propos"]
)
with estimer:
    onglet_estimer()
with explorer:
    onglet_explorer()
with comparer:
    onglet_comparer()
with modele:
    onglet_modele()
with a_propos:
    onglet_a_propos()
pied_de_page()
