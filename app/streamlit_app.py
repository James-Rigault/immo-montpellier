"""Application Streamlit : estimer le prix d'un logement à Montpellier."""
import sys
from html import escape
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from pydeck.types import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import AdresseIntrouvable, Estimateur, niveau_confiance, suggestions  # noqa: E402
from affichage import (  # noqa: E402
    badge_confiance, carte_resultat, graphique_comparaison, graphique_courbe, graphique_facteurs,
    graphique_importance, importance_variables, regrouper_facteurs,
)

# à remplir avec tes adresses ; un lien laissé vide n'est pas affiché
LIEN_GITHUB = ""
LIEN_LINKEDIN = ""

RESULTATS_2025 = Path(__file__).resolve().parents[1] / "reports" / "resultats_test_2025.csv"
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
}

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


def remplir_adresse(exemple):
    """Appelé au clic sur un exemple, avant le réaffichage de la page."""
    st.session_state["texte"] = exemple

def euros(x):
    return f"{x:,.0f} €".replace(",", "\u202f")


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
nb_ventes = f"{len(ventes):,}".replace(",", "\u202f")
annees = ventes["date_mutation"].dt.year
chiffres = chiffres_cles()


def hero():
    """Ouverture plein écran : titre, champ d'adresse et chiffres clés. Renvoie l'adresse choisie."""
    adresse = ""
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
            texte = st.text_input(
                "Adresse",
                placeholder="10 rue de la Loge",
                help="Saisissez le début de l'adresse, puis Entrée pour afficher les suggestions.",
                key="texte",
            )
            if texte.strip():
                try:
                    propositions = chercher_adresses(texte)
                except requests.RequestException:
                    propositions = []
                    st.warning("Le service d'adresses ne répond pas. Réessayez dans un instant.")
                if propositions:
                    adresse = st.selectbox("Adresse exacte", propositions)
                elif len(texte.strip()) >= 3:
                    st.info("Aucune adresse trouvée à Montpellier. Vérifiez l'orthographe.")
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
        with droite, st.form("formulaire", border=False):
            # une rangée de colonnes par ligne : sur téléphone, les champs s'empilent dans le bon ordre
            col1, col2 = st.columns(2, gap="large")
            type_local = col1.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
            surface = col2.number_input("Surface habitable (m²)", min_value=9, max_value=400, value=60)
            col1, col2 = st.columns(2, gap="large")
            pieces = col1.number_input("Nombre de pièces", min_value=1, max_value=15, value=3)
            dependances = col2.number_input("Caves et parkings", min_value=0, max_value=5, value=0)
            col1, _ = st.columns(2, gap="large")
            terrain = col1.number_input("Terrain (m², pour une maison)", min_value=0, max_value=5000, value=0)
            valider = st.form_submit_button("Estimer le prix", type="primary")

        if not valider:
            return
        if not adresse:
            droite.warning("Saisissez une adresse en haut de page et choisissez-la dans les suggestions.")
            return
        try:
            with st.spinner("Calcul en cours…"):
                r = charger_estimateur().estimer(adresse, type_local, surface, pieces, dependances, terrain)
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
        "dependances": dependances, "terrain": terrain,
    }
    # nouvelle estimation : les curseurs du simulateur repartent des valeurs saisies
    for cle in ["sim_type", "sim_surface", "sim_pieces", "sim_dep"]:
        st.session_state.pop(cle, None)


def section_resultat(r, bien):
    """Le prix, sa fourchette et la confiance, sur fond alterné."""
    estimateur = charger_estimateur()
    nb_voisins = estimateur.compter_voisins(r["latitude"], r["longitude"], bien["type_local"], bien["surface"])
    largeur = (r["prix_haut"] - r["prix_bas"]) / r["prix"]
    atypique = estimateur.surface_atypique(bien["type_local"], bien["surface"])
    niveau, raisons = niveau_confiance(nb_voisins, largeur, atypique)

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


def section_pourquoi(r):
    """02 — l'explication SHAP du prix."""
    with st.container(key="section-pourquoi"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete(
            "02 — Lecture", "Pourquoi ce prix",
            f"Le modèle part d'un prix moyen de {euros(r['base'])} pour cette surface. Chaque caractéristique "
            "l'augmente (en noir) ou le diminue (en gris). Méthode : valeurs SHAP.",
        ), unsafe_allow_html=True)
        principaux = regrouper_facteurs(r["facteurs"]).head(8).rename(index=NOMS)
        droite.altair_chart(graphique_facteurs(principaux), width="stretch")


def section_comparables(r):
    """03 — carte et tableau des ventes comparables, numérotées."""
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

    with st.container(key="section-ombre-comparables"):
        st.markdown(entete(
            "03 — Références", "Ventes comparables",
            "Les ventes de même type et de surface proche les plus près de l'adresse. "
            "En vert, le bien estimé ; en noir, les ventes numérotées du tableau.",
        ), unsafe_allow_html=True)
        st.pydeck_chart(pdk.Deck(
            layers=[cercles, numeros], initial_view_state=vue, tooltip={"text": "{info}"},
            map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
        ))
        st.dataframe(tableau, hide_index=True, width="stretch")


def section_simulateur(r, bien):
    """04 — « Et si » : on change les caractéristiques, au même emplacement."""
    estimateur = charger_estimateur()
    with st.container(key="section-simulateur"):
        gauche, droite = st.columns([4, 8], gap="large")
        with gauche:
            st.markdown(entete(
                "04 — Simulation", "Et si…",
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
            r["latitude"], r["longitude"], sim_type, sim_surface, sim_pieces, sim_dep, bien["terrain"]
        )
        ecart = prix_sim - r["prix"]
        comparaison = (
            f"{ecart:+,.0f} € par rapport à l'estimation".replace(",", " ").replace("-", "−")
            if round(ecart) else "Identique à l'estimation"
        )
        # courbe : comment évolue le prix avec la surface, à cet endroit
        surfaces = list(range(15, 201, 5))
        prix_courbe = estimateur.prix_selon_surface(
            r["latitude"], r["longitude"], sim_type, sim_pieces, sim_dep, bien["terrain"], surfaces
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
    section_resultat(r, bien)
    section_pourquoi(r)
    section_comparables(r)
    section_simulateur(r, bien)


def liste_filets(elements):
    """Liste numérotée éditoriale : chaque élément (titre, texte) entre deux filets."""
    lignes = "".join(
        f'<li><span class="liste-numero">{i:02d}</span><div><b>{titre}</b><p>{texte}</p></div></li>'
        for i, (titre, texte) in enumerate(elements, start=1)
    )
    return f'<ol class="liste-filets">{lignes}</ol>'


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

    with st.container(key="section-modele-limites"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("03 — Limites", "Ce qu'il ignore"), unsafe_allow_html=True)
        droite.markdown(liste_filets([
            ("L'état du bien", "Ni l'étage, ni la vue, ni les travaux, ni le diagnostic énergétique."),
            ("Le territoire", "La seule commune de Montpellier."),
            ("Les annexes", "Les prix incluent les caves et parkings vendus avec le logement."),
            ("Le temps", "Appris sur 2021–2024 : un changement récent du marché peut le décaler."),
            ("La fourchette", f"Elle vise 8 vrais prix sur 10 ; sur 2025, elle en contient "
                              f"{pourcent(chiffres['couverture'])}. Elle est un peu trop étroite."),
        ]), unsafe_allow_html=True)


def onglet_a_propos():
    """Démarche et sources des données."""
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

    with st.container(key="section-ombre-a-propos-sources"):
        gauche, droite = st.columns([4, 8], gap="large")
        gauche.markdown(entete("02 — Sources", "Les données"), unsafe_allow_html=True)
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


adresse = hero()
estimer, modele, a_propos = st.tabs(["Estimer", "Le modèle", "À propos"])
with estimer:
    onglet_estimer(adresse)
with modele:
    onglet_modele()
with a_propos:
    onglet_a_propos()
pied_de_page()
