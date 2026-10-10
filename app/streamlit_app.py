"""Application Streamlit : estimer le prix d'un logement à Montpellier."""
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from pydeck.types import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import AdresseIntrouvable, Estimateur, niveau_confiance, suggestions  # noqa: E402
from affichage import (  # noqa: E402
    STYLE, badge_confiance, carte_resultat, graphique_comparaison, graphique_facteurs, graphique_importance,
    importance_variables, regrouper_facteurs,
)

# à remplir avec tes adresses ; un lien laissé vide n'est pas affiché
LIEN_GITHUB = ""
LIEN_LINKEDIN = ""

RESULTATS_2025 = Path(__file__).resolve().parents[1] / "reports" / "resultats_test_2025.csv"

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

st.set_page_config(page_title="Estimateur immobilier Montpellier", page_icon="🏠")


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
    return f"{x:.1f} %".replace(".", ",")


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


# styles des éléments faits sur mesure (badges, carte de résultat...), en plus du thème
st.markdown(STYLE, unsafe_allow_html=True)

st.title("Estimateur de prix immobilier à Montpellier")
ventes = charger_estimateur().ventes
nb_ventes = f"{len(ventes):,}".replace(",", "\u202f")
annees = ventes["date_mutation"].dt.year
st.markdown(
    "Tape une adresse et décris le bien : un modèle d'apprentissage automatique estime son prix, "
    f"à partir de **{nb_ventes} ventes réelles** enregistrées entre {annees.min()} et {annees.max()}."
)
chiffres = chiffres_cles()
st.markdown(f"""
<div class="badges">
  <div class="badge"><b>{pourcent(chiffres["erreur_mediane"])}</b><span>erreur médiane sur les ventes de 2025</span></div>
  <div class="badge"><b>{chiffres["part_10"]:.0f} %</b><span>des prix estimés à ±10 % près</span></div>
  <div class="badge"><b>{chiffres["couverture"] / 10:.0f} sur 10</b><span>vrais prix dans la fourchette</span></div>
</div>
""", unsafe_allow_html=True)


def onglet_estimer():
    """Adresse, formulaire, résultat, comparables et simulateur « Et si »."""
    texte = st.text_input(
        "Adresse à Montpellier",
        placeholder="ex. : 10 rue de la Loge",
        help="Tape le début de l'adresse puis appuie sur Entrée pour voir les suggestions.",
        key="texte",
    )
    st.caption("Ou essaie un exemple :")
    for colonne, (quartier, exemple) in zip(st.columns(len(EXEMPLES)), EXEMPLES.items()):
        colonne.button(quartier, on_click=remplir_adresse, args=(exemple,), width="stretch")
    adresse = ""
    if texte.strip():
        try:
            propositions = chercher_adresses(texte)
        except requests.RequestException:
            propositions = []
            st.warning("Le service d'adresses ne répond pas. Réessaie dans un instant.")
        if propositions:
            adresse = st.selectbox("Choisis l'adresse exacte", propositions)
        elif len(texte.strip()) >= 3:
            st.info("Aucune adresse trouvée à Montpellier. Vérifie l'orthographe.")
    with st.form("formulaire"):

        col1, col2 = st.columns(2)
        type_local = col1.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
        surface = col2.number_input("Surface habitable (m²)", min_value=9, max_value=400, value=60)
        pieces = col1.number_input("Nombre de pièces", min_value=1, max_value=15, value=3)
        dependances = col2.number_input("Caves et parkings", min_value=0, max_value=5, value=0)
        terrain = st.number_input("Surface de terrain (m², pour une maison)", min_value=0, max_value=5000, value=0)
        valider = st.form_submit_button("Estimer le prix", type="primary")

    if valider:
        if not adresse:
            st.warning("Tape une adresse et choisis-la dans la liste des suggestions.")
            return
        try:
            with st.spinner("Calcul en cours..."):
                r = charger_estimateur().estimer(adresse, type_local, surface, pieces, dependances, terrain)
        except AdresseIntrouvable:
            st.error("Adresse introuvable dans la commune de Montpellier. Vérifie l'orthographe.")
            return
        except requests.RequestException:
            st.error("Le service de géocodage ne répond pas. Réessaie dans un instant.")
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

    if "resultat" in st.session_state:
        r = st.session_state["resultat"]
        st.markdown(
            carte_resultat(r["adresse"], r["prix"], r["prix_m2"], r["prix_bas"], r["prix_haut"], euros),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Fourchette probable : sur les ventes de 2025, environ {chiffres['couverture'] / 10:.0f} vrais prix "
            "sur 10 tombaient dans ce type de fourchette."
        )
        bien = st.session_state["bien"]
        estimateur = charger_estimateur()
        nb_voisins = estimateur.compter_voisins(r["latitude"], r["longitude"], bien["type_local"], bien["surface"])
        largeur = (r["prix_haut"] - r["prix_bas"]) / r["prix"]
        atypique = estimateur.surface_atypique(bien["type_local"], bien["surface"])
        niveau, raisons = niveau_confiance(nb_voisins, largeur, atypique)
        st.markdown(badge_confiance(niveau, raisons), unsafe_allow_html=True)
        st.markdown("#### Pourquoi ce prix ?")
        principaux = regrouper_facteurs(r["facteurs"]).head(8).rename(index=NOMS)
        st.altair_chart(graphique_facteurs(principaux), width="stretch")
        st.caption(
            f"Le modèle part d'un prix moyen de {euros(r['base'])} pour cette surface, puis chaque "
            "caractéristique l'augmente (en vert) ou le diminue (en rouge). Survole une barre pour voir sa valeur. "
            "Méthode : valeurs SHAP."
        )
        st.markdown("#### Ventes comparables à proximité")
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

        bien = pd.DataFrame({
            "lat": [r["latitude"]], "lon": [r["longitude"]],
            "repere": [""], "info": [f"Bien estimé : {r['adresse']}"],
        })
        points = pd.concat([bien, groupes], ignore_index=True)
        points["couleur"] = [[198, 40, 40]] + [[21, 101, 192]] * len(groupes)

        cercles = pdk.Layer(
            "ScatterplotLayer", data=points, get_position="[lon, lat]", get_fill_color="couleur",
            get_radius=8, radius_min_pixels=11, radius_max_pixels=16, pickable=True,
            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5,
        )
        numeros = pdk.Layer(
            "TextLayer", data=points[points["repere"] != ""], get_position="[lon, lat]", get_text="repere",
            get_size=12, get_color=[255, 255, 255],
            get_text_anchor=String("middle"), get_alignment_baseline=String("center"),
        )
        vue = pdk.ViewState(latitude=r["latitude"], longitude=r["longitude"], zoom=16)
        st.pydeck_chart(pdk.Deck(layers=[cercles, numeros], initial_view_state=vue, tooltip={"text": "{info}"}))
        st.caption("En rouge : le bien estimé. En bleu, numérotées : les ventes du tableau. Survole un point pour voir son adresse.")

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


        st.dataframe(tableau, hide_index=True, width="stretch")
        st.markdown("#### Et si… ? Simule des changements")
        st.caption("Modifie les caractéristiques du bien : le prix est recalculé au même emplacement.")
        bien = st.session_state["bien"]
        estimateur = charger_estimateur()

        col1, col2 = st.columns(2)
        sim_type = col1.radio(
            "Type de bien", ["Appartement", "Maison"],
            index=["Appartement", "Maison"].index(bien["type_local"]), horizontal=True, key="sim_type",
        )
        sim_surface = col2.slider("Surface (m²)", 9, 250, int(bien["surface"]), key="sim_surface")
        sim_pieces = col1.slider("Nombre de pièces", 1, 10, int(bien["pieces"]), key="sim_pieces")
        sim_dep = col2.slider("Caves et parkings", 0, 3, int(bien["dependances"]), key="sim_dep")

        prix_sim = estimateur.prix(
            r["latitude"], r["longitude"], sim_type, sim_surface, sim_pieces, sim_dep, bien["terrain"]
        )
        ecart = prix_sim - r["prix"]
        col1, col2 = st.columns(2)
        col1.metric(
            "Prix simulé", euros(prix_sim),
            delta=f"{ecart:+,.0f} € par rapport à l'estimation".replace(",", "\u202f") if round(ecart) else None,
        )
        col2.metric("Prix au m² simulé", euros(prix_sim / sim_surface).replace("€", "€/m²"))

        # courbe : comment évolue le prix avec la surface, à cet endroit
        surfaces = list(range(15, 201, 5))
        prix_courbe = estimateur.prix_selon_surface(
            r["latitude"], r["longitude"], sim_type, sim_pieces, sim_dep, bien["terrain"], surfaces
        )
        courbe = pd.DataFrame({"Surface (m²)": surfaces, "Prix estimé (€)": prix_courbe}).set_index("Surface (m²)")
        st.line_chart(courbe)
        st.caption(
            f"Prix estimé selon la surface, pour un bien de {sim_pieces} pièce(s) à cette adresse. "
            "Le prix n'est pas toujours proportionnel à la surface : le prix au m² baisse souvent pour les grandes surfaces."
        )

def onglet_modele():
    """Résultats sur les ventes de 2025, importance des variables et limites."""
    st.markdown("#### Résultats sur les ventes de 2025")
    st.markdown(
        "Le modèle a appris sur les ventes de 2021 à 2024, puis a été testé sur les ventes de 2025, "
        "qu'il n'avait jamais vues. On le compare à une **baseline** simple : le prix au m² médian "
        "du code postal et du type de bien, multiplié par la surface."
    )
    resultats = pd.read_csv(RESULTATS_2025, index_col=0)
    st.altair_chart(graphique_comparaison(resultats), width="stretch")
    tableau = pd.DataFrame({
        "Erreur moyenne": resultats["MAE (€)"].map(euros),
        "Erreur moyenne (%)": resultats["MAPE (%)"].map(pourcent),
        "Erreur médiane": resultats["Erreur médiane (%)"].map(pourcent),
        "Prix à ±10 %": resultats["Part à ±10 % (%)"].map(pourcent),
        "Vrai prix dans la fourchette": resultats["Couverture fourchette (%)"].map(
            lambda x: "—" if pd.isna(x) else pourcent(x)
        ),
    }, index=resultats.index.rename("Modèle")).rename(index={"LightGBM final": "Modèle final (LightGBM)"})
    st.dataframe(tableau, width="stretch")
    gain = 1 - resultats.loc["LightGBM final", "MAE (€)"] / resultats.loc["Baseline", "MAE (€)"]
    st.caption(f"L'erreur moyenne en euros baisse de {gain:.0%} par rapport à la baseline.".replace("%", " %"))

    st.markdown("#### Les variables les plus utiles au modèle")
    importance = importance_variables(charger_estimateur().modele).rename(index=NOMS).head(10)
    st.altair_chart(graphique_importance(importance), width="stretch")
    st.caption(
        "Part de chaque variable dans les améliorations apportées par les arbres du modèle (importance « gain » "
        "de LightGBM). Latitude et longitude sont regroupées dans « Emplacement dans la ville »."
    )

    st.markdown("#### Limites")
    st.markdown(
        "- Il ne connaît ni l'état du bien, ni l'étage, ni la vue, ni le DPE.\n"
        "- Il ne couvre que la commune de Montpellier.\n"
        "- Les prix incluent les caves et parkings vendus avec le logement.\n"
        "- Il a appris sur 2021-2024 : un changement récent du marché peut le décaler.\n"
        f"- La fourchette vise 8 vrais prix sur 10, mais n'en contient que {pourcent(chiffres['couverture'])} "
        "sur les ventes de 2025 : elle est un peu trop étroite."
    )


def onglet_a_propos():
    """Démarche, sources des données et liens."""
    st.markdown("#### La démarche")
    st.markdown(
        "1. **Collecte** : toutes les ventes de logements de Montpellier publiées dans DVF.\n"
        "2. **Nettoyage** : ventes d'un seul logement, sans prix aberrants, soit "
        f"{nb_ventes} ventes conservées.\n"
        "3. **Variables** : surface, pièces, type de bien, et environnement du logement calculé avec "
        "OpenStreetMap (distance au centre, au tram, aux écoles, aux parcs, commerces et restaurants).\n"
        "4. **Modèles** : comparaison d'une baseline, d'une régression linéaire, d'une forêt aléatoire et de "
        "LightGBM, réglé ensuite avec Optuna ; deux modèles quantiles donnent la fourchette.\n"
        "5. **Explicabilité** : valeurs SHAP pour montrer ce qui fait monter ou baisser chaque estimation."
    )
    st.markdown("#### Sources des données")
    st.markdown(
        "- [Demandes de valeurs foncières (DVF)](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/), "
        "data.gouv.fr : les ventes immobilières.\n"
        "- [OpenStreetMap](https://www.openstreetmap.org) : tram, écoles, parcs, commerces et restaurants.\n"
        "- [Service de géocodage de l'IGN](https://geoservices.ign.fr/documentation/services/services-geoplateforme/geocodage) "
        ": suggestions d'adresses et coordonnées GPS."
    )
    st.markdown("#### Me retrouver")
    liens = liens_auteur()
    st.markdown(liens if liens else "Liens à venir.")


def liens_auteur():
    """Liens GitHub et LinkedIn, en ignorant ceux qui ne sont pas encore renseignés."""
    liens = {"GitHub": LIEN_GITHUB, "LinkedIn": LIEN_LINKEDIN}
    return " · ".join(f"[{nom}]({url})" for nom, url in liens.items() if url)


estimer, modele, a_propos = st.tabs(["Estimer", "Le modèle", "À propos"])
with estimer:
    onglet_estimer()
with modele:
    onglet_modele()
with a_propos:
    onglet_a_propos()

st.divider()
pied = "Projet de data science · données DVF et OpenStreetMap"
if liens_auteur():
    pied += " · " + liens_auteur()
st.caption(pied)
