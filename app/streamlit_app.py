"""Application Streamlit : estimer le prix d'un logement à Montpellier."""


import pydeck as pdk
from pydeck.types import String
import sys
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import AdresseIntrouvable, Estimateur  # noqa: E402

NOMS = {
    "surface_reelle_bati": "Surface",
    "nombre_pieces_principales": "Nombre de pièces",
    "surface_par_piece": "Surface par pièce",
    "est_maison": "Maison ou appartement",
    "nb_dependances": "Caves et parkings",
    "surface_terrain_totale": "Terrain",
    "latitude": "Position (nord-sud)",
    "longitude": "Position (est-ouest)",
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


def euros(x):
    return f"{x:,.0f} €".replace(",", "\u202f")


st.title("Estimateur de prix immobilier à Montpellier")
st.caption(
        "En rouge : le bien estimé. En bleu : les ventes du tableau. "
        "Plusieurs numéros sur un même point = ventes dans le même immeuble. Survole un point pour voir les adresses."
    )


with st.form("formulaire"):
    adresse = st.text_input("Adresse à Montpellier", placeholder="Exemple : 10 rue de la Loge, Montpellier")
    col1, col2 = st.columns(2)
    type_local = col1.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
    surface = col2.number_input("Surface habitable (m²)", min_value=9, max_value=400, value=60)
    pieces = col1.number_input("Nombre de pièces", min_value=1, max_value=15, value=3)
    dependances = col2.number_input("Caves et parkings", min_value=0, max_value=5, value=0)
    terrain = st.number_input("Surface de terrain (m², pour une maison)", min_value=0, max_value=5000, value=0)
    valider = st.form_submit_button("Estimer le prix", type="primary")

if valider:
    if not adresse.strip():
        st.warning("Indique une adresse.")
        st.stop()
    try:
        with st.spinner("Calcul en cours..."):
            r = charger_estimateur().estimer(adresse, type_local, surface, pieces, dependances, terrain)
    except AdresseIntrouvable:
        st.error("Adresse introuvable dans la commune de Montpellier. Vérifie l'orthographe.")
        st.stop()
    except requests.RequestException:
        st.error("Le service de géocodage ne répond pas. Réessaie dans un instant.")
        st.stop()

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
    st.subheader(r["adresse"])
    col1, col2 = st.columns(2)
    col1.metric("Prix estimé", euros(r["prix"]))
    col2.metric("Prix au m²", euros(r["prix_m2"]).replace("€", "€/m²"))
    st.write(f"Fourchette probable : **{euros(r['prix_bas'])}** à **{euros(r['prix_haut'])}**")
    st.caption("Sur les ventes de 2025, environ 8 vrais prix sur 10 tombaient dans ce type de fourchette.")

    st.markdown("#### Pourquoi ce prix ?")
    principaux = r["facteurs"].head(8).rename(index=NOMS)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    couleurs = ["#2e7d32" if v > 0 else "#c62828" for v in principaux.values]
    barres = ax.barh(principaux.index, principaux.values, color=couleurs)

    # axe avec une partie négative toujours visible, en k€
    ecart = principaux.abs().max()
    ax.set_xlim(min(principaux.min(), -0.25 * ecart) * 1.3, max(principaux.max(), 0.25 * ecart) * 1.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:+.0f} k€"))

    # valeur écrite au bout de chaque barre
    ax.bar_label(barres, labels=[f"{v / 1000:+.0f} k€" for v in principaux.values], padding=3, fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Effet sur le prix (€)")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Le modèle part d'un prix moyen de {euros(r['base'])} pour cette surface, puis chaque "
        "caractéristique l'augmente (en vert) ou le diminue (en rouge). Méthode : valeurs SHAP."
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
    

    st.dataframe(tableau, hide_index=True, use_container_width=True)
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

with st.expander("Limites de ce modèle"):
    st.markdown(
        "- Il ne connaît ni l'état du bien, ni l'étage, ni la vue, ni le DPE.\n"
        "- Il ne couvre que la commune de Montpellier.\n"
        "- Les prix incluent les caves et parkings vendus avec le logement.\n"
        "- Il a appris sur 2021-2024 : un changement récent du marché peut le décaler.\n"
        "- Source des données : DVF (data.gouv.fr) et OpenStreetMap."
    )