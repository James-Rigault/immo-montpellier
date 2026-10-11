"""Composants d'interface : résultat, confiance et graphiques, dans le style « Édition ».

Le HTML produit ici est mis en forme par app/style.css ; les graphiques Altair reprennent
la même palette (encre, gris chauds, un seul accent) et la même police.
"""
from html import escape

import altair as alt
import pandas as pd

ENCRE = "#141414"
GRIS = "#8C877F"
GRIS_CLAIR = "#B9B4AC"
FILET = "#D6D2CB"
ACCENT = "#1F3B2D"  # vert bouteille : la seule couleur, réservée à ce qui compte
POLICE = "Hanken Grotesk"

# nombre de traits remplis de la jauge, pour chaque niveau de confiance
NIVEAUX = {"faible": 1, "moyenne": 2, "élevée": 3}


def style_graphique(graphique):
    """Applique la charte du site à un graphique Altair : sans cadre ni fond, typographie du site."""
    return (
        graphique
        .configure(font=POLICE, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=GRIS, titleColor=GRIS, labelFontSize=12, titleFontSize=11, titleFontWeight=500,
            titlePadding=12, domain=False, ticks=False, gridColor=FILET, gridWidth=0.6,
        )
        .configure_legend(labelColor=ENCRE, labelFontSize=12, symbolType="square", symbolSize=90, orient="top-left")
    )


# ---------- Résultat ----------

def position_estimation(prix, prix_bas, prix_haut):
    """Place de l'estimation sur le filet de fourchette, en % de sa longueur (0 = prix bas)."""
    if prix_haut <= prix_bas:
        return 50.0
    return min(max(100 * (prix - prix_bas) / (prix_haut - prix_bas), 0.0), 100.0)


def carte_resultat(adresse, prix, prix_m2, prix_bas, prix_haut, euros):
    """Prix en très grand, prix au m², puis la fourchette dessinée comme un filet gradué."""
    position = position_estimation(prix, prix_bas, prix_haut)
    return f"""
<div class="resultat">
  <span class="etiquette">Estimation</span>
  <p class="resultat-adresse">{escape(adresse)}</p>
  <p class="resultat-prix">{euros(prix)}</p>
  <p class="resultat-m2">{euros(prix_m2)} le m²</p>
  <div class="fourchette">
    <div class="fourchette-filet"><span class="fourchette-repere" style="left: {position:.1f}%"></span></div>
    <div class="fourchette-bornes">
      <div><span class="etiquette">Fourchette basse</span><b>{euros(prix_bas)}</b></div>
      <div><span class="etiquette">Fourchette haute</span><b>{euros(prix_haut)}</b></div>
    </div>
  </div>
</div>
"""


def badge_confiance(niveau, raisons):
    """Niveau de confiance : trois traits, dont 1, 2 ou 3 remplis, le mot, puis les raisons."""
    remplis = NIVEAUX[niveau]
    traits = "".join('<i class="plein"></i>' if i < remplis else "<i></i>" for i in range(3))
    liste = "".join(f"<li>{escape(raison[0].upper() + raison[1:])}</li>" for raison in raisons)
    return f"""
<div class="confiance">
  <span class="etiquette">Confiance</span>
  <p class="confiance-niveau"><span class="jauge">{traits}</span>{niveau.capitalize()}</p>
  <ul class="raisons">{liste}</ul>
</div>
"""


# ---------- « Pourquoi ce prix ? » ----------

def regrouper_facteurs(facteurs):
    """Additionne latitude et longitude en un seul facteur : les valeurs SHAP s'additionnent."""
    facteurs = facteurs.copy()
    emplacement = facteurs.pop("latitude") + facteurs.pop("longitude")
    facteurs["emplacement"] = emplacement
    return facteurs.reindex(facteurs.abs().sort_values(ascending=False).index)


def pas_rond(x):
    """Plus grand pas « rond » (0,1 ; 0,2 ; 0,5 ; 1 ; 2 ; 5 ; 10...) inférieur ou égal à x."""
    pas = 0.1
    for candidat in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if candidat <= x:
            pas = candidat
    return pas


def graduations(minimum, maximum):
    """Graduations de l'axe en k€ : toujours une valeur négative, 0, puis environ 3 valeurs positives."""
    pas = pas_rond(maximum / 3)
    positives = [round(pas * i, 1) for i in range(1, int(maximum / pas + 1e-9) + 1)]  # 1e-9 : évite 5,999… au lieu de 6
    return [-pas_rond(-minimum), 0] + positives


def graphique_facteurs(facteurs):
    """Effet de chaque facteur sur le prix, en k€ : hausses en encre, baisses en gris."""
    donnees = pd.DataFrame({"Facteur": facteurs.index, "effet": facteurs.values / 1000})
    donnees["Effet"] = [f"{v:+.1f} k€".replace(".", ",").replace("-", "−") for v in donnees["effet"]]
    donnees["sens"] = ["hausse" if v > 0 else "baisse" for v in donnees["effet"]]

    # axe toujours un peu ouvert des deux côtés de 0, avec de la place pour les valeurs à droite
    ecart = donnees["effet"].abs().max()
    domaine = [min(donnees["effet"].min(), -0.25 * ecart) * 1.15, max(donnees["effet"].max(), 0.25 * ecart) * 1.7]

    base = alt.Chart(donnees).encode(
        y=alt.Y("Facteur:N", sort=None, title=None, axis=alt.Axis(labelLimit=180, labelColor=ENCRE, labelPadding=10)),
        tooltip=[alt.Tooltip("Facteur:N"), alt.Tooltip("Effet:N", title="Effet sur le prix")],
    )
    couleur = alt.Color("sens:N", scale=alt.Scale(domain=["hausse", "baisse"], range=[ENCRE, GRIS_CLAIR]), legend=None)
    barres = base.mark_bar(height={"band": 0.42}).encode(
        x=alt.X("effet:Q", title="Effet sur le prix, en milliers d'euros", scale=alt.Scale(domain=domaine, nice=False),
                axis=alt.Axis(values=graduations(*domaine), grid=True,
                              labelExpr="datum.value == 0 ? '0' : format(datum.value, '+~g')")),
        color=couleur,
    )
    # valeur écrite à droite : au bout des hausses, et juste après la ligne 0 pour les baisses,
    # pour ne jamais chevaucher les noms des facteurs à gauche
    style = {"align": "left", "dx": 6, "fontSize": 12, "color": ENCRE}
    hausses = base.transform_filter("datum.effet > 0").mark_text(**style).encode(x="effet:Q", text="Effet:N")
    baisses = base.transform_filter("datum.effet <= 0").mark_text(**{**style, "color": GRIS}).encode(
        x=alt.datum(0), text="Effet:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ENCRE, strokeWidth=1).encode(x="x:Q")
    return style_graphique((barres + hausses + baisses + zero).properties(height=38 * len(donnees)))


# ---------- Simulateur ----------

def graphique_courbe(surfaces, prix, surface_actuelle, prix_actuel):
    """Prix estimé selon la surface : un trait d'accent, et un point sur la surface simulée."""
    donnees = pd.DataFrame({"surface": surfaces, "prix": prix})
    donnees["Prix"] = [f"{p:,.0f} €".replace(",", " ") for p in donnees["prix"]]
    axe_prix = alt.Axis(labelExpr="format(datum.value / 1000, ',.0f') + ' k€'", grid=True, tickCount=5)
    ligne = alt.Chart(donnees).mark_line(color=ACCENT, strokeWidth=1.6).encode(
        x=alt.X("surface:Q", title="Surface, en m²", axis=alt.Axis(grid=False, tickCount=8)),
        y=alt.Y("prix:Q", title=None, axis=axe_prix),
        tooltip=[alt.Tooltip("surface:Q", title="Surface (m²)"), alt.Tooltip("Prix:N", title="Prix estimé")],
    )
    actuel = pd.DataFrame({"surface": [surface_actuelle], "prix": [prix_actuel]})
    repere = alt.Chart(actuel).mark_rule(color=GRIS_CLAIR, strokeDash=[2, 3]).encode(x="surface:Q")
    point = alt.Chart(actuel).mark_point(filled=True, color=ENCRE, size=55).encode(x="surface:Q", y="prix:Q")
    return style_graphique((ligne + repere + point).properties(height=300))


# ---------- Le modèle ----------

def importance_variables(modele):
    """Importance « gain » de chaque variable, en % du total, latitude et longitude regroupées."""
    gains = pd.Series(modele.booster_.feature_importance("gain"), index=modele.booster_.feature_name())
    gains["emplacement"] = gains.pop("latitude") + gains.pop("longitude")
    return (100 * gains / gains.sum()).sort_values(ascending=False)


def graphique_importance(importance):
    """Importance de chaque variable, en % : la première en accent, les autres en gris."""
    donnees = pd.DataFrame({"Variable": importance.index, "part": importance.values})
    donnees["Part"] = [f"{v:.1f}\u202f%".replace(".", ",") for v in donnees["part"]]
    donnees["rang"] = ["premier"] + ["autre"] * (len(donnees) - 1)
    base = alt.Chart(donnees).encode(
        y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelLimit=180, labelColor=ENCRE, labelPadding=10)),
        x=alt.X("part:Q", title="Part de l'importance totale, en %",
                scale=alt.Scale(domain=[0, donnees["part"].max() * 1.2]), axis=alt.Axis(grid=True, tickCount=5)),
        tooltip=[alt.Tooltip("Variable:N"), alt.Tooltip("Part:N", title="Importance")],
    )
    barres = base.mark_bar(height={"band": 0.42}).encode(
        color=alt.Color("rang:N", scale=alt.Scale(domain=["premier", "autre"], range=[ACCENT, GRIS_CLAIR]), legend=None)
    )
    valeurs = base.mark_text(align="left", dx=6, fontSize=12, color=ENCRE).encode(text="Part:N")
    return style_graphique((barres + valeurs).properties(height=34 * len(donnees)))


def graphique_comparaison(resultats):
    """Point de comparaison (gris) et modèle final (accent) côte à côte sur 3 scores en %."""
    # le « | » sépare les deux lignes du nom de chaque score
    scores = {
        "MAPE (%)": "Erreur moyenne|plus bas = mieux",
        "Erreur médiane (%)": "Erreur médiane|plus bas = mieux",
        "Part à ±10 % (%)": "Prix à ±10\u202f%|plus haut = mieux",
    }
    modeles = ["Point de comparaison", "Modèle final (LightGBM)"]
    donnees = (
        resultats[list(scores)].rename(columns=scores)
        .rename(index={"Baseline": modeles[0], "LightGBM final": modeles[1]})
        .rename_axis("Modèle").reset_index()
        .melt(id_vars="Modèle", var_name="Score", value_name="valeur")
    )
    donnees["Valeur"] = [f"{v:.1f}\u202f%".replace(".", ",") for v in donnees["valeur"]]
    base = alt.Chart(donnees).encode(
        y=alt.Y("Score:N", title=None, sort=list(scores.values()),
                axis=alt.Axis(labelExpr="split(datum.label, '|')", labelColor=ENCRE, labelLimit=160, labelPadding=10)),
        yOffset=alt.YOffset("Modèle:N", sort=modeles),
        x=alt.X("valeur:Q", title=None, axis=None, scale=alt.Scale(domain=[0, donnees["valeur"].max() * 1.25])),
        tooltip=[alt.Tooltip("Modèle:N"), alt.Tooltip("Score:N"), alt.Tooltip("Valeur:N")],
    )
    couleur = alt.Color("Modèle:N", title=None, scale=alt.Scale(domain=modeles, range=[GRIS_CLAIR, ACCENT]))
    barres = base.mark_bar(height={"band": 0.8}).encode(color=couleur)
    valeurs = base.mark_text(align="left", dx=6, fontSize=12, color=ENCRE).encode(text="Valeur:N")
    return style_graphique((barres + valeurs).properties(height=alt.Step(24)))
