"""Morceaux d'interface : badges, carte de résultat, badge de confiance, graphique « Pourquoi ce prix ? »."""
from html import escape

import altair as alt
import pandas as pd

TERRE_CUITE = "#C0562F"

VERT, ROUGE = "#2E7D32", "#C62828"

# couleur et nombre de segments remplis de la jauge, pour chaque niveau de confiance
NIVEAUX = {"faible": (ROUGE, 1), "moyenne": ("#D08A1E", 2), "élevée": (VERT, 3)}

STYLE = f"""
<style>
.badges {{display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.25rem 0 1.5rem;}}
.badge {{flex: 1 1 10rem; background: #F3E9DD; border: 1px solid #E4D6C6; border-radius: 0.6rem;
        padding: 0.6rem 0.8rem;}}
.badge b {{display: block; font-size: 1.5rem; color: {TERRE_CUITE}; line-height: 1.2;}}
.badge span {{display: block; font-size: 0.85rem; line-height: 1.3; color: #6B5B50;}}

.carte {{background: #FFFFFF; border: 1px solid #E4D6C6; border-radius: 0.8rem; padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(43, 33, 28, 0.06); margin-bottom: 1rem;}}
.carte .lieu {{font-size: 0.9rem; color: #6B5B50; margin-bottom: 0.3rem;}}
.carte .prix {{font-size: 2.6rem; font-weight: 700; color: {TERRE_CUITE}; line-height: 1.1;}}
.carte .m2 {{font-size: 1.05rem; color: #2B211C; margin-bottom: 1.4rem;}}
.fourchette {{position: relative; height: 0.7rem; border-radius: 1rem; margin: 2.2rem 0 0.5rem;
             background: linear-gradient(90deg, #F1D9C7, {TERRE_CUITE}, #F1D9C7);}}
.fourchette .repere {{position: absolute; top: 50%; width: 1.1rem; height: 1.1rem; border-radius: 50%;
                     background: #2B211C; border: 3px solid #FFFFFF; transform: translate(-50%, -50%);}}
.fourchette .etiquette {{position: absolute; bottom: 1.1rem; transform: translateX(-50%); white-space: nowrap;
                        font-size: 0.85rem; font-weight: 600;}}
.bornes {{display: flex; justify-content: space-between; font-size: 0.85rem; color: #6B5B50;}}
.bornes b {{display: block; color: #2B211C; font-size: 0.95rem;}}
.bornes div:last-child {{text-align: right;}}

.confiance {{display: inline-flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0.9rem;
            border-radius: 2rem; font-weight: 600; background: #FFFFFF; border: 2px solid var(--c); color: var(--c);}}
.jauge {{display: inline-flex; gap: 3px;}}
.jauge i {{width: 1.3rem; height: 0.55rem; border-radius: 0.2rem; background: #E4D6C6;}}
.jauge i.plein {{background: var(--c);}}
.raisons {{margin: 0.5rem 0 1rem; padding-left: 1.2rem; color: #4A3D35; font-size: 0.92rem;}}
</style>
"""


def position_estimation(prix, prix_bas, prix_haut):
    """Place de l'estimation sur la barre de fourchette, en % de sa longueur (0 = prix bas)."""
    if prix_haut <= prix_bas:
        return 50.0
    return min(max(100 * (prix - prix_bas) / (prix_haut - prix_bas), 0.0), 100.0)


def carte_resultat(adresse, prix, prix_m2, prix_bas, prix_haut, euros):
    """Carte avec le prix en grand, le prix au m² et la barre de fourchette."""
    position = position_estimation(prix, prix_bas, prix_haut)
    # l'étiquette reste dans la barre même quand l'estimation est tout au bord
    etiquette = min(max(position, 18.0), 82.0)
    return f"""
<div class="carte">
  <div class="lieu">📍 {escape(adresse)}</div>
  <div class="prix">{euros(prix)}</div>
  <div class="m2">soit <b>{euros(prix_m2)}/m²</b></div>
  <div class="fourchette">
    <span class="etiquette" style="left: {etiquette:.1f}%">Estimation</span>
    <span class="repere" style="left: {position:.1f}%"></span>
  </div>
  <div class="bornes">
    <div>Prix bas<b>{euros(prix_bas)}</b></div>
    <div>Prix haut<b>{euros(prix_haut)}</b></div>
  </div>
</div>
"""


def badge_confiance(niveau, raisons):
    """Pastille colorée avec une jauge à 3 segments, et la liste des raisons dessous."""
    couleur, remplis = NIVEAUX[niveau]
    segments = "".join('<i class="plein"></i>' if i < remplis else "<i></i>" for i in range(3))
    liste = "".join(f"<li>{escape(raison[0].upper() + raison[1:])}</li>" for raison in raisons)
    return f"""
<div style="--c: {couleur}">
  <span class="confiance"><span class="jauge">{segments}</span>Confiance {niveau}</span>
  <ul class="raisons">{liste}</ul>
</div>
"""


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
    """Barres horizontales de l'effet de chaque facteur sur le prix, en k€ (vert = hausse, rouge = baisse)."""
    donnees = pd.DataFrame({"Facteur": facteurs.index, "effet": facteurs.values / 1000})
    donnees["Effet"] = [f"{v:+.1f} k€".replace(".", ",") for v in donnees["effet"]]
    donnees["sens"] = ["hausse" if v > 0 else "baisse" for v in donnees["effet"]]

    # axe toujours un peu ouvert des deux côtés de 0, avec de la place pour les étiquettes à droite
    ecart = donnees["effet"].abs().max()
    domaine = [min(donnees["effet"].min(), -0.25 * ecart) * 1.15, max(donnees["effet"].max(), 0.25 * ecart) * 1.3]

    base = alt.Chart(donnees).encode(
        y=alt.Y("Facteur:N", sort=None, title=None, axis=alt.Axis(labelLimit=170, labelFontSize=12)),
        tooltip=[alt.Tooltip("Facteur:N"), alt.Tooltip("Effet:N", title="Effet sur le prix")],
    )
    couleur = alt.Color("sens:N", scale=alt.Scale(domain=["hausse", "baisse"], range=[VERT, ROUGE]), legend=None)
    barres = base.mark_bar(cornerRadius=3, height={"band": 0.7}).encode(
        x=alt.X("effet:Q", title="Effet sur le prix (k€)", scale=alt.Scale(domain=domaine, nice=False),
                axis=alt.Axis(values=graduations(*domaine), labelFontSize=11,
                              labelExpr="datum.value == 0 ? '0' : format(datum.value, '+~g') + ' k€'")),
        color=couleur,
    )
    # valeur écrite à droite : au bout des hausses, et juste après la ligne 0 pour les baisses,
    # pour ne jamais chevaucher les noms des facteurs à gauche
    style = {"align": "left", "dx": 4, "fontSize": 11, "fontWeight": "bold"}
    hausses = base.transform_filter("datum.effet > 0").mark_text(**style).encode(x="effet:Q", text="Effet:N", color=couleur)
    baisses = base.transform_filter("datum.effet <= 0").mark_text(**style).encode(x=alt.datum(0), text="Effet:N", color=couleur)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6B5B50").encode(x="x:Q")
    return (barres + hausses + baisses + zero).properties(height=34 * len(donnees)).configure_view(stroke=None)


def importance_variables(modele):
    """Importance « gain » de chaque variable, en % du total, latitude et longitude regroupées."""
    gains = pd.Series(modele.booster_.feature_importance("gain"), index=modele.booster_.feature_name())
    gains["emplacement"] = gains.pop("latitude") + gains.pop("longitude")
    return (100 * gains / gains.sum()).sort_values(ascending=False)


def graphique_importance(importance):
    """Barres horizontales terre cuite de l'importance de chaque variable, en %."""
    donnees = pd.DataFrame({"Variable": importance.index, "part": importance.values})
    donnees["Part"] = [f"{v:.1f} %".replace(".", ",") for v in donnees["part"]]
    base = alt.Chart(donnees).encode(
        y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelLimit=170, labelFontSize=12)),
        x=alt.X("part:Q", title="Part de l'importance totale (%)",
                scale=alt.Scale(domain=[0, donnees["part"].max() * 1.25])),
        tooltip=[alt.Tooltip("Variable:N"), alt.Tooltip("Part:N", title="Importance")],
    )
    barres = base.mark_bar(cornerRadius=3, height={"band": 0.7}, color=TERRE_CUITE)
    valeurs = base.mark_text(align="left", dx=4, fontSize=11, color="#2B211C").encode(text="Part:N")
    return (barres + valeurs).properties(height=30 * len(donnees)).configure_view(stroke=None)


def graphique_comparaison(resultats):
    """Baseline et modèle final côte à côte sur 3 scores en %, avec le sens « meilleur » indiqué."""
    # le « | » sépare les deux lignes du nom de chaque score
    scores = {
        "MAPE (%)": "Erreur moyenne|plus bas = mieux",
        "Erreur médiane (%)": "Erreur médiane|plus bas = mieux",
        "Part à ±10 % (%)": "Prix à ±10 %|plus haut = mieux",
    }
    modeles = ["Baseline", "Modèle final (LightGBM)"]
    donnees = (
        resultats[list(scores)].rename(columns=scores)
        .rename(index={"LightGBM final": modeles[1]})
        .rename_axis("Modèle").reset_index()
        .melt(id_vars="Modèle", var_name="Score", value_name="valeur")
    )
    donnees["Valeur"] = [f"{v:.1f} %".replace(".", ",") for v in donnees["valeur"]]
    base = alt.Chart(donnees).encode(
        y=alt.Y("Score:N", title=None, sort=list(scores.values()),
                axis=alt.Axis(labelExpr="split(datum.label, '|')", labelFontSize=12, labelLimit=150, ticks=False)),
        yOffset=alt.YOffset("Modèle:N", sort=modeles),
        x=alt.X("valeur:Q", title=None, axis=None, scale=alt.Scale(domain=[0, donnees["valeur"].max() * 1.25])),
        tooltip=[alt.Tooltip("Modèle:N"), alt.Tooltip("Score:N"), alt.Tooltip("Valeur:N")],
    )
    couleur = alt.Color("Modèle:N", title=None, legend=alt.Legend(orient="top", labelFontSize=12),
                        scale=alt.Scale(domain=modeles, range=["#B8A99A", TERRE_CUITE]))
    barres = base.mark_bar(cornerRadius=3).encode(color=couleur)
    valeurs = base.mark_text(align="left", dx=4, fontSize=11, color="#2B211C").encode(text="Valeur:N")
    return (barres + valeurs).properties(height=alt.Step(26)).configure_view(stroke=None)
