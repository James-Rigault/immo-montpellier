"""Rapport PDF d'une estimation, généré en mémoire avec fpdf2, dans la charte de l'app.

Les polices de l'app (Hanken Grotesk, Cormorant Garamond, licence OFL) sont incluses dans app/polices :
accents et symbole € s'affichent correctement, sans dépendre des polices du serveur.
"""
from datetime import date
from pathlib import Path

from fpdf import FPDF

from affichage import adresses_comparables

POLICES = Path(__file__).resolve().parent / "polices"
LOGO = Path(__file__).resolve().parents[1] / "assets" / "logo.svg"
NOM_APP = "Valeur Montpellier"
# même palette que assets/style.css : le vert et le doré du logo, l'encre, des gris chauds
ENCRE, GRIS, GRIS_CLAIR, FILET = (20, 20, 20), (110, 106, 98), (185, 180, 172), (214, 210, 203)
VERT, DORE = (42, 61, 47), (201, 168, 106)
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
        "novembre", "décembre"]


def texte(valeur):
    """Hanken Grotesk n'a pas l'espace fine insécable : on la remplace par une espace insécable."""
    return str(valeur).replace(" ", " ")


def euros(x):
    return f"{x:,.0f} €".replace(",", " ")


def pourcent(x):
    return f"{x:.1f} %".replace(".", ",")


def date_longue(jour):
    return f"{jour.day} {MOIS[jour.month - 1]} {jour.year}"


class Rapport(FPDF):
    """Une page A4 sobre : étiquettes en capitales grises, titres serif, filets fins."""

    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(True, margin=18)
        self.add_font("Hanken", "", POLICES / "HankenGrotesk-Regular.ttf")
        self.add_font("Hanken", "B", POLICES / "HankenGrotesk-Medium.ttf")
        self.add_font("Cormorant", "", POLICES / "CormorantGaramond-Light.ttf")

    def footer(self):
        self.set_y(-12)
        self.set_font("Hanken", "", 7.5)
        self.set_text_color(*GRIS)
        self.cell(0, 4, f"Page {self.page_no()} sur {{nb}}", align="C")

    def titre(self, valeur, taille=20):
        self.set_font("Cormorant", "", taille)
        self.set_text_color(*ENCRE)
        self.multi_cell(0, taille * 0.45, texte(valeur), align="L", new_x="LMARGIN", new_y="NEXT")

    def paragraphe(self, valeur, taille=9.5, couleur=(74, 71, 67)):
        self.set_font("Hanken", "", taille)
        self.set_text_color(*couleur)
        self.multi_cell(0, taille * 0.5, texte(valeur), align="L", new_x="LMARGIN", new_y="NEXT")

    def filet(self, couleur=FILET, epaisseur=0.2):
        self.set_draw_color(*couleur)
        self.set_line_width(epaisseur)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())

    def section(self, titre, hauteur=40):
        """Nouvelle section : un filet encre puis le titre serif.

        Si la section (titre compris, environ 25 mm, plus `hauteur` mm de contenu) ne tient pas
        sur la page, elle commence sur la suivante : un graphique n'est jamais coupé en deux.
        """
        if self.get_y() + 25 + hauteur > self.page_break_trigger:
            self.add_page()
        self.ln(5)
        self.filet(ENCRE, 0.3)
        self.ln(4)
        self.titre(titre, 17)
        self.ln(2)

    def lignes(self, paires, largeur_cle=55):
        """Tableau à deux colonnes (libellé, valeur), séparé par des filets."""
        for cle, valeur in paires:
            self.set_font("Hanken", "", 9.5)
            self.set_text_color(*GRIS)
            self.cell(largeur_cle, 7, texte(cle))
            self.set_text_color(*ENCRE)
            self.cell(0, 7, texte(valeur), new_x="LMARGIN", new_y="NEXT")
            self.filet()

    def chiffres(self, elements, premier_en_vert=False):
        """Trois grands chiffres côte à côte, à la manière de l'app.

        En Hanken Grotesk : Cormorant dessine des chiffres de hauteur variable (elzéviriens), que l'app
        redresse en CSS mais que fpdf2 ne sait pas redresser sans moteur de mise en forme supplémentaire.
        """
        largeur = (self.w - self.l_margin - self.r_margin) / len(elements)
        y = self.get_y()
        for i, (valeur, legende) in enumerate(elements):
            x = self.l_margin + i * largeur
            self.set_xy(x, y)
            self.set_font("Hanken", "B" if premier_en_vert and i == 0 else "", 14)
            self.set_text_color(*(VERT if premier_en_vert and i == 0 else ENCRE))
            self.cell(largeur, 9, texte(valeur))
            self.set_xy(x, y + 10)
            self.set_font("Hanken", "", 8)
            self.set_text_color(*GRIS)
            self.multi_cell(largeur - 4, 4, texte(legende), align="L")
        self.set_xy(self.l_margin, y + 20)

    def barres(self, facteurs):
        """Effet de chaque facteur sur le prix, en k€ : hausses en encre, baisses en gris, autour d'un axe 0."""
        largeur_nom, largeur_valeur = 55, 18
        zone = self.w - self.l_margin - self.r_margin - largeur_nom - largeur_valeur
        bas, haut = min(facteurs.min(), 0), max(facteurs.max(), 0)
        echelle = zone / max(haut - bas, 1)
        zero = self.l_margin + largeur_nom + (-bas) * echelle
        y0 = self.get_y()
        for nom, effet in facteurs.items():
            y = self.get_y()
            self.set_font("Hanken", "", 8.5)
            self.set_text_color(*ENCRE)
            self.cell(largeur_nom - 3, 6, texte(nom), align="R")
            self.set_fill_color(*(ENCRE if effet > 0 else GRIS_CLAIR))
            gauche = zero if effet > 0 else zero + effet * echelle
            self.rect(gauche, y + 1.8, abs(effet) * echelle, 2.6, style="F")
            self.set_xy(max(zero, zero + effet * echelle) + 1.5 if effet > 0 else zero + 1.5, y)
            self.set_text_color(*(ENCRE if effet > 0 else GRIS))
            self.cell(largeur_valeur, 6, texte(f"{effet / 1000:+.1f} k€".replace(".", ",").replace("-", "−")))
            self.set_xy(self.l_margin, y + 6)
        self.set_draw_color(*ENCRE)
        self.set_line_width(0.25)
        self.line(zero, y0, zero, self.get_y())

    def tableau(self, entetes, lignes, largeurs, alignements):
        """Tableau à filets horizontaux, en-têtes gris, chiffres alignés à droite."""
        self.set_font("Hanken", "", 8)
        self.set_text_color(*GRIS)
        for entete, largeur, alignement in zip(entetes, largeurs, alignements):
            self.cell(largeur, 6, texte(entete), align=alignement)
        self.ln(6)
        self.filet(ENCRE, 0.3)
        self.set_font("Hanken", "", 8.5)
        self.set_text_color(*ENCRE)
        for ligne in lignes:
            for valeur, largeur, alignement in zip(ligne, largeurs, alignements):
                self.cell(largeur, 6.5, texte(valeur), align=alignement)
            self.ln(6.5)
            self.filet()


def rapport_pdf(r, bien, niveau, raisons, facteurs, chiffres, affaire=None, jour=None):
    """Le rapport complet, en octets, prêt pour st.download_button.

    r : résultat d'Estimateur.estimer ; bien : caractéristiques saisies ; facteurs : effets SHAP déjà
    regroupés et nommés (en €) ; chiffres : scores du test 2025 ; affaire : analyse « Bonne affaire ? » ou None.
    """
    pdf = Rapport()
    pdf.add_page()

    # en-tête : le logo tel quel (fpdf2 lit le SVG), le nom de l'app et la date, puis un filet doré
    pdf.image(LOGO, x=pdf.l_margin, y=pdf.t_margin, h=11)
    pdf.set_xy(pdf.l_margin + 14, pdf.t_margin + 0.5)
    pdf.set_font("Cormorant", "", 15)
    pdf.set_text_color(*VERT)
    pdf.cell(0, 6, NOM_APP, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 14)
    pdf.set_font("Hanken", "", 8.5)
    pdf.set_text_color(*GRIS)
    pdf.cell(0, 4.5, texte(f"Estimation du {date_longue(jour or date.today())}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.filet(DORE, 0.6)
    pdf.ln(5)
    pdf.titre(r["adresse"], 24)

    pdf.section("Le logement")
    pdf.lignes([
        ("Type de bien", bien["type_local"]),
        ("Surface habitable", f"{bien['surface']:.0f} m²"),
        ("Nombre de pièces", str(bien["pieces"])),
        ("Caves et parkings", str(bien["dependances"])),
        ("Terrain", f"{bien['terrain']:.0f} m²" if bien["terrain"] else "aucun"),
        ("Année de construction", str(bien["annee"]) if bien.get("annee") else "Inconnue"),
        ("Classe DPE du bâtiment", f"{bien['dpe']} (données du bâtiment, BDNB et ADEME)" if bien.get("dpe")
         else "Inconnue (aucun DPE trouvé pour le bâtiment)"),
    ])

    pdf.section("Prix estimé")
    pdf.chiffres([
        (euros(r["prix"]), f"Soit {euros(r['prix_m2'])} le m²"),
        (euros(r["prix_bas"]), "Fourchette basse"),
        (euros(r["prix_haut"]), "Fourchette haute"),
    ], premier_en_vert=True)
    pdf.lignes([("Confiance", f"{niveau.capitalize()} : " + ", ".join(raisons))], largeur_cle=30)
    pdf.ln(1)
    pdf.paragraphe(f"Sur les ventes de 2025, {chiffres['couverture'] / 10:.0f} vrais prix sur 10 se trouvaient "
                   "dans une fourchette de ce type.", 8.5, GRIS)

    if affaire:
        pdf.section("Bonne affaire ?")
        pdf.titre(affaire["verdict"], 14)
        pdf.ln(1)
        if round(affaire["ecart"]) == 0:
            ecart = "est égal à l'estimation"
        else:
            sens = "au-dessus" if affaire["ecart"] > 0 else "en dessous"
            ecart = (f"est {euros(abs(affaire['ecart']))} {sens} de l'estimation "
                     f"({affaire['ecart_pct']:+.1f}\u00a0%)".replace(".", ",").replace("-", "−"))
        pdf.paragraphe(f"Le prix demandé, {euros(affaire['prix_demande'])}, {ecart}. Au m², il est plus cher que "
                       f"{affaire['moins_chers']} des {affaire['nb_comparables']} ventes comparables les plus proches.")
        if affaire["prudent"]:
            pdf.paragraphe("La confiance est faible : ce verdict est une simple indication.", 8.5, GRIS)

    pdf.section("Pourquoi ce prix", hauteur=15 + 6 * len(facteurs))
    pdf.paragraphe(f"Le modèle part d'un prix moyen de {euros(r['base'])} pour cette surface. Chaque "
                   "caractéristique l'augmente (en noir) ou le diminue (en gris). Ces effets sont calculés avec les "
                   "valeurs SHAP.", 8.5)
    pdf.ln(2)
    pdf.barres(facteurs)

    comparables = r["comparables"].reset_index(drop=True)
    pdf.section("Ventes comparables", hauteur=8 + 6.5 * len(comparables))
    pdf.tableau(
        ["N°", "Date", "Adresse", "Surface", "Pièces", "Prix", "Prix au m²", "Distance"],
        [
            [str(i + 1), ligne.date_mutation.strftime("%m/%Y"), adresse, f"{ligne.surface_reelle_bati:.0f} m²",
             f"{ligne.nombre_pieces_principales:.0f}", euros(ligne.valeur_fonciere), euros(ligne.prix_m2),
             f"{ligne.distance_m:.0f} m"]
            for i, (ligne, adresse) in enumerate(zip(comparables.itertuples(), adresses_comparables(comparables)))
        ],
        [8, 15, 52, 17, 13, 24, 22, 23], ["L", "L", "L", "R", "R", "R", "R", "R"],
    )

    pdf.section("Méthode et limites")
    pdf.paragraphe(
        "L'estimation vient d'un modèle LightGBM appris sur les ventes réelles de Montpellier (DVF) de 2021 à 2024. "
        "Il utilise aussi les lieux proches (OpenStreetMap), l'année de construction du bâtiment (BDNB) et sa classe "
        f"DPE médiane (ADEME). Sur les ventes de 2025, qu'il n'avait jamais vues, l'erreur médiane est de "
        f"{pourcent(chiffres['erreur_mediane'])} et {chiffres['part_10']:.0f}\u00a0% des estimations sont à moins de "
        "10\u00a0% du vrai prix. La fourchette vient de deux modèles quantiles (10\u00a0% et 90\u00a0%)."
    )
    pdf.ln(2)
    pdf.paragraphe(
        "C'est une estimation statistique : le modèle ne connaît ni l'état du logement, ni l'étage, ni la vue. Elle "
        "ne remplace pas l'avis d'un professionnel. Le DPE utilisé est celui du bâtiment, pas celui du logement, et "
        "les prix de vente incluent les caves et parkings vendus avec le logement."
    )
    return bytes(pdf.output())
