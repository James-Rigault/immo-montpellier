import io
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from rapport_pdf import rapport_pdf  # noqa: E402

CHIFFRES = {"couverture": 73.0, "erreur_mediane": 10.5, "part_10": 48.4}


def estimation():
    """Un résultat d'estimation complet, sans modèle ni réseau."""
    comparables = pd.DataFrame({
        "date_mutation": pd.to_datetime(["2024-02-01", "2023-06-01"]),
        "adresse_numero": [5, None], "adresse_nom_voie": ["RUE DU BAYLE", "RUE DE L'ÉCOLE"],
        "surface_reelle_bati": [42.0, 53.0], "nombre_pieces_principales": [2.0, 3.0],
        "valeur_fonciere": [148_000.0, 175_000.0], "prix_m2": [3_524.0, 3_302.0], "distance_m": [9.0, 14.0],
    })
    r = {
        "adresse": "7 Rue du Bayle 34000 Montpellier", "prix": 222_869, "prix_m2": 3_714, "prix_bas": 169_691,
        "prix_haut": 279_131, "base": 206_585, "comparables": comparables,
    }
    bien = {"type_local": "Appartement", "surface": 60, "pieces": 3, "dependances": 1, "terrain": 0,
            "annee": 1930, "dpe": "D"}
    facteurs = pd.Series({"Emplacement dans la ville": 28_600.0, "Classe DPE du bâtiment": -5_300.0,
                          "Année de construction": 1_200.0})
    return r, bien, facteurs


def texte_pdf(contenu):
    """Tout le texte du PDF, espaces insécables remplacées par des espaces simples."""
    lecteur = PdfReader(io.BytesIO(contenu))
    return "\n".join(page.extract_text() for page in lecteur.pages).replace(" ", " "), len(lecteur.pages)


def test_rapport_complet_avec_les_bonnes_valeurs():
    r, bien, facteurs = estimation()
    affaire = {"verdict": "Au-dessus du marché", "prix_demande": 300_000, "ecart": 77_131, "ecart_pct": 34.6,
               "moins_chers": 8, "nb_comparables": 10, "prudent": False}
    debut = time.perf_counter()
    contenu = rapport_pdf(r, bien, "moyenne", ["fourchette de prix moyenne"], facteurs, CHIFFRES, affaire,
                          jour=date(2026, 10, 16))
    assert time.perf_counter() - debut < 3  # généré en mémoire en quelques secondes au plus
    assert contenu.startswith(b"%PDF") and len(contenu) > 5_000
    texte, pages = texte_pdf(contenu)
    assert pages >= 1
    for attendu in [
        "16 OCTOBRE 2026", "7 Rue du Bayle 34000 Montpellier",          # date et adresse
        "Appartement", "60 m", "1930", "Classe DPE",                      # caractéristiques, dont année et DPE
        "222 869 €", "3 714 €", "169 691 €", "279 131 €", "Moyenne",     # prix, prix au m², fourchette, confiance
        "Emplacement dans la ville", "Classe DPE du bâtiment", "+28,6 k€", "−5,3 k€",  # facteurs SHAP
        "5 Rue Du Bayle", "148 000 €", "Rue De L'École",                 # ventes comparables
        "Au-dessus du marché", "300 000 €", "8 sur 10",                  # « Bonne affaire ? »
        "MÉTHODE ET LIMITES", "pas une expertise", "10,5 %",             # méthode et limites
    ]:
        assert attendu in texte, attendu


def test_rapport_sans_prix_demande_ni_annee():
    r, bien, facteurs = estimation()
    bien = {**bien, "annee": None, "dpe": "Inconnu"}
    texte, _ = texte_pdf(rapport_pdf(r, bien, "faible", ["peu de ventes comparables"], facteurs, CHIFFRES))
    assert "BONNE AFFAIRE" not in texte and "Inconnue" in texte and "Faible" in texte
    assert "03 — LECTURE" in texte  # sans « Bonne affaire ? », les sections sont renumérotées
