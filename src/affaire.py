"""« Bonne affaire ? » : un prix demandé comparé à l'estimation, à sa fourchette et aux ventes comparables."""
import numpy as np

# Seuils exprimés en largeurs de fourchette au-dessus du prix haut : au-delà d'une demi-largeur,
# l'écart n'est plus une affaire de négociation, il sort nettement du marché.
SEUILS_VERDICT = {"nettement_au_dessus": 0.5}

VERDICTS = {
    "dessous": "En dessous du marché",
    "dans": "Dans la fourchette",
    "dessus": "Au-dessus du marché",
    "nettement": "Nettement au-dessus",
}
# confiance faible : le verdict est atténué, il n'est plus affirmé
VERDICTS_PRUDENTS = {
    "dessous": "Probablement en dessous du marché",
    "dans": "Probablement dans la fourchette",
    "dessus": "Probablement au-dessus du marché",
    "nettement": "Probablement au-dessus du marché",
}


def niveau_verdict(prix_demande, prix_bas, prix_haut, seuils=SEUILS_VERDICT):
    """Clé du verdict (dessous, dans, dessus, nettement) selon la place du prix demandé autour de la fourchette."""
    if prix_demande < prix_bas:
        return "dessous"
    if prix_demande <= prix_haut:
        return "dans"
    largeur = max(prix_haut - prix_bas, 1)
    return "nettement" if prix_demande - prix_haut > seuils["nettement_au_dessus"] * largeur else "dessus"


def analyser_prix(prix_demande, prix, prix_bas, prix_haut, confiance):
    """Écart à l'estimation, place dans la fourchette et verdict, prudent si la confiance est faible."""
    cle = niveau_verdict(prix_demande, prix_bas, prix_haut)
    prudent = confiance == "faible"
    return {
        "ecart": prix_demande - prix,
        "ecart_pct": 100 * (prix_demande - prix) / prix,
        "cle": cle,
        "verdict": (VERDICTS_PRUDENTS if prudent else VERDICTS)[cle],
        "prudent": prudent,
    }


def position_comparables(prix_m2_demande, prix_m2_comparables):
    """Combien de ventes comparables étaient moins chères au m² que le prix demandé, et sur combien."""
    prix = np.asarray(prix_m2_comparables, dtype=float)
    return int((prix < prix_m2_demande).sum()), len(prix)
