# immo-montpellier

ESTIMATEUR DE PRIX IMMOBILIER EXPLICABLE – MONTPELLIER

Combien vaut un logement à Montpellier, et pourquoi ? Cette application estime le prix d'un appartement ou d'une maison à partir de son adresse, donne une fourchette de prix, explique son estimation facteur par facteur et montre les ventes comparables du quartier.

Essayer l'application : 

https://estimateur-immo-montpellier.streamlit.app


EN BREF

Données : 23 703 ventes immobilières réelles à Montpellier (DVF, 2021-2025), enrichies avec OpenStreetMap.
Modèle : LightGBM optimisé avec Optuna, avec une validation temporelle.
Résultat sur 2025 (ventes jamais vues par le modèle) : erreur médiane de 10,8 %, et 47 % des estimations à moins de 10 % du vrai prix.
Fourchette de prix : environ 7 vrais prix sur 10 tombent dans la fourchette proposée.
Outils : Python, pandas, GeoPandas, scikit-learn, LightGBM, Optuna, SHAP, Streamlit, pytest.


FONCTIONNALITÉS

- Suggestions d'adresses : on tape le début d'une adresse, l'application propose les adresses officielles de Montpellier (géocodage public de l'IGN).
- Estimation et fourchette : un prix estimé, un prix au m², et une fourchette « bas – haut » obtenue par régression quantile.
- Explication du prix : chaque estimation est décomposée facteur par facteur avec les valeurs SHAP (par exemple « +12 k€ grâce aux commerces, −7 k€ à cause de la surface »).
- Indicateur de confiance : élevée, moyenne ou faible, selon le nombre de ventes similaires à proximité, la largeur de la fourchette et le caractère inhabituel de la surface.
- Ventes comparables : une carte et un tableau des ventes réelles les plus proches, de même type et de surface proche, reliés par des numéros.
- Simulateur « Et si… ? » : modifier la surface, le nombre de pièces ou les dépendances et voir le prix évoluer, avec une courbe du prix selon la surface.


RÉSULTATS

Le modèle est entraîné sur les ventes de 2021 à 2023, comparé aux autres sur 2024, puis évalué une seule fois sur 2025.

Comparaison des modèles sur la validation 2024 (première version) :
- Baseline (prix médian du code postal) : erreur moyenne 46 960 €, erreur médiane 18,8 %, 28 % des estimations à ±10 %
- Régression linéaire (Ridge) : erreur moyenne 42 807 €, erreur médiane 16,4 %, 32 % des estimations à ±10 %
- Random Forest : erreur moyenne 32 316 €, erreur médiane 11,6 %, 45 % des estimations à ±10 %
- LightGBM (retenu) : erreur moyenne 32 239 €, erreur médiane 11,4 %, 45 % des estimations à ±10 %

LightGBM réduit l'erreur médiane d'environ 40 % par rapport à une méthode simple. La régression linéaire fait à peine mieux que la baseline : les effets sont non linéaires (la proximité du tram compte beaucoup à 200 m, plus du tout à 2 km).

Résultat final sur les ventes de 2025, après nettoyage complémentaire et optimisation : erreur médiane de 10,8 %, et 47 % des estimations à moins de 10 % du vrai prix.


DÉMARCHE

1. Données (src/ingest.py) : téléchargement automatique des ventes DVF géolocalisées de Montpellier.
2. Nettoyage (src/clean.py) : une ligne DVF n'est pas une vente. Le script dédoublonne les lignes, exclut les ventes de plusieurs logements (prix impossible à répartir), retire les surfaces aberrantes, les prix au m² extrêmes et les ventes hors marché (moins de 30 % du prix médian local : viagers, ventes partielles, ventes en bloc). Les caves et parkings sont conservés comme variable. Attention : la règle des ventes hors marché a été ajoutée après la production des données du modèle actuel ; elle n'y est pas encore appliquée (voir l'onglet « Le modèle » de l'application).
3. Variables géographiques (src/features.py) : distances au centre, au tram, aux écoles, aux parcs, aux commerces et aux restaurants, et densités à 500 m, calculées en Lambert-93 à partir d'OpenStreetMap.
4. Modélisation (src/train.py) : comparaison d'une baseline, d'une régression linéaire, d'une Random Forest et de LightGBM, avec une validation temporelle pour éviter toute fuite de données.
5. Optimisation et modèle final (src/tune.py, src/final.py) : réglages trouvés par Optuna, réentraînement sur 2021-2024, modèles quantiles pour la fourchette, test unique sur 2025.
6. Explicabilité (notebooks/05_explicabilite.ipynb) : analyse SHAP globale et vente par vente, vérification de l'indicateur de confiance.
7. Application (src/predict.py, app/streamlit_app.py) : le calcul est séparé de l'affichage, et les variables sont calculées avec exactement les mêmes fonctions qu'à l'entraînement.


CE QUE J'AI APPRIS

- Une validation honnête change tout : un découpage aléatoire laisse le modèle « voir le futur » et donne des scores trop optimistes.
- Les pires erreurs n'étaient pas des erreurs du modèle : les estimations les plus fausses concernaient des ventes autour de 600 €/m², dont plusieurs dans le même immeuble au même prix. Plutôt que de supprimer les prédictions gênantes, j'ai ajouté une règle de nettoyage fixée à l'avance pour toutes les années. Elle reste à appliquer aux données du modèle actuel.
- Corrélation n'est pas causalité : les logements proches du tram sont plus chers, mais ils sont aussi plus proches du centre. SHAP aide à séparer les deux effets dans le modèle, sans prouver de relation de cause à effet.


STRUCTURE DU PROJET

- app/streamlit_app.py : l'application
- app/requirements.txt : les dépendances (versions figées) pour le déploiement
- src/ingest.py : téléchargement des données DVF
- src/clean.py : nettoyage
- src/features.py : variables géographiques (OpenStreetMap)
- src/train.py : comparaison des modèles
- src/tune.py : optimisation avec Optuna
- src/final.py : modèle final et test sur 2025
- src/predict.py : géocodage, estimation, explication, comparables
- src/marche.py : agrégations de l'onglet « Explorer le marché » (carte en hexagones, évolution, pièces)
- src/erreurs.py : erreurs sur 2025 par groupe et poids des ventes hors marché
- src/rapport_nettoyage.py : entonnoir du nettoyage, écrit dans reports/ pour l'application
- notebooks/ : exploration, nettoyage, variables, modèles, explicabilité
- tests/ : tests automatisés (pytest)
- models/ : modèles entraînés
- data/ : données (seules celles utiles à l'application sont versionnées)
- reports/ : résultats chiffrés
- NOTES.md : journal de bord détaillé du projet


LANCER LE PROJET EN LOCAL

Installation :
git clone https://github.com/James-Rigault/immo-montpellier.git
cd immo-montpellier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Lancer l'application :
python -m streamlit run app/streamlit_app.py

Reconstruire toute la chaîne, des données brutes au modèle final :
python src/ingest.py
python src/clean.py
python src/features.py
python src/train.py
python src/tune.py
python src/final.py
python src/rapport_nettoyage.py

Le rapport vérifie que ses étapes reproduisent les données du modèle : il s'arrête avec une erreur si elles ont changé.

Lancer les tests :
python -m pytest


LIMITES

- Les données DVF ne contiennent ni l'état du bien, ni l'étage, ni la vue, ni le diagnostic énergétique : une partie des écarts de prix est impossible à prévoir avec ces seules données.
- Le prix enregistré inclut les caves et parkings vendus avec le logement.
- Le modèle ne couvre que la commune de Montpellier.
- Les lieux OpenStreetMap sont ceux d'aujourd'hui, pas ceux de la date de chaque vente.
- L'année n'est pas une variable du modèle : une évolution rapide du marché peut le décaler.


PISTES D'AMÉLIORATION

- Intégrer les ventes de logements neufs (VEFA) avec une variable « neuf / ancien ».
- Ajouter la classe énergétique (DPE, données ADEME).
- Suivre l'évolution du marché avec un indice des prix mensuel.
- Estimer la rentabilité locative à partir des données publiques de loyers.


SOURCES DES DONNÉES

- Demandes de valeurs foncières géolocalisées (DVF) : DGFiP et Etalab, data.gouv.fr, Licence Ouverte 2.0.
- OpenStreetMap : © les contributeurs d'OpenStreetMap, licence ODbL.
- Géocodage : service de géocodage de la Géoplateforme (IGN).

Les estimations de cette application sont indicatives et réalisées dans un cadre pédagogique.


AUTEUR

James Rigault, étudiant en 2e année d'école d'ingénieur.
