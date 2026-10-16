# immo-montpellier

ESTIMATEUR DE PRIX IMMOBILIER EXPLICABLE – MONTPELLIER

Combien vaut un logement à Montpellier, et pourquoi ? Cette application estime le prix d'un appartement ou d'une maison à partir de son adresse, donne une fourchette de prix, explique son estimation facteur par facteur et montre les ventes comparables du quartier.

Essayer l'application : 

https://estimateur-immo-montpellier.streamlit.app


EN BREF

Données : 23 338 ventes immobilières réelles à Montpellier (DVF, 2021-2025), enrichies avec OpenStreetMap, l'année de construction du bâtiment (BDNB) et sa classe DPE (ADEME).
Modèle : LightGBM optimisé avec Optuna, avec une validation temporelle.
Résultat sur 2025 (ventes jamais vues par le modèle) : erreur médiane de 10,5 %, et 48 % des estimations à moins de 10 % du vrai prix.
Fourchette de prix : environ 7 vrais prix sur 10 tombent dans la fourchette proposée.
Outils : Python, pandas, GeoPandas, scikit-learn, LightGBM, Optuna, SHAP, Streamlit, pytest.


FONCTIONNALITÉS

- Suggestions d'adresses : on tape le début d'une adresse, l'application propose les adresses officielles de Montpellier (géocodage public de l'IGN).
- Estimation et fourchette : un prix estimé, un prix au m², et une fourchette « bas – haut » obtenue par régression quantile.
- Explication du prix : chaque estimation est décomposée facteur par facteur avec les valeurs SHAP (par exemple « +12 k€ grâce aux commerces, −7 k€ à cause de la surface »).
- Indicateur de confiance : élevée, moyenne ou faible, selon le nombre de ventes similaires à proximité, la largeur de la fourchette et le caractère inhabituel de la surface.
- Ventes comparables : une carte et un tableau des ventes réelles les plus proches, de même type et de surface proche, reliés par des numéros.
- Année de construction et DPE : pré-remplis à partir du bâtiment de l'adresse (BDNB et ADEME), modifiables (DPE de A à G, ou « Inconnu »), et visibles dans l'explication SHAP.
- « Bonne affaire ? » : on saisit le prix d'une annonce ; l'application donne l'écart à l'estimation (en € et en %), sa place par rapport à la fourchette, un verdict (en dessous du marché, dans la fourchette, au-dessus, nettement au-dessus au-delà d'une demi-largeur de fourchette, seuil défini dans src/affaire.py), une jauge, et la position du prix au m² parmi les 10 ventes comparables. Si la confiance est faible, le verdict est formulé avec prudence.
- Simulateur « Et si… ? » : modifier la surface, le nombre de pièces, les dépendances ou la classe DPE et voir le prix évoluer, avec une courbe du prix selon la surface.


RÉSULTATS

Le modèle est entraîné sur les ventes de 2021 à 2023, comparé aux autres sur 2024, puis évalué une seule fois sur 2025.

Comparaison des modèles sur la validation 2024 (première version) :
- Baseline (prix médian du code postal) : erreur moyenne 46 960 €, erreur médiane 18,8 %, 28 % des estimations à ±10 %
- Régression linéaire (Ridge) : erreur moyenne 42 807 €, erreur médiane 16,4 %, 32 % des estimations à ±10 %
- Random Forest : erreur moyenne 32 316 €, erreur médiane 11,6 %, 45 % des estimations à ±10 %
- LightGBM (retenu) : erreur moyenne 32 239 €, erreur médiane 11,4 %, 45 % des estimations à ±10 %

LightGBM réduit l'erreur médiane d'environ 40 % par rapport à une méthode simple. La régression linéaire fait à peine mieux que la baseline : les effets sont non linéaires (la proximité du tram compte beaucoup à 200 m, plus du tout à 2 km).

Année de construction et DPE (validation 2024, mêmes réglages Optuna, moyenne de 5 entraînements, fichier reports/comparaison_variables.csv) :
- Ancien modèle (données sans la règle des ventes hors marché) : erreur moyenne 31 449 €, MAPE 17,6 %, erreur médiane 11,3 %, 45,5 % à ±10 %
- Mêmes variables, données nettoyées : erreur moyenne 31 679 €, MAPE 17,6 %, erreur médiane 11,3 %, 45,7 % à ±10 %
- + année de construction et DPE (retenu) : erreur moyenne 31 398 €, MAPE 17,8 %, erreur médiane 11,2 %, 45,5 % à ±10 %
- + indicateur « DPE manquant » : aucun gain, non retenu

Le gain est modeste. L'erreur moyenne en euros baisse à chaque entraînement (de 210 à 320 €), l'erreur médiane et la part à ±10 % restent dans le bruit, et l'erreur moyenne en pourcentage (MAPE) augmente d'environ 0,2 point à chaque fois.

Résultat final sur les ventes de 2025 (test lancé une seule fois, pour le modèle retenu) : erreur moyenne 29 782 €, MAPE 15,7 %, erreur médiane de 10,5 %, et 48 % des estimations à moins de 10 % du vrai prix ; 73 % des vrais prix dans la fourchette. La première version obtenait 10,8 % et 47 %, mais sur un jeu 2025 qui contenait encore 57 ventes hors marché : les deux chiffres ne sont pas directement comparables. L'indicateur de confiance reste cohérent sur 2025 : erreur médiane de 7,9 % en confiance élevée, 12,3 % en moyenne, 17,5 % en faible.


DÉMARCHE

1. Données (src/ingest.py) : téléchargement automatique des ventes DVF géolocalisées de Montpellier.
2. Nettoyage (src/clean.py) : une ligne DVF n'est pas une vente. Le script dédoublonne les lignes, exclut les ventes de plusieurs logements (prix impossible à répartir), retire les surfaces aberrantes, les prix au m² extrêmes et les ventes hors marché (moins de 30 % du prix médian local : viagers, ventes partielles, ventes en bloc). Les caves et parkings sont conservés comme variable. La règle des ventes hors marché retire 370 ventes.
3. Variables géographiques (src/features.py) : distances au centre, au tram, aux écoles, aux parcs, aux commerces et aux restaurants, et densités à 500 m, calculées en Lambert-93 à partir d'OpenStreetMap.
4. Bâtiment (src/batiments.py) : chaque vente est reliée à son bâtiment de la BDNB, par sa parcelle cadastrale (93,3 % des ventes), sinon par son adresse BAN (6,4 %), sinon par proximité à moins de 50 m (0,3 %) ; 7 ventes sur 23 338 restent sans bâtiment. On en tire l'année de construction (Fichiers fonciers, ou à défaut l'année médiane déclarée dans les DPE du bâtiment) et la classe DPE médiane des logements du bâtiment. Pour éviter toute fuite de données, seuls les DPE établis avant la date de la vente sont utilisés : les ventes de 2021 en ont donc rarement un (20 %), celles de 2024-2025 presque toujours (94-95 %). Les valeurs manquantes sont gardées, LightGBM les traite nativement. Taux détaillés : reports/correspondance_batiments.csv.
5. Modélisation (src/train.py) : comparaison d'une baseline, d'une régression linéaire, d'une Random Forest et de LightGBM, avec une validation temporelle pour éviter toute fuite de données.
6. Optimisation et modèle final (src/tune.py, src/final.py) : réglages trouvés par Optuna, réentraînement sur 2021-2024, modèles quantiles pour la fourchette, test unique sur 2025.
7. Explicabilité (notebooks/05_explicabilite.ipynb) : analyse SHAP globale et vente par vente, vérification de l'indicateur de confiance.
8. Application (src/predict.py, app/streamlit_app.py) : le calcul est séparé de l'affichage, et les variables sont calculées avec exactement les mêmes fonctions qu'à l'entraînement.


CE QUE J'AI APPRIS

- Une validation honnête change tout : un découpage aléatoire laisse le modèle « voir le futur » et donne des scores trop optimistes.
- Les pires erreurs n'étaient pas des erreurs du modèle : les estimations les plus fausses concernaient des ventes autour de 600 €/m², dont plusieurs dans le même immeuble au même prix. Plutôt que de supprimer les prédictions gênantes, j'ai ajouté une règle de nettoyage fixée à l'avance pour toutes les années.
- Une variable intuitive n'apporte pas forcément beaucoup : le DPE et l'année de construction améliorent à peine le modèle, car l'emplacement et la surface en disent déjà beaucoup. Répéter l'entraînement avec plusieurs graines aléatoires permet de séparer un vrai gain du hasard.
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
- src/batiments.py : récupération BDNB et DPE, correspondance vente → bâtiment, année et DPE sans fuite de données
- src/comparer_variables.py : comparaison sans / avec année et DPE sur la validation 2024
- src/affaire.py : verdict « Bonne affaire ? » et position parmi les ventes comparables
- src/predict.py : géocodage, estimation, explication, comparables, bâtiment de l'adresse
- src/marche.py : agrégations de l'onglet « Explorer le marché » (carte en hexagones, évolution, pièces)
- src/erreurs.py : erreurs sur 2025 par groupe
- src/rapport_nettoyage.py : entonnoir du nettoyage et ventes hors marché, écrits dans reports/ pour l'application
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
python src/comparer_variables.py
python src/final.py
python src/rapport_nettoyage.py

src/features.py télécharge une seule fois l'export BDNB de l'Hérault (environ 640 Mo) et les DPE de Montpellier (API de l'ADEME) dans data/external/, qui n'est pas versionné ; seule une petite table des bâtiments de Montpellier (data/processed/batiments.parquet) l'est, pour l'application.
Le rapport vérifie que ses étapes reproduisent les données du modèle : il s'arrête avec une erreur si elles ont changé.

Lancer les tests :
python -m pytest


LIMITES

- Les données DVF ne contiennent ni l'état du bien, ni l'étage, ni la vue : une partie des écarts de prix est impossible à prévoir.
- Le DPE utilisé est la classe médiane du bâtiment, pas celle du logement vendu ; le simulateur « passer de F à C » donne donc un ordre de grandeur, pas la valeur de travaux. Les DPE ne commencent qu'en juillet 2021.
- Le prix enregistré inclut les caves et parkings vendus avec le logement.
- Le modèle ne couvre que la commune de Montpellier.
- Les lieux OpenStreetMap sont ceux d'aujourd'hui, pas ceux de la date de chaque vente.
- L'année n'est pas une variable du modèle : une évolution rapide du marché peut le décaler.


PISTES D'AMÉLIORATION

- Intégrer les ventes de logements neufs (VEFA) avec une variable « neuf / ancien ».
- Suivre l'évolution du marché avec un indice des prix mensuel.
- Estimer la rentabilité locative à partir des données publiques de loyers.


SOURCES DES DONNÉES

- Demandes de valeurs foncières géolocalisées (DVF) : DGFiP et Etalab, data.gouv.fr, Licence Ouverte 2.0.
- OpenStreetMap : © les contributeurs d'OpenStreetMap, licence ODbL.
- Géocodage : service de géocodage de la Géoplateforme (IGN).
- Base de données nationale des bâtiments (BDNB), export open data du département 34, millésime 2026-02.b (CSTB), Licence Ouverte 2.0.
- DPE Logements existants (depuis juillet 2021), jeu dpe03existant : ADEME, data.ademe.fr, Licence Ouverte 2.0.

Les estimations de cette application sont indicatives et réalisées dans un cadre pédagogique.


AUTEUR

James Rigault, étudiant en 2e année d'école d'ingénieur.
