Notes d'exploration: 
Chiffres clés
Nombre de lignes / de ventes uniques : 81361 / 35963
Années couvertes : 2021 - 2025
Prix au m² médian appartements / maisons : Appart : 3319.0 , Maison : 3917.0
Pièges repérés
Ventes à plusieurs lignes, prix total répété
Caves et parkings inclus dans le prix
Valeurs absurdes (exemples d'id_mutation) :
       id_mutation    type_local          valeur_fonciere  surface_reelle_bati  \
59537  2024-400977       Maison              1.0                200.0   
71288  2025-423883  Appartement              1.0                125.0   
9613   2021-557206  Appartement              1.0                 85.0 

Notes additionnels : Beaucoup de rouge dans les quartiers antigone , comédie , écusson , port marianne , bouttonet . Il y a beaucoup de bleu sur le coté ouest de Montpellier , mosson , cévènnes , croix d'argent , près d'arènes

Questions pour la semaine 2
Quels seuils pour retirer les valeurs extrêmes ? Un seuil de 50 000 euros sur la valuer_fonciere
Que faire des ventes avec plusieurs logements ?

Après nettoyage:

Lignes brutes                   81361
Ventes (mutations)              30699
Ventes d'un seul logement       24516
Après filtres                   23703

Décisions :

Ventes de plusieurs logements retirées : impossible de répartir le prix total.
Caves et parkings gardés, comptés dans nb_dependances.
Surfaces gardées entre 9 et 400 m².
Prix au m² : retrait du 1 % le plus bas et du 1 % le plus haut, séparément pour appartements et maisons.
Limites :

Le prix inclut les dépendances, qu'on ne peut pas séparer.
DVF ne dit pas si le bien est neuf, son état ni son DPE.

Variables géographiques (semaine 3)
Source : OpenStreetMap, lieux dans un rayon de 10 km autour de la Comédie
Variables créées :
tram            227 lieux trouvés
ecole           356 lieux trouvés
parc            338 lieux trouvés
commerce       4336 lieux trouvés
restaurant     1373 lieux trouvés
13 nouvelles variables : dist_centre, dist_tram, dist_ecole, nb_ecole_500m, dist_parc, dist_commerce, nb_commerce_500m, dist_restaurant, nb_restaurant_500m, est_maison, surface_par_piece, annee, mois

Variables les plus corrélées au prix au m² : ...
Prix au m² médian des appartements à moins de 200 m du tram : 3490.0 €, à plus de 1 km : 3066.0 €
Attention : cet écart mélange l'effet tram et l'effet centre-ville

OSM est collaboratif : certains lieux peuvent manquer ou être mal tagués
Ce sont les lieux d'aujourd'hui, pas ceux de la date de la vente

MODELISATION (SEMAINE 4)

Méthode
La cible est le prix au m², multiplié ensuite par la surface. Les erreurs sont mesurées sur le prix total.
Découpage temporel : entraînement sur 2021 à 2023 (15 333 ventes), validation sur 2024 (3 891 ventes).
L'année 2025 est gardée de côté comme jeu de test jusqu'à la semaine 5.
Baseline : prix au m² médian du code postal et du type de bien, calculé sur l'entraînement, multiplié par la surface.
17 variables : surface, pièces, type, dépendances, terrain, coordonnées, distances et densités OpenStreetMap.
L'année n'est pas utilisée, car les modèles à arbres ne savent pas extrapoler une tendance.

Résultats v1 (validation 2024)
Baseline : MAE 46 960 €, MAPE 30,0 %, erreur médiane 18,8 %, 28,4 % des estimations à 10 % près
Régression linéaire (Ridge) : MAE 42 807 €, MAPE 26,7 %, erreur médiane 16,4 %, 31,9 % à 10 % près
Random Forest : MAE 32 316 €, MAPE 19,2 %, erreur médiane 11,6 %, 44,9 % à 10 % près
LightGBM : MAE 32 239 €, MAPE 19,3 %, erreur médiane 11,4 %, 45,2 % à 10 % près

Observations
LightGBM réduit l'erreur médiane d'environ 40 % par rapport à la baseline (de 18,8 % à 11,4 %).
Ridge fait à peine mieux que la baseline, ce qui montre que les effets sont non linéaires.
Random Forest et LightGBM sont presque à égalité. Je retiens LightGBM car il est plus rapide.
La MAPE (19 %) est bien plus haute que l'erreur médiane (11 %) : quelques très grosses erreurs tirent la moyenne vers le haut.

Analyse des pires erreurs
Les 10 pires erreurs sont des ventes à environ 600 €/m², alors que les 4 modèles prédisent un prix normal.
Plusieurs ventes se trouvent dans le même immeuble, au même prix au m² (623 €/m²) : probablement une vente en bloc.
Causes probables : viagers (seul le bouquet est enregistré), ventes partielles (nue-propriété, quote-part), ventes familiales, ventes en bloc.
Ce ne sont pas des erreurs du modèle mais des ventes hors marché.

Correction : nouvelle règle de nettoyage
Je retire les ventes dont le prix au m² est inférieur à 30 % de la médiane de leur code postal, de leur type de bien et de leur année.
C'est une règle métier fixée à l'avance et appliquée à toutes les années. Je n'ai pas supprimé les pires prédictions, ce qui fausserait l'évaluation.


Résultats v2 (après la nouvelle règle)
                                     MAE (€)  MAPE (%)  Erreur médiane (%)  Part à ±10 % (%)
Baseline (médiane code postal)  46960.4      30.0                18.8              28.4
Régression linéaire (Ridge)     42806.5      26.7                16.4              31.9
Random Forest                   32316.0      19.2                11.6              44.9
LightGBM                        32238.6      19.3                11.4              45.2
Attention : une partie du gain vient du fait que la validation est plus facile (elle ne contient plus de ventes imprévisibles par nature), pas seulement d'un meilleur modèle.

Où le modèle se trompe le plus (notebook 04, cellule 3)
Appartements ou maisons : X
Petites ou grandes surfaces : X
Code postal le moins bien prédit : X

Meilleure erreur médiane sur 2024 : 11.22 %
Meilleurs réglages : {'n_estimators': 875, 'learning_rate': 0.014482713684487343, 'num_leaves': 100, 'min_child_samples': 13, 'subsample': 0.7520657266851974, 'colsample_bytree': 0.9174998188274811, 'reg_lambda': 7.3078468454405625}


"Test 2025 lancé le 09/10/2026 , réglages figés". 

OPTIMISATION ET MODELE FINAL (SEMAINE 5)

Optuna, 40 essais : erreur médiane sur 2024 de  % (avant) à X % (après).
Meilleurs réglages : (copie le contenu de reports/meilleurs_parametres.json)

Test 2025 lancé le (date), une seule fois, réglages figés.
Baseline : erreur médiane X %, X % des estimations à 10 % près
LightGBM final : erreur médiane X %, X % des estimations à 10 % près
Fourchette 10 %-90 % : X % des vrais prix sont dedans
Ratio médian estimation / prix réel en 2025 : X

Explicabilité (SHAP)
Variables les plus importantes : X, X, X
Effet du tram à autres caractéristiques égales : X €/m² à moins de 200 m, contre X €/m² à plus de 1 km
Comparaison avec la semaine 3 : l'écart brut était de X €/m², donc une partie venait de l'effet centre-ville.