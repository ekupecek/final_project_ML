# Prédiction de température — Stockage de déchets nucléaires

Projet de machine learning pour prédire les températures dans un tunnel de stockage de déchets nucléaires à partir de mesures de capteurs spatiaux et temporels.

---

## Contexte

Des capteurs répartis en 3D dans un tunnel mesurent la température en continu. L'objectif est de prédire la température à des positions où aucun capteur n'existe, à partir des mesures des capteurs voisins, du temps écoulé et de la puissance de chauffage.

**Données :**
- `train.parquet` — 6 626 928 lignes (capteur, temps, puissance, température)
- `test.parquet`  — 2 190 480 lignes (à prédire)
- `sensors.parquet` — 232 capteurs avec coordonnées (x, y, z)

**Métrique Kaggle :** RMSE (Root Mean Squared Error) — erreur en °C

---

## Structure du projet

```
final_project_ML/
├── data_parquet_2026/          # Données brutes (non versionnées)
├── data/processed/             # Données nettoyées (générées, non versionnées)
├── src/
│   ├── cleaning.py             # Pipeline de nettoyage
│   ├── train_baseline.py       # Modèle baseline HistGradientBoosting
│   ├── train_lgbm.py           # LightGBM seul
│   ├── train_features.py       # HistGB + features physiques
│   ├── train_neighbor.py       # HistGB + température voisin (k=1)
│   ├── train_neighbor_hybrid.py# HistGB + features spatiales hybrides ← meilleur
│   ├── train_lgbm_hybrid.py    # LightGBM + features spatiales hybrides
│   ├── train_lgbm_neighbor.py  # LightGBM + température voisins
│   └── search_k_neighbors.py   # Recherche du meilleur k
├── models/                     # Modèles entraînés (non versionnés)
├── reports/                    # Métriques et importances de features
├── submissions/                # Fichiers CSV à soumettre sur Kaggle
└── README.md
```

---

## Reproduire les résultats

```bash
# 1. Nettoyage des données
python src/cleaning.py

# 2. Meilleur modèle (HistGB + features spatiales hybrides)
python src/train_neighbor_hybrid.py

# 3. Soumission : submissions/neighbor_hybrid_submission.csv
```

---

## Pipeline de nettoyage (`cleaning.py`)

Le script effectue un nettoyage multi-couches :

| Étape | Description | Paramètre |
|---|---|---|
| Limites physiques | Supprime les températures < 0°C ou > 200°C | `TEMP_MIN=0`, `TEMP_MAX=200` |
| IQR par capteur | Supprime les outliers à 4× l'IQR de chaque capteur | `IQR_MULTIPLIER=4.0` |
| Spikes temporels | Détecte les pics isolés via z-score robuste sur fenêtre glissante | fenêtre=21, seuil=8σ |
| Capteurs défaillants | Supprime les capteurs avec >20% de données mauvaises | seuil=20% |
| Dérive temporelle | Identifie les capteurs avec une dérive anormale (slope z-score > 4) | seuil=4σ |

**Résultats du nettoyage :**
- Lignes supprimées : 311 862 (4.7%)
- Capteurs défaillants retirés : 10
- Données nettoyées : **6 180 661 lignes**, 232 → 222 capteurs actifs

**Features ajoutées :**

| Feature | Formule | Rôle |
|---|---|---|
| `time_years` | `time / (365.25 × 24 × 3600)` | Temps en années |
| `r_xy` | `√(x² + y²)` | Distance radiale 2D |
| `r_xyz` | `√(x² + y² + z²)` | Distance 3D |
| `abs_y` | `\|y\|` | Symétrie axiale |
| `power_x_time` | `power × time_years` | Énergie cumulée |
| `power_over_r_xy` | `power / r_xy` | Densité de puissance 2D |

---

## Stratégie de validation

**Split par capteur entier** (pas aléatoire) : 80% des capteurs pour l'entraînement, 20% pour la validation.

Pourquoi ? Le but est de prédire des températures à des **positions sans capteur** — tenir des capteurs entiers en validation teste la généralisation spatiale, ce qui correspond au vrai test Kaggle.

- Entraînement : **185 capteurs** — 4 930 181 lignes
- Validation : **47 capteurs** — 1 250 480 lignes (jamais vus à l'entraînement)

---

## Expériences et progression

### Scores locaux (validation 47 capteurs)

| Script | Modèle | RMSE | MAE | R² |
|---|---|---|---|---|
| `train_baseline.py` | HistGradientBoosting | 5.770 | 2.806 | 0.834 |
| `train_lgbm.py` | LightGBM seul | 5.539 | 2.508 | 0.847 |
| `train_features.py` | HistGB + features physiques | 5.869 | 2.811 | 0.828 |
| `train_neighbor.py` | HistGB + voisin k=1 | 5.224 | 2.372 | 0.864 |
| `train_neighbor_hybrid.py` | **HistGB + hybride k=1+5** | **5.289** | **2.403** | **0.860** |
| `train_lgbm_hybrid.py` | LightGBM + hybride k=1+5 | 5.524 | 2.565 | 0.848 |

### Scores Kaggle (public leaderboard)

| Soumission | Score Kaggle | Δ vs baseline |
|---|---|---|
| `baseline_submission.csv` | 4.054 | référence |
| `lgbm_submission.csv` | 4.171 | +0.117 ↑ pire |
| `neighbor_submission.csv` | 3.747 | −0.307 ✓ |
| `neighbor_hybrid_submission.csv` | ~3.5 | −0.55 ✓ meilleur |

---

## Meilleure approche : features spatiales hybrides

### Principe

Pour chaque point à prédire `(capteur_j, temps_t)`, on regarde ce que font les capteurs voisins **au même moment** :

```
nn_temp        = température du capteur le plus proche (k=1)
nn_dist        = distance au capteur le plus proche
broad_temp_mean = moyenne des 5 capteurs les plus proches (k=5)
broad_temp_std  = écart-type des 5 capteurs (cohérence locale)
```

**k=1** apporte le signal fort (le voisin immédiat est très prédictif).
**k=5** apporte la robustesse (lisse les mesures bruitées d'un capteur isolé).

### Recherche du meilleur k (`search_k_neighbors.py`)

| k | RMSE local |
|---|---|
| **1** | **5.200** |
| 3 | 5.688 |
| 5 | 5.900 |
| 8 | 5.651 |
| 10 | 6.001 |
| 20 | 6.192 |

k=1 domine car la **dilution spatiale** (moyenner des capteurs lointains) introduit plus de biais que le bruit d'une seule mesure.

### Absence de fuite de données

Les températures des 47 capteurs de **validation ne servent jamais de source** pour les features voisines des données d'entraînement. La carte de voisins de validation est construite exclusivement à partir des 185 capteurs d'entraînement.

### Features les plus importantes (HistGB + hybride)

| Feature | Rôle |
|---|---|
| `nn_temp` | Température du voisin le plus proche — signal spatial direct |
| `power_over_r_xy` | Densité de puissance — très physique |
| `power` | Puissance de chauffage brute |
| `broad_temp_mean` | Moyenne des 5 voisins — robustesse |
| `time` | Temps écoulé |
| `coor_x` | Position spatiale |

---

## Pourquoi LightGBM n'améliore pas ici

LightGBM seul a introduit de l'overfitting spatial (score Kaggle 4.17 vs 4.05 baseline). Combiné aux features hybrides, il reste moins bon que HistGB (RMSE 5.52 vs 5.29) car `nn_temp` est déjà une quasi-prédiction directe — HistGradientBoosting l'exploite plus efficacement que l'approche par arbres de LightGBM.

---

## Pistes d'amélioration futures

- **Ensemble** : moyenne pondérée de plusieurs modèles
- **Plus de voisins contextuels** : ajouter le 2e et 3e voisin comme features séparées plutôt qu'en moyenne
- **Features temporelles** : encoder les patterns de variation de température dans le temps par capteur
- **Modèle physique** : intégrer l'équation de diffusion thermique comme prior
