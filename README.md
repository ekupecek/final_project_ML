# Temperature Prediction — Nuclear Waste Storage Tunnel

Machine learning project to predict rock temperatures in a nuclear waste storage tunnel (Mont-Terri experiment, EPFL) from spatial and temporal sensor measurements.

**Group 12 — Anna Billon, Eva Kupecek**

---

## Problem Statement

Sensors distributed in a 2D cross-section of a tunnel measure temperature continuously over ~250 years. The goal is to predict temperature at positions where no sensor exists.

The geometry consists of two zones around a central heated canister:
- **Buffer** (coor_x < 1.4 m) — granular bentonite, many hot sensors
- **OPA** (coor_x > 1.4 m) — Opalinus Clay host rock, fewer and cooler sensors

The buffer boundary is a **vertical line at x = 1.4 m** (not a circle of radius 1.4 m).

> OPA predictions are weighted more heavily in the final Kaggle metric.

**Dataset:**
| File | Rows | Description |
|---|---|---|
| `train.parquet` | 6 626 928 | 232 sensors × 9128 time steps |
| `test.parquet` | 2 190 480 | 80 unknown sensors × 9127 time steps |
| `sensors.parquet` | — | 3D coordinates of each sensor |

**Key insight:** Test sensors are 0.2–1.5 units away from train sensors and share the same time steps. The problem is a **spatial interpolation**: at each time t, we know the temperature of 232 train sensors → predict the 80 test sensors.

**Kaggle metric:** RMSE (°C, lower is better)

---

## Project Structure

```
final_project_ML/
├── data_parquet_2026/           # Raw data (not versioned)
├── data/processed/              # Cleaned data (generated, not versioned)
├── src/
│   ├── cleaning.py              # Cleaning pipeline + feature engineering
│   └── train_best.py            # Best model: HistGB + global IDW + KNN hybrid
├── models/
│   └── best_model.joblib        # Saved model (not versioned)
├── reports/
│   ├── best_metrics.csv         # Validation metrics
│   └── best_feature_importance.csv  # Feature importance
├── submissions/
│   └── best_submission.csv      # Kaggle submission (Kaggle score: 3.42)
└── train.ipynb                  # Full pipeline notebook (cleaning + model + analysis)
```

---

## How to Reproduce

```bash
# 1. Clean data and engineer base features
python src/cleaning.py

# 2. Train best model
python src/train_best.py
# → submissions/best_submission.csv
```

---

## Cleaning Pipeline (`cleaning.py`)

Four cascaded filters applied in order:

| Step | Description | Parameter |
|---|---|---|
| Physical limits | Remove temperatures < 0 °C or > 200 °C | `TEMP_MIN=0`, `TEMP_MAX=200` |
| IQR per sensor | Outliers beyond 4× each sensor's IQR | `IQR_MULTIPLIER=4.0` |
| Temporal spikes | Robust z-score on rolling window | window=21, threshold=8σ |
| Failed sensors | Sensors with > 20% bad readings excluded | threshold=20% |

**Results:** 311 862 rows removed (4.7%), 10 sensors excluded → **6 180 661 rows**, 222 active sensors.

Physical limits are applied first so extreme values (e.g. −9999 °C) do not inflate the IQR and mask moderate outliers.

**Engineered base features:**

| Feature | Formula | Physical meaning |
|---|---|---|
| `r_xy` | `√(x² + y²)` | 2D distance to canister (geometry is 2D) |
| `r_xyz` | `√(x² + y² + z²)` | 3D distance to origin |
| `time_years` | `time / (365.25 × 24 × 3600)` | Time in years |
| `abs_y` | `\|y\|` | Vertical position |
| `power_x_time` | `power × time_years` | Cumulative energy proxy |
| `power_over_r_xy` | `power / r_xy` | Heat flux proxy |

---

## Validation Strategy

**Sensor-level split** — entire sensors are held out to prevent data leakage. No validation sensor is used as a source for spatial features.

| Set | Sensors | Rows |
|---|---|---|
| Training | 185 (80%) | ~4 930 000 |
| Validation | 47 (20%) | ~1 250 000 |

---

## Model Architecture

### Global IDW — Inverse Distance Weighting

At each time step t, the known sensor temperatures form a spatial field. IDW interpolates the temperature at a query point by weighting each source sensor by the inverse square of its distance:

```
T_idw(j, t) = Σᵢ [T_i(t) / d(j,i)²] / Σᵢ [1 / d(j,i)²]
```

Uses all 222 source sensors. The IDW estimate is passed as a **feature** to HistGB (not used directly as a prediction).

### KNN Hybrid Features

Nearest-neighbour temperatures are looked up at each time step:

| Feature | k | Description |
|---|---|---|
| `nn_temp` | 1 | Temperature of the single nearest sensor |
| `nn_dist` | 1 | Distance to the nearest sensor |
| `broad_temp_mean` | 5 | Mean temperature of 5 nearest sensors |
| `broad_temp_std` | 5 | Std of 5 nearest sensor temperatures |

### HistGradientBoostingRegressor

Direct temperature prediction from 16 features:
- 11 base features (coordinates, time, power, engineered)
- 1 global IDW estimate
- 4 KNN spatial features

No blending, no OPA sample weights.

---

## Results

| Submission | Val RMSE | Kaggle Score |
|---|---|---|
| `baseline_submission.csv` | — | 4.054 |
| `neighbor_submission.csv` | — | 3.747 |
| `best_submission.csv` | 5.190 | **3.42** |

---

## Architecture Decision Log

| Decision | Rationale |
|---|---|
| `r_xy` over `r_xyz` | Geometry is 2D (cross-section); z adds noise |
| `coor_x`, `coor_y` separately | Tunnel is not radially symmetric; material zones break symmetry |
| Global IDW over local (k=20) | Local IDW k=20 tested → worse results (4.2 on Kaggle vs 3.42) |
| IDW as feature, not blend | Blending α×IDW+(1-α)×HistGB tested → consistently worse than direct prediction |
| No OPA sample weights | Adding OPA weights (×3) tested → did not improve overall RMSE |
| Sensor-level train/val split | Prevents any spatial leakage between train and validation |
| Keep all 16 features | Removing low-importance features (permutation) degraded RMSE — correlated features protect each other during training |
| Buffer boundary at coor_x = 1.4 m | Tunnel cross-section shows rectangular buffer zone, not circular |

---

## Feature Importance (Native — Split Gain)

| Feature | Importance |
|---|---|
| `power_over_r_xy` | 7.28 |
| `power` | 4.10 |
| `time` | 1.69 |
| `coor_x` | 1.64 |
| `idw_temp` | 1.25 |
| `power_x_time` | 1.14 |
| `r_xy` | 0.63 |
| `coor_y` | 0.47 |
| `nn_dist` | 0.15 |
| `r_xyz`, `coor_z`, `time_years`, `abs_y` | ≈ 0 |

> Features with near-zero native importance were kept: removing them caused val RMSE to increase from 5.190 → 5.430, because permutation importance misleads when features are correlated.

---

## Possible Further Improvements

- **Tune IDW power** `p`: test p=1, p=3 to adjust spatial decay
- **Add angle feature** θ = `atan2(y, x)`: tunnel is not symmetric, angle matters
- **Temporal gradient**: rate of temperature change per neighbouring sensor
- **Kriging / GPR**: models spatial covariance — more principled than IDW
- **Sensor drift detection**: bonus points available on Kaggle leaderboard
