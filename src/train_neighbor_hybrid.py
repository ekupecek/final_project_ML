"""
HistGradientBoosting + hybrid spatial features:
  - nearest neighbor temperature (k=1) : signal fort et précis
  - distance au voisin le plus proche  : contexte spatial
  - moyenne k=5 voisins                : robustesse au bruit
  - std k=5 voisins                    : cohérence locale

Run from the repository root:
    python src/train_neighbor_hybrid.py

Requires:
    python src/cleaning.py

Outputs:
    reports/neighbor_hybrid_metrics.csv
    models/neighbor_hybrid_hist_gradient_boosting.joblib
    submissions/neighbor_hybrid_submission.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KDTree

# -----------------------------
# Configuration
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR   = REPO_ROOT / "reports"
MODELS_DIR    = REPO_ROOT / "models"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

for folder in [REPORTS_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SENSOR_FRACTION = 0.20
TARGET = "temperature"

K_NEAR  = 1   # signal fort : capteur le plus proche
K_BROAD = 5   # robustesse : moyenne des 5 plus proches

BASE_FEATURES = [
    "time", "power",
    "coor_x", "coor_y", "coor_z",
    "time_years", "r_xy", "r_xyz", "abs_y",
    "power_x_time", "power_over_r_xy",
]

# Features hybrides : précision (k=1) + robustesse (k=5)
SPATIAL_FEATURES = [
    "nn_temp",        # température du voisin le plus proche (k=1)
    "nn_dist",        # distance au voisin le plus proche
    "broad_temp_mean",# moyenne des 5 voisins — lisse le bruit
    "broad_temp_std", # std des 5 voisins — mesure la cohérence locale
]

ALL_FEATURES = BASE_FEATURES + SPATIAL_FEATURES

MODEL_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=63,
    l2_regularization=0.01,
    early_stopping=True,
    validation_fraction=0.10,
    random_state=RANDOM_STATE,
)


# -----------------------------
# Sensor split
# -----------------------------
def sensor_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SENSOR_FRACTION, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(splitter.split(train, groups=train["sensor"]))
    return train.iloc[train_idx].copy(), train.iloc[val_idx].copy()


# -----------------------------
# Neighbor map (k_max voisins en une seule requête KD-tree)
# -----------------------------
def build_neighbor_map(
    query_sensors: pd.DataFrame,
    source_sensors: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    coord_cols = ["coor_x", "coor_y", "coor_z"]
    source_u = source_sensors.groupby("sensor")[coord_cols].mean()
    query_u  = query_sensors.groupby("sensor")[coord_cols].mean()

    tree = KDTree(source_u[coord_cols].values)
    dists, idxs = tree.query(query_u[coord_cols].values, k=k + 1)

    rows = []
    src_index = source_u.index.tolist()
    for i, sensor in enumerate(query_u.index):
        rank = 0
        for dist, idx in zip(dists[i], idxs[i]):
            neighbor = src_index[idx]
            if neighbor == sensor:
                continue
            rows.append({"sensor": sensor, "neighbor_sensor": neighbor,
                         "neighbor_dist": dist, "neighbor_rank": rank})
            rank += 1
            if rank >= k:
                break
    return pd.DataFrame(rows)


# -----------------------------
# Temperature lookup vectorisé
# -----------------------------
def build_temp_lookup(source: pd.DataFrame) -> dict:
    lookup = {}
    for sensor, grp in source.groupby("sensor", observed=True):
        s = grp.set_index("time")["temperature"].sort_index()
        lookup[sensor] = s[~s.index.duplicated(keep="first")]
    return lookup


def expand_neighbors(
    df: pd.DataFrame,
    neighbor_map: pd.DataFrame,
    temp_lookup: dict,
) -> pd.DataFrame:
    """Long-format : une ligne par (row_idx, voisin)."""
    expanded = (
        df[["sensor", "time"]]
        .reset_index(names="row_idx")
        .merge(neighbor_map, on="sensor", how="left")
    )
    parts = []
    for nbr, grp in expanded.groupby("neighbor_sensor", observed=True):
        series = temp_lookup.get(nbr)
        if series is None or series.empty:
            grp = grp.copy()
            grp["neighbor_temp"] = np.nan
            parts.append(grp)
            continue
        times     = grp["time"].values
        sorted_t  = series.index.values
        idxs      = np.searchsorted(sorted_t, times)
        idxs      = np.clip(idxs, 0, len(sorted_t) - 1)
        prev      = np.clip(idxs - 1, 0, len(sorted_t) - 1)
        use_prev  = np.abs(sorted_t[prev] - times) < np.abs(sorted_t[idxs] - times)
        idxs      = np.where(use_prev, prev, idxs)
        grp = grp.copy()
        grp["neighbor_temp"] = series.iloc[idxs].values
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


# -----------------------------
# Calcul des features hybrides
# -----------------------------
def compute_hybrid_features(
    df: pd.DataFrame,
    expanded: pd.DataFrame,
) -> pd.DataFrame:
    """
    nn_temp / nn_dist       — voisin le plus proche (rank 0)
    broad_temp_mean / std   — moyenne sur K_BROAD voisins
    """
    # Signal fort : voisin le plus proche (rank=0)
    near = (
        expanded[expanded["neighbor_rank"] == 0]
        [["row_idx", "neighbor_temp", "neighbor_dist"]]
        .rename(columns={"neighbor_temp": "nn_temp", "neighbor_dist": "nn_dist"})
        .set_index("row_idx")
    )

    # Robustesse : moyenne sur les K_BROAD premiers voisins
    broad = (
        expanded[expanded["neighbor_rank"] < K_BROAD]
        .groupby("row_idx")
        .agg(
            broad_temp_mean=("neighbor_temp", "mean"),
            broad_temp_std =("neighbor_temp", "std"),
        )
    )

    return (
        df.reset_index(drop=True)
        .join(near)
        .join(broad)
    )


def fill_spatial_nans(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    for col in SPATIAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(reference[col].median())
    return df


# -----------------------------
# Evaluation & submission
# -----------------------------
def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }


def save_submission(test: pd.DataFrame, predictions: np.ndarray) -> Path:
    id_candidates = ["id", "Id", "ID", "sample_id", "row_id"]
    id_col = next((c for c in id_candidates if c in test.columns), None)
    if id_col:
        sub = pd.DataFrame({id_col: test[id_col].to_numpy(), TARGET: predictions})
    else:
        sub = pd.DataFrame({"Id": np.arange(len(test)), TARGET: predictions})
    path = SUBMISSIONS_DIR / "neighbor_hybrid_submission.csv"
    sub.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Chargement des données...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    train = train.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    print(f"  Train : {train.shape} | Test : {test.shape}")

    # Séparation capteurs (avant features → pas de fuite)
    print("Séparation par capteurs...")
    train_part, val_part = sensor_split(train)
    print(f"  Entraînement : {train_part['sensor'].nunique()} capteurs | {len(train_part):,} lignes")
    print(f"  Validation   : {val_part['sensor'].nunique()} capteurs | {len(val_part):,} lignes")

    # Cartes de voisins (K_BROAD = nombre max de voisins utiles)
    print(f"Construction des cartes de voisins (k_max={K_BROAD})...")
    train_coords     = train_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    all_train_coords = train[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    test_coords      = test[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")

    nm_train = build_neighbor_map(train_coords,     train_coords,     K_BROAD)
    nm_val   = build_neighbor_map(
        val_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor"),
        train_coords, K_BROAD
    )
    nm_test  = build_neighbor_map(test_coords, all_train_coords, K_BROAD)

    # Lookup de températures (une seule fois par split)
    print("Lookup de températures...")
    lookup_train = build_temp_lookup(train_part)
    lookup_all   = build_temp_lookup(train)

    # Expansion long-format puis features hybrides
    print("Calcul des features spatiales hybrides...")
    train_part = compute_hybrid_features(train_part, expand_neighbors(train_part, nm_train, lookup_train))
    val_part   = compute_hybrid_features(val_part,   expand_neighbors(val_part,   nm_val,   lookup_train))
    test       = compute_hybrid_features(test,       expand_neighbors(test,       nm_test,  lookup_all))

    # Remplissage des NaN avec la médiane d'entraînement
    for col in SPATIAL_FEATURES:
        if col in train_part.columns:
            med = train_part[col].median()
            train_part[col] = train_part[col].fillna(med)
            val_part[col]   = val_part[col].fillna(med)
            test[col]       = test[col].fillna(med)

    features = [f for f in ALL_FEATURES if f in train_part.columns and f in test.columns]
    print(f"Features totales : {len(features)}  "
          f"(base={len(BASE_FEATURES)}, spatiales={len(SPATIAL_FEATURES)})")

    # Entraînement
    print("\nEntraînement HistGradientBoosting...")
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(train_part[features], train_part[TARGET])

    # Évaluation
    val_pred = model.predict(val_part[features])
    metrics  = evaluate(val_part[TARGET], val_pred)
    print(f"\nValidation (capteurs non vus) :")
    print(f"  RMSE : {metrics['rmse']:.5f}")
    print(f"  MAE  : {metrics['mae']:.5f}")
    print(f"  R²   : {metrics['r2']:.5f}")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "neighbor_hybrid_metrics.csv", index=False)

    # Tableau comparatif
    print("\n--- Comparaison locale ---")
    print(f"  {'Modèle':35} {'RMSE':>9} {'MAE':>9} {'R²':>8}")
    print(f"  {'-'*65}")
    comparisons = {
        "baseline_metrics.csv":        "Baseline (HistGB)",
        "neighbor_metrics.csv":        "HistGB + voisin k=1",
        "neighbor_hybrid_metrics.csv": "HistGB + hybride k=1+5  ← actuel",
    }
    for fname, label in comparisons.items():
        p = REPORTS_DIR / fname
        if p.exists():
            r = pd.read_csv(p).iloc[0]
            print(f"  {label:35} {r['rmse']:9.5f} {r['mae']:9.5f} {r['r2']:8.5f}")

    # Modèle final sur toutes les données d'entraînement
    print("\nEntraînement final sur toutes les données nettoyées...")
    nm_full   = build_neighbor_map(all_train_coords, all_train_coords, K_BROAD)
    train     = compute_hybrid_features(train, expand_neighbors(train, nm_full, lookup_all))
    for col in SPATIAL_FEATURES:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())

    final_model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(train[features], train[TARGET])

    joblib.dump({"model": final_model, "features": features},
                MODELS_DIR / "neighbor_hybrid_hist_gradient_boosting.joblib")

    test_f = test[features].copy()
    if test_f.isna().any().any():
        test_f = test_f.fillna(train[features].median())

    test_pred = final_model.predict(test_f)
    path = save_submission(test, test_pred)
    print(f"Soumission sauvegardée : {path}")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
