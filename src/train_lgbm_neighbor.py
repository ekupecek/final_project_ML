"""
LightGBM + neighbor sensor temperature features.

Combines the spatial context of neighbor temperatures with LightGBM's
capacity to learn complex interactions.

Run from the repository root:
    python src/train_lgbm_neighbor.py

Requires:
    python src/cleaning.py

Outputs:
    reports/lgbm_neighbor_metrics.csv
    reports/lgbm_neighbor_feature_importance.csv
    models/lgbm_neighbor_model.pkl
    submissions/lgbm_neighbor_submission.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KDTree

# -----------------------------
# Configuration
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

for folder in [REPORTS_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SENSOR_FRACTION = 0.20
TARGET = "temperature"
K_NEIGHBORS = 5

BASE_FEATURES = [
    "time", "power",
    "coor_x", "coor_y", "coor_z",
    "time_years", "r_xy", "r_xyz", "abs_y",
    "power_x_time", "power_over_r_xy",
]

NEIGHBOR_FEATURES = [
    "neighbor_temp_mean",
    "neighbor_temp_std",
    "neighbor_temp_min",
    "neighbor_temp_max",
    "neighbor_temp_dist_weighted",
    "neighbor_dist_mean",
]

ALL_FEATURES = BASE_FEATURES + NEIGHBOR_FEATURES

# Moderate regularization — avoid the overfitting seen in the first LightGBM attempt
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 0.5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


# -----------------------------
# Sensor split
# -----------------------------
def sensor_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SENSOR_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(splitter.split(train, groups=train["sensor"]))
    return train.iloc[train_idx].copy(), train.iloc[val_idx].copy()


# -----------------------------
# Neighbor features (identical logic to train_neighbor.py)
# -----------------------------
def build_neighbor_map(
    query_sensors: pd.DataFrame,
    source_sensors: pd.DataFrame,
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    coord_cols = ["coor_x", "coor_y", "coor_z"]
    source_unique = source_sensors.groupby("sensor")[coord_cols].mean()
    query_unique  = query_sensors.groupby("sensor")[coord_cols].mean()

    tree = KDTree(source_unique[coord_cols].values)
    dists, idxs = tree.query(query_unique[coord_cols].values, k=k + 1)

    rows = []
    source_index = source_unique.index.tolist()
    for i, sensor in enumerate(query_unique.index):
        rank = 0
        for dist, idx in zip(dists[i], idxs[i]):
            neighbor = source_index[idx]
            if neighbor == sensor:
                continue
            rows.append({
                "sensor": sensor,
                "neighbor_sensor": neighbor,
                "neighbor_dist": dist,
                "neighbor_rank": rank,
            })
            rank += 1
            if rank >= k:
                break
    return pd.DataFrame(rows)


def add_neighbor_features(
    df: pd.DataFrame,
    source_train: pd.DataFrame,
    neighbor_map: pd.DataFrame,
) -> pd.DataFrame:
    temp_by_sensor: dict = {}
    for sensor, grp in source_train.groupby("sensor", observed=True):
        series = grp.set_index("time")["temperature"].sort_index()
        temp_by_sensor[sensor] = series[~series.index.duplicated(keep="first")]

    expanded = (
        df[["sensor", "time"]]
        .reset_index(names="row_idx")
        .merge(neighbor_map, on="sensor", how="left")
    )

    parts = []
    for nbr_sensor, grp in expanded.groupby("neighbor_sensor", observed=True):
        series = temp_by_sensor.get(nbr_sensor)
        if series is None or series.empty:
            grp = grp.copy()
            grp["neighbor_temp"] = np.nan
            parts.append(grp)
            continue
        times = grp["time"].values
        sorted_times = series.index.values
        idxs = np.searchsorted(sorted_times, times)
        idxs = np.clip(idxs, 0, len(sorted_times) - 1)
        prev_idxs = np.clip(idxs - 1, 0, len(sorted_times) - 1)
        use_prev = np.abs(sorted_times[prev_idxs] - times) < np.abs(sorted_times[idxs] - times)
        idxs = np.where(use_prev, prev_idxs, idxs)
        grp = grp.copy()
        grp["neighbor_temp"] = series.iloc[idxs].values
        parts.append(grp)

    expanded = pd.concat(parts, ignore_index=True)
    expanded["weight"] = 1.0 / (expanded["neighbor_dist"] + 1e-6)
    expanded["weighted_temp"] = expanded["neighbor_temp"] * expanded["weight"]

    agg = (
        expanded.groupby("row_idx")
        .agg(
            neighbor_temp_mean=("neighbor_temp", "mean"),
            neighbor_temp_std=("neighbor_temp", "std"),
            neighbor_temp_min=("neighbor_temp", "min"),
            neighbor_temp_max=("neighbor_temp", "max"),
            weight_sum=("weight", "sum"),
            weighted_temp_sum=("weighted_temp", "sum"),
            neighbor_dist_mean=("neighbor_dist", "mean"),
        )
    )
    agg["neighbor_temp_dist_weighted"] = agg["weighted_temp_sum"] / agg["weight_sum"]
    agg = agg.drop(columns=["weight_sum", "weighted_temp_sum"])

    return df.reset_index(drop=True).join(agg)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }


def save_submission(test: pd.DataFrame, predictions: np.ndarray) -> Path:
    id_candidates = ["id", "Id", "ID", "sample_id", "row_id"]
    id_col = next((c for c in id_candidates if c in test.columns), None)
    if id_col:
        submission = pd.DataFrame({id_col: test[id_col].to_numpy(), TARGET: predictions})
    else:
        submission = pd.DataFrame({"Id": np.arange(len(test)), TARGET: predictions})
    path = SUBMISSIONS_DIR / "lgbm_neighbor_submission.csv"
    submission.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    train = train.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    print(f"  Train: {train.shape} | Test: {test.shape}")

    # Sensor split first (prevents leakage)
    print("Creating sensor-based validation split...")
    train_part, val_part = sensor_split(train)
    print(f"  Train: {train_part['sensor'].nunique()} sensors | {len(train_part):,} rows")
    print(f"  Val:   {val_part['sensor'].nunique()} sensors | {len(val_part):,} rows")

    # Build neighbor maps
    print(f"Building neighbor maps (k={K_NEIGHBORS})...")
    train_coords     = train_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    all_train_coords = train[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    test_coords      = test[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")

    neighbor_map_train = build_neighbor_map(train_coords, train_coords)
    neighbor_map_val   = build_neighbor_map(
        val_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor"), train_coords
    )
    neighbor_map_test  = build_neighbor_map(test_coords, all_train_coords)

    # Add neighbor features
    print("Adding neighbor features to train split...")
    train_part = add_neighbor_features(train_part, train_part, neighbor_map_train)
    print("Adding neighbor features to validation split...")
    val_part   = add_neighbor_features(val_part,   train_part, neighbor_map_val)
    print("Adding neighbor features to test set...")
    test       = add_neighbor_features(test,       train,      neighbor_map_test)

    # Fill NaN with training medians
    for col in NEIGHBOR_FEATURES:
        if col in train_part.columns:
            median_val = train_part[col].median()
            train_part[col] = train_part[col].fillna(median_val)
            val_part[col]   = val_part[col].fillna(median_val)
            test[col]       = test[col].fillna(median_val)

    features = [f for f in ALL_FEATURES if f in train_part.columns and f in test.columns]
    print(f"Features: {len(features)} total ({len(NEIGHBOR_FEATURES)} neighbor)")

    X_train, y_train = train_part[features], train_part[TARGET]
    X_val,   y_val   = val_part[features],   val_part[TARGET]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=features, reference=dtrain)

    print("Training LightGBM...")
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    print(f"Best iteration: {model.best_iteration}")

    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    metrics  = evaluate(y_val, val_pred)
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  MAE:  {metrics['mae']:.5f}")
    print(f"  R²:   {metrics['r2']:.5f}")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "lgbm_neighbor_metrics.csv", index=False)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_gain": model.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "lgbm_neighbor_feature_importance.csv", index=False)
    print(f"\nTop features:\n{importance_df.head(8).to_string(index=False)}")

    # Final model on all training data
    print("\nAdding neighbor features to full training set...")
    train = add_neighbor_features(
        train, train,
        build_neighbor_map(all_train_coords, all_train_coords)
    )
    for col in NEIGHBOR_FEATURES:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())

    print("Training final LightGBM on all cleaned data...")
    dfull = lgb.Dataset(train[features], label=train[TARGET], feature_name=features)
    final_model = lgb.train(
        LGBM_PARAMS,
        dfull,
        num_boost_round=model.best_iteration,
    )

    joblib.dump({"model": final_model, "features": features},
                MODELS_DIR / "lgbm_neighbor_model.pkl")

    test_features = test[features].copy()
    if test_features.isna().any().any():
        test_features = test_features.fillna(train[features].median())

    test_predictions = final_model.predict(test_features)
    submission_path  = save_submission(test, test_predictions)
    print(f"Submission saved to: {submission_path}")

    # Comparison table
    print("\n--- Comparaison complète ---")
    print(f"  {'Modèle':30} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print(f"  {'-'*60}")
    scores = {
        "baseline_metrics.csv":       "Baseline (HistGB)",
        "neighbor_metrics.csv":       "HistGB + Voisins",
        "lgbm_neighbor_metrics.csv":  "LightGBM + Voisins",
    }
    for fname, label in scores.items():
        p = REPORTS_DIR / fname
        if p.exists():
            row = pd.read_csv(p).iloc[0]
            marker = " ← actuel" if fname == "lgbm_neighbor_metrics.csv" else ""
            print(f"  {label:30} {row['rmse']:10.5f} {row['mae']:10.5f} {row['r2']:8.5f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
