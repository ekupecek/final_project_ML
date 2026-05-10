"""
HistGradientBoosting + neighbor sensor temperature features.

For each sensor at time t, adds the mean/std/distance-weighted temperature
of its k spatially nearest sensors as features.

No data leakage: validation sensor temperatures are NEVER used as neighbor
sources when building training or validation features.

Run from the repository root:
    python src/train_neighbor.py

Requires:
    python src/cleaning.py

Outputs:
    reports/neighbor_metrics.csv
    reports/neighbor_feature_importance.csv
    models/neighbor_hist_gradient_boosting.joblib
    submissions/neighbor_submission.csv
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
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

for folder in [REPORTS_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SENSOR_FRACTION = 0.20
TARGET = "temperature"
K_NEIGHBORS = 1

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
        n_splits=1,
        test_size=VALIDATION_SENSOR_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(splitter.split(train, groups=train["sensor"]))
    return train.iloc[train_idx].copy(), train.iloc[val_idx].copy()


# -----------------------------
# Neighbor feature computation
# -----------------------------
def build_neighbor_map(
    query_sensors: pd.DataFrame,
    source_sensors: pd.DataFrame,
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    """
    For each sensor in query_sensors, find the k nearest sensors
    in source_sensors by 3D coordinate distance.

    Returns a DataFrame with columns:
        sensor, neighbor_sensor, neighbor_dist, neighbor_rank
    """
    coord_cols = ["coor_x", "coor_y", "coor_z"]

    source_unique = (
        source_sensors
        .groupby("sensor")[coord_cols]
        .mean()
    )
    query_unique = (
        query_sensors
        .groupby("sensor")[coord_cols]
        .mean()
    )

    tree = KDTree(source_unique[coord_cols].values)
    # Query k+1 in case the sensor itself appears in the source
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
    """
    For each row in df, look up the temperatures of its k nearest source
    sensors at the closest available time, then compute aggregate features.

    source_train must NOT contain any sensors that are in df's validation/test
    group (no leakage).
    """
    coord_cols = ["coor_x", "coor_y", "coor_z"]

    # Build per-sensor temperature series from source (sorted by time)
    temp_by_sensor: dict[str, pd.Series] = {}
    for sensor, grp in source_train.groupby("sensor", observed=True):
        series = grp.set_index("time")["temperature"].sort_index()
        # Keep only first occurrence per time
        temp_by_sensor[sensor] = series[~series.index.duplicated(keep="first")]

    # Expand: one row per (df_row, neighbor)
    expanded = (
        df[["sensor", "time"]]
        .reset_index(names="row_idx")
        .merge(neighbor_map, on="sensor", how="left")
    )

    # Look up neighbor temperature at the nearest time in source
    def lookup_temp(row) -> float:
        series = temp_by_sensor.get(row["neighbor_sensor"])
        if series is None or series.empty:
            return np.nan
        idx = series.index.searchsorted(row["time"])
        # Clamp to valid range
        if idx >= len(series):
            idx = len(series) - 1
        elif idx > 0:
            # Choose the nearest of idx-1 and idx
            if abs(series.index[idx - 1] - row["time"]) < abs(series.index[idx] - row["time"]):
                idx -= 1
        return series.iloc[idx]

    print("  Looking up neighbor temperatures (this may take a moment)...")

    # Vectorised per-neighbor-sensor to avoid row-by-row apply
    parts = []
    for nbr_sensor, grp in expanded.groupby("neighbor_sensor", observed=True):
        series = temp_by_sensor.get(nbr_sensor)
        if series is None or series.empty:
            grp = grp.copy()
            grp["neighbor_temp"] = np.nan
            parts.append(grp)
            continue

        # Nearest-time lookup using searchsorted
        times = grp["time"].values
        sorted_times = series.index.values
        idxs = np.searchsorted(sorted_times, times)
        idxs = np.clip(idxs, 0, len(sorted_times) - 1)

        # Compare with previous index to find truly nearest
        prev_idxs = np.clip(idxs - 1, 0, len(sorted_times) - 1)
        use_prev = np.abs(sorted_times[prev_idxs] - times) < np.abs(sorted_times[idxs] - times)
        idxs = np.where(use_prev, prev_idxs, idxs)

        grp = grp.copy()
        grp["neighbor_temp"] = series.iloc[idxs].values
        parts.append(grp)

    expanded = pd.concat(parts, ignore_index=True)

    # Distance-weighted temperature (inverse distance)
    expanded["weight"] = 1.0 / (expanded["neighbor_dist"] + 1e-6)
    expanded["weighted_temp"] = expanded["neighbor_temp"] * expanded["weight"]

    agg = (
        expanded
        .groupby("row_idx")
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

    df = df.reset_index(drop=True)
    return df.join(agg)


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
    path = SUBMISSIONS_DIR / "neighbor_submission.csv"
    submission.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    print(f"  Train: {train.shape} | Test: {test.shape}")

    train = train.dropna(subset=BASE_FEATURES + [TARGET]).copy()

    # --- Sensor split (before computing neighbor features to avoid leakage) ---
    print("Creating sensor-based validation split...")
    train_part, val_part = sensor_split(train)
    print(f"  Train: {train_part['sensor'].nunique()} sensors | {len(train_part):,} rows")
    print(f"  Val:   {val_part['sensor'].nunique()} sensors | {len(val_part):,} rows")

    # --- Build neighbor maps ---
    # Source for training and validation features: only train_part sensors
    # Source for test features: all training sensors
    print(f"Building neighbor maps (k={K_NEIGHBORS})...")

    train_sensor_coords = train_part[["sensor", "coor_x", "coor_y", "coor_z"]].drop_duplicates("sensor")
    all_train_coords    = train[["sensor", "coor_x", "coor_y", "coor_z"]].drop_duplicates("sensor")
    test_sensor_coords  = test[["sensor", "coor_x", "coor_y", "coor_z"]].drop_duplicates("sensor")

    # train_part rows → neighbors from train_part sensors (exclude self handled inside)
    neighbor_map_train = build_neighbor_map(train_sensor_coords, train_sensor_coords)
    # val_part rows → neighbors from train_part sensors (no leakage)
    neighbor_map_val   = build_neighbor_map(val_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor"),
                                            train_sensor_coords)
    # test rows → neighbors from all training sensors
    neighbor_map_test  = build_neighbor_map(test_sensor_coords, all_train_coords)

    # --- Add neighbor features ---
    print("Adding neighbor features to train split...")
    train_part = add_neighbor_features(train_part, train_part, neighbor_map_train)

    print("Adding neighbor features to validation split...")
    val_part = add_neighbor_features(val_part, train_part, neighbor_map_val)

    print("Adding neighbor features to test set...")
    test = add_neighbor_features(test, train, neighbor_map_test)

    # --- Feature list ---
    features = [f for f in ALL_FEATURES if f in train_part.columns and f in test.columns]
    missing_new = [f for f in NEIGHBOR_FEATURES if f not in features]
    if missing_new:
        print(f"Warning: missing neighbor features: {missing_new}")

    # Fill remaining NaN in neighbor features with median
    for col in NEIGHBOR_FEATURES:
        if col in train_part.columns:
            median_val = train_part[col].median()
            train_part[col] = train_part[col].fillna(median_val)
            val_part[col]   = val_part[col].fillna(median_val)
            test[col]       = test[col].fillna(median_val)

    X_train = train_part[features]
    y_train = train_part[TARGET]
    X_val   = val_part[features]
    y_val   = val_part[TARGET]

    print(f"\nTraining HistGradientBoosting with {len(features)} features "
          f"({len(NEIGHBOR_FEATURES)} neighbor)...")
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    val_pred = model.predict(X_val)
    metrics  = evaluate(y_val, val_pred)
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  MAE:  {metrics['mae']:.5f}")
    print(f"  R²:   {metrics['r2']:.5f}")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "neighbor_metrics.csv", index=False)

    # --- Final model on all training data ---
    print("\nAdding neighbor features to full training set...")
    train = add_neighbor_features(train, train, build_neighbor_map(all_train_coords, all_train_coords))
    for col in NEIGHBOR_FEATURES:
        if col in train.columns:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)

    print("Training final model on all cleaned data...")
    final_model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(train[features], train[TARGET])

    joblib.dump({"model": final_model, "features": features},
                MODELS_DIR / "neighbor_hist_gradient_boosting.joblib")

    test_features = test[features].copy()
    if test_features.isna().any().any():
        medians = train[features].median(numeric_only=True)
        test_features = test_features.fillna(medians)

    test_predictions = final_model.predict(test_features)
    submission_path = save_submission(test, test_predictions)
    print(f"Submission saved to: {submission_path}")

    # --- Comparison ---
    baseline_path = REPORTS_DIR / "baseline_metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path).iloc[0]
        print("\n--- Comparison vs Baseline ---")
        print(f"  {'':22} {'Baseline':>10} {'+ Voisins':>10} {'Gain':>10}")
        for m in ["rmse", "mae", "r2"]:
            b, n = baseline[m], metrics[m]
            gain = n - b
            arrow = "↓ mieux" if (m != "r2" and gain < 0) or (m == "r2" and gain > 0) else "↑ pire"
            print(f"  {m:22} {b:10.5f} {n:10.5f}   {arrow}")

    print("\nDone.")


if __name__ == "__main__":
    main()
