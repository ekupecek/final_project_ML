"""
Grid search over k (number of spatial neighbors) for the neighbor temperature feature.

Queries k_max neighbors once, then subsets for each k to avoid redundant lookups.
Uses HistGradientBoosting with the same config as train_neighbor.py.

Run from the repository root:
    python src/search_k_neighbors.py

Outputs:
    reports/k_search_results.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KDTree

# -----------------------------
# Configuration
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR  = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SENSOR_FRACTION = 0.20
TARGET = "temperature"

K_VALUES  = [1, 3, 5, 8, 10, 15, 20]
K_MAX     = max(K_VALUES)

BASE_FEATURES = [
    "time", "power",
    "coor_x", "coor_y", "coor_z",
    "time_years", "r_xy", "r_xyz", "abs_y",
    "power_x_time", "power_over_r_xy",
]

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
# Neighbor map with k_max neighbors
# -----------------------------
def build_neighbor_map_kmax(
    query_sensors: pd.DataFrame,
    source_sensors: pd.DataFrame,
    k: int,
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


# -----------------------------
# Temperature lookup (shared across all k)
# -----------------------------
def build_temp_lookup(source_train: pd.DataFrame) -> dict:
    temp_by_sensor = {}
    for sensor, grp in source_train.groupby("sensor", observed=True):
        series = grp.set_index("time")["temperature"].sort_index()
        temp_by_sensor[sensor] = series[~series.index.duplicated(keep="first")]
    return temp_by_sensor


def compute_expanded(
    df: pd.DataFrame,
    neighbor_map: pd.DataFrame,
    temp_by_sensor: dict,
) -> pd.DataFrame:
    """Return long-format table: (row_idx, neighbor_sensor, neighbor_dist, neighbor_temp)."""
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
        times      = grp["time"].values
        sorted_t   = series.index.values
        idxs       = np.searchsorted(sorted_t, times)
        idxs       = np.clip(idxs, 0, len(sorted_t) - 1)
        prev_idxs  = np.clip(idxs - 1, 0, len(sorted_t) - 1)
        use_prev   = np.abs(sorted_t[prev_idxs] - times) < np.abs(sorted_t[idxs] - times)
        idxs       = np.where(use_prev, prev_idxs, idxs)
        grp = grp.copy()
        grp["neighbor_temp"] = series.iloc[idxs].values
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


# -----------------------------
# Aggregate for a given k
# -----------------------------
def aggregate_for_k(expanded_full: pd.DataFrame, k: int) -> pd.DataFrame:
    """Subset to the first k neighbors and aggregate."""
    sub = expanded_full[expanded_full["neighbor_rank"] < k].copy()
    sub["weight"]        = 1.0 / (sub["neighbor_dist"] + 1e-6)
    sub["weighted_temp"] = sub["neighbor_temp"] * sub["weight"]

    agg = (
        sub.groupby("row_idx")
        .agg(
            neighbor_temp_mean    = ("neighbor_temp", "mean"),
            neighbor_temp_std     = ("neighbor_temp", "std"),
            neighbor_temp_min     = ("neighbor_temp", "min"),
            neighbor_temp_max     = ("neighbor_temp", "max"),
            weight_sum            = ("weight", "sum"),
            weighted_temp_sum     = ("weighted_temp", "sum"),
            neighbor_dist_mean    = ("neighbor_dist", "mean"),
        )
    )
    agg["neighbor_temp_dist_weighted"] = agg["weighted_temp_sum"] / agg["weight_sum"]
    return agg.drop(columns=["weight_sum", "weighted_temp_sum"])


def attach_features(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    neighbor_cols = [
        "neighbor_temp_mean", "neighbor_temp_std",
        "neighbor_temp_min",  "neighbor_temp_max",
        "neighbor_temp_dist_weighted", "neighbor_dist_mean",
    ]
    result = df.reset_index(drop=True).join(agg)
    for col in neighbor_cols:
        if col in result.columns:
            result[col] = result[col].fillna(result[col].median())
    return result


# -----------------------------
# Evaluate
# -----------------------------
def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    train = train.dropna(subset=BASE_FEATURES + [TARGET]).copy()

    print("Creating sensor-based split...")
    train_part, val_part = sensor_split(train)
    print(f"  Train: {train_part['sensor'].nunique()} sensors | {len(train_part):,} rows")
    print(f"  Val:   {val_part['sensor'].nunique()} sensors | {len(val_part):,} rows")

    # Build neighbor maps with k_max (done once)
    print(f"\nBuilding neighbor maps (k_max={K_MAX})...")
    train_coords = train_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    val_coords   = val_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")

    neighbor_map_train = build_neighbor_map_kmax(train_coords, train_coords, K_MAX)
    neighbor_map_val   = build_neighbor_map_kmax(val_coords,   train_coords, K_MAX)

    # Build temperature lookup (done once)
    print("Building temperature lookup...")
    temp_lookup = build_temp_lookup(train_part)

    # Expand to long format (done once for k_max)
    print("Expanding neighbor rows (done once for all k values)...")
    expanded_train = compute_expanded(train_part, neighbor_map_train, temp_lookup)
    expanded_val   = compute_expanded(val_part,   neighbor_map_val,   temp_lookup)

    neighbor_cols = [
        "neighbor_temp_mean", "neighbor_temp_std",
        "neighbor_temp_min",  "neighbor_temp_max",
        "neighbor_temp_dist_weighted", "neighbor_dist_mean",
    ]
    all_features = BASE_FEATURES + neighbor_cols

    # Grid search over k
    print(f"\n{'k':>4} | {'RMSE':>9} | {'MAE':>9} | {'R²':>8} | {'Temps':>7}")
    print("-" * 50)

    results = []
    for k in K_VALUES:
        t0 = time.time()

        agg_train = aggregate_for_k(expanded_train, k)
        agg_val   = aggregate_for_k(expanded_val,   k)

        train_k = attach_features(train_part, agg_train)
        val_k   = attach_features(val_part,   agg_val)

        features = [f for f in all_features if f in train_k.columns]

        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(train_k[features], train_k[TARGET])

        val_pred = model.predict(val_k[features])
        m = evaluate(val_k[TARGET], val_pred)
        elapsed = time.time() - t0

        print(f"{k:>4} | {m['rmse']:9.5f} | {m['mae']:9.5f} | {m['r2']:8.5f} | {elapsed:6.1f}s")
        results.append({"k": k, **m, "time_s": elapsed})

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "k_search_results.csv", index=False)

    best_k   = results_df.loc[results_df["rmse"].idxmin(), "k"]
    best_rmse = results_df["rmse"].min()
    print(f"\nMeilleur k : {int(best_k)}  (RMSE = {best_rmse:.5f})")
    print(f"Résultats sauvegardés dans reports/k_search_results.csv")


if __name__ == "__main__":
    main()
