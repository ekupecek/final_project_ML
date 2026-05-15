"""
v12 — IDW + hybrid k=1+5 + gradient temporel (dT/dt) + HistGB.

Hypothèse : dans la zone bas-gauche du scatter plot (t ≈ 0, T ≈ 12°C),
les features spatiales sont inutiles car tous les capteurs sont à la même
température. La vitesse de montée en température (dT/dt) du capteur voisin
le plus proche discrimine les positions même quand les valeurs absolues
sont encore similaires.

Feature ajoutée :
  nn_dtdt : taux de variation de température du capteur le plus proche,
            estimé par régression linéaire sur une fenêtre glissante de
            GRAD_WINDOW pas de temps → pente en °C/s.

Run depuis la racine :
    python src/train_best_v12.py

Requires :
    python src/cleaning.py

Outputs :
    reports/v12_metrics.csv
    reports/v12_feature_importance.csv
    submissions/v12_submission.csv
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
from sklearn.metrics.pairwise import euclidean_distances

# ----------------------------
# Configuration
# ----------------------------
REPO_ROOT       = Path(__file__).resolve().parents[1]
PROCESSED_DIR   = REPO_ROOT / "data" / "processed"
REPORTS_DIR     = REPO_ROOT / "reports"
MODELS_DIR      = REPO_ROOT / "models"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

for folder in [REPORTS_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VAL_FRACTION = 0.20
TARGET       = "temperature"
K_BROAD      = 5
IDW_POWER    = 2
GRAD_WINDOW  = 10   # nombre de pas de temps pour estimer dT/dt

BASE_FEATURES = [
    "time", "power",
    "coor_x", "coor_y", "coor_z",
    "time_years", "r_xy", "r_xyz", "abs_y",
    "power_x_time", "power_over_r_xy",
]
IDW_FEATURES     = ["idw_temp"]
SPATIAL_FEATURES = ["nn_temp", "nn_dist", "broad_temp_mean", "broad_temp_std"]
GRAD_FEATURES    = ["nn_dtdt"]
ALL_FEATURES     = BASE_FEATURES + IDW_FEATURES + SPATIAL_FEATURES + GRAD_FEATURES

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


# ----------------------------
# Split
# ----------------------------
def sensor_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_STATE
    )
    idx_tr, idx_val = next(splitter.split(df, groups=df["sensor"]))
    return df.iloc[idx_tr].copy(), df.iloc[idx_val].copy()


# ----------------------------
# KNN hybrid features
# ----------------------------
def build_neighbor_map(
    query_sensors: pd.DataFrame,
    source_sensors: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    coord_cols = ["coor_x", "coor_y", "coor_z"]
    source_u   = source_sensors.groupby("sensor")[coord_cols].mean()
    query_u    = query_sensors.groupby("sensor")[coord_cols].mean()
    tree = KDTree(source_u[coord_cols].values)
    dists, idxs = tree.query(query_u[coord_cols].values, k=k + 1)
    src_index = source_u.index.tolist()
    rows = []
    for i, sensor in enumerate(query_u.index):
        rank = 0
        for dist, idx in zip(dists[i], idxs[i]):
            neighbor = src_index[idx]
            if neighbor == sensor:
                continue
            rows.append({
                "sensor": sensor, "neighbor_sensor": neighbor,
                "neighbor_dist": dist, "neighbor_rank": rank,
            })
            rank += 1
            if rank >= k:
                break
    return pd.DataFrame(rows)


def build_temp_lookup(source: pd.DataFrame) -> dict:
    return {
        sensor: (
            grp.set_index("time")["temperature"]
            .sort_index()
            .pipe(lambda s: s[~s.index.duplicated(keep="first")])
        )
        for sensor, grp in source.groupby("sensor", observed=True)
    }


def expand_neighbors(
    df: pd.DataFrame,
    neighbor_map: pd.DataFrame,
    temp_lookup: dict,
) -> pd.DataFrame:
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
        times    = grp["time"].values
        sorted_t = series.index.values
        idxs     = np.clip(np.searchsorted(sorted_t, times), 0, len(sorted_t) - 1)
        prev     = np.clip(idxs - 1, 0, len(sorted_t) - 1)
        use_prev = np.abs(sorted_t[prev] - times) < np.abs(sorted_t[idxs] - times)
        grp = grp.copy()
        grp["neighbor_temp"] = series.iloc[np.where(use_prev, prev, idxs)].values
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def compute_hybrid_features(
    df: pd.DataFrame,
    expanded: pd.DataFrame,
) -> pd.DataFrame:
    near = (
        expanded[expanded["neighbor_rank"] == 0]
        [["row_idx", "neighbor_temp", "neighbor_dist"]]
        .rename(columns={"neighbor_temp": "nn_temp", "neighbor_dist": "nn_dist"})
        .set_index("row_idx")
    )
    broad = (
        expanded[expanded["neighbor_rank"] < K_BROAD]
        .groupby("row_idx")
        .agg(
            broad_temp_mean=("neighbor_temp", "mean"),
            broad_temp_std =("neighbor_temp", "std"),
        )
    )
    return df.reset_index(drop=True).join(near).join(broad)


# ----------------------------
# Gradient temporel dT/dt
# ----------------------------
def build_dtdt_lookup(source: pd.DataFrame, window: int = GRAD_WINDOW) -> dict:
    """
    Pour chaque capteur source, estime dT/dt à chaque pas de temps par
    régression linéaire sur une fenêtre glissante de `window` points.
    Retourne un dict sensor -> Series(time -> dT/dt en °C/s).
    """
    lookup = {}
    for sensor, grp in source.groupby("sensor", observed=True):
        series = (
            grp.set_index("time")["temperature"]
            .sort_index()
            .pipe(lambda s: s[~s.index.duplicated(keep="first")])
        )
        times = series.index.values.astype(np.float64)
        temps = series.values.astype(np.float64)
        n = len(times)
        dtdt = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - window // 2)
            hi = min(n, i + window // 2 + 1)
            if hi - lo < 2:
                continue
            t_w = times[lo:hi]
            T_w = temps[lo:hi]
            # pente par moindres carrés
            t_c = t_w - t_w.mean()
            denom = (t_c ** 2).sum()
            if denom == 0:
                continue
            dtdt[i] = (t_c * T_w).sum() / denom
        lookup[sensor] = pd.Series(dtdt, index=series.index)
    return lookup


def compute_dtdt_feature(
    df: pd.DataFrame,
    neighbor_map: pd.DataFrame,
    dtdt_lookup: dict,
) -> pd.DataFrame:
    """
    Ajoute nn_dtdt : dT/dt du capteur le plus proche (rank=0)
    au pas de temps le plus proche.
    """
    nn_map = neighbor_map[neighbor_map["neighbor_rank"] == 0][
        ["sensor", "neighbor_sensor"]
    ]
    expanded = (
        df[["sensor", "time"]]
        .reset_index(names="row_idx")
        .merge(nn_map, on="sensor", how="left")
    )
    parts = []
    for nbr, grp in expanded.groupby("neighbor_sensor", observed=True):
        series = dtdt_lookup.get(nbr)
        if series is None or series.empty:
            grp = grp.copy()
            grp["nn_dtdt"] = np.nan
            parts.append(grp)
            continue
        times    = grp["time"].values
        sorted_t = series.index.values
        idxs     = np.clip(np.searchsorted(sorted_t, times), 0, len(sorted_t) - 1)
        prev     = np.clip(idxs - 1, 0, len(sorted_t) - 1)
        use_prev = np.abs(sorted_t[prev] - times) < np.abs(sorted_t[idxs] - times)
        grp = grp.copy()
        grp["nn_dtdt"] = series.iloc[np.where(use_prev, prev, idxs)].values
        parts.append(grp)

    dtdt_df = pd.concat(parts, ignore_index=True).set_index("row_idx")[["nn_dtdt"]]
    return df.reset_index(drop=True).join(dtdt_df)


# ----------------------------
# IDW feature
# ----------------------------
def compute_idw_feature(
    df: pd.DataFrame,
    query_coords: pd.DataFrame,
    source_coords: pd.DataFrame,
    source_data: pd.DataFrame,
    power: int = IDW_POWER,
) -> pd.DataFrame:
    coord_cols = ["coor_x", "coor_y", "coor_z"]
    query_u  = query_coords.groupby("sensor")[coord_cols].mean()
    source_u = source_coords.groupby("sensor")[coord_cols].mean()
    q_list = query_u.index.tolist()
    s_list = source_u.index.tolist()
    dist_mat = euclidean_distances(query_u[coord_cols].values, source_u[coord_cols].values)
    s_idx_map = {s: j for j, s in enumerate(s_list)}
    for i, q in enumerate(q_list):
        if q in s_idx_map:
            dist_mat[i, s_idx_map[q]] = np.inf
    weights_mat = 1.0 / (dist_mat ** power + 1e-9)
    q_idx_map = {s: i for i, s in enumerate(q_list)}
    source_pivot = (
        source_data.groupby(["time", "sensor"])["temperature"]
        .mean().unstack("sensor").reindex(columns=s_list)
    )
    pivot_times = source_pivot.index.values
    pivot_vals  = source_pivot.values
    df = df.reset_index(drop=True)
    df_times  = df["time"].values
    df_q_idx  = df["sensor"].map(q_idx_map).values
    t_idxs = np.clip(np.searchsorted(pivot_times, df_times), 0, len(pivot_times) - 1)
    t_prev = np.clip(t_idxs - 1, 0, len(pivot_times) - 1)
    use_prev = np.abs(pivot_times[t_prev] - df_times) < np.abs(pivot_times[t_idxs] - df_times)
    t_idxs = np.where(use_prev, t_prev, t_idxs)
    idw_values = np.full(len(df), np.nan, dtype=np.float64)
    for t_uniq in np.unique(t_idxs):
        mask      = t_idxs == t_uniq
        src_temps = pivot_vals[t_uniq, :]
        valid     = ~np.isnan(src_temps)
        if not valid.any():
            continue
        row_w   = weights_mat[df_q_idx[mask], :]
        w_valid = row_w * valid
        denom   = w_valid.sum(axis=1)
        numer   = (w_valid * np.where(valid, src_temps, 0.0)).sum(axis=1)
        idw_values[mask] = np.where(denom > 0, numer / denom, np.nan)
    df = df.copy()
    df["idw_temp"] = idw_values
    return df


# ----------------------------
# Évaluation & soumission
# ----------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def save_submission(test: pd.DataFrame, predictions: np.ndarray) -> Path:
    id_col = next((c for c in ["id", "Id", "ID"] if c in test.columns), None)
    sub = pd.DataFrame({
        (id_col if id_col else "Id"): (
            test[id_col].to_numpy() if id_col else np.arange(len(test))
        ),
        TARGET: predictions,
    })
    path = SUBMISSIONS_DIR / "v12_submission.csv"
    sub.to_csv(path, index=False)
    return path


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("=== v12 : IDW + hybrid k=1+5 + dT/dt (nn_dtdt) + HistGB ===\n")

    print("Chargement des données...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    train = train.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    print(f"  Train : {train.shape} | Test : {test.shape}")

    print("Split par capteurs...")
    train_part, val_part = sensor_split(train)
    print(f"  Train : {train_part['sensor'].nunique()} capteurs | {len(train_part):,} lignes")
    print(f"  Val   : {val_part['sensor'].nunique()} capteurs  | {len(val_part):,} lignes")

    train_coords     = train_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    all_train_coords = train[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    val_coords       = val_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    test_coords      = test[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")

    # ---- KNN hybrid ----
    print(f"\nFeatures hybrides (k={K_BROAD})...")
    nm_train = build_neighbor_map(train_coords, train_coords, K_BROAD)
    nm_val   = build_neighbor_map(val_coords,   train_coords, K_BROAD)
    nm_test  = build_neighbor_map(test_coords,  all_train_coords, K_BROAD)

    lookup_train = build_temp_lookup(train_part)
    lookup_all   = build_temp_lookup(train)

    train_part = compute_hybrid_features(train_part, expand_neighbors(train_part, nm_train, lookup_train))
    val_part   = compute_hybrid_features(val_part,   expand_neighbors(val_part,   nm_val,   lookup_train))
    test       = compute_hybrid_features(test,        expand_neighbors(test,       nm_test,  lookup_all))

    for col in SPATIAL_FEATURES:
        med = train_part[col].median()
        train_part[col] = train_part[col].fillna(med)
        val_part[col]   = val_part[col].fillna(med)
        test[col]       = test[col].fillna(med)

    # ---- Gradient temporel dT/dt ----
    print(f"Feature gradient temporel (window={GRAD_WINDOW})...")
    dtdt_train = build_dtdt_lookup(train_part)
    dtdt_all   = build_dtdt_lookup(train)

    train_part = compute_dtdt_feature(train_part, nm_train, dtdt_train)
    val_part   = compute_dtdt_feature(val_part,   nm_val,   dtdt_train)
    test       = compute_dtdt_feature(test,        nm_test,  dtdt_all)

    dtdt_med = train_part["nn_dtdt"].median()
    train_part["nn_dtdt"] = train_part["nn_dtdt"].fillna(dtdt_med)
    val_part["nn_dtdt"]   = val_part["nn_dtdt"].fillna(dtdt_med)
    test["nn_dtdt"]       = test["nn_dtdt"].fillna(dtdt_med)

    # ---- IDW ----
    print("Feature IDW (tous capteurs source)...")
    print("  Train split...")
    train_part = compute_idw_feature(train_part, train_coords, train_coords, train_part)
    print("  Validation split...")
    val_part   = compute_idw_feature(val_part,   val_coords,   train_coords, train_part)
    print("  Test...")
    test       = compute_idw_feature(test,        test_coords,  all_train_coords, train)

    idw_med = train_part["idw_temp"].median()
    train_part["idw_temp"] = train_part["idw_temp"].fillna(idw_med)
    val_part["idw_temp"]   = val_part["idw_temp"].fillna(idw_med)
    test["idw_temp"]       = test["idw_temp"].fillna(idw_med)

    features = [f for f in ALL_FEATURES if f in train_part.columns and f in test.columns]
    print(f"\nFeatures totales ({len(features)}) : {features}")

    # ---- Entraînement ----
    print("\nEntraînement HistGradientBoosting...")
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(train_part[features], train_part[TARGET])

    val_pred = model.predict(val_part[features])
    metrics  = evaluate(val_part[TARGET].values, val_pred)
    print(f"\nValidation ({val_part['sensor'].nunique()} capteurs non vus) :")
    print(f"  RMSE : {metrics['rmse']:.5f}")
    print(f"  MAE  : {metrics['mae']:.5f}")
    print(f"  R²   : {metrics['r2']:.5f}")

    # Comparaison avec v1
    best_path = REPORTS_DIR / "best_metrics.csv"
    if best_path.exists():
        best = pd.read_csv(best_path).iloc[0]
        delta = metrics["rmse"] - best["rmse"]
        sign = "+" if delta > 0 else ""
        print(f"\n  vs v1 (best) : {sign}{delta:.5f}  ({'pire' if delta > 0 else 'meilleur'})")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "v12_metrics.csv", index=False)

    # ---- Soumission finale ----
    print("\nEntraînement final sur toutes les données nettoyées...")
    nm_full = build_neighbor_map(all_train_coords, all_train_coords, K_BROAD)
    train   = compute_hybrid_features(train, expand_neighbors(train, nm_full, lookup_all))
    for col in SPATIAL_FEATURES:
        train[col] = train[col].fillna(train[col].median())

    dtdt_all_full = build_dtdt_lookup(train)
    nm_full_self  = build_neighbor_map(all_train_coords, all_train_coords, K_BROAD)
    train = compute_dtdt_feature(train, nm_full_self, dtdt_all_full)
    train["nn_dtdt"] = train["nn_dtdt"].fillna(train["nn_dtdt"].median())
    test["nn_dtdt"]  = test["nn_dtdt"].fillna(train["nn_dtdt"].median())

    print("  IDW final...")
    train = compute_idw_feature(train, all_train_coords, all_train_coords, train)
    train["idw_temp"] = train["idw_temp"].fillna(train["idw_temp"].median())
    test["idw_temp"]  = test["idw_temp"].fillna(train["idw_temp"].median())

    final_model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(train[features], train[TARGET])

    test_f = test[features].copy()
    if test_f.isna().any().any():
        test_f = test_f.fillna(train[features].median())

    test_pred = final_model.predict(test_f)
    path = save_submission(test, test_pred)
    print(f"\nSoumission sauvegardée : {path}")
    print("Terminé.")


if __name__ == "__main__":
    main()
