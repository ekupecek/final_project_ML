"""

Meilleur modèle : IDW (tous les capteurs) + features hybrides k=1+5 + HistGB.

Architecture :
  - idw_temp  : Inverse Distance Weighting sur TOUS les capteurs source
                T_idw = Σ T_i/d² / Σ 1/d²  (i = tous capteurs source)
  - nn_temp / nn_dist       : capteur le plus proche (k=1)
  - broad_temp_mean / std   : moyenne/std des 5 voisins (k=5)
  + 11 features de base (géométrie, temps, puissance)

Différence vs features hybrides seules : l'IDW exploite les 222 capteurs
(au lieu de 5) avec une pondération continue → signal global plus riche.

Run depuis la racine :
    python src/train_best.py

Requires :
    python src/cleaning.py

Outputs :
    reports/best_metrics.csv
    reports/best_feature_importance.csv
    models/best_model.joblib
    submissions/best_submission.csv
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
IDW_POWER    = 2   # poids = 1 / d^IDW_POWER

BASE_FEATURES = [
    "time", "power",
    "coor_x", "coor_y", "coor_z",
    "time_years", "r_xy", "r_xyz", "abs_y",
    "power_x_time", "power_over_r_xy",
]

IDW_FEATURES     = ["idw_temp"]
SPATIAL_FEATURES = ["nn_temp", "nn_dist", "broad_temp_mean", "broad_temp_std"]
ALL_FEATURES     = BASE_FEATURES + IDW_FEATURES + SPATIAL_FEATURES

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
# Utilitaires — split
# ----------------------------
def sensor_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split par capteur entier pour éviter toute fuite."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_STATE
    )
    idx_tr, idx_val = next(splitter.split(df, groups=df["sensor"]))
    return df.iloc[idx_tr].copy(), df.iloc[idx_val].copy()


# ----------------------------
# Utilitaires — features hybrides k=1+5
# ----------------------------
def build_neighbor_map(
    query_sensors: pd.DataFrame,
    source_sensors: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """KD-tree : k voisins 3D les plus proches, self exclu."""
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
    """Index temporel par capteur pour lookup O(log n)."""
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
    """Lookup de la température de chaque voisin au temps le plus proche."""
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
        idxs     = np.searchsorted(sorted_t, times)
        idxs     = np.clip(idxs, 0, len(sorted_t) - 1)
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
    """nn (k=1) + broad mean/std (k=5)."""
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
# Utilitaires — IDW
# ----------------------------
def compute_idw_feature(
    df: pd.DataFrame,
    query_coords: pd.DataFrame,
    source_coords: pd.DataFrame,
    source_data: pd.DataFrame,
    power: int = IDW_POWER,
) -> pd.DataFrame:
    """
    Ajoute idw_temp : interpolation par distance inverse sur tous les capteurs source.

    T_idw(j,t) = Σ_i [T_i(t) / d(j,i)^p] / Σ_i [1 / d(j,i)^p]

    Traitement par pas de temps unique → mémoire O(n_source × n_query_sensors).
    Self-contribution exclue pour les capteurs qui sont à la fois query et source.
    """
    coord_cols = ["coor_x", "coor_y", "coor_z"]
    query_u  = query_coords.groupby("sensor")[coord_cols].mean()
    source_u = source_coords.groupby("sensor")[coord_cols].mean()

    q_list = query_u.index.tolist()
    s_list = source_u.index.tolist()

    # Matrice de distances (n_query_sensors × n_source_sensors) — constante dans le temps
    dist_mat = euclidean_distances(
        query_u[coord_cols].values,
        source_u[coord_cols].values,
    )

    # Mettre la distance self à +inf pour exclure la contribution propre
    s_idx_map = {s: j for j, s in enumerate(s_list)}
    for i, q in enumerate(q_list):
        if q in s_idx_map:
            dist_mat[i, s_idx_map[q]] = np.inf

    weights_mat = 1.0 / (dist_mat ** power + 1e-9)  # (n_q, n_s)

    q_idx_map = {s: i for i, s in enumerate(q_list)}

    # Pivot des températures source : (n_temps_uniques × n_source)
    source_pivot = (
        source_data.groupby(["time", "sensor"])["temperature"]
        .mean()
        .unstack("sensor")
        .reindex(columns=s_list)
    )
    pivot_times = source_pivot.index.values
    pivot_vals  = source_pivot.values  # (n_times, n_source)

    df = df.reset_index(drop=True)
    df_times    = df["time"].values
    df_q_idx    = df["sensor"].map(q_idx_map).values

    # Mapping temps → index dans pivot (recherche binaire)
    t_idxs = np.searchsorted(pivot_times, df_times)
    t_idxs = np.clip(t_idxs, 0, len(pivot_times) - 1)
    t_prev = np.clip(t_idxs - 1, 0, len(pivot_times) - 1)
    use_prev = np.abs(pivot_times[t_prev] - df_times) < np.abs(pivot_times[t_idxs] - df_times)
    t_idxs = np.where(use_prev, t_prev, t_idxs)

    # Calcul IDW par pas de temps unique (mémoire constante)
    idw_values = np.full(len(df), np.nan, dtype=np.float64)
    for t_uniq in np.unique(t_idxs):
        mask      = t_idxs == t_uniq
        src_temps = pivot_vals[t_uniq, :]          # (n_source,)
        valid     = ~np.isnan(src_temps)
        if not valid.any():
            continue
        row_q_idx = df_q_idx[mask]                 # indices query sensor
        row_w     = weights_mat[row_q_idx, :]      # (n_rows, n_source)
        w_valid   = row_w * valid                  # zero poids si temp manquante
        denom     = w_valid.sum(axis=1)
        numer     = (w_valid * np.where(valid, src_temps, 0.0)).sum(axis=1)
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
    path = SUBMISSIONS_DIR / "best_submission.csv"
    sub.to_csv(path, index=False)
    return path


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("=== IDW + hybrid k=1+5 + HistGB ===\n")

    print("Chargement des données...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    train = train.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    print(f"  Train : {train.shape} | Test : {test.shape}")

    print("Split par capteurs...")
    train_part, val_part = sensor_split(train)
    print(f"  Train : {train_part['sensor'].nunique()} capteurs | {len(train_part):,} lignes")
    print(f"  Val   : {val_part['sensor'].nunique()} capteurs  | {len(val_part):,} lignes")

    # Coordonnées uniques
    train_coords     = train_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    all_train_coords = train[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    val_coords       = val_part[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")
    test_coords      = test[["sensor","coor_x","coor_y","coor_z"]].drop_duplicates("sensor")

    # ---- Features hybrides k=1+5 ----
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

    # ---- Feature IDW ----
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

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "best_metrics.csv", index=False)

    # Importance (permutation)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        model, val_part[features], val_part[TARGET],
        n_repeats=5, random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    imp_df = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
    }).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(REPORTS_DIR / "best_feature_importance.csv", index=False)
    print(f"\nTop features :\n{imp_df.head(8).to_string(index=False)}")

    # Tableau comparatif
    print("\n--- Comparaison ---")
    print(f"  {'Modèle':45} {'RMSE':>9} {'MAE':>9}")
    print(f"  {'-'*65}")
    for fname, label in [
        ("baseline_metrics.csv",        "Baseline HistGB"),
        ("neighbor_hybrid_metrics.csv", "HistGB + hybride k=1+5"),
        ("lgbm_hybrid_metrics.csv",     "LightGBM + hybride k=1+5"),
        ("best_metrics.csv",            "IDW + hybride k=1+5 (actuel) ←"),
    ]:
        p = REPORTS_DIR / fname
        if p.exists():
            r = pd.read_csv(p).iloc[0]
            print(f"  {label:45} {r['rmse']:9.5f} {r['mae']:9.5f}")

    # ---- Modèle final sur toutes les données ----
    print("\nEntraînement final sur toutes les données nettoyées...")
    nm_full = build_neighbor_map(all_train_coords, all_train_coords, K_BROAD)
    train   = compute_hybrid_features(train, expand_neighbors(train, nm_full, lookup_all))
    for col in SPATIAL_FEATURES:
        train[col] = train[col].fillna(train[col].median())

    print("  IDW final...")
    train = compute_idw_feature(train, all_train_coords, all_train_coords, train)
    train["idw_temp"] = train["idw_temp"].fillna(train["idw_temp"].median())
    test["idw_temp"]  = test["idw_temp"].fillna(train["idw_temp"].median())

    final_model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(train[features], train[TARGET])

    joblib.dump({"model": final_model, "features": features},
                MODELS_DIR / "best_model.joblib")

    test_f = test[features].copy()
    if test_f.isna().any().any():
        test_f = test_f.fillna(train[features].median())

    test_pred = final_model.predict(test_f)
    path = save_submission(test, test_pred)
    print(f"\nSoumission sauvegardée : {path}")
    print("Terminé.")


if __name__ == "__main__":
    main()
