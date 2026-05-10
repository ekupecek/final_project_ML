"""
HistGradientBoosting + engineered features for the nuclear waste temperature project.

Improvement tracked independently from baseline and LightGBM.
New features added here (not in cleaning.py) to stay modular.

Run from the repository root:
    python src/train_features.py

Requires:
    python src/cleaning.py

Outputs:
    reports/features_metrics.csv
    reports/features_feature_importance.csv
    models/features_hist_gradient_boosting.joblib
    submissions/features_submission.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

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

# Same base features as baseline
BASE_FEATURES = [
    "time",
    "power",
    "coor_x",
    "coor_y",
    "coor_z",
    "time_years",
    "r_xy",
    "r_xyz",
    "abs_y",
    "power_x_time",
    "power_over_r_xy",
]

# New features added in this script — only physics-meaningful ones
NEW_FEATURES = [
    "log_time",           # log(time) — heat diffusion is logarithmic
    "log_power_r",        # log(t) * power / r_xy — physics-inspired
    "power_over_r_xyz",   # 3D version of power_over_r_xy
    "power_density",      # power / r_xyz^2 — 3D power density
]

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES

# Same HistGB config as baseline (only features change)
MODEL_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    l2_regularization=0.01,
    early_stopping=True,
    validation_fraction=0.10,
    random_state=RANDOM_STATE,
)


# -----------------------------
# Feature engineering
# -----------------------------
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["log_time"]         = np.log1p(df["time_years"])
    df["power_over_r_xyz"] = df["power"] / (df["r_xyz"] + eps)
    df["log_power_r"]      = df["log_time"] * df["power"] / (df["r_xy"] + eps)
    df["power_density"]    = df["power"] / (df["r_xyz"] ** 2 + eps)
    return df


# -----------------------------
# Helpers
# -----------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    return train, test


def sensor_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SENSOR_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(splitter.split(train, groups=train["sensor"]))
    return train.iloc[train_idx].copy(), train.iloc[val_idx].copy()


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }


def get_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURES if c in train.columns and c in test.columns]


def save_submission(test: pd.DataFrame, predictions: np.ndarray) -> Path:
    id_candidates = ["id", "Id", "ID", "sample_id", "row_id"]
    id_col = next((c for c in id_candidates if c in test.columns), None)
    if id_col:
        submission = pd.DataFrame({id_col: test[id_col].to_numpy(), TARGET: predictions})
    else:
        submission = pd.DataFrame({"Id": np.arange(len(test)), TARGET: predictions})
    path = SUBMISSIONS_DIR / "features_submission.csv"
    submission.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading data...")
    train, test = load_data()

    print("Adding new features...")
    train = add_new_features(train)
    test  = add_new_features(test)

    features = get_features(train, test)
    print(f"Total features: {len(features)} ({len(NEW_FEATURES)} new)")

    train = train.dropna(subset=features + [TARGET]).copy()
    test_features = test[features].copy()
    if test_features.isna().any().any():
        medians = train[features].median(numeric_only=True)
        test_features = test_features.fillna(medians)

    print("Creating sensor-based validation split...")
    train_part, val_part = sensor_split(train)
    print(f"  Train: {train_part['sensor'].nunique()} sensors | {len(train_part):,} rows")
    print(f"  Val:   {val_part['sensor'].nunique()} sensors | {len(val_part):,} rows")

    print("Training HistGradientBoosting with new features...")
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(train_part[features], train_part[TARGET])

    print("Evaluating...")
    val_pred = model.predict(val_part[features])
    metrics = evaluate(val_part[TARGET], val_pred)
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  MAE:  {metrics['mae']:.5f}")
    print(f"  R²:   {metrics['r2']:.5f}")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "features_metrics.csv", index=False)

    # Feature importance
    print("Computing feature importance...")
    sample = val_part.sample(n=min(20_000, len(val_part)), random_state=RANDOM_STATE)
    result = permutation_importance(
        model, sample[features], sample[TARGET],
        n_repeats=3, random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "features_feature_importance.csv", index=False)
    print(f"\nTop features:\n{importance_df[['feature','importance_mean']].head(8).to_string(index=False)}")

    # Final model on all data
    print("\nTraining final model on all cleaned data...")
    final_model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(train[features], train[TARGET])

    joblib.dump({"model": final_model, "features": features},
                MODELS_DIR / "features_hist_gradient_boosting.joblib")

    test_predictions = final_model.predict(test_features)
    submission_path = save_submission(test, test_predictions)
    print(f"Submission saved to: {submission_path}")

    # Comparison table
    baseline_path = REPORTS_DIR / "baseline_metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path).iloc[0]
        print("\n--- Comparison vs Baseline ---")
        print(f"  {'':20} {'Baseline':>10} {'+ Features':>10} {'Gain':>10}")
        for m in ["rmse", "mae", "r2"]:
            b, f = baseline[m], metrics[m]
            gain = f - b
            sign = "+" if gain > 0 else "-"
            print(f"  {m:20} {b:10.5f} {f:10.5f} {sign}{abs(gain):.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
