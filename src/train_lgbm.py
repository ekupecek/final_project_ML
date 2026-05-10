"""
LightGBM model for the nuclear waste temperature project.

Run from the repository root:
    python src/train_lgbm.py

Requires that cleaning has already been run:
    python src/cleaning.py

Input:
    data/processed/train_cleaned.parquet
    data/processed/test_with_features.parquet

Outputs:
    reports/lgbm_metrics.csv
    reports/lgbm_feature_importance.csv
    models/lgbm_model.pkl
    submissions/lgbm_submission.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

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

FEATURE_COLUMNS = [
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

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 7,
    "min_child_samples": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


# -----------------------------
# Helpers
# -----------------------------
def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_with_features.parquet")
    return train, test


def get_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in train.columns and c in test.columns]


def sensor_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SENSOR_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(splitter.split(train, groups=train["sensor"]))
    return train.iloc[train_idx].copy(), train.iloc[val_idx].copy()


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_submission(test: pd.DataFrame, predictions: np.ndarray) -> Path:
    id_candidates = ["id", "Id", "ID", "sample_id", "row_id"]
    id_col = next((c for c in id_candidates if c in test.columns), None)
    if id_col:
        submission = pd.DataFrame({id_col: test[id_col].to_numpy(), TARGET: predictions})
    else:
        submission = pd.DataFrame({"Id": np.arange(len(test)), TARGET: predictions})
    path = SUBMISSIONS_DIR / "lgbm_submission.csv"
    submission.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading processed data...")
    train, test = load_processed_data()
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")

    features = get_features(train, test)
    train = train.dropna(subset=features + [TARGET]).copy()

    test_features = test[features].copy()
    if test_features.isna().any().any():
        medians = train[features].median(numeric_only=True)
        test_features = test_features.fillna(medians)

    print("Creating sensor-based validation split...")
    train_part, val_part = sensor_split(train)
    print(f"  Train sensors: {train_part['sensor'].nunique()} | rows: {len(train_part):,}")
    print(f"  Val   sensors: {val_part['sensor'].nunique()} | rows: {len(val_part):,}")

    X_train = train_part[features]
    y_train = train_part[TARGET]
    X_val = val_part[features]
    y_val = val_part[TARGET]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain)

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

    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = evaluate(y_val, val_pred)
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  MAE:  {metrics['mae']:.5f}")
    print(f"  R²:   {metrics['r2']:.5f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(REPORTS_DIR / "lgbm_metrics.csv", index=False)

    importance_df = pd.DataFrame({
        "feature": features,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "lgbm_feature_importance.csv", index=False)
    print(f"\nTop features:\n{importance_df[['feature','importance_gain']].head(6).to_string(index=False)}")

    # Final model on all cleaned data
    print("\nTraining final LightGBM on all cleaned data...")
    dfull = lgb.Dataset(train[features], label=train[TARGET], feature_name=features)
    final_model = lgb.train(
        LGBM_PARAMS,
        dfull,
        num_boost_round=model.best_iteration,
    )

    model_path = MODELS_DIR / "lgbm_model.pkl"
    joblib.dump({"model": final_model, "features": features}, model_path)
    print(f"Model saved to: {model_path}")

    test_predictions = final_model.predict(test_features)
    submission_path = save_submission(test, test_predictions)
    print(f"Submission saved to: {submission_path}")

    # Compare with baseline
    baseline_metrics_path = REPORTS_DIR / "baseline_metrics.csv"
    if baseline_metrics_path.exists():
        baseline = pd.read_csv(baseline_metrics_path).iloc[0]
        print("\n--- Comparison ---")
        print(f"{'':20} {'Baseline':>10} {'LightGBM':>10} {'Gain':>10}")
        for m in ["rmse", "mae", "r2"]:
            b = baseline[m]
            l = metrics[m]
            gain = l - b
            sign = "-" if gain < 0 else "+"
            print(f"  {m:18} {b:10.5f} {l:10.5f} {sign}{abs(gain):.5f}")

    print("\nLightGBM pipeline complete.")


if __name__ == "__main__":
    main()
