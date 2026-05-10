"""
Data cleaning and outlier handling for the nuclear waste temperature project.

Run from the repository root:
    python src/cleaning.py

Expected raw files:
    train.parquet
    test.parquet
    sensors.parquet

The script automatically searches common raw-data folders.
Outputs are written to:
    data/processed/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR_CANDIDATES = [
    REPO_ROOT / "data_parquet_2026" / "data_parquet_2026",
    REPO_ROOT / "data_parquet_2026",
    REPO_ROOT / "data",
    REPO_ROOT,
]

OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Broad physical limits. These are intentionally conservative.
TEMP_MIN = 0.0
TEMP_MAX = 200.0

# Per-sensor robust outlier limits.
IQR_MULTIPLIER = 4.0
ROLLING_WINDOW = 21
ROLLING_RESIDUAL_Z_LIMIT = 8.0

# Failed/drift sensor flags.
MAX_BAD_FRACTION_FOR_FAILED_SENSOR = 0.20
DRIFT_SLOPE_Z_LIMIT = 4.0


# -----------------------------
# Loading helpers
# -----------------------------
def find_raw_dir() -> Path:
    """Find the folder containing train/test/sensors parquet files."""
    required = {"train.parquet", "test.parquet", "sensors.parquet"}
    for folder in RAW_DIR_CANDIDATES:
        if folder.exists():
            files = {p.name for p in folder.iterdir() if p.is_file()}
            if required.issubset(files):
                return folder
    raise FileNotFoundError(
        "Could not find train.parquet, test.parquet and sensors.parquet. "
        "Put them in data_parquet_2026/data_parquet_2026, data_parquet_2026, data, or the repo root."
    )


def load_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(raw_dir / "train.parquet")
    test = pd.read_parquet(raw_dir / "test.parquet")
    sensors = pd.read_parquet(raw_dir / "sensors.parquet")
    return train, test, sensors


# -----------------------------
# Feature helpers
# -----------------------------
def add_sensor_coordinates(df, sensors):
    """
    Merge sensor coordinates into train/test.

    Some sensors appear more than once in sensors.parquet.
    We average their coordinates so each sensor has one coordinate row.
    """
    sensors_unique = (
        sensors
        .groupby("sensor", as_index=False)
        .mean(numeric_only=True)
    )

    return df.merge(sensors_unique, on="sensor", how="left")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple geometry/time features useful later for validation and modeling."""
    df = df.copy()
    df["time_years"] = df["time"] / SECONDS_PER_YEAR
    df["r_xy"] = np.sqrt(df["coor_x"] ** 2 + df["coor_y"] ** 2)
    df["r_xyz"] = np.sqrt(df["coor_x"] ** 2 + df["coor_y"] ** 2 + df["coor_z"] ** 2)
    df["abs_y"] = df["coor_y"].abs()
    df["power_x_time"] = df["power"] * df["time_years"]
    df["power_over_r_xy"] = df["power"] / (df["r_xy"] + 1e-6)
    df["log_time"] = np.log1p(df["time_years"])
    df["sqrt_time"] = np.sqrt(df["time_years"])
    df["inv_r_xy"] = 1 / (df["r_xy"] + 1e-3)

    df["x2"] = df["coor_x"] ** 2
    df["y2"] = df["coor_y"] ** 2
    df["z2"] = df["coor_z"] ** 2
    df["xy"] = df["coor_x"] * df["coor_y"]
    df["xz"] = df["coor_x"] * df["coor_z"]
    df["yz"] = df["coor_y"] * df["coor_z"]

    df["power_log_time"] = df["power"] * df["log_time"]
    df["power_sqrt_time"] = df["power"] * df["sqrt_time"]
    df["power_inv_r_xy"] = df["power"] * df["inv_r_xy"]
    df["time_over_r_xy"] = df["time_years"] / (df["r_xy"] + 1e-3)
    return df 


# -----------------------------
# Cleaning and outlier logic
# -----------------------------
def create_cleaning_flags(train: pd.DataFrame) -> pd.DataFrame:
    """Create row-level cleaning flags without dropping yet."""
    df = train.copy()

    df["flag_missing_temperature"] = df["temperature"].isna()
    df["flag_bad_coordinates"] = df[["coor_x", "coor_y", "coor_z"]].isna().any(axis=1)
    df["flag_physical_outlier"] = (
        df["temperature"].notna()
        & ((df["temperature"] < TEMP_MIN) | (df["temperature"] > TEMP_MAX))
    )

    # Per-sensor IQR outliers.
    valid_temp = df["temperature"].notna() & ~df["flag_physical_outlier"]
    q1 = df.loc[valid_temp].groupby("sensor", observed=True)["temperature"].quantile(0.25)
    q3 = df.loc[valid_temp].groupby("sensor", observed=True)["temperature"].quantile(0.75)
    iqr = q3 - q1

    bounds = pd.DataFrame({
        "sensor": q1.index,
        "sensor_q1": q1.values,
        "sensor_q3": q3.values,
        "sensor_iqr": iqr.values,
    })
    bounds["sensor_lower"] = bounds["sensor_q1"] - IQR_MULTIPLIER * bounds["sensor_iqr"]
    bounds["sensor_upper"] = bounds["sensor_q3"] + IQR_MULTIPLIER * bounds["sensor_iqr"]

    df = df.merge(bounds, on="sensor", how="left", validate="many_to_one")
    df["flag_sensor_iqr_outlier"] = (
        valid_temp
        & df["sensor_lower"].notna()
        & ((df["temperature"] < df["sensor_lower"]) | (df["temperature"] > df["sensor_upper"]))
    )

    # Rolling median residual outliers per sensor.
    # This catches isolated spikes that may be inside broad IQR limits.
    df = df.sort_values(["sensor", "time"]).reset_index(drop=True)
    rolling_median = (
        df.groupby("sensor", observed=True)["temperature"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, center=True, min_periods=5).median())
    )
    residual = df["temperature"] - rolling_median
    mad_by_sensor = residual.abs().groupby(df["sensor"], observed=True).transform("median")
    robust_z = residual.abs() / (1.4826 * mad_by_sensor + 1e-6)

    df["rolling_median_temperature"] = rolling_median
    df["rolling_residual"] = residual
    df["rolling_residual_robust_z"] = robust_z
    df["flag_rolling_spike"] = (
        df["temperature"].notna()
        & rolling_median.notna()
        & (robust_z > ROLLING_RESIDUAL_Z_LIMIT)
    )

    df["flag_row_outlier"] = (
        df["flag_missing_temperature"]
        | df["flag_bad_coordinates"]
        | df["flag_physical_outlier"]
        | df["flag_sensor_iqr_outlier"]
        | df["flag_rolling_spike"]
    )

    return df


def create_sensor_quality_report(flagged: pd.DataFrame) -> pd.DataFrame:
    """Create a sensor-level report for failed sensors and possible drift."""
    grouped = flagged.groupby("sensor", observed=True)

    summary = grouped.agg(
        n_rows=("temperature", "size"),
        n_missing=("flag_missing_temperature", "sum"),
        n_physical_outliers=("flag_physical_outlier", "sum"),
        n_iqr_outliers=("flag_sensor_iqr_outlier", "sum"),
        n_rolling_spikes=("flag_rolling_spike", "sum"),
        n_total_bad=("flag_row_outlier", "sum"),
        temp_mean=("temperature", "mean"),
        temp_median=("temperature", "median"),
        temp_std=("temperature", "std"),
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        time_min=("time_years", "min"),
        time_max=("time_years", "max"),
    ).reset_index()

    summary["bad_fraction"] = summary["n_total_bad"] / summary["n_rows"]
    summary["possible_failed_sensor"] = summary["bad_fraction"] > MAX_BAD_FRACTION_FOR_FAILED_SENSOR

    # Simple drift score: robust slope of temperature vs time per sensor.
    slopes = []
    for sensor, g in flagged.loc[~flagged["flag_row_outlier"]].groupby("sensor", observed=True):
        if len(g) < 50 or g["time_years"].nunique() < 2:
            slope = np.nan
        else:
            # Linear slope in degC/year.
            slope = np.polyfit(g["time_years"].to_numpy(), g["temperature"].to_numpy(), 1)[0]
        slopes.append((sensor, slope))

    slopes = pd.DataFrame(slopes, columns=["sensor", "temperature_slope_degC_per_year"])
    summary = summary.merge(slopes, on="sensor", how="left")

    med = summary["temperature_slope_degC_per_year"].median(skipna=True)
    mad = (summary["temperature_slope_degC_per_year"] - med).abs().median(skipna=True)
    summary["drift_slope_robust_z"] = (
        (summary["temperature_slope_degC_per_year"] - med).abs() / (1.4826 * mad + 1e-12)
    )
    summary["possible_drift_sensor"] = summary["drift_slope_robust_z"] > DRIFT_SLOPE_Z_LIMIT

    return summary.sort_values(["possible_failed_sensor", "possible_drift_sensor", "bad_fraction"], ascending=False)


def clean_train(flagged: pd.DataFrame, sensor_report: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows and optionally remove highly problematic sensors."""
    bad_sensors = set(sensor_report.loc[sensor_report["possible_failed_sensor"], "sensor"].astype(str))

    cleaned = flagged.loc[~flagged["flag_row_outlier"]].copy()
    cleaned = cleaned.loc[~cleaned["sensor"].astype(str).isin(bad_sensors)].copy()

    helper_cols = [
        "sensor_q1", "sensor_q3", "sensor_iqr", "sensor_lower", "sensor_upper",
        "rolling_median_temperature", "rolling_residual", "rolling_residual_robust_z",
        "flag_missing_temperature", "flag_bad_coordinates", "flag_physical_outlier",
        "flag_sensor_iqr_outlier", "flag_rolling_spike", "flag_row_outlier",
    ]
    cleaned = cleaned.drop(columns=[c for c in helper_cols if c in cleaned.columns])
    return cleaned.reset_index(drop=True)


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    raw_dir = find_raw_dir()
    print(f"Raw data folder: {raw_dir}")

    train, test, sensors = load_data(raw_dir)
    print(f"Raw train shape: {train.shape}")
    print(f"Raw test shape:  {test.shape}")
    print(f"Sensors shape:   {sensors.shape}")

    train = add_sensor_coordinates(train, sensors)
    test = add_sensor_coordinates(test, sensors)
    train = add_basic_features(train)
    test = add_basic_features(test)

    flagged = create_cleaning_flags(train)
    sensor_report = create_sensor_quality_report(flagged)
    cleaned = clean_train(flagged, sensor_report)

    removed_rows = flagged.loc[flagged["flag_row_outlier"]].copy()

    # Save outputs.
    cleaned.to_parquet(OUTPUT_DIR / "train_cleaned.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "test_with_features.parquet", index=False)
    sensor_report.to_csv(OUTPUT_DIR / "sensor_quality_report.csv", index=False)
    removed_rows[[
        "sensor", "time", "power", "temperature",
        "flag_missing_temperature", "flag_bad_coordinates", "flag_physical_outlier",
        "flag_sensor_iqr_outlier", "flag_rolling_spike", "flag_row_outlier",
    ]].to_csv(OUTPUT_DIR / "removed_training_rows.csv", index=False)

    print("\nCleaning complete.")
    print(f"Cleaned train shape: {cleaned.shape}")
    print(f"Removed rows: {len(removed_rows):,}")
    print(f"Possible failed sensors: {sensor_report['possible_failed_sensor'].sum()}")
    print(f"Possible drift sensors:  {sensor_report['possible_drift_sensor'].sum()}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
