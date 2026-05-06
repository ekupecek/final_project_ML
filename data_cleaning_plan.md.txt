# Data Cleaning for the Nuclear Waste Temperature Project

## 1. Goal

The objective is to clean the training data before building machine learning models to predict temperatures in the repository tunnel.

Main issues mentioned in the project slides:

* Missing values
* Outliers
* Failed sensors
* Sensor drift

The training dataset contains:

| Column      | Description          |
| ----------- | -------------------- |
| sensor      | Sensor ID            |
| time        | Time in seconds      |
| power       | Heating power        |
| temperature | Measured temperature |

The sensors dataset contains sensor coordinates:

| Column | Description  |
| ------ | ------------ |
| sensor | Sensor ID    |
| coor_x | x coordinate |
| coor_y | y coordinate |
| coor_z | z coordinate |

---

# 2. First Observations From the Dataset

## Dataset size

Training rows:

* 6,626,928 rows

Missing temperatures:

* 99,403 missing values

Potential issues detected:

* Temperatures as low as -292°C
* Temperatures as high as 6039°C

These values are physically unrealistic and are almost certainly outliers or broken sensor measurements.

---

# 3. Recommended Cleaning Pipeline

## Step 1 — Load Data

```python
import pandas as pd

train = pd.read_parquet("train.parquet")
test = pd.read_parquet("test.parquet")
sensors = pd.read_parquet("sensors.parquet")
```

---

## Step 2 — Merge Sensor Coordinates

Coordinates are extremely important for prediction.

```python
train = train.merge(sensors, on="sensor", how="left")
test = test.merge(sensors, on="sensor", how="left")
```

---

## Step 3 — Check Missing Values

```python
print(train.isnull().sum())
```

You will notice:

```python
temperature ≈ 99k missing values
```

Since temperature is the target variable:

* Remove rows where temperature is missing.

```python
train = train.dropna(subset=["temperature"])
```

---

# 4. Outlier Cleaning

## Why?

Some temperatures are physically impossible.

Examples found:

* -292°C
* 6039°C

These are almost certainly:

* failed sensors
* corrupted measurements
* numerical errors

---

## Simple Physical Filtering

A reasonable first filter:

```python
train = train[
    (train["temperature"] > 0) &
    (train["temperature"] < 300)
]
```

You can later refine this threshold.

---

## Step 5 — Detect Statistical Outliers

### Option A — Z-score

```python
from scipy.stats import zscore

z = zscore(train["temperature"])
train = train[abs(z) < 4]
```

---

### Option B — IQR filtering

```python
Q1 = train["temperature"].quantile(0.25)
Q3 = train["temperature"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

train = train[
    (train["temperature"] >= lower) &
    (train["temperature"] <= upper)
]
```

IQR is usually safer for skewed data.

---

# 5. Sensor Drift Detection

The project explicitly mentions sensor drift.

Sensor drift means:

* a sensor slowly becomes biased over time.

Example:

* neighboring sensors remain stable
* one sensor slowly increases abnormally

---

## Basic Drift Detection Strategy

For each sensor:

* plot temperature vs time
* compare nearby sensors
* look for sudden jumps or long-term divergence

Example:

```python
import matplotlib.pyplot as plt

sensor_id = "N102"

subset = train[train["sensor"] == sensor_id]

plt.figure(figsize=(10,4))
plt.plot(subset["time"], subset["temperature"])
plt.title(sensor_id)
plt.xlabel("time")
plt.ylabel("temperature")
plt.show()
```

---

# 6. Feature Engineering (Very Important)

You should create useful features.

## Recommended Features

### Spatial features

```python
train["distance"] = (
    train["coor_x"]**2 +
    train["coor_y"]**2
) ** 0.5
```

---

### Time features

Convert seconds into years:

```python
SECONDS_PER_YEAR = 365 * 24 * 3600

train["time_years"] = train["time"] / SECONDS_PER_YEAR
```

---

### Interaction features

```python
train["power_distance"] = train["power"] * train["distance"]
```

---

# 7. Train / Validation Split

You should create your own validation set.

Example:

```python
from sklearn.model_selection import train_test_split

X = train.drop(columns=["temperature"])
y = train["temperature"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 8. Important Recommendation

Do NOT use the sensor name directly as a categorical variable initially.

Instead:

* use coordinates
* use distances
* use geometry-based features

This usually generalizes better to unseen locations.

---

# 9. Good First Models

Recommended baseline models:

## Easy baselines

* Linear Regression
* Random Forest
* XGBoost
* LightGBM

## Advanced models

* Neural Networks
* Temporal models
* Physics-informed ML

For this project:

XGBoost or LightGBM will probably be very strong baselines.

---

# 10. Suggested Workflow

1. Clean missing values
2. Remove impossible temperatures
3. Detect outliers
4. Add spatial features
5. Create validation split
6. Train baseline model
7. Evaluate RMSE/MAE
8. Improve features
9. Detect drifted sensors
10. Submit to Kaggle

---

# 11. Strong Recommendation for Collaboration

Avoid editing the same notebook simultaneously.

Better structure:

```text
project/
│
├── data/
├── notebooks/
├── src/
│   ├── cleaning.py
│   ├── features.py
│   ├── train.py
│   └── models.py
├── submissions/
└── README.md
```

This prevents notebook corruption and merge conflicts.
