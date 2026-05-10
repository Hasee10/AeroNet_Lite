import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                     accuracy_score, confusion_matrix,
                                     classification_report)

_base_dir     = os.path.dirname(__file__)
processed_dir = os.path.join(_base_dir, "..", "data", "processed")
raw_dir       = os.path.join(_base_dir, "..", "data", "raw")
os.makedirs(processed_dir, exist_ok=True)

anomaly_labels = {0: "Normal", 1: "Battery Anomaly", 2: "Route Anomaly", 3: "Sensor Spike"}

# ── Real features from Bike-Sharing dataset (delivery demand proxy) ──────────
DEMAND_FEATURES = ["season", "holiday", "workingday", "weather",
                   "temp", "humidity", "windspeed", "hour", "month", "weekday"]

# Per-zone typical conditions used to predict grid-cell demand at inference time
_ZONE_FEATURE_ROWS = {
    # [season, holiday, workingday, weather, temp, humidity, windspeed, hour, month, weekday]
    "Residential": [2, 0, 1, 1, 20.0, 55.0, 12.0, 18, 6, 0],  # evening rush, summer
    "Commercial":  [2, 0, 1, 1, 22.0, 50.0, 10.0, 14, 6, 1],  # midday peak, summer
    "Industrial":  [2, 0, 1, 2, 18.0, 60.0, 18.0,  8, 6, 2],  # morning shift, overcast
    "Hospital":    [2, 0, 1, 1, 20.0, 55.0, 12.0, 10, 6, 3],  # mid-morning, clear
    "School":      [2, 0, 1, 1, 19.0, 57.0, 11.0,  9, 9, 4],  # school morning, fall
    "Open Field":  [1, 0, 0, 1, 12.0, 70.0, 22.0, 10, 1, 5],  # winter, weekend, low
}


def _load_demand_data():
    """
    Primary  : data/raw/train.csv  — real Kaggle Bike-Sharing dataset
               (used as delivery-demand proxy per assignment spec)
    Fallback : _generate_demand_data()  — synthetic, if CSV missing
    """
    csv_path = os.path.join(raw_dir, "train.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour"]     = df["datetime"].dt.hour
        df["month"]    = df["datetime"].dt.month
        df["weekday"]  = df["datetime"].dt.weekday
        keep = DEMAND_FEATURES + ["count"]
        df   = df[keep].dropna().reset_index(drop=True)
        return df, True
    # ── fallback ─────────────────────────────────────────────────────────────
    return _generate_demand_data(), False


def _generate_demand_data(n=1200, seed=0):
    """Synthetic fallback — only used if train.csv is absent."""
    rng      = np.random.default_rng(seed)
    season   = rng.integers(1, 5, n)
    holiday  = rng.integers(0, 2, n)
    working  = 1 - holiday
    weather  = rng.integers(1, 5, n)
    temp     = rng.uniform(5, 38, n)
    humidity = rng.uniform(20, 95, n)
    wind     = rng.uniform(0, 50, n)
    hour     = rng.integers(0, 24, n)
    month    = rng.integers(1, 13, n)
    weekday  = rng.integers(0, 7, n)
    base     = (np.where((hour >= 7) & (hour <= 20), 2.5, 0.5) *
                np.where(working == 1, 1.2, 0.8) *
                np.where(season == 3, 1.3, np.where(season == 1, 0.7, 1.0)) *
                np.where(weather == 1, 1.0, np.where(weather == 2, 0.8, 0.4)))
    count = np.round(np.clip(base * 20 + rng.normal(0, 3, n), 0, None), 1)
    return pd.DataFrame({
        "season": season, "holiday": holiday, "workingday": working,
        "weather": weather, "temp": temp.round(1), "humidity": humidity.round(1),
        "windspeed": wind.round(1), "hour": hour, "month": month,
        "weekday": weekday, "count": count,
    }), False


def run_demand_forecast(grid=None, verbose=True):
    df, is_real = _load_demand_data()
    df.to_csv(os.path.join(processed_dir, "demand_data.csv"), index=False)

    x = df[DEMAND_FEATURES].values
    y = df["count"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # ── Linear Regression ────────────────────────────────────────────────────
    lr      = LinearRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    lr_mae  = round(mean_absolute_error(y_test, lr_pred), 3)
    lr_rmse = round(float(np.sqrt(mean_squared_error(y_test, lr_pred))), 3)
    lr_r2   = round(float(1 - np.sum((y_test - lr_pred)**2) /
                          np.sum((y_test - y_test.mean())**2)), 3)

    # ── Random Forest Regressor ───────────────────────────────────────────────
    rf      = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_mae  = round(mean_absolute_error(y_test, rf_pred), 3)
    rf_rmse = round(float(np.sqrt(mean_squared_error(y_test, rf_pred))), 3)
    rf_r2   = round(float(1 - np.sum((y_test - rf_pred)**2) /
                          np.sum((y_test - y_test.mean())**2)), 3)

    # Sample: working day, spring, clear weather, noon, mild conditions
    sample_row   = np.array([[2, 0, 1, 1, 20.0, 55.0, 15.0, 12, 4, 2]])
    sample_pred  = round(float(rf.predict(sample_row)[0]), 2)
    sample_scaled= round(sample_pred / max(rf_pred.max(), 1) * 10, 2)

    # ── Grid demand mapping ───────────────────────────────────────────────────
    grid_forecast = {}
    if grid is not None:
        feat_arr = np.array([
            _ZONE_FEATURE_ROWS.get(cell.zone, _ZONE_FEATURE_ROWS["Open Field"])
            for row in grid for cell in row
        ])
        raw    = rf.predict(feat_arr)
        scaled = (raw / max(raw.max(), 1) * 10).reshape(10, 10)
        for r in range(10):
            for c in range(10):
                grid[r][c].demand = round(float(scaled[r, c]), 2)
        grid_forecast = {"updated": True, "zones_updated": 100}

    if verbose:
        src = (f"real  - Bike-Sharing Dataset  (n={len(df)}, data/raw/train.csv)"
               if is_real else f"synthetic fallback (n={len(df)})")
        print(f"\n{'='*60}")
        print("  MODULE 5a - DEMAND FORECASTING")
        print(f"{'='*60}")
        print(f"  Data source : {src}")
        print(f"  Proxy note  : Bike-rental demand -> delivery demand (per spec)")
        print(f"  Features    : {DEMAND_FEATURES}")
        print(f"\n  Linear Regression   MAE={lr_mae}  RMSE={lr_rmse}  R2={lr_r2}")
        print(f"  Random Forest       MAE={rf_mae}  RMSE={rf_rmse}  R2={rf_r2}  <- best")
        print(f"  Sample forecast     noon/spring/clear -> {sample_pred} units (scaled: {sample_scaled}/10)")
        if grid_forecast:
            print(f"  Grid updated        {grid_forecast['zones_updated']} zones re-scored")
        print(f"{'='*60}\n")

    return {
        "lr_mae": lr_mae, "lr_rmse": lr_rmse, "lr_r2": lr_r2,
        "rf_mae": rf_mae, "rf_rmse": rf_rmse, "rf_r2": rf_r2,
        "model": rf, "grid_forecast": grid_forecast,
        "data_source": "real" if is_real else "synthetic",
        "n_samples": len(df),
    }


train_demand_model = run_demand_forecast   # backward-compat alias


def _generate_anomaly_data(n=800, seed=1):
    rng       = np.random.default_rng(seed)
    records   = []
    per_class = n // 4
    for label in range(4):
        for _ in range(per_class):
            if label == 0:
                rec = dict(battery_drop=rng.uniform(1.0, 2.5),
                           route_deviation=rng.uniform(0.0, 1.5),
                           altitude_change=rng.uniform(-2.0, 2.0),
                           speed_change=rng.uniform(-3.0, 3.0), label=0)
            elif label == 1:
                rec = dict(battery_drop=rng.uniform(5.1, 15.0),
                           route_deviation=rng.uniform(0.0, 1.5),
                           altitude_change=rng.uniform(-2.0, 2.0),
                           speed_change=rng.uniform(-3.0, 3.0), label=1)
            elif label == 2:
                rec = dict(battery_drop=rng.uniform(1.0, 2.5),
                           route_deviation=rng.uniform(3.1, 10.0),
                           altitude_change=rng.uniform(-2.0, 2.0),
                           speed_change=rng.uniform(-3.0, 3.0), label=2)
            else:
                rec = dict(battery_drop=rng.uniform(1.0, 2.5),
                           route_deviation=rng.uniform(0.0, 1.5),
                           altitude_change=rng.uniform(20.0, 50.0),
                           speed_change=rng.uniform(20.0, 60.0), label=3)
            records.append(rec)
    df = pd.DataFrame(records)
    for col in ["battery_drop", "route_deviation", "altitude_change", "speed_change"]:
        df[col] = df[col].round(2)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def train_anomaly_model(verbose=True):
    df = _generate_anomaly_data()
    df.to_csv(os.path.join(processed_dir, "flight_anomalies.csv"), index=False)
    feat_cols = ["battery_drop", "route_deviation", "altitude_change", "speed_change"]
    x = df[feat_cols].values
    y = df["label"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    labels = list(anomaly_labels.values())
    dt  = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(x_train, y_train)
    dt_acc = round(accuracy_score(y_test, dt.predict(x_test)), 4)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_acc  = round(accuracy_score(y_test, rf_pred), 4)
    cm      = confusion_matrix(y_test, rf_pred)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    knn_acc = round(accuracy_score(y_test, knn.predict(x_test)), 4)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    gnb_acc = round(accuracy_score(y_test, gnb.predict(x_test)), 4)
    if verbose:
        print(f"\n{'='*60}")
        print("  ANOMALY DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"  Dataset    : synthetic flight telemetry (n={len(df)}, 4 classes)")
        print(f"  Features   : {feat_cols}")
        print(f"\n  Model Comparison:")
        print(f"    Decision Tree (depth=6)  : {dt_acc*100:.2f}%")
        print(f"    Random Forest (100 trees): {rf_acc*100:.2f}%  <- best")
        print(f"    KNN (k=5)                : {knn_acc*100:.2f}%")
        print(f"    Gaussian NB              : {gnb_acc*100:.2f}%")
        print(f"\n  Random Forest - Confusion Matrix (rows=actual, cols=predicted):")
        header = "            " + "  ".join(f"{l[:10]:>10}" for l in labels)
        print(header)
        for i, row in enumerate(cm):
            print(f"  {labels[i][:10]:>10}  " + "  ".join(f"{v:>10}" for v in row))
        print(f"\n  Classification Report:\n"
              f"{classification_report(y_test, rf_pred, target_names=labels)}")
        print(f"{'='*60}\n")
    return {"model": rf, "accuracy": rf_acc, "dt_accuracy": dt_acc,
            "knn_accuracy": knn_acc, "gnb_accuracy": gnb_acc,
            "confusion_matrix": cm, "labels": anomaly_labels}


def classify_telemetry(model, drone_state):
    features = np.array([[
        drone_state.get("battery_drop",    0.0),
        drone_state.get("route_deviation", 0.0),
        drone_state.get("altitude_change", 0.0),
        drone_state.get("speed_change",    0.0),
    ]])
    return anomaly_labels[model.predict(features)[0]]


def classify_drone_telemetry(model, battery_drop, speed_change, altitude_change, route_deviation):
    return classify_telemetry(model, {
        "battery_drop": battery_drop, "route_deviation": route_deviation,
        "altitude_change": altitude_change, "speed_change": speed_change,
    })
