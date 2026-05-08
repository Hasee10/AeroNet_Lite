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

processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

anomaly_labels = {0: "Normal", 1: "Battery Anomaly", 2: "Route Anomaly", 3: "Sensor Spike"}


def _generate_demand_data(n=1200, seed=0):
    rng         = np.random.default_rng(seed)
    hour        = rng.integers(0, 24, n)
    day         = rng.integers(1, 8, n)
    month       = rng.integers(1, 13, n)
    season      = np.where(month <= 3, 1, np.where(month <= 6, 2, np.where(month <= 9, 3, 4)))
    temp        = rng.uniform(5, 38, n)
    humidity    = rng.uniform(20, 95, n)
    weather     = rng.integers(1, 5, n)
    zone_density= rng.uniform(0.5, 5.0, n)
    base         = 3 * zone_density
    hour_effect  = np.where((hour >= 8) & (hour <= 20), 2.5, 0.5)
    day_effect   = np.where(day <= 5, 1.2, 0.8)
    season_effect= np.where(season == 3, 1.3, np.where(season == 1, 0.7, 1.0))
    temp_effect  = np.clip((temp - 15) / 10, -1, 1) * 0.8
    weather_eff  = np.where(weather == 1, 1.0, np.where(weather == 2, 0.8,
                   np.where(weather == 3, 0.5, 0.3)))
    noise        = rng.normal(0, 0.5, n)
    count = np.round(np.clip(
        base * hour_effect * day_effect * season_effect * weather_eff + temp_effect + noise, 0, None), 1)
    return pd.DataFrame({
        "hour": hour, "day": day, "month": month, "season": season,
        "temp": temp.round(1), "humidity": humidity.round(1),
        "weather": weather, "zone_density": zone_density.round(2), "count": count,
    })


def run_demand_forecast(grid=None, verbose=True):
    df       = _generate_demand_data()
    df.to_csv(os.path.join(processed_dir, "demand_data.csv"), index=False)
    features = ["hour", "day", "month", "season", "temp", "humidity", "weather", "zone_density"]
    x        = df[features].values
    y        = df["count"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_mae  = round(mean_absolute_error(y_test, lr.predict(x_test)), 3)
    lr_rmse = round(np.sqrt(mean_squared_error(y_test, lr.predict(x_test))), 3)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_mae  = round(mean_absolute_error(y_test, rf_pred), 3)
    rf_rmse = round(np.sqrt(mean_squared_error(y_test, rf_pred)), 3)
    sample_pred   = round(float(rf.predict(np.array([[14, 3, 6, 2, 25.0, 60.0, 1, 3.5]]))[0]), 2)
    sample_scaled = round(sample_pred / max(rf_pred.max(), 1) * 10, 2)
    grid_forecast = {}
    if grid is not None:
        feat_arr = np.array([[12, 3, 6, 2, 25.0, 55.0, 1, cell.density / 1000.0]
                             for row in grid for cell in row])
        raw      = rf.predict(feat_arr)
        scaled   = (raw / max(raw.max(), 1) * 10).reshape(10, 10)
        for r in range(10):
            for c in range(10):
                grid[r][c].demand = round(float(scaled[r, c]), 2)
        grid_forecast = {"updated": True, "zones_updated": 100}
    if verbose:
        print(f"\n{'='*60}")
        print("  DEMAND FORECASTING RESULTS")
        print(f"{'='*60}")
        print(f"  Dataset    : synthetic (n={len(df)}, Bike-Sharing style)")
        print(f"  Features   : {features}")
        print(f"  Model      : Random Forest Regressor (n_estimators=100)")
        print(f"  Linear Regression  ->  MAE={lr_mae}  |  RMSE={lr_rmse}")
        print(f"  Random Forest      ->  MAE={rf_mae}  |  RMSE={rf_rmse}")
        print(f"  Sample forecast    : Hour 14 demand = {sample_pred} (scaled: {sample_scaled})")
        if grid_forecast:
            print(f"  Demand mapped to grid: {grid_forecast['zones_updated']} zones updated.")
        print(f"{'='*60}\n")
    return {"lr_mae": lr_mae, "lr_rmse": lr_rmse, "rf_mae": rf_mae,
            "rf_rmse": rf_rmse, "model": rf, "grid_forecast": grid_forecast}


train_demand_model = run_demand_forecast


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
