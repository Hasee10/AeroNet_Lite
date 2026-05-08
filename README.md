# AeroNet Lite – Autonomous Drone Delivery Simulation

**Course:** AI – Semester 6  
**Project:** BSDS Semester Project SP2026

---

## Overview

AeroNet Lite simulates a drone delivery system over a 10×10 city grid using five AI techniques:

| Module | File | AI Technique |
|--------|------|-------------|
| Grid & Layout Validator | `layout_validator.py` | Constraint Satisfaction (CSP) |
| Fleet Selector | `fleet_selector.py` | Genetic Algorithm |
| Delivery Path Planner | `astar_planner.py` | A* Search |
| Disruption Handler | `delivery_simulator.py` | Optimization / Rerouting |
| ML Pipeline | `ml_pipeline.py` | Regression + Classification |

---

## How to Run

```bash
cd aeronet_lite/src
python main.py
```

Requires: Python 3.10+, numpy, pandas, scikit-learn, matplotlib (all standard Anaconda packages).

Output figures are saved to `report/figures/`.

---

## Project Structure

```
aeronet_lite/
  data/
    raw/            (place downloaded Kaggle CSVs here if available)
    processed/      (auto-generated synthetic datasets)
  src/
    grid_model.py         Shared 10x10 grid data model
    layout_validator.py   CSP constraints R1–R4
    fleet_selector.py     Genetic Algorithm fleet selection
    astar_planner.py      A* path planning with Commercial corridor cost
    delivery_simulator.py Drone management & disruption handling
    ml_pipeline.py        Demand forecasting + anomaly classification
    visualization.py      matplotlib figures
    main.py               20-step simulation entry point
  notebooks/
    demand_forecasting.ipynb
    anomaly_classifier.ipynb
  report/
    figures/        (auto-generated plots)
  README.md
```

---

## Modules

### Module 1 – CSP Layout Validator

Four hard constraints checked on the 10×10 grid:

| Rule | Constraint |
|------|-----------|
| R1 | Industrial cells not adjacent to School/Hospital |
| R2 | Every Residential within 3 Manhattan cells of a Hub |
| R3 | Every Hub within 2 cells of a Charging Pad |
| R4 | At least one Hospital has a Medical Pickup within 1 cell |

### Module 2 – Fleet Selector (Genetic Algorithm)

- Chromosome: `[light_count, heavy_count]`
- Fitness: `0.75 × coverage_pct − 0.25 × budget_used_pct`
- Budget: $15,000 | Light: $1,000 | Heavy: $1,800
- 60 generations, tournament selection, single-point crossover

### Module 3 – A* Path Planner

- State: `(row, col)`, Actions: 4-directional movement
- Step cost: 0.8 (Commercial corridor), 1.0 (all others)
- Blocked: `no_fly == True` cells
- Heuristic: Manhattan distance (admissible)
- Full route: hub → pickup → dropoff → hub

### Module 4 – Disruption Handler

At step 11 a no-fly cell is activated. Any drone whose remaining path crosses that cell is rerouted using A* from its current position.

### Module 5 – ML Pipeline

**Demand Forecasting:**  
- Synthetic dataset (1200 rows, Bike-Sharing style)  
- Features: hour, day_of_week, temperature, weather, zone_density  
- Models: Linear Regression and Random Forest Regressor  
- Metric: MAE and RMSE

**Anomaly Detection:**  
- Synthetic telemetry (800 rows, 4 classes)  
- Features: battery_drop, speed, altitude_change, route_deviation  
- Classes: Normal, Battery anomaly, Route anomaly, Sensor spike  
- Model: Decision Tree Classifier  
- Metric: Accuracy + Confusion Matrix

---

## Datasets Used

| Purpose | Source |
|---------|--------|
| Demand forecasting | Synthetic (Bike-Sharing Demand structure) |
| Anomaly detection | Synthetic (UAV telemetry structure) |
| Population density | Grid density values from ZONE_DENSITIES mapping |

Real Kaggle datasets can be loaded by replacing the synthetic generation functions in `ml_pipeline.py`.

---

## Sample Output (Step Log)

```
Step  1: Grid initialized (10x10).
Step  2: Layout validation FAILED rules ['R2'].
Step  3: Fleet selected: 15 light drone(s), 0 heavy drone(s).
Step  5: Delivery assignment complete. Assigned: 4.
Step  7: Drones moved. Completed so far: 0.
Step 11: No-fly cell activated at (4, 4).
Step 12: Rerouting pass 1.
Step 15: Demand model trained. RF MAE=0.567  RMSE=0.701.
Step 18: Anomaly classifier output for D1: 'Battery anomaly'.
Step 20: Simulation complete. Completed: 2  Delayed: 0  Failed: 0  Pending: 6.
```
