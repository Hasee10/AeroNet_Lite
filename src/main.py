import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from module_0 import create_grid, print_grid_summary
from module_1 import validate_layout, print_validation_report
from module_2 import select_fleet, print_fleet_report
from module_4 import (
    build_fleet, generate_deliveries,
    assign_deliveries, move_drones,
    activate_no_fly, activate_nofly,
    check_routes_for_nofly, reroute_drones,
    inject_anomaly, simulation_summary,
    step_simulation,
)
from module_5 import (
    run_demand_forecast, train_demand_model,
    train_anomaly_model, classify_drone_telemetry, classify_telemetry,
)
from visualization import generate_all_figures

event_log = []


def log(step, message):
    entry = f"Step {step:>2}: {message}"
    event_log.append(entry)
    print(entry)


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_simulation():
    anomaly_log = []

    section("STEPS 1-3 | Initialization, Validation & Fleet Selection")

    grid = create_grid(seed=42)
    log(1, "Grid initialized (10x10).")
    print_grid_summary(grid)

    report = validate_layout(grid)
    print_validation_report(report)
    status = "passed" if report["valid"] else f"FAILED rules {report['failed']}"
    log(2, f"Layout validation {status}.")

    fleet_result = select_fleet(grid, budget=10_000)
    print_fleet_report(fleet_result)
    log(3, f"Fleet selected: {fleet_result['light']} Light + "
           f"{fleet_result['heavy']} Heavy drones. "
           f"Cost: {fleet_result['cost']:,} / {fleet_result['budget']:,} units.")

    drones = build_fleet(fleet_result, grid)

    section("STEPS 4-6 | Delivery Generation & Route Planning")

    deliveries = generate_deliveries(grid, count=8, seed=7)
    log(4, f"Generated {len(deliveries)} deliveries.")
    for d in deliveries:
        log(4, f"  {d.delivery_id}: pickup={d.pickup} dropoff={d.dropoff} ({d.payload_kg}kg)")

    pre_assign_log = []
    assign_deliveries(deliveries, drones, grid, pre_assign_log)
    for entry in pre_assign_log:
        event_log.append(f"Step  5: {entry.strip()}")
        print(f"Step  5: {entry.strip()}")
    assigned_count = sum(1 for d in deliveries if d.status == "assigned")
    failed_assign  = sum(1 for d in deliveries if d.status == "failed")
    log(5, f"Delivery assignment complete. Assigned: {assigned_count}  "
           f"Not routable (payload/path): {failed_assign}.")

    log(6, f"All routes computed via A*. "
           f"Drones en route: {sum(1 for dr in drones if dr.status == 'en_route')}.")

    section("STEPS 7-10 | Drone Movement")

    for step in range(7, 11):
        move_drones(drones, deliveries, event_log)
        positions = {dr.drone_id: dr.position for dr in drones if dr.status == "en_route"}
        log(step, f"Drones advanced. En-route: {positions}. "
                  f"Completed so far: {sum(1 for d in deliveries if d.status=='completed')}.")

    section("STEP 11 | Disruption - No-Fly Cell Activated")

    no_fly_cell = (4, 7)
    affected    = activate_nofly(no_fly_cell[0], no_fly_cell[1], grid, drones)
    event_log.append(f"Step 11:   No-fly cell activated at {no_fly_cell}. Scanning routes.")
    print(      f"Step 11:   No-fly cell activated at {no_fly_cell}. Scanning routes.")
    log(11, f"Affected drone(s): {affected if affected else 'none on that cell'}.")

    section("STEPS 12-14 | Disruption Rerouting")

    for step in range(12, 15):
        reroute_log = []
        reroute_drones(drones, deliveries, grid, reroute_log)
        for entry in reroute_log:
            event_log.append(f"Step {step:>2}: {entry.strip()}")
            print(      f"Step {step:>2}: {entry.strip()}")
        move_drones(drones, deliveries, event_log)
        log(step, f"Rerouting pass {step - 11} complete. "
                  f"En-route: {sum(1 for dr in drones if dr.status=='en_route')}  "
                  f"Completed: {sum(1 for d in deliveries if d.status=='completed')}.")

    section("STEPS 15-17 | Demand Forecasting")

    demand_result = run_demand_forecast(grid=grid, verbose=True)
    log(15, f"Demand forecast run. "
            f"RF MAE={demand_result['rf_mae']}  RMSE={demand_result['rf_rmse']}.")
    log(16, "Grid demand values updated from ML forecast.")

    high_demand = max(
        [(r, c) for r in range(10) for c in range(10) if grid[r][c].zone == "Residential"],
        key=lambda pos: grid[pos[0]][pos[1]].demand,
    )
    extra = next((d for d in deliveries if d.status == "pending"), None)
    if extra:
        log(17, f"High-demand cell {high_demand} detected. Dispatching {extra.delivery_id}.")
        extra_log = []
        assign_deliveries([extra], drones, grid, extra_log)
        for entry in extra_log:
            print(f"Step 17: {entry.strip()}")
    else:
        log(17, f"High-demand cell {high_demand} - no pending deliveries to dispatch.")

    section("STEP 18 | Anomaly Detection & Classification")

    anomaly_result = train_anomaly_model(verbose=True)
    clf = anomaly_result["model"]

    d3 = next((dr for dr in drones if dr.drone_id == "D3"), None)
    target_drone = d3 if d3 else (drones[0] if drones else None)

    if target_drone:
        inject_anomaly(target_drone, "battery", event_log)
        anomaly_type = classify_drone_telemetry(
            clf, battery_drop=35.0, speed_change=0.5,
            altitude_change=0.5, route_deviation=0.8
        )
        log(18, f"Real-time use: Drone {target_drone.drone_id} telemetry classified "
                f"as '{anomaly_type}' at Step 18.")
        anomaly_log.append({
            "drone": target_drone.drone_id, "type": anomaly_type,
            "battery_drop": 35.0, "route_deviation": 0.8,
        })

    for dr in drones:
        if target_drone and dr.drone_id == target_drone.drone_id:
            continue
        t = classify_drone_telemetry(clf, 1.5, 1.0, 0.3, 0.2)
        anomaly_log.append({"drone": dr.drone_id, "type": t,
                             "battery_drop": 1.5, "route_deviation": 0.2})

    section("STEP 19 | Post-Anomaly Response")

    if target_drone is not None and target_drone.status == "returning":
        target_drone.position = target_drone.hub
        target_drone.status   = "idle"
        log(19, f"D3 returning to hub {target_drone.hub} safely.")
    else:
        move_drones(drones, deliveries, event_log)
        log(19, f"Drones advanced one step. "
                f"Completed: {sum(1 for d in deliveries if d.status=='completed')}.")

    section("STEP 20 | Simulation Complete")

    summary = simulation_summary(deliveries)
    log(20, f"Simulation complete. "
            f"Completed: {summary['completed']}  "
            f"Delayed: {summary['delayed']}  "
            f"Failed: {summary['failed']}  "
            f"Pending: {summary['pending']}.")

    print(f"\n{'='*60}")
    print("  DELIVERY STATUS TABLE")
    print(f"{'='*60}")
    print(f"  {'ID':<10} {'Pickup':<10} {'Dropoff':<10} {'Payload':>7}  {'Status':<12}  {'Drone'}")
    print(f"  {'-'*9} {'-'*9} {'-'*9} {'-'*7}  {'-'*11}  {'-'*5}")
    for d in deliveries:
        print(f"  {d.delivery_id:<10} {str(d.pickup):<10} {str(d.dropoff):<10} "
              f"{d.payload_kg:>6.1f}kg  {d.status:<12}  {d.assigned_drone or '-'}")
    print(f"{'='*60}")

    section("Generating Visualizations")

    paths = generate_all_figures(grid, drones, deliveries, summary, anomaly_log)
    for p in paths:
        print(f"  Saved: {os.path.relpath(p)}")
    print(f"\n  All figures saved to report/figures/")

    print(f"\n{'='*60}")
    print("  AeroNet Lite simulation finished successfully.")
    print(f"{'='*60}\n")

    return {
        "grid": grid, "drones": drones, "deliveries": deliveries,
        "summary": summary, "validation": report,
        "fleet": fleet_result, "demand": demand_result, "anomaly": anomaly_result,
    }


if __name__ == "__main__":
    run_simulation()
