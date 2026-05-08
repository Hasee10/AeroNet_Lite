import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

zone_colors = {
    "Residential": "#90EE90",
    "Commercial":  "#FFD700",
    "Industrial":  "#A9A9A9",
    "Hospital":    "#FF6B6B",
    "School":      "#87CEEB",
    "Open Field":  "#F5F5DC",
}
no_fly_color = "#FF0000"

ZONE_COLORS  = zone_colors
NO_FLY_COLOR = no_fly_color

figures_dir = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
os.makedirs(figures_dir, exist_ok=True)


def _save(fig, name):
    path = os.path.join(figures_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_grid_base(ax, grid):
    for r in range(10):
        for c in range(10):
            cell  = grid[r][c]
            yr    = 9 - r
            color = no_fly_color if cell.no_fly else zone_colors.get(cell.zone, "#ffffff")
            rect  = plt.Rectangle((c, yr), 1, 1, facecolor=color,
                                   edgecolor="#888888", linewidth=0.5)
            ax.add_patch(rect)

    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            yr   = 9 - r
            if cell.no_fly:
                ax.text(c + 0.5, yr + 0.5, "NO-FLY", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold")
            if cell.is_hub:
                ax.plot(c + 0.5, yr + 0.5, "*", color="#00008b", markersize=10)
                ax.text(c + 0.5, yr + 0.18, "Hub", ha="center", va="center",
                        fontsize=5, color="#00008b")
            if cell.is_charging:
                ax.text(c + 0.85, yr + 0.85, "C", ha="center", va="center",
                        fontsize=6, color="#555500", fontweight="bold")
            if cell.is_medical_pickup:
                ax.text(c + 0.85, yr + 0.15, "+", ha="center", va="center",
                        fontsize=9, color="darkred", fontweight="bold")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_xticklabels(range(11), fontsize=7)
    ax.set_yticklabels(list(range(9, -1, -1)) + [""], fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)


def plot_zone_map(grid):
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_grid_base(ax, grid)
    ax.set_title("AeroNet Lite - Zone Map", fontsize=13, fontweight="bold", pad=10)
    patches = [mpatches.Patch(facecolor=col, edgecolor="#888", label=zone)
               for zone, col in zone_colors.items()]
    patches += [
        mpatches.Patch(facecolor=no_fly_color, edgecolor="#888", label="No-Fly Zone"),
        mpatches.Patch(facecolor="#ffffff", edgecolor="#ffffff", label="* = Drone Hub"),
        mpatches.Patch(facecolor="#ffffff", edgecolor="#ffffff", label="C = Charging Pad"),
        mpatches.Patch(facecolor="#ffffff", edgecolor="#ffffff", label="+ = Medical Pickup"),
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8, framealpha=0.9)
    return _save(fig, "zone_map.png")


def plot_route_map(grid, drones, deliveries):
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_grid_base(ax, grid)
    ax.set_title("AeroNet Lite - Drone Routes", fontsize=13, fontweight="bold", pad=10)
    colors = plt.cm.tab10.colors
    for idx, drone in enumerate(drones):
        if not drone.route:
            continue
        color = colors[idx % len(colors)]
        xs = [c + 0.5 for _, c in drone.route]
        ys = [9 - r + 0.5 for r, _ in drone.route]
        ax.plot(xs, ys, "-o", color=color, markersize=3, linewidth=1.5,
                label=f"{drone.drone_id} ({drone.drone_type})", alpha=0.8)
    if drones:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.9)
    return _save(fig, "route_map.png")


def plot_demand_heatmap(grid):
    data = np.array([[grid[r][c].demand for c in range(10)] for r in range(10)])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="equal")
    plt.colorbar(im, ax=ax, label="Demand intensity")
    ax.set_title("AeroNet Lite - Demand Heatmap", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for r in range(10):
        for c in range(10):
            ax.text(c, r, f"{data[r,c]:.1f}", ha="center", va="center",
                    fontsize=6, color="black")
    return _save(fig, "demand_heatmap.png")


def plot_anomaly_summary(anomaly_log):
    if not anomaly_log:
        anomaly_log = [{"drone": "D1", "type": "Normal", "battery_drop": 4.0, "route_deviation": 0.5}]
    drone_ids  = [e["drone"] for e in anomaly_log]
    types      = [e["type"]  for e in anomaly_log]
    colors_map = {"Normal": "green", "Battery Anomaly": "red",
                  "Route Anomaly": "orange", "Sensor Spike": "purple"}
    bar_colors = [colors_map.get(t, "blue") for t in types]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(drone_ids, [1] * len(drone_ids), color=bar_colors, edgecolor="black")
    for bar, t in zip(bars, types):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, t, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Drone ID")
    ax.set_title("Anomaly Detection Summary", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(facecolor=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    return _save(fig, "anomaly_summary.png")


def plot_simulation_summary(summary):
    labels = ["Completed", "Delayed", "Failed", "Pending"]
    values = [summary["completed"], summary["delayed"], summary["failed"], summary["pending"]]
    colors = ["#4caf50", "#ff9800", "#f44336", "#9e9e9e"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Deliveries")
    ax.set_title("Simulation Final Summary", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) + 1.5 if values else 2)
    return _save(fig, "simulation_summary.png")


def generate_all_figures(grid, drones, deliveries, summary, anomaly_log=None):
    paths = []
    paths.append(plot_zone_map(grid))
    paths.append(plot_route_map(grid, drones, deliveries))
    paths.append(plot_demand_heatmap(grid))
    paths.append(plot_anomaly_summary(anomaly_log or []))
    paths.append(plot_simulation_summary(summary))
    return paths
