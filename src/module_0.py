from dataclasses import dataclass
import os
import random

try:
    import pandas as pd
    _pd_available = True
except ImportError:
    _pd_available = False

zone_types = ["Residential", "Commercial", "Industrial", "Hospital", "School", "Open Field"]

zone_colors = {
    "Residential": "#90EE90",
    "Commercial":  "#FFD700",
    "Industrial":  "#A9A9A9",
    "Hospital":    "#FF6B6B",
    "School":      "#87CEEB",
    "Open Field":  "#F5F5DC",
}

# ── Hardcoded fallback densities (persons per sq mile) ───────────────────────
_DENSITY_FALLBACK = {
    "Residential": 5000,
    "Commercial":  3000,
    "Industrial":  800,
    "Hospital":    500,
    "School":      1200,
    "Open Field":  200,
}


def _load_zone_densities():
    """
    Derive per-zone density values from the real US City Population Density
    dataset (data/raw/uscitypopdensity.csv).

    Mapping logic:
      Residential  → 75th-percentile city density  (dense urban neighbourhoods)
      Commercial   → 50th-percentile city density  (mixed-use downtown)
      Industrial   → 25th-percentile city density  (sparse industrial zones)
      Hospital     → 25th-percentile / 2            (campus-style, low surrounding pop)
      School       → 25th–50th midpoint             (suburban school areas)
      Open Field   → minimum observed density       (rural / undeveloped)
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                            "uscitypopdensity.csv")
    if not _pd_available or not os.path.exists(csv_path):
        return _DENSITY_FALLBACK.copy()

    try:
        df  = pd.read_csv(csv_path)
        col = "Population Density (Persons/Square Mile)"
        d   = df[col].dropna().astype(float)

        p25  = int(d.quantile(0.25))
        p50  = int(d.quantile(0.50))
        p75  = int(d.quantile(0.75))
        dmin = max(int(d.min()), 50)          # floor at 50 to avoid zeros

        densities = {
            "Residential": p75,
            "Commercial":  p50,
            "Industrial":  p25,
            "Hospital":    max(p25 // 2, 50),
            "School":      (p25 + p50) // 2,
            "Open Field":  dmin,
        }
        return densities

    except Exception:
        return _DENSITY_FALLBACK.copy()


# Computed once at import — uses real data when available, fallback otherwise
zone_densities = _load_zone_densities()

ZONE_COLORS    = zone_colors
ZONE_DENSITIES = zone_densities


@dataclass
class Cell:
    row: int
    col: int
    zone: str
    density: int = 0
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0


_layout = [
    ["Open Field","Open Field","Residential","Residential","Residential","Residential","Open Field","Open Field","Open Field","Open Field"],
    ["Open Field","Residential","Residential","Commercial","Commercial","Residential","Residential","Open Field","Open Field","Open Field"],
    ["Residential","Residential","Commercial","Commercial","Commercial","Commercial","Residential","Residential","Open Field","Open Field"],
    ["Open Field","Commercial","Commercial","Industrial","Industrial","Commercial","Commercial","Residential","Residential","Open Field"],
    ["Open Field","Residential","Commercial","Industrial","Open Field","Commercial","Residential","Residential","Open Field","Open Field"],
    ["Hospital","Residential","Residential","Commercial","Commercial","Residential","Residential","School","Open Field","Open Field"],
    ["Open Field","Residential","Residential","Residential","Commercial","Commercial","Residential","Residential","Open Field","Open Field"],
    ["Open Field","Open Field","Residential","Residential","Commercial","Residential","Residential","Residential","Open Field","Open Field"],
    ["Open Field","Open Field","Open Field","Residential","Residential","Residential","Commercial","Open Field","Open Field","Open Field"],
    ["Open Field","Open Field","Open Field","Open Field","Residential","Residential","Open Field","Open Field","Open Field","Open Field"],
]

_hub_positions      = [(1, 4), (5, 3), (7, 5)]
_charging_positions = [(1, 5), (5, 4), (7, 4)]
_medical_positions  = [(6, 0)]


def create_grid(seed=42):
    random.seed(seed)
    grid = [
        [Cell(row=r, col=c, zone=_layout[r][c], density=zone_densities.get(_layout[r][c], 200))
         for c in range(10)]
        for r in range(10)
    ]
    for r, c in _hub_positions:
        grid[r][c].is_hub = True
    for r, c in _charging_positions:
        grid[r][c].is_charging = True
    for r, c in _medical_positions:
        grid[r][c].is_medical_pickup = True
    for r in range(10):
        for c in range(10):
            base = grid[r][c].density / 1000.0
            grid[r][c].demand = round(max(0.0, base + random.uniform(-0.3, 0.3)), 2)
    return grid


def get_cells_by(grid, **flags):
    result = []
    for row in grid:
        for cell in row:
            if all(getattr(cell, k) == v for k, v in flags.items()):
                result.append(cell)
    return result


def print_grid_summary(grid):
    hubs     = [(c.row, c.col) for c in get_cells_by(grid, is_hub=True)]
    charging = [(c.row, c.col) for c in get_cells_by(grid, is_charging=True)]
    medical  = [(c.row, c.col) for c in get_cells_by(grid, is_medical_pickup=True)]
    hospitals= [(c.row, c.col) for c in get_cells_by(grid, zone="Hospital")]
    no_fly   = [(c.row, c.col) for c in get_cells_by(grid, no_fly=True)]
    print(f"  Hubs           : {hubs}")
    print(f"  Charging pads  : {charging}")
    print(f"  Medical pickups: {medical}")
    print(f"  Hospitals      : {hospitals}")
    print(f"  No-fly cells   : {no_fly}")
