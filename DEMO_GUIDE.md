# AeroNet Lite — Complete Demo Guide
### Every line of code explained. Zero jargon. Nothing skipped.

---

## BEFORE YOU START — What Is AeroNet Lite?

Imagine a small city made up of a 10×10 grid of squares (like a chessboard — 100 squares total). Each square is a neighbourhood: some are Residential areas, some are Commercial shops, some are Industrial factories, one is a Hospital, one is a School, and some are Open Fields.

We have 7 delivery drones that pick up packages from shops and deliver them to houses. The drones need to:
1. Follow safety rules (do not fly near hazardous zones)
2. Be selected smartly (how many small vs large drones to buy)
3. Find the shortest path through the city
4. Actually fly the routes, handle emergencies, and re-route
5. Predict demand and detect problems using machine learning

The dashboard is a live visual of everything happening.

---

## HOW TO SPLIT THE DEMO (5 People, ~25 Minutes Total)

| Person | Topic | Files | Time |
|--------|-------|-------|------|
| **Person 1** | City grid + Safety rules | module_0.py + module_1.py | ~5 min |
| **Person 2** | Fleet selection (Genetic Algorithm) | module_2.py | ~5 min |
| **Person 3** | Pathfinding + Drone simulation | module_3.py + module_4.py | ~5 min |
| **Person 4** | Machine Learning (demand + anomaly) | module_5.py | ~5 min |
| **Person 5** | Dashboard — every panel explained | dashboard.html | ~5 min |

---

---

# PERSON 1 — City Grid + Safety Rules
## Files: `src/module_0.py` and `src/module_1.py`
## Time: ~5 minutes

---

### WHAT THIS DOES IN PLAIN ENGLISH

Person 1 built the "city map." Before any drone flies, we need to know what the city looks like — which squares are residential houses, which are shops, where the hospitals are, where the drones park (hubs), and so on. Person 1 also built a "safety checker" that verifies the city layout follows 4 rules (e.g. factories must not be next to hospitals).

---

### MODULE 0 — `module_0.py` — THE CITY GRID

#### Line 1: Imports

```python
from dataclasses import dataclass
import os
import random
```

- `dataclass` — Python shortcut for making a simple data container (like a record card). We use it to store info about each grid square.
- `os` — lets Python interact with your file system (read files, check if they exist).
- `random` — generates random numbers. Used to add small random variation to demand values.

---

#### Lines 11–20: Zone types and colours

```python
zone_types = ["Residential", "Commercial", "Industrial", "Hospital", "School", "Open Field"]

zone_colors = {
    "Residential": "#90EE90",   # light green
    "Commercial":  "#FFD700",   # gold/yellow
    "Industrial":  "#A9A9A9",   # grey
    "Hospital":    "#FF6B6B",   # red
    "School":      "#87CEEB",   # sky blue
    "Open Field":  "#F5F5DC",   # beige
}
```

These are just two lookup tables. `zone_types` lists all 6 zone categories. `zone_colors` maps each zone name to a hex colour code — these exact colours appear in the dashboard grid squares.

---

#### Lines 22–30: Fallback population densities

```python
_DENSITY_FALLBACK = {
    "Residential": 5000,
    "Commercial":  3000,
    "Industrial":  800,
    "Hospital":    500,
    "School":      1200,
    "Open Field":  200,
}
```

These numbers are "people per square mile" for each zone type. If the real dataset file is missing, we use these hardcoded numbers instead. Residential is most dense (5000), Open Field is least (200). The underscore `_` at the start of `_DENSITY_FALLBACK` is a Python convention meaning "this is for internal use."

---

#### Lines 33–72: Loading real population data

```python
def _load_zone_densities():
```

This function tries to read a real US cities population CSV file. Here is what it does, step by step:

```python
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "uscitypopdensity.csv")
```
Builds the file path to `data/raw/uscitypopdensity.csv`. `__file__` is the current Python file. `..` means "go up one folder."

```python
    if not _pd_available or not os.path.exists(csv_path):
        return _DENSITY_FALLBACK.copy()
```
If pandas is not installed OR the file does not exist, immediately return the hardcoded fallback numbers. This prevents crashes.

```python
    df  = pd.read_csv(csv_path)
    col = "Population Density (Persons/Square Mile)"
    d   = df[col].dropna().astype(float)
```
Read the CSV into a table (`df`). Grab the density column. Remove any blank rows (`.dropna()`). Convert to decimal numbers (`.astype(float)`).

```python
    p25  = int(d.quantile(0.25))
    p50  = int(d.quantile(0.50))
    p75  = int(d.quantile(0.75))
    dmin = max(int(d.min()), 50)
```
- `quantile(0.25)` = the 25th percentile — 25% of cities are less dense than this number.
- `quantile(0.50)` = the median — the middle value.
- `quantile(0.75)` = 75th percentile — 75% of cities are less dense.
- `dmin` = the lowest density city in the dataset (minimum 50 so we never get zero).

With the real data these come out as approximately: p25=2076, p50=3128, p75=4720.

```python
    densities = {
        "Residential": p75,       # most dense — 4720
        "Commercial":  p50,       # medium — 3128
        "Industrial":  p25,       # sparse — 2076
        "Hospital":    p25 // 2,  # campus-like — 1038
        "School":      (p25+p50)//2, # suburban — 2602
        "Open Field":  dmin,      # rural — 172
    }
```
Maps percentiles to zone types logically: residential neighbourhoods are the densest, open fields the least.

---

#### Lines 82–93: The Cell data structure

```python
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
```

Think of `Cell` as a record card for one square on the grid. It stores:
- `row`, `col` — its position (like coordinates, e.g. row=3, col=5)
- `zone` — what type of area it is
- `density` — how many people live/work there
- `is_hub` — True if drones park here
- `is_charging` — True if drones recharge here
- `is_medical_pickup` — True if medical packages are collected here
- `no_fly` — True if drones are banned from this square
- `demand` — how much delivery demand exists here (0–10 scale)

The `@dataclass` decorator automatically creates an `__init__` method so you can write `Cell(row=0, col=0, zone="Residential")` without manually writing the constructor.

---

#### Lines 95–110: The city layout

```python
_layout = [
    ["Open Field","Open Field","Residential",...],
    ...
]
```

This is a hand-crafted 10×10 list of lists. `_layout[r][c]` gives the zone type for the square at row `r`, column `c`. This is the "blueprint" of the city.

```python
_hub_positions      = [(1, 4), (5, 3), (7, 5)]   # 3 drone hubs
_charging_positions = [(1, 5), (5, 4), (7, 4)]   # 3 charging pads
_medical_positions  = [(6, 0)]                     # 1 medical pickup
```

These hardcode exactly which squares are special infrastructure.

---

#### Lines 113–130: Building the grid

```python
def create_grid(seed=42):
    random.seed(seed)
```
`seed=42` makes the random numbers repeatable — every time you run the program, you get exactly the same grid. If you change the seed, you'd get different random demand values.

```python
    grid = [
        [Cell(row=r, col=c, zone=_layout[r][c], density=zone_densities.get(_layout[r][c], 200))
         for c in range(10)]
        for r in range(10)
    ]
```
This creates a 10×10 list of Cell objects. For each row `r` (0–9) and column `c` (0–9), it creates a Cell with:
- its zone from `_layout[r][c]`
- its density looked up from `zone_densities` (defaulting to 200 if the zone type is not in the dict)

```python
    for r, c in _hub_positions:
        grid[r][c].is_hub = True
    for r, c in _charging_positions:
        grid[r][c].is_charging = True
    for r, c in _medical_positions:
        grid[r][c].is_medical_pickup = True
```
Marks the special squares using our hardcoded position lists.

```python
    for r in range(10):
        for c in range(10):
            base = grid[r][c].density / 1000.0
            grid[r][c].demand = round(max(0.0, base + random.uniform(-0.3, 0.3)), 2)
```
Sets the initial demand for each square:
- `base = density / 1000.0` — converts density to a small number (e.g. 4720 becomes 4.72)
- `random.uniform(-0.3, 0.3)` — adds a small random nudge
- `max(0.0, ...)` — ensures demand never goes negative
- `round(..., 2)` — round to 2 decimal places

---

#### Lines 133–139: Helper functions

```python
def get_cells_by(grid, **flags):
    result = []
    for row in grid:
        for cell in row:
            if all(getattr(cell, k) == v for k, v in flags.items()):
                result.append(cell)
    return result
```
A utility that finds all cells matching certain criteria. `**flags` means keyword arguments. Example: `get_cells_by(grid, is_hub=True)` returns all hub cells. It loops every cell and checks if all the requested attributes match.

---

### WHAT YOU SEE ON THE DASHBOARD (from Module 0)

- The **10×10 coloured grid** — each square's colour comes from `zone_colors`
- The **H markers** (hub squares), **lightning bolt** (charging), **medical cross** icons
- The **demand numbers** in each cell come from `cell.demand`
- The **Population Density panel** on the right shows the real values loaded from CSV

---

### MODULE 1 — `module_1.py` — SAFETY RULES (CSP)

CSP stands for "Constraint Satisfaction Problem." In plain English: we have rules that the city layout MUST follow. This module checks all 4 rules.

---

#### Lines 4–12: get_neighbors

```python
def get_neighbors(row, col, grid=None):
    coords = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 10 and 0 <= nc < 10:
            coords.append((nr, nc))
```

For a given square (row, col), this finds the 4 squares directly touching it (up, down, left, right). The loop uses direction offsets: `(-1,0)` = up one row, `(1,0)` = down, `(0,-1)` = left, `(0,1)` = right. The `if` check ensures we don't go outside the 10×10 boundary.

If `grid` is provided, it returns the actual Cell objects; otherwise just coordinates.

---

#### Lines 15–16: manhattan

```python
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
```

"Manhattan distance" is like measuring distance in a city where you can only move in straight lines (no diagonals), just like walking city blocks. `abs` means "absolute value" (ignore negative sign). Example: from (2,3) to (5,7) = |2-5| + |3-7| = 3 + 4 = 7 steps.

---

#### Lines 19–29: Rule 1 — Industrial Safety

```python
def check_industrial_safety(grid):
    errors = []
    for row in grid:
        for cell in row:
            if cell.zone == "Industrial":
                for nb in get_neighbors(cell.row, cell.col, grid):
                    if nb.zone in ("School", "Hospital"):
                        errors.append(
                            f"R1 FAIL: Industrial at ({cell.row},{cell.col}) "
                            f"is adjacent to {nb.zone} at ({nb.row},{nb.col})"
                        )
    return errors
```

Rule 1: Factories cannot be next to hospitals or schools (safety concern). 

How it works:
1. Loop through every cell
2. If a cell is "Industrial", get its 4 neighbours
3. If any neighbour is "School" or "Hospital", add an error message
4. Return the list of errors (empty list = all good)

---

#### Lines 33–48: Rule 2 — Residential Coverage

```python
def check_residential_coverage(grid):
    errors = []
    hubs = [(c.row, c.col) for row in grid for c in row if c.is_hub]
    for row in grid:
        for cell in row:
            if cell.zone == "Residential":
                pos = (cell.row, cell.col)
                if not any(manhattan(pos, h) <= 3 for h in hubs):
                    ...errors.append(...)
    return errors
```

Rule 2: Every residential square must have a drone hub within 3 steps (Manhattan distance ≤ 3).

How it works:
1. First, collect all hub positions into a list called `hubs`
2. Loop through every residential cell
3. Check if ANY hub is within distance 3 from this cell — `any(manhattan(pos, h) <= 3 for h in hubs)`
4. If none are close enough, add an error with a helpful suggestion of where to add a hub

---

#### Lines 52–62: Rule 3 — Hub Charging Access

```python
def check_hub_charging(grid):
    errors = []
    charging_pads = [(c.row, c.col) for row in grid for c in row if c.is_charging]
    for row in grid:
        for cell in row:
            if cell.is_hub:
                pos = (cell.row, cell.col)
                if not any(manhattan(pos, cp) <= 2 for cp in charging_pads):
                    errors.append(...)
    return errors
```

Rule 3: Every hub must have a charging pad within 2 steps. Drones must be able to recharge near where they park.

Same pattern as Rule 2 but checking `is_charging` instead of `is_hub`, and using distance ≤ 2.

---

#### Lines 66–81: Rule 4 — Medical Access

```python
def check_medical_access(grid):
    medical_pickups = [(c.row, c.col) for row in grid for c in row if c.is_medical_pickup]
    hospitals = [c for row in grid for c in row if c.zone == "Hospital"]
    covered = any(
        any(manhattan((h.row, h.col), mp) <= 1 for mp in medical_pickups)
        for h in hospitals
    )
    if not covered:
        errors.append(...)
```

Rule 4: At least one hospital must have a medical pickup point within 1 step (directly adjacent).

The double `any(... for h in hospitals)` means: "for at least ONE hospital, is there at least ONE medical pickup within distance 1?"

---

#### Lines 85–101: validate_layout — The Master Checker

```python
def validate_layout(grid):
    rule_results = {
        "R1": check_industrial_safety(grid),
        "R2": check_residential_coverage(grid),
        "R3": check_hub_charging(grid),
        "R4": check_medical_access(grid),
    }
    passed = [r for r, errs in rule_results.items() if not errs]
    failed = [r for r, errs in rule_results.items() if errs]
    errors = [e for errs in rule_results.values() for e in errs]
    return {
        "passed":  passed,
        "failed":  failed,
        "errors":  errors,
        "valid":   len(failed) == 0,
        "details": rule_results,
    }
```

Runs all 4 checks and packages the results into one dictionary:
- `passed` = list of rules with no errors (e.g. `["R1","R3","R4"]`)
- `failed` = list of rules that have violations (e.g. `["R2"]`)
- `valid` = True only if ALL 4 rules pass (`len(failed) == 0`)

**Our actual results: R1, R2, R3, R4 all PASS.** The city layout was designed to satisfy all constraints.

---

### WHAT YOU SEE ON THE DASHBOARD (from Module 1)

- The **CSP Constraint Checker panel** — shows R1/R2/R3/R4 with green ticks or red crosses
- The **constraint descriptions** next to each rule
- The note "Layout valid: True" at the bottom

---

---

# PERSON 2 — Fleet Selection (Genetic Algorithm)
## File: `src/module_2.py`
## Time: ~5 minutes

---

### WHAT THIS DOES IN PLAIN ENGLISH

We need to decide: how many small drones (Light, carry 2kg) and how many large drones (Heavy, carry 5kg) should we buy? We have a budget of $10,000. 

A "Genetic Algorithm" is inspired by natural evolution. It:
1. Creates 30 random combinations (population) — e.g. "4 light + 3 heavy", "6 light + 1 heavy", etc.
2. Scores each combination (better coverage + lower cost = higher score)
3. Keeps the best ones, combines them (crossover), adds small tweaks (mutation)
4. Repeats for 60 rounds (generations)
5. The last surviving "fittest" combination is our answer

---

#### Lines 1–16: Constants

```python
default_budget  = 10000
light_cost      = 1000    # $1000 per light drone
heavy_cost      = 1800    # $1800 per heavy drone
light_payload   = 2       # 2kg max cargo
heavy_payload   = 5       # 5kg max cargo
light_range     = 12      # 12 grid cells max distance
heavy_range     = 20      # 20 grid cells max distance
light_daily_cap = 3       # max 3 deliveries per day
heavy_daily_cap = 2       # max 2 deliveries per day
pop_size        = 30      # 30 candidates at once
generations     = 60      # evolve for 60 rounds
mutation_rate   = 0.15    # 15% chance of mutation per candidate
```

These are the "game rules." Light drones are cheaper but carry less. Heavy drones cost more but carry more and fly farther.

---

#### Lines 18–60: compute_score — The Fitness Function

```python
def compute_score(n_light, n_heavy, grid, budget=default_budget):
```

This function answers: "How good is this fleet combination?"

```python
    cost = n_light * light_cost + n_heavy * heavy_cost
    if cost > budget:
        return -999
```
Calculate total cost. If it exceeds the $10,000 budget, immediately return -999 (disqualified).

```python
    residential = [c for row in grid for c in row if c.zone == "Residential"]
```
Get all residential cells — these are the "customers" we need to serve.

```python
    hubs = [(c.row, c.col) for row in grid for c in row if c.is_hub]
    covered = 0
    for cell in residential:
        pos = (cell.row, cell.col)
        for hub_pos in hubs:
            dist = abs(pos[0]-hub_pos[0]) + abs(pos[1]-hub_pos[1])
            if dist <= light_range or dist <= heavy_range:
                covered += 1
                break
```
Counts how many residential cells are within flying range of at least one hub.

```python
    coverage_pct   = covered / max(len(residential), 1)
    budget_used_pct = cost / budget
    score = 0.75 * coverage_pct - 0.25 * budget_used_pct
    return round(score, 4)
```

**The fitness formula:** `score = 0.75 × coverage% - 0.25 × cost%`

- Coverage is weighted 75% (more important — serve as many houses as possible)
- Cost is weighted 25% (less important — but we still want to save money)
- A score of 1.0 = perfect (100% coverage, zero cost — impossible)
- A score around 0.75–0.85 is excellent

---

#### Lines 62–75: Random Chromosome

```python
def _random_chromosome(budget=default_budget):
    max_light = budget // light_cost     # max 10 light drones
    max_heavy = budget // heavy_cost     # max 5 heavy drones
    n_light = random.randint(0, max_light)
    n_heavy = random.randint(0, max_heavy)
    while n_light * light_cost + n_heavy * heavy_cost > budget:
        if n_light > 0:
            n_light -= 1
        else:
            n_heavy -= 1
    return [n_light, n_heavy]
```

A "chromosome" = one candidate solution = `[number_of_light, number_of_heavy]`. This function creates a random valid one. The `while` loop reduces counts until the combo fits within budget.

---

#### Lines 77–85: Tournament Selection

```python
def _tournament(pop, scores, k=3):
    candidates = random.sample(range(len(pop)), k)
    best = max(candidates, key=lambda i: scores[i])
    return pop[best]
```

To pick a "parent" for the next generation, we randomly pick 3 candidates and choose the one with the best score. This is "tournament selection" — like a small competition. We run this twice to get 2 parents.

---

#### Lines 87–94: Crossover

```python
def _crossover(p1, p2):
    c1 = [p1[0], p2[1]]
    c2 = [p2[0], p1[1]]
    return c1, c2
```

"Crossover" = breeding. Parent 1 = `[5, 1]`, Parent 2 = `[3, 2]`. Their children are `[5, 2]` and `[3, 1]`. Child 1 gets Parent 1's light count and Parent 2's heavy count. This mixes good traits from both parents.

---

#### Lines 96–107: Mutation

```python
def _mutate(chrom, budget=default_budget):
    if random.random() >= mutation_rate:
        return chrom
    gene = random.randint(0, 1)
    chrom[gene] += random.choice([-1, 1])
    chrom[gene] = max(0, chrom[gene])
    while chrom[0]*light_cost + chrom[1]*heavy_cost > budget:
        chrom[gene] = max(0, chrom[gene] - 1)
    return chrom
```

With 15% probability, randomly change one number by ±1. This introduces variety — prevents the algorithm from getting "stuck." The `while` loop trims back if the mutation breaks the budget limit.

---

#### Lines 109–148: The Main GA Loop — select_fleet_ga

```python
def select_fleet_ga(grid=None, budget=default_budget, verbose=True):
    pop = [_random_chromosome(budget) for _ in range(pop_size)]
```
Start with 30 random chromosomes.

```python
    for gen in range(generations):
        scores = [compute_score(c[0], c[1], grid, budget) for c in pop]
```
Every generation, score all 30 candidates.

```python
        next_pop = []
        while len(next_pop) < pop_size:
            p1 = _tournament(pop, scores)
            p2 = _tournament(pop, scores)
            c1, c2 = _crossover(p1[:], p2[:])
            next_pop.extend([_mutate(c1[:], budget), _mutate(c2[:], budget)])
        pop = next_pop[:pop_size]
```
Build the next generation: pick 2 parents by tournament, breed them, mutate offspring, repeat until we have 30 new candidates.

```python
    best_idx   = max(range(len(pop)), key=lambda i: scores[i])
    best_chrom = pop[best_idx]
    best_score = scores[best_idx]
    return {
        "light": best_chrom[0],
        "heavy": best_chrom[1],
        "cost":  best_chrom[0]*light_cost + best_chrom[1]*heavy_cost,
        "score": best_score,
        "method": "Genetic Algorithm"
    }
```

After 60 generations, pick the highest-scoring chromosome. **Our result: 5 Light + 2 Heavy, cost = $8,600, score = 0.84.**

---

#### Lines 150–170: Brute Force Alternative

```python
def select_fleet_brute(grid=None, budget=default_budget, verbose=True):
    best_score = -float("inf")
    best = None
    for n_light in range(0, budget//light_cost + 1):
        for n_heavy in range(0, budget//heavy_cost + 1):
            if n_light*light_cost + n_heavy*heavy_cost > budget:
                continue
            score = compute_score(n_light, n_heavy, grid, budget)
            if score > best_score:
                best_score = score
                best = (n_light, n_heavy)
```

Tries every possible combination (e.g. 0 light + 0 heavy, 0 light + 1 heavy, ... up to 10 light + 5 heavy). It's guaranteed to find the best answer but is much slower. The GA is faster and usually finds the same answer.

---

### WHAT YOU SEE ON THE DASHBOARD (from Module 2)

- **Fleet Selection table** — shows 5 Light and 2 Heavy drones, their costs, payload, range
- **GA Convergence Chart** (line chart) — shows how the best score improved across 60 generations — starts low, rises, plateaus
- **Score of 0.84** displayed in the fleet summary
- Total fleet cost: **$8,600 out of $10,000 budget**

---

---

# PERSON 3 — Pathfinding + Drone Simulation
## Files: `src/module_3.py` and `src/module_4.py`
## Time: ~5 minutes

---

### WHAT THIS DOES IN PLAIN ENGLISH

**Module 3 (Pathfinding):** Given a start square and a destination square, find the shortest route through the grid. Drones avoid "no-fly" zones and prefer commercial streets (they're cheaper to traverse — lower air traffic).

**Module 4 (Simulation):** Actually flies the drones. Assigns 8 deliveries to 7 drones, moves them step-by-step, handles a no-fly zone appearing at step 11, and a battery emergency at step 17.

---

### MODULE 3 — `module_3.py` — A* PATHFINDING

#### Lines 1–5: Imports and Manhattan

```python
import heapq

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
```

`heapq` is Python's "priority queue" — a data structure that always gives you the smallest item first. We use this to always explore the most promising grid square next. `manhattan` is the same distance formula as Module 1.

---

#### Lines 8–9: Move Cost

```python
def get_move_cost(cell):
    return 0.8 if cell.zone == "Commercial" else 1.0
```

Moving into a Commercial square costs only 0.8 instead of 1.0. This models real-world logistics: commercial zones have better infrastructure (wider roads/air corridors), so drones prefer flying over them. Everything else costs 1.0.

---

#### Lines 12–18: Reconstruct Path

```python
def reconstruct_path(came_from, current):
    path = []
    node = current
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path
```

`came_from` is a dictionary where each square records "which square did I come from?" After A* finds the goal, we trace backwards using this dictionary — like following breadcrumbs back to the start. Then we `.reverse()` so the path goes start→end.

---

#### Lines 22–52: A* Search — The Main Algorithm

```python
def astar(start, goal, grid):
    if grid[goal[0]][goal[1]].no_fly:
        return None, None
```
Immediately return failure if the destination itself is a no-fly zone.

```python
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_heap  = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0.0, start))
    g_cost    = {start: 0.0}
    came_from = {start: None}
    closed    = set()
```

Setting up the algorithm:
- `directions` = the 4 possible moves (up/down/left/right)
- `open_heap` = squares we still need to explore, sorted by best guess of total cost
- The initial entry is `(estimated_total_cost, actual_cost_so_far, position)` = `(manhattan(start,goal), 0.0, start)`
- `g_cost` = dictionary tracking the best known actual cost to reach each square
- `came_from` = breadcrumb trail
- `closed` = squares we've already fully processed (we won't revisit these)

```python
    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path, round(g, 2)
```

Main loop:
- Pop the square with the lowest estimated total cost (`f`) from the heap
- Skip it if already processed
- Mark it as processed (add to `closed`)
- If it's the goal — done! Reconstruct and return the path

```python
        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 10 and 0 <= nc < 10):
                continue
            neighbor = (nr, nc)
            if grid[nr][nc].no_fly:
                continue
            new_g = g + get_move_cost(grid[nr][nc])
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor]    = new_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (new_g + manhattan(neighbor, goal), new_g, neighbor))
```

For each neighbour of the current square:
1. Skip if outside the grid
2. Skip if it's a no-fly zone
3. Calculate new cost = current cost + move cost of neighbour
4. If this is the best-known route to this neighbour, update records and add to heap
5. The heap entry is `(new_g + manhattan_to_goal, new_g, neighbour)` — this is the "f = g + h" formula of A*

**Why A* is clever:** It doesn't blindly explore in all directions. The heuristic `manhattan(neighbour, goal)` guides it towards the destination, so it finds the path much faster than searching every square.

---

#### Lines 64–105: Plan Delivery Route

```python
def plan_delivery_route(drone, delivery, grid):
```

A complete 3-leg journey:
1. **Hub → Pickup** (drone leaves its base, goes to collect the package)
2. **Pickup → Dropoff** (drone carries the package to the customer)
3. **Dropoff → Hub** (drone returns to base for charging)

```python
    legs = [
        ("hub->pickup",     hub,     pickup),
        ("pickup->dropoff", pickup,  dropoff),
        ("dropoff->hub",    dropoff, hub),
    ]
    segments   = []
    full_path  = []
    total_cost = 0.0
    for label, src, dst in legs:
        res = astar_result(src, dst, grid)
        res["label"] = label
        segments.append(res)
        if not res["found"]:
            return {"success": False, ...}
        if full_path:
            full_path.extend(res["path"][1:])
        else:
            full_path.extend(res["path"])
        total_cost += res["cost"]
```

For each of the 3 legs, run A*. Combine the paths into one continuous `full_path` (the `[1:]` skips the first point of legs 2 and 3 because it's the same as the last point of the previous leg).

---

### MODULE 4 — `module_4.py` — DRONE SIMULATION

#### Lines 7–50: The Drone Class

```python
@dataclass
class Drone:
    drone_id: str           # e.g. "D1"
    drone_type: str         # "light" or "heavy"
    payload_kg: float       # how much it can carry
    max_range: int          # max grid cells it can fly
    hub: tuple              # home base position e.g. (1, 4)
    position: tuple         # current position (starts at hub)
    battery: float = 100.0  # 100% battery at start
    status: str = "idle"    # "idle", "en_route", "returning", "failed"
    current_delivery: Optional[str] = None  # which delivery it's working on
    route: list = field(default_factory=list)  # list of grid positions in its path
    route_index: int = 0    # which step along the route it's currently at
    battery_drain_per_step: float = 4.0
```

Every drone is one of these objects. `field(default_factory=list)` creates a new empty list for each drone — important because Python would share one list between all drones if you wrote `route=[]` directly.

```python
    def __post_init__(self):
        self.position = self.hub
```
`__post_init__` runs automatically after the dataclass `__init__`. Sets starting position = hub.

```python
    def advance(self):
        if self.route_index + 1 < len(self.route):
            self.route_index += 1
            self.position = self.route[self.route_index]
            self.battery  = max(0.0, self.battery - self.battery_drain_per_step)
```
Moves the drone one step forward along its route. Drains battery by `battery_drain_per_step` (4% for light, 3.5% for heavy). Battery never goes below 0.

---

#### Lines 52–60: The Delivery Class

```python
@dataclass
class Delivery:
    delivery_id: str        # e.g. "DEL_01"
    pickup: tuple           # where to collect from
    dropoff: tuple          # where to deliver to
    payload_kg: float       # package weight
    status: str = "pending" # "pending", "assigned", "completed", "failed", "delayed"
    assigned_drone: Optional[str] = None  # which drone got this job
```

---

#### Lines 70–84: build_fleet

```python
def build_fleet(fleet_result, grid):
    hubs = [(c.row, c.col) for row in grid for c in row if c.is_hub]
    drones = []
    drone_num = 1
    for i in range(fleet_result.get("light", 0)):
        drones.append(Drone(drone_id=f"D{drone_num}", drone_type="light",
                            payload_kg=2.0, max_range=12,
                            hub=hubs[i % len(hubs)], battery_drain_per_step=5.0))
        drone_num += 1
    for i in range(fleet_result.get("heavy", 0)):
        drones.append(Drone(..., battery_drain_per_step=3.5))
        drone_num += 1
    return drones
```

Creates the actual drone objects from the GA's fleet decision. `hubs[i % len(hubs)]` distributes drones across hubs using modulo — drone 1 goes to hub 0, drone 2 to hub 1, drone 3 to hub 2, drone 4 back to hub 0, etc. Note: light drones drain 5% per step, heavy drones drain 3.5% (heavier but more efficient motors).

---

#### Lines 87–102: generate_deliveries

```python
def generate_deliveries(grid, count=8, seed=7):
    import random
    random.seed(seed)
    residential = [...]  # all residential cells
    commercial  = [...]  # all commercial cells
    medical     = [...]  # medical pickup cells
    deliveries  = []
    for i in range(1, count + 1):
        pickup  = random.choice(commercial + medical)
        dropoff = random.choice(residential)
        while dropoff == pickup:
            dropoff = random.choice(residential)
        payload = random.choice([1.5, 2.0, 3.0, 4.5])
        deliveries.append(Delivery(...))
    return deliveries
```

Creates 8 random deliveries. Packages are picked up from commercial/medical locations and delivered to residential homes. `seed=7` makes it repeatable.

---

#### Lines 105–132: assign_deliveries

```python
def assign_deliveries(deliveries, drones, grid, log):
    idle_drones = [d for d in drones if d.status == "idle" and d.battery > 20]
    for delivery in deliveries:
        if delivery.status != "pending":
            continue
        if not idle_drones:
            break
        drone = min(idle_drones,
                    key=lambda d: abs(d.position[0] - delivery.pickup[0]) +
                                  abs(d.position[1] - delivery.pickup[1]))
```

For each unassigned delivery, pick the closest available idle drone (with battery > 20%). `min(..., key=...)` finds the drone with the smallest Manhattan distance to the pickup point.

```python
        if drone.payload_kg < delivery.payload_kg:
            log.append(...)
            continue
```
Skip if the drone can't carry this package.

```python
        result = plan_delivery_route(drone, delivery, grid)
        if result["success"]:
            drone.route       = result["full_path"]
            drone.route_index = 0
            drone.status      = "en_route"
            drone.current_delivery = delivery.delivery_id
            delivery.status   = "assigned"
```
Plan the full 3-leg route and assign it to the drone. The drone's `route` is now the full list of grid positions it will visit.

---

#### Lines 159–171: No-Fly Zone Activation

```python
def activate_nofly(row, col, grid, drones=None):
    grid[row][col].no_fly = True
    if drones is None:
        return []
    return [d.drone_id for d in drones
            if d.status == "en_route" and any(pos == (row, col) for pos in d.route[d.route_index:])]
```

Marks a cell as no-fly. Returns list of drone IDs whose remaining routes pass through that cell — these drones need rerouting.

---

#### Lines 185–197: reroute_drone

```python
def reroute_drone(drone, grid):
    if not drone.route:
        return f"  {drone.drone_id}: no active route to reroute."
    goal = drone.route[-1]
    path, cost = astar(drone.position, goal, grid)
    if path is not None:
        drone.route       = [drone.position] + path[1:]
        drone.route_index = 0
        return (f"  Drone {drone.drone_id} rerouted via A* ...")
    drone.status = "failed"
    return (f"  Drone {drone.drone_id} cannot reach destination ...")
```

When a no-fly zone blocks a drone's route, re-run A* from the drone's current position to its final destination (end of its original route). If a new path exists, update the route. If no path exists, mark the drone as "failed."

---

#### Lines 254–268: inject_anomaly

```python
def inject_anomaly(drone, anomaly_type, log):
    if anomaly_type == "battery":
        drone.battery = max(0.0, drone.battery - 40.0)
        if drone.battery < 20:
            drone.status = "returning"
    elif anomaly_type == "route":
        log.append(...)
    elif anomaly_type == "sensor":
        log.append(...)
```

Simulates emergencies:
- **Battery anomaly** — drops battery by 40%. If below 20%, drone returns to hub immediately.
- **Route anomaly** — logs a deviation warning.
- **Sensor anomaly** — logs an altitude/speed spike.

In the simulation, at step 11 a no-fly zone appears at (4,4) and Drone 3 reroutes. At step 17, Drone 5 gets a battery anomaly.

---

#### Lines 270–276: simulation_summary

```python
def simulation_summary(deliveries):
    return {
        "completed": sum(1 for d in deliveries if d.status == "completed"),
        "delayed":   sum(1 for d in deliveries if d.status == "delayed"),
        "failed":    sum(1 for d in deliveries if d.status == "failed"),
        "pending":   sum(1 for d in deliveries if d.status in ("pending", "assigned")),
    }
```

At the end of all 20 simulation steps, counts how many deliveries succeeded, got delayed, failed, or are still in progress.

---

### WHAT YOU SEE ON THE DASHBOARD (from Modules 3 + 4)

- **Drone position dots** moving across the grid each step
- **Coloured path trails** — each drone has a unique colour showing its full route
- **Hover on a path** → tooltip showing drone ID, delivery, battery, status
- **Direction arrows** on paths showing which way the drone is travelling
- **Simulation log** panel (right side) — live text messages at each step
- **Step counter** and progress bar at top
- **Health Alerts panel** — shows battery warnings, rerouting events
- **Delivery counters** — completed / delayed / failed

---

---

# PERSON 4 — Machine Learning
## File: `src/module_5.py`
## Time: ~5 minutes

---

### WHAT THIS DOES IN PLAIN ENGLISH

Module 5 uses two real machine learning models:

1. **Demand Forecasting** — Predict how many deliveries each city zone needs. We use a real Kaggle dataset (Bike Sharing — 10,886 records) as a proxy for delivery demand. We train two models (Linear Regression and Random Forest) and compare them.

2. **Anomaly Detection** — Detect when a drone is behaving strangely. We generate synthetic flight data with 4 categories: Normal, Battery Anomaly, Route Anomaly, Sensor Spike. We train 4 classifiers and compare accuracy.

---

### PART A — DEMAND FORECASTING

#### Lines 14–16: File paths

```python
_base_dir     = os.path.dirname(__file__)
processed_dir = os.path.join(_base_dir, "..", "data", "processed")
raw_dir       = os.path.join(_base_dir, "..", "data", "raw")
```

`__file__` is the path to `module_5.py`. Everything is relative to that. `processed_dir` is where we save our cleaned data after loading.

---

#### Lines 22–34: Features and zone feature rows

```python
DEMAND_FEATURES = ["season", "holiday", "workingday", "weather",
                   "temp", "humidity", "windspeed", "hour", "month", "weekday"]
```

These are the 10 "inputs" we feed to the model. Each one is a number: season (1-4), holiday (0 or 1), temperature (Celsius), etc.

```python
_ZONE_FEATURE_ROWS = {
    "Residential": [2, 0, 1, 1, 20.0, 55.0, 12.0, 18, 6, 0],
    "Commercial":  [2, 0, 1, 1, 22.0, 50.0, 10.0, 14, 6, 1],
    ...
}
```

Each zone type is mapped to a "typical" scenario (summer weekday, typical weather). When we want to predict demand for each grid cell, we feed these representative feature vectors into the trained model.

---

#### Lines 37–54: _load_demand_data

```python
def _load_demand_data():
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
    return _generate_demand_data(), False
```

Step by step:
1. Check if `data/raw/train.csv` exists
2. Read it into a pandas DataFrame (like an Excel table in Python)
3. Parse the "datetime" column into a proper date object
4. Extract hour/month/weekday from the date (the raw CSV only has one "datetime" column)
5. Keep only the 10 feature columns + "count" (the target — number of bike rentals)
6. Drop any rows with missing values
7. Return the data AND `True` (means "this is real data")

If the file is missing, call the synthetic fallback and return `False`.

---

#### Lines 83–150: run_demand_forecast

```python
def run_demand_forecast(grid=None, verbose=True):
    df, is_real = _load_demand_data()
    df.to_csv(os.path.join(processed_dir, "demand_data.csv"), index=False)
```
Load data, save processed version to disk.

```python
    x = df[DEMAND_FEATURES].values
    y = df["count"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
- `x` = the 10 feature columns (inputs)
- `y` = the "count" column (what we're trying to predict)
- `train_test_split(..., test_size=0.2)` — 80% of data for training, 20% held back for testing
- `random_state=42` — same split every time (reproducible)

```python
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    lr_mae  = round(mean_absolute_error(y_test, lr_pred), 3)
    lr_rmse = round(float(np.sqrt(mean_squared_error(y_test, lr_pred))), 3)
    lr_r2   = round(float(1 - np.sum((y_test - lr_pred)**2) /
                          np.sum((y_test - y_test.mean())**2)), 3)
```

**Linear Regression:** Finds the best straight-line formula connecting features to count. Trains in milliseconds but oversimplifies.
- **MAE** (Mean Absolute Error) — on average, predictions are off by this many bike rentals
- **RMSE** (Root Mean Squared Error) — like MAE but penalises big errors more
- **R²** — how much of the variation in count the model explains (0 = terrible, 1 = perfect)

**Our Linear Regression results:** MAE=107.656, RMSE=147.233, R²=0.343 (poor — too simple)

```python
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
```

**Random Forest:** 100 decision trees, each trained on a random subset of data. Each tree makes its own prediction; the forest averages them. Much more powerful than linear regression.

**Our Random Forest results:** MAE=44.074, RMSE=65.988, R²=0.868 (excellent — explains 87% of variation)

```python
    feat_arr = np.array([
        _ZONE_FEATURE_ROWS.get(cell.zone, _ZONE_FEATURE_ROWS["Open Field"])
        for row in grid for cell in row
    ])
    raw    = rf.predict(feat_arr)
    scaled = (raw / max(raw.max(), 1) * 10).reshape(10, 10)
    for r in range(10):
        for c in range(10):
            grid[r][c].demand = round(float(scaled[r, c]), 2)
```

After training, predict demand for all 100 grid cells using their zone feature vectors. Scale predictions to 0–10 range and store in `cell.demand`. This updates the heatmap on the dashboard.

---

### PART B — ANOMALY DETECTION

#### Lines 156–186: _generate_anomaly_data

```python
def _generate_anomaly_data(n=800, seed=1):
    rng = np.random.default_rng(seed)
    records = []
    per_class = n // 4   # 200 records per class
    for label in range(4):
        for _ in range(per_class):
            if label == 0:    # Normal
                rec = dict(battery_drop=rng.uniform(1.0, 2.5), ...)
            elif label == 1:  # Battery Anomaly
                rec = dict(battery_drop=rng.uniform(5.1, 15.0), ...)
            elif label == 2:  # Route Anomaly
                rec = dict(route_deviation=rng.uniform(3.1, 10.0), ...)
            else:             # Sensor Spike
                rec = dict(altitude_change=rng.uniform(20.0, 50.0),
                           speed_change=rng.uniform(20.0, 60.0), ...)
```

Generates 800 fake flight records (200 per category). Each record has 4 features:
- `battery_drop` — how much battery dropped in one step
- `route_deviation` — how far off course the drone went
- `altitude_change` — how much altitude changed
- `speed_change` — how much speed changed

The 4 classes are designed to have distinct patterns:
- Normal: small values in all features
- Battery Anomaly: high `battery_drop` (5.1–15%)
- Route Anomaly: high `route_deviation` (3.1–10 cells)
- Sensor Spike: extreme `altitude_change` and `speed_change`

---

#### Lines 189–232: train_anomaly_model

Four classifiers compared:

```python
dt  = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(x_train, y_train)
dt_acc = round(accuracy_score(y_test, dt.predict(x_test)), 4)
```
**Decision Tree** — like a flowchart of yes/no questions (e.g. "battery_drop > 5? → likely Battery Anomaly"). Limited to 6 levels deep to avoid overfitting. Accuracy: **91.75%**

```python
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_acc = round(accuracy_score(y_test, rf_pred), 4)
```
**Random Forest Classifier** — 100 decision trees vote on the class. Best performer. Accuracy: **96.25%**

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_acc = round(accuracy_score(y_test, knn.predict(x_test)), 4)
```
**KNN (K-Nearest Neighbours)** — looks at the 5 most similar records and takes a majority vote. Simple but effective. Accuracy: **89.50%**

```python
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_acc = round(accuracy_score(y_test, gnb.predict(x_test)), 4)
```
**Gaussian Naive Bayes** — uses probability theory (Bayes' theorem), assumes features are independent. Fastest to train. Accuracy: **85.25%**

---

#### Lines 235–249: classify_telemetry / classify_drone_telemetry

```python
def classify_telemetry(model, drone_state):
    features = np.array([[
        drone_state.get("battery_drop",    0.0),
        drone_state.get("route_deviation", 0.0),
        drone_state.get("altitude_change", 0.0),
        drone_state.get("speed_change",    0.0),
    ]])
    return anomaly_labels[model.predict(features)[0]]
```

Given a live drone's current telemetry (4 numbers), feed them into the trained model and get back a label: "Normal", "Battery Anomaly", "Route Anomaly", or "Sensor Spike."

`model.predict(features)[0]` returns an integer (0–3). `anomaly_labels[...]` converts that integer to the human-readable label string.

---

### WHAT YOU SEE ON THE DASHBOARD (from Module 5)

- **Demand Heatmap** — appears at step 15, showing demand per cell as colour intensity
- **Demand Forecast chart** — bar chart comparing LR vs RF with MAE/RMSE/R² values
- **ML Metrics table** — side-by-side comparison of both models
- **Anomaly Detection table** — shows accuracy of all 4 classifiers
- **Health Alerts** — real-time anomaly labels on active drones
- **Heatmap legend** — navy=low demand, bright cyan=high demand

---

---

# PERSON 5 — The Dashboard
## File: `dashboard.html`
## Time: ~5 minutes

---

### WHAT THIS DOES IN PLAIN ENGLISH

The dashboard is a single HTML file that visualises everything the Python modules compute. It is entirely self-contained — no server needed, just open in a browser. It has a city grid, charts, tables, controls, and a simulation that animates step by step.

---

### OVERALL STRUCTURE

The dashboard uses:
- **HTML** — the skeleton (elements, text, layout)
- **CSS** (in `<style>` tags) — colours, fonts, grid layout
- **JavaScript** (in `<script>` tags) — all the logic, charts, animation
- **Chart.js** — library for drawing donut charts, bar charts, line charts
- **SVG** — scalable vector graphics, drawn on top of the grid to show drone paths

The page is divided into panels:
1. **Header** — title + simulation controls (Pause/Resume, Reset, Inject No-Fly, Inject Anomaly)
2. **Left column** — City Grid (with SVG path overlay) + Demand Heatmap
3. **Middle column** — Module panels (CSP, Fleet, Delivery Status, GA Chart, Demand Chart)
4. **Right column** — Simulation Log + Health Alerts + ML Tables + Summary Panel

---

### THE CITY GRID

```javascript
const ZONE_COLORS = {
    Residential: "#90EE90",
    Commercial:  "#FFD700",
    ...
};
const layout = [
    ["Open Field","Open Field","Residential",...],
    ...
];
```

JavaScript mirrors the Python `_layout` and `zone_colors`. The grid is drawn as 100 `<div>` elements, each coloured based on its zone type.

```javascript
function buildGrid() {
    const g = document.getElementById("city-grid");
    for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
            const div = document.createElement("div");
            div.className = "cell";
            div.style.background = ZONE_COLORS[layout[r][c]];
            div.id = `cell-${r}-${c}`;
            ...
            g.appendChild(div);
        }
    }
}
```
Creates 100 divs, assigns colour and ID, and appends to the grid container.

---

### THE SVG PATH OVERLAY

```javascript
const svg = document.getElementById("path-svg");
svg.setAttribute("viewBox", "0 0 10 10");
svg.setAttribute("preserveAspectRatio", "none");
```

The SVG sits on top of the grid. `viewBox="0 0 10 10"` means the coordinate system goes from 0 to 10 in both directions, perfectly matching the grid. Cell (row, col) has its centre at `(col + 0.5, row + 0.5)` in SVG coordinates.

```javascript
function updateDronePaths(step) {
    svg.innerHTML = "";  // clear old paths
    // Add arrow markers to SVG defs...
    const defs = document.createElementNS(..., "defs");
    ...
    drones.forEach(d => {
        if (!d.path || d.path.length < 2) return;
        const points = d.path.map(([r,c]) => `${c+0.5},${r+0.5}`).join(" ");
        // Draw glow line (blurry, wide)
        const glow = mkPoly(points, d.color, 0.35, 0.15, "none");
        glow.style.filter = "blur(1px)";
        svg.appendChild(glow);
        // Draw main coloured path
        const mainLine = mkPoly(points, d.color, 0.12, 0.7, "none");
        mainLine.setAttribute("marker-end", `url(#arrow-${d.id})`);
        svg.appendChild(mainLine);
        // Draw invisible wide hit area for hover
        const hitArea = mkPoly(points, "transparent", 0.7, 0, "stroke");
        hitArea.addEventListener("mouseenter", e => showPathTooltip(d.id, step, e));
        hitArea.addEventListener("mouseleave", () => hideTooltip());
        svg.appendChild(hitArea);
    });
}
```

For each drone with a path:
1. Convert `[row, col]` pairs to SVG `"x,y"` strings
2. Draw a glowing blurry underline (visual effect)
3. Draw the actual coloured path line with an arrowhead at the end
4. Draw an invisible wider line on top — this catches mouse hover events without blocking cell clicks

---

### HOVER TOOLTIP

```javascript
function showPathTooltip(droneId, step, e) {
    const d = drones.find(x => x.id === droneId);
    const tt = document.getElementById("tt");
    tt.innerHTML = `
        <b>${d.id}</b> (${d.type})<br>
        Delivery: ${d.delivery}<br>
        Status: ${d.status}<br>
        Battery: ${d.battery}%<br>
        Hub: ${JSON.stringify(d.hub)}
    `;
    tt.style.display = "block";
    tt.style.left = (e.pageX + 12) + "px";
    tt.style.top  = (e.pageY - 20) + "px";
}
```

When you hover over a path, this fills a tooltip div with drone info and positions it near the mouse cursor.

---

### DEMAND HEATMAP

```javascript
const ZONE_HEAT = {
    Residential: 7.8, Commercial: 6.2, Industrial: 3.1,
    Hospital: 2.8, School: 4.9, "Open Field": 0.8
};

function buildHeatmap() {
    const hm = document.getElementById("heatmap-grid");
    for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
            const div = document.createElement("div");
            div.className = "hmap-cell";
            div.id = `hm-${r}-${c}`;
            hm.appendChild(div);
        }
    }
}

function updateHeatmap() {
    for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
            const v = ZONE_HEAT[layout[r][c]] || 0.8;
            const div = document.getElementById(`hm-${r}-${c}`);
            div.style.background = heatColor(v);
            div.style.opacity = "1";
        }
    }
}

function heatColor(v) {
    // v is 0–10
    const t = v / 10;
    const r = Math.round(0   + t * 0);   // always 0
    const g = Math.round(10  + t * 235); // 10 → 245
    const b = Math.round(60  + t * 195); // 60 → 255
    return `rgb(${r},${g},${b})`;
}
```

The heatmap is a second 10×10 grid hidden beneath the city grid. At step 15, `updateHeatmap()` is called which colours each cell based on its zone's typical demand. `heatColor(v)` maps 0–10 to a colour gradient: dark navy (low demand) → bright cyan (high demand).

---

### SIMULATION CONTROLS

```javascript
let simRunning = false;
let simPaused  = false;

async function runSimulation() {
    simRunning = true;
    document.getElementById("pauseBtn").disabled = false;
    ...
    for (let step = 1; step <= 20; step++) {
        while (simPaused) await sleep(120);  // wait while paused
        if (!simRunning) break;              // stop if reset pressed
        
        // move drones
        // update logs
        // at step 11: inject no-fly zone
        // at step 15: update heatmap
        // at step 17: inject battery anomaly
        
        await sleep(800);  // pause 800ms between steps
    }
    showSummaryPanel();
}

function togglePause() {
    simPaused = !simPaused;
    document.getElementById("pauseBtn").textContent = simPaused ? "Resume" : "Pause";
}

function resetSim() {
    simRunning = false;
    simPaused  = false;
    // reset all drone positions and delivery statuses
    // clear SVG paths
    // clear logs
}
```

`async/await` is JavaScript's way of handling time — `await sleep(800)` pauses for 800 milliseconds without freezing the browser. The `while(simPaused)` loop keeps checking every 120ms whether to continue.

---

### THE CHARTS

**Donut Chart (Fleet Composition):**
```javascript
new Chart(ctx, {
    type: "doughnut",
    data: { labels: ["Light (×5)", "Heavy (×2)"],
            datasets: [{ data: [5, 2], backgroundColor: ["#60a5fa","#f59e0b"] }] }
});
```
Shows 5 Light vs 2 Heavy drones as proportional arcs.

**GA Convergence Chart (Line Chart):**
Shows how the genetic algorithm's best score improved over 60 generations. X-axis = generation number, Y-axis = fitness score. The line starts around 0.6–0.65 and plateaus near 0.84.

**Demand Forecast Chart (Bar Chart):**
Side-by-side bars for Linear Regression and Random Forest, grouped by metric (MAE, RMSE, R²). Lower bars = better for MAE/RMSE. Higher bar = better for R². Random Forest bars are dramatically better.

---

### MODULE LABELS

Each panel header has a coloured badge:
- **M1** (green) — CSP panel
- **M2** (cyan) — Fleet Selection panel
- **M3** (purple) — A* Route panel
- **M4** (yellow) — Drone Simulation panel
- **M5** (pink) — ML / Demand panel

```html
<span class="mod-tag m1">M1</span>
```
```css
.mod-tag { font-size:10px; padding:2px 5px; border-radius:4px; }
.m1 { background:#22c55e; color:#000; }
```

---

### FINAL SUMMARY PANEL

```javascript
function showSummaryPanel() {
    document.getElementById("summary-panel").style.display = "block";
    // fill in KPIs:
    // completed deliveries, drones active, CSP status, GA best score, RF R² score
    // module checklist: M0 grid ✓, M1 CSP ✓, M2 GA ✓, M3 A* ✓, M4 sim ✓, M5 ML ✓
    // ML model comparison table
}
```

After step 20, a summary panel appears showing:
- Total deliveries completed vs delayed vs failed
- All 5 modules used (with green ticks)
- Side-by-side ML comparison
- The GA winner and final score

---

---

## QUICK REFERENCE — NUMBERS TO KNOW FOR THE DEMO

| Item | Value |
|------|-------|
| Grid size | 10×10 = 100 cells |
| Zone types | 6 (Residential, Commercial, Industrial, Hospital, School, Open Field) |
| Drone hubs | 3 (at positions (1,4), (5,3), (7,5)) |
| Fleet | 5 Light + 2 Heavy = 7 drones total |
| Fleet cost | $8,600 out of $10,000 budget |
| GA score | 0.84 |
| Deliveries | 8 total |
| Simulation steps | 20 |
| No-fly event | Step 11, cell (4,4) — Drone 3 reroutes |
| Battery anomaly | Step 17, Drone 5 — battery drops 40% |
| Training data | 10,886 records (real Bike Sharing Dataset from Kaggle) |
| LR performance | MAE=107.66, RMSE=147.23, R²=0.343 (poor) |
| RF performance | MAE=44.07, RMSE=65.99, R²=0.868 (great) |
| Anomaly models | DT=91.75%, RF=96.25%, KNN=89.50%, NB=85.25% |
| CSP rules | R1, R2, R3, R4 — all PASS |
| A* heuristic | Manhattan distance |
| Commercial cell cost | 0.8 (vs 1.0 for others) |
| Battery drain | Light=5%/step, Heavy=3.5%/step |
| Population density (real) | Residential=4720, Commercial=3128, Industrial=2076 |

---

## COMMON QUESTIONS YOU MIGHT BE ASKED

**Q: Why use Bike Sharing data for drone delivery demand?**
A: The assignment spec says to use it as a "proxy." Bike rental demand is driven by the same factors as delivery demand — time of day, weather, whether it's a workday. The patterns transfer reasonably well.

**Q: Why Random Forest and not just Linear Regression?**
A: Linear Regression assumes the relationship between features and count is a straight line. Demand doesn't work that way — it spikes at rush hour, drops on rainy days, etc. Random Forest handles these non-linear patterns perfectly.

**Q: Why does the GA use 30 population and 60 generations?**
A: These are common defaults. Too few and it doesn't explore enough; too many wastes time. 30×60 = 1800 evaluations is enough to reliably converge on a good answer.

**Q: Why Manhattan distance and not straight-line distance?**
A: Drones on a grid can only move up/down/left/right (4 directions). They can't cut across diagonally. So actual travel distance = Manhattan distance, not Euclidean.

**Q: What happens if the CSV files are missing?**
A: Both Module 0 and Module 5 have fallback modes — Module 0 uses hardcoded density values, Module 5 generates synthetic data. The system never crashes.

**Q: What is the difference between A* and Dijkstra's algorithm?**
A: Dijkstra's explores squares in order of actual cost from start. A* adds a heuristic (estimated remaining distance to goal) so it focuses exploration toward the destination. A* is faster.

---

*End of Demo Guide — covers every line of every module and every dashboard element.*
