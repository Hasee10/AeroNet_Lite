import random

default_budget  = 10_000
light_cost      = 1_000
heavy_cost      = 1_800
light_payload   = 2
heavy_payload   = 5
light_range     = 12
heavy_range     = 20
light_daily_cap = 3
heavy_daily_cap = 2

pop_size      = 30
generations   = 60
mutation_rate = 0.15


def compute_coverage(drones, grid):
    demand_zones = sum(
        1 for row in grid for cell in row
        if cell.demand > 1.5 and cell.zone in ("Residential", "Commercial")
    )
    if demand_zones == 0:
        return 100.0
    fleet_capacity = sum(
        light_daily_cap if d.get("type") == "light" else heavy_daily_cap
        for d in drones
    )
    return min(100.0, fleet_capacity / demand_zones * 100)


def compute_score(n_light, n_heavy, grid, budget=default_budget):
    cost = n_light * light_cost + n_heavy * heavy_cost
    if cost > budget:
        return -999.0
    demand_zones = max(
        sum(1 for row in grid for cell in row
            if cell.demand > 1.5 and cell.zone in ("Residential", "Commercial")),
        10,
    )
    fleet_capacity  = n_light * light_daily_cap + n_heavy * heavy_daily_cap
    coverage_pct    = min(100.0, fleet_capacity / demand_zones * 100)
    budget_used_pct = cost / budget * 100
    return 0.75 * coverage_pct - 0.25 * budget_used_pct


def create_drone_objects(result, grid=None):
    hubs = []
    if grid is not None:
        hubs = [(c.row, c.col) for row in grid for c in row if c.is_hub]
    if not hubs:
        hubs = [(0, 0)]
    drones = []
    drone_id = 1
    for i in range(result.get("light", 0)):
        drones.append({"id": f"D{drone_id}", "type": "light",
                       "hub": hubs[i % len(hubs)],
                       "payload": light_payload, "range": light_range})
        drone_id += 1
    for i in range(result.get("heavy", 0)):
        drones.append({"id": f"D{drone_id}", "type": "heavy",
                       "hub": hubs[i % len(hubs)],
                       "payload": heavy_payload, "range": heavy_range})
        drone_id += 1
    return drones


def select_fleet_brute(grid, budget=default_budget):
    best_score = -9999.0
    best = (0, 0)
    for l in range(budget // light_cost + 1):
        for h in range(budget // heavy_cost + 1):
            s = compute_score(l, h, grid, budget)
            if s > best_score:
                best_score = s
                best = (l, h)
    cost = best[0] * light_cost + best[1] * heavy_cost
    return {"light": best[0], "heavy": best[1], "cost": cost,
            "score": round(best_score, 2), "budget": budget,
            "budget_used_pct": round(cost / budget * 100, 1),
            "method": "Brute Force"}


def _random_chromosome(budget):
    return [random.randint(0, budget // light_cost),
            random.randint(0, budget // heavy_cost)]


def _tournament(pop, scores, k=3):
    contestants = random.sample(range(len(pop)), k)
    winner = max(contestants, key=lambda i: scores[i])
    return pop[winner][:]


def _crossover(p1, p2):
    return [p1[0], p2[1]], [p2[0], p1[1]]


def _mutate(chrom, budget):
    gene  = random.randint(0, 1)
    delta = random.choice([-1, 1])
    limit = budget // (light_cost if gene == 0 else heavy_cost)
    chrom[gene] = max(0, min(limit, chrom[gene] + delta))
    return chrom


def select_fleet_ga(grid, budget=default_budget, seed=42):
    random.seed(seed)
    population  = [_random_chromosome(budget) for _ in range(pop_size)]
    best_chrom  = population[0][:]
    best_score  = -9999.0
    for _ in range(generations):
        scores      = [compute_score(c[0], c[1], grid, budget) for c in population]
        gen_best    = max(range(len(scores)), key=lambda i: scores[i])
        if scores[gen_best] > best_score:
            best_score = scores[gen_best]
            best_chrom = population[gen_best][:]
        next_pop = [best_chrom[:]]
        while len(next_pop) < pop_size:
            p1 = _tournament(population, scores)
            p2 = _tournament(population, scores)
            c1, c2 = _crossover(p1, p2)
            if random.random() < mutation_rate:
                c1 = _mutate(c1, budget)
            if random.random() < mutation_rate:
                c2 = _mutate(c2, budget)
            next_pop.extend([c1, c2])
        population = next_pop[:pop_size]
    l, h = best_chrom
    cost = l * light_cost + h * heavy_cost
    return {"light": l, "heavy": h, "cost": cost,
            "score": round(best_score, 2), "budget": budget,
            "budget_used_pct": round(cost / budget * 100, 1),
            "method": "Genetic Algorithm"}


def select_fleet(grid, budget=default_budget):
    return select_fleet_ga(grid, budget)


def print_fleet_report(fleet):
    budget = fleet.get("budget", default_budget)
    print(f"\n{'='*60}")
    print("  Fleet Selection")
    print(f"{'='*60}")
    print(f"  Method        : {fleet['method']}")
    print(f"  Budget        : {budget:,} units")
    print(f"  Light drones  : {fleet['light']}  ({light_cost:,} each | {light_range}-cell range | {light_payload}kg)")
    print(f"  Heavy drones  : {fleet['heavy']}  ({heavy_cost:,} each | {heavy_range}-cell range | {heavy_payload}kg)")
    print(f"  Total cost    : {fleet['cost']:,} units  ({fleet['budget_used_pct']}% of budget)")
    print(f"  Coverage score: {fleet['score']}")
    print(f"{'='*60}\n")
