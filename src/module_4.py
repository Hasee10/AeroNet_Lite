from dataclasses import dataclass, field
from typing import Optional
from module_3 import plan_delivery_route, astar, astar_result
from module_0 import get_cells_by


@dataclass
class Drone:
    drone_id: str
    drone_type: str
    payload_kg: float
    max_range: int
    hub: tuple
    position: tuple = field(init=False)
    battery: float = 100.0
    status: str = "idle"
    current_delivery: Optional[str] = None
    route: list = field(default_factory=list)
    route_index: int = 0
    battery_drain_per_step: float = 4.0

    @property
    def id(self):
        return self.drone_id

    @property
    def current_pos(self):
        return self.position

    @property
    def current_route(self):
        return self.route[self.route_index:]

    @property
    def delivery_id(self):
        return self.current_delivery

    def __post_init__(self):
        self.position = self.hub

    @property
    def has_route(self):
        return bool(self.route) and self.route_index < len(self.route)

    def advance(self):
        if self.route_index + 1 < len(self.route):
            self.route_index += 1
            self.position = self.route[self.route_index]
            self.battery  = max(0.0, self.battery - self.battery_drain_per_step)


@dataclass
class Delivery:
    delivery_id: str
    pickup: tuple
    dropoff: tuple
    payload_kg: float
    status: str = "pending"
    assigned_drone: Optional[str] = None

    @property
    def id(self):
        return self.delivery_id

    @property
    def weight(self):
        return self.payload_kg


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
        drones.append(Drone(drone_id=f"D{drone_num}", drone_type="heavy",
                            payload_kg=5.0, max_range=20,
                            hub=hubs[i % len(hubs)], battery_drain_per_step=3.5))
        drone_num += 1
    return drones


def generate_deliveries(grid, count=8, seed=7):
    import random
    random.seed(seed)
    residential = [(c.row, c.col) for row in grid for c in row if c.zone == "Residential"]
    commercial  = [(c.row, c.col) for row in grid for c in row if c.zone == "Commercial"]
    medical     = [(c.row, c.col) for row in grid for c in row if c.is_medical_pickup]
    deliveries  = []
    for i in range(1, count + 1):
        pickup  = random.choice(commercial + medical)
        dropoff = random.choice(residential)
        while dropoff == pickup:
            dropoff = random.choice(residential)
        payload = random.choice([1.5, 2.0, 3.0, 4.5])
        deliveries.append(Delivery(delivery_id=f"DEL_{i:02d}",
                                   pickup=pickup, dropoff=dropoff, payload_kg=payload))
    return deliveries


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
        if drone.payload_kg < delivery.payload_kg:
            log.append(f"  [{delivery.delivery_id}] No available drone with sufficient payload "
                       f"({delivery.payload_kg}kg required, {drone.payload_kg}kg available).")
            continue
        result = plan_delivery_route(drone, delivery, grid)
        if result["success"]:
            drone.route            = result["full_path"]
            drone.route_index      = 0
            drone.status           = "en_route"
            drone.current_delivery = delivery.delivery_id
            delivery.status        = "assigned"
            delivery.assigned_drone = drone.drone_id
            idle_drones.remove(drone)
            log.append(f"  {delivery.delivery_id} assigned to {drone.drone_id} ({drone.drone_type}). "
                       f"Route: {delivery.pickup} -> {delivery.dropoff}. Cost: {result['total_cost']}.")
        else:
            delivery.status = "failed"
            log.append(f"  {delivery.delivery_id} FAILED routing: {result['message']}")


def move_drones(drones, deliveries, log):
    delivery_map = {d.delivery_id: d for d in deliveries}
    for drone in drones:
        if drone.status != "en_route":
            continue
        if not drone.has_route:
            drone.status = "idle"
            drone.current_delivery = None
            continue
        drone.advance()
        if drone.route_index == len(drone.route) - 1:
            if drone.current_delivery:
                delivery = delivery_map.get(drone.current_delivery)
                if delivery:
                    delivery.status = "completed"
                log.append(f"  {drone.drone_id} completed delivery {drone.current_delivery}. "
                           f"Returned to hub {drone.hub}.")
            drone.route            = []
            drone.route_index      = 0
            drone.position         = drone.hub
            drone.status           = "idle"
            drone.current_delivery = None


def activate_nofly(row, col, grid, drones=None):
    grid[row][col].no_fly = True
    if drones is None:
        return []
    return [d.drone_id for d in drones
            if d.status == "en_route" and any(pos == (row, col) for pos in d.route[d.route_index:])]


def activate_no_fly(grid, position, log):
    r, c = position
    affected = activate_nofly(r, c, grid)
    log.append(f"  No-fly cell activated at {position}.")
    return affected


def check_routes_for_nofly(cell, drones, grid):
    reroute_log = []
    for drone in drones:
        if drone.status != "en_route" or not drone.route:
            continue
        if cell in drone.route[drone.route_index:]:
            msg = reroute_drone(drone, grid)
            reroute_log.append(msg)
    return reroute_log


def reroute_drone(drone, grid):
    if not drone.route:
        return f"  {drone.drone_id}: no active route to reroute."
    goal = drone.route[-1]
    path, cost = astar(drone.position, goal, grid)
    if path is not None:
        drone.route       = [drone.position] + path[1:]
        drone.route_index = 0
        return (f"  Drone {drone.drone_id} rerouted via A* from "
                f"{drone.position} to {goal}. New cost: {cost}.")
    drone.status = "failed"
    return (f"  Drone {drone.drone_id} cannot reach destination {goal}. "
            f"Delivery {drone.current_delivery} delayed.")


def reroute_drones(drones, deliveries, grid, log):
    delivery_map = {d.delivery_id: d for d in deliveries}
    for drone in drones:
        if drone.status != "en_route" or not drone.route:
            continue
        remaining = drone.route[drone.route_index:]
        conflict  = next((pos for pos in remaining if grid[pos[0]][pos[1]].no_fly), None)
        if conflict is None:
            continue
        msg = reroute_drone(drone, grid)
        log.append(msg)
        if drone.status == "failed":
            delivery = delivery_map.get(drone.current_delivery)
            if delivery:
                delivery.status = "delayed"
            log.append(f"  Delivery {drone.current_delivery} marked as delayed.")


def step_simulation(step, drones, deliveries, grid):
    step_log     = []
    delivery_map = {d.delivery_id: d for d in deliveries}
    for drone in drones:
        if drone.status != "en_route" or not drone.route:
            continue
        remaining = drone.route[drone.route_index:]
        conflict  = next((pos for pos in remaining if grid[pos[0]][pos[1]].no_fly), None)
        if conflict:
            step_log.append(reroute_drone(drone, grid))
    for drone in drones:
        if drone.status != "en_route":
            continue
        if not drone.has_route:
            drone.status = "idle"
            drone.current_delivery = None
            continue
        drone.advance()
        if drone.route_index == len(drone.route) - 1:
            if drone.current_delivery:
                delivery = delivery_map.get(drone.current_delivery)
                if delivery:
                    delivery.status = "completed"
                step_log.append(f"Step {step}: {drone.drone_id} completed delivery "
                                f"{drone.current_delivery}.")
            drone.route            = []
            drone.route_index      = 0
            drone.position         = drone.hub
            drone.status           = "idle"
            drone.current_delivery = None
        else:
            step_log.append(f"Step {step}: {drone.drone_id} at {drone.position}. "
                            f"Battery: {drone.battery:.1f}%.")
    return drones, step_log


def inject_anomaly(drone, anomaly_type, log):
    if anomaly_type == "battery":
        drone.battery = max(0.0, drone.battery - 40.0)
        log.append(f"  Battery anomaly detected for {drone.drone_id}! "
                   f"Battery dropped to {drone.battery:.1f}%.")
        if drone.battery < 20:
            drone.status = "returning"
            log.append(f"  {drone.drone_id} returning to hub due to critical battery level.")
    elif anomaly_type == "route":
        log.append(f"  Route anomaly detected for {drone.drone_id} - "
                   f"unexpected deviation from planned path.")
    elif anomaly_type == "sensor":
        log.append(f"  Sensor spike detected for {drone.drone_id} - "
                   f"altitude/speed reading anomalous.")


def simulation_summary(deliveries):
    return {
        "completed": sum(1 for d in deliveries if d.status == "completed"),
        "delayed":   sum(1 for d in deliveries if d.status == "delayed"),
        "failed":    sum(1 for d in deliveries if d.status == "failed"),
        "pending":   sum(1 for d in deliveries if d.status in ("pending", "assigned")),
    }
