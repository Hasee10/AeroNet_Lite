import heapq


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_move_cost(cell):
    return 0.8 if cell.zone == "Commercial" else 1.0


def reconstruct_path(came_from, current):
    path = []
    node = current
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def astar(start, goal, grid):
    if grid[goal[0]][goal[1]].no_fly:
        return None, None
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_heap  = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0.0, start))
    g_cost    = {start: 0.0}
    came_from = {start: None}
    closed    = set()
    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path, round(g, 2)
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
    return None, None


def astar_result(start, goal, grid):
    path, cost = astar(start, goal, grid)
    if path is None:
        return {"path": [], "cost": 0, "found": False,
                "message": f"No safe path from {start} to {goal} (all routes blocked)."}
    return {"path": path, "cost": cost, "found": True,
            "message": f"Path found ({len(path)} steps, cost {cost})."}


def plan_delivery_route(drone, delivery, grid):
    if isinstance(drone, tuple):
        hub = drone
    elif isinstance(drone, dict):
        hub = drone["hub"]
    else:
        hub = drone.hub

    if isinstance(delivery, (tuple, list)):
        pickup, dropoff = delivery[0], delivery[1]
    elif isinstance(delivery, dict):
        pickup  = delivery["pickup"]
        dropoff = delivery["dropoff"]
    else:
        pickup  = delivery.pickup
        dropoff = delivery.dropoff

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
            return {"segments": segments, "full_path": [], "total_cost": 0,
                    "success": False, "found": False,
                    "message": f"Leg '{label}' failed: {res['message']}"}
        if full_path:
            full_path.extend(res["path"][1:])
        else:
            full_path.extend(res["path"])
        total_cost += res["cost"]
    total_cost = round(total_cost, 2)
    return {"segments": segments, "full_path": full_path, "total_cost": total_cost,
            "success": True, "found": True,
            "message": f"Full route planned ({len(full_path)} steps, cost {total_cost})."}
