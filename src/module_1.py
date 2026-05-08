from module_0 import Cell


def get_neighbors(row, col, grid=None):
    coords = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 10 and 0 <= nc < 10:
            coords.append((nr, nc))
    if grid is None:
        return coords
    return [grid[r][c] for r, c in coords]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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


def check_residential_coverage(grid):
    errors = []
    hubs = [(c.row, c.col) for row in grid for c in row if c.is_hub]
    for row in grid:
        for cell in row:
            if cell.zone == "Residential":
                pos = (cell.row, cell.col)
                if not any(manhattan(pos, h) <= 3 for h in hubs):
                    nearest = min(hubs, key=lambda h: manhattan(pos, h))
                    sug_r = max(0, min(9, cell.row - 1))
                    sug_c = max(0, min(9, cell.col))
                    errors.append(
                        f"R2 FAIL: Residential at ({cell.row},{cell.col}) has no hub within 3 cells. "
                        f"Nearest hub: {nearest} (dist {manhattan(pos, nearest)}). "
                        f"Suggested fix: add a hub near ({sug_r},{sug_c}) or convert cell to Open Field."
                    )
    return errors


def check_hub_charging(grid):
    errors = []
    charging_pads = [(c.row, c.col) for row in grid for c in row if c.is_charging]
    for row in grid:
        for cell in row:
            if cell.is_hub:
                pos = (cell.row, cell.col)
                if not any(manhattan(pos, cp) <= 2 for cp in charging_pads):
                    errors.append(
                        f"R3 FAIL: Hub at ({cell.row},{cell.col}) has no charging pad within 2 cells."
                    )
    return errors


def check_medical_access(grid):
    errors = []
    medical_pickups = [(c.row, c.col) for row in grid for c in row if c.is_medical_pickup]
    hospitals = [c for row in grid for c in row if c.zone == "Hospital"]
    covered = any(
        any(manhattan((h.row, h.col), mp) <= 1 for mp in medical_pickups)
        for h in hospitals
    )
    if not covered:
        if hospitals:
            errors.append(
                f"R4 FAIL: No hospital has a medical pickup within 1 cell. "
                f"Hospitals at: {[(h.row, h.col) for h in hospitals]}."
            )
        else:
            errors.append("R4 FAIL: No Hospital zone found in the grid.")
    return errors


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


def print_validation_report(report):
    valid   = report["valid"]
    passed  = report["passed"]
    failed  = report["failed"]
    details = report["details"]
    print(f"\n{'='*60}")
    print("  Layout Validation Report")
    print(f"{'='*60}")
    print(f"  PASSED: {', '.join(passed) if passed else 'none'}")
    print(f"  FAILED: {', '.join(failed) if failed else 'none'}")
    for rule, errs in details.items():
        if not errs:
            print(f"\n  [{rule}] PASSED - constraint satisfied.")
        else:
            print(f"\n  [{rule}] FAILED ({len(errs)} violation(s)):")
            for e in errs:
                print(f"    - {e}")
    print(f"\n  Layout valid: {valid}")
    print(f"{'='*60}\n")
