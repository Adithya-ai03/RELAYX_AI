from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PRIORITY_VALUE = {"Critical": 1.0, "High": .75, "Medium": .5, "Normal": .25}
DEFAULT_WEIGHTS = {"urgency": .25, "deadline": .25, "priority": .20, "delay": .15, "value": .10, "route_risk": .05}


def validate_data(shipments: pd.DataFrame, vehicles: pd.DataFrame, hubs: pd.DataFrame, routes: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if shipments["shipment_id"].duplicated().any(): issues.append("Duplicate shipment IDs found.")
    if vehicles["vehicle_id"].duplicated().any(): issues.append("Duplicate vehicle IDs found.")
    if (shipments["weight_kg"] < 0).any(): issues.append("Negative shipment weights found.")
    if (vehicles["capacity_kg"] < vehicles["current_load_kg"]).any(): issues.append("Some vehicles have current load above capacity.")
    if not pd.to_datetime(shipments["deadline"], errors="coerce").notna().all(): issues.append("Some shipment deadlines are invalid.")
    route_pairs = set(zip(routes.origin, routes.destination))
    network_locations = {location for pair in route_pairs for location in pair}
    invalid = [row.shipment_id for row in shipments.itertuples() if row.current_location not in network_locations or row.destination not in network_locations]
    if invalid: issues.append(f"{len(invalid)} shipment location reference(s) are not present in the route network.")
    return issues


def analyze_shipments(shipments: pd.DataFrame, weights: dict[str, float] | None = None, now: datetime | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    now = now or datetime.now()
    result = shipments.copy()
    result["deadline"] = pd.to_datetime(result["deadline"])
    remaining = (result["deadline"] - now).dt.total_seconds() / 3600
    result["remaining_hours"] = remaining.clip(lower=0)
    result["overdue"] = remaining < 0
    result["urgency_score"] = (1 - remaining.clip(lower=0).clip(upper=72) / 72).clip(0, 1)
    result["deadline_pressure"] = (result["expected_transport_hours"] / remaining.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(2).clip(0, 2)
    result["deadline_risk"] = (result["deadline_pressure"] / 1.5).clip(0, 1)
    result["deadline_risk_label"] = pd.cut(result["deadline_pressure"], [-np.inf, .55, 1, 1.5, np.inf], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]).astype(str)
    value_score = (result["shipment_value"] - result["shipment_value"].min()) / max(result["shipment_value"].max() - result["shipment_value"].min(), 1)
    result["priority_score"] = sum([weights["urgency"] * result["urgency_score"], weights["deadline"] * result["deadline_risk"], weights["priority"] * result["priority"].map(PRIORITY_VALUE), weights["delay"] * result["delay_probability"], weights["value"] * value_score, weights["route_risk"] * result["route_risk"]]).clip(0, 1)
    result["recommended_action"] = np.where(result["deadline_risk"] > .75, "Recover immediately", np.where(result["priority_score"] > .5, "Prioritize recovery", "Monitor and pack"))
    return result.sort_values("priority_score", ascending=False).reset_index(drop=True)


def _route_time(routes: pd.DataFrame, origin: str, destination: str) -> float | None:
    match = routes[(routes.origin == origin) & (routes.destination == destination) & (routes.route_status == "Active")]
    return float(match.iloc[0].estimated_hours) if not match.empty else None


def candidates_for_shipment(shipment: pd.Series, vehicles: pd.DataFrame, hubs: pd.DataFrame, routes: pd.DataFrame, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now()
    candidates: list[dict[str, Any]] = []
    route_times = {(row.origin, row.destination): float(row.estimated_hours) for row in routes.itertuples() if row.route_status == "Active"}
    hub_costs = {row.city: float(row.handling_cost) for row in hubs.itertuples() if row.operational_status == "Operational"}
    vehicle_rows = list(vehicles.itertuples())
    vehicles_by_location: dict[str, list[Any]] = {}
    vehicles_by_leg: dict[tuple[str, str], list[Any]] = {}
    for vehicle in vehicle_rows:
        if vehicle.vehicle_status == "Unavailable":
            continue
        vehicles_by_location.setdefault(vehicle.current_location, []).append(vehicle)
        vehicles_by_leg.setdefault((vehicle.current_location, vehicle.route_destination), []).append(vehicle)
    for vehicle in vehicles_by_location.get(shipment.current_location, []):
        available = max(float(getattr(vehicle, "available_capacity_kg", vehicle.capacity_kg - vehicle.current_load_kg)), 0)
        if vehicle.route_destination == shipment.destination:
            travel = route_times.get((vehicle.route_origin, vehicle.route_destination))
            if travel is None:
                continue
            eta_hours = max(travel, (pd.Timestamp(vehicle.eta).to_pydatetime() - now).total_seconds() / 3600)
            candidates.append({"vehicle_id": vehicle.vehicle_id, "strategy": "DIRECT PIGGYBACK", "route": f"{vehicle.route_origin} -> {vehicle.route_destination}", "eta_hours": eta_hours, "capacity_available": available, "transport_cost": float(vehicle.transport_cost), "transfer_cost": 0.0, "hub": None})
    # Search connected paths with up to four operational transfer hubs.
    active_edges = list(route_times)

    def paths_from(current: str, destination: str, visited: tuple[str, ...] = ()) -> list[list[str]]:
        if current == destination:
            return [[current]]
        if len(visited) >= 5:
            return []
        paths: list[list[str]] = []
        for origin, next_city in active_edges:
            if origin != current or next_city in visited or next_city == shipment.current_location:
                continue
            if next_city != destination and next_city not in hub_costs:
                continue
            for suffix in paths_from(next_city, destination, visited + (current,)):
                paths.append([current] + suffix)
        return paths

    for path in paths_from(shipment.current_location, shipment.destination):
        if len(path) < 3:
            continue
        leg_options = [vehicles_by_leg.get((origin, destination), []) for origin, destination in zip(path, path[1:])]
        if any(not options for options in leg_options):
            continue
        # A small bounded Cartesian product is appropriate for the prototype fleet size.
        combinations: list[list[Any]] = [[]]
        for options in leg_options:
            combinations = [prefix + [vehicle] for prefix in combinations for vehicle in options if vehicle.vehicle_id not in {item.vehicle_id for item in prefix}]
        for combination in combinations:
            capacities = [max(float(getattr(item, "available_capacity_kg", item.capacity_kg - item.current_load_kg)), 0) for item in combination]
            route_hours = sum(route_times[(origin, destination)] for origin, destination in zip(path, path[1:]))
            handling_hours = 2 * (len(path) - 2)
            first_eta = max(route_times[(path[0], path[1])], (pd.Timestamp(combination[0].eta).to_pydatetime() - now).total_seconds() / 3600)
            eta_hours = first_eta + route_hours - route_times[(path[0], path[1])] + handling_hours
            hub_cost = sum(hub_costs[hub] for hub in path[1:-1])
            candidates.append({"vehicle_id": " + ".join(item.vehicle_id for item in combination), "vehicle_ids": [item.vehicle_id for item in combination], "strategy": "ONE-HUB PIGGYBACK" if len(path) == 3 else "MULTI-HUB PIGGYBACK", "route": " -> ".join(path), "eta_hours": eta_hours, "capacity_available": min(capacities), "transport_cost": float(sum(item.transport_cost for item in combination)), "transfer_cost": float(hub_cost), "hub": " + ".join(path[1:-1])})
    return candidates


def evaluate_candidates(shipment: pd.Series, vehicles: pd.DataFrame, hubs: pd.DataFrame, routes: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Evaluate planner candidates with the same hard checks used by allocation."""
    now = now or datetime.now()
    rows = []
    for candidate in candidates_for_shipment(shipment, vehicles, hubs, routes, now):
        required = float(shipment.weight_kg)
        cost = (candidate["transport_cost"] * required / max(candidate["capacity_available"], required)) + candidate["transfer_cost"]
        margin = float(shipment.remaining_hours - candidate["eta_hours"])
        reasons = []
        if candidate["capacity_available"] < required:
            reasons.append(f"capacity {candidate['capacity_available']:.0f}kg < {required:.0f}kg required")
        if margin < 0:
            reasons.append(f"ETA misses deadline by {abs(margin):.1f}h")
        rows.append({**candidate, "cost": round(cost, 2), "deadline_margin": round(margin, 2), "required_capacity": required, "utilization": required / max(candidate["capacity_available"], 1), "feasible": not reasons, "reason": "; ".join(reasons) if reasons else "Route, capacity, availability, and deadline checks passed."})
    return pd.DataFrame(rows)


def run_simulation(shipments: pd.DataFrame, vehicles: pd.DataFrame, hubs: pd.DataFrame, routes: pd.DataFrame, weights: dict[str, float] | None = None, capacity_factor: float = 1.0, unavailable_vehicle_ids: list[str] | None = None, deadline_shift_hours: float = 0, route_cost_factor: float = 1.0, now: datetime | None = None) -> dict[str, Any]:
    """Recalculate on copied frames so what-if controls never mutate source data."""
    simulated_vehicles = vehicles.copy()
    simulated_vehicles["capacity_kg"] = (simulated_vehicles["capacity_kg"] * capacity_factor).round().clip(lower=1)
    if unavailable_vehicle_ids:
        simulated_vehicles.loc[simulated_vehicles.vehicle_id.isin(unavailable_vehicle_ids), "vehicle_status"] = "Unavailable"
    simulated_vehicles["transport_cost"] = simulated_vehicles["transport_cost"] * route_cost_factor
    simulated_shipments = shipments.copy()
    simulated_shipments["deadline"] = pd.to_datetime(simulated_shipments["deadline"]) - pd.to_timedelta(deadline_shift_hours, unit="h")
    return allocate_plan(simulated_shipments, simulated_vehicles, hubs.copy(), routes.copy(), weights, now)


def allocate_plan(shipments: pd.DataFrame, vehicles: pd.DataFrame, hubs: pd.DataFrame, routes: pd.DataFrame, weights: dict[str, float] | None = None, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now()
    analyzed = analyze_shipments(shipments, weights, now)
    vehicle_state = vehicles.copy()
    vehicle_state["available_capacity_kg"] = (vehicle_state.capacity_kg - vehicle_state.current_load_kg).clip(lower=0)
    allocations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for _, shipment in analyzed.iterrows():
        options = candidates_for_shipment(shipment, vehicle_state, hubs, routes, now)
        feasible = []
        for option in options:
            required = float(shipment.weight_kg)
            if option["capacity_available"] < required:
                rejected.append({"shipment_id": shipment.shipment_id, "vehicle_id": option["vehicle_id"], "reason": f"Insufficient capacity: {option['capacity_available']:.0f}kg available, {required:.0f}kg required."})
                continue
            margin = float(shipment.remaining_hours - option["eta_hours"])
            if margin < 0:
                rejected.append({"shipment_id": shipment.shipment_id, "vehicle_id": option["vehicle_id"], "reason": f"ETA exceeds deadline by {abs(margin):.1f} hours."})
                continue
            cost = (option["transport_cost"] * required / max(option["capacity_available"], required)) + option["transfer_cost"]
            # Prefer deadline safety and lower estimated cost; use utilization as a tie-breaker.
            score = .35 * min(margin / max(shipment.remaining_hours, 1), 1) + .20 * shipment.priority_score + .40 * (1 - min(cost / 8000, 1)) + .05 * min(required / max(option["capacity_available"], required), 1)
            feasible.append({**option, "cost": round(cost, 2), "deadline_margin": round(margin, 2), "score": round(float(score), 4)})
        if feasible:
            selected = max(feasible, key=lambda item: item["score"])
            selected.update({"shipment_id": shipment.shipment_id, "weight_kg": float(shipment.weight_kg), "priority": shipment.priority, "priority_score": round(float(shipment.priority_score), 4), "deadline": shipment.deadline, "status": "Recovered", "reason": "Compatible route, sufficient capacity, deadline-feasible ETA, and strongest weighted recovery score."})
            allocations.append(selected)
            for vehicle_id in selected.get("vehicle_ids", [selected["vehicle_id"]]):
                if vehicle_id in set(vehicle_state.vehicle_id):
                    vehicle_state.loc[vehicle_state.vehicle_id == vehicle_id, "available_capacity_kg"] -= shipment.weight_kg
    allocation_df = pd.DataFrame(allocations)
    recovered_ids = set(allocation_df.shipment_id) if not allocation_df.empty else set()
    escalated = analyzed[~analyzed.shipment_id.isin(recovered_ids)].copy()
    escalated["status"] = "Escalated"
    escalated["reason"] = "No feasible piggyback option found within capacity and deadline constraints."
    total_cost = float(allocation_df.cost.sum()) if not allocation_df.empty else 0.0
    dedicated = float(analyzed.loc[analyzed.shipment_id.isin(recovered_ids), "weight_kg"].sum() * 8) if recovered_ids else 0.0
    return {"shipments": analyzed, "allocations": allocation_df, "escalated": escalated, "rejected": pd.DataFrame(rejected), "vehicles": vehicle_state, "total_cost": total_cost, "dedicated_cost": dedicated, "savings": dedicated - total_cost, "validation": validate_data(shipments, vehicles, hubs, routes)}
