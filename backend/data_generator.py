from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

CITIES = ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi", "Pune", "Kolkata", "Ahmedabad", "Vijayawada", "Visakhapatnam", "Kochi", "Jaipur"]
PRIORITIES = ["Critical", "High", "Medium", "Normal"]
CATEGORIES = ["Medicine", "Electronics", "Food", "Automotive", "Consumer Goods", "Clothing", "Documents", "Other"]


def _route_rows() -> pd.DataFrame:
    pairs = [("Bengaluru", "Chennai", 350), ("Bengaluru", "Hyderabad", 570), ("Hyderabad", "Delhi", 1_570), ("Chennai", "Vijayawada", 430), ("Mumbai", "Pune", 150), ("Delhi", "Jaipur", 280), ("Mumbai", "Ahmedabad", 530), ("Hyderabad", "Visakhapatnam", 620), ("Chennai", "Kochi", 690), ("Delhi", "Kolkata", 1_500), ("Chennai", "Hyderabad", 600)]
    return pd.DataFrame([{"route_id": f"R{i+1:03d}", "origin": a, "destination": b, "distance_km": d, "estimated_hours": round(d / 48, 1), "base_cost": round(800 + d * 2.2), "delay_risk": round(0.08 + (i % 4) * 0.06, 2), "route_status": "Active"} for i, (a, b, d) in enumerate(pairs)])


def demo_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    shipments = pd.DataFrame([
        {"shipment_id": "S101", "origin": "Bengaluru", "current_location": "Bengaluru", "destination": "Chennai", "weight_kg": 100, "volume_m3": 1.1, "priority": "Critical", "deadline": now + timedelta(hours=4), "status": "Misplaced", "shipment_value": 120000, "created_at": now - timedelta(hours=20), "expected_transport_hours": 7, "delay_probability": .95, "route_risk": .30, "business_category": "Medicine"},
        {"shipment_id": "S102", "origin": "Bengaluru", "current_location": "Bengaluru", "destination": "Chennai", "weight_kg": 150, "volume_m3": 1.6, "priority": "High", "deadline": now + timedelta(hours=12), "status": "Misplaced", "shipment_value": 70000, "created_at": now - timedelta(hours=12), "expected_transport_hours": 8, "delay_probability": .75, "route_risk": .20, "business_category": "Electronics"},
        {"shipment_id": "S103", "origin": "Bengaluru", "current_location": "Bengaluru", "destination": "Hyderabad", "weight_kg": 200, "volume_m3": 2.2, "priority": "Normal", "deadline": now + timedelta(hours=28), "status": "Misplaced", "shipment_value": 30000, "created_at": now - timedelta(hours=7), "expected_transport_hours": 12, "delay_probability": .20, "route_risk": .12, "business_category": "Consumer Goods"},
        {"shipment_id": "S104", "origin": "Bengaluru", "current_location": "Bengaluru", "destination": "Chennai", "weight_kg": 100, "volume_m3": 1.0, "priority": "Critical", "deadline": now + timedelta(hours=9), "status": "Misplaced", "shipment_value": 90000, "created_at": now - timedelta(hours=10), "expected_transport_hours": 8, "delay_probability": .90, "route_risk": .20, "business_category": "Medicine"},
        {"shipment_id": "S105", "origin": "Bengaluru", "current_location": "Bengaluru", "destination": "Delhi", "weight_kg": 150, "volume_m3": 1.8, "priority": "High", "deadline": now + timedelta(hours=48), "status": "Misplaced", "shipment_value": 85000, "created_at": now - timedelta(hours=9), "expected_transport_hours": 30, "delay_probability": .35, "route_risk": .18, "business_category": "Automotive"},
        {"shipment_id": "S106", "origin": "Bengaluru", "current_location": "Bengaluru", "destination": "Chennai", "weight_kg": 500, "volume_m3": 5.0, "priority": "Medium", "deadline": now + timedelta(hours=10), "status": "Misplaced", "shipment_value": 50000, "created_at": now - timedelta(hours=5), "expected_transport_hours": 8, "delay_probability": .40, "route_risk": .20, "business_category": "Automotive"},
    ])
    vehicles = pd.DataFrame([
        {"vehicle_id": "V101", "current_location": "Bengaluru", "route_origin": "Bengaluru", "route_destination": "Chennai", "capacity_kg": 400, "current_load_kg": 0, "departure_time": now, "eta": now + timedelta(hours=7), "vehicle_status": "Available", "transport_cost": 2400, "vehicle_type": "Truck", "average_speed_kmph": 50},
        {"vehicle_id": "V102", "current_location": "Bengaluru", "route_origin": "Bengaluru", "route_destination": "Chennai", "capacity_kg": 250, "current_load_kg": 50, "departure_time": now + timedelta(hours=1), "eta": now + timedelta(hours=8), "vehicle_status": "In Transit", "transport_cost": 1800, "vehicle_type": "Van", "average_speed_kmph": 50},
        {"vehicle_id": "V103", "current_location": "Bengaluru", "route_origin": "Bengaluru", "route_destination": "Hyderabad", "capacity_kg": 350, "current_load_kg": 350, "departure_time": now + timedelta(hours=2), "eta": now + timedelta(hours=14), "vehicle_status": "Available", "transport_cost": 2600, "vehicle_type": "Truck", "average_speed_kmph": 48},
        {"vehicle_id": "V104", "current_location": "Hyderabad", "route_origin": "Hyderabad", "route_destination": "Delhi", "capacity_kg": 500, "current_load_kg": 100, "departure_time": now + timedelta(hours=3), "eta": now + timedelta(hours=34), "vehicle_status": "Available", "transport_cost": 5200, "vehicle_type": "Large Truck", "average_speed_kmph": 50},
        {"vehicle_id": "V105", "current_location": "Bengaluru", "route_origin": "Bengaluru", "route_destination": "Mumbai", "capacity_kg": 800, "current_load_kg": 0, "departure_time": now, "eta": now + timedelta(hours=20), "vehicle_status": "Available", "transport_cost": 4000, "vehicle_type": "Large Truck", "average_speed_kmph": 45},
        {"vehicle_id": "V106", "current_location": "Bengaluru", "route_origin": "Bengaluru", "route_destination": "Chennai", "capacity_kg": 200, "current_load_kg": 0, "departure_time": now, "eta": now + timedelta(hours=6), "vehicle_status": "Unavailable", "transport_cost": 1200, "vehicle_type": "Van", "average_speed_kmph": 55},
        {"vehicle_id": "V107", "current_location": "Chennai", "route_origin": "Chennai", "route_destination": "Hyderabad", "capacity_kg": 400, "current_load_kg": 0, "departure_time": now + timedelta(hours=2), "eta": now + timedelta(hours=14), "vehicle_status": "Available", "transport_cost": 2000, "vehicle_type": "Truck", "average_speed_kmph": 50},
    ])
    hubs = pd.DataFrame([{"hub_id": f"H{i+1:02d}", "city": city, "handling_capacity": 5000, "handling_cost": 180 + i * 20, "operational_status": "Operational"} for i, city in enumerate(["Hyderabad", "Chennai", "Mumbai", "Pune", "Delhi"])])
    return shipments, vehicles, hubs, _route_rows()


def synthetic_data(seed: int = 42, shipment_count: int = 360, vehicle_count: int = 72) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    routes = _route_rows()
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    chosen = routes.sample(shipment_count, replace=True, random_state=seed).reset_index(drop=True)
    priorities = rng.choice(PRIORITIES, shipment_count, p=[.08, .22, .35, .35])
    weights = rng.integers(30, 480, shipment_count)
    priority_bias = pd.Series(priorities).map({"Critical": .30, "High": .18, "Medium": .08, "Normal": .02}).to_numpy()
    shipments = pd.DataFrame({"shipment_id": [f"S{i+1:04d}" for i in range(shipment_count)], "origin": chosen.origin, "current_location": chosen.origin, "destination": chosen.destination, "weight_kg": weights, "volume_m3": np.round(weights / 90, 2), "priority": priorities, "deadline": [now + timedelta(hours=int(x)) for x in rng.integers(6, 72, shipment_count)], "status": "Misplaced", "shipment_value": rng.integers(10000, 150000, shipment_count), "created_at": [now - timedelta(hours=int(x)) for x in rng.integers(1, 72, shipment_count)], "expected_transport_hours": chosen.estimated_hours.to_numpy(), "delay_probability": np.clip(priority_bias + chosen.delay_risk.to_numpy() + rng.uniform(0, .15, shipment_count), 0, .98), "route_risk": chosen.delay_risk.to_numpy(), "business_category": rng.choice(CATEGORIES, shipment_count)})
    vehicle_routes = routes.sample(vehicle_count, replace=True, random_state=seed + 1).reset_index(drop=True)
    capacity = rng.choice([250, 400, 700, 1200], vehicle_count, p=[.3, .35, .25, .1])
    current_load = (capacity * rng.uniform(.1, .65, vehicle_count)).astype(int)
    statuses = rng.choice(["Available", "In Transit", "Delayed", "Unavailable"], vehicle_count, p=[.55, .25, .12, .08])
    vehicles = pd.DataFrame({"vehicle_id": [f"V{i+1:03d}" for i in range(vehicle_count)], "current_location": vehicle_routes.origin, "route_origin": vehicle_routes.origin, "route_destination": vehicle_routes.destination, "capacity_kg": capacity, "current_load_kg": current_load, "departure_time": now + pd.to_timedelta(rng.integers(0, 8, vehicle_count), unit="h"), "eta": now + pd.to_timedelta(vehicle_routes.estimated_hours.to_numpy() + rng.integers(0, 5, vehicle_count), unit="h"), "vehicle_status": statuses, "transport_cost": (vehicle_routes.base_cost.to_numpy() * rng.uniform(.8, 1.2, vehicle_count)).round(), "vehicle_type": rng.choice(["Van", "Truck", "Large Truck"], vehicle_count), "average_speed_kmph": rng.integers(42, 62, vehicle_count)})
    hubs = pd.DataFrame([{"hub_id": f"H{i+1:02d}", "city": city, "handling_capacity": int(rng.integers(2500, 8000)), "handling_cost": int(rng.integers(150, 450)), "operational_status": "Operational" if i != 3 else "Maintenance"} for i, city in enumerate(CITIES[:10])])
    return shipments, vehicles, hubs, routes
