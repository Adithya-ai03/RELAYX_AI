from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backend.data_generator import demo_data, synthetic_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"


def _normalize_data(datasets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep demo and full mode on the exact same typed, validated data contract."""
    shipments, vehicles, hubs, routes = datasets
    shipments = shipments.copy()
    vehicles = vehicles.copy()
    shipments["deadline"] = pd.to_datetime(shipments["deadline"], errors="coerce")
    shipments["created_at"] = pd.to_datetime(shipments["created_at"], errors="coerce")
    vehicles["departure_time"] = pd.to_datetime(vehicles["departure_time"], errors="coerce")
    vehicles["eta"] = pd.to_datetime(vehicles["eta"], errors="coerce")
    for column in ("weight_kg", "volume_m3", "shipment_value", "expected_transport_hours", "delay_probability", "route_risk"):
        shipments[column] = pd.to_numeric(shipments[column], errors="coerce").fillna(0)
    for column in ("capacity_kg", "current_load_kg", "transport_cost", "average_speed_kmph"):
        vehicles[column] = pd.to_numeric(vehicles[column], errors="coerce").fillna(0)
    vehicles["capacity_kg"] = vehicles["capacity_kg"].clip(lower=0)
    vehicles["current_load_kg"] = vehicles[["current_load_kg", "capacity_kg"]].min(axis=1).clip(lower=0)
    vehicles["available_capacity_kg"] = (vehicles["capacity_kg"] - vehicles["current_load_kg"]).clip(lower=0)
    return shipments, vehicles, hubs, routes


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def _load_demo_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    shipments = _read_csv("shipments.csv")
    shipments["deadline"] = [now + timedelta(hours=float(value)) for value in shipments.pop("deadline_offset_hours")]
    shipments["created_at"] = [now - timedelta(hours=float(value)) for value in shipments.pop("created_offset_hours")]
    vehicles = _read_csv("vehicles.csv")
    vehicles["departure_time"] = [now + timedelta(hours=float(value)) for value in vehicles.pop("departure_offset_hours")]
    vehicles["eta"] = [now + timedelta(hours=float(value)) for value in vehicles.pop("eta_offset_hours")]
    hubs = _read_csv("hubs.csv")
    routes = _read_csv("routes.csv")
    return _normalize_data((shipments, vehicles, hubs, routes))


def _write_bundle(directory: Path, datasets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("shipments.csv", "vehicles.csv", "hubs.csv", "routes.csv"), datasets):
        frame.to_csv(directory / name, index=False)


def load_csv_data(demo: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load editable CSV data; generate a larger CSV bundle on demand."""
    if demo:
        try:
            return _load_demo_csv()
        except (FileNotFoundError, KeyError, ValueError):
            return demo_data()

    generated_shipments = GENERATED_DIR / "shipments.csv"
    if not generated_shipments.exists():
        datasets = synthetic_data(seed=42, shipment_count=360, vehicle_count=72)
        _write_bundle(GENERATED_DIR, datasets)
    datasets = tuple(pd.read_csv(GENERATED_DIR / name, parse_dates=date_columns) for name, date_columns in [
        ("shipments.csv", ["deadline", "created_at"]),
        ("vehicles.csv", ["departure_time", "eta"]),
        ("hubs.csv", []),
        ("routes.csv", []),
    ])
    # Regenerate only when a persisted demo bundle has aged past its planning window.
    if datasets[0]["deadline"].max() < pd.Timestamp.now() or datasets[1]["eta"].max() < pd.Timestamp.now():
        datasets = synthetic_data(seed=42, shipment_count=360, vehicle_count=72)
        _write_bundle(GENERATED_DIR, datasets)
    return _normalize_data(datasets)
