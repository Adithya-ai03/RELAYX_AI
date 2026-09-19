from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.engine import validate_data
from backend.models import Hub, Route, Shipment, Vehicle

DATA_NAMES = ("shipments", "vehicles", "hubs", "routes")
MODEL_BY_NAME = {"shipments": Shipment, "vehicles": Vehicle, "hubs": Hub, "routes": Route}
ID_COLUMNS = {"shipments": "shipment_id", "vehicles": "vehicle_id", "hubs": "hub_id", "routes": "route_id"}
REQUIRED_COLUMNS = {
    "shipments": {"shipment_id", "origin", "current_location", "destination", "weight_kg", "volume_m3", "priority", "deadline", "status", "shipment_value", "created_at", "expected_transport_hours", "delay_probability", "route_risk", "business_category"},
    "vehicles": {"vehicle_id", "current_location", "route_origin", "route_destination", "capacity_kg", "current_load_kg", "departure_time", "eta", "vehicle_status", "transport_cost", "vehicle_type", "average_speed_kmph"},
    "hubs": {"hub_id", "city", "handling_capacity", "handling_cost", "operational_status"},
    "routes": {"route_id", "origin", "destination", "distance_km", "estimated_hours", "base_cost", "delay_risk", "route_status"},
}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def model_to_frame(session: Session, name: str) -> pd.DataFrame:
    model = MODEL_BY_NAME[name]
    rows = [row.__dict__.copy() for row in session.scalars(select(model)).all()]
    for row in rows:
        row.pop("_sa_instance_state", None)
    return pd.DataFrame(rows).drop(columns=["created_at"], errors="ignore") if name != "shipments" else pd.DataFrame(rows)


def load_frames(session: Session) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(model_to_frame(session, name) for name in DATA_NAMES)  # type: ignore[return-value]


def validate_frame(name: str, frame: pd.DataFrame) -> list[str]:
    missing = sorted(REQUIRED_COLUMNS[name] - set(frame.columns))
    if missing:
        return [f"{name} is missing required columns: {', '.join(missing)}"]
    if frame[ID_COLUMNS[name]].duplicated().any():
        return [f"Duplicate {ID_COLUMNS[name]} values found in {name}."]
    return []


def replace_dataset(session: Session, name: str, frame: pd.DataFrame) -> None:
    model = MODEL_BY_NAME[name]
    identity = ID_COLUMNS[name]
    model_columns = {column.name for column in model.__table__.columns}
    for record in frame_to_records(frame):
        record = {key: value for key, value in record.items() if key in model_columns}
        key = record.get(identity)
        existing = session.get(model, key)
        if existing is None:
            session.add(model(**record))
        else:
            for column, value in record.items():
                setattr(existing, column, value)


def import_dataset(session: Session, name: str, frame: pd.DataFrame) -> list[str]:
    issues = validate_frame(name, frame)
    if issues:
        return issues
    candidate = {dataset: model_to_frame(session, dataset) for dataset in DATA_NAMES}
    candidate[name] = frame
    try:
        issues = validate_data(*(candidate[dataset] for dataset in DATA_NAMES))
    except (KeyError, TypeError, ValueError) as error:
        return [f"Invalid {name} data: {error}"]
    if issues:
        return issues
    replace_dataset(session, name, frame)
    return []


def replace_all_datasets(session: Session, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> list[str]:
    for name, frame in zip(DATA_NAMES, frames):
        issues = validate_frame(name, frame)
        if issues:
            return issues
    issues = validate_data(*frames)
    if issues:
        return issues
    for name, frame in zip(DATA_NAMES, frames):
        replace_dataset(session, name, frame)
    return []