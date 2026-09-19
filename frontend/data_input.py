from __future__ import annotations

import io
from datetime import datetime, time
from typing import Any

import pandas as pd
import streamlit as st

from frontend.api_client import ApiError, RecoveryApi

DATASETS = ("shipments", "vehicles", "hubs", "routes")
DATASET_LABELS = {"shipments": "Shipments", "vehicles": "Vehicles", "hubs": "Hubs", "routes": "Routes"}
PREVIEW_ROWS = 50
PRIORITIES = ["Critical", "High", "Medium", "Normal"]
SHIPMENT_STATUSES = ["Misplaced", "In Transit", "Delivered"]
VEHICLE_STATUSES = ["Available", "In Transit", "Delayed", "Unavailable"]
VEHICLE_TYPES = ["Van", "Truck", "Large Truck"]
HUB_STATUSES = ["Operational", "Maintenance"]
ROUTE_STATUSES = ["Active", "Inactive"]
CATEGORIES = ["Medicine", "Electronics", "Food", "Automotive", "Consumer Goods", "Clothing", "Documents", "Other"]


def _datetime_value(value: datetime | None = None) -> datetime:
    current = value or datetime.now()
    return current.replace(second=0, microsecond=0)


def datetime_widget(label: str, key: str, value: datetime | None = None) -> datetime:
    default = _datetime_value(value)
    date_column, time_column = st.columns(2)
    with date_column:
        chosen_date = st.date_input(label, value=default.date(), key=f"{key}_date")
    with time_column:
        chosen_time = st.time_input("Time", value=default.time().replace(second=0, microsecond=0), key=f"{key}_time")
    if isinstance(chosen_time, time):
        return datetime.combine(chosen_date, chosen_time)
    return datetime.combine(chosen_date, time(hour=default.hour, minute=default.minute))


def json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if hasattr(value, "item"):
        return value.item()
    return value


def frames_from_payload(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in DATASETS:
        frame = pd.DataFrame(payload.get(name, []))
        if name == "shipments":
            for column in ("deadline", "created_at"):
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if name == "vehicles":
            for column in ("departure_time", "eta"):
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frames[name] = frame
    return frames


def records_from_frames(frames: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for name in DATASETS:
        frame = frames.get(name, pd.DataFrame())
        serializable = frame.copy()
        for column in serializable.columns:
            if pd.api.types.is_datetime64_any_dtype(serializable[column]):
                serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
        payload[name] = serializable.astype(object).where(pd.notna(serializable), None).to_dict(orient="records")
    return payload


def location_choices(frames: dict[str, pd.DataFrame]) -> list[str]:
    cities: set[str] = set()
    routes = frames.get("routes", pd.DataFrame())
    hubs = frames.get("hubs", pd.DataFrame())
    shipments = frames.get("shipments", pd.DataFrame())
    vehicles = frames.get("vehicles", pd.DataFrame())
    for frame, columns in (
        (routes, ("origin", "destination")),
        (hubs, ("city",)),
        (shipments, ("origin", "current_location", "destination")),
        (vehicles, ("current_location", "route_origin", "route_destination")),
    ):
        for column in columns:
            if column in frame.columns:
                cities.update(str(value) for value in frame[column].dropna().unique())
    return sorted(cities) or ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi"]


def handle_api_error(error: ApiError) -> None:
    message = str(error)
    st.error(message)
    if "Session expired" in message:
        st.session_state.auth_error = message
        for key in ("authenticated", "user", "role", "remember", "access_token"):
            st.session_state.pop(key, None)
        st.rerun()


def bump_data_revision() -> None:
    st.session_state.data_revision = int(st.session_state.get("data_revision", 0)) + 1


def fetch_backend_data(token: str) -> dict[str, Any]:
    return RecoveryApi(token=token).get_data()


def render_dataset_preview(payload: dict[str, Any]) -> None:
    st.markdown("## Current Dataset")
    frames = frames_from_payload(payload)
    counts = st.columns(4)
    for column, name in zip(counts, DATASETS):
        with column:
            st.metric(DATASET_LABELS[name], f"{len(frames[name])} records")
    for name in DATASETS:
        frame = frames[name]
        with st.expander(f"{DATASET_LABELS[name]} ({len(frame)} records)", expanded=False):
            if frame.empty:
                st.info(f"No {name} records in the current backend dataset.")
            else:
                st.caption(f"Showing first {min(PREVIEW_ROWS, len(frame))} of {len(frame)} records.")
                st.dataframe(frame.head(PREVIEW_ROWS), width="stretch", hide_index=True)


def location_input(label: str, cities: list[str], key: str) -> str:
    options = list(cities) + ["Other (type below)"]
    choice = st.selectbox(label, options, key=key)
    if choice == "Other (type below)":
        return st.text_input(f"{label} (custom)", key=f"{key}_custom").strip()
    return choice


def _required_text(label: str, key: str) -> str:
    return st.text_input(label, key=key).strip()


def shipment_form(cities: list[str]) -> dict[str, Any] | None:
    st.markdown("## Shipment Details")
    shipment_id = _required_text("Shipment ID", "manual_shipment_id")
    origin = location_input("Origin", cities, "manual_shipment_origin")
    current_location = location_input("Current location", cities, "manual_shipment_current")
    destination = location_input("Destination", cities, "manual_shipment_destination")
    weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=100.0, step=1.0, key="manual_shipment_weight")
    volume_m3 = st.number_input("Volume (m3)", min_value=0.0, value=1.0, step=0.1, key="manual_shipment_volume")
    priority = st.selectbox("Priority", PRIORITIES, key="manual_shipment_priority")
    deadline = datetime_widget("Deadline", "manual_shipment_deadline")
    status = st.selectbox("Status", SHIPMENT_STATUSES, key="manual_shipment_status")
    shipment_value = st.number_input("Shipment value", min_value=0.0, value=50000.0, step=1000.0, key="manual_shipment_value")
    created_at = datetime_widget("Created at", "manual_shipment_created", value=datetime.now())
    expected_transport_hours = st.number_input("Expected transport hours", min_value=0.0, value=8.0, step=0.5, key="manual_shipment_hours")
    delay_probability = st.number_input("Delay probability", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="manual_shipment_delay")
    route_risk = st.number_input("Route risk", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="manual_shipment_risk")
    business_category = st.selectbox("Business category", CATEGORIES, key="manual_shipment_category")
    if not shipment_id:
        return None
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "current_location": current_location,
        "destination": destination,
        "weight_kg": float(weight_kg),
        "volume_m3": float(volume_m3),
        "priority": priority,
        "deadline": json_ready(deadline),
        "status": status,
        "shipment_value": float(shipment_value),
        "created_at": json_ready(created_at),
        "expected_transport_hours": float(expected_transport_hours),
        "delay_probability": float(delay_probability),
        "route_risk": float(route_risk),
        "business_category": business_category,
    }


def vehicle_form(cities: list[str]) -> dict[str, Any] | None:
    st.markdown("## Vehicle Details")
    vehicle_id = _required_text("Vehicle ID", "manual_vehicle_id")
    current_location = location_input("Current location", cities, "manual_vehicle_current")
    route_origin = location_input("Route origin", cities, "manual_vehicle_origin")
    route_destination = location_input("Route destination", cities, "manual_vehicle_destination")
    capacity_kg = st.number_input("Capacity (kg)", min_value=0.0, value=400.0, step=10.0, key="manual_vehicle_capacity")
    current_load_kg = st.number_input("Current load (kg)", min_value=0.0, value=0.0, step=10.0, key="manual_vehicle_load")
    departure_time = datetime_widget("Departure time", "manual_vehicle_departure")
    eta = datetime_widget("ETA", "manual_vehicle_eta")
    vehicle_status = st.selectbox("Vehicle status", VEHICLE_STATUSES, key="manual_vehicle_status")
    transport_cost = st.number_input("Transport cost", min_value=0.0, value=2400.0, step=50.0, key="manual_vehicle_cost")
    vehicle_type = st.selectbox("Vehicle type", VEHICLE_TYPES, key="manual_vehicle_type")
    average_speed_kmph = st.number_input("Average speed (km/h)", min_value=0.0, value=50.0, step=1.0, key="manual_vehicle_speed")
    if not vehicle_id:
        return None
    if current_load_kg > capacity_kg:
        st.warning("Current load cannot be above capacity.")
        return None
    return {
        "vehicle_id": vehicle_id,
        "current_location": current_location,
        "route_origin": route_origin,
        "route_destination": route_destination,
        "capacity_kg": float(capacity_kg),
        "current_load_kg": float(current_load_kg),
        "departure_time": json_ready(departure_time),
        "eta": json_ready(eta),
        "vehicle_status": vehicle_status,
        "transport_cost": float(transport_cost),
        "vehicle_type": vehicle_type,
        "average_speed_kmph": float(average_speed_kmph),
    }


def hub_form(cities: list[str]) -> dict[str, Any] | None:
    st.markdown("## Hub Details")
    hub_id = _required_text("Hub ID", "manual_hub_id")
    city = location_input("City", cities, "manual_hub_city")
    handling_capacity = st.number_input("Handling capacity", min_value=0, value=5000, step=100, key="manual_hub_capacity")
    handling_cost = st.number_input("Handling cost", min_value=0, value=180, step=10, key="manual_hub_cost")
    operational_status = st.selectbox("Operational status", HUB_STATUSES, key="manual_hub_status")
    if not hub_id:
        return None
    return {
        "hub_id": hub_id,
        "city": city,
        "handling_capacity": int(handling_capacity),
        "handling_cost": int(handling_cost),
        "operational_status": operational_status,
    }


def route_form(cities: list[str]) -> dict[str, Any] | None:
    st.markdown("## Route Details")
    route_id = _required_text("Route ID", "manual_route_id")
    origin = location_input("Origin", cities, "manual_route_origin")
    destination = location_input("Destination", cities, "manual_route_destination")
    distance_km = st.number_input("Distance (km)", min_value=0.0, value=350.0, step=1.0, key="manual_route_distance")
    estimated_hours = st.number_input("Estimated hours", min_value=0.0, value=7.0, step=0.1, key="manual_route_hours")
    base_cost = st.number_input("Base cost", min_value=0.0, value=1500.0, step=10.0, key="manual_route_cost")
    delay_risk = st.number_input("Delay risk", min_value=0.0, max_value=1.0, value=0.08, step=0.01, key="manual_route_risk")
    route_status = st.selectbox("Route status", ROUTE_STATUSES, key="manual_route_status")
    if not route_id:
        return None
    return {
        "route_id": route_id,
        "origin": origin,
        "destination": destination,
        "distance_km": float(distance_km),
        "estimated_hours": float(estimated_hours),
        "base_cost": float(base_cost),
        "delay_risk": float(delay_risk),
        "route_status": route_status,
    }


def collect_manual_record(dataset_name: str, cities: list[str]) -> dict[str, Any] | None:
    builders = {
        "shipments": shipment_form,
        "vehicles": vehicle_form,
        "hubs": hub_form,
        "routes": route_form,
    }
    return builders[dataset_name](cities)


def id_column(dataset_name: str) -> str:
    return {"shipments": "shipment_id", "vehicles": "vehicle_id", "hubs": "hub_id", "routes": "route_id"}[dataset_name]


def _flash(kind: str, text: str, extra: str | None = None) -> None:
    st.session_state.data_input_flash = {"kind": kind, "text": text, "extra": extra}
    bump_data_revision()
    st.rerun()


def local_append_record(payload: dict[str, Any], dataset_name: str, record: dict[str, Any]) -> dict[str, Any]:
    working = {name: list(values) for name, values in payload.items()}
    working.setdefault(dataset_name, [])
    working[dataset_name] = [*working[dataset_name], record]
    return working


def local_append_csv(payload: dict[str, Any], dataset_name: str, csv_text: str) -> dict[str, Any]:
    working = {name: list(values) for name, values in payload.items()}
    frame = pd.read_csv(io.StringIO(csv_text))
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    working.setdefault(dataset_name, [])
    working[dataset_name] = [*working[dataset_name], *rows]
    return working


def render_data_input_page(fallback_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    st.markdown("### DATA INPUT")
    st.caption("Add records manually or upload a CSV. Both methods work in the local app and with the authenticated API.")
    token = st.session_state.get("access_token")
    if st.session_state.get("role") != "ADMIN":
        st.warning("You do not have permission to perform this action.")
        return

    flash = st.session_state.pop("data_input_flash", None)
    if flash:
        if flash["kind"] == "success":
            st.success(flash["text"])
        else:
            st.error(flash["text"])
        if flash.get("extra"):
            st.write(flash["extra"])

    payload = st.session_state.get("backend_dataset") or records_from_frames(dict(zip(DATASETS, fallback_frames)))
    if token:
        api = RecoveryApi(token=token)
        try:
            payload = fetch_backend_data(token)
            st.session_state.backend_dataset = payload
        except ApiError as error:
            handle_api_error(error)
            payload = st.session_state.get("backend_dataset") or records_from_frames(dict(zip(DATASETS, fallback_frames)))

    if st.button("Refresh Data"):
        if token:
            try:
                payload = fetch_backend_data(token)
                st.session_state.backend_dataset = payload
                _flash("success", "Backend dataset refreshed.")
            except ApiError as error:
                handle_api_error(error)
        else:
            payload = st.session_state.get("backend_dataset") or records_from_frames(dict(zip(DATASETS, fallback_frames)))
            st.session_state.backend_dataset = payload
            st.success("Local dataset refreshed.")

    frames = frames_from_payload(payload)
    cities = location_choices(frames)
    st.markdown("## Manual Entry")
    manual_label = st.selectbox("Manual record type", list(DATASET_LABELS.values()), key="manual_dataset")
    manual_name = next(name for name, label in DATASET_LABELS.items() if label == manual_label)
    manual_record = collect_manual_record(manual_name, cities)
    if st.button("Save manual record", type="primary"):
        if manual_record is None:
            st.error("Complete the required ID field before saving.")
        elif token:
            try:
                api.create_record(manual_name, manual_record)
                st.session_state.backend_dataset = fetch_backend_data(token)
                _flash("success", f"{manual_label} record saved to PostgreSQL.")
            except ApiError as error:
                st.error(str(error))
        else:
            st.error("A database-backed login is required to save records.")

    st.markdown("## CSV Upload")
    csv_tab = st.container()

    with csv_tab:
        st.markdown("Select Method: **CSV Upload**")
        dataset_label = st.selectbox("Dataset", list(DATASET_LABELS.values()), key="csv_dataset")
        dataset_name = next(name for name, label in DATASET_LABELS.items() if label == dataset_label)
        csv_text = st.text_input(
            "CSV content",
            key="csv_content",
            placeholder="Paste a CSV row or use\\n between rows",
        )
        if st.button("Upload CSV", type="primary"):
            if not csv_text.strip():
                st.error("Paste CSV content before uploading.")
            else:
                if token:
                    try:
                        csv_text = csv_text.replace("\\n", "\n")
                        pd.read_csv(io.StringIO(csv_text))
                        uploaded = io.BytesIO(csv_text.encode("utf-8"))
                        uploaded.name = f"{dataset_name}.csv"
                        result = api.upload_csv(dataset_name, uploaded)
                        extra = f"{DATASET_LABELS[dataset_name]} uploaded records: {result.get('records', 0)}"
                        st.session_state.backend_dataset = fetch_backend_data(token)
                        _flash("success", "CSV uploaded successfully", extra)
                    except (pd.errors.ParserError, UnicodeDecodeError) as error:
                        st.error(f"Invalid CSV content: {error}")
                    except ApiError as error:
                        handle_api_error(error)
                else:
                    try:
                        current = st.session_state.get("backend_dataset") or records_from_frames(dict(zip(DATASETS, fallback_frames)))
                        st.session_state.backend_dataset = local_append_csv(current, dataset_name, csv_text.replace("\\n", "\n"))
                        _flash("success", f"{DATASET_LABELS[dataset_name]} CSV added locally.")
                    except (pd.errors.ParserError, UnicodeDecodeError) as error:
                        st.error(f"Invalid CSV content: {error}")

    render_dataset_preview(st.session_state.get("backend_dataset", payload))
