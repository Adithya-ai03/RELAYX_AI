from __future__ import annotations

import inspect
import io
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from starlette.routing import Router

if "on_startup" not in inspect.signature(Router.__init__).parameters:
    _router_init = Router.__init__

    def _compatible_router_init(self: Router, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("on_startup", None)
        kwargs.pop("on_shutdown", None)
        _router_init(self, *args, **kwargs)

    Router.__init__ = _compatible_router_init  # type: ignore[method-assign]

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.ai import analyze_question
from backend.auth import authenticate_database_user, create_access_token, current_user, require_admin, require_manager
from backend.database import check_database, get_db
from backend.engine import allocate_plan, analyze_shipments, run_simulation, validate_data
from backend.models import AIAnalysis, Allocation, DecisionAudit, RecoveryRun, RejectedRecoveryOption, SimulationRun
from backend.repositories import DATA_NAMES, import_dataset, load_frames, replace_all_datasets

app = FastAPI(title="Ignite Shipment Recovery API", version="2.0.0")
if not hasattr(app, "max_body_size"):
    app.max_body_size = None
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8502", "http://localhost:8503"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class DatasetPayload(BaseModel):
    shipments: list[dict[str, Any]] = Field(default_factory=list)
    vehicles: list[dict[str, Any]] = Field(default_factory=list)
    hubs: list[dict[str, Any]] = Field(default_factory=list)
    routes: list[dict[str, Any]] = Field(default_factory=list)


class PlanRequest(DatasetPayload):
    weights: dict[str, float] | None = None


class SimulationRequest(PlanRequest):
    capacity_factor: float = 1.0
    unavailable_vehicle_ids: list[str] = Field(default_factory=list)
    deadline_shift_hours: float = 0
    route_cost_factor: float = 1.0


class AIRequest(BaseModel):
    question: str
    plan: dict[str, Any] | None = None


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return safe.astype(object).where(pd.notna(safe), None).to_dict(orient="records")


def serialize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in plan.items() if not isinstance(value, pd.DataFrame)}
    for key in ("shipments", "allocations", "candidate_options", "escalated", "rejected", "vehicles"):
        if isinstance(plan.get(key), pd.DataFrame):
            result[key] = records(plan[key])
    return result


def _database_frames(session: Session) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = load_frames(session)
    if any(frame.empty for frame in frames):
        raise HTTPException(status_code=422, detail="Database datasets are incomplete. Import the four CSV datasets before planning.")
    return frames


def _persist_plan(session: Session, plan: dict[str, Any], user: dict[str, str]) -> None:
    run = RecoveryRun(username=user["username"], total_cost=float(plan["total_cost"]), savings=float(plan["savings"]))
    session.add(run)
    session.flush()
    allocations = plan.get("allocations", pd.DataFrame())
    for row in allocations.to_dict(orient="records") if isinstance(allocations, pd.DataFrame) else []:
        session.add(Allocation(run_id=run.run_id, shipment_id=str(row["shipment_id"]), vehicle_id=str(row["vehicle_id"]), strategy=str(row["strategy"]), route=str(row["route"]), hub=row.get("hub"), eta_hours=float(row["eta_hours"]), deadline_margin=float(row["deadline_margin"]), weight_kg=float(row["weight_kg"]), cost=float(row["cost"]), score=float(row["score"]), priority=str(row["priority"]), priority_score=float(row["priority_score"]), reason=str(row["reason"])))
        session.add(DecisionAudit(run_id=run.run_id, shipment_id=str(row["shipment_id"]), username=user["username"], vehicle_id=str(row["vehicle_id"]), strategy=str(row["strategy"]), reason=str(row["reason"]), priority_score=float(row["priority_score"]), deadline_margin=float(row["deadline_margin"]), eta_hours=float(row["eta_hours"]), cost=float(row["cost"])))
    rejected = plan.get("rejected", pd.DataFrame())
    for row in rejected.to_dict(orient="records") if isinstance(rejected, pd.DataFrame) else []:
        session.add(RejectedRecoveryOption(run_id=run.run_id, shipment_id=str(row["shipment_id"]), vehicle_id=str(row["vehicle_id"]), reason=str(row["reason"])))
    session.commit()


@app.get("/health")
def health() -> dict[str, str]:
    connected = check_database()
    return {"status": "ok" if connected else "degraded", "database": "connected" if connected else "disconnected"}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_db)) -> dict[str, str]:
    user = authenticate_database_user(session, form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token(user["username"], user["role"]), "token_type": "bearer", **user}


@app.get("/auth/me")
def me(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
    return user


@app.get("/data")
def data(user: dict[str, str] = Depends(require_manager), session: Session = Depends(get_db)) -> dict[str, Any]:
    return {name: records(frame) for name, frame in zip(DATA_NAMES, load_frames(session))}


@app.post("/data/{name}/upload")
async def upload(name: str, file: UploadFile = File(...), user: dict[str, str] = Depends(require_admin), session: Session = Depends(get_db)) -> dict[str, Any]:
    if name not in DATA_NAMES:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    try:
        frame = pd.read_csv(io.BytesIO(await file.read()))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {error}") from error
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    if name == "shipments" and "deadline" not in frame.columns and "deadline_offset_hours" in frame.columns:
        frame["deadline"] = [now + timedelta(hours=float(value)) for value in frame.pop("deadline_offset_hours")]
        if "created_at" not in frame.columns and "created_offset_hours" in frame.columns:
            frame["created_at"] = [now - timedelta(hours=float(value)) for value in frame.pop("created_offset_hours")]
    if name == "vehicles" and "eta" not in frame.columns and "eta_offset_hours" in frame.columns:
        frame["eta"] = [now + timedelta(hours=float(value)) for value in frame.pop("eta_offset_hours")]
        if "departure_time" not in frame.columns and "departure_offset_hours" in frame.columns:
            frame["departure_time"] = [now + timedelta(hours=float(value)) for value in frame.pop("departure_offset_hours")]
    for column in ("deadline", "created_at", "eta", "departure_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    issues = import_dataset(session, name, frame)
    if issues:
        session.rollback()
        raise HTTPException(status_code=422, detail=issues)
    session.commit()
    return {"dataset": name, "records": len(frame), "validation": []}


@app.post("/data/{name}")
def create_record(name: str, record: dict[str, Any], user: dict[str, str] = Depends(require_admin), session: Session = Depends(get_db)) -> dict[str, Any]:
    if name not in DATA_NAMES:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    existing = load_frames(session)
    frame = pd.DataFrame([record])
    issues = import_dataset(session, name, pd.concat([existing[DATA_NAMES.index(name)], frame], ignore_index=True))
    if issues:
        session.rollback()
        raise HTTPException(status_code=422, detail=issues)
    session.commit()
    return {"dataset": name, "record": record, "validation": []}


@app.post("/data/submit")
def submit_data(payload: DatasetPayload, user: dict[str, str] = Depends(require_admin), session: Session = Depends(get_db)) -> dict[str, Any]:
    frames = tuple(pd.DataFrame(getattr(payload, name)) for name in DATA_NAMES)
    issues = replace_all_datasets(session, frames)  # type: ignore[arg-type]
    if issues:
        session.rollback()
        raise HTTPException(status_code=422, detail=issues)
    session.commit()
    return {"records": {name: len(frame) for name, frame in zip(DATA_NAMES, frames)}, "validation": []}


@app.post("/shipments/analyze")
def analyze(payload: PlanRequest, user: dict[str, str] = Depends(require_manager), session: Session = Depends(get_db)) -> dict[str, Any]:
    shipments, _, _, _ = _database_frames(session)
    return {"shipments": records(analyze_shipments(shipments, payload.weights))}


@app.post("/recovery/plan")
def recovery_plan(payload: PlanRequest, user: dict[str, str] = Depends(require_manager), session: Session = Depends(get_db)) -> dict[str, Any]:
    frames = _database_frames(session)
    issues = validate_data(*frames)
    if issues:
        raise HTTPException(status_code=422, detail=issues)
    plan = allocate_plan(*frames, payload.weights)
    _persist_plan(session, plan, user)
    return serialize_plan(plan)


@app.post("/simulation/run")
def simulation(payload: SimulationRequest, user: dict[str, str] = Depends(require_manager), session: Session = Depends(get_db)) -> dict[str, Any]:
    frames = _database_frames(session)
    plan = run_simulation(*frames, payload.weights, payload.capacity_factor, payload.unavailable_vehicle_ids, payload.deadline_shift_hours, payload.route_cost_factor)
    result = serialize_plan(plan)
    session.add(SimulationRun(username=user["username"], parameters=payload.model_dump(), result=result))
    session.commit()
    return result


@app.get("/audit")
def audit(user: dict[str, str] = Depends(require_manager), session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = session.scalars(select(DecisionAudit).order_by(DecisionAudit.created_at.desc())).all()
    return [{"shipment_id": row.shipment_id, "vehicle_id": row.vehicle_id, "strategy": row.strategy, "reason": row.reason, "priority_score": row.priority_score, "deadline_margin": row.deadline_margin, "eta_hours": row.eta_hours, "cost": row.cost, "username": row.username, "created_at": row.created_at.isoformat()} for row in rows]


@app.post("/ai/analyze")
def ai_analyze(payload: AIRequest, user: dict[str, str] = Depends(require_manager), session: Session = Depends(get_db)) -> dict[str, str]:
    if not payload.plan:
        raise HTTPException(status_code=422, detail="A calculated plan is required")
    frames = tuple(pd.DataFrame(payload.plan.get(name, [])) for name in DATA_NAMES)
    plan = {key: pd.DataFrame(value) if isinstance(value, list) else value for key, value in payload.plan.items()}
    answer, source = analyze_question(plan, payload.question, frames[1], frames[2], frames[3])
    session.add(AIAnalysis(username=user["username"], question=payload.question, answer=answer, source=source))
    session.commit()
    return {"answer": answer, "source": source}
