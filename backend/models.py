from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(80), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="MANAGER")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    audit_records: Mapped[list[DecisionAudit]] = relationship(back_populates="user")


class Shipment(Base):
    __tablename__ = "shipments"

    shipment_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    origin: Mapped[str] = mapped_column(String(120), nullable=False)
    current_location: Mapped[str] = mapped_column(String(120), nullable=False)
    destination: Mapped[str] = mapped_column(String(120), nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    volume_m3: Mapped[float] = mapped_column(Float, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    shipment_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_transport_hours: Mapped[float] = mapped_column(Float, nullable=False)
    delay_probability: Mapped[float] = mapped_column(Float, nullable=False)
    route_risk: Mapped[float] = mapped_column(Float, nullable=False)
    business_category: Mapped[str] = mapped_column(String(80), nullable=False)
    allocations: Mapped[list[Allocation]] = relationship(back_populates="shipment")
    candidates: Mapped[list[CandidateRecoveryOption]] = relationship(back_populates="shipment")
    rejected_options: Mapped[list[RejectedRecoveryOption]] = relationship(back_populates="shipment")

    __table_args__ = (Index("ix_shipments_priority_deadline", "priority", "deadline"),)


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    current_location: Mapped[str] = mapped_column(String(120), nullable=False)
    route_origin: Mapped[str] = mapped_column(String(120), nullable=False)
    route_destination: Mapped[str] = mapped_column(String(120), nullable=False)
    capacity_kg: Mapped[float] = mapped_column(Float, nullable=False)
    current_load_kg: Mapped[float] = mapped_column(Float, nullable=False)
    departure_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    eta: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    vehicle_status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    transport_cost: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String(50), nullable=False)
    average_speed_kmph: Mapped[float] = mapped_column(Float, nullable=False)


class Hub(Base):
    __tablename__ = "hubs"

    hub_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    city: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    handling_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    handling_cost: Mapped[float] = mapped_column(Float, nullable=False)
    operational_status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class Route(Base):
    __tablename__ = "routes"

    route_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    origin: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    destination: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_hours: Mapped[float] = mapped_column(Float, nullable=False)
    base_cost: Mapped[float] = mapped_column(Float, nullable=False)
    delay_risk: Mapped[float] = mapped_column(Float, nullable=False)
    route_status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class RecoveryRun(Base):
    __tablename__ = "recovery_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str | None] = mapped_column(ForeignKey("users.username"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    savings: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    allocations: Mapped[list[Allocation]] = relationship(back_populates="run", cascade="all, delete-orphan")
    candidates: Mapped[list[CandidateRecoveryOption]] = relationship(back_populates="run", cascade="all, delete-orphan")
    rejected_options: Mapped[list[RejectedRecoveryOption]] = relationship(back_populates="run", cascade="all, delete-orphan")
    audits: Mapped[list[DecisionAudit]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Allocation(Base):
    __tablename__ = "allocations"

    allocation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("recovery_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    shipment_id: Mapped[str] = mapped_column(ForeignKey("shipments.shipment_id"), nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(80), nullable=False)
    route: Mapped[str] = mapped_column(String(500), nullable=False)
    hub: Mapped[str | None] = mapped_column(String(255), nullable=True)
    eta_hours: Mapped[float] = mapped_column(Float, nullable=False)
    deadline_margin: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    shipment: Mapped[Shipment] = relationship(back_populates="allocations")
    run: Mapped[RecoveryRun] = relationship(back_populates="allocations")


class CandidateRecoveryOption(Base):
    __tablename__ = "candidate_recovery_options"

    option_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("recovery_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    shipment_id: Mapped[str] = mapped_column(ForeignKey("shipments.shipment_id"), nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String(255), nullable=False)
    strategy: Mapped[str] = mapped_column(String(80), nullable=False)
    option_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    shipment: Mapped[Shipment] = relationship(back_populates="candidates")
    run: Mapped[RecoveryRun] = relationship(back_populates="candidates")


class RejectedRecoveryOption(Base):
    __tablename__ = "rejected_recovery_options"

    rejection_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("recovery_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    shipment_id: Mapped[str] = mapped_column(ForeignKey("shipments.shipment_id"), nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String(255), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    shipment: Mapped[Shipment] = relationship(back_populates="rejected_options")
    run: Mapped[RecoveryRun] = relationship(back_populates="rejected_options")


class DecisionAudit(Base):
    __tablename__ = "decision_audit"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("recovery_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    shipment_id: Mapped[str] = mapped_column(ForeignKey("shipments.shipment_id"), nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(ForeignKey("users.username"), nullable=True, index=True)
    vehicle_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(80), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    deadline_margin: Mapped[float] = mapped_column(Float, nullable=False)
    eta_hours: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    user: Mapped[User | None] = relationship(back_populates="audit_records", foreign_keys=[username])
    run: Mapped[RecoveryRun] = relationship(back_populates="audits")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    simulation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str | None] = mapped_column(ForeignKey("users.username"), nullable=True)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)


class AIAnalysis(Base):
    __tablename__ = "ai_analysis"

    analysis_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str | None] = mapped_column(ForeignKey("users.username"), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)