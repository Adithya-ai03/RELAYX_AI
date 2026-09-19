from __future__ import annotations

from typing import Any

import pandas as pd


def _frame(plan: dict[str, Any], key: str) -> pd.DataFrame:
    value = plan.get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def answer_from_facts(plan: dict[str, Any], question: str) -> str:
    """Answer broad logistics questions from calculated facts without inventing data."""
    shipments = _frame(plan, "shipments")
    allocations = _frame(plan, "allocations")
    escalated = _frame(plan, "escalated")
    rejected = _frame(plan, "rejected")
    query = question.strip()
    lower = query.lower()
    if not query:
        return "Please enter a logistics question."
    if shipments.empty:
        return "No shipment data is available for analysis."
    if any(word in lower for word in ("risk", "urgent", "danger", "deadline")):
        ranked = shipments.sort_values("priority_score", ascending=False).head(5)
        return "Risk view: " + "; ".join(f"{row.shipment_id}: {row.deadline_risk_label}, {row.remaining_hours:.1f}h remaining, priority score {row.priority_score:.2f}" for row in ranked.itertuples()) + "."
    if any(word in lower for word in ("cost", "save", "saving", "price", "expensive")):
        return f"Estimated recovery cost is ₹{plan.get('total_cost', 0):,.0f}; the dedicated baseline is ₹{plan.get('dedicated_cost', 0):,.0f}; estimated savings are ₹{plan.get('savings', 0):,.0f}."
    if any(word in lower for word in ("escalat", "unrecover", "failed", "not recover")):
        ids = ", ".join(escalated["shipment_id"].astype(str)) if "shipment_id" in escalated else "none"
        return f"Escalated shipments: {ids or 'none'}. Rejected candidate explanations available: {len(rejected)}. Alternatives are a dedicated vehicle, next route, alternate hub, or manual intervention."
    if any(word in lower for word in ("vehicle", "assign", "allocation", "selected", "route")) and not allocations.empty:
        return "Current calculated assignments: " + "; ".join(f"{row.shipment_id} → {row.vehicle_id}, {row.strategy}, ₹{row.cost:,.0f}, {row.deadline_margin:.1f}h deadline margin" for row in allocations.itertuples()) + "."
    if any(word in lower for word in ("priority", "priorit", "important")):
        row = shipments.sort_values("priority_score", ascending=False).iloc[0]
        return f"{row.shipment_id} ranks first with score {row.priority_score:.2f}: urgency {row.urgency_score:.2f}, deadline risk {row.deadline_risk:.2f}, business priority {row.priority}, delay probability {row.delay_probability:.0%}."
    recovered = len(allocations)
    return f"The current plan contains {len(shipments)} shipments, recovers {recovered}, and escalates {len(escalated)}. Ask about a shipment, vehicle, route, deadline, risk, cost, capacity, or recovery decision for a more specific analysis."
