from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a logistics recovery analyst for a shipment piggybacking control tower.
Use only the structured application data provided in the user message. Never invent shipment IDs,
vehicle IDs, routes, costs, ETAs, capacities, or optimization results. Treat capacity, route
compatibility, and deadline feasibility as hard constraints. Clearly distinguish calculated facts
from assumptions. Answer the user's question directly, concisely, and with useful business reasoning.
"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def build_data_context(plan: dict[str, Any], vehicles: pd.DataFrame | None = None, hubs: pd.DataFrame | None = None, routes: pd.DataFrame | None = None) -> dict[str, Any]:
    """Build the complete factual context supplied to either the LLM or fallback."""
    return {
        "shipments_with_risk_analysis": _records(plan.get("shipments")),
        "vehicles": _records(vehicles if vehicles is not None else plan.get("vehicles")),
        "hubs": _records(hubs),
        "routes": _records(routes),
        "allocations": _records(plan.get("allocations")),
        "escalated_shipments": _records(plan.get("escalated")),
        "rejected_candidates": _records(plan.get("rejected")),
        "calculated_metrics": {
            "estimated_recovery_cost": _json_safe(plan.get("total_cost", 0)),
            "dedicated_baseline_cost": _json_safe(plan.get("dedicated_cost", 0)),
            "estimated_savings": _json_safe(plan.get("savings", 0)),
            "recovered_count": len(plan.get("allocations", pd.DataFrame())),
            "escalated_count": len(plan.get("escalated", pd.DataFrame())),
        },
    }


def _fallback_answer(plan: dict[str, Any], question: str) -> str:
    allocations = plan.get("allocations", pd.DataFrame())
    shipments = plan.get("shipments", pd.DataFrame())
    escalated = plan.get("escalated", pd.DataFrame())
    rejected = plan.get("rejected", pd.DataFrame())
    q = question.lower().strip()
    if not q:
        return "Type a logistics question to analyze the current shipment recovery plan."
    if any(term in q for term in ("risk", "urgent", "danger")):
        top = shipments.sort_values("priority_score", ascending=False).head(5)
        return "Most at-risk shipments: " + "; ".join(f"{row.shipment_id} ({row.deadline_risk_label}, priority score {row.priority_score:.2f}, remaining {row.remaining_hours:.1f}h)" for row in top.itertuples()) + "."
    if any(term in q for term in ("save", "cost", "expensive", "cheap")):
        return f"Calculated estimated recovery cost is ₹{plan.get('total_cost', 0):,.0f} versus ₹{plan.get('dedicated_cost', 0):,.0f} dedicated baseline, for estimated savings of ₹{plan.get('savings', 0):,.0f}."
    if any(term in q for term in ("escalat", "unrecover", "cannot recover")):
        ids = ", ".join(escalated.shipment_id.tolist()) or "none"
        return f"Escalated shipments: {ids}. The optimizer could not find a route, capacity, and deadline-feasible piggyback option. Rejected candidate records available: {len(rejected)}."
    if "vehicle" in q or "assign" in q or "selected" in q:
        if allocations.empty:
            return "No vehicle allocation is currently feasible."
        return "Current assignments: " + "; ".join(f"{row.shipment_id} → {row.vehicle_id} ({row.strategy}, ₹{row.cost:,.0f}, {row.deadline_margin:.1f}h margin)" for row in allocations.itertuples()) + "."
    if "priorit" in q:
        row = shipments.iloc[0]
        return f"{row.shipment_id} is currently first because its weighted priority score is {row.priority_score:.2f}; urgency is {row.urgency_score:.2f}, deadline risk is {row.deadline_risk:.2f}, business priority is {row.priority}, and delay probability is {row.delay_probability:.0%}."
    if allocations.empty:
        return f"The complete plan contains {len(shipments)} analyzed shipments but no feasible allocations. Escalated count is {len(escalated)}."
    first = allocations.iloc[0]
    return f"The plan recovered {len(allocations)} of {len(shipments)} analyzed shipments. Example recommendation: {first.shipment_id} on {first.vehicle_id} using {first.strategy.lower()}, with estimated cost ₹{first.cost:,.0f} and {first.deadline_margin:.1f} hours of deadline margin."


def _llm_settings() -> tuple[str | None, str, str]:
    return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def ask_llm(question: str, context: dict[str, Any]) -> str:
    api_key, base_url, model = _llm_settings()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    payload = {"model": model, "temperature": 0.1, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Question:\n{question}\n\nComplete application data:\n{json.dumps(context, ensure_ascii=False)}"}]}
    request = Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"LLM request failed: {error}") from error


def analyze_question(plan: dict[str, Any], question: str, vehicles: pd.DataFrame | None = None, hubs: pd.DataFrame | None = None, routes: pd.DataFrame | None = None) -> tuple[str, str]:
    """Answer a manual question with the LLM when configured, otherwise fallback."""
    context = build_data_context(plan, vehicles, hubs, routes)
    api_key, _, _ = _llm_settings()
    if api_key:
        try:
            return ask_llm(question, context), "LLM"
        except RuntimeError as error:
            return f"LLM unavailable ({error}). Deterministic answer from the same application data:\n\n{_fallback_answer(plan, question)}", "Fallback"
    return _fallback_answer(plan, question), "Deterministic fallback"


def explain_plan(plan: dict[str, Any], question: str = "Explain the current recovery plan.") -> str:
    return analyze_question(plan, question)[0]
