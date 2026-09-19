from __future__ import annotations

from typing import Any

import pandas as pd


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def retrieve_context(question: str, plan: dict[str, Any], vehicles: pd.DataFrame | None = None, hubs: pd.DataFrame | None = None, routes: pd.DataFrame | None = None, limit: int = 12) -> dict[str, Any]:
    """Lightweight deterministic RAG: retrieve matching IDs/locations plus plan summaries."""
    query = question.lower()
    tokens = {token for token in query.replace("?", " ").replace(",", " ").split() if len(token) > 2}
    result: dict[str, Any] = {"question": question, "retrieval_mode": "deterministic keyword retrieval", "shipments": [], "vehicles": [], "hubs": [], "routes": [], "allocations": [], "escalated": [], "metrics": {}}
    for key, frame in (("shipments", plan.get("shipments")), ("allocations", plan.get("allocations")), ("escalated", plan.get("escalated")), ("vehicles", vehicles), ("hubs", hubs), ("routes", routes)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        searchable = frame.apply(lambda row: " ".join(str(value).lower() for value in row.tolist()), axis=1)
        matches = frame[searchable.map(lambda text: any(token in text for token in tokens))]
        result[key] = _records((matches if not matches.empty else frame).head(limit))
    result["metrics"] = {"recovered": len(plan.get("allocations", pd.DataFrame())), "escalated": len(plan.get("escalated", pd.DataFrame())), "estimated_cost": plan.get("total_cost", 0), "estimated_savings": plan.get("savings", 0)}
    return result
