from __future__ import annotations

from typing import Any

import pandas as pd

from backend.decision import answer_from_facts
from backend.llm import complete, configured
from backend.rag import retrieve_context


def answer(question: str, plan: dict[str, Any], vehicles: pd.DataFrame | None = None, hubs: pd.DataFrame | None = None, routes: pd.DataFrame | None = None) -> tuple[str, str]:
    """Answer exactly one latest prompt using RAG facts and optional LLM reasoning."""
    if not question.strip():
        return "Please enter a logistics question.", "Decision engine"
    context = retrieve_context(question, plan, vehicles, hubs, routes)
    if configured():
        try:
            return complete(question, context), "LLM + RAG"
        except RuntimeError as error:
            return f"LLM unavailable ({error}).\n\n{answer_from_facts(plan, question)}", "Decision engine + RAG fallback"
    return answer_from_facts(plan, question), "Decision engine + RAG"
