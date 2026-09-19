import json

from backend.agent import answer
from backend.data_generator import demo_data
from backend.engine import allocate_plan
from backend.rag import retrieve_context


def build_plan():
    shipments, vehicles, hubs, routes = demo_data()
    return allocate_plan(shipments, vehicles, hubs, routes), vehicles, hubs, routes


def test_agent_answers_latest_prompt_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan, vehicles, hubs, routes = build_plan()
    answer_text, source = answer("Which shipments are most at risk?", plan, vehicles, hubs, routes)
    assert "S101" in answer_text or "S104" in answer_text
    assert source == "Decision engine + RAG"


def test_rag_retrieves_requested_shipment():
    plan, vehicles, hubs, routes = build_plan()
    context = retrieve_context("Why was S101 prioritized?", plan, vehicles, hubs, routes)
    ids = {row["shipment_id"] for row in context["shipments"]}
    assert "S101" in ids


def test_agent_falls_back_when_llm_fails(monkeypatch):
    plan, vehicles, hubs, routes = build_plan()
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setattr("backend.agent.complete", lambda *_args: (_ for _ in ()).throw(RuntimeError("offline")))
    answer_text, source = answer("How much cost is saved?", plan, vehicles, hubs, routes)
    assert "estimated" in answer_text.lower()
    assert source == "Decision engine + RAG fallback"
