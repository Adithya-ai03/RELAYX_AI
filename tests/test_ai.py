import json

from backend.ai import analyze_question, ask_llm, build_data_context
from backend.data_generator import demo_data
from backend.engine import allocate_plan


def _plan():
    shipments, vehicles, hubs, routes = demo_data()
    return allocate_plan(shipments, vehicles, hubs, routes), vehicles, hubs, routes


def test_context_contains_complete_loaded_data():
    plan, vehicles, hubs, routes = _plan()
    context = build_data_context(plan, vehicles, hubs, routes)
    assert len(context["shipments_with_risk_analysis"]) == len(plan["shipments"])
    assert len(context["vehicles"]) == len(vehicles)
    assert len(context["hubs"]) == len(hubs)
    assert len(context["routes"]) == len(routes)
    assert context["allocations"]


def test_manual_question_uses_data_backed_fallback_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan, vehicles, hubs, routes = _plan()
    answer, source = analyze_question(plan, "Which shipments are most at risk?", vehicles, hubs, routes)
    assert source == "Deterministic fallback"
    assert "S101" in answer or "S104" in answer


def test_llm_client_parses_openai_compatible_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "The calculated plan is capacity-feasible."}}]}).encode()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("backend.ai.urlopen", lambda request, timeout: FakeResponse())
    answer = ask_llm("Is the plan feasible?", {"calculated_metrics": {"recovered_count": 3}})
    assert answer == "The calculated plan is capacity-feasible."
