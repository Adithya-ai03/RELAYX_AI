from backend.data_generator import demo_data
from backend.engine import allocate_plan
from backend.rag import retrieve_context


def test_rag_returns_complete_plan_metrics():
    shipments, vehicles, hubs, routes = demo_data()
    plan = allocate_plan(shipments, vehicles, hubs, routes)
    context = retrieve_context("show current plan", plan, vehicles, hubs, routes)
    assert context["metrics"]["recovered"] == len(plan["allocations"])
    assert context["metrics"]["escalated"] == len(plan["escalated"])
    assert context["routes"]
    assert context["vehicles"]
