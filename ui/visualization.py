from __future__ import annotations

import pandas as pd
import plotly.express as px

COLORS = {
    "Recovered": "#16A34A", "Escalated": "#EF4444", "At Risk": "#F59E0B", "Pending": "#64748B",
    "Critical": "#EF4444", "High": "#F59E0B", "Medium": "#2563EB", "Normal": "#64748B",
}


def _style(fig, height: int = 320):
    fig.update_layout(height=height, margin={"l": 12, "r": 12, "t": 38, "b": 12}, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"family": "Inter, system-ui, sans-serif", "color": "#0F172A"}, title_font={"size": 16, "color": "#0F172A"}, legend_title_text="")
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")
    return fig


def status_chart(plan: dict):
    allocations = plan["allocations"]
    values = pd.DataFrame({"Status": ["Recovered", "Escalated"], "Count": [len(allocations), len(plan["escalated"])]})
    return _style(px.pie(values, names="Status", values="Count", hole=.62, color="Status", color_discrete_map=COLORS, title="Shipment recovery status"))


def priority_chart(shipments: pd.DataFrame):
    values = shipments.priority.value_counts().rename_axis("Priority").reset_index(name="Shipments")
    return _style(px.bar(values, x="Shipments", y="Priority", orientation="h", color="Priority", color_discrete_map=COLORS, title="Priority distribution"))


def risk_chart(shipments: pd.DataFrame):
    fig = px.scatter(shipments, x="remaining_hours", y="priority_score", size="weight_kg", color="priority", hover_name="shipment_id", hover_data=["destination", "deadline_risk_label", "delay_probability"], color_discrete_map=COLORS, title="Deadline risk and priority")
    return _style(fig, 360)


def fleet_chart(vehicles: pd.DataFrame):
    frame = vehicles.copy()
    frame["current_load"] = frame["current_load_kg"]
    frame["remaining_capacity"] = (frame["capacity_kg"] - frame["current_load_kg"]).clip(lower=0)
    long = frame.head(24).melt(id_vars="vehicle_id", value_vars=["current_load", "remaining_capacity"], var_name="Measure", value_name="Kg")
    return _style(px.bar(long, x="vehicle_id", y="Kg", color="Measure", barmode="stack", title="Fleet capacity utilization"), 360)
