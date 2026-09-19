from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent import answer as answer_question
from backend.data_loader import load_csv_data
from backend.engine import DEFAULT_WEIGHTS, allocate_plan, evaluate_candidates, run_simulation
from frontend.api_client import ApiError, RecoveryApi
from frontend.data_input import bump_data_revision, frames_from_payload, handle_api_error, render_data_input_page
from ui.visualization import fleet_chart, priority_chart, risk_chart, status_chart


st.set_page_config(page_title="RELAYX | AI Shipment Recovery Control Tower", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, h4, h5, h6, p, label, span, li { color: #0F172A !important; }
h1, h2, h3, h4, h5, h6 { letter-spacing: 0; }
.stApp { background: #F5F7FA; }
.block-container { padding-top: .7rem; max-width: 1600px; }
.stSidebar, [data-testid="stSidebar"] { background: #0B1F33; }
[data-testid="stSidebar"] * { color: #F8FAFC !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #A8C0D4 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { color: #0F172A !important; background: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #0F172A !important; }
.stButton > button { color: #FFFFFF; background: #00A6A6; border: 1px solid #087F8C; font-weight: 700; border-radius: 8px; min-height: 2.5rem; }
.stButton > button:hover { color: #FFFFFF; background: #087F8C; border-color: #00A6A6; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .18rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { background: #102A43; border: 1px solid #1D405B; border-radius: 9px; padding: .62rem .7rem; color: #CBD5E1 !important; font-weight: 600; margin: .22rem 0; transition: background .15s ease, border-color .15s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #173A56; color: #FFFFFF !important; border-color: #22D3EE; transform: translateX(2px); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: #00A6A6; color: #FFFFFF !important; border-color: #22D3EE; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color: #22D3EE; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: .83rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label small { color: #8FB4C8 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(1) { border-left: 3px solid #22D3EE; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(2) { border-left: 3px solid #2563EB; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(3) { border-left: 3px solid #00A6A6; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(4) { border-left: 3px solid #16A34A; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(5) { border-left: 3px solid #F59E0B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(6) { border-left: 3px solid #64748B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(7) { border-left: 3px solid #F59E0B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(8) { border-left: 3px solid #22D3EE; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: .8rem; }
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] { color: #0F172A !important; }
[data-testid="stDataFrame"] { border: 1px solid #CBD5E1; border-radius: 8px; }
[data-testid="stDataFrame"] * { color: #0F172A; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] { background: #FFFFFF; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color: #0F172A !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #64748B !important; }
.stAlert p, .stAlert span { color: #0F172A !important; }
.hero { background: #0B1F33; color: #F8FAFC; padding: 1.65rem 2rem; border-radius: 12px; margin-bottom: 1rem; border-left: 5px solid #00A6A6; }
.hero h1, .hero h1 span, .hero h1 strong { color: #F8FAFC !important; font-size: 2.1rem; margin: 0; }
.hero p, .hero p span { color: #C7D8E5 !important; margin: .35rem 0 0; font-size: 1rem; }
.kpi { background: #FFFFFF; border: 1px solid #D9E2EC; border-left: 4px solid #00A6A6; border-radius: 12px; padding: .85rem; min-height: 92px; box-shadow: 0 2px 8px rgba(15, 23, 42, .04); }
.kpi-label { color: #64748B; text-transform: uppercase; font-size: .7rem; font-weight: 700; letter-spacing: .08em; }
.kpi-value { color: #0F172A; font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }
.section { color: #0F172A; border-bottom: 2px solid #00A6A6; padding-bottom: .35rem; }
.audit-card { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: 1rem; margin-bottom: .75rem; }
.brand-mark { color: #F8FAFC; font-size: 1.25rem; font-weight: 800; letter-spacing: .08em; }
.brand-sub { color: #A8C0D4; font-size: .68rem; margin-left: .55rem; }
.status-online { color: #86EFAC; font-size: .78rem; font-weight: 700; padding-top: .45rem; }
.top-shell { background: #0B1F33; border-radius: 0 0 12px 12px; padding: .5rem .9rem .35rem; margin: -.7rem -1rem 1.1rem; }
.top-shell [data-testid="stRadio"] label { color: #CBD5E1 !important; }
.top-shell [data-testid="stRadio"] label:has(input:checked) { color: #FFFFFF !important; }
.top-shell [data-testid="stRadio"] label:has(input:checked) p { color: #FFFFFF !important; }
.top-search input { background: #FFFFFF !important; color: #0F172A !important; }
.workspace-card { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: 1rem; min-height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, .04); }
.workspace-title { color: #0F172A; font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .6rem; }
.queue-row { display: grid; grid-template-columns: 5.2rem minmax(0, 1fr) 5.8rem; align-items: center; gap: .6rem; padding: .62rem 0; border-bottom: 1px solid #E2E8F0; font-size: .78rem; }
.queue-row:last-child { border-bottom: 0; }
.queue-priority { font-weight: 800; }
.queue-critical { color: #EF4444; }
.queue-high { color: #F59E0B; }
.queue-medium { color: #2563EB; }
.queue-normal { color: #64748B; }
.queue-action { color: #087F8C; font-weight: 700; text-align: right; white-space: nowrap; font-size: .74rem; }
.sidebar-section { color: #8FB4C8; font-size: .68rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin: .9rem 0 .35rem; }
.sidebar-status { background: #102A43; border: 1px solid #1D405B; border-radius: 9px; padding: .75rem; color: #DCEAF2 !important; font-size: .76rem; line-height: 1.7; }
.sidebar-status strong { color: #86EFAC !important; }
.mode-status { border-radius: 8px; padding: .55rem .65rem; margin: .55rem 0 .75rem; font-size: .75rem; font-weight: 800; letter-spacing: .04em; }
.network-note { background: #EEF6F8; border: 1px solid #B7DCE2; border-left: 4px solid #2563EB; border-radius: 9px; color: #102A43 !important; padding: .7rem 1rem; margin: .55rem 0 1rem; font-size: .82rem; }
.network-note strong, .network-note span { color: #102A43 !important; }
.page-kicker { color: #087F8C !important; font-size: .7rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .2rem; }
.module-header { display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 1px solid #D9E2EC; border-left: 4px solid #00A6A6; border-radius: 10px; padding: .75rem 1rem; margin-bottom: 1rem; }
.module-header-title { color: #0F172A !important; font-weight: 800; font-size: 1.1rem; }
.module-header-meta { color: #64748B !important; font-size: .78rem; }
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #0F172A !important; }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: #0F172A !important; }
[data-testid="stChatInput"] textarea { color: #0F172A !important; background: #FFFFFF !important; }
.planner-state { background: #E8F1F5; border: 1px solid #8FBCC7; border-left: 4px solid #087F8C; border-radius: 10px; color: #0B1F33 !important; padding: .9rem 1rem; margin: .7rem 0; font-weight: 600; }
.planner-state strong, .planner-state span { color: #0B1F33 !important; }
.planner-note { background: #EEF6F8; border: 1px solid #B7DCE2; border-left: 4px solid #22D3EE; border-radius: 10px; color: #102A43 !important; padding: .9rem 1rem; margin: .7rem 0; line-height: 1.45; }
.planner-note strong, .planner-note span { color: #102A43 !important; }
.ai-question-card { background: #EEF6F8; border: 1px solid #B7DCE2; border-left: 4px solid #22D3EE; border-radius: 10px; padding: .8rem 1rem; margin: .8rem 0; color: #102A43 !important; }
.ai-question-label { color: #087F8C !important; font-size: .68rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.ai-question-text { color: #0F172A !important; font-size: 1rem; font-weight: 600; margin-top: .25rem; }
.ai-answer-card { background: linear-gradient(135deg, #0B1F33 0%, #102A43 100%); border: 1px solid #087F8C; border-top: 4px solid #22D3EE; border-radius: 12px; padding: 1.1rem 1.25rem; margin: .8rem 0; box-shadow: 0 5px 18px rgba(11, 31, 51, .14); }
.ai-answer-label { color: #67E8F9 !important; font-size: .7rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .55rem; }
.ai-answer-body, .ai-answer-body p, .ai-answer-body li, .ai-answer-body strong { color: #F8FAFC !important; line-height: 1.6; }
.ai-source-badge { display: inline-block; color: #0B1F33 !important; background: #86EFAC; border-radius: 999px; padding: .25rem .6rem; font-size: .7rem; font-weight: 800; margin-bottom: .6rem; }
.login-card { background: #0B1F33; color: #F8FAFC; border-left: 5px solid #00A6A6; border-radius: 12px; padding: 1.4rem 1.5rem 1.25rem; margin-top: 2rem; }
.login-card h1, .login-card h1 span { color: #F8FAFC !important; font-size: 1.55rem; line-height: 1.2; margin: .25rem 0 .55rem; }
.login-card p, .login-card p span { color: #C7D8E5 !important; margin: 0; }
</style>
""", unsafe_allow_html=True)


DEMO_USERS = {
    "admin": {"password": "admin123", "role": "ADMIN"},
    "manager": {"password": "manager123", "role": "MANAGER"},
}
ROLE_PAGES = {
    "ADMIN": ["Control Tower", "Data Input", "Shipment Priority", "Recovery Planner", "Global Recovery Plan", "Decision Audit", "Network Intelligence", "What-If Simulator", "AI Logistics Analyzer", "Data Monitoring"],
    "MANAGER": ["Control Tower", "Shipment Priority", "Recovery Planner", "Global Recovery Plan", "Decision Audit", "Network Intelligence", "What-If Simulator", "AI Logistics Analyzer"],
}

legacy_roles = {"Administrator": "ADMIN", "Logistics Manager": "MANAGER"}
if st.session_state.get("role") in legacy_roles:
    st.session_state.role = legacy_roles[st.session_state.role]


def show_login() -> None:
    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown('<div class="login-card"><h1>Intelligent Shipment Recovery Control Tower</h1><p>Intelligent Shipment Piggybacking &amp; Recovery Platform</p></div>', unsafe_allow_html=True)
        if st.session_state.get("auth_error"):
            st.error(st.session_state.pop("auth_error"))
        with st.form("demo_login"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="Enter demo password")
            remember = st.checkbox("Remember Me")
            submitted = st.form_submit_button("LOGIN TO CONTROL TOWER", width="stretch")
            if submitted:
                clean_username = username.strip().lower()
                try:
                    login_result = RecoveryApi().login(clean_username, password)
                    st.session_state.authenticated = True
                    st.session_state.user = login_result.get("username", clean_username)
                    st.session_state.role = login_result["role"]
                    st.session_state.access_token = login_result["access_token"]
                    st.session_state.remember = remember
                    st.rerun()
                except ApiError as error:
                    st.error(str(error))
        st.caption("Demo accounts: admin / admin123 · manager / manager123")


def restore_demo_session() -> None:
    st.session_state.authenticated = True
    st.session_state.user = "admin"
    st.session_state.role = "ADMIN"
    st.session_state.access_token = None


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def kpi(label: str, value: str):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def shipment_status_table(shipments_view: pd.DataFrame, plan: dict) -> pd.DataFrame:
    recovered = set(plan["allocations"].shipment_id) if not plan["allocations"].empty else set()
    escalated = set(plan["escalated"].shipment_id)
    statuses = shipments_view.shipment_id.map(lambda value: "Recovered" if value in recovered else "Escalated" if value in escalated else "Recoverable")
    result = shipments_view[["shipment_id", "origin", "destination", "current_location", "priority", "status", "deadline", "deadline_risk_label", "recommended_action"]].copy()
    result["Recovery status"] = statuses
    return result.rename(columns={"shipment_id": "Shipment ID", "origin": "Origin", "destination": "Destination", "current_location": "Current location", "priority": "Priority", "status": "Status", "deadline": "Deadline", "deadline_risk_label": "Delay risk", "recommended_action": "Recommended strategy"})


def data_health(loaded_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], validation: list[str]) -> pd.DataFrame:
    rows = []
    for name, frame in zip(("shipments.csv", "vehicles.csv", "hubs.csv", "routes.csv"), loaded_data):
        rows.append({"File": name, "Status": "Healthy" if not frame.empty else "Empty", "Records": len(frame), "Columns": len(frame.columns), "Last loaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Validation": "Passed" if not validation else "Review"})
    return pd.DataFrame(rows)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return serializable.astype(object).where(pd.notna(serializable), None).to_dict(orient="records")


def network_figure(routes: pd.DataFrame, vehicles: pd.DataFrame):
    """Build the operational network used by the Control Tower and Network page."""
    edge_counts = routes.merge(vehicles.groupby(["route_origin", "route_destination"], as_index=False).size().rename(columns={"size": "vehicles"}), left_on=["origin", "destination"], right_on=["route_origin", "route_destination"], how="left").fillna({"vehicles": 0})
    positions = {city: (i % 4, -(i // 4)) for i, city in enumerate(sorted(set(routes.origin) | set(routes.destination)))}
    figure = go.Figure()
    for row in edge_counts.itertuples():
        x0, y0 = positions[row.origin]
        x1, y1 = positions[row.destination]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"width": 1.5, "color": "#5B8EA3"}, hovertext=f"{row.origin} → {row.destination} · {int(row.vehicles)} vehicles", showlegend=False))
        figure.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#00A6A6")
    figure.add_trace(go.Scatter(x=[positions[c][0] for c in positions], y=[positions[c][1] for c in positions], mode="markers", hovertext=list(positions), hovertemplate="%{hovertext}<extra></extra>", marker={"size": 12, "color": "#00A6A6", "line": {"width": 1.5, "color": "#FFFFFF"}}, showlegend=False))
    for city, (x_position, y_position) in positions.items():
        figure.add_annotation(x=x_position, y=y_position, text=f"<b>{city}</b>", showarrow=False, yshift=24, font={"family": "Inter, sans-serif", "size": 12, "color": "#0F172A"}, bgcolor="#FFFFFF", bordercolor="#00A6A6", borderwidth=1, borderpad=4)
    figure.update_layout(height=390, xaxis={"visible": False, "range": [-.7, 3.7]}, yaxis={"visible": False, "range": [-3.2, .8], "scaleanchor": "x", "scaleratio": 1}, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF", margin={"l": 8, "r": 8, "t": 8, "b": 8}, hoverlabel={"bgcolor": "#0B1F33", "font": {"color": "#FFFFFF"}})
    return figure


def build_dispatch_views(shipments: pd.DataFrame, vehicles: pd.DataFrame, allocations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create lane, dispatch-sequence, and vehicle-selection views from calculated data."""
    ranked = shipments.copy()
    ranked["lane"] = ranked["current_location"] + " → " + ranked["destination"]
    ranked = ranked.sort_values(["lane", "priority_score", "remaining_hours"], ascending=[True, False, True]).reset_index(drop=True)
    direct_capacity = vehicles[(vehicles.vehicle_status != "Unavailable") & (vehicles.current_location == vehicles.route_origin)].copy()
    if "available_capacity_kg" not in direct_capacity.columns:
        direct_capacity["available_capacity_kg"] = (
            pd.to_numeric(direct_capacity["capacity_kg"], errors="coerce").fillna(0)
            - pd.to_numeric(direct_capacity["current_load_kg"], errors="coerce").fillna(0)
        ).clip(lower=0)
    lane_capacity = direct_capacity.groupby(["route_origin", "route_destination"]).available_capacity_kg.sum().to_dict()
    ranked["lane_capacity_kg"] = [lane_capacity.get((row.current_location, row.destination), 0) for row in ranked.itertuples()]
    ranked["cumulative_lane_weight_kg"] = ranked.groupby("lane")["weight_kg"].cumsum()
    ranked["dispatch_wave"] = np.where(ranked["lane_capacity_kg"] > 0, np.ceil(ranked["cumulative_lane_weight_kg"] / ranked["lane_capacity_kg"]), np.nan)
    ranked["dispatch_wave"] = pd.array(ranked["dispatch_wave"], dtype="Int64")
    ranked["dispatch_priority"] = ranked.groupby("lane").cumcount() + 1
    allocation_vehicle = allocations.set_index("shipment_id")["vehicle_id"].to_dict() if not allocations.empty else {}
    ranked["selected_vehicle"] = ranked.shipment_id.map(allocation_vehicle).fillna("No feasible vehicle")
    ranked["dispatch_status"] = ranked.shipment_id.isin(set(allocations.shipment_id) if not allocations.empty else set()).map({True: "RECOVERY READY", False: "ESCALATION / REVIEW"})
    lane_view = ranked.groupby("lane", as_index=False).agg({"shipment_id": "count", "weight_kg": "sum", "priority_score": "max", "remaining_hours": "min", "dispatch_wave": "max"}).rename(columns={"shipment_id": "shipments", "weight_kg": "total_weight_kg", "priority_score": "highest_priority_score", "remaining_hours": "nearest_deadline_hours", "dispatch_wave": "dispatch_waves"})
    lane_view = lane_view.sort_values(["highest_priority_score", "nearest_deadline_hours"], ascending=[False, True])
    return lane_view, ranked, direct_capacity


@st.cache_data(show_spinner=False)
def load_data(demo: bool, access_token: str | None = None, data_revision: int = 0):
    if access_token:
        try:
            payload = RecoveryApi(token=access_token).get_data()
            frames = frames_from_payload(payload)
            return tuple(frames[name] for name in ("shipments", "vehicles", "hubs", "routes"))
        except ApiError as error:
            if "Session expired" in str(error):
                raise
    return load_csv_data(demo=demo)


@st.cache_data(show_spinner=False)
def build_plan(demo: bool, weight_values: tuple[tuple[str, float], ...], access_token: str | None = None, data_revision: int = 0):
    loaded_data = load_data(demo, access_token, data_revision)
    configured_weights = dict(weight_values)
    if access_token:
        try:
            api = RecoveryApi(token=access_token)
            datasets = {name: _json_records(frame) for name, frame in zip(("shipments", "vehicles", "hubs", "routes"), loaded_data)}
            response = api.plan(datasets, configured_weights)
            frame_keys = ("shipments", "allocations", "candidate_options", "escalated", "rejected", "vehicles")
            api_plan = {key: pd.DataFrame(response[key]) if key in frame_keys and isinstance(response.get(key), list) else response.get(key) for key in response}
            for key in frame_keys:
                if key not in api_plan:
                    api_plan[key] = pd.DataFrame()
            if not isinstance(api_plan.get("validation"), list):
                api_plan["validation"] = []
            return loaded_data, api_plan
        except ApiError as error:
            if "Session expired" in str(error):
                raise
    return loaded_data, allocate_plan(*loaded_data, configured_weights)


with st.sidebar:
    st.markdown('<div class="brand-mark">◈ RELAYX</div><div class="brand-sub">AI SHIPMENT RECOVERY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-online">● SYSTEM ONLINE</div><div class="sidebar-status">Signed in as <strong>{st.session_state.user}</strong><br>{st.session_state.role}</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-section">Operations workspace</div>', unsafe_allow_html=True)
    icons = {"Control Tower": "▦", "Data Input": "⇧", "Shipment Priority": "◉", "Recovery Planner": "⇄", "Global Recovery Plan": "▤", "Decision Audit": "✓", "Network Intelligence": "⌁", "What-If Simulator": "◌", "AI Logistics Analyzer": "✦", "Data Monitoring": "⚙"}
    permitted_pages = ROLE_PAGES[st.session_state.role]
    navigation_labels = [f"{icons[item]}  {item}" for item in permitted_pages]
    navigation_target = {f"{icons[item]}  {item}": item for item in permitted_pages}
    page = navigation_target[st.radio("Operations workspace", navigation_labels, label_visibility="collapsed")]
    st.divider()
    st.markdown('<div class="sidebar-section">Workspace controls</div>', unsafe_allow_html=True)
    st.caption("Live synthetic logistics data · deterministic optimization source of truth")
    with st.expander("Optimization weights"):
        weights = {key: st.slider(key.title(), 0.0, 1.0, float(value), 0.05) for key, value in DEFAULT_WEIGHTS.items()}
    st.markdown('<div class="sidebar-section">System status</div><div class="sidebar-status"><strong>● Optimization engine online</strong><br>● Live simulation data<br>● Analyzer ready</div>', unsafe_allow_html=True)
    if st.button("LOG OUT", width="stretch"):
        for key in ("authenticated", "user", "role", "remember", "ai_question", "ai_answer", "ai_source", "access_token", "backend_dataset", "data_revision"):
            st.session_state.pop(key, None)
        st.session_state.authenticated = False
        st.rerun()

try:
    (shipments, vehicles, hubs, routes), plan = build_plan(False, tuple(sorted(weights.items())), st.session_state.get("access_token"), int(st.session_state.get("data_revision", 0)))
except ApiError as error:
    handle_api_error(error)
    st.stop()
except Exception as error:
    st.error(f"Recovery analysis could not be completed: {error}")
    st.info("Please verify the shipment, vehicle, route, and network data, then restart the workspace.")
    st.stop()

st.markdown('<div class="hero"><h1>RELAYX · AI SHIPMENT RECOVERY CONTROL TOWER</h1><p>Recover more. Use existing capacity. Move smarter.</p></div>', unsafe_allow_html=True)
if plan["validation"]:
    st.markdown(f'<div class="network-note"><strong>Network routing note:</strong> {" · ".join(plan["validation"])}</div>', unsafe_allow_html=True)

shipments_view = plan["shipments"]
allocations = plan["allocations"]
recovered_count = len(allocations)
critical = int((shipments_view.priority == "Critical").sum())
at_risk = int((shipments_view.deadline_risk_label.isin(["HIGH", "CRITICAL"])).sum())
available_vehicles = int((vehicles.vehicle_status != "Unavailable").sum())
utilization = float(((vehicles.current_load_kg.sum() + allocations.weight_kg.sum()) / vehicles.capacity_kg.sum()) * 100) if not allocations.empty else float((vehicles.current_load_kg.sum() / vehicles.capacity_kg.sum()) * 100)

module_descriptions = {
    "Data Input": ("DATA INPUT", "Upload CSV datasets to update the operational network and recovery plan.", "#22D3EE"),
    "Shipment Priority": ("SHIPMENT PRIORITY", "Rank misplaced shipments by urgency, risk, value and deadline.", "#2563EB"),
    "Recovery Planner": ("RECOVERY PLANNER", "Compare feasible direct, one-hop, and multi-hub recovery opportunities.", "#00A6A6"),
    "Global Recovery Plan": ("GLOBAL RECOVERY PLAN", "See the fleet-wide allocation across shared capacity.", "#16A34A"),
    "Decision Audit": ("DECISION AUDIT", "Inspect what was selected and why alternatives were rejected.", "#F59E0B"),
    "Network Intelligence": ("NETWORK INTELLIGENCE", "Explore active corridors, hubs, vehicles and route risk.", "#22D3EE"),
    "What-If Simulator": ("WHAT-IF SIMULATOR", "Stress-test the recovery network against disruption.", "#F59E0B"),
    "AI Logistics Analyzer": ("AI LOGISTICS ANALYZER", "Ask the agent about the calculated recovery network.", "#22D3EE"),
    "Data Monitoring": ("DATA MONITORING", "Inspect loaded records and validation status.", "#F59E0B"),
}
if page in module_descriptions:
    module_title, module_description, module_color = module_descriptions[page]
    st.markdown(f'<div class="module-header" style="border-left-color:{module_color}"><span class="module-header-title">{module_title}</span><span class="module-header-meta">{module_description}</span></div>', unsafe_allow_html=True)

if page == "Control Tower":
    st.markdown("### Monitor misplaced shipments, recovery opportunities, fleet capacity, and delivery risk across the network.")
    cols = st.columns(8)
    for col, label, value in zip(cols, ["Misplaced shipments", "Recovered shipments", "Escalations", "At-risk shipments", "Critical shipments", "Vehicles available", "Fleet utilization", "Estimated cost impact"], [len(shipments_view), recovered_count, len(plan["escalated"]), at_risk, critical, available_vehicles, f"{utilization:.1f}%", money(plan["savings"])]):
        with col: kpi(label, str(value))
    st.write("")
    network_column, queue_column = st.columns([7, 5], gap="medium")
    with network_column:
        st.markdown('<div class="workspace-card"><div class="workspace-title">Network recovery overview · active routes and recovery paths</div>', unsafe_allow_html=True)
        st.plotly_chart(network_figure(routes, vehicles), width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with queue_column:
        st.markdown('<div class="workspace-card"><div class="workspace-title">Priority queue · earliest operational attention</div>', unsafe_allow_html=True)
        for row in shipments_view.head(7).itertuples():
            priority_class = str(row.priority).lower()
            status = "ESCALATION" if row.shipment_id in set(plan["escalated"].shipment_id) else "RECOVERY READY"
            action = "Recover now" if row.recommended_action == "Recover immediately" else "Review"
            st.markdown(f'<div class="queue-row"><span class="queue-priority queue-{priority_class}">{row.priority.upper()}</span><span><b>{row.shipment_id}</b> · {row.current_location} → {row.destination}<br><small>{row.remaining_hours:.1f}h remaining · {status}</small></span><span class="queue-action">{action} →</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(status_chart(plan), width="stretch")
    with right:
        st.plotly_chart(priority_chart(shipments_view), width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(risk_chart(shipments_view), width="stretch")
    with right:
        st.plotly_chart(fleet_chart(vehicles), width="stretch")
    impact_left, impact_right = st.columns(2)
    with impact_left:
        st.markdown("#### BUSINESS IMPACT")
        st.dataframe(
            pd.DataFrame({
                "Metric": ["Capacity reused", "Dedicated trips avoided", "On-time recovery", "Estimated recovery cost"],
                "Value": [
                    f"{allocations.weight_kg.sum() if not allocations.empty else 0:,.0f} kg",
                    str(recovered_count),
                    f"{(allocations.deadline_margin.ge(0).mean() * 100 if not allocations.empty else 0):.1f}%",
                    money(plan["total_cost"]),
                ],
            }),
            hide_index=True,
            width="stretch",
        )
    with impact_right:
        st.markdown("#### RECOVERY IMPACT")
        st.dataframe(
            pd.DataFrame({
                "Metric": ["Shipments considered", "Feasible / recovered", "Shipments escalated", "Vehicles used", "One-hop transfers"],
                "Value": [
                    str(len(shipments_view)),
                    f"{recovered_count} / {len(shipments_view)}",
                    str(len(plan["escalated"])),
                    str(allocations.vehicle_id.nunique() if not allocations.empty else 0),
                    str(int(allocations.strategy.eq("ONE-HUB PIGGYBACK").sum()) if not allocations.empty else 0),
                ],
            }),
            hide_index=True,
            width="stretch",
        )
    st.markdown("#### Shipment status monitor")
    status_view = shipment_status_table(shipments_view, plan)
    status_search = st.text_input("Search shipment status", placeholder="Shipment, location, priority, or status")
    if status_search:
        status_view = status_view[status_view.astype(str).apply(lambda column: column.str.contains(status_search, case=False, na=False)).any(axis=1)]
    st.dataframe(status_view, width="stretch", hide_index=True)

elif page == "Shipment Priority":
    st.markdown("### Shipment Priority Center")
    st.caption("Same-lane shipments are ranked first by recovery priority, then grouped into dispatch waves, then matched to the strongest available transfer vehicle.")
    lane_view, dispatch_view, direct_capacity = build_dispatch_views(shipments_view, vehicles, allocations)

    st.markdown("#### 1 · Lane priority")
    st.caption("When multiple shipments share an origin and destination, the lane with the highest urgency and nearest deadline receives attention first.")
    lane_table = lane_view.rename(columns={
        "lane": "Origin → destination lane",
        "shipments": "Shipments",
        "total_weight_kg": "Total weight (kg)",
        "highest_priority_score": "Highest priority score",
        "nearest_deadline_hours": "Nearest deadline (hrs)",
        "dispatch_waves": "Dispatch waves",
    })
    st.dataframe(
        lane_table,
        width="stretch",
        hide_index=True,
        column_config={"Highest priority score": st.column_config.ProgressColumn("Highest priority score", min_value=0, max_value=1)},
    )

    lane_options = lane_view.lane.tolist()
    selected_lane = st.selectbox("Select a lane to plan dispatch priority", lane_options)
    selected_dispatch = dispatch_view[dispatch_view.lane == selected_lane].copy()
    selected_dispatch["Dispatch priority"] = selected_dispatch["dispatch_priority"]
    selected_dispatch = selected_dispatch.sort_values("Dispatch priority")

    st.markdown("#### 2 · Shipment dispatch sequence")
    st.caption("Within the selected lane, the highest-priority shipments are assigned the earliest dispatch sequence. Dispatch waves show how many capacity batches are required.")
    dispatch_table = selected_dispatch[[
        "dispatch_priority",
        "dispatch_wave",
        "shipment_id",
        "priority",
        "weight_kg",
        "remaining_hours",
        "deadline_risk_label",
        "priority_score",
        "selected_vehicle",
        "dispatch_status",
    ]].rename(columns={
        "dispatch_priority": "Dispatch priority",
        "dispatch_wave": "Dispatch wave",
        "shipment_id": "Shipment ID",
        "priority": "Recovery priority",
        "weight_kg": "Weight (kg)",
        "remaining_hours": "Time to deadline (hrs)",
        "deadline_risk_label": "Recovery risk",
        "priority_score": "Priority score",
        "selected_vehicle": "Selected vehicle",
        "dispatch_status": "Recovery status",
    })
    st.dataframe(
        dispatch_table,
        width="stretch",
        hide_index=True,
        column_config={"Priority score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=1)},
    )

    st.markdown("#### 3 · Candidate vehicle selection")
    st.caption("Vehicles are ranked for the selected lane by route compatibility, usable capacity, availability, ETA, and estimated transport cost.")
    origin, destination = selected_lane.split(" → ")
    vehicle_table = direct_capacity[(direct_capacity.route_origin == origin) & (direct_capacity.route_destination == destination)].copy()
    if vehicle_table.empty:
        st.info("No vehicles currently satisfy this lane's direct recovery constraints. Consider an intermediate-hub recovery path or escalation.")
    else:
        vehicle_table["vehicle_selection_priority"] = vehicle_table.apply(
            lambda row: (row.available_capacity_kg > 0, row.vehicle_status == "Available", -row.transport_cost),
            axis=1,
        )
        vehicle_table = vehicle_table.sort_values("vehicle_selection_priority", ascending=False).reset_index(drop=True)
        vehicle_table.insert(0, "Vehicle priority", range(1, len(vehicle_table) + 1))
        vehicle_table["route_compatibility"] = "DIRECT ROUTE"
        vehicle_table["selection_reason"] = vehicle_table.apply(
            lambda row: f"{row.available_capacity_kg:.0f}kg available · {row.vehicle_status.lower()} · estimated cost {money(row.transport_cost)}",
            axis=1,
        )
        vehicle_table = vehicle_table.rename(columns={
            "vehicle_id": "Candidate vehicle",
            "vehicle_status": "Vehicle availability",
            "available_capacity_kg": "Capacity available (kg)",
            "eta": "Estimated arrival",
            "route_destination": "Destination",
            "selection_reason": "Selection rationale",
        })
        st.dataframe(
            vehicle_table[[
                "Vehicle priority",
                "Candidate vehicle",
                "route_compatibility",
                "Vehicle availability",
                "Capacity available (kg)",
                "transport_cost",
                "Estimated arrival",
                "Selection rationale",
            ]],
            width="stretch",
            hide_index=True,
            column_config={"transport_cost": st.column_config.NumberColumn("Estimated transport cost", format="₹%,.0f")},
        )

elif page == "Recovery Planner":
    selected_id = st.selectbox("Select shipment for recovery", shipments_view.shipment_id.tolist())
    shipment = shipments_view[shipments_view.shipment_id == selected_id].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Shipment", selected_id); b.metric("Priority", shipment.priority); c.metric("Weight", f"{shipment.weight_kg:.0f} kg"); d.metric("Deadline risk", shipment.deadline_risk_label)
    st.write(f"Current route: **{shipment.current_location} → {shipment.destination}** · Time to deadline: **{shipment.remaining_hours:.1f} hrs** · Delay probability: **{shipment.delay_probability:.0%}**")
    options_df = evaluate_candidates(shipment, vehicles, hubs, routes)
    feasible_options = options_df[options_df.feasible].copy() if not options_df.empty else options_df
    if not feasible_options.empty:
        candidate_view = feasible_options[["vehicle_id", "strategy", "route", "hub", "capacity_available", "required_capacity", "cost", "eta_hours", "deadline_margin", "utilization", "reason"]].rename(columns={"vehicle_id": "Candidate vehicle", "strategy": "Recovery strategy", "route": "Route", "hub": "Transfer hub", "capacity_available": "Capacity available (kg)", "required_capacity": "Required capacity (kg)", "cost": "Recovery cost", "eta_hours": "Estimated arrival (hrs)", "deadline_margin": "Deadline margin (hrs)", "utilization": "Utilization", "reason": "Feasibility"})
        st.dataframe(candidate_view, width="stretch", hide_index=True)
        best = feasible_options.iloc[0]
        st.success(f"Selected recovery opportunity: {best.vehicle_id} · {best.strategy} · {money(best.cost)} estimated · {best.eta_hours:.1f} hrs to arrival")
        st.markdown(f"**Why this opportunity was selected:** {best.reason}")
    else:
        st.markdown('<div class="planner-state">No feasible piggyback option found for this shipment.</div>', unsafe_allow_html=True)
        if not options_df.empty:
            st.markdown(f'<div class="planner-note"><strong>Why this happened:</strong> every candidate fails a hard recovery constraint. This shipment has {shipment.remaining_hours:.1f} hours to deadline; recovery requires sufficient capacity and an arrival before the deadline.</div>', unsafe_allow_html=True)
            st.dataframe(options_df[["vehicle_id", "strategy", "route", "hub", "capacity_available", "required_capacity", "cost", "eta_hours", "deadline_margin", "reason"]], width="stretch", hide_index=True)
        else:
            st.markdown('<div class="planner-note"><strong>Why this happened:</strong> no compatible vehicle or one-hub route was found from the shipment\'s current location.</div>', unsafe_allow_html=True)
        st.info("Available alternatives: dedicated vehicle, next available route, alternate hub, or manual logistics intervention.")

elif page == "Global Recovery Plan":
    st.markdown("### Fleet-Wide Allocation")
    st.caption("Recovery decisions across all misplaced shipments, using shared vehicle capacity and route constraints.")
    a, b, c, d = st.columns(4)
    a.metric("Shipments recovered", recovered_count); b.metric("Shipments escalated", len(plan["escalated"])); c.metric("Recovery cost", money(plan["total_cost"])); d.metric("Estimated cost impact", money(plan["savings"]))
    if not allocations.empty:
        allocation_view = allocations[["shipment_id", "vehicle_id", "strategy", "route", "hub", "weight_kg", "eta_hours", "deadline_margin", "cost", "priority"]].copy()
        allocation_view["hub"] = allocation_view["hub"].fillna("-")
        allocation_view.columns = ["Shipment", "Vehicle(s)", "Strategy", "Route", "Transfer Hub(s)", "Weight kg", "ETA hrs", "Margin hrs", "Est. cost", "Priority"]
        st.dataframe(allocation_view, width="stretch", hide_index=True, column_config={"Est. cost": st.column_config.NumberColumn(format="₹%,.0f"), "ETA hrs": st.column_config.NumberColumn(format="%.1f"), "Margin hrs": st.column_config.NumberColumn(format="%.1f")})
        with st.expander("Inspect calculated allocation details"):
            st.dataframe(allocations[["shipment_id", "vehicle_id", "strategy", "route", "cost", "eta_hours", "deadline_margin", "score", "reason"]], width="stretch", hide_index=True)
    if not plan["escalated"].empty:
        st.markdown("#### Escalated shipments")
        st.dataframe(plan["escalated"][["shipment_id", "destination", "weight_kg", "priority", "reason"]].rename(columns={"shipment_id": "Shipment ID", "destination": "Destination", "weight_kg": "Weight (kg)", "priority": "Recovery priority", "reason": "Escalation reason"}), width="stretch", hide_index=True)
    if not plan["rejected"].empty:
        with st.expander("Inspect alternative candidate evaluations"):
            st.dataframe(plan["rejected"], width="stretch", hide_index=True)

elif page == "Decision Audit":
    st.markdown("### Decision Audit")
    st.caption("Understand what the recovery engine decided, why it was selected, and why alternatives were not used.")
    persisted_audit = []
    if st.session_state.get("access_token"):
        try:
            persisted_audit = RecoveryApi(token=st.session_state.access_token).get_audit()
        except ApiError as error:
            st.warning(str(error))
    if persisted_audit:
        st.dataframe(pd.DataFrame(persisted_audit), width="stretch", hide_index=True)
    elif allocations.empty:
        st.info("No recovery decisions have been recorded yet.")
    else:
        audit_rows = []
        for row in allocations.itertuples():
            audit_rows.append({"Shipment": row.shipment_id, "Selected vehicle(s)": row.vehicle_id, "Strategy": row.strategy, "Cost": money(row.cost), "ETA": f"{row.eta_hours:.1f} hrs", "Deadline margin": f"{row.deadline_margin:.1f} hrs", "Priority score": f"{row.priority_score:.2f}", "Status": "RECOVERED"})
        st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)

        selected_audit_id = st.selectbox(
            "Select shipment to inspect audit details",
            allocations.shipment_id.tolist()
        )

        row = allocations[allocations.shipment_id == selected_audit_id].iloc[0]

        st.markdown(
            f"#### Audit Details: {row.shipment_id} · {row.vehicle_id} · {row.strategy}"
        )

        st.markdown('<div class="audit-card">', unsafe_allow_html=True)

        st.markdown(f"**WHAT WAS DECIDED?** {row.reason}")

        st.markdown("**WHY WAS IT SELECTED?**")

        factors = pd.DataFrame({
            "Decision factor": [
                "Route compatibility",
                "Deadline compliance",
                "Capacity availability",
                "Cost efficiency",
                "Priority satisfaction"
            ],
            "Assessment": [
                "100%",
                f"{max(0, min(row.deadline_margin / max(row.eta_hours, 1), 1) * 100):.0f}%",
                "FEASIBLE",
                f"{max(0, (1 - row.cost / 8000) * 100):.0f}%",
                f"{row.priority_score * 100:.0f}%"
            ]
        })

        st.dataframe(factors, hide_index=True, width="stretch")

        rejected = plan["rejected"]

        if not rejected.empty:
            rejected_for_shipment = rejected[
                rejected.shipment_id == row.shipment_id
            ]

            if not rejected_for_shipment.empty:
                st.markdown("**WHY WERE ALTERNATIVES NOT SELECTED?**")

                st.dataframe(
                    rejected_for_shipment,
                    hide_index=True,
                    width="stretch"
                )

        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Network Intelligence":
    st.markdown("### Recovery Network")
    st.caption("Explore routes, transfer hubs, vehicle movement, shipment flows, and recovery opportunities across the transportation network.")
    edge_counts = routes.merge(vehicles.groupby(["route_origin", "route_destination"], as_index=False).size().rename(columns={"size": "vehicles"}), left_on=["origin", "destination"], right_on=["route_origin", "route_destination"], how="left").fillna({"vehicles": 0})
    fig = go.Figure()
    positions = {city: (i % 4, -(i // 4)) for i, city in enumerate(sorted(set(routes.origin) | set(routes.destination)))}
    for row in edge_counts.itertuples():
        x0, y0 = positions[row.origin]; x1, y1 = positions[row.destination]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"width": 1.5, "color": "#8aa4aa"}, hovertext=f"{row.origin} → {row.destination} · {int(row.vehicles)} vehicles", showlegend=False))
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#14566a")
    fig.add_trace(go.Scatter(x=[positions[c][0] for c in positions], y=[positions[c][1] for c in positions], mode="markers", hovertext=list(positions), hovertemplate="%{hovertext}<extra></extra>", marker={"size": 12, "color": "#d47a3f", "line": {"width": 1.5, "color": "#ffffff"}}, showlegend=False))
    for city, (x_position, y_position) in positions.items():
        fig.add_annotation(x=x_position, y=y_position, text=f"<b>{city}</b>", showarrow=False, yshift=25, font={"family": "DM Sans, sans-serif", "size": 13, "color": "#17313d"}, bgcolor="#ffffff", bordercolor="#d47a3f", borderwidth=1, borderpad=4, opacity=0.98)
    fig.update_layout(height=600, xaxis={"visible": False, "range": [-0.7, 3.7]}, yaxis={"visible": False, "range": [-3.2, 0.9]}, plot_bgcolor="#e9f0f2", paper_bgcolor="#f3f6f8", margin={"l": 20, "r": 20, "t": 30, "b": 20}, hoverlabel={"bgcolor": "#092b3d", "font": {"color": "#ffffff"}})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(edge_counts[["route_id", "origin", "destination", "distance_km", "estimated_hours", "delay_risk", "vehicles"]], width="stretch", hide_index=True)

elif page == "What-If Simulator":
    st.markdown("### Scenario Simulator")
    st.caption("Test network disruptions and see how the recovery plan changes.")
    vehicle_change = st.selectbox("Vehicle unavailable", ["None"] + vehicles.vehicle_id.tolist())
    capacity_loss = st.slider("Reduce vehicle capacity (%)", 0, 70, 0, 5)
    shorter_deadlines = st.slider("Shorten shipment deadlines (hrs)", 0, 12, 0)
    cost_change = st.slider("Increase transport cost (%)", 0, 50, 0, 5)
    if st.button("Recalculate recovery plan", type="primary"):
        simulation_args = {"shipments": _json_records(shipments), "vehicles": _json_records(vehicles), "hubs": _json_records(hubs), "routes": _json_records(routes), "weights": weights, "capacity_factor": 1 - capacity_loss / 100, "unavailable_vehicle_ids": [] if vehicle_change == "None" else [vehicle_change], "deadline_shift_hours": shorter_deadlines, "route_cost_factor": 1 + cost_change / 100}
        try:
            if st.session_state.get("access_token"):
                simulated = {key: pd.DataFrame(value) if isinstance(value, list) else value for key, value in RecoveryApi(token=st.session_state.access_token).simulation(simulation_args).items()}
            else:
                simulated = run_simulation(shipments, vehicles, hubs, routes, weights, capacity_factor=1 - capacity_loss / 100, unavailable_vehicle_ids=[] if vehicle_change == "None" else [vehicle_change], deadline_shift_hours=shorter_deadlines, route_cost_factor=1 + cost_change / 100)
        except ApiError as error:
            st.warning(str(error))
            simulated = run_simulation(shipments, vehicles, hubs, routes, weights, capacity_factor=1 - capacity_loss / 100, unavailable_vehicle_ids=[] if vehicle_change == "None" else [vehicle_change], deadline_shift_hours=shorter_deadlines, route_cost_factor=1 + cost_change / 100)
        before_recovery = len(plan["allocations"]); after_recovery = len(simulated["allocations"])
        st.markdown("#### Scenario impact")
        x, y, z = st.columns(3); x.metric("Recovered shipments", after_recovery, after_recovery - before_recovery); y.metric("Escalations", len(simulated["escalated"]), len(simulated["escalated"]) - len(plan["escalated"])); z.metric("Estimated recovery cost", money(simulated["total_cost"]), money(simulated["total_cost"] - plan["total_cost"]))
        st.dataframe(simulated["allocations"][["vehicle_id", "shipment_id", "strategy", "cost", "deadline_margin"]], width="stretch", hide_index=True)
        st.info("The recovery plan was recalculated globally. Changes can cascade because vehicle capacity is shared across shipments.")

elif page == "Data Monitoring":
    st.markdown("### System and data monitoring")
    st.caption("Administrator view of the CSV sources and deterministic validation contract.")
    if st.button("Refresh data"):
        bump_data_revision()
        load_data.clear()
        build_plan.clear()
        st.rerun()
    st.dataframe(data_health((shipments, vehicles, hubs, routes), plan["validation"]), width="stretch", hide_index=True)
    if plan["validation"]:
        st.warning(" · ".join(plan["validation"]))
    else:
        st.success("All loaded datasets passed the backend validation checks.")
    st.markdown("#### Loaded schema")
    st.dataframe(pd.DataFrame({"Dataset": ["shipments", "vehicles", "hubs", "routes"], "Columns": [", ".join(frame.columns) for frame in (shipments, vehicles, hubs, routes)]}), width="stretch", hide_index=True)
elif page == "Data Input":
    render_data_input_page((shipments, vehicles, hubs, routes))
else:
    st.markdown("### Ask the Recovery Analyst")
    st.write("Ask about recovery decisions, shipment risk, fleet capacity, route constraints, or scenario impact.")
    st.caption("The AI is an explanation layer. Operational decisions remain governed by the deterministic recovery engine.")
    question = st.chat_input("Ask about the current recovery plan...")
    if question:
        st.session_state.ai_question = question
        try:
            if st.session_state.get("access_token"):
                api_payload = {key: _json_records(value) if isinstance(value, pd.DataFrame) else value for key, value in plan.items()}
                ai_result = RecoveryApi(token=st.session_state.access_token).ai(question, api_payload)
                st.session_state.ai_answer, st.session_state.ai_source = ai_result["answer"], ai_result["source"]
            else:
                st.session_state.ai_answer, st.session_state.ai_source = answer_question(question, plan, vehicles, hubs, routes)
        except ApiError:
            st.session_state.ai_answer, st.session_state.ai_source = answer_question(question, plan, vehicles, hubs, routes)
    if st.session_state.get("ai_question"):
        st.markdown(f'<div class="ai-question-card"><div class="ai-question-label">Your latest question</div><div class="ai-question-text">{st.session_state.ai_question}</div></div>', unsafe_allow_html=True)
        answer_html = html.escape(str(st.session_state.ai_answer)).replace("\n", "<br>")
        st.markdown(f'<div class="ai-answer-card"><div class="ai-answer-label">Recovery analyst response</div><div class="ai-source-badge">{html.escape(str(st.session_state.ai_source))}</div><div class="ai-answer-body">{answer_html}</div></div>', unsafe_allow_html=True)
        st.caption("AI-generated explanation based on current recovery data. Costs, ETAs, and risk scores are synthetic estimates.")
        if st.button("Clear previous answer"):
            st.session_state.pop("ai_question", None)
            st.session_state.pop("ai_answer", None)
            st.session_state.pop("ai_source", None)
            st.rerun()
