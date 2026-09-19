from __future__ import annotations

import os
from typing import Any

import requests


class ApiError(RuntimeError):
    pass


def _api_base_url() -> str:
    return (
        os.getenv("API_BASE_URL")
        or os.getenv("BACKEND_API_URL")
        or "http://127.0.0.1:8001"
    ).rstrip("/")


def _format_detail(detail: Any) -> str:
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                loc = item.get("loc", [])
                msg = item.get("msg", item)
                loc_text = " → ".join(str(part) for part in loc) if loc else ""
                parts.append(f"{loc_text}: {msg}" if loc_text else str(msg))
            else:
                parts.append(str(item))
        return " ".join(parts) if parts else str(detail)
    return str(detail)


class RecoveryApi:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or _api_base_url()).rstrip("/")
        self.token = token

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = requests.request(method, f"{self.base_url}{path}", headers=headers, timeout=30, **kwargs)
        except requests.RequestException as error:
            raise ApiError("Unable to connect to FastAPI backend. Make sure the backend is running on port 8001.") from error
        if response.status_code == 401:
            if path == "/auth/login":
                raise ApiError("Invalid demo credentials. Use admin / admin123 or manager / manager123.")
            raise ApiError("Session expired. Please login again.")
        if response.status_code == 403:
            raise ApiError("You do not have permission to perform this action.")
        if response.status_code >= 400:
            try:
                payload = response.json()
                detail = payload.get("detail", payload)
            except ValueError:
                detail = response.text
            if response.status_code == 422:
                raise ApiError(_format_detail(detail))
            raise ApiError(_format_detail(detail))
        return response.json()

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/auth/login", data={"username": username, "password": password})

    def get_current_user(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def get_data(self) -> dict[str, Any]:
        return self._request("GET", "/data")

    def get_audit(self) -> list[dict[str, Any]]:
        return self._request("GET", "/audit")  # type: ignore[return-value]

    def submit_data(self, payload: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return self._request("POST", "/data/submit", json=payload)

    def upload(self, name: str, uploaded_file: Any) -> dict[str, Any]:
        filename = getattr(uploaded_file, "name", f"{name}.csv")
        contents = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        return self._request("POST", f"/data/{name}/upload", files={"file": (filename, contents, "text/csv")})

    def upload_csv(self, name: str, uploaded_file: Any) -> dict[str, Any]:
        return self.upload(name, uploaded_file)

    def create_record(self, name: str, record: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/data/{name}", json=record)

    def plan(self, datasets: dict[str, list[dict[str, Any]]], weights: dict[str, float] | None = None) -> dict[str, Any]:
        return self._request("POST", "/recovery/plan", json={**datasets, "weights": weights})

    def recovery_plan(self, datasets: dict[str, list[dict[str, Any]]], weights: dict[str, float] | None = None) -> dict[str, Any]:
        return self.plan(datasets, weights)

    def analyze_shipments(self, datasets: dict[str, list[dict[str, Any]]], weights: dict[str, float] | None = None) -> dict[str, Any]:
        return self._request("POST", "/shipments/analyze", json={**datasets, "weights": weights})

    def simulation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/simulation/run", json=payload)

    def run_simulation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.simulation(payload)

    def ai(self, question: str, plan: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/ai/analyze", json={"question": question, "plan": plan})

    def ai_analyze(self, question: str, plan: dict[str, Any]) -> dict[str, Any]:
        return self.ai(question, plan)


def login(username: str, password: str) -> dict[str, Any]:
    return RecoveryApi().login(username, password)


def get_current_user(token: str) -> dict[str, Any]:
    return RecoveryApi(token=token).get_current_user()


def get_data(token: str) -> dict[str, Any]:
    return RecoveryApi(token=token).get_data()


def submit_data(token: str, payload: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return RecoveryApi(token=token).submit_data(payload)


def upload_csv(token: str, name: str, uploaded_file: Any) -> dict[str, Any]:
    return RecoveryApi(token=token).upload_csv(name, uploaded_file)


def analyze_shipments(token: str, datasets: dict[str, list[dict[str, Any]]], weights: dict[str, float] | None = None) -> dict[str, Any]:
    return RecoveryApi(token=token).analyze_shipments(datasets, weights)


def recovery_plan(token: str, datasets: dict[str, list[dict[str, Any]]], weights: dict[str, float] | None = None) -> dict[str, Any]:
    return RecoveryApi(token=token).recovery_plan(datasets, weights)


def run_simulation(token: str, payload: dict[str, Any]) -> dict[str, Any]:
    return RecoveryApi(token=token).run_simulation(payload)


def ai_analyze(token: str, question: str, plan: dict[str, Any]) -> dict[str, Any]:
    return RecoveryApi(token=token).ai_analyze(question, plan)
