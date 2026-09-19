from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an intelligent logistics recovery agent. Answer only the user's latest question using the retrieved application facts. Never invent IDs, routes, costs, ETAs, capacities, or decisions. State when data is unavailable. Explain trade-offs clearly and respect hard capacity, route, and deadline constraints."""


def configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def complete(question: str, context: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {"model": model, "temperature": 0.1, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps({"question": question, "retrieved_facts": context}, ensure_ascii=False, default=str)}]}
    request = Request(f"{base_url}/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode())
        return body["choices"][0]["message"]["content"].strip()
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"LLM request failed: {error}") from error
