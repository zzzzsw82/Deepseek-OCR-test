from __future__ import annotations

import asyncio
import os
from typing import Optional

import requests

from agents import get_invoice_runtime


def test_ollama_connection(base_url: Optional[str] = None) -> None:
    """Simple health-check to ensure the local Ollama server is reachable."""
    url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    response = requests.get(f"{url}/api/tags", timeout=5)
    response.raise_for_status()
    models = [entry.get("name") for entry in response.json().get("models", [])]
    print(f"Ollama reachable at {url}. Available models: {models}")


async def run_agent() -> None:
    runtime = get_invoice_runtime()
    result = await runtime.invoke("请对 input/ 中的票据执行 OCR 并整理成 CSV")
    print(runtime.latest_content(result))


if __name__ == "__main__":
    test_ollama_connection()
    asyncio.run(run_agent())
