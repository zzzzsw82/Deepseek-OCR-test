from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
try:
    from langchain_ollama import ChatOllama  # type: ignore
except ImportError:  # fallback for older environments
    from langchain_community.chat_models import ChatOllama  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from langgraph_tools import build_formatter_tool, build_ocr_tool

logger = logging.getLogger(__name__)


class InvoiceAgentRuntime:
    """Runtime that wires DeepSeek OCR + CSV formatting tools into a LangGraph agent."""

    def __init__(
        self,
        thread_id: str = "invoice",
        prompt_path: Optional[Path] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        self._base_path = base_path or Path(__file__).resolve().parent.parent
        load_dotenv(self._base_path / ".env", override=False)

        prompt_file = prompt_path or (self._base_path / "prompts" / "agent_prompt.txt")
        if not prompt_file.exists():
            raise FileNotFoundError(f"Agent prompt not found at {prompt_file}")
        self._prompt = prompt_file.read_text(encoding="utf-8")

        self._config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        self._checkpointer = InMemorySaver()
        self._init_lock = asyncio.Lock()

        self._agent = None
        self._tools = []

        self._model_name = os.getenv("INVOICE_AGENT_MODEL", "qwen3:14b")
        self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def _build_tools(self) -> None:
        ocr_tool = build_ocr_tool()
        formatter_tool = build_formatter_tool()
        self._tools = [ocr_tool, formatter_tool]
        logger.info("Registered LangGraph tools: %s", [tool.name for tool in self._tools])

    async def initialize(self) -> None:
        """Create the LangGraph agent if it has not been initialized already."""
        if self._agent is not None:
            return

        async with self._init_lock:
            if self._agent is not None:
                return

            self._build_tools()
            model = ChatOllama(
                model=self._model_name,
                base_url=self._ollama_url,
                temperature=self._ollama_temperature,
            )
            self._agent = create_react_agent(
                model=model,
                tools=self._tools,
                prompt=self._prompt,
                checkpointer=self._checkpointer,
            )

    async def invoke(self, message: str) -> Dict[str, Any]:
        if not message:
            raise ValueError("Message cannot be empty.")

        await self.initialize()
        assert self._agent is not None

        return await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            self._config,
        )

    @staticmethod
    def latest_content(result: Dict[str, Any]) -> str:
        messages = result.get("messages") or []
        if not messages:
            return ""
        return messages[-1].content  # type: ignore[union-attr,index]


_invoice_runtime_singleton: Optional[InvoiceAgentRuntime] = None


def get_invoice_runtime() -> InvoiceAgentRuntime:
    global _invoice_runtime_singleton
    if _invoice_runtime_singleton is None:
        _invoice_runtime_singleton = InvoiceAgentRuntime()
    return _invoice_runtime_singleton
