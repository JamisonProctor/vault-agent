"""Ollama API client with tool-calling support for Qwen3.5."""

import json
import httpx
from vault_agent.config import DEFAULT_OLLAMA_URL, DEFAULT_MODEL


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=600.0)  # 10 min timeout for slow reasoning

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat request to Ollama with optional tool definitions.

        Returns the full response dict including message and any tool_calls.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 16384,
            },
        }
        if tools:
            payload["tools"] = tools

        resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    def is_tool_call(self, response: dict) -> bool:
        """Check if the response contains tool calls."""
        msg = response.get("message", {})
        return bool(msg.get("tool_calls"))

    def get_tool_calls(self, response: dict) -> list[dict]:
        """Extract tool calls from a response."""
        return response.get("message", {}).get("tool_calls", [])

    def get_content(self, response: dict) -> str:
        """Extract text content from a response."""
        return response.get("message", {}).get("content", "")

    def get_message(self, response: dict) -> dict:
        """Extract the full message dict (for appending to history)."""
        return response.get("message", {})
