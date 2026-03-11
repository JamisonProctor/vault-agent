"""Ollama API client with streaming and tool-calling support for Qwen3.5."""

import json
import sys
import time
import httpx
from vault_agent.config import DEFAULT_OLLAMA_URL, DEFAULT_MODEL, DEFAULT_THINK


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        think: bool = DEFAULT_THINK,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.verbose = verbose
        self.think = think
        self.client = httpx.Client(timeout=600.0)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a streaming chat request to Ollama.

        Streams the response to show real-time progress (thinking tokens,
        content tokens, elapsed time). Assembles and returns the full
        response dict as if it were non-streaming.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "options": {
                "num_ctx": 16384,
            },
        }
        if tools:
            payload["tools"] = tools

        start = time.time()
        content_parts = []
        thinking_parts = []
        tool_calls = []
        role = "assistant"
        content_count = 0
        thinking_count = 0
        last_status_time = 0

        with self.client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                role = msg.get("role", role)

                # Ollama sends thinking tokens in a separate "thinking" field
                thinking_token = msg.get("thinking", "")
                if thinking_token:
                    thinking_parts.append(thinking_token)
                    thinking_count += 1
                    now = time.time()
                    if self.verbose and now - last_status_time > 2:
                        elapsed = now - start
                        recent = "".join(thinking_parts[-5:]).strip()
                        if len(recent) > 60:
                            recent = "..." + recent[-60:]
                        print(
                            f"\r  [thinking {elapsed:.0f}s] {recent}",
                            end="", flush=True,
                        )
                        last_status_time = now

                # Handle content tokens
                content_token = msg.get("content", "")
                if content_token:
                    content_parts.append(content_token)
                    content_count += 1
                    if self.verbose:
                        if thinking_count > 0 and content_count == 1:
                            print()  # newline after thinking status
                        sys.stdout.write(content_token)
                        sys.stdout.flush()

                # Handle tool calls
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])

                # Check if done
                if chunk.get("done"):
                    break

        elapsed = time.time() - start

        if self.verbose:
            print()  # newline after streaming output
            parts = [f"{elapsed:.1f}s"]
            if thinking_count:
                parts.append(f"{thinking_count} thinking tokens")
            parts.append(f"{content_count} output tokens")
            if tool_calls:
                parts.append(f"{len(tool_calls)} tool call(s)")
            print(f"  [llm] {' | '.join(parts)}")

        # Assemble the response in the same format as non-streaming
        assembled = {
            "message": {
                "role": role,
                "content": "".join(content_parts),
            },
            "done": True,
            "total_duration_s": elapsed,
        }
        if thinking_parts:
            assembled["thinking"] = "".join(thinking_parts)
        if tool_calls:
            assembled["message"]["tool_calls"] = tool_calls

        return assembled

    def is_tool_call(self, response: dict) -> bool:
        msg = response.get("message", {})
        return bool(msg.get("tool_calls"))

    def get_tool_calls(self, response: dict) -> list[dict]:
        return response.get("message", {}).get("tool_calls", [])

    def get_content(self, response: dict) -> str:
        return response.get("message", {}).get("content", "")

    def get_message(self, response: dict) -> dict:
        return response.get("message", {})
