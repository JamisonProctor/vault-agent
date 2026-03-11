"""Core agent loop — sends prompts + tools to the LLM, executes tool calls, loops.

Mirrors Claude Code's agentic pattern:
1. Send system prompt + user message + tool definitions to the LLM
2. If the LLM responds with tool calls, execute them and feed results back
3. Loop until the LLM responds with a final text message (no more tool calls)
"""

import json
import time
from pathlib import Path

from vault_agent.config import MAX_AGENT_ITERATIONS
from vault_agent.ollama_client import OllamaClient
from vault_agent.state import VaultState
from vault_agent.tools import TOOL_DEFINITIONS, execute_tool


SYSTEM_PROMPT = """\
You are a knowledge management agent for an Obsidian vault. Your job is to analyze \
notes and discover connections between them.

You have access to tools for reading, searching, and analyzing notes in the vault. \
Use these tools to understand the content of each note you are asked to process.

When analyzing a note, you should:
1. Read the note content
2. Extract 3-8 descriptive tags/themes that capture the key topics
3. Search the vault for related notes using keywords from the content
4. Suggest wiki-links to other notes that are semantically related

Respond with a JSON object in this exact format:
{
    "tags": ["tag1", "tag2", "tag3"],
    "links": [
        {"target": "Other Note.md", "reason": "Brief explanation of why these notes are related"}
    ],
    "summary": "One sentence summary of the note"
}

Rules:
- Tags should be lowercase, use hyphens for multi-word tags (e.g. "machine-learning")
- Only suggest links to notes that actually exist in the vault
- Only suggest links that are not already present in the note
- Be conservative — only suggest genuinely meaningful connections
- Do NOT suggest links to the note itself
"""


class Agent:
    def __init__(
        self,
        vault_path: Path,
        client: OllamaClient,
        state: VaultState,
        verbose: bool = False,
    ):
        self.vault_path = vault_path
        self.client = client
        self.state = state
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [agent] {msg}")

    def _run_agent_loop(self, messages: list[dict]) -> str:
        """Run the tool-calling loop until the LLM produces a final text response."""
        for iteration in range(MAX_AGENT_ITERATIONS):
            self._log(f"iteration {iteration + 1}")

            response = self.client.chat(messages, tools=TOOL_DEFINITIONS)

            if not self.client.is_tool_call(response):
                return self.client.get_content(response)

            # Append assistant message with tool calls to history
            assistant_msg = self.client.get_message(response)
            messages.append(assistant_msg)

            # Execute each tool call and add results
            for tool_call in self.client.get_tool_calls(response):
                func_name = tool_call["function"]["name"]
                arguments = tool_call["function"].get("arguments", {})
                self._log(f"tool call: {func_name}({arguments})")

                result = execute_tool(func_name, arguments, self.vault_path)
                self._log(f"result: {result[:200]}...")

                messages.append({
                    "role": "tool",
                    "content": result,
                })

        return "Error: agent exceeded maximum iterations without producing a final response."

    def process_note(self, note_path: str) -> dict | None:
        """Process a single note: extract tags and discover links.

        Returns parsed result dict or None on failure.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyze the note at '{note_path}'. "
                    f"Read it, search for related notes, and suggest tags and links. "
                    f"Respond with JSON only."
                ),
            },
        ]

        raw_response = self._run_agent_loop(messages)

        # Parse JSON from response (handle markdown code blocks)
        try:
            text = raw_response.strip()
            if text.startswith("```"):
                # Strip markdown code fence
                lines = text.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            self._log(f"Failed to parse JSON response: {raw_response[:300]}")
            return None

    def run(self, force: bool = False, progress_callback=None) -> dict:
        """Run the agent over all notes that need processing.

        Returns a summary dict with counts and results.
        """
        self.state.mark_run_started()
        self.state.save()

        notes = self.state.get_notes_to_process(force=force)
        total = len(notes)

        if total == 0:
            print("All notes are up to date. Use --force to re-process.")
            return {"processed": 0, "skipped": 0, "errors": 0}

        print(f"Processing {total} note(s)...\n")

        processed = 0
        errors = 0
        start_time = time.time()

        for i, note_path in enumerate(notes, 1):
            elapsed = time.time() - start_time
            if i > 1:
                avg_per_note = elapsed / (i - 1)
                remaining = avg_per_note * (total - i + 1)
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                eta = f"~{mins}m{secs:02d}s remaining"
            else:
                eta = "estimating..."

            # Progress bar
            bar_width = 30
            filled = int(bar_width * i / total)
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
            pct = int(100 * i / total)
            print(f"[{i}/{total}] Processing \"{note_path}\"...")
            print(f"  [{bar}] {pct}% | {eta}")

            if progress_callback:
                progress_callback(i, total, note_path)

            result = self.process_note(note_path)

            if result:
                content_hash = self.state.get_content_hash(note_path)
                tags = result.get("tags", [])
                links = result.get("links", [])

                self.state.mark_processed(
                    note_path=note_path,
                    content_hash=content_hash,
                    proposed_tags=tags,
                    proposed_links=links,
                )
                self.state.save()  # Save after each note for resume support

                tag_str = ", ".join(f"#{t}" for t in tags) if tags else "none"
                print(f"  Tags: {tag_str}")
                print(f"  Links: {len(links)} suggested")
                processed += 1
            else:
                print(f"  Error: failed to process")
                errors += 1

            print()

        self.state.mark_run_completed()
        self.state.save()

        total_time = time.time() - start_time
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        print(f"Done! {processed} processed, {errors} errors in {mins}m{secs:02d}s")

        return {"processed": processed, "skipped": total - processed - errors, "errors": errors}
