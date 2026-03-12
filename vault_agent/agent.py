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

        try:
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

        except KeyboardInterrupt:
            self.state.save()
            total_time = time.time() - start_time
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            remaining_count = total - i
            print(f"\n\nInterrupted! {processed} processed, {errors} errors in {mins}m{secs:02d}s")
            print(f"{remaining_count} note(s) remaining — run again to resume.")
            return {"processed": processed, "skipped": remaining_count, "errors": errors, "interrupted": True}

        self.state.mark_run_completed()
        self.state.save()

        total_time = time.time() - start_time
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        print(f"Done! {processed} processed, {errors} errors in {mins}m{secs:02d}s")

        return {"processed": processed, "skipped": total - processed - errors, "errors": errors}

    # --- Link Discovery (tag-based + LLM-confirmed) ---

    def _build_tag_index(self) -> dict[str, list[str]]:
        """Build a reverse index: tag -> list of note paths."""
        index: dict[str, list[str]] = {}
        for path, info in self.state.data["notes"].items():
            for tag in info.get("proposed_tags", []):
                index.setdefault(tag, []).append(path)
        return index

    def _find_candidate_pairs(
        self, min_shared_tags: int = 2, max_tag_frequency: int = 0,
    ) -> list[tuple[str, str, list[str]]]:
        """Find pairs of notes that share at least min_shared_tags tags.

        Args:
            min_shared_tags: Minimum number of shared tags to consider a pair.
            max_tag_frequency: If > 0, ignore tags that appear on more than this
                many notes (they're too generic to indicate real connections).

        Returns list of (note_a, note_b, shared_tags) sorted by overlap count descending.
        """
        notes = self.state.data["notes"]
        pairs: dict[tuple[str, str], list[str]] = {}

        tag_index = self._build_tag_index()

        # Filter out overly common tags
        skipped_tags = []
        if max_tag_frequency > 0:
            for tag in list(tag_index.keys()):
                if len(tag_index[tag]) > max_tag_frequency:
                    skipped_tags.append((tag, len(tag_index[tag])))
                    del tag_index[tag]
            if skipped_tags:
                skipped_tags.sort(key=lambda x: -x[1])
                print(f"Skipping {len(skipped_tags)} overly common tag(s):")
                for tag, count in skipped_tags:
                    print(f"  #{tag} ({count} notes)")
                print()

        for tag, note_paths in tag_index.items():
            for i, a in enumerate(note_paths):
                for b in note_paths[i + 1:]:
                    key = tuple(sorted([a, b]))
                    pairs.setdefault(key, []).append(tag)

        # Filter by minimum overlap and sort by count
        results = [
            (a, b, tags) for (a, b), tags in pairs.items()
            if len(tags) >= min_shared_tags
        ]
        results.sort(key=lambda x: len(x[2]), reverse=True)
        return results

    def _get_existing_links(self, note_path: str) -> set[str]:
        """Get the set of note names this note already links to."""
        import re
        filepath = self.vault_path / note_path
        if not filepath.exists():
            return set()
        content = filepath.read_text(encoding="utf-8")
        return set(re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content))

    def confirm_link(self, note_a: str, note_b: str, shared_tags: list[str]) -> dict | None:
        """Ask the LLM whether two notes should be linked, given their shared tags.

        Returns a dict with link details or None if no link should be made.
        """
        # Read summaries from state if available
        info_a = self.state.data["notes"].get(note_a, {})
        info_b = self.state.data["notes"].get(note_b, {})

        # Read first 500 chars of each note for context
        try:
            content_a = (self.vault_path / note_a).read_text(encoding="utf-8")[:500]
            content_b = (self.vault_path / note_b).read_text(encoding="utf-8")[:500]
        except (FileNotFoundError, UnicodeDecodeError):
            return None

        prompt = f"""You are evaluating whether two Obsidian notes should be linked.

Note A: "{note_a}"
Tags: {info_a.get('proposed_tags', [])}
Preview:
{content_a}

Note B: "{note_b}"
Tags: {info_b.get('proposed_tags', [])}
Preview:
{content_b}

Shared tags: {shared_tags}

Should these notes be linked? Only say yes if there's a genuine, meaningful connection — not just surface-level keyword overlap.

Respond with JSON only:
{{"should_link": true/false, "reason": "why they should or should not be linked", "link_from": "which note should contain the link (path)", "link_to": "the note being linked to (path)"}}
"""
        response = self.client.chat([
            {"role": "system", "content": "You evaluate connections between notes. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])

        raw = self.client.get_content(response)
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            result = json.loads(text)
            if result.get("should_link"):
                return result
            return None
        except json.JSONDecodeError:
            self._log(f"Failed to parse link confirmation: {raw[:200]}")
            return None

    def discover_links(self, min_shared_tags: int = 2, max_tag_frequency: int = 0) -> dict:
        """Discover links between notes based on tag overlap, confirmed by LLM.

        Resumable — tracks evaluated pairs in state so interrupted runs
        pick up where they left off. Use --force to re-evaluate all pairs.

        Returns a summary dict.
        """
        candidates = self._find_candidate_pairs(min_shared_tags, max_tag_frequency)

        if not candidates:
            print(f"No note pairs found with {min_shared_tags}+ shared tags.")
            print("Run 'vault-agent run' first to extract tags.")
            return {"candidates": 0, "confirmed": 0, "skipped": 0}

        # Filter out already-evaluated pairs
        remaining = [
            (a, b, tags) for a, b, tags in candidates
            if not self.state.is_pair_evaluated(a, b)
        ]
        already_done = len(candidates) - len(remaining)

        if already_done > 0:
            print(f"Found {len(candidates)} candidate pair(s), {already_done} already evaluated.")
            print(f"Resuming with {len(remaining)} remaining pair(s).\n")
        else:
            print(f"Found {len(candidates)} candidate pair(s) with {min_shared_tags}+ shared tags.\n")

        if not remaining:
            print("All pairs already evaluated. Use --force to re-evaluate.")
            return {"candidates": len(candidates), "confirmed": 0, "skipped": 0}

        confirmed = 0
        skipped = 0
        evaluated = 0
        start_time = time.time()

        try:
            for i, (note_a, note_b, shared_tags) in enumerate(remaining, 1):
                # Skip if already linked in the actual note content
                links_a = self._get_existing_links(note_a)
                links_b = self._get_existing_links(note_b)
                name_a = Path(note_a).stem
                name_b = Path(note_b).stem

                if name_b in links_a or name_a in links_b:
                    self._log(f"Already linked: {note_a} <-> {note_b}")
                    self.state.mark_pair_evaluated(note_a, note_b)
                    skipped += 1
                    evaluated += 1
                    continue

                elapsed = time.time() - start_time
                if evaluated > 0:
                    avg = elapsed / evaluated
                    eta_remaining = avg * (len(remaining) - i + 1)
                    mins = int(eta_remaining // 60)
                    secs = int(eta_remaining % 60)
                    eta = f"~{mins}m{secs:02d}s remaining"
                else:
                    eta = "estimating..."

                overall_i = already_done + i
                pct = int(100 * overall_i / len(candidates))
                print(f"[{overall_i}/{len(candidates)}] {note_a} <-> {note_b}")
                print(f"  Shared tags: {', '.join(f'#{t}' for t in shared_tags)}")
                print(f"  [{pct}%] {eta}")

                result = self.confirm_link(note_a, note_b, shared_tags)

                # Mark as evaluated regardless of outcome
                self.state.mark_pair_evaluated(note_a, note_b)

                if result:
                    link_from = result.get("link_from", note_a)
                    link_to = result.get("link_to", note_b)
                    reason = result.get("reason", "")

                    # Store in state, skipping duplicates
                    note_info = self.state.data["notes"].get(link_from, {})
                    existing_links = note_info.get("proposed_links", [])
                    existing_targets = {l.get("target") for l in existing_links}
                    if link_to not in existing_targets:
                        existing_links.append({
                            "target": link_to,
                            "reason": reason,
                        })
                        note_info["proposed_links"] = existing_links
                        note_info["links_applied"] = False
                        self.state.data["notes"][link_from] = note_info

                    print(f"  -> LINK: [[{Path(link_to).stem}]] in {link_from}")
                    print(f"     Reason: {reason}")
                    confirmed += 1
                else:
                    print(f"  -> No link needed")
                    skipped += 1

                evaluated += 1
                self.state.save()  # Save after each pair for resume support
                print()

        except KeyboardInterrupt:
            self.state.save()
            total_time = time.time() - start_time
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            remaining_count = len(remaining) - i
            print(f"\n\nInterrupted! {confirmed} links confirmed, {skipped} skipped in {mins}m{secs:02d}s")
            print(f"{remaining_count} pair(s) remaining — run again to resume.")
            return {"candidates": len(candidates), "confirmed": confirmed, "skipped": skipped, "interrupted": True}

        total_time = time.time() - start_time
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        print(f"Done! {confirmed} links confirmed, {skipped} skipped in {mins}m{secs:02d}s")

        return {"candidates": len(candidates), "confirmed": confirmed, "skipped": skipped}
