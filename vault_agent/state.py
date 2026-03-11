"""State file management for tracking vault processing progress.

Maintains a JSON state file that records which notes have been processed,
content hashes for change detection, and proposed tags/links.
"""

import hashlib
import json
import time
from pathlib import Path

from vault_agent.config import STATE_FILENAME


def _hash_content(content: str) -> str:
    """Create a short content hash for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class VaultState:
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.state_file = vault_path / STATE_FILENAME
        self.data: dict = self._load()

    def _load(self) -> dict:
        """Load state from disk, or return empty state."""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        return {
            "version": 1,
            "notes": {},
            "last_run": None,
            "last_completed_run": None,
        }

    def save(self) -> None:
        """Persist state to disk."""
        self.state_file.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_notes_to_process(self, force: bool = False) -> list[str]:
        """Return list of note paths that need processing.

        A note needs processing if:
        - force=True (re-process everything)
        - It's not in the state file (new note)
        - Its content hash has changed (modified note)
        """
        all_notes = sorted(self.vault_path.rglob("*.md"))
        to_process = []

        for note_path in all_notes:
            if note_path.name.startswith("."):
                continue
            relative = str(note_path.relative_to(self.vault_path))

            if force:
                to_process.append(relative)
                continue

            try:
                content = note_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            current_hash = _hash_content(content)
            stored = self.data["notes"].get(relative)

            if stored is None or stored.get("content_hash") != current_hash:
                to_process.append(relative)

        return to_process

    def mark_processed(
        self,
        note_path: str,
        content_hash: str,
        proposed_tags: list[str] | None = None,
        proposed_links: list[dict] | None = None,
    ) -> None:
        """Record that a note has been processed."""
        self.data["notes"][note_path] = {
            "content_hash": content_hash,
            "processed_at": time.time(),
            "proposed_tags": proposed_tags or [],
            "proposed_links": proposed_links or [],
            "tags_applied": False,
            "links_applied": False,
        }

    def mark_run_started(self) -> None:
        self.data["last_run"] = time.time()

    def mark_run_completed(self) -> None:
        self.data["last_completed_run"] = time.time()

    def get_content_hash(self, note_path: str) -> str:
        """Compute current content hash for a note."""
        filepath = self.vault_path / note_path
        content = filepath.read_text(encoding="utf-8")
        return _hash_content(content)

    def get_pending_tags(self) -> dict[str, list[str]]:
        """Return all notes with proposed but unapplied tags."""
        pending = {}
        for path, info in self.data["notes"].items():
            if info.get("proposed_tags") and not info.get("tags_applied"):
                pending[path] = info["proposed_tags"]
        return pending

    def get_pending_links(self) -> dict[str, list[dict]]:
        """Return all notes with proposed but unapplied links."""
        pending = {}
        for path, info in self.data["notes"].items():
            if info.get("proposed_links") and not info.get("links_applied"):
                pending[path] = info["proposed_links"]
        return pending

    def mark_tags_applied(self, note_path: str) -> None:
        if note_path in self.data["notes"]:
            self.data["notes"][note_path]["tags_applied"] = True

    def mark_links_applied(self, note_path: str) -> None:
        if note_path in self.data["notes"]:
            self.data["notes"][note_path]["links_applied"] = True

    def summary(self) -> str:
        """Return a human-readable status summary."""
        total = len(self.data["notes"])
        pending_tags = len(self.get_pending_tags())
        pending_links = len(self.get_pending_links())
        last_run = self.data.get("last_completed_run")

        lines = [
            f"Notes processed: {total}",
            f"Pending tag proposals: {pending_tags}",
            f"Pending link proposals: {pending_links}",
        ]
        if last_run:
            from datetime import datetime
            ts = datetime.fromtimestamp(last_run).strftime("%Y-%m-%d %H:%M")
            lines.append(f"Last completed run: {ts}")
        else:
            lines.append("Last completed run: never")
        return "\n".join(lines)
