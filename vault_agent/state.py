"""State file management for tracking vault processing progress.

Maintains a JSON state file that records which notes have been processed,
content hashes for change detection, and proposed tags/links.
"""

import hashlib
import json
import time
from collections import defaultdict
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
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            # Ensure evaluated_pairs exists (added in v1.1)
            data.setdefault("evaluated_pairs", [])
            return data
        return {
            "version": 1,
            "notes": {},
            "evaluated_pairs": [],
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

    def is_pair_evaluated(self, note_a: str, note_b: str) -> bool:
        """Check if a candidate pair has already been evaluated."""
        key = "|".join(sorted([note_a, note_b]))
        return key in set(self.data.get("evaluated_pairs", []))

    def mark_pair_evaluated(self, note_a: str, note_b: str) -> None:
        """Record that a candidate pair has been evaluated (linked or rejected)."""
        key = "|".join(sorted([note_a, note_b]))
        pairs = self.data.setdefault("evaluated_pairs", [])
        if key not in pairs:
            pairs.append(key)

    def clear_evaluated_pairs(self) -> None:
        """Reset evaluated pairs tracking (for full re-evaluation)."""
        self.data["evaluated_pairs"] = []

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

    # --- Validation ---

    def _get_all_vault_files(self) -> set[str]:
        """Return set of all .md file paths relative to vault root."""
        return {
            str(p.relative_to(self.vault_path))
            for p in self.vault_path.rglob("*.md")
            if not p.name.startswith(".")
        }

    def _fuzzy_match_target(self, bad_target: str, valid_files: set[str]) -> str | None:
        """Try to fuzzy-match a bad link target to a real file.

        Strategies:
        1. Case-insensitive match
        2. Substring match on filename
        3. Levenshtein distance on filename (threshold <= 3)
        """
        from vault_agent.tags import _levenshtein

        bad_name = Path(bad_target).name.lower()
        bad_stem = Path(bad_target).stem.lower()

        # Build lookup structures
        name_to_path: dict[str, str] = {}
        for f in valid_files:
            name_to_path[Path(f).name.lower()] = f

        # Strategy 1: exact filename, case-insensitive
        if bad_name in name_to_path:
            return name_to_path[bad_name]

        # Strategy 2: stem match (ignore .md extension issues)
        for f in valid_files:
            if Path(f).stem.lower() == bad_stem:
                return f

        # Strategy 3: Levenshtein on stems
        best_match = None
        best_dist = 4  # threshold
        for f in valid_files:
            dist = _levenshtein(bad_stem, Path(f).stem.lower())
            if dist < best_dist:
                best_dist = dist
                best_match = f

        return best_match

    def validate_links(self, fix: bool = False) -> dict:
        """Validate all proposed link targets exist as real files.

        Args:
            fix: If True, attempt to correct bad targets via fuzzy matching
                 and remove links that can't be fixed.

        Returns:
            dict with validation results:
            - valid: count of valid links
            - invalid: count of invalid links
            - fixed: count of links corrected (when fix=True)
            - removed: count of links removed (when fix=True)
            - details: list of (note_path, target, status, correction) tuples
        """
        valid_files = self._get_all_vault_files()
        results = {
            "valid": 0,
            "invalid": 0,
            "fixed": 0,
            "removed": 0,
            "details": [],
        }

        for note_path, info in self.data["notes"].items():
            proposed = info.get("proposed_links", [])
            if not proposed:
                continue

            cleaned_links = []
            for link in proposed:
                target = link.get("target", "")
                if target in valid_files:
                    results["valid"] += 1
                    cleaned_links.append(link)
                    continue

                # Also check if stem.md matches (LLM might omit folder path)
                stem_match = None
                for f in valid_files:
                    if Path(f).name == target or Path(f).name == target + ".md":
                        stem_match = f
                        break

                if stem_match:
                    results["valid"] += 1
                    if fix:
                        link["target"] = stem_match
                        results["fixed"] += 1
                        results["details"].append((note_path, target, "fixed", stem_match))
                    cleaned_links.append(link)
                    continue

                # Try fuzzy match
                fuzzy = self._fuzzy_match_target(target, valid_files)
                if fuzzy and fix:
                    link["target"] = fuzzy
                    results["fixed"] += 1
                    results["details"].append((note_path, target, "fixed", fuzzy))
                    cleaned_links.append(link)
                elif fuzzy:
                    results["invalid"] += 1
                    results["details"].append((note_path, target, "fixable", fuzzy))
                else:
                    results["invalid"] += 1
                    if fix:
                        results["removed"] += 1
                        results["details"].append((note_path, target, "removed", None))
                    else:
                        results["details"].append((note_path, target, "invalid", None))

            if fix:
                info["proposed_links"] = cleaned_links
                if not cleaned_links:
                    info["links_applied"] = False

        if fix:
            self.save()

        return results

    # --- Tag normalization ---

    def apply_tag_normalization(self, mapping: dict[str, str]) -> int:
        """Apply a tag normalization mapping to all proposed tags in state.

        Args:
            mapping: dict of variant_tag -> canonical_tag

        Returns:
            Number of tag replacements made.
        """
        replacements = 0
        for info in self.data["notes"].values():
            tags = info.get("proposed_tags", [])
            new_tags = []
            for tag in tags:
                canonical = mapping.get(tag, tag)
                if canonical != tag:
                    replacements += 1
                new_tags.append(canonical)
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for t in new_tags:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            info["proposed_tags"] = deduped

        self.save()
        return replacements
