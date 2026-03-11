"""Tool definitions and implementations for the vault agent.

Each tool mirrors Claude Code's approach: read, list, search, edit notes.
Tools are defined as Ollama-compatible function schemas and implemented
as plain Python functions operating on the filesystem.
"""

import re
from pathlib import Path

from vault_agent.config import DEFAULT_VAULT_PATH


# --- Tool Definitions (sent to the LLM) ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": (
                "List all markdown notes in the vault or a specific subfolder. "
                "Returns a list of note paths relative to the vault root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": (
                            "Subfolder to list (relative to vault root). "
                            "Leave empty to list all notes in the vault."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": (
                "Read the full content of a note by its path. "
                "Returns the raw markdown content including frontmatter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the note relative to the vault root (e.g. 'Projects/My Note.md')",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": (
                "Search note contents by keyword or regex pattern. "
                "Returns matching note paths and the lines that matched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex pattern to search for across all notes.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_note_links",
            "description": (
                "Get all existing wiki-links ([[...]]) in a note, both outgoing links "
                "found in the note content and a list of other notes that link back to this note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the note relative to the vault root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_note_metadata",
            "description": (
                "Read the YAML frontmatter and tags from a note. "
                "Returns parsed frontmatter fields and any inline tags (#tag)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the note relative to the vault root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_note",
            "description": (
                "Insert or replace content in a note. Use this to add wiki-links, "
                "tags, or other content. Supports append, prepend, or replacing a specific string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the note relative to the vault root.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["append", "prepend", "replace"],
                        "description": "How to insert the content.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to insert.",
                    },
                    "target": {
                        "type": "string",
                        "description": "For 'replace' operation: the exact string to replace. Ignored for append/prepend.",
                    },
                },
                "required": ["path", "operation", "content"],
            },
        },
    },
]


# --- Tool Implementations ---

def _resolve_path(vault_path: Path, relative: str) -> Path:
    """Resolve a relative note path to an absolute path, with safety check."""
    resolved = (vault_path / relative).resolve()
    if not str(resolved).startswith(str(vault_path.resolve())):
        raise ValueError(f"Path {relative} escapes the vault directory")
    return resolved


def list_notes(vault_path: Path = DEFAULT_VAULT_PATH, folder: str = "") -> str:
    """List all .md files in the vault or a subfolder."""
    search_dir = vault_path / folder if folder else vault_path
    if not search_dir.exists():
        return f"Error: folder '{folder}' does not exist"

    notes = sorted(search_dir.rglob("*.md"))
    relative = [str(n.relative_to(vault_path)) for n in notes if not n.name.startswith(".")]
    if not relative:
        return "No notes found."
    return "\n".join(relative)


def read_note(vault_path: Path = DEFAULT_VAULT_PATH, path: str = "") -> str:
    """Read the full content of a note."""
    if not path:
        return "Error: path is required"
    filepath = _resolve_path(vault_path, path)
    if not filepath.exists():
        return f"Error: note '{path}' not found"
    return filepath.read_text(encoding="utf-8")


def search_vault(vault_path: Path = DEFAULT_VAULT_PATH, query: str = "") -> str:
    """Search all notes for a text pattern. Returns matching lines with context."""
    if not query:
        return "Error: query is required"

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    results = []
    for note in sorted(vault_path.rglob("*.md")):
        if note.name.startswith("."):
            continue
        try:
            content = note.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                rel = str(note.relative_to(vault_path))
                results.append(f"{rel}:{i}: {line.strip()}")
        if len(results) > 100:
            results.append("... (truncated, too many results)")
            break

    return "\n".join(results) if results else "No matches found."


def get_note_links(vault_path: Path = DEFAULT_VAULT_PATH, path: str = "") -> str:
    """Get outgoing wiki-links and backlinks for a note."""
    if not path:
        return "Error: path is required"

    filepath = _resolve_path(vault_path, path)
    if not filepath.exists():
        return f"Error: note '{path}' not found"

    content = filepath.read_text(encoding="utf-8")

    # Extract outgoing [[links]]
    outgoing = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)

    # Find backlinks (other notes that link to this note)
    note_name = filepath.stem
    backlinks = []
    for other in vault_path.rglob("*.md"):
        if other == filepath or other.name.startswith("."):
            continue
        try:
            other_content = other.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if f"[[{note_name}]]" in other_content or f"[[{note_name}|" in other_content:
            backlinks.append(str(other.relative_to(vault_path)))

    lines = []
    lines.append(f"Outgoing links: {outgoing if outgoing else 'none'}")
    lines.append(f"Backlinks: {backlinks if backlinks else 'none'}")
    return "\n".join(lines)


def get_note_metadata(vault_path: Path = DEFAULT_VAULT_PATH, path: str = "") -> str:
    """Extract frontmatter and inline tags from a note."""
    if not path:
        return "Error: path is required"

    filepath = _resolve_path(vault_path, path)
    if not filepath.exists():
        return f"Error: note '{path}' not found"

    content = filepath.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    frontmatter = {}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end].strip()
            for line in fm_text.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    frontmatter[key.strip()] = val.strip()

    # Find inline tags
    inline_tags = re.findall(r'(?:^|\s)#([a-zA-Z][\w/-]*)', content)
    inline_tags = sorted(set(inline_tags))

    lines = []
    lines.append(f"Frontmatter: {frontmatter if frontmatter else 'none'}")
    lines.append(f"Inline tags: {inline_tags if inline_tags else 'none'}")
    return "\n".join(lines)


def edit_note(
    vault_path: Path = DEFAULT_VAULT_PATH,
    path: str = "",
    operation: str = "append",
    content: str = "",
    target: str = "",
) -> str:
    """Edit a note: append, prepend, or replace content."""
    if not path:
        return "Error: path is required"
    if not content:
        return "Error: content is required"

    filepath = _resolve_path(vault_path, path)
    if not filepath.exists():
        return f"Error: note '{path}' not found"

    existing = filepath.read_text(encoding="utf-8")

    if operation == "append":
        new_content = existing + "\n" + content
    elif operation == "prepend":
        new_content = content + "\n" + existing
    elif operation == "replace":
        if not target:
            return "Error: 'target' is required for replace operation"
        if target not in existing:
            return f"Error: target string not found in '{path}'"
        new_content = existing.replace(target, content, 1)
    else:
        return f"Error: unknown operation '{operation}'"

    filepath.write_text(new_content, encoding="utf-8")
    return f"Successfully edited '{path}' ({operation})"


# --- Tool Dispatcher ---

TOOL_IMPLEMENTATIONS = {
    "list_notes": list_notes,
    "read_note": read_note,
    "search_vault": search_vault,
    "get_note_links": get_note_links,
    "get_note_metadata": get_note_metadata,
    "edit_note": edit_note,
}


def execute_tool(name: str, arguments: dict, vault_path: Path = DEFAULT_VAULT_PATH) -> str:
    """Execute a tool by name with the given arguments."""
    func = TOOL_IMPLEMENTATIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    try:
        return func(vault_path=vault_path, **arguments)
    except Exception as e:
        return f"Error executing {name}: {e}"
