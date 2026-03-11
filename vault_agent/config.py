from pathlib import Path

DEFAULT_VAULT_PATH = Path.home() / "Documents" / "Obsidian Vault"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
STATE_FILENAME = ".vault_agent_state.json"
MAX_AGENT_ITERATIONS = 20
