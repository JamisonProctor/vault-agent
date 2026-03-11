from pathlib import Path

DEFAULT_VAULT_PATH = Path.home() / "Documents" / "Obsidian Vault"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
STATE_FILENAME = ".vault_agent_state.json"
MAX_AGENT_ITERATIONS = 20
DEFAULT_THINK = False  # Thinking spirals on Qwen3.5:9b — disable by default
THINKING_TIMEOUT_S = 60  # If thinking enabled, cut off after this many seconds
