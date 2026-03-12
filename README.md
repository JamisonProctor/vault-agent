# Vault Agent

**Your notes are full of hidden connections. Vault Agent finds them for you.**

Vault Agent is an AI-powered assistant for [Obsidian](https://obsidian.md) that reads through your notes, understands what they're about, and automatically discovers how they relate to each other. It adds tags, suggests links between related notes, and helps you turn a scattered collection of documents into a connected knowledge base.

Everything runs locally on your machine. Your notes never leave your computer.

## What it does

**Automatic tagging** — Vault Agent reads each note and assigns descriptive tags based on the content. No more manually tagging hundreds of notes.

**Link discovery** — It finds meaningful connections between notes you might never have noticed. Two notes about similar topics, related projects, or overlapping ideas get linked together with an explanation of why.

**Smart deduplication** — Tags like `#product-manager`, `#product-management`, and `#senior-product-manager` get normalized into a clean, consistent taxonomy.

**Non-destructive workflow** — Nothing gets written to your notes until you say so. Vault Agent proposes changes first, lets you review them, then applies only what you approve.

**Resumable** — Processing a large vault can take time. You can stop and restart anytime without losing progress. Vault Agent picks up right where it left off.

## How it works

1. **Process** — Vault Agent sends each note to a local AI model, which extracts tags and searches your vault for related notes
2. **Review** — You see all proposed tags and links before anything changes
3. **Apply** — Tags get added to your note frontmatter, links get added as a Related Notes section

The AI runs entirely on your machine using [Ollama](https://ollama.com). No API keys, no cloud services, no data leaving your computer.

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- An Obsidian vault

### Install

```bash
git clone https://github.com/JamisonProctor/vault-agent.git
cd vault-agent
pip install -e .
ollama pull qwen3.5:9b
```

### Run

```bash
# Process your vault (extracts tags, discovers links)
vault-agent run

# See what it found
vault-agent proposals

# Happy with the results? Apply them
vault-agent apply-tags
vault-agent apply-links
```

## Commands

| Command | What it does |
|---|---|
| `vault-agent run` | Process notes — extract tags and discover links |
| `vault-agent status` | See how many notes are processed and what's pending |
| `vault-agent proposals` | Preview all proposed tags and links |
| `vault-agent apply-tags` | Write proposed tags to your notes |
| `vault-agent apply-links` | Write proposed links to your notes |
| `vault-agent discover-links` | Find connections between notes based on shared tags |
| `vault-agent normalize-tags` | Clean up near-duplicate tags |
| `vault-agent validate` | Check that all proposed link targets point to real files |

## Options

```bash
vault-agent --vault /path/to/vault    # Custom vault location
vault-agent --model qwen3.5:9b        # Choose a different Ollama model
vault-agent run --force                # Re-process all notes from scratch
vault-agent discover-links --force     # Re-evaluate all link candidates
vault-agent validate --fix             # Auto-correct bad link targets
vault-agent normalize-tags --apply     # Apply tag cleanup
```

## Built with

- [Ollama](https://ollama.com) for local AI inference
- [Click](https://click.palletsprojects.com) for the CLI
- [Obsidian](https://obsidian.md) as the target knowledge base
