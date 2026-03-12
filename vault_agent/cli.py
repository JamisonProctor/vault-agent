"""CLI entry point for the vault agent."""

import click
from pathlib import Path

from vault_agent.config import DEFAULT_VAULT_PATH, DEFAULT_OLLAMA_URL, DEFAULT_MODEL


@click.group()
@click.option("--vault", type=click.Path(exists=True), default=str(DEFAULT_VAULT_PATH), help="Path to Obsidian vault")
@click.option("--model", default=DEFAULT_MODEL, help="Ollama model name")
@click.option("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
@click.pass_context
def cli(ctx, vault, model, ollama_url):
    """Vault Agent — local LLM-powered Obsidian link discovery and tagging."""
    ctx.ensure_object(dict)
    ctx.obj["vault_path"] = Path(vault)
    ctx.obj["model"] = model
    ctx.obj["ollama_url"] = ollama_url


@cli.command()
@click.option("--force", is_flag=True, help="Re-process all notes, ignoring state")
@click.option("--verbose", "-v", is_flag=True, help="Show agent debug output")
@click.option("--think/--no-think", default=False, help="Enable model thinking (slow, disabled by default)")
@click.pass_context
def run(ctx, force, verbose, think):
    """Process vault notes — extract tags and discover links."""
    from vault_agent.agent import Agent
    from vault_agent.ollama_client import OllamaClient
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    client = OllamaClient(base_url=ctx.obj["ollama_url"], model=ctx.obj["model"], verbose=verbose, think=think)
    state = VaultState(vault_path)
    agent = Agent(vault_path=vault_path, client=client, state=state, verbose=verbose)

    agent.run(force=force)


@cli.command()
@click.pass_context
def status(ctx):
    """Show processing status and pending proposals."""
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    state = VaultState(vault_path)

    print("=== Vault Agent Status ===\n")
    print(state.summary())

    # Count unprocessed notes
    to_process = state.get_notes_to_process()
    if to_process:
        print(f"\nNotes needing processing: {len(to_process)}")
        for n in to_process[:10]:
            print(f"  - {n}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
    else:
        print("\nAll notes are up to date.")


@cli.command(name="discover-links")
@click.option("--min-tags", default=2, help="Minimum shared tags to consider a pair (default: 2)")
@click.option("--verbose", "-v", is_flag=True, help="Show agent debug output")
@click.option("--think/--no-think", default=False, help="Enable model thinking")
@click.pass_context
def discover_links(ctx, min_tags, verbose, think):
    """Discover links between notes based on shared tags (run after tagging)."""
    from vault_agent.agent import Agent
    from vault_agent.ollama_client import OllamaClient
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    client = OllamaClient(base_url=ctx.obj["ollama_url"], model=ctx.obj["model"], verbose=verbose, think=think)
    state = VaultState(vault_path)
    agent = Agent(vault_path=vault_path, client=client, state=state, verbose=verbose)

    agent.discover_links(min_shared_tags=min_tags)


@cli.command()
@click.pass_context
def proposals(ctx):
    """Show all pending tag and link proposals."""
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    state = VaultState(vault_path)

    pending_tags = state.get_pending_tags()
    pending_links = state.get_pending_links()

    if not pending_tags and not pending_links:
        print("No pending proposals.")
        return

    if pending_tags:
        print("=== Pending Tag Proposals ===\n")
        for path, tags in pending_tags.items():
            tag_str = ", ".join(f"#{t}" for t in tags)
            print(f"  {path}: {tag_str}")
        print()

    if pending_links:
        print("=== Pending Link Proposals ===\n")
        for path, links in pending_links.items():
            print(f"  {path}:")
            for link in links:
                print(f"    -> [[{link['target']}]] — {link.get('reason', '')}")
        print()


@cli.command()
@click.option("--fix", is_flag=True, help="Auto-fix bad targets (fuzzy match) and remove unfixable ones")
@click.pass_context
def validate(ctx, fix):
    """Validate proposed link targets exist as real files."""
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    state = VaultState(vault_path)

    action = "Validating and fixing" if fix else "Validating"
    print(f"{action} proposed link targets...\n")

    results = state.validate_links(fix=fix)

    print(f"Valid links:   {results['valid']}")
    print(f"Invalid links: {results['invalid']}")
    if fix:
        print(f"Fixed:         {results['fixed']}")
        print(f"Removed:       {results['removed']}")

    # Show details
    details = results["details"]
    if details:
        print()
        for note_path, target, status, correction in details:
            if status == "fixed":
                print(f"  FIXED   {note_path}")
                print(f"          {target} -> {correction}")
            elif status == "fixable":
                print(f"  FIXABLE {note_path}")
                print(f"          {target} -> {correction}")
                print(f"          (run with --fix to correct)")
            elif status == "removed":
                print(f"  REMOVED {note_path}")
                print(f"          {target} (no match found)")
            else:
                print(f"  INVALID {note_path}")
                print(f"          {target} (no match found)")

    if not fix and results["invalid"] > 0:
        print(f"\nRun 'vault-agent validate --fix' to auto-correct fixable targets and remove the rest.")


@cli.command(name="normalize-tags")
@click.option("--apply", is_flag=True, help="Apply the normalization (default: preview only)")
@click.option("--threshold", default=2, help="Levenshtein distance threshold for fuzzy matching (default: 2)")
@click.pass_context
def normalize_tags(ctx, apply, threshold):
    """Preview and apply tag normalization to merge near-duplicates."""
    from vault_agent.state import VaultState
    from vault_agent.tags import build_normalization_map, get_tag_counts

    vault_path = ctx.obj["vault_path"]
    state = VaultState(vault_path)

    tag_counts = get_tag_counts(state.data)
    total_unique = len(tag_counts)

    mapping = build_normalization_map(tag_counts, distance_threshold=threshold)

    if not mapping:
        print(f"{total_unique} unique tags found. No near-duplicates detected.")
        return

    # Group by canonical tag for display
    from collections import defaultdict
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for variant, canonical in mapping.items():
        groups[canonical].append((variant, tag_counts.get(variant, 0)))

    print(f"{total_unique} unique tags found. {len(mapping)} tags will be merged:\n")
    for canonical, variants in sorted(groups.items()):
        canonical_count = tag_counts.get(canonical, 0)
        print(f"  #{canonical} ({canonical_count} notes) <-")
        for variant, count in sorted(variants, key=lambda x: -x[1]):
            print(f"    #{variant} ({count} notes)")

    after_count = total_unique - len(mapping)
    print(f"\nBefore: {total_unique} unique tags")
    print(f"After:  {after_count} unique tags (-{len(mapping)})")

    if apply:
        replacements = state.apply_tag_normalization(mapping)
        print(f"\nApplied {replacements} tag replacements across all notes.")
    else:
        print(f"\nThis is a preview. Run with --apply to make changes.")


@cli.command(name="apply-tags")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def apply_tags(ctx, yes):
    """Apply all pending tag proposals to notes."""
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    state = VaultState(vault_path)
    pending = state.get_pending_tags()

    if not pending:
        print("No pending tag proposals.")
        return

    print(f"Will apply tags to {len(pending)} note(s):\n")
    for path, tags in pending.items():
        tag_str = ", ".join(f"#{t}" for t in tags)
        print(f"  {path}: {tag_str}")

    if not yes:
        click.confirm("\nProceed?", abort=True)

    for path, tags in pending.items():
        filepath = vault_path / path
        if not filepath.exists():
            print(f"  Skipping {path} (file not found)")
            continue

        content = filepath.read_text(encoding="utf-8")

        # Build or update frontmatter tags
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                fm = content[3:end]
                body = content[end + 3:]
                if "tags:" in fm:
                    # Merge with existing tags
                    import re
                    match = re.search(r'tags:\s*\[([^\]]*)\]', fm)
                    if match:
                        existing = [t.strip().strip('"').strip("'") for t in match.group(1).split(",") if t.strip()]
                        merged = sorted(set(existing + tags))
                        tag_list = ", ".join(merged)
                        fm = re.sub(r'tags:\s*\[[^\]]*\]', f'tags: [{tag_list}]', fm)
                    else:
                        # tags as list items
                        tag_lines = "\n".join(f"  - {t}" for t in sorted(set(tags)))
                        fm = fm.rstrip() + f"\ntags:\n{tag_lines}\n"
                else:
                    tag_list = ", ".join(sorted(tags))
                    fm = fm.rstrip() + f"\ntags: [{tag_list}]\n"
                content = f"---{fm}---{body}"
            else:
                tag_list = ", ".join(sorted(tags))
                content = f"---\ntags: [{tag_list}]\n---\n{content}"
        else:
            tag_list = ", ".join(sorted(tags))
            content = f"---\ntags: [{tag_list}]\n---\n{content}"

        filepath.write_text(content, encoding="utf-8")
        state.mark_tags_applied(path)
        print(f"  Applied tags to {path}")

    state.save()
    print("\nDone!")


@cli.command(name="apply-links")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def apply_links(ctx, yes):
    """Apply all pending link proposals to notes."""
    from vault_agent.state import VaultState

    vault_path = ctx.obj["vault_path"]
    state = VaultState(vault_path)
    pending = state.get_pending_links()

    if not pending:
        print("No pending link proposals.")
        return

    print(f"Will add links to {len(pending)} note(s):\n")
    for path, links in pending.items():
        print(f"  {path}:")
        for link in links:
            print(f"    -> [[{link['target']}]] — {link.get('reason', '')}")

    if not yes:
        click.confirm("\nProceed?", abort=True)

    for path, links in pending.items():
        filepath = vault_path / path
        if not filepath.exists():
            print(f"  Skipping {path} (file not found)")
            continue

        content = filepath.read_text(encoding="utf-8")

        # Append a "Related Notes" section
        link_lines = []
        for link in links:
            target = link["target"].replace(".md", "")
            reason = link.get("reason", "")
            if reason:
                link_lines.append(f"- [[{target}]] — {reason}")
            else:
                link_lines.append(f"- [[{target}]]")

        section = "\n\n## Related Notes\n" + "\n".join(link_lines)

        if "## Related Notes" in content:
            # Append to existing section
            content = content.rstrip() + "\n" + "\n".join(link_lines)
        else:
            content = content.rstrip() + section

        filepath.write_text(content, encoding="utf-8")
        state.mark_links_applied(path)
        print(f"  Added {len(links)} link(s) to {path}")

    state.save()
    print("\nDone!")


if __name__ == "__main__":
    cli()
