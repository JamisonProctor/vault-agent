"""Tag normalization and deduplication utilities.

Merges near-duplicate tags (e.g. product-manager / product-management)
using stemming heuristics and Levenshtein distance.
"""

from __future__ import annotations

import re
from collections import defaultdict


# Common suffix groups that should be treated as equivalent
SUFFIX_GROUPS = [
    # Role vs domain
    ("manager", "management"),
    ("engineer", "engineering"),
    ("designer", "design"),
    ("developer", "development"),
    ("analyst", "analytics", "analysis"),
    ("architect", "architecture"),
    ("scientist", "science"),
    ("consultant", "consulting"),
    ("strategist", "strategy"),
    ("leader", "leadership"),
    ("coordinator", "coordination"),
    ("director", "direction"),
    ("operator", "operations"),
    ("administrator", "administration"),
    # Verb forms
    ("automate", "automation", "automated"),
    ("optimize", "optimization", "optimized"),
    ("integrate", "integration", "integrated"),
    ("collaborate", "collaboration", "collaborative"),
    ("communicate", "communication"),
    ("negotiate", "negotiation"),
    ("innovate", "innovation", "innovative"),
    ("transform", "transformation"),
    ("implement", "implementation"),
    ("migrate", "migration"),
    ("visualize", "visualization"),
]


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _stem_tag(tag: str) -> str:
    """Reduce a tag to a canonical stem for grouping.

    Strips common prefixes (senior-, lead-, junior-, head-of-, vp-)
    and normalizes suffixes using SUFFIX_GROUPS.
    """
    # Strip seniority/role prefixes
    prefixes_to_strip = [
        "senior-", "sr-", "junior-", "jr-", "lead-",
        "head-of-", "head-", "vp-", "chief-", "principal-",
        "staff-", "associate-",
    ]
    stemmed = tag.lower()
    for prefix in prefixes_to_strip:
        if stemmed.startswith(prefix):
            stemmed = stemmed[len(prefix):]
            break

    # Normalize suffixes
    parts = stemmed.split("-")
    normalized_parts = []
    for part in parts:
        replaced = False
        for group in SUFFIX_GROUPS:
            if part in group:
                normalized_parts.append(group[0])  # canonical form is first in group
                replaced = True
                break
        if not replaced:
            normalized_parts.append(part)

    return "-".join(normalized_parts)


def _are_similar(stem_a: str, stem_b: str, distance_threshold: int = 2) -> bool:
    """Check if two stems are similar enough to merge.

    Uses segment-aware comparison to avoid merging tags where only
    a prefix differs (e.g. 'hr-analytics' vs 'ai-analytics').
    """
    dist = _levenshtein(stem_a, stem_b)
    if dist > distance_threshold:
        return False

    # For short stems, require much closer matches
    max_len = max(len(stem_a), len(stem_b))
    if max_len <= 3:
        return dist <= 0  # exact match only for very short tags
    if max_len <= 5:
        return dist <= 1  # at most 1 edit for short tags

    # Segment-aware comparison: if both tags have the same structure
    # (same number of hyphen-separated parts), each part must be close
    parts_a = stem_a.split("-")
    parts_b = stem_b.split("-")
    if len(parts_a) == len(parts_b) and len(parts_a) > 1:
        for pa, pb in zip(parts_a, parts_b):
            seg_len = max(len(pa), len(pb))
            seg_dist = _levenshtein(pa, pb)
            # Each segment must be very close (pluralization, typo)
            if seg_len <= 3 and seg_dist > 0:
                return False  # short segments must match exactly
            if seg_dist > 1:
                return False  # at most 1 edit per segment
        return True

    # For single-segment or different-structure tags, use ratio
    return dist / max_len < 0.2


def find_tag_groups(tags: list[str], distance_threshold: int = 2) -> dict[str, list[str]]:
    """Group tags that are near-duplicates.

    Returns a dict mapping canonical tag -> list of variant tags (including itself).
    Uses two strategies:
    1. Stem-based grouping (strips prefixes, normalizes suffixes)
    2. Levenshtein distance for remaining close matches (length-aware)
    """
    # Phase 1: Group by stem
    stem_groups: dict[str, list[str]] = defaultdict(list)
    for tag in sorted(tags):
        stem = _stem_tag(tag)
        stem_groups[stem].append(tag)

    # Phase 2: Merge groups whose stems are similar (length-aware)
    stems = list(stem_groups.keys())
    merged: dict[str, list[str]] = {}
    consumed: set[str] = set()

    for i, stem_a in enumerate(stems):
        if stem_a in consumed:
            continue
        group = list(stem_groups[stem_a])
        for stem_b in stems[i + 1:]:
            if stem_b in consumed:
                continue
            if _are_similar(stem_a, stem_b, distance_threshold):
                group.extend(stem_groups[stem_b])
                consumed.add(stem_b)
        consumed.add(stem_a)

        # Pick the most common tag as canonical (or shortest if tied)
        canonical = min(group, key=lambda t: (-tags.count(t), len(t)))
        if len(group) > 1:
            merged[canonical] = sorted(group)

    return merged


def build_normalization_map(
    tags_with_counts: dict[str, int],
    distance_threshold: int = 2,
) -> dict[str, str]:
    """Build a mapping from variant tags to their canonical form.

    Args:
        tags_with_counts: dict of tag -> number of notes using this tag
        distance_threshold: max Levenshtein distance to consider tags similar

    Returns:
        dict mapping each variant tag to its canonical form.
        Tags that don't need normalization are not included.
    """
    tags = list(tags_with_counts.keys())
    groups = find_tag_groups(tags, distance_threshold)

    mapping: dict[str, str] = {}
    for canonical, variants in groups.items():
        # Pick the variant with the highest count as canonical
        best = max(variants, key=lambda t: (tags_with_counts.get(t, 0), -len(t)))
        for variant in variants:
            if variant != best:
                mapping[variant] = best

    return mapping


def get_tag_counts(state_data: dict) -> dict[str, int]:
    """Count how many notes use each tag across the state file."""
    counts: dict[str, int] = defaultdict(int)
    for info in state_data.get("notes", {}).values():
        for tag in info.get("proposed_tags", []):
            counts[tag] += 1
    return dict(counts)
