"""
Stage 3: Taxonomy Mapper
-------------------------
Reads every raw/<dataset>/<raw_class>/ folder, matches the raw_class name
against taxonomy.json aliases, and copies images into
clean/<unified_class>/ using the unified name.

Matching strategy (in order of confidence):
  1. Exact normalized alias match (e.g. "lumpy_skin" -> skin/lumpy_skin_disease)
  2. Keyword scoring: the raw class name is tokenized into words, and every
     candidate class is scored by how many of its alias words/phrases appear.
     The highest-scoring class wins. Ties are flagged as ambiguous and sent to
     clean/_UNMAPPED/ for manual review.

Anything that doesn't match any alias goes into clean/_UNMAPPED/<raw_name>/
so you can review it and add a new alias to taxonomy.json rather than
silently losing data.

The clean/ folder is wiped at the start of every run so results are always
reproducible from scratch.

Run this AFTER extract.py and BEFORE dedup.py.
"""

import json
import re
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
CLEAN_DIR = Path(__file__).parent / "clean"
TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(name: str) -> str:
    """Lowercase, strip punctuation/underscores/spaces for exact matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def tokenize(name: str) -> list:
    """Split a name into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", name.lower())


def build_alias_lookup(taxonomy: dict) -> dict:
    """Flattens taxonomy.json into {normalized_alias: 'group/unified_class'}"""
    lookup = {}
    for group, classes in taxonomy.items():
        if group.startswith("_") or group == "excluded_not_photo_diagnosable":
            continue
        if not isinstance(classes, dict):
            continue
        for unified_class, aliases in classes.items():
            for alias in aliases:
                lookup[normalize(alias)] = f"{group}/{unified_class}"
    return lookup


def build_class_index(taxonomy: dict) -> dict:
    """
    Builds {class_key: {'phrases': set of tuples, 'tokens': set}} for scoring.
    Multi-word aliases become phrase tuples; every alias contributes its words
    to the token set.
    """
    index = {}
    for group, classes in taxonomy.items():
        if group.startswith("_") or group == "excluded_not_photo_diagnosable":
            continue
        if not isinstance(classes, dict):
            continue
        for unified_class, aliases in classes.items():
            key = f"{group}/{unified_class}"
            phrases = set()
            tokens = set()
            for alias in aliases:
                words = tokenize(alias)
                if len(words) > 1:
                    phrases.add(tuple(words))
                tokens.update(words)
            index[key] = {"phrases": phrases, "tokens": tokens}
    return index


def phrase_matches(raw_tokens: list, phrase: tuple) -> bool:
    """True if the phrase's words appear consecutively in raw_tokens."""
    n = len(phrase)
    for i in range(len(raw_tokens) - n + 1):
        if tuple(raw_tokens[i:i + n]) == phrase:
            return True
    return False


def score_class(raw_tokens: list, class_info: dict) -> int:
    """
    Scores a candidate class against the raw tokens.
    Phrase matches are weighted higher than single-token matches.
    """
    score = 0
    for phrase in class_info["phrases"]:
        if phrase_matches(raw_tokens, phrase):
            score += len(phrase) * 2
    for tok in raw_tokens:
        if tok in class_info["tokens"]:
            score += 1
    return score


def best_class(raw_name: str, index: dict):
    """
    Returns (class_key, score, ambiguous) for the raw class name.
    ambiguous is True when two or more classes tie for the top score.
    """
    raw_tokens = tokenize(raw_name)
    scores = {key: score_class(raw_tokens, info) for key, info in index.items()}
    best_score = max(scores.values()) if scores else 0
    if best_score == 0:
        return None, 0, False
    tied = [k for k, s in scores.items() if s == best_score]
    return tied[0], best_score, len(tied) > 1


def main():
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    lookup = build_alias_lookup(taxonomy)
    index = build_class_index(taxonomy)
    excluded = {normalize(x) for x in taxonomy.get("excluded_not_photo_diagnosable", [])}

    # Start fresh every run so stale files never linger after taxonomy changes
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    unmapped_log = []
    ambiguous_log = []
    excluded_log = []
    mapped_count = 0

    for dataset_dir in RAW_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir():
                continue

            norm = normalize(class_dir.name)

            if norm in excluded:
                excluded_log.append((dataset_dir.name, class_dir.name))
                continue

            # 1) Exact normalized alias match — highest confidence
            unified = lookup.get(norm)

            # 2) Keyword scoring fallback
            if unified is None:
                unified, score, ambiguous = best_class(class_dir.name, index)
                if unified is not None and ambiguous:
                    ambiguous_log.append((dataset_dir.name, class_dir.name, score))
                    unified = None  # send to _UNMAPPED for manual review

            if unified is None:
                unmapped_log.append((dataset_dir.name, class_dir.name))
                dest = CLEAN_DIR / "_UNMAPPED" / class_dir.name
            else:
                dest = CLEAN_DIR / unified

            dest.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    # prefix filename with source dataset to avoid collisions
                    new_name = f"{dataset_dir.name}__{img.name}"
                    shutil.copy2(img, dest / new_name)
                    mapped_count += 1

    print(f"Mapped {mapped_count} images into clean/")
    if excluded_log:
        print(f"\nExcluded (not photo-diagnosable) — {len(excluded_log)} class folders skipped:")
        for ds, cls in excluded_log:
            print(f"  {ds} / {cls}")
    if ambiguous_log:
        print(f"\n!! {len(ambiguous_log)} class folders were ambiguous (tied scores) — check clean/_UNMAPPED/")
        print("   Add a more specific alias to taxonomy.json and re-run this script.")
        for ds, cls, score in ambiguous_log:
            print(f"  {ds} / {cls} (score {score})")
    if unmapped_log:
        print(f"\n!! {len(unmapped_log)} class folders had no taxonomy match — check clean/_UNMAPPED/")
        print("   Add missing aliases to taxonomy.json and re-run this script.")
        for ds, cls in unmapped_log:
            print(f"  {ds} / {cls}")


if __name__ == "__main__":
    main()