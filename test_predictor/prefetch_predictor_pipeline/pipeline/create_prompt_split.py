#!/usr/bin/env python3
"""Create a stratified prompt split while preserving an existing assignment."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")


def load_prompts(paths: list[Path]) -> dict[int, dict[str, Any]]:
    """Load prompt corpora and reject duplicate prompt identifiers.

    Args:
        paths: JSONL corpora containing ``id`` and ``category`` fields.

    Returns:
        Prompt records indexed by ID.

    Raises:
        ValueError: If an ID is duplicated or a required field is missing.
    """
    prompts: dict[int, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                if "id" not in record or "category" not in record:
                    raise ValueError(f"{path}:{line_number}: id and category are required.")
                prompt_id = int(record["id"])
                if prompt_id in prompts:
                    raise ValueError(f"Duplicate prompt ID {prompt_id} in {path}:{line_number}.")
                prompts[prompt_id] = record
    return prompts


def load_existing_split(path: Path) -> dict[int, str]:
    """Load an optional existing prompt-level split.

    Args:
        path: JSON file containing a ``split_per_prompt`` mapping.

    Returns:
        Existing mapping with integer prompt IDs.

    Raises:
        ValueError: If split labels are invalid.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {int(key): value for key, value in payload["split_per_prompt"].items()}
    invalid = {value for value in mapping.values() if value not in SPLITS}
    if invalid:
        raise ValueError(f"Invalid split labels: {sorted(invalid)}.")
    return mapping


def target_counts(prompt_count: int) -> dict[str, int]:
    """Return the historical 80/10/10 integer allocation for one category.

    Args:
        prompt_count: Number of prompts in a category.

    Returns:
        Target counts for train, validation and test.
    """
    train = int(prompt_count * 0.8)
    validation = int(prompt_count * 0.1)
    return {"train": train, "validation": validation, "test": prompt_count - train - validation}


def create_split(
    prompts: dict[int, dict[str, Any]], existing: dict[int, str], seed: int
) -> dict[int, str]:
    """Preserve existing assignments and stratify the remaining prompts.

    Args:
        prompts: All prompt records indexed by ID.
        existing: Assignments to preserve exactly.
        seed: Pseudo-random seed for new assignments.

    Returns:
        Complete mapping from prompt ID to split.

    Raises:
        ValueError: If preserved IDs are unknown or exceed category targets.
    """
    unknown = set(existing) - set(prompts)
    if unknown:
        raise ValueError(f"Existing split references unknown prompt IDs: {sorted(unknown)}.")

    by_category: dict[str, list[int]] = defaultdict(list)
    for prompt_id, record in prompts.items():
        by_category[str(record["category"])].append(prompt_id)

    generator = random.Random(seed)
    complete = dict(existing)
    for category, prompt_ids in sorted(by_category.items()):
        prompt_ids.sort()
        targets = target_counts(len(prompt_ids))
        assigned = Counter(complete[prompt_id] for prompt_id in prompt_ids if prompt_id in complete)
        if any(assigned[split] > targets[split] for split in SPLITS):
            raise ValueError(f"Preserved assignments exceed targets in category {category}.")

        pending = [prompt_id for prompt_id in prompt_ids if prompt_id not in complete]
        generator.shuffle(pending)
        cursor = 0
        for split in SPLITS:
            needed = targets[split] - assigned[split]
            for prompt_id in pending[cursor:cursor + needed]:
                complete[prompt_id] = split
            cursor += needed

    return complete


def main() -> None:
    """Parse arguments, create the split and write one JSON artifact.

    Returns:
        ``None``. The function writes the requested split JSON file.
    """
    parser = argparse.ArgumentParser(description="Create a stratified prompt-level split.")
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preserve-split", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    prompts = load_prompts(args.prompts)
    existing = load_existing_split(args.preserve_split) if args.preserve_split else {}
    split_per_prompt = create_split(prompts, existing, args.seed)
    counts = Counter(split_per_prompt.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "strategy": "stratified_by_category_preserving_existing_assignments",
        "preserved_split": str(args.preserve_split) if args.preserve_split else None,
        "split_per_prompt": {str(prompt_id): split_per_prompt[prompt_id] for prompt_id in sorted(split_per_prompt)},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Prompt split complete: {len(split_per_prompt)} prompts "
        f"(train={counts['train']}, validation={counts['validation']}, test={counts['test']}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
