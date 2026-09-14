#!/usr/bin/env python3
"""Merge compatible derived prefetch JSONL datasets without recomputing traces."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "schema_version",
    "source_dataset",
    "prompt_id",
    "sequence_id",
    "routing_history",
    "router_logits",
    "hidden_raw_ref",
    "target_experts_next_token",
}


def validate_record(record: dict[str, Any], path: Path, line_number: int) -> None:
    """Validate the fields required to merge a BUILD_2-compatible record.

    Args:
        record: Parsed JSONL record.
        path: Source JSONL path.
        line_number: One-based source line number.

    Raises:
        ValueError: If the record lacks a required field or has an invalid ID.
    """
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}.")
    if not isinstance(record["prompt_id"], int):
        raise ValueError(f"{path}:{line_number}: prompt_id must be an integer.")
    if not record["source_dataset"]:
        raise ValueError(f"{path}:{line_number}: source_dataset must not be empty.")


def merge_datasets(input_paths: list[Path], output_path: Path) -> tuple[int, int, Counter[str]]:
    """Stream-merge derived datasets while rejecting incompatible provenance.

    Args:
        input_paths: Ordered JSONL inputs. Their prompt ID sets must be disjoint.
        output_path: New JSONL destination. It must not exist already.

    Returns:
        Number of written examples, distinct prompt IDs and examples per source.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If input schemas differ, IDs overlap or a prompt has mixed provenance.
    """
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    all_prompt_ids: set[int] = set()
    prompt_sources: dict[int, str] = {}
    schema_versions: set[str] = set()
    examples = 0
    examples_by_source: Counter[str] = Counter()

    try:
        with temporary_path.open("w", encoding="utf-8") as destination:
            for input_path in input_paths:
                input_prompt_ids: set[int] = set()
                with input_path.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        record = json.loads(line)
                        validate_record(record, input_path, line_number)

                        prompt_id = record["prompt_id"]
                        source_dataset = record["source_dataset"]
                        previous_source = prompt_sources.get(prompt_id)
                        if previous_source is not None and previous_source != source_dataset:
                            raise ValueError(
                                f"{input_path}:{line_number}: prompt {prompt_id} has mixed provenance."
                            )

                        prompt_sources[prompt_id] = source_dataset
                        input_prompt_ids.add(prompt_id)
                        schema_versions.add(record["schema_version"])
                        examples_by_source[source_dataset] += 1
                        json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
                        destination.write("\n")
                        examples += 1

                duplicates = all_prompt_ids & input_prompt_ids
                if duplicates:
                    raise ValueError(
                        f"Prompt IDs occur in more than one input: {sorted(duplicates)}."
                    )
                all_prompt_ids.update(input_prompt_ids)

        if len(schema_versions) != 1:
            raise ValueError(f"Inputs use incompatible schema versions: {sorted(schema_versions)}.")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return examples, len(all_prompt_ids), examples_by_source


def main() -> None:
    """Parse arguments, merge inputs and print a concise summary.

    Returns:
        ``None``. The function writes one new JSONL file.
    """
    parser = argparse.ArgumentParser(description="Merge derived prefetch JSONL datasets.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples, prompts, by_source = merge_datasets(args.inputs, args.output)
    sources = ", ".join(f"{name}={count}" for name, count in sorted(by_source.items()))
    print(f"Merged dataset complete: {examples} examples from {prompts} prompts ({sources}) -> {args.output}")


if __name__ == "__main__":
    main()
