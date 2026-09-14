#!/usr/bin/env python3
"""Create one reproducible prompt corpus from existing JSONL corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_input(value: str) -> tuple[str, Path]:
    """Parse one ``source_dataset=path`` input specification."""
    source_dataset, separator, raw_path = value.partition("=")
    if not separator or not source_dataset or not raw_path:
        raise argparse.ArgumentTypeError("Usa il formato source_dataset=percorso.jsonl.")
    return source_dataset, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unisce corpus di prompt JSONL senza modificare gli input.")
    parser.add_argument(
        "--input", action="append", required=True, type=parse_input,
        help="Corpus nel formato source_dataset=percorso.jsonl; ripetere per ogni input.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"L'output esiste gia': {args.output}")

    records: list[dict[str, object]] = []
    prompt_ids: set[int] = set()
    source_counts: dict[str, int] = {}
    for source_dataset, path in args.input:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus non trovato: {path}")
        count = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                if not all(field in record for field in ("id", "category", "prompt")):
                    raise ValueError(f"{path}:{line_number}: mancano id, category o prompt.")
                prompt_id = int(record["id"])
                if prompt_id in prompt_ids:
                    raise ValueError(f"Prompt ID duplicato: {prompt_id}")
                prompt_ids.add(prompt_id)
                record["id"] = prompt_id
                record["source_dataset"] = source_dataset
                records.append(record)
                count += 1
        source_counts[source_dataset] = count

    records.sort(key=lambda record: int(record["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")
    print(f"Combined prompt corpus complete: {len(records)} prompts -> {args.output}")
    print("Sources: " + ", ".join(f"{name}={count}" for name, count in source_counts.items()))


if __name__ == "__main__":
    main()
