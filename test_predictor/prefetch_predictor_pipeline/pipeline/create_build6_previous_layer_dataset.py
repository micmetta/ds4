#!/usr/bin/env python3
"""Create Build 6 records by attaching the preceding MoE layer of each token.

The input is the final Build 5 derived JSONL with current KV statistics.  It is
read only.  Output records keep the Build 5 current-layer fields and add a
``previous_layer`` object for the same token.  Only target layers 4..42 are
kept, because layer 3 has no compatible preceding MoE router state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


FIRST_TARGET_LAYER = 4
LAST_TARGET_LAYER = 42
REQUIRED_PREVIOUS_FIELDS = (
    "routing_history",
    "router_logits",
    "hidden_raw_ref",
    "kv_raw_ref",
    "kv_features_current",
)


def token_key(record: dict[str, Any]) -> tuple[str, int]:
    """Return the identifier shared by all layer records of one token."""
    try:
        return str(record["sequence_id"]), int(record["token_position"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Record senza sequence_id o token_position valido.") from error


def previous_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Copy exactly the preceding-layer fields used by Build 6."""
    missing = [field for field in REQUIRED_PREVIOUS_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record layer precedente incompleto: {missing}")
    return {"layer": int(record["layer"]), **{field: record[field] for field in REQUIRED_PREVIOUS_FIELDS}}


def write_record(destination, current: dict[str, Any], previous: dict[str, Any]) -> None:
    """Write one Build 6 record without the obsolete token-embedding metadata."""
    output = dict(current)
    output["schema_version"] = "prefetch_dataset_build6_previous_layer_v1"
    output.pop("token_embedding", None)
    output["previous_layer"] = previous_payload(previous)
    json.dump(output, destination, ensure_ascii=False, separators=(",", ":"))
    destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea il derived dataset Build 6 con feature del layer MoE precedente."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL Build 5 finale, letto senza modificarlo.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input Build 5 non trovato: {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Output Build 6 gia' esistente: {args.output}")
    report = args.report or args.output.with_name(f"{args.output.stem}_summary.json")
    if report.exists():
        raise FileExistsError(f"Report gia' esistente: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"File temporaneo gia' esistente: {temporary}")

    # Build 5 is ordered primarily by layer (all token records of layer 3,
    # then all token records of layer 4, ...), not by token.  Keep only the
    # immediately preceding layer keyed by token instead of accumulating every
    # layer of every token in memory.
    previous_by_token: dict[tuple[str, int], dict[str, Any]] = {}
    last_layer: int | None = None
    source_records = 0
    written_records = 0
    skipped_layer_three = 0

    print("1) Collegamento causale dei record Build 5 al layer precedente...")
    try:
        with args.input.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = token_key(record)
                layer = int(record.get("layer", -1))
                if not FIRST_TARGET_LAYER - 1 <= layer <= LAST_TARGET_LAYER:
                    raise ValueError(f"Linea {line_number}: layer MoE inatteso: {layer}.")
                # A decrease 42 -> 3 starts another recorded sequence block.
                # Its keys are independent, but clearing keeps memory bounded
                # and prevents any accidental cross-block association.
                if last_layer is not None and layer < last_layer:
                    previous_by_token.clear()
                last_layer = layer
                if layer == FIRST_TARGET_LAYER - 1:
                    skipped_layer_three += 1
                else:
                    previous = previous_by_token.get(key)
                    if previous is None or int(previous["layer"]) != layer - 1:
                        raise ValueError(
                            f"Linea {line_number}: manca layer {layer - 1} per token {key}."
                        )
                    write_record(destination, record, previous)
                    written_records += 1
                previous_by_token[key] = record
                source_records += 1
        if written_records == 0:
            raise ValueError("Nessun record Build 6 scritto.")
        os.replace(temporary, args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report.write_text(
        json.dumps(
            {
                "experiment": "build_dataset_6_previous_layer_features",
                "source_build5_jsonl": str(args.input),
                "output": str(args.output),
                "source_records": source_records,
                "output_records": written_records,
                "target_layers": [FIRST_TARGET_LAYER, LAST_TARGET_LAYER],
                "excluded_target_layer": 3,
                "previous_layer_fields": list(REQUIRED_PREVIOUS_FIELDS),
                "token_embedding_metadata_removed": True,
                "source_build5_modified": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Build 6 derived dataset completato: {written_records:,} esempi.")
    print(f"Output: {args.output}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
