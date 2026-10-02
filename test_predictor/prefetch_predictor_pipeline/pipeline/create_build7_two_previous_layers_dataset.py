#!/usr/bin/env python3
"""Create Build 7 records with the two preceding routed MoE layers.

Build 7 keeps the causal current-layer fields of Build 5 and attaches the
states of ``l - 1`` and ``l - 2`` for the same decoded token.  It deliberately
starts at target layer 5: for target layer 4, ``l - 2`` would be the
deterministic layer 2 and would not have router state comparable to a MoE
layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


FIRST_TARGET_LAYER = 5
LAST_TARGET_LAYER = 42
REQUIRED_CONTEXT_FIELDS = (
    "routing_history",
    "router_logits",
    "hidden_raw_ref",
    "kv_raw_ref",
    "kv_features_current",
)


def token_key(record: dict[str, Any]) -> tuple[str, int]:
    """Return the key shared by the routed layers of one decoded token."""
    try:
        return str(record["sequence_id"]), int(record["token_position"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Record senza sequence_id o token_position valido.") from error


def context_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Copy the state and raw references of one preceding MoE layer."""
    missing = [field for field in REQUIRED_CONTEXT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record del layer precedente incompleto: {missing}")
    return {"layer": int(record["layer"]), **{field: record[field] for field in REQUIRED_CONTEXT_FIELDS}}


def write_record(destination, current: dict[str, Any], previous: dict[str, Any], previous_previous: dict[str, Any]) -> None:
    """Write one Build 7 record without obsolete token-embedding metadata."""
    output = dict(current)
    output["schema_version"] = "prefetch_dataset_build7_two_previous_layers_v1"
    output.pop("token_embedding", None)
    output["previous_layer"] = context_payload(previous)
    output["previous_previous_layer"] = context_payload(previous_previous)
    json.dump(output, destination, ensure_ascii=False, separators=(",", ":"))
    destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea il derived dataset Build 7 con i due layer MoE precedenti."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL Build 5 finale, letto senza modificarlo.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input Build 5 non trovato: {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Output Build 7 gia' esistente: {args.output}")
    report = args.report or args.output.with_name(f"{args.output.stem}_summary.json")
    if report.exists():
        raise FileExistsError(f"Report gia' esistente: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"File temporaneo gia' esistente: {temporary}")

    # The Build 5 JSONL is ordered by layer.  Keep the latest two layers for
    # every token; clearing at a layer decrease bounds memory and prevents
    # accidental links across independent sequence blocks.
    recent_layers_by_token: dict[tuple[str, int], tuple[dict[str, Any] | None, dict[str, Any]]] = {}
    last_layer: int | None = None
    source_records = 0
    written_records = 0

    print("1) Collegamento causale dei record Build 5 ai layer l-1 e l-2...")
    try:
        with args.input.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = token_key(record)
                layer = int(record.get("layer", -1))
                if not FIRST_TARGET_LAYER - 2 <= layer <= LAST_TARGET_LAYER:
                    raise ValueError(f"Linea {line_number}: layer MoE inatteso: {layer}.")
                if last_layer is not None and layer < last_layer:
                    recent_layers_by_token.clear()
                last_layer = layer

                older, previous = recent_layers_by_token.get(key, (None, None))
                if layer >= FIRST_TARGET_LAYER:
                    if previous is None or int(previous["layer"]) != layer - 1:
                        raise ValueError(f"Linea {line_number}: manca layer {layer - 1} per token {key}.")
                    if older is None or int(older["layer"]) != layer - 2:
                        raise ValueError(f"Linea {line_number}: manca layer {layer - 2} per token {key}.")
                    write_record(destination, record, previous, older)
                    written_records += 1

                recent_layers_by_token[key] = (previous, record)
                source_records += 1
        if written_records == 0:
            raise ValueError("Nessun record Build 7 scritto.")
        os.replace(temporary, args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report.write_text(
        json.dumps(
            {
                "experiment": "build_dataset_7_two_previous_layer_features",
                "source_build5_jsonl": str(args.input),
                "output": str(args.output),
                "source_records": source_records,
                "output_records": written_records,
                "target_layers": [FIRST_TARGET_LAYER, LAST_TARGET_LAYER],
                "excluded_target_layers": [3, 4],
                "previous_layer_fields": list(REQUIRED_CONTEXT_FIELDS),
                "previous_previous_layer_fields": list(REQUIRED_CONTEXT_FIELDS),
                "token_embedding_metadata_removed": True,
                "source_build5_modified": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Build 7 derived dataset completato: {written_records:,} esempi.")
    print(f"Output: {args.output}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
