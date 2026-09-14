#!/usr/bin/env python3
"""Build a supervised prefetch JSONL dataset from generic collector traces."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "prefetch_dataset_v1"
EMBEDDING_TABLE_ID = "deepseek_v4_flash_token_embedding_gguf"


def load_prompts(path: Path) -> dict[int, dict[str, Any]]:
    """
    Load the V0 prompt corpus and index records by prompt identifier.

    Args:
        path: JSONL file containing ``id``, ``category`` and ``prompt`` fields.

    Returns:
        Mapping from prompt ID to its original JSON object.

    Raises:
        ValueError: If a prompt identifier appears more than once.
    """
    prompts: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            prompt_id = int(record["id"])
            if prompt_id in prompts:
                raise ValueError(f"Duplicate prompt ID {prompt_id} at line {line_number}.")
            prompts[prompt_id] = record
    return prompts


def load_successful_manifests(path: Path) -> list[dict[str, Any]]:
    """
    Load the latest successful collection entry for every prompt.

    Args:
        path: Collector ``manifest.jsonl`` file.

    Returns:
        Successful manifest entries ordered by prompt ID.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    latest: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            if entry.get("status") == "success":
                latest[int(entry["prompt_id"])] = entry
    return [latest[prompt_id] for prompt_id in sorted(latest)]


def load_decode_events(
    path: Path, require_hidden: bool, require_kv: bool
) -> dict[int, list[dict[str, Any]]]:
    """
    Read routed decode events and group them by layer.

    Args:
        path: Per-prompt routing JSONL trace written by DS4.
        require_hidden: Require a valid ``hidden_raw_ref`` for each routed event.
        require_kv: Require a valid ``kv_raw_ref`` for each routed event.

    Returns:
        Mapping ``layer -> events`` ordered by token position.

    Raises:
        ValueError: If an event is incomplete, duplicated or lacks a requested
            hidden-state reference.
    """
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            event = json.loads(line)
            if event.get("phase") != "decode" or event.get("is_hash"):
                continue
            required = ("token_index", "token_id", "layer", "experts")
            if any(field not in event for field in required):
                raise ValueError(f"{path}:{line_number}: incomplete routing event.")
            if require_hidden and not event.get("hidden_raw_ref"):
                raise ValueError(f"{path}:{line_number}: hidden_raw_ref is required.")
            if require_kv and not event.get("kv_raw_ref"):
                raise ValueError(f"{path}:{line_number}: kv_raw_ref is required.")
            by_layer[int(event["layer"])].append(event)

    for layer_events in by_layer.values():
        layer_events.sort(key=lambda event: int(event["token_index"]))
        positions = [int(event["token_index"]) for event in layer_events]
        if len(positions) != len(set(positions)):
            raise ValueError(f"{path}: duplicate token position in one routed layer.")
    return by_layer


def make_record(
    prompt: dict[str, Any],
    manifest: dict[str, Any],
    source_dataset: str,
    layer: int,
    current: dict[str, Any],
    history: list[dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    """
    Create one supervised record for ``E_l(t) -> E_l(t+1)``.

    Args:
        prompt: Original V0 prompt metadata.
        manifest: Successful collector manifest entry for the prompt.
        source_dataset: Identifier of the prompt corpus that produced the record.
        layer: Routed layer of the current event.
        current: Routing event at token position ``t``.
        history: Consecutive routing events ending at ``t``.
        target: Routing event at token position ``t+1``.

    Returns:
        JSON-serializable dataset record. If collected, ``hidden_raw_ref`` is
        preserved as an intermediate reference for the later PCA step.
    """
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": str(prompt.get("source_dataset", source_dataset)),
        "prompt_id": prompt["id"],
        "prompt_category": prompt["category"],
        "sequence_id": manifest["sequence_id"],
        "token_position": current["token_index"],
        "token_id": current["token_id"],
        "layer": layer,
        "is_hash": False,
        "experts_t": current["experts"],
        "routing_history": {
            "history_length": len(history),
            "token_positions": [event["token_index"] for event in history],
            "expert_ids": [event["experts"] for event in history],
        },
        "router_logits": current.get("router_logits"),
        "token_embedding": {
            "kind": "model_token_embedding",
            "storage": "external_lookup",
            "embedding_table_id": EMBEDDING_TABLE_ID,
            "source_tensor": "token_embd.weight",
            "dimension": 4096,
            "token_id_source": "token_id",
        },
        "target_experts_next_token": target["experts"],
    }
    if current.get("hidden_raw_ref"):
        record["hidden_raw_ref"] = current["hidden_raw_ref"]
    if current.get("kv_raw_ref"):
        record["kv_raw_ref"] = current["kv_raw_ref"]
    return record


def build_dataset(
    prompts: dict[int, dict[str, Any]],
    manifests: list[dict[str, Any]],
    history_length: int,
    output_path: Path,
    require_hidden: bool,
    require_kv: bool,
    source_dataset: str,
) -> tuple[int, int]:
    """
    Convert collector traces into a supervised JSONL dataset.

    Args:
        prompts: V0 prompts indexed by ID.
        manifests: Successful collection entries.
        history_length: Number of observed routing steps included in history.
        output_path: JSONL destination.
        require_hidden: Require feature-4 raw references in all output records.
        source_dataset: Identifier written in every output record.

    Returns:
        Tuple ``(examples, sequences)`` written to the destination.
    """
    examples = 0
    sequences = 0
    with output_path.open("w", encoding="utf-8") as destination:
        for manifest in manifests:
            prompt_id = int(manifest["prompt_id"])
            if prompt_id not in prompts:
                raise ValueError(f"Manifest references unknown prompt ID {prompt_id}.")
            by_layer = load_decode_events(
                Path(manifest["trace_path"]), require_hidden, require_kv
            )
            wrote_sequence = False
            for layer, events in sorted(by_layer.items()):
                for index in range(history_length - 1, len(events) - 1):
                    current = events[index]
                    target = events[index + 1]
                    if int(target["token_index"]) != int(current["token_index"]) + 1:
                        raise ValueError("Decode positions must be consecutive.")
                    history = events[index - history_length + 1:index + 1]
                    json.dump(
                        make_record(
                            prompts[prompt_id], manifest, source_dataset,
                            layer, current, history, target,
                        ),
                        destination,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    destination.write("\n")
                    examples += 1
                    wrote_sequence = True
            if wrote_sequence:
                sequences += 1
    return examples, sequences


def main() -> None:
    """
    Parse command-line options and build the derived dataset.

    Args:
        No Python arguments. Paths and options are read from the command line.

    Returns:
        ``None``. The function writes one derived JSONL file and prints its size.
    """
    parser = argparse.ArgumentParser(description="Build a derived prefetch JSONL dataset.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument("--require-hidden", action="store_true")
    parser.add_argument("--require-kv", action="store_true")
    parser.add_argument("--source-dataset", default="prefetch_prompts_v0")
    args = parser.parse_args()
    if args.history_length < 1:
        parser.error("--history-length must be at least 1")

    manifests = load_successful_manifests(args.raw_dir / "manifest.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples, sequences = build_dataset(
        load_prompts(args.prompts), manifests, args.history_length, args.output,
        args.require_hidden, args.require_kv, args.source_dataset,
    )
    print(f"Derived dataset complete: {examples} examples from {sequences} sequences -> {args.output}")


if __name__ == "__main__":
    main()
