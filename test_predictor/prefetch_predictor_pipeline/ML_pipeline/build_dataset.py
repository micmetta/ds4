#!/usr/bin/env python3
"""Build the supervised BUILD_1 JSONL dataset from DS4 routing traces."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = "prefetch_dataset_v1"
EMBEDDING_TABLE_ID = "deepseek_v4_flash_token_embedding_gguf"


def load_prompts(path: Path) -> dict[int, dict]:
    prompts = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            prompt_id = record["id"]
            if prompt_id in prompts:
                raise ValueError(f"duplicate prompt id {prompt_id} at line {line_number}")
            prompts[prompt_id] = record
    return prompts


def successful_manifest_entries(path: Path) -> list[dict]:
    latest_by_prompt_id = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            if entry.get("status") == "success":
                latest_by_prompt_id[entry["prompt_id"]] = entry
    return [latest_by_prompt_id[prompt_id]
            for prompt_id in sorted(latest_by_prompt_id)]


def load_routing_trace(path: Path) -> dict[int, list[dict]]:
    by_layer = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            event = json.loads(line)
            if event.get("phase") != "decode":
                continue
            if event.get("is_hash"):
                continue
            required = ("token_index", "token_id", "layer", "experts")
            if any(field not in event for field in required):
                raise ValueError(f"{path}:{line_number}: incomplete routing event")
            by_layer[event["layer"]].append(event)

    for layer_events in by_layer.values():
        layer_events.sort(key=lambda event: event["token_index"])
        positions = [event["token_index"] for event in layer_events]
        if len(positions) != len(set(positions)):
            raise ValueError(f"{path}: duplicate token position in one routed layer")
    return by_layer


def make_record(prompt: dict, manifest: dict, layer: int, current: dict,
                history: list[dict], target: dict) -> dict:
    history_positions = [event["token_index"] for event in history]
    history_experts = [event["experts"] for event in history]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": "prefetch_prompts_v0",
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
            "token_positions": history_positions,
            "expert_ids": history_experts,
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


def build_dataset(prompts: dict[int, dict], manifests: list[dict],
                  history_length: int, output_path: Path) -> tuple[int, int]:
    examples = 0
    sequences = 0
    with output_path.open("w", encoding="utf-8") as destination:
        for manifest in manifests:
            prompt_id = manifest["prompt_id"]
            if prompt_id not in prompts:
                raise ValueError(f"manifest references unknown prompt_id {prompt_id}")
            trace_path = Path(manifest["trace_path"])
            if not trace_path.is_file():
                raise FileNotFoundError(f"missing trace for prompt_id {prompt_id}: {trace_path}")

            by_layer = load_routing_trace(trace_path)
            wrote_sequence = False
            for layer, events in sorted(by_layer.items()):
                for index in range(history_length - 1, len(events) - 1):
                    history = events[index - history_length + 1:index + 1]
                    current = events[index]
                    target = events[index + 1]
                    if target["token_index"] != current["token_index"] + 1:
                        raise ValueError(
                            f"{trace_path}: non-consecutive decode positions "
                            f"{current['token_index']} -> {target['token_index']}"
                        )
                    record = make_record(prompts[prompt_id], manifest, layer,
                                         current, history, target)
                    json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
                    destination.write("\n")
                    examples += 1
                    wrote_sequence = True
            if wrote_sequence:
                sequences += 1
    return examples, sequences


def write_embedding_manifest(path: Path, model_path: str) -> None:
    record = {
        "embedding_table_id": EMBEDDING_TABLE_ID,
        "source_model": model_path,
        "source_tensor": "token_embd.weight",
        "dimension": 4096,
        "storage": "external_lookup",
        "note": (
            "BUILD_1 records token_id and this stable lookup reference. "
            "The embedding matrix remains in the source GGUF; it is not "
            "duplicated once per routing event."
        ),
    }
    with path.open("w", encoding="utf-8") as destination:
        json.dump(record, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert BUILD_1 raw routing traces into supervised JSONL.")
    parser.add_argument(
        "--prompts",
        default="test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--raw-dir",
        default="test_predictor/prefetch_predictor_pipeline/build_dataset_1/raw",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=(
            "test_predictor/prefetch_predictor_pipeline/build_dataset_1/derived/"
            "prefetch_dataset_build_1.jsonl"
        ),
        type=Path,
    )
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument(
        "--model",
        default="gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
    )
    args = parser.parse_args()

    if args.history_length < 1:
        parser.error("--history-length must be at least 1")

    manifest_path = args.raw_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    manifests = successful_manifest_entries(manifest_path)
    examples, sequences = build_dataset(prompts, manifests,
                                        args.history_length, args.output)
    write_embedding_manifest(args.output.parent / "embedding_table_manifest.json",
                             args.model)
    print(
        f"BUILD_1 preprocessing complete: {examples} examples from "
        f"{sequences} sequences -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
