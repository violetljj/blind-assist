"""Source-only instance/reference shell gate; never opens model predictions.

Passing this bounded probe does not admit the other Final Reckoning strata.
Only raw probe shard records and their sensor payloads are read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import dtr_carla_c2_rich_scene as c2
import join_dtr_carla_c2_rich_scene as join

SCHEMA = "blindassist-dtr-final-visual-shell-probe-v1"
PASS = "SOURCE_SHELL_GATE_MET_PENDING_FULL_ROSTER_GATES"
FAIL = "SOURCE_SHELL_GATE_NOT_EVALUABLE"
POSITION_TOLERANCE = 1e-4
ANGLE_TOLERANCE = 1e-3
TIME_TOLERANCE = 1e-8
SAMPLE_SECONDS = 0.1
MIN_PARTIAL_RATIO = 0.05
MAX_PARTIAL_RATIO = 0.45
MIN_WINDOW_FRAMES = 6
MIN_VISIBLE_CONTEXT = 8


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def close(first: Any, second: Any, path: str = "") -> bool:
    """Compare source geometry, preserving exact categories and Boolean truth."""
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            close(value, second[key], f"{path}.{key}") for key, value in first.items()
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(close(a, b, path) for a, b in zip(first, second))
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        tolerance = TIME_TOLERANCE if path.endswith("time_s") else POSITION_TOLERANCE
        if path.rsplit(".", 1)[-1] in {"yaw", "pitch", "roll"}:
            return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0) <= ANGLE_TOLERANCE
        return math.isfinite(first) and math.isfinite(second) and abs(first - second) <= tolerance
    return first == second


def check_rows(rows: Sequence[Mapping[str, Any]], episode_id: str) -> None:
    require(bool(rows), f"empty_episode:{episode_id}")
    for index, row in enumerate(rows):
        require(row["episode_id"] == episode_id and row["sample_index"] == index,
                f"frame_identity:{episode_id}:{index}")
        require(abs(float(row["time_s"]) - index * SAMPLE_SECONDS) <= TIME_TOLERANCE,
                f"frame_time:{episode_id}:{index}")


def check_outcome(rows: Sequence[Mapping[str, Any]], scenario: Mapping[str, Any]) -> None:
    episode_id = str(scenario["episode_id"])
    check_rows(rows, episode_id)
    contacts = [row for row in rows if row["truth"]["current_contact"] is True]
    outcome = "CONTACT" if contacts else "SAFE"
    responsible = sorted({key for row in contacts for key in row["truth"]["responsible_assets"]})
    require(outcome == scenario["expected_outcome"], f"outcome:{episode_id}")
    require(responsible == sorted(scenario["expected_responsible_assets"]), f"responsible:{episode_id}")
    for row in rows:
        truth = row["truth"]
        require(type(truth["current_contact"]) is bool, f"contact_type:{episode_id}")
        require(truth["expected_outcome"] == scenario["expected_outcome"] and
                sorted(truth["expected_responsible_assets"]) == sorted(scenario["expected_responsible_assets"]),
                f"expected_truth_binding:{episode_id}")
        require(bool(truth["responsible_assets"]) == truth["current_contact"],
                f"responsible_contact_consistency:{episode_id}")


def runs(indices: Sequence[int]) -> list[list[int]]:
    output: list[list[int]] = []
    for index in indices:
        if not output or output[-1][-1] + 1 != index:
            output.append([])
        output[-1].append(index)
    return output


def evaluate_pair(pair: Mapping[str, Any], primary: Sequence[Mapping[str, Any]],
                  reference: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label = str(pair["episode_id"])
    check_rows(primary, label)
    check_rows(reference, str(pair["reference_episode_id"]))
    require(len(primary) == len(reference), f"twin_count:{label}")
    target = str(pair["target_asset"])
    counts: list[int] = []
    refs: list[int] = []
    for left, right in zip(primary, reference):
        require(close(left["time_s"], right["time_s"], "time_s"), f"twin_time:{label}")
        for key in ("wearer_transform", "camera_transform"):
            require(close(left[key], right[key]), f"twin_{key}:{label}:{left['sample_index']}")
        for key in ("transform", "bounding_box", "local_position", "command_velocity"):
            require(close(left["actors"][target][key], right["actors"][target][key]),
                    f"twin_target_{key}:{label}:{left['sample_index']}")
        for row, destination in ((left, counts), (right, refs)):
            pixels = row["instance_visibility"][target]["pixels"]
            require(type(pixels) is int and pixels >= 0, f"pixel_count:{label}")
            destination.append(pixels)
    start, end = map(float, pair["window_s"])
    window = [i for i, row in enumerate(primary) if start - TIME_TOLERANCE <= float(row["time_s"]) <= end + TIME_TOLERANCE]
    require(len(window) >= MIN_WINDOW_FRAMES, f"window_coverage:{label}")
    require(all(refs[i] > 0 for i in window), f"reference_invisible:{label}")
    ratios = [counts[i] / refs[i] for i in window]
    if pair["role"] == "PARTIAL":
        require(all(counts[i] > 0 for i in window), f"partial_zero_pixels:{label}")
        qualifying = runs([i for i in window if MIN_PARTIAL_RATIO <= counts[i] / refs[i] <= MAX_PARTIAL_RATIO])
        require(any(len(run) >= MIN_WINDOW_FRAMES for run in qualifying), f"partial_ratio_run:{label}")
    else:
        require(pair["role"] == "FULL", f"unknown_pair_role:{label}")
        qualifying = runs([i for i in window if counts[i] == 0])
        valid = []
        for run in qualifying:
            if len(run) < MIN_WINDOW_FRAMES:
                continue
            # Context is contiguous and immediately brackets the complete disappearance.
            first_zero = run[0]
            while first_zero > 0 and counts[first_zero - 1] == 0:
                first_zero -= 1
            if first_zero < MIN_VISIBLE_CONTEXT:
                continue
            before = list(range(first_zero - MIN_VISIBLE_CONTEXT, first_zero))
            last_zero = run[-1]
            while last_zero + 1 < len(counts) and counts[last_zero + 1] == 0:
                last_zero += 1
            after = list(range(last_zero + 1, last_zero + 1 + MIN_VISIBLE_CONTEXT))
            if after[-1] < len(counts) and all(counts[i] > 0 and refs[i] > 0 for i in before + after):
                valid.append(run)
        require(bool(valid), f"full_disappearance_context:{label}")
    return {"episode_id": label, "role": pair["role"], "window_frames": len(window),
            "minimum_ratio": min(ratios), "maximum_ratio": max(ratios),
            "maximum_qualifying_run": max(map(len, qualifying), default=0)}


def load_shard(protocol: Mapping[str, Any], protocol_path: Path, root: Path,
               sensor: str) -> dict[str, list[dict[str, Any]]]:
    shard = root / "shards" / sensor
    result = join.read_json(shard / "result.json")
    require(result["status"] == "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE", f"shard_incomplete:{sensor}")
    require(result["sensor"] == sensor and result["protocol_sha256"] == c2.sha256_file(protocol_path), f"shard_binding:{sensor}")
    require(bool(result["checks"]) and all(value is True for value in result["checks"].values()), f"shard_checks:{sensor}")
    require(c2.sha256_file(shard / "payload_inventory.json") == result["payload_inventory_sha256"], f"inventory_hash:{sensor}")
    require(not join.verify_payload_inventory(shard), f"payload_integrity:{sensor}")
    inventory = join.read_json(shard / "payload_inventory.json")
    payloads = {item["relative_path"]: item for item in inventory}
    require(len(payloads) == len(inventory) == result["payload_count"], f"payload_count:{sensor}")
    expected = {row["episode_id"] for row in protocol["scenarios"]}
    require(len(result["episodes"]) == len(expected) and {row["episode_id"] for row in result["episodes"]} == expected,
            f"episode_set:{sensor}")
    output = {}
    used_payloads: set[str] = set()
    for scenario in protocol["scenarios"]:
        episode_id = scenario["episode_id"]
        directory = shard / "episodes" / episode_id
        manifest = join.read_json(directory / "manifest.json")
        for name, key in (("frames.jsonl", "frames_sha256"), ("summary.json", "summary_sha256"),
                          ("payload_inventory.json", "payload_inventory_sha256"), ("asset_manifest.json", "asset_manifest_sha256")):
            require(c2.sha256_file(directory / name) == manifest[key], f"episode_hash:{sensor}:{episode_id}:{name}")
        rows = join.read_jsonl(directory / "frames.jsonl")
        duration = float(protocol["layouts"][scenario["layout_id"]]["duration_seconds"])
        count = int(round(duration / SAMPLE_SECONDS)) + 1
        require(len(rows) == manifest["frames"] == count and all(row["sensor"] == sensor for row in rows), f"frame_count_sensor:{sensor}:{episode_id}")
        require(manifest["episode_id"] == episode_id and manifest["sensor"] == sensor, f"manifest_identity:{sensor}:{episode_id}")
        for row in rows:
            path = row["sensor_path"]
            require(path in payloads and path not in used_payloads, f"payload_frame_membership:{sensor}:{episode_id}")
            payload = payloads[path]
            require(payload["sha256"] == row["sensor_payload_sha256"] and payload["bytes"] == row["sensor_payload_bytes"],
                    f"payload_frame_binding:{sensor}:{episode_id}")
            used_payloads.add(path)
        check_outcome(rows, scenario)
        output[episode_id] = rows
    require(used_payloads == payloads.keys(), f"unmapped_payloads:{sensor}")
    return output


def check_witness(primary: Sequence[dict[str, Any]], witness: Sequence[dict[str, Any]], *,
                  mode: str, authority: frozenset[str]) -> None:
    require(len(primary) == len(witness), "witness_count")
    for left, right in zip(primary, witness):
        require(close(join.replay_projection(left, mode=mode, authoritative_actor_keys=authority),
                      join.replay_projection(right, mode=mode, authoritative_actor_keys=authority)),
                f"witness_replay:{left['episode_id']}:{left['sample_index']}")


def evaluate(protocol_path: Path, root: Path, require_witness: bool = False) -> dict[str, Any]:
    protocol = join.read_json(protocol_path)
    spec = protocol["final_visual_shell_probe"]
    require(spec["schema"] == SCHEMA, "probe_schema")
    require(len(spec["pairs"]) == 2 and {p["role"] for p in spec["pairs"]} == {"PARTIAL", "FULL"}, "probe_pairs")
    scenarios = {row["episode_id"]: row for row in protocol["scenarios"]}
    require(len(protocol["scenarios"]) == len(scenarios) == protocol["admission"]["expected_episode_count"] == 12,
            "twelve_episode_roster")
    for pair in spec["pairs"]:
        layout = protocol["layouts"][scenarios[pair["episode_id"]]["layout_id"]]
        asset = next(a for a in layout["assets"] if a["asset_key"] == pair["shell_asset"])
        template = protocol["asset_templates"][asset["template"]]
        require(template.get("collision_relevant") is False and
                asset.get("collisions_enabled", template.get("collisions_enabled")) is False,
                f"shell_collision_authority:{pair['shell_asset']}")
    instance = load_shard(protocol, protocol_path, root, "instance")
    pairs = [evaluate_pair(pair, instance[pair["episode_id"]], instance[pair["reference_episode_id"]]) for pair in spec["pairs"]]
    if require_witness:
        witness = load_shard(protocol, protocol_path, root, "witness")
        mode = protocol["capture"].get("witness_replay_projection", join.ACTUAL_ALL_REPLAY)
        for episode_id, rows in instance.items():
            assets = c2.materialize_layout_assets(protocol, scenarios[episode_id]["layout_id"])
            authority = frozenset(a["asset_key"] for a in assets if a.get("scripted_pose_authority", False))
            check_witness(rows, witness[episode_id], mode=mode, authority=authority)
    return {"schema": SCHEMA, "status": PASS, "protocol_sha256": c2.sha256_file(protocol_path),
            "episodes_checked": len(instance), "witness_checked": require_witness, "pairs": pairs,
            "full_roster_authorized": False, "method_predictions_opened": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-witness", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.exists(), f"output_exists:{args.output}")
    try:
        result = evaluate(args.protocol.resolve(strict=True), args.root.resolve(strict=True), args.require_witness)
    except (ValueError, KeyError, OSError, TypeError, StopIteration) as error:
        result = {"schema": SCHEMA, "status": FAIL, "failure": str(error), "full_roster_authorized": False}
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
