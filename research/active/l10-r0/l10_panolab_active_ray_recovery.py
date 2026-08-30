#!/usr/bin/env python3
"""Evaluate causal entrance-ray recovery on real adjacent Panoramax frames."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from l10_panolab_entrance_ray import project_entrance_ray
from named_poi_entity_linked_entrance_ray import _distance, _visibility


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"HASH_MISMATCH:{path}")
    if "bytes" in spec and path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"BYTE_COUNT_MISMATCH:{path}")
    return path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reciprocal_link(item: dict[str, Any], relation: str, target_id: str) -> bool:
    return any(
        link.get("rel") == relation and str(link.get("id")) == str(target_id)
        for link in item.get("links", [])
    )


def opposite(relation: str) -> str:
    return "next" if relation == "prev" else "prev"


def strict_ray(
    frame: dict[str, Any], projection_protocol: dict[str, Any]
) -> dict[str, Any]:
    panorama = frame["panorama"]
    entrance = frame["target"]["entrance_node"]
    lon_lat = entrance["lon_lat"]
    return project_entrance_ray(
        panorama["provider_item"],
        {"id": entrance["id"], "lon": lon_lat[0], "lat": lon_lat[1]},
        projection_protocol,
        downloaded_image_size=tuple(panorama["image_size"]),
    )


def validate_image(frame: dict[str, Any]) -> None:
    panorama = frame["panorama"]
    path = resolve(panorama["local_path"])
    require(sha256(path) == panorama["image_sha256"], f"IMAGE_HASH_MISMATCH:{frame['key']}")
    require(path.stat().st_size == int(panorama["image_bytes"]), f"IMAGE_BYTES_MISMATCH:{frame['key']}")
    with Image.open(path) as image:
        actual = [image.width, image.height]
        image.load()
    require(actual == panorama["image_size"] and actual[0] == 2 * actual[1], f"IMAGE_SIZE_MISMATCH:{frame['key']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    projection_evaluator = Path(__file__).with_name("l10_panolab_entrance_ray.py")
    require(sha256(projection_evaluator) == protocol["orientation_projection"]["evaluator"]["sha256"], "PROJECTION_EVALUATOR_HASH_MISMATCH")
    source_path = verify_file(protocol["source"])
    projection_protocol = load(verify_file(protocol["orientation_projection"]["protocol"]))
    source = load(source_path)
    runtime = {"python": sys.version.split()[0], "pillow": importlib.metadata.version("Pillow")}
    require(runtime == protocol["runtime"]["versions"], f"RUNTIME_MISMATCH:{runtime}")
    require(len(source["episodes"]) == 4, "EPISODE_COUNT_NOT_4")
    require(len({episode["target"]["building"]["id"] for episode in source["episodes"]}) == 4, "TARGET_WAYS_NOT_UNIQUE")
    require(len({episode["sequence_id"] for episode in source["episodes"]}) == 4, "SEQUENCES_NOT_UNIQUE")

    clearance_m = float(protocol["geometry"]["pre_entrance_clearance_m"])
    rows = []
    for episode in source["episodes"]:
        start = episode["start"]
        after = episode["after"]
        validate_image(start)
        validate_image(after)
        start_item = start["panorama"]["provider_item"]
        after_item = after["panorama"]["provider_item"]
        relation = episode["provider_link_relation"]
        require(start_item["collection"] == after_item["collection"] == episode["sequence_id"], f"SEQUENCE_MISMATCH:{episode['episode_id']}")
        forward = reciprocal_link(start_item, relation, after_item["id"])
        backward = reciprocal_link(after_item, opposite(relation), start_item["id"])
        movement_m = _distance(start["panorama"]["camera_lon_lat"], after["panorama"]["camera_lon_lat"])
        require(abs(movement_m - float(episode["camera_displacement_m"])) <= 0.35, f"MOVEMENT_RECEIPT_MISMATCH:{episode['episode_id']}")
        require(3.0 <= movement_m <= 15.0, f"MOVEMENT_OUT_OF_RANGE:{episode['episode_id']}")
        start_visibility = _visibility(start, clearance_m)
        after_visibility = _visibility(after, clearance_m)
        start_ray = strict_ray(start, projection_protocol)
        after_ray = strict_ray(after, projection_protocol)
        expected_start = episode["start_expected_visibility"]
        start_correct = start_visibility["class"] == expected_start
        after_visible = after_visibility["class"] == "VISIBLE_TARGET_ENTRANCE"
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "target_name": episode["target"]["entity_name"],
                "start_expected_visibility": expected_start,
                "start_visibility": start_visibility,
                "after_visibility": after_visibility,
                "start_visibility_correct": start_correct,
                "provider_link_relation": relation,
                "reciprocal_provider_links_verified": forward and backward,
                "movement_distance_m": round(movement_m, 3),
                "action": "SIDESTEP_TO_ENTRANCE_FACE",
                "start_ray_gate": start_ray["projection_gate"],
                "after_ray_gate": after_ray["projection_gate"],
                "start_ray_authorized": start_visibility["class"] == "VISIBLE_TARGET_ENTRANCE",
                "after_ray_authorized": after_visible,
                "after_entrance_ray": after_ray if after_visible else None,
                "authority_count_before": int(start_visibility["class"] == "VISIBLE_TARGET_ENTRANCE"),
                "authority_count_after": int(after_visible),
                "active_recovery": start_correct and not (start_visibility["class"] == "VISIBLE_TARGET_ENTRANCE") and after_visible,
            }
        )

    metrics = {
        "strict_orientation_images": sum(
            row[phase]["eligible"]
            for row in rows
            for phase in ("start_ray_gate", "after_ray_gate")
        ),
        "strict_orientation_image_total": 8,
        "initial_visibility_role_correct": sum(row["start_visibility_correct"] for row in rows),
        "reciprocal_action_receipts": sum(row["reciprocal_provider_links_verified"] for row in rows),
        "occluded_false_authorizations": sum(row["start_ray_authorized"] for row in rows),
        "post_action_ray_authorizations": sum(row["after_ray_authorized"] for row in rows),
        "active_recoveries": sum(row["active_recovery"] for row in rows),
        "active_recovery_rate": sum(row["active_recovery"] for row in rows) / len(rows),
        "mean_authority_count_delta": sum(row["authority_count_after"] - row["authority_count_before"] for row in rows) / len(rows),
        "mean_camera_displacement_m": sum(row["movement_distance_m"] for row in rows) / len(rows),
    }
    gate = {
        "strict_orientation_8_of_8": metrics["strict_orientation_images"] == 8,
        "initial_visibility_role_4_of_4": metrics["initial_visibility_role_correct"] == 4,
        "reciprocal_action_receipt_4_of_4": metrics["reciprocal_action_receipts"] == 4,
        "zero_occluded_false_authorization": metrics["occluded_false_authorizations"] == 0,
        "post_action_ray_authorization_4_of_4": metrics["post_action_ray_authorizations"] == 4,
        "active_recovery_4_of_4": metrics["active_recoveries"] == 4,
    }
    gate["passed"] = all(gate.values())
    decision = protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"]
    result = {
        "schema": protocol["result_schema"],
        "decision": decision,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "source_sha256": protocol["source"]["sha256"],
        "evaluator_sha256": protocol["evaluator"]["sha256"],
        "runtime": runtime,
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
