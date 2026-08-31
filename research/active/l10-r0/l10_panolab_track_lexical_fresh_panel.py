#!/usr/bin/env python3
"""Freeze and materialize a pixel-unseen Panoramax lexical-ledger panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from l10_panolab_entrance_ray import projection_gate


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist-l10-panolab-track-lexical-fresh-source-protocol-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"MISSING_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    return path


def ascii_tokens(value: str, ignored: set[str]) -> list[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return [token for token in re.findall(r"[a-z0-9]+", folded) if token not in ignored]


def reciprocal(item_a: dict[str, Any], item_b: dict[str, Any], relation: str) -> bool:
    forward = relation.lower()
    backward = "prev" if forward == "next" else "next"
    return (
        any(link.get("rel") == forward and link.get("id") == item_b["id"] for link in item_a.get("links", []))
        and any(link.get("rel") == backward and link.get("id") == item_a["id"] for link in item_b.get("links", []))
    )


def candidate_pairs(
    candidates: dict[str, Any],
    orientation_protocol: dict[str, Any],
    prior_source: dict[str, Any],
    prior_local_ids: set[str],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    prior_way_ids = {episode["target"]["building"]["id"] for episode in prior_source["episodes"]}
    prior_item_ids = {
        episode[phase]["panorama"]["provider_item"]["id"]
        for episode in prior_source["episodes"]
        for phase in ("start", "after")
    }
    ignored = set(contract["ignored_name_tokens"])
    pair_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for candidate in candidates["candidates"]:
        way_id = int(candidate["target_way"]["id"])
        target_name = str(candidate["target_way"].get("tags", {}).get("name") or "").strip()
        if not target_name or way_id in prior_way_ids:
            continue
        for source_role, support_key in (
            ("TARGET_SELF_OCCLUDED", "self_pairs"),
            ("NON_TARGET_BUILDING_OCCLUDED", "other_pairs"),
        ):
            for pair in candidate["supports"][support_key]:
                start_id = str(pair["occluded_item_id"])
                after_id = str(pair["reacquired_item_id"])
                signature = (way_id, start_id, after_id)
                if signature in seen:
                    continue
                seen.add(signature)
                if {start_id, after_id} & (prior_item_ids | prior_local_ids):
                    continue
                start_item = candidate["items"].get(start_id)
                after_item = candidate["items"].get(after_id)
                if start_item is None or after_item is None:
                    continue
                movement = float(pair["camera_displacement_m"])
                after_distance = float(pair["reacquired_classification"]["first_intersection"]["distance_from_camera_m"])
                if not (
                    float(contract["minimum_camera_displacement_m"])
                    <= movement
                    <= float(contract["maximum_camera_displacement_m"])
                ):
                    continue
                if not (
                    float(contract["minimum_after_entrance_distance_m"])
                    <= after_distance
                    <= float(contract["maximum_after_entrance_distance_m"])
                ):
                    continue
                if pair["reacquired_classification"]["stratum"] != "DIRECT":
                    continue
                expected_start = "TARGET_SELF_OCCLUDED" if support_key == "self_pairs" else "OTHER_BUILDING_OCCLUDED"
                if pair["occluded_classification"]["stratum"] != expected_start:
                    continue
                start_dimensions = tuple(start_item["properties"]["pers:interior_orientation"]["sensor_array_dimensions"])
                after_dimensions = tuple(after_item["properties"]["pers:interior_orientation"]["sensor_array_dimensions"])
                start_gate = projection_gate(start_item, orientation_protocol, downloaded_image_size=start_dimensions)
                after_gate = projection_gate(after_item, orientation_protocol, downloaded_image_size=after_dimensions)
                if not start_gate["eligible"] or not after_gate["eligible"]:
                    continue
                relation = str(pair["action_from_occluded"]).lower()
                if relation not in {"prev", "next"} or not pair["reciprocal_provider_links_verified"]:
                    continue
                if not reciprocal(start_item, after_item, relation):
                    continue
                pair_rows.append({
                    "query_index": int(candidate["query_index"]),
                    "source_city": candidate["source_city"],
                    "target_way": candidate["target_way"],
                    "target_polygon_lon_lat": candidate["target_polygon_lon_lat"],
                    "main_entrance_node": candidate["main_entrance_node"],
                    "target_name": target_name,
                    "significant_name_token_count": len(ascii_tokens(target_name, ignored)),
                    "role": source_role,
                    "sequence_id": pair["sequence_id"],
                    "provider_link_relation": relation,
                    "start_item": start_item,
                    "after_item": after_item,
                    "camera_displacement_m": movement,
                    "after_entrance_distance_m": after_distance,
                    "start_classification": pair["occluded_classification"],
                    "after_classification": pair["reacquired_classification"],
                })
    return pair_rows


def select_panel(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    per_way: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        per_way.setdefault(int(row["target_way"]["id"]), []).append(row)
    way_choices = [
        sorted(
            choices,
            key=lambda row: (
                row["significant_name_token_count"],
                row["after_entrance_distance_m"],
                row["sequence_id"],
                row["start_item"]["id"],
                row["after_item"]["id"],
            ),
        )[0]
        for choices in per_way.values()
    ]
    city_rows: dict[str, list[dict[str, Any]]] = {}
    for row in way_choices:
        city_rows.setdefault(row["source_city"], []).append(row)
    for city in city_rows:
        city_rows[city].sort(key=lambda row: (
            row["significant_name_token_count"],
            row["after_entrance_distance_m"],
            int(row["target_way"]["id"]),
            row["sequence_id"],
        ))
    selected: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    round_index = 0
    while len(selected) < count:
        added = False
        for city in sorted(city_rows):
            candidates = city_rows[city]
            if round_index >= len(candidates):
                continue
            for row in candidates[round_index:]:
                if row["sequence_id"] not in used_sequences:
                    selected.append(row)
                    used_sequences.add(row["sequence_id"])
                    added = True
                    break
            if len(selected) == count:
                break
        require(added, "INSUFFICIENT_DISTINCT_WAY_SEQUENCE_CITY_PANEL")
        round_index += 1
    return selected


def freeze(protocol: dict[str, Any], output_path: Path) -> None:
    require(not output_path.exists(), f"OUTPUT_EXISTS:{output_path}")
    candidates_path = verify(protocol["inputs"]["candidate_metadata"])
    orientation_path = verify(protocol["inputs"]["orientation_protocol"])
    prior_source_path = verify(protocol["inputs"]["prior_consumed_active_source"])
    candidates = load(candidates_path)
    orientation = load(orientation_path)
    prior_source = load(prior_source_path)
    prior_image_dir = resolve(protocol["selection"]["prior_local_image_directory"])
    prior_local_ids = {path.stem for path in prior_image_dir.glob("*.jpg")}
    rows = candidate_pairs(candidates, orientation, prior_source, prior_local_ids, protocol["selection"])
    selected = select_panel(rows, int(protocol["selection"]["episode_count"]))
    require(len({row["target_way"]["id"] for row in selected}) == len(selected), "TARGET_WAYS_NOT_UNIQUE")
    require(len({row["sequence_id"] for row in selected}) == len(selected), "SEQUENCES_NOT_UNIQUE")
    selected_ids = {row[phase]["id"] for row in selected for phase in ("start_item", "after_item")}
    require(not (selected_ids & prior_local_ids), "SELECTED_PIXEL_ALREADY_LOCAL")
    receipt = {
        "schema": "blindassist-l10-panolab-track-lexical-fresh-selection-v1",
        "status": "FROZEN_BEFORE_SELECTED_PIXEL_DOWNLOAD_OR_MODEL_CALL",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(resolve(protocol["protocol_path"])),
        "candidate_metadata_sha256": protocol["inputs"]["candidate_metadata"]["sha256"],
        "prior_consumed_active_source_sha256": protocol["inputs"]["prior_consumed_active_source"]["sha256"],
        "selection_rule": protocol["selection_rule"],
        "eligible_pair_count": len(rows),
        "eligible_distinct_target_way_count": len({row["target_way"]["id"] for row in rows}),
        "selected_episode_count": len(selected),
        "selected_distinct_city_count": len({row["source_city"] for row in selected}),
        "selected_distinct_target_way_count": len({row["target_way"]["id"] for row in selected}),
        "selected_distinct_sequence_count": len({row["sequence_id"] for row in selected}),
        "selected_overlap_with_prior_active_target_ways": 0,
        "selected_overlap_with_prior_active_item_ids": 0,
        "selected_overlap_with_existing_local_image_ids": 0,
        "selected_pixel_views_before_freeze": 0,
        "selected_model_calls_before_freeze": 0,
        "episodes": [dict(row, episode_id=f"FL{index:02d}") for index, row in enumerate(selected, start=1)],
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selection": str(output_path),
        "sha256": sha256(output_path),
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "city": row["source_city"],
                "target_way_id": row["target_way"]["id"],
                "target_name": row["target_name"],
                "sequence_id": row["sequence_id"],
                "movement_m": row["camera_displacement_m"],
            }
            for row in receipt["episodes"]
        ],
    }, ensure_ascii=False, indent=2))


def download(url: str, output_path: Path) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    require(not temporary.exists(), f"STALE_TEMPORARY:{temporary}")
    request = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-L10-Development/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize(protocol: dict[str, Any], selection_path: Path, output_path: Path) -> None:
    require(not output_path.exists(), f"OUTPUT_EXISTS:{output_path}")
    selection = load(selection_path)
    require(selection["status"] == "FROZEN_BEFORE_SELECTED_PIXEL_DOWNLOAD_OR_MODEL_CALL", "SELECTION_NOT_FROZEN")
    image_dir = resolve(protocol["materialization"]["image_directory"])
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for episode in selection["episodes"]:
        for phase in ("start", "after"):
            item = episode[f"{phase}_item"]
            path = image_dir / f"{item['id']}.jpg"
            require(not path.exists(), f"SELECTED_IMAGE_ALREADY_EXISTS:{path}")
            download(item["assets"]["hd"]["href"], path)
            with Image.open(path) as image:
                image.load()
                size = [image.width, image.height]
            expected = item["properties"]["pers:interior_orientation"]["sensor_array_dimensions"]
            require(size == expected and size[0] == 2 * size[1], f"IMAGE_SIZE_MISMATCH:{item['id']}")
            images.append({
                "episode_id": episode["episode_id"],
                "phase": phase,
                "item_id": item["id"],
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "image_size": size,
                "url": item["assets"]["hd"]["href"],
            })
    manifest = {
        "schema": "blindassist-l10-panolab-track-lexical-fresh-materialization-v1",
        "selection": str(selection_path.relative_to(ROOT)).replace("\\", "/"),
        "selection_sha256": sha256(selection_path),
        "pixel_views_before_frozen_selection": selection["selected_pixel_views_before_freeze"],
        "model_calls_before_frozen_selection": selection["selected_model_calls_before_freeze"],
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(output_path),
        "sha256": sha256(output_path),
        "image_count": len(images),
        "bytes": sum(row["bytes"] for row in images),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "materialize"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load(protocol_path)
    require(protocol.get("schema") == SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    protocol["protocol_path"] = str(protocol_path)
    if args.mode == "freeze":
        freeze(protocol, args.selection.resolve())
        return
    require(args.manifest is not None, "MANIFEST_REQUIRED_FOR_MATERIALIZE")
    materialize(protocol, args.selection.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
