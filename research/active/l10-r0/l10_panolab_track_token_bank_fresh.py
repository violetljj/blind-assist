#!/usr/bin/env python3
"""Fresh Development replay of a long/short-term exact-token evidence bank."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from paddleocr import PaddleOCR

import l10_panolab_track_lexical_consensus as consensus
import l10_panolab_track_lexical_fresh_panel as panel
import l10_panolab_track_lexical_ledger as ledger


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-track-token-bank-fresh-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-track-token-bank-fresh-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(spec: dict[str, Any]) -> Path:
    path = ledger.resolve(spec["path"])
    require(path.is_file(), f"MISSING_FROZEN_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    if "bytes" in spec:
        require(path.stat().st_size == int(spec["bytes"]), f"BYTE_COUNT_MISMATCH:{path}")
    return path


def build_frame(
    episode: dict[str, Any],
    phase: str,
    image_receipt: dict[str, Any],
) -> dict[str, Any]:
    entrance = episode["main_entrance_node"]
    return {
        "key": f"{episode['episode_id']}_{phase}",
        "target": {
            "entrance_node": {
                "id": entrance["id"],
                "lon_lat": [entrance["lon"], entrance["lat"]],
            }
        },
        "panorama": {
            "local_path": image_receipt["path"],
            "image_sha256": image_receipt["sha256"],
            "image_bytes": image_receipt["bytes"],
            "image_size": image_receipt["image_size"],
            "provider_item": episode[f"{phase}_item"],
        },
    }


def evaluate_frame(
    pipeline: PaddleOCR,
    frame: dict[str, Any],
    targets: list[dict[str, Any]],
    own_episode_id: str,
    projection_protocol: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    panorama = frame["panorama"]
    path = ledger.resolve(panorama["local_path"])
    require(sha256(path) == panorama["image_sha256"], f"IMAGE_HASH_MISMATCH:{frame['key']}")
    require(path.stat().st_size == int(panorama["image_bytes"]), f"IMAGE_BYTES_MISMATCH:{frame['key']}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(image is not None, f"IMAGE_DECODE_FAILED:{path}")
    require([image.shape[1], image.shape[0]] == panorama["image_size"], f"IMAGE_SIZE_MISMATCH:{frame['key']}")
    ray = ledger.strict_ray(frame, projection_protocol)
    crop, window = ledger.entrance_window(image, ray, protocol["observation_window"])
    rows, ocr = ledger.run_ocr(pipeline, crop)
    predecessor_matches = {
        target["episode_id"]: consensus.match_target_name(rows, target["entity_name"], protocol["lexical_match"])
        for target in targets
    }
    return {
        "frame_key": frame["key"],
        "panorama_item_id": panorama["provider_item"]["id"],
        "strict_entrance_ray": ray,
        "window": window,
        "ocr": ocr,
        "ocr_rows": rows,
        "predecessor_own_target_match": predecessor_matches[own_episode_id],
        "predecessor_counterfactual_matches": {
            target_id: match
            for target_id, match in predecessor_matches.items()
            if target_id != own_episode_id
        },
    }


def token_bank_match(
    frames: list[dict[str, Any]],
    entity_name: str,
    lexical_contract: dict[str, Any],
    bank_contract: dict[str, Any],
) -> dict[str, Any]:
    target_tokens = ledger.significant_name_tokens(entity_name, lexical_contract)
    require(target_tokens, f"EMPTY_TARGET_TOKENS:{entity_name}")
    witnesses: dict[str, list[dict[str, Any]]] = {token: [] for token in target_tokens}
    minimum_score = float(bank_contract["minimum_ocr_row_score"])
    for frame in frames:
        for row in frame["ocr_rows"]:
            if row["score"] < minimum_score:
                continue
            for token in target_tokens:
                if token in row["ascii_tokens"]:
                    witnesses[token].append({
                        "phase": frame["frame_key"].rsplit("_", 1)[-1],
                        "panorama_item_id": frame["panorama_item_id"],
                        "text": row["text"],
                        "score": row["score"],
                        "box_xyxy": row["box_xyxy"],
                    })
    observed = [token for token in target_tokens if witnesses[token]]
    if len(target_tokens) == 1:
        required = 1
    else:
        required = max(
            int(bank_contract["minimum_distinct_target_tokens"]),
            math.ceil(float(bank_contract["minimum_target_token_coverage"]) * len(target_tokens)),
        )
        required = min(required, len(target_tokens))
    return {
        "matched": len(observed) >= required,
        "tier": "LONG_SHORT_EXACT_TOKEN_BANK" if len(observed) >= required else "NONE",
        "target_tokens": target_tokens,
        "required_distinct_target_tokens": required,
        "observed_target_tokens": observed,
        "target_token_coverage": round(len(observed) / len(target_tokens), 6),
        "witnesses": {token: witnesses[token] for token in observed},
        "phase_count": len(frames),
        "portal_ownership_authority": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    frozen = protocol["frozen_inputs"]
    verify(frozen["fresh_source_protocol"])
    selection = load(verify(frozen["fresh_selection"]))
    manifest = load(verify(frozen["materialization_manifest"]))
    prior_source = load(verify(frozen["prior_consumed_active_source"]))
    projection_protocol = load(verify(frozen["orientation_projection_protocol"]))
    verify(frozen["orientation_projection_evaluator"])
    verify(frozen["frozen_predecessor_evaluator"])
    verify(frozen["frozen_predecessor_protocol"])
    verify(frozen["posthoc_predecessor_result"])
    require(selection["status"] == "FROZEN_BEFORE_SELECTED_PIXEL_DOWNLOAD_OR_MODEL_CALL", "SELECTION_NOT_FROZEN")
    require(manifest["selection_sha256"] == frozen["fresh_selection"]["sha256"], "MANIFEST_SELECTION_HASH_MISMATCH")
    require(manifest["pixel_views_before_frozen_selection"] == 0, "PIXELS_VIEWED_BEFORE_SELECTION")
    require(manifest["model_calls_before_frozen_selection"] == 0, "MODEL_CALLED_BEFORE_SELECTION")
    require(len(selection["episodes"]) == len(manifest["images"]) // 2 == 4, "PANEL_SIZE_MISMATCH")

    prior_way_ids = {episode["target"]["building"]["id"] for episode in prior_source["episodes"]}
    prior_item_ids = {
        episode[phase]["panorama"]["provider_item"]["id"]
        for episode in prior_source["episodes"]
        for phase in ("start", "after")
    }
    selected_way_ids = {episode["target_way"]["id"] for episode in selection["episodes"]}
    selected_item_ids = {
        episode[f"{phase}_item"]["id"]
        for episode in selection["episodes"]
        for phase in ("start", "after")
    }
    require(not (selected_way_ids & prior_way_ids), "PRIOR_TARGET_WAY_OVERLAP")
    require(not (selected_item_ids & prior_item_ids), "PRIOR_ITEM_OVERLAP")
    require(len({episode["sequence_id"] for episode in selection["episodes"]}) == 4, "SEQUENCES_NOT_UNIQUE")
    require(len({episode["source_city"] for episode in selection["episodes"]}) >= 3, "CITY_DIVERSITY_NOT_3")

    runtime = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
    }
    require(runtime == protocol["runtime"]["versions"], f"RUNTIME_MISMATCH:{runtime}")
    detection_root = ledger.verify_model(protocol["models"]["medium_detection"])
    recognition_root = ledger.verify_model(protocol["models"]["medium_recognition"])
    pipeline = PaddleOCR(
        text_detection_model_dir=str(detection_root),
        text_recognition_model_dir=str(recognition_root),
        engine="onnxruntime",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    image_index = {(row["episode_id"], row["phase"]): row for row in manifest["images"]}
    targets = [
        {"episode_id": episode["episode_id"], "entity_name": episode["target_name"]}
        for episode in selection["episodes"]
    ]
    result_rows = []
    started = time.perf_counter()
    for episode in selection["episodes"]:
        relation = episode["provider_link_relation"]
        require(panel.reciprocal(episode["start_item"], episode["after_item"], relation), f"RECIPROCAL_LINK_FAILURE:{episode['episode_id']}")
        require(episode["start_classification"]["stratum"] in {"TARGET_SELF_OCCLUDED", "OTHER_BUILDING_OCCLUDED"}, f"START_NOT_OCCLUDED:{episode['episode_id']}")
        require(episode["after_classification"]["stratum"] == "DIRECT", f"AFTER_NOT_DIRECT:{episode['episode_id']}")
        start_frame = build_frame(episode, "start", image_index[(episode["episode_id"], "start")])
        after_frame = build_frame(episode, "after", image_index[(episode["episode_id"], "after")])
        start = evaluate_frame(pipeline, start_frame, targets, episode["episode_id"], projection_protocol, protocol)
        after = evaluate_frame(pipeline, after_frame, targets, episode["episode_id"], projection_protocol, protocol)
        predecessor_after = bool(after["predecessor_own_target_match"]["matched"])
        predecessor_track = bool(start["predecessor_own_target_match"]["matched"] or predecessor_after)
        token_after = token_bank_match([after], episode["target_name"], protocol["lexical_match"], protocol["long_short_token_bank"])
        token_track = token_bank_match([start, after], episode["target_name"], protocol["lexical_match"], protocol["long_short_token_bank"])
        controls = {
            target["episode_id"]: token_bank_match(
                [start, after], target["entity_name"], protocol["lexical_match"], protocol["long_short_token_bank"]
            )
            for target in targets
            if target["episode_id"] != episode["episode_id"]
        }
        result_rows.append({
            "episode_id": episode["episode_id"],
            "source_city": episode["source_city"],
            "target_way_id": episode["target_way"]["id"],
            "target_name": episode["target_name"],
            "sequence_id": episode["sequence_id"],
            "action": "SIDESTEP_TO_ENTRANCE_FACE",
            "movement_distance_m": episode["camera_displacement_m"],
            "geometry": {
                "start_class": episode["start_classification"]["stratum"],
                "after_class": episode["after_classification"]["stratum"],
                "after_entrance_ray_authorized": True,
                "reciprocal_provider_links_verified": True,
            },
            "frames": {"start": start, "after": after},
            "predecessor": {
                "after_frame_only_candidate": predecessor_after,
                "track_carried_candidate": predecessor_track,
            },
            "successor": {
                "after_frame_only_token_bank": token_after,
                "long_short_track_token_bank": token_track,
                "counterfactual_track_token_banks": controls,
                "portal_ownership_binding_emitted": False,
            },
        })

    count = len(result_rows)
    predecessor_after = sum(row["predecessor"]["after_frame_only_candidate"] for row in result_rows)
    predecessor_track = sum(row["predecessor"]["track_carried_candidate"] for row in result_rows)
    successor_after = sum(row["successor"]["after_frame_only_token_bank"]["matched"] for row in result_rows)
    successor_track = sum(row["successor"]["long_short_track_token_bank"]["matched"] for row in result_rows)
    counterfactual_trials = sum(len(row["successor"]["counterfactual_track_token_banks"]) for row in result_rows)
    counterfactual_matches = sum(
        match["matched"]
        for row in result_rows
        for match in row["successor"]["counterfactual_track_token_banks"].values()
    )
    metrics = {
        "episodes": count,
        "source_cities": len({row["source_city"] for row in result_rows}),
        "target_way_overlap_with_prior_active": len(selected_way_ids & prior_way_ids),
        "item_overlap_with_prior_active": len(selected_item_ids & prior_item_ids),
        "strict_orientation_images": 2 * count,
        "reciprocal_action_receipts": count,
        "start_occluded_geometry_roles": count,
        "post_action_direct_geometry_roles": count,
        "post_action_authorized_entrance_rays": count,
        "predecessor_after_frame_only_candidates": predecessor_after,
        "predecessor_track_carried_candidates": predecessor_track,
        "successor_after_frame_only_token_banks": successor_after,
        "successor_long_short_track_token_banks": successor_track,
        "successor_track_gain_over_predecessor_track": successor_track - predecessor_track,
        "successor_track_gain_over_successor_after_only": successor_track - successor_after,
        "successor_long_short_track_rate": round(successor_track / count, 6),
        "counterfactual_track_trials": counterfactual_trials,
        "counterfactual_track_matches": counterfactual_matches,
        "portal_ownership_bindings_emitted": 0,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    gate_spec = protocol["gate"]
    gate = {
        "four_fresh_episodes_evaluated": count == 4,
        "minimum_three_source_cities": metrics["source_cities"] >= 3,
        "zero_prior_active_target_way_overlap": metrics["target_way_overlap_with_prior_active"] == 0,
        "zero_prior_active_item_overlap": metrics["item_overlap_with_prior_active"] == 0,
        "strict_orientation_8_of_8": metrics["strict_orientation_images"] == 8,
        "reciprocal_action_receipts_4_of_4": metrics["reciprocal_action_receipts"] == 4,
        "post_action_authorized_entrance_rays_4_of_4": metrics["post_action_authorized_entrance_rays"] == 4,
        "minimum_successor_long_short_track_candidates": successor_track >= int(gate_spec["minimum_successor_long_short_track_candidates"]),
        "minimum_gain_over_frozen_predecessor_track": successor_track - predecessor_track >= int(gate_spec["minimum_gain_over_frozen_predecessor_track"]),
        "long_short_track_not_worse_than_after_only": successor_track >= successor_after,
        "zero_counterfactual_track_matches": counterfactual_matches == 0,
        "zero_portal_ownership_bindings": metrics["portal_ownership_bindings_emitted"] == 0,
    }
    gate["passed"] = all(gate.values())
    decision = protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"]
    result = {
        "schema": RESULT_SCHEMA,
        "decision": decision,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": protocol["evaluator"]["sha256"],
        "runtime": runtime,
        "metrics": metrics,
        "gate": gate,
        "rows": result_rows,
        "evidence_semantics": protocol["evidence_semantics"],
        "research_basis": protocol["research_basis"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "metrics": metrics,
        "gate": gate,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
