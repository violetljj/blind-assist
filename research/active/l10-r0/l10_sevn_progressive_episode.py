#!/usr/bin/env python3
"""Run the first pixel-derived L10 SEVN progressive-evidence episodes.

The three policies share one frozen observation cache. Evaluator-private SEVN
door truth is applied only after each runtime output has been sealed in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_evidence_authority_lattice import Evidence, EvidenceAuthorityLattice
from l10_panolab import require, sha256_file, utc_now
import l10_sevn_panolab as sevn
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2
import l10_sevn_ppocrv6_medium_fallback as base
import l10_sevn_ppocrv6_medium_portal_witness as witness


SCHEMA = "blindassist-l10-sevn-progressive-episode-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-sevn-progressive-episode-result-v1"
ARMS = ("PASSIVE", "FIXED_SWEEP", "TRIGGERED_ACTIVE")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def verify_sha256(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"missing frozen input: {path}")
    require(sha256_file(path) == spec["sha256"], f"SHA-256 mismatch: {path}")
    return path


def verify_model_directory(spec: dict[str, Any]) -> Path:
    root = resolve(spec["path"])
    require(root.is_dir(), f"missing model directory: {root}")
    for filename, expected in spec["sha256"].items():
        path = root / filename
        require(path.is_file(), f"missing model file: {path}")
        require(sha256_file(path) == expected, f"model SHA-256 mismatch: {path}")
    return root


def runtime_versions() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
    }


def selected_confidence(output: dict[str, Any]) -> float | None:
    selected = output.get("selected_binding")
    if selected is None:
        return None
    text_score = float(selected["text"].get("score", 0.0))
    portal_score = float(selected["portal"].get("confidence", 0.0))
    return round(text_score * portal_score, 8)


def concise_output(output: dict[str, Any]) -> dict[str, Any]:
    selected = output.get("selected_binding")
    exact_rows = [
        {
            "canonical": row.get("canonical"),
            "score": row.get("score"),
            "box_xyxy": row.get("box_xyxy"),
        }
        for row in output.get("exact_target_text_candidates", [])
        if isinstance(row, dict)
    ]
    return {
        "state": output["state"],
        "observation_branch": output.get("observation_branch"),
        "exact_target_text_candidate_count": len(exact_rows),
        "exact_target_text_candidates": exact_rows,
        "portal_candidate_count": len(output.get("portal_mask_candidates", [])),
        "portal_witness_count": len(output.get("portal_witness_set", [])),
        "binding_confidence": selected_confidence(output),
        "selected_binding": None
        if selected is None
        else {
            "portal_id": selected["portal"]["candidate_id"],
            "portal_confidence": selected["portal"]["confidence"],
            "portal_box_xyxy": selected["portal"]["mask_box_xyxy"],
            "text": selected["text"]["canonical"],
            "text_score": selected["text"]["score"],
            "authority_edge": selected.get("authority_edge"),
        },
    }


def trigger_action(
    output: dict[str, Any], episode: dict[str, Any], viewport_width: int
) -> tuple[str, str]:
    """Choose one truth-free observation action from the current deficit."""
    if output.get("selected_binding") is not None:
        return "HOLD", "CURRENT_TYPED_BINDING_PRESENT"

    candidates = [
        row
        for row in output.get("exact_target_text_candidates", [])
        if isinstance(row, dict) and len(row.get("box_xyxy", [])) == 4
    ]
    if candidates:
        best = sorted(
            candidates,
            key=lambda row: (-float(row.get("score", 0.0)), tuple(row["box_xyxy"])),
        )[0]
        center_x = 0.5 * (float(best["box_xyxy"][0]) + float(best["box_xyxy"][2]))
        action = "PAN_LEFT" if center_x < viewport_width / 2.0 else "PAN_RIGHT"
        return action, "EXACT_TARGET_TOKEN_EDGE_RECENTER"

    start = episode["start_observation_id"]
    approach = episode["transitions"][start]["APPROACH"]
    if approach["action_executed"]:
        return "APPROACH", "NO_TARGET_TOKEN_AND_GRAPH_APPROACH_AVAILABLE"
    return "SWEEP", "NO_TARGET_TOKEN_AND_NO_GRAPH_APPROACH"


def authority_for(output: dict[str, Any], mission: dict[str, Any]) -> dict[str, Any]:
    lattice = EvidenceAuthorityLattice()
    if output.get("selected_binding") is None:
        decision = lattice.step(Evidence())
        scope = "NO_BINDING_AUTHORITY"
    else:
        address = f"SEVN_ADDRESS:{mission['street_name']}:{mission['house_number']}"
        decision = lattice.step(
            Evidence(
                referent_id=str(mission["house_number"]),
                facade_id=address,
                same_referent_continuity_supported=True,
                sibling_exclusion_supported=True,
                sign_facade_association_supported=True,
            )
        )
        scope = "SEVN_ADDRESS_DOOR_SURROGATE_ONLY"
    return {
        "authority": decision.authority.value,
        "authorized_action": decision.action.value,
        "reason": decision.reason,
        "scope": scope,
    }


def empirical_risk_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emitted = [row for row in rows if row["binding_confidence"] is not None]
    emitted.sort(key=lambda row: (-row["binding_confidence"], row["episode_id"]))
    curve = []
    wrong = 0
    total = len(rows)
    for index, row in enumerate(emitted, start=1):
        wrong += row["outcome"] == "WRONG_BINDING"
        curve.append(
            {
                "accepted": index,
                "minimum_confidence": row["binding_confidence"],
                "coverage": round(index / total, 6),
                "risk": round(wrong / index, 6),
                "precision": round((index - wrong) / index, 6),
            }
        )
    aurc = sum(point["risk"] for point in curve) / total if total else 0.0
    return {
        "curve": curve,
        "empirical_aurc_over_episode_denominator": round(aurc, 8),
        "maximum_coverage": round(len(emitted) / total, 6) if total else 0.0,
        "calibration_authority": "NONE_EMPIRICAL_RANKING_ONLY",
    }


def summarize(rows: list[dict[str, Any]], passive_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["outcome"] == "CORRECT_BINDING" for row in rows)
    wrong = sum(row["outcome"] == "WRONG_BINDING" for row in rows)
    unknown = total - correct - wrong
    emitted = correct + wrong
    passive_by_id = {row["episode_id"]: row for row in passive_rows}
    miss_ids = {
        episode_id
        for episode_id, row in passive_by_id.items()
        if row["outcome"] != "CORRECT_BINDING"
    }
    recovered = sum(
        row["episode_id"] in miss_ids and row["outcome"] == "CORRECT_BINDING"
        for row in rows
    )
    extra_views = sum(row["extra_observation_count"] for row in rows)
    actions = sum(row["action_executed"] for row in rows)
    return {
        "episode_count": total,
        "correct_binding": correct,
        "wrong_binding": wrong,
        "unknown": unknown,
        "coverage": round(emitted / total, 6) if total else 0.0,
        "binding_precision": round(correct / emitted, 6) if emitted else None,
        "passive_miss_count": len(miss_ids),
        "recovery_after_passive_miss": recovered,
        "action_count": actions,
        "extra_observation_count": extra_views,
        "mean_extra_observations": round(extra_views / total, 6) if total else 0.0,
        "correct_recoveries_per_extra_observation": round(recovered / extra_views, 6)
        if extra_views
        else None,
        "risk_coverage": empirical_risk_coverage(rows),
    }


def self_check() -> dict[str, str]:
    episode = {
        "start_observation_id": "S",
        "transitions": {"S": {"APPROACH": {"action_executed": False}}},
    }
    unknown = {"selected_binding": None, "exact_target_text_candidates": []}
    assert trigger_action(unknown, episode, 100) == (
        "SWEEP",
        "NO_TARGET_TOKEN_AND_NO_GRAPH_APPROACH",
    )
    left = {
        "selected_binding": None,
        "exact_target_text_candidates": [{"score": 0.9, "box_xyxy": [0, 0, 20, 10]}],
    }
    assert trigger_action(left, episode, 100)[0] == "PAN_LEFT"
    bound = {"selected_binding": {"text": {"score": 1.0}, "portal": {"confidence": 0.5}}}
    assert trigger_action(bound, episode, 100)[0] == "HOLD"
    assert selected_confidence(bound) == 0.5
    rows = [
        {"episode_id": "a", "outcome": "CORRECT_BINDING", "binding_confidence": 0.8},
        {"episode_id": "b", "outcome": "WRONG_BINDING", "binding_confidence": 0.4},
        {"episode_id": "c", "outcome": "UNKNOWN", "binding_confidence": None},
    ]
    curve = empirical_risk_coverage(rows)
    assert curve["maximum_coverage"] == 0.666667
    assert curve["curve"][0]["precision"] == 1.0
    return {"status": "PASS"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite result: {output_path}")
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA, "protocol schema mismatch")
    require(sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "evaluator hash mismatch")

    source_path = verify_sha256(protocol["frozen_inputs"]["public_source"])
    truth_path = verify_sha256(protocol["frozen_inputs"]["evaluator_truth"])
    v2_protocol_path = verify_sha256(protocol["frozen_inputs"]["v2_protocol"])
    verify_sha256(protocol["frozen_inputs"]["authority_lattice"])
    source = read_json(source_path)
    truth = read_json(truth_path)
    v2_protocol = read_json(v2_protocol_path)
    sevn.validate_public(source, int(protocol["cohort"]["episodes_per_scenario"]))
    sevn.validate_truth(truth, source, int(protocol["cohort"]["episodes_per_scenario"]))
    require(len(source["episodes"]) == int(protocol["cohort"]["episode_count"]), "episode count mismatch")

    versions = runtime_versions()
    require(versions == protocol["runtime"]["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable for portal model")

    portal_model_path = verify_sha256(v2_protocol["models"]["portal_model"]["weights"])
    postprocessor_path = verify_sha256(v2_protocol["models"]["portal_model"]["postprocessor"])
    primary_root = resolve(v2_protocol["models"]["ocr"]["model_root"])
    for filename, expected in v2_protocol["models"]["ocr"]["sha256"].items():
        require(sha256_file(primary_root / filename) == expected, f"primary OCR model mismatch: {filename}")
    medium_det_root = verify_model_directory(protocol["models"]["medium_detection"])
    medium_rec_root = verify_model_directory(protocol["models"]["medium_recognition"])

    postprocessor = v1.load_module(postprocessor_path)
    portal_model = YOLO(str(portal_model_path))
    require(portal_model.task == "segment", "portal model task mismatch")
    primary_ocr = RapidOCR(
        params={
            "Global.model_root_dir": str(primary_root),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": int(
                v2_protocol["pixel_contract"]["ocr"]["intra_op_threads"]
            ),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    medium_ocr = PaddleOCR(
        text_detection_model_dir=str(medium_det_root),
        text_recognition_model_dir=str(medium_rec_root),
        engine="onnxruntime",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    witness.PORTAL_CONTRACT = protocol["portal_medium_observation"]

    archive_path = args.highres_archive.resolve()
    archive_contract = v2_protocol["high_resolution_archive"]
    require(archive_path.is_file(), f"missing high-resolution archive: {archive_path}")
    require(archive_path.stat().st_size == int(archive_contract["bytes"]), "archive byte mismatch")
    member_pattern = archive_contract["member_pattern"]
    evaluation_contract = v2_protocol["pixel_contract"]["evaluation"]
    canvas_width = float(v2_protocol["render_contract"]["annotation_canvas_width"])
    canvas_height = float(v2_protocol["render_contract"]["annotation_canvas_height"])

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    episode_traces = []
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path, "r") as archive_zip:
        for sequence, episode in enumerate(source["episodes"], start=1):
            episode_id = episode["episode_id"]
            episode_truth = truth["episodes"][episode_id]
            panorama_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
            view_cache: dict[tuple[int, float], dict[str, Any]] = {}

            def observe(observation_id: str, heading: float) -> dict[str, Any]:
                observation = source["observations"][observation_id]
                frame_id = int(observation["frame_id"])
                key = (frame_id, round(float(heading), 6))
                if key in view_cache:
                    return view_cache[key]
                if frame_id not in panorama_cache:
                    member_name = member_pattern.format(frame_id=frame_id)
                    info = archive_zip.getinfo(member_name)
                    encoded = archive_zip.read(info)
                    panorama = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                    require(panorama is not None, f"{episode_id}: panorama decode failed")
                    require(
                        list(panorama.shape) == v2_protocol["render_contract"]["panorama_shape_hwc"],
                        f"{episode_id}: panorama shape mismatch",
                    )
                    panorama_cache[frame_id] = (
                        panorama,
                        {
                            "archive_member": member_name,
                            "archive_member_crc32": f"{info.CRC:08x}",
                            "archive_member_bytes": info.file_size,
                            "archive_member_sha256": hashlib.sha256(encoded).hexdigest(),
                        },
                    )
                panorama, archive_receipt = panorama_cache[frame_id]
                viewport, render_receipt = v1.render_viewport(
                    panorama,
                    float(observation["camera_pose"]["panorama_angle_degrees"]),
                    float(heading),
                    float(observation["horizontal_fov_degrees"]),
                )
                render_receipt.update(archive_receipt)
                _, output, masks = witness.infer_successor(
                    viewport,
                    episode["mission"],
                    primary_ocr,
                    medium_ocr,
                    portal_model,
                    postprocessor,
                    v2_protocol["pixel_contract"],
                )
                target_annotation = episode_truth["target_door_annotation"]
                target_same_frame = int(target_annotation["frame_id"]) == frame_id
                target_box = v1.annotation_to_viewport(
                    target_annotation,
                    render_receipt,
                    canvas_width,
                    canvas_height,
                )
                height, width = viewport.shape[:2]
                target_visible = target_same_frame and (
                    0.0 <= target_box[0] <= target_box[2] <= width
                    and 0.0 <= target_box[1] <= target_box[3] <= height
                )
                selected = output.get("selected_binding")
                if selected is None:
                    outcome = "UNKNOWN"
                    mask_metrics = None
                elif not target_visible:
                    outcome = "WRONG_BINDING"
                    mask_metrics = None
                else:
                    mask = masks[selected["portal"]["candidate_id"]]
                    mask_metrics = v2.truth_mask_metrics(mask, target_box, evaluation_contract)
                    outcome = "CORRECT_BINDING" if mask_metrics["correct"] else "WRONG_BINDING"
                row = {
                    "observation_id": observation_id,
                    "frame_id": frame_id,
                    "heading_degrees": round(float(heading), 6),
                    "render_receipt": render_receipt,
                    "runtime_output": concise_output(output),
                    "evaluator_truth": {
                        "target_visible": target_visible,
                        "target_box_xyxy": [round(value, 6) for value in target_box]
                        if target_same_frame
                        else None,
                        "selected_mask_metrics": mask_metrics,
                        "outcome": outcome,
                    },
                }
                view_cache[key] = row
                return row

            def views_for(action: str) -> list[dict[str, Any]]:
                start_id = episode["start_observation_id"]
                edge = episode["transitions"][start_id][action]
                require(edge["action_executed"], f"{episode_id}: unavailable action {action}")
                observation_id = edge["to_observation_id"]
                observation = source["observations"][observation_id]
                return [observe(observation_id, heading) for heading in observation["viewport_headings_degrees"]]

            def select_view(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
                emitted = [row for row in rows if row["runtime_output"]["binding_confidence"] is not None]
                if not emitted:
                    return None
                return sorted(
                    emitted,
                    key=lambda row: (
                        -row["runtime_output"]["binding_confidence"],
                        row["frame_id"],
                        row["heading_degrees"],
                    ),
                )[0]

            def arm_row(
                arm: str,
                action: str,
                reason: str,
                rows: list[dict[str, Any]],
                extra_observation_count: int,
            ) -> dict[str, Any]:
                selected_view = select_view(rows)
                final = selected_view or rows[-1]
                output = final["runtime_output"]
                outcome = final["evaluator_truth"]["outcome"] if selected_view else "UNKNOWN"
                return {
                    "episode_id": episode_id,
                    "arm": arm,
                    "action": action,
                    "action_reason": reason,
                    "action_executed": int(action != "HOLD"),
                    "extra_observation_count": extra_observation_count,
                    "observed_view_count": len(rows),
                    "selected_observation_id": final["observation_id"] if selected_view else None,
                    "binding_confidence": output["binding_confidence"] if selected_view else None,
                    "outcome": outcome,
                    "authority": authority_for(output if selected_view else {"selected_binding": None}, episode["mission"]),
                }

            start_id = episode["start_observation_id"]
            start_observation = source["observations"][start_id]
            start_view = observe(start_id, start_observation["viewport_headings_degrees"][0])
            passive = arm_row("PASSIVE", "HOLD", "NO_EXTRA_OBSERVATION", [start_view], 0)
            arm_rows["PASSIVE"].append(passive)

            sweep_views = views_for("SWEEP")
            fixed = arm_row("FIXED_SWEEP", "SWEEP", "ALWAYS_SWEEP", sweep_views, 3)
            arm_rows["FIXED_SWEEP"].append(fixed)

            trigger, trigger_reason = trigger_action(
                start_view["runtime_output"], episode, int(start_view["render_receipt"]["viewport_shape_hwc"][1])
            )
            triggered_views = [start_view] if trigger == "HOLD" else views_for(trigger)
            triggered_extra = 0 if trigger == "HOLD" else (3 if trigger == "SWEEP" else 1)
            triggered = arm_row(
                "TRIGGERED_ACTIVE",
                trigger,
                trigger_reason,
                triggered_views,
                triggered_extra,
            )
            arm_rows["TRIGGERED_ACTIVE"].append(triggered)

            episode_traces.append(
                {
                    "episode_id": episode_id,
                    "scenario_class": episode_truth["scenario_class"],
                    "mission": episode["mission"],
                    "arms": {arm: arm_rows[arm][-1] for arm in ARMS},
                    "observation_trace": list(view_cache.values()),
                }
            )
            print(
                json.dumps(
                    {
                        "episode": episode_id,
                        "progress": f"{sequence}/{len(source['episodes'])}",
                        "passive": passive["outcome"],
                        "fixed": fixed["outcome"],
                        "triggered": triggered["outcome"],
                        "triggered_action": trigger,
                    }
                ),
                flush=True,
            )

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent portal fallback: {actual_device}")
    metrics = {
        arm: summarize(arm_rows[arm], arm_rows["PASSIVE"])
        for arm in ARMS
    }
    gates = {
        "triggered_correct_at_least_fixed": metrics["TRIGGERED_ACTIVE"]["correct_binding"]
        >= metrics["FIXED_SWEEP"]["correct_binding"],
        "triggered_wrong_no_more_than_fixed": metrics["TRIGGERED_ACTIVE"]["wrong_binding"]
        <= metrics["FIXED_SWEEP"]["wrong_binding"],
        "triggered_uses_fewer_extra_observations": metrics["TRIGGERED_ACTIVE"]["extra_observation_count"]
        < metrics["FIXED_SWEEP"]["extra_observation_count"],
        "triggered_improves_over_passive": metrics["TRIGGERED_ACTIVE"]["correct_binding"]
        > metrics["PASSIVE"]["correct_binding"],
    }
    gate_met = all(gates.values())
    result = {
        "schema": RESULT_SCHEMA,
        "generated_at_utc": utc_now(),
        "decision": "L10_SEVN_TRIGGERED_PROGRESSIVE_EPISODE_DEVELOPMENT_GATE_MET"
        if gate_met
        else "L10_SEVN_TRIGGERED_PROGRESSIVE_EPISODE_DEVELOPMENT_GATE_NOT_MET",
        "claim_scope": "CONSUMED_SEVN_ADDRESS_DOOR_PIXEL_EPISODE_DEVELOPMENT_ONLY",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "v2_protocol": {"path": str(v2_protocol_path), "sha256": sha256_file(v2_protocol_path)},
            "high_resolution_archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size},
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "portal_device": actual_device,
            "device_name": torch.cuda.get_device_name(0),
            "medium_ocr_device": "cpu",
        },
        "policy": protocol["policy"],
        "metrics": metrics,
        "gates": gates,
        "gate_met": gate_met,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "episode_results": episode_traces,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--highres-archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    require(args.protocol is not None, "--protocol is required")
    require(args.highres_archive is not None, "--highres-archive is required")
    require(args.output is not None, "--output is required")
    result = run(args)
    print(json.dumps({"decision": result["decision"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
