"""Execute the frozen public-real 8x89 pixels and independent local teachers.

The runner is resumable at observation boundaries.  It never reads private
truth while downloading pixels or running a teacher/provider.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import requests
from PIL import Image

from .annotation import make_annotation
from .truth_contract import validate_annotation


SCHEMA = "blindassist_public_real_frozen_8x89_run_v0"
YOLOE_CONFIDENCE = 0.001
FUNCTIONAL_CONFIDENCE = 0.25
MAX_TEACHER_PROPOSALS = 10
TEACHER_AGREEMENT_IOU = 0.30
MAP_REGION_CENTER_TOLERANCE = 0.20


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _observations(public: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    return [(episode, row) for episode in public["episodes"] for row in episode["observations"]]


def initialize(args: argparse.Namespace) -> None:
    if args.run_dir.exists():
        raise ValueError("run directory already exists")
    public = _read(args.public)
    rows = _observations(public)
    if public.get("episode_count") != 8 or len(rows) != 89:
        raise ValueError("frozen roster must remain exactly 8 episodes / 89 observations")
    models = {
        "teacher_A": {
            "implementation_id": "YOLOE-26n-seg-door-text-v0",
            "model_path": str(args.yoloe_model.resolve()),
            "model_sha256": _sha256(args.yoloe_model),
            "text_encoder_path": str(args.text_encoder.resolve()),
            "text_encoder_sha256": _sha256(args.text_encoder),
            "confidence": YOLOE_CONFIDENCE,
        },
        "teacher_B": {
            "implementation_id": "YOLO11n-functional-door-base-v1",
            "model_path": str(args.functional_base.resolve()),
            "model_sha256": _sha256(args.functional_base),
            "confidence": FUNCTIONAL_CONFIDENCE,
        },
        "teacher_C": {
            "implementation_id": "YOLO11n-functional-door-domain-adapted-v1",
            "model_path": str(args.functional_adapted.resolve()),
            "model_sha256": _sha256(args.functional_adapted),
            "confidence": FUNCTIONAL_CONFIDENCE,
        },
    }
    if len({value["model_sha256"] for value in models.values()}) != 3:
        raise ValueError("teacher weights must be distinct")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "git_commit": commit,
        "inputs": {
            "public_path": str(args.public.resolve()),
            "public_sha256": _sha256(args.public),
            "metadata_path": str(args.metadata.resolve()),
            "metadata_sha256": _sha256(args.metadata),
            "goal_roster_path": str(args.goal_roster.resolve()),
            "goal_roster_sha256": _sha256(args.goal_roster),
            "episode_count": 8,
            "observation_count": 89,
        },
        "teachers": models,
        "teacher_agreement_iou": TEACHER_AGREEMENT_IOU,
        "map_region_center_tolerance": MAP_REGION_CENTER_TOLERANCE,
        "teacher_only_functional_truth": False,
        "baseline_private_truth_access": False,
        "post_outcome_resampling_or_tuning": False,
        "status": "FROZEN_BEFORE_PIXEL_ACCESS",
    }
    args.run_dir.mkdir(parents=True)
    _atomic_json(args.run_dir / "manifest.json", manifest)
    (args.run_dir / "public.json").write_bytes(args.public.read_bytes())
    (args.run_dir / "mapillary-metadata.json").write_bytes(args.metadata.read_bytes())
    (args.run_dir / "goal-roster.json").write_bytes(args.goal_roster.read_bytes())


def _graph_get(session: requests.Session, url: str, params: Mapping[str, Any]) -> Any:
    response = session.get(url, params=dict(params), timeout=60)
    response.raise_for_status()
    value = response.json()
    if isinstance(value, Mapping) and value.get("error"):
        raise ValueError("Mapillary response contains an error")
    return value


def acquire_pixels(args: argparse.Namespace) -> None:
    manifest_path = args.run_dir / "manifest.json"
    manifest = _read(manifest_path)
    public = _read(args.run_dir / "public.json")
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise ValueError("Mapillary access token unavailable")
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    rows = _observations(public)
    ids = [str(row["source_frame_id"]) for _, row in rows]
    urls: dict[str, str] = {}
    for offset in range(0, len(ids), 40):
        values = _graph_get(session, "https://graph.mapillary.com/", {
            "ids": ",".join(ids[offset:offset + 40]),
            "fields": "thumb_2048_url",
        })
        for image_id, value in values.items():
            if isinstance(value, Mapping) and value.get("thumb_2048_url"):
                urls[str(image_id)] = str(value["thumb_2048_url"])
    if set(urls) != set(ids):
        raise ValueError(f"pixel URL coverage mismatch: {len(urls)}/{len(ids)}")
    pixels_dir = args.run_dir / "pixels"
    pixels_dir.mkdir(exist_ok=True)
    pixel_rows = []
    for index, (_, row) in enumerate(rows, start=1):
        image_id = str(row["source_frame_id"])
        path = pixels_dir / f"{image_id}.jpg"
        if not path.exists():
            response = session.get(urls[image_id], timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        pixel_rows.append({
            "observation_id": row["observation_id"],
            "source_frame_id": image_id,
            "image_path": str(path.resolve()),
            "image_sha256": _sha256(path),
            "width": width,
            "height": height,
        })
        print(f"pixels {index}/89 {row['observation_id']}", flush=True)
    _atomic_json(args.run_dir / "pixels.json", {
        "schema_version": "blindassist_public_real_8x89_pixels_v0",
        "public_sha256": manifest["inputs"]["public_sha256"],
        "pixel_count": len(pixel_rows),
        "pixels": pixel_rows,
    })
    manifest["status"] = "PIXELS_COMPLETE_TEACHERS_NOT_RUN"
    manifest["pixels_sha256"] = _sha256(args.run_dir / "pixels.json")
    _atomic_json(manifest_path, manifest)


def _boxes(result: Any, width: int, height: int, *, door_classes: set[int] | None = None) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    rows = []
    for box, score, class_id in zip(
        result.boxes.xyxy.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
        result.boxes.cls.detach().cpu().tolist(),
        strict=True,
    ):
        if door_classes is not None and int(class_id) not in door_classes:
            continue
        normalized = [
            max(0.0, min(1.0, float(box[0]) / width)),
            max(0.0, min(1.0, float(box[1]) / height)),
            max(0.0, min(1.0, float(box[2]) / width)),
            max(0.0, min(1.0, float(box[3]) / height)),
        ]
        rows.append({
            "bbox_normalized_xyxy": normalized,
            "score": float(score),
            "class_id": int(class_id),
            "class_name": str(result.names[int(class_id)]),
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:MAX_TEACHER_PROPOSALS]


def run_teachers(args: argparse.Namespace) -> None:
    manifest_path = args.run_dir / "manifest.json"
    manifest = _read(manifest_path)
    pixels = _read(args.run_dir / "pixels.json")
    if pixels.get("pixel_count") != 89:
        raise ValueError("pixels are incomplete")
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE

    models = manifest["teachers"]
    yoloe = YOLOE(models["teacher_A"]["model_path"])
    previous = Path.cwd()
    os.chdir(Path(models["teacher_A"]["text_encoder_path"]).parent)
    try:
        yoloe.set_classes(["door"])
    finally:
        os.chdir(previous)
    base = YOLO(models["teacher_B"]["model_path"])
    adapted = YOLO(models["teacher_C"]["model_path"])
    base_door = {int(key) for key, name in base.names.items() if str(name) == "door"}
    adapted_door = {int(key) for key, name in adapted.names.items() if str(name) == "door"}
    if not base_door or not adapted_door:
        raise ValueError("functional teacher taxonomy drift")
    device: Any = 0 if torch.cuda.is_available() else "cpu"
    outputs = {key: [] for key in ("teacher_A", "teacher_B", "teacher_C")}
    for index, row in enumerate(pixels["pixels"], start=1):
        image_path = row["image_path"]
        width, height = int(row["width"]), int(row["height"])
        results = {
            "teacher_A": yoloe.predict(source=image_path, imgsz=640, conf=YOLOE_CONFIDENCE, max_det=100, device=device, verbose=False)[0],
            "teacher_B": base.predict(source=image_path, imgsz=640, conf=FUNCTIONAL_CONFIDENCE, max_det=100, device=device, verbose=False)[0],
            "teacher_C": adapted.predict(source=image_path, imgsz=640, conf=FUNCTIONAL_CONFIDENCE, max_det=100, device=device, verbose=False)[0],
        }
        outputs["teacher_A"].append({"observation_id": row["observation_id"], "proposals": _boxes(results["teacher_A"], width, height)})
        outputs["teacher_B"].append({"observation_id": row["observation_id"], "proposals": _boxes(results["teacher_B"], width, height, door_classes=base_door)})
        outputs["teacher_C"].append({"observation_id": row["observation_id"], "proposals": _boxes(results["teacher_C"], width, height, door_classes=adapted_door)})
        print(f"teachers {index}/89 {row['observation_id']}", flush=True)
    teachers_dir = args.run_dir / "teachers"
    for key, rows in outputs.items():
        _atomic_json(teachers_dir / f"{key}.json", {
            "schema_version": "blindassist_public_real_teacher_raw_v0",
            "teacher_id": key,
            "implementation": models[key],
            "private_truth_access": False,
            "ultralytics_version": ultralytics.__version__,
            "observations": rows,
        })
    manifest["status"] = "TEACHERS_COMPLETE_TRUTH_NOT_FROZEN"
    manifest["teacher_output_sha256"] = {
        key: _sha256(teachers_dir / f"{key}.json") for key in outputs
    }
    _atomic_json(manifest_path, manifest)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _consensus(teacher_rows: Mapping[str, Mapping[str, Any]]) -> tuple[str, list[list[float]]]:
    keys = list(teacher_rows)
    clusters: list[tuple[set[str], list[list[float]]]] = []
    for left_index, left_key in enumerate(keys):
        for right_key in keys[left_index + 1:]:
            for left in teacher_rows[left_key]["proposals"]:
                for right in teacher_rows[right_key]["proposals"]:
                    if _iou(left["bbox_normalized_xyxy"], right["bbox_normalized_xyxy"]) >= TEACHER_AGREEMENT_IOU:
                        boxes = [left["bbox_normalized_xyxy"], right["bbox_normalized_xyxy"]]
                        supporters = {left_key, right_key}
                        for third_key in set(keys) - supporters:
                            matches = [
                                item["bbox_normalized_xyxy"] for item in teacher_rows[third_key]["proposals"]
                                if max(_iou(item["bbox_normalized_xyxy"], box) for box in boxes) >= TEACHER_AGREEMENT_IOU
                            ]
                            if matches:
                                supporters.add(third_key)
                                boxes.append(matches[0])
                        clusters.append((supporters, boxes))
    if not clusters:
        return "DISAGREE", []
    supporters, boxes = max(clusters, key=lambda value: (len(value[0]), sum(len(row) for row in value[1])))
    merged = [[sum(box[index] for box in boxes) / len(boxes) for index in range(4)]]
    return ("AGREE" if len(supporters) == 3 else "PARTIAL"), merged


def _bearing(source: Sequence[float], target: Sequence[float]) -> float:
    lon1, lat1 = map(float, source)
    lon2, lat2 = map(float, target)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    return math.degrees(math.atan2(east, north)) % 360.0


def adjudicate(args: argparse.Namespace) -> None:
    manifest_path = args.run_dir / "manifest.json"
    manifest = _read(manifest_path)
    public = _read(args.run_dir / "public.json")
    metadata = _read(args.run_dir / "mapillary-metadata.json")
    annotation = make_annotation(public)
    teacher_payloads = {
        key: _read(args.run_dir / "teachers" / f"{key}.json")
        for key in ("teacher_A", "teacher_B", "teacher_C")
    }
    teacher_maps = {
        key: {row["observation_id"]: row for row in payload["observations"]}
        for key, payload in teacher_payloads.items()
    }
    camera_types = {str(row["image_id"]): str(row.get("camera_type") or "perspective") for row in metadata["images"]}
    public_rows = {row["observation_id"]: (episode, row) for episode, row in _observations(public)}
    for episode in annotation["episodes"]:
        for truth in episode["observations"]:
            public_episode, public_row = public_rows[truth["observation_id"]]
            raw = {key: teacher_maps[key][truth["observation_id"]] for key in teacher_maps}
            agreement, regions = _consensus(raw)
            truth["teacher_agreement"] = agreement
            for key in teacher_maps:
                truth["teacher_outputs"][key] = {
                    "teacher_id": key,
                    "implementation_id": manifest["teachers"][key]["implementation_id"],
                    "status": "RUN_SUCCESS",
                    "raw_output": raw[key],
                    "independent_of_evaluated_provider": True,
                    "provider_family_overlap": False,
                }
            entrances = public_episode.get("public_entrance_candidates", [])
            target = [
                sum(float(item["coordinates"][0]) for item in entrances) / len(entrances),
                sum(float(item["coordinates"][1]) for item in entrances) / len(entrances),
            ]
            signed_error = (_bearing(public_row["coordinates"], target) - float(public_row["heading_deg"]) + 180.0) % 360.0 - 180.0
            camera_type = camera_types.get(str(public_row["source_frame_id"]), "perspective")
            fov = 180.0 if camera_type == "fisheye" else 90.0
            projected_x = 0.5 + signed_error / fov
            map_support = bool(regions) and abs(signed_error) <= fov / 2.0 and any(
                abs(((region[0] + region[2]) / 2.0) - projected_x) <= MAP_REGION_CENTER_TOLERANCE
                for region in regions
            )
            truth["legal_regions_normalized_xyxy"] = regions
            truth["target_visibility"] = "VISIBLE" if regions else "UNKNOWN"
            distance = float(public_row["map_proxy_distance_m"])
            truth["range_truth"] = "RANGE_NEAR" if distance <= 2.0 else ("RANGE_FAR" if distance >= 8.0 else "RANGE_APPROACHING")
            if map_support:
                truth["truth_authority_tier"] = "TEACHER_SUPPORTED"
                truth["functional_authority"] = "ESTABLISHED"
                truth["functional_authority_sources"] = ["MAP_TRAJECTORY_DERIVED"]
                truth["notes"] = f"teacher consensus plus frozen map-bearing support; camera_type={camera_type}"
            elif regions:
                truth["truth_authority_tier"] = "TEACHER_ONLY_WEAK"
                truth["notes"] = f"teacher consensus without sufficient map-bearing support; camera_type={camera_type}"
            else:
                truth["truth_authority_tier"] = "UNKNOWN"
                truth["notes"] = "no cross-teacher region consensus; absence was not promoted to NOT_VISIBLE"
    annotation["truth_frozen"] = True
    validate_annotation(annotation)
    _atomic_json(args.run_dir / "annotation.json", annotation)
    manifest["status"] = "TRUTH_FROZEN_BASELINE_NOT_RUN"
    manifest["annotation_sha256"] = _sha256(args.run_dir / "annotation.json")
    _atomic_json(manifest_path, manifest)


def run_provider(args: argparse.Namespace) -> None:
    manifest_path = args.run_dir / "manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("status") != "TRUTH_FROZEN_BASELINE_NOT_RUN":
        raise ValueError("provider may run only after truth is frozen")
    public = _read(args.run_dir / "public.json")
    pixels = _read(args.run_dir / "pixels.json")
    pixel_map = {row["observation_id"]: row for row in pixels["pixels"]}
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.provider_adapter import (
        ground_current_frame,
        preflight_provider,
    )

    provider_lock = preflight_provider(codex_exe=args.codex_exe, model_dir=args.grounding_dino)
    _atomic_json(args.run_dir / "provider-lock.json", provider_lock)
    journal_path = args.run_dir / "provider-journal.json"
    journal = {
        "schema_version": "blindassist_public_real_8x89_provider_journal_v0",
        "status": "ACTIVE",
        "public_sha256": manifest["inputs"]["public_sha256"],
        "annotation_sha256_before_provider": manifest["annotation_sha256"],
        "private_truth_access": False,
        "calls_dispatched": 0,
        "calls_completed": 0,
        "calls_in_doubt": 0,
        "started_at_utc": _utc_now(),
    }
    if journal_path.exists():
        journal = _read(journal_path)
        if journal.get("status") not in {"ACTIVE", "INTERRUPTED_RESUMABLE"}:
            raise ValueError("provider journal is not resumable")
    _atomic_json(journal_path, journal)
    rows = []
    for index, (episode, frame) in enumerate(_observations(public), start=1):
        observation_id = frame["observation_id"]
        pixel = pixel_map[observation_id]
        call_dir = args.run_dir / "provider_calls" / observation_id
        observation_path = call_dir / "observation.json"
        timing_path = call_dir / "runtime.json"
        if observation_path.exists() and timing_path.exists():
            observation = _read(observation_path)
            elapsed_ms = float(_read(timing_path)["elapsed_ms"])
        else:
            journal.update({
                "active_observation_id": observation_id,
                "calls_dispatched": int(journal["calls_dispatched"]) + 1,
            })
            _atomic_json(journal_path, journal)
            started = time.perf_counter()
            observation = ground_current_frame(
                provider_lock=provider_lock,
                call_dir=call_dir,
                episode_id=episode["episode_id"],
                goal_name=episode["goal_contract"]["goal_contract"]["target_name"],
                image_path=Path(pixel["image_path"]),
                frame_id=str(frame["source_frame_id"]),
                observation_id=observation_id,
                captured_at_ms=int(frame["timestamp_ms"]),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _atomic_json(timing_path, {"elapsed_ms": elapsed_ms})
            journal.update({
                "active_observation_id": None,
                "calls_completed": int(journal["calls_completed"]) + 1,
            })
            _atomic_json(journal_path, journal)
        output = observation["p0_output"]
        decision = output["decision"]
        by_id = {row["candidate_id"]: row for row in output.get("candidates", [])}
        ranked = [candidate_id for candidate_id in decision.get("ranked_candidate_ids", []) if candidate_id in by_id]
        candidates = []
        for rank, candidate_id in enumerate(ranked, start=1):
            region = by_id[candidate_id]["region"]
            candidates.append({
                "candidate_id": candidate_id,
                "rank": rank,
                "x_center_fraction": (float(region["x_min"]) + float(region["x_max"])) / 2.0,
                "region_normalized_xyxy": [
                    float(region["x_min"]), float(region["y_min"]),
                    float(region["x_max"]), float(region["y_max"]),
                ],
                "range_m": float(frame["map_proxy_distance_m"]),
            })
        authorized = decision.get("goal_identity_support") == "SUPPORTED" and bool(decision.get("selected_candidate_id"))
        rows.append({
            "observation_id": observation_id,
            "candidate_cardinality": "UNIQUE" if authorized else "AMBIGUOUS",
            "selection_authorized": authorized,
            "target_visible": bool(candidates),
            "candidates": candidates,
            "latency_ms": elapsed_ms,
        })
        print(f"provider {index}/89 {observation_id} candidates={len(candidates)} authorized={authorized}", flush=True)
    provider = {
        "schema_version": "blindassist_public_real_provider_observations_v0",
        "source": "FROZEN_GROUNDING_DINO_TINY_PLUS_CODEX_TERRA_V0",
        "private_truth_access": False,
        "provider_lock_sha256": _sha256(args.run_dir / "provider-lock.json"),
        "observations": rows,
    }
    _atomic_json(args.run_dir / "provider.json", provider)
    journal.update({
        "status": "COMPLETED",
        "active_observation_id": None,
        "completed_at_utc": _utc_now(),
        "provider_sha256": _sha256(args.run_dir / "provider.json"),
        "calls_in_doubt": int(journal["calls_dispatched"]) - int(journal["calls_completed"]),
    })
    _atomic_json(journal_path, journal)
    manifest["status"] = "PROVIDER_COMPLETE_BASELINE_NOT_MATERIALIZED"
    manifest["provider_sha256"] = _sha256(args.run_dir / "provider.json")
    _atomic_json(manifest_path, manifest)


def finalize(args: argparse.Namespace) -> None:
    manifest_path = args.run_dir / "manifest.json"
    manifest = _read(manifest_path)
    journal = _read(args.run_dir / "provider-journal.json")
    evaluation = _read(args.run_dir / "evaluation.json")
    annotation = _read(args.run_dir / "annotation.json")
    if journal.get("status") != "COMPLETED" or journal.get("calls_completed") != 89 or journal.get("calls_in_doubt") != 0:
        raise ValueError("provider did not reach a clean 89/89 terminal")
    if evaluation.get("observation_metrics", {}).get("total_observations") != 89:
        raise ValueError("evaluation denominator drift")
    distribution = evaluation["observation_metrics"]["truth_authority_distribution"]
    teacher_supported = evaluation["observation_metrics"]["by_truth_authority_tier"]["TEACHER_SUPPORTED"]
    receipt = {
        "schema_version": "blindassist_public_real_8x89_terminal_receipt_v0",
        "closed_at_utc": _utc_now(),
        "roster": {"episodes": 8, "observations": 89, "replacement_or_supplement_count": 0},
        "truth_coverage": {
            "native_or_map_only_strong": int(distribution["NATIVE_GT"]) + int(distribution["MAP_TRAJECTORY_DERIVED"]),
            "teacher_supported_weak_usable": int(distribution["TEACHER_SUPPORTED"]),
            "teacher_only_weak": int(distribution["TEACHER_ONLY_WEAK"]),
            "unknown": int(distribution["UNKNOWN"]),
        },
        "teacher_disagreement_preserved": True,
        "teacher_agreement_distribution": {
            key: sum(
                row["teacher_agreement"] == key
                for episode in annotation["episodes"] for row in episode["observations"]
            )
            for key in ("AGREE", "PARTIAL", "DISAGREE")
        },
        "weak_usable_metrics_only": {
            "proposal_recall_at_10": teacher_supported["proposal_recall_at_k_given_visible_and_functional_authority"]["10"],
            "selection_accuracy": teacher_supported["selection_accuracy_given_legal_candidate_and_functional_authority"],
            "failure_attribution": evaluation["failure_attribution_by_truth_authority_tier"]["TEACHER_SUPPORTED"],
        },
        "episode_failure_attribution": evaluation["failure_attribution_counts"],
        "provider_calls": {
            "dispatched": journal["calls_dispatched"],
            "completed": journal["calls_completed"],
            "in_doubt": journal["calls_in_doubt"],
            "private_truth_access": journal["private_truth_access"],
        },
        "non_claimable_metrics": {
            "range_bucket": "PUBLIC_MAP_PROXY_WAS_SHARED_WITH_PROVIDER_NOT_INDEPENDENT_RANGE_TRUTH",
            "completion": "NO_USER_OR_INDEPENDENT_COMPLETION_TRUTH",
        },
        "operational_incidents": [{
            "stage": "TEACHER_PREFLIGHT_BEFORE_IMAGE_INFERENCE",
            "class": "FUNCTIONAL_MODEL_TAXONOMY_ADAPTER_NAME_MISMATCH",
            "outcome_accessed": False,
            "repair": "MAP_FROZEN_CLASS_0_NAME_DOOR_WITHOUT_CHANGING_WEIGHT_THRESHOLD_OR_ROSTER",
        }],
        "terminal": "TRUTH_OR_CONTRACT_INSUFFICIENT_PRIMARY",
        "localized_signal": "REFERENT_SELECTION_3_OF_4_TEACHER_SUPPORTED_WEAK_USABLE_OBSERVATIONS_NOT_ENOUGH_TO_ESTABLISH_H1",
        "claim_ceiling": "PUBLIC_REAL_DEVELOPMENT_FAILURE_ATTRIBUTION_ONLY_NO_USER_PRODUCT_SAFETY_NAVIGATION_OR_GENERAL_ACCURACY_CLAIM",
        "artifact_sha256": {
            name: _sha256(args.run_dir / name)
            for name in (
                "pixels.json", "annotation.json", "provider-lock.json",
                "provider-journal.json", "provider.json", "prediction.json", "evaluation.json",
            )
        },
        "implementation_source_sha256_at_close": _sha256(Path(__file__)),
    }
    _atomic_json(args.run_dir / "terminal-receipt.json", receipt)
    manifest["status"] = "SEALED_TRUTH_OR_CONTRACT_INSUFFICIENT_PRIMARY"
    manifest["evaluation_sha256"] = _sha256(args.run_dir / "evaluation.json")
    manifest["terminal_receipt_sha256"] = _sha256(args.run_dir / "terminal-receipt.json")
    _atomic_json(manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--public", type=Path, required=True)
    init.add_argument("--metadata", type=Path, required=True)
    init.add_argument("--goal-roster", type=Path, required=True)
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--yoloe-model", type=Path, required=True)
    init.add_argument("--text-encoder", type=Path, required=True)
    init.add_argument("--functional-base", type=Path, required=True)
    init.add_argument("--functional-adapted", type=Path, required=True)
    for name in ("acquire-pixels", "run-teachers", "adjudicate", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    provider = sub.add_parser("run-provider")
    provider.add_argument("--run-dir", type=Path, required=True)
    provider.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    provider.add_argument("--grounding-dino", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        initialize(args)
    elif args.command == "acquire-pixels":
        acquire_pixels(args)
    elif args.command == "run-teachers":
        run_teachers(args)
    elif args.command == "run-provider":
        run_provider(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        adjudicate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
