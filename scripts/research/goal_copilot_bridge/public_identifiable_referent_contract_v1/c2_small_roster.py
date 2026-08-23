"""Freeze and materialize one small reference-image UNIQUE roster without algorithms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from PIL import Image

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import contract
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0 import run_sun3d_door_approach_v0 as sun3d


SCHEMA_VERSION = "blindassist_public_identifiable_referent_c2_small_roster_v1"
PROTOCOL_ID = "PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_V1"
EXCLUDED_SEQUENCE = "hotel_umd/maryland_hotel3"
EXPECTED_SOURCE_COUNT = 7
MIN_EPISODES = 5
MAX_EPISODES = 7
LATER_PER_EPISODE = 3
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
REFERENCE_MIN_AREA = 0.02
REFERENCE_MAX_AREA = 0.80
LATER_MIN_AREA = 0.01
DISTRACTOR_MIN_AREA = 0.005
MIN_SOURCE_FRAME_GAP = 30
MIN_TRANSLATION_M = 0.30
MIN_VIEW_RAY_DEG = 15.0
MAX_METADATA_GETS = 15
MAX_IMAGE_GETS = 28
MAX_IMAGE_BYTES = 20_000_000
STRUCTURAL_LABELS = {"wall", "floor", "ceiling"}
CLAIM_CEILING = (
    "PUBLIC_IDENTIFIABLE_REFERENCE_IMAGE_SMALL_ROSTER_MATERIALIZATION_ONLY_NO_"
    "IDENTITY_BASELINE_ALGORITHM_NAVIGATION_CONTROL_SAFETY_OR_PRODUCT_CLAIM"
)


class C2Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2Error(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _body_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("body_sha256", None)
    return contract.content_sha256(payload)


def _normalize_label(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip().lower())
    text = re.sub(r"\s*:\s*\d+\s*$", "", text)
    text = re.sub(r"\s+\d+\s*$", "", text)
    return text


def _parse_matrices(extrinsic_text: str) -> list[list[float]]:
    values = [float(value) for value in extrinsic_text.split()]
    matrices = [values[index : index + 12] for index in range(0, len(values), 12)]
    _require(bool(matrices) and len(matrices[-1]) == 12, "SUN3D extrinsics are malformed")
    return matrices


def _clipped_bbox(polygon: Mapping[str, Any]) -> tuple[list[float], float] | None:
    if len(polygon.get("x", [])) < 3 or len(polygon.get("y", [])) < 3:
        return None
    x0 = max(0.0, min(float(value) for value in polygon["x"]))
    y0 = max(0.0, min(float(value) for value in polygon["y"]))
    x1 = min(float(IMAGE_WIDTH), max(float(value) for value in polygon["x"]))
    y1 = min(float(IMAGE_HEIGHT), max(float(value) for value in polygon["y"]))
    if x1 <= x0 or y1 <= y0:
        return None
    bbox = [x0 / IMAGE_WIDTH, y0 / IMAGE_HEIGHT, x1 / IMAGE_WIDTH, y1 / IMAGE_HEIGHT]
    return bbox, (x1 - x0) * (y1 - y0) / (IMAGE_WIDTH * IMAGE_HEIGHT)


def _camera_center(matrix: Sequence[float]) -> tuple[float, float, float]:
    return float(matrix[3]), float(matrix[7]), float(matrix[11])


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def _view_ray_angle(camera_a: Sequence[float], camera_b: Sequence[float], target: Sequence[float]) -> float:
    ray_a = [float(target[index]) - float(camera_a[index]) for index in range(3)]
    ray_b = [float(target[index]) - float(camera_b[index]) for index in range(3)]
    norm_a = math.sqrt(sum(value * value for value in ray_a))
    norm_b = math.sqrt(sum(value * value for value in ray_b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    cosine = sum(ray_a[index] * ray_b[index] for index in range(3)) / (norm_a * norm_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _valid_object_records(
    object_id: int, annotation: Mapping[str, Any], matrices: Sequence[Sequence[float]]
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    records: list[dict[str, Any]] = []
    world_points: list[list[float]] = []
    for frame_id, frame in enumerate(annotation["frames"]):
        candidates = []
        for polygon in frame.get("polygon", []):
            if int(polygon["object"]) != object_id:
                continue
            bbox = _clipped_bbox(polygon)
            center = sun3d._camera_centroid(polygon.get("XYZ", []))
            if bbox is not None and center is not None:
                candidates.append((bbox[1], polygon, bbox, center))
        if not candidates:
            continue
        _, polygon, (bbox_xyxy, area), center = max(candidates, key=lambda item: item[0])
        source_frame_id = int(annotation["fileList"][frame_id].split("-", 1)[0]) - 1
        _require(0 <= source_frame_id < len(matrices), "frame-to-extrinsics join is out of bounds")
        world = list(sun3d._world_point(matrices[source_frame_id], center))
        world_points.append(world)
        records.append(
            {
                "frame_id": frame_id,
                "source_frame_id": source_frame_id,
                "source_filename": annotation["fileList"][frame_id],
                "bbox_xyxy_normalized": bbox_xyxy,
                "bbox_area_fraction": area,
                "camera_center_world": list(_camera_center(matrices[source_frame_id])),
            }
        )
    return records, world_points


def _same_class_distractors(
    frame: Mapping[str, Any], annotation: Mapping[str, Any], target_id: int, normalized_label: str
) -> list[int]:
    found = set()
    for polygon in frame.get("polygon", []):
        object_id = int(polygon["object"])
        if object_id == target_id or object_id >= len(annotation["objects"]):
            continue
        item = annotation["objects"][object_id]
        bbox = _clipped_bbox(polygon)
        if item is not None and _normalize_label(item["name"]) == normalized_label and bbox is not None and bbox[1] >= DISTRACTOR_MIN_AREA:
            found.add(object_id)
    return sorted(found)


def _target_candidate(
    sequence: str, object_id: int, annotation: Mapping[str, Any], matrices: Sequence[Sequence[float]]
) -> dict[str, Any] | None:
    item = annotation["objects"][object_id]
    if item is None:
        return None
    label = _normalize_label(item["name"])
    if not label or label in STRUCTURAL_LABELS or label.startswith("wall"):
        return None
    records, world_points = _valid_object_records(object_id, annotation, matrices)
    if len(records) < 4:
        return None
    target_world = [statistics.median(point[axis] for point in world_points) for axis in range(3)]
    selected_reference = None
    selected_qualified: list[dict[str, Any]] = []
    for reference in records:
        if not REFERENCE_MIN_AREA <= reference["bbox_area_fraction"] <= REFERENCE_MAX_AREA:
            continue
        qualified = []
        for later in records:
            if later["frame_id"] <= reference["frame_id"] or later["bbox_area_fraction"] < LATER_MIN_AREA:
                continue
            if later["source_frame_id"] - reference["source_frame_id"] < MIN_SOURCE_FRAME_GAP:
                continue
            translation = _distance(reference["camera_center_world"], later["camera_center_world"])
            ray_angle = _view_ray_angle(reference["camera_center_world"], later["camera_center_world"], target_world)
            if translation < MIN_TRANSLATION_M and ray_angle < MIN_VIEW_RAY_DEG:
                continue
            qualified.append({**later, "translation_from_reference_m": translation, "view_ray_angle_deg": ray_angle})
        if len(qualified) >= LATER_PER_EPISODE:
            selected_reference = reference
            selected_qualified = qualified
            break
    if selected_reference is None:
        return None
    indices = [0, len(selected_qualified) // 2, len(selected_qualified) - 1]
    selected_later = [dict(selected_qualified[index]) for index in indices]
    for later in selected_later:
        later["same_class_distractor_object_ids"] = _same_class_distractors(
            annotation["frames"][later["frame_id"]], annotation, object_id, label
        )
    distractor_frames = sum(bool(item["same_class_distractor_object_ids"]) for item in selected_later)
    private_instance_id = f"sun3d:{sequence}:object:{object_id}"
    return {
        "sequence": sequence,
        "native_object_id": object_id,
        "private_physical_instance_id": private_instance_id,
        "private_normalized_label": label,
        "target_world_xyz_map_derived": target_world,
        "reference": selected_reference,
        "later_observations": selected_later,
        "qualifying_later_view_count": len(selected_qualified),
        "selected_later_distractor_frame_count": distractor_frames,
    }


def _episode_from_metadata(
    sequence: str, annotation: Mapping[str, Any], extrinsic_text: str
) -> dict[str, Any] | None:
    matrices = _parse_matrices(extrinsic_text)
    candidates = [
        candidate
        for object_id in range(len(annotation["objects"]))
        if (candidate := _target_candidate(sequence, object_id, annotation, matrices)) is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item["selected_later_distractor_frame_count"],
            -item["qualifying_later_view_count"],
            item["private_normalized_label"],
            item["native_object_id"],
        )
    )
    return candidates[0]


def _implementation_lock(protocol_path: Path) -> dict[str, str]:
    runner_path = Path(__file__).resolve()
    contract_path = Path(contract.__file__).resolve()
    schema_path = contract_path.with_name("public_identifiable_referent_contract_v1.schema.json")
    return {
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256_file(protocol_path),
        "runner_path": str(runner_path),
        "runner_sha256": _sha256_file(runner_path),
        "contract_path": str(contract_path),
        "contract_sha256": _sha256_file(contract_path),
        "schema_path": str(schema_path),
        "schema_sha256": _sha256_file(schema_path),
    }


def freeze(protocol_path: Path, output: Path) -> dict[str, Any]:
    _require(not output.exists(), "frozen roster output already exists")
    protocol_path = protocol_path.resolve()
    list_bytes = sun3d._fetch(sun3d.OFFICIAL_LIST_URL)
    sequences = sun3d._fully_annotated_sequences(list_bytes.decode("utf-8"))
    sources = [sequence for sequence in sequences if sequence != EXCLUDED_SEQUENCE]
    _require(len(sources) == EXPECTED_SOURCE_COUNT, "official source count drift")
    episodes = []
    dispositions = []
    metadata_gets = 1
    for sequence in sources:
        base = f"{sun3d.DATA_ROOT}/{sequence}"
        annotation_bytes = sun3d._fetch(f"{base}/annotation/index.json")
        annotation = json.loads(annotation_bytes)
        extrinsics_name = annotation["extrinsics"]
        extrinsic_bytes = sun3d._fetch(f"{base}/extrinsics/{extrinsics_name}")
        metadata_gets += 2
        episode = _episode_from_metadata(sequence, annotation, extrinsic_bytes.decode("utf-8"))
        if episode is None:
            dispositions.append({"sequence": sequence, "status": "NO_QUALIFYING_REFERENCE_INSTANCE"})
            continue
        case_id = f"c2-ref-{len(episodes) + 1:03d}"
        episode["case_id"] = case_id
        episode["source"] = {
            "annotation_url": f"{base}/annotation/index.json",
            "annotation_sha256": _sha256_bytes(annotation_bytes),
            "extrinsics_url": f"{base}/extrinsics/{extrinsics_name}",
            "extrinsics_sha256": _sha256_bytes(extrinsic_bytes),
        }
        episode["reference"]["image_url"] = f"{base}/image/{episode['reference']['source_filename']}"
        for index, later in enumerate(episode["later_observations"], start=1):
            later["observation_id"] = f"{case_id}-later-{index:02d}"
            later["image_url"] = f"{base}/image/{later['source_filename']}"
        episode["private_identity_source_sha256"] = contract.content_sha256(
            {
                "sequence": sequence,
                "native_object_id": episode["native_object_id"],
                "target_world_xyz_map_derived": episode["target_world_xyz_map_derived"],
                "annotation_sha256": episode["source"]["annotation_sha256"],
                "extrinsics_sha256": episode["source"]["extrinsics_sha256"],
            }
        )
        episodes.append(episode)
        dispositions.append({"sequence": sequence, "status": "QUALIFIED", "case_id": case_id})
    _require(metadata_gets <= MAX_METADATA_GETS, "metadata GET budget exceeded")
    terminal = "C2_ROSTER_FROZEN_READY_FOR_ONE_MATERIALIZATION"
    if len(episodes) < MIN_EPISODES:
        terminal = "C2_NOT_EVALUABLE_METADATA_ROSTER_LT_5"
    roster = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "data_role": "FRESH_C2_MATERIALIZATION",
        "frozen_at_utc": _utc_now(),
        "official_list_url": sun3d.OFFICIAL_LIST_URL,
        "official_list_sha256": _sha256_bytes(list_bytes),
        "excluded_consumed_sequence": EXCLUDED_SEQUENCE,
        "source_count": len(sources),
        "source_dispositions": dispositions,
        "episodes": episodes,
        "episode_count": len(episodes),
        "image_gets_before_freeze": 0,
        "provider_calls": 0,
        "teacher_calls": 0,
        "baseline_calls": 0,
        "budgets": {
            "metadata_gets_max": MAX_METADATA_GETS,
            "metadata_gets_used": metadata_gets,
            "image_gets_max": MAX_IMAGE_GETS,
            "image_bytes_max": MAX_IMAGE_BYTES,
            "later_observations_per_episode": LATER_PER_EPISODE,
        },
        "selection_rule": (
            "ALL_7_UNCONSUMED_OFFICIAL_POSE_CORRECTED_SOURCES_ONE_EPISODE_EACH_"
            "REFERENCE_FIRST_SUPPORTING_3_REAL_VIEW_CHANGES_TARGET_RANK_DISTRACTOR_THEN_VIEW_COUNT_THEN_LABEL_ID"
        ),
        "implementation_lock": _implementation_lock(protocol_path),
        "materialization_authorized": len(episodes) >= MIN_EPISODES,
        "passive_baseline_authorized": False,
        "algorithm_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": terminal,
    }
    roster["body_sha256"] = _body_hash(roster)
    _atomic_json(output, roster)
    return roster


def _verify_roster(roster: Mapping[str, Any], roster_path: Path, roster_file_sha256: str, protocol_path: Path) -> None:
    _require(_sha256_file(roster_path) == roster_file_sha256, "roster file SHA mismatch")
    _require(roster.get("body_sha256") == _body_hash(roster), "roster body hash mismatch")
    _require(roster.get("terminal") == "C2_ROSTER_FROZEN_READY_FOR_ONE_MATERIALIZATION", "roster is not materializable")
    _require(MIN_EPISODES <= roster.get("episode_count", 0) <= MAX_EPISODES, "roster episode count is outside 5--7")
    _require(roster.get("materialization_authorized") is True, "materialization is not authorized")
    _require(roster.get("passive_baseline_authorized") is False and roster.get("algorithm_authorized") is False, "execution authority drift")
    live_lock = _implementation_lock(protocol_path.resolve())
    _require(roster.get("implementation_lock") == live_lock, "implementation lock drift")


def _download_image(
    url: str,
    path: Path,
    role: str,
    journal: list[dict[str, Any]],
    journal_path: Path,
    counters: dict[str, int],
) -> tuple[str, int, int]:
    journal.append({"event": "dispatch", "url": url, "role": role, "at_utc": _utc_now()})
    _atomic_json(journal_path, journal)
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.content
    counters["image_gets"] += 1
    counters["image_bytes"] += len(payload)
    _require(counters["image_gets"] <= MAX_IMAGE_GETS, "image GET budget exceeded")
    _require(counters["image_bytes"] <= MAX_IMAGE_BYTES, "image byte budget exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    _require((width, height) == (IMAGE_WIDTH, IMAGE_HEIGHT), "SUN3D image dimensions drifted")
    sha256 = _sha256_bytes(payload)
    journal.append({"event": "complete", "url": url, "role": role, "sha256": sha256, "bytes": len(payload), "at_utc": _utc_now()})
    _atomic_json(journal_path, journal)
    return sha256, width, height


def _materialize_inner(roster: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    counters = {"image_gets": 0, "image_bytes": 0, "reference_gets": 0, "later_gets": 0}
    journal: list[dict[str, Any]] = []
    image_hashes: set[str] = set()
    public_episodes = []
    private_episodes = []
    contract_pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    journal_path = output_dir / "download-journal.json"

    for episode in roster["episodes"]:
        case_id = episode["case_id"]
        reference_path = output_dir / "reference-images" / case_id / "reference.jpg"
        sha256, width, height = _download_image(
            episode["reference"]["image_url"], reference_path, "REFERENCE", journal, journal_path, counters
        )
        counters["reference_gets"] += 1
        _require(sha256 not in image_hashes, "reference image hash is not unique")
        image_hashes.add(sha256)
        bundle = {
            "schema_version": contract.FREEZE_SCHEMA,
            "protocol_id": contract.PROTOCOL_ID,
            "contract_id": case_id,
            "frozen_at_utc": _utc_now(),
            "freeze_states": {
                "episode_observation_pixels": "NOT_CAPTURED",
                "provider_output": "NOT_CREATED",
                "candidate_output": "NOT_CREATED",
                "outcome_access": "NONE",
            },
            "public_goal": {
                "goal_text": "Find the same physical instance shown inside the reference region.",
                "modality": "REFERENCE_IMAGE_INSTANCE",
                "reference_mode": "UNIQUE",
                "reference_anchor_id": f"public-{case_id}",
                "reference_image": {
                    "reference_image_id": f"{case_id}-reference",
                    "image_sha256": sha256,
                    "width": width,
                    "height": height,
                    "target_selector": "PUBLIC_TARGET_REGION",
                    "public_target_region_xyxy": episode["reference"]["bbox_xyxy_normalized"],
                },
                "language_description": None,
            },
            "private_binding": {
                "binding_authority": "SOURCE_NATIVE_REFERENCE_LINK",
                "binding_created_at_utc": roster["frozen_at_utc"],
                "binding_created_before_episode_observations": True,
                "binding_created_before_provider_output": True,
                "model_or_teacher_used_for_binding": False,
                "source_record_sha256": episode["private_identity_source_sha256"],
                "bound_reference_image_sha256": sha256,
                "legal_physical_instance_ids": [episode["private_physical_instance_id"]],
                "world_anchors": [
                    {
                        "physical_instance_id": episode["private_physical_instance_id"],
                        "coordinate_frame_id": f"sun3d:{episode['sequence']}:corrected-world",
                        "position_xyz_m": episode["target_world_xyz_map_derived"],
                        "authority": "SUN3D_NATIVE_POLYGON_CORRECTED_EXTRINSICS_MAP_DERIVED",
                    }
                ],
            },
        }
        public_receipt, private_receipt = contract.freeze_contract(bundle)
        public_path = output_dir / "public-contracts" / f"{case_id}.json"
        private_path = output_dir / "private-identity-locks" / f"{case_id}.json"
        contract.atomic_json(public_path, public_receipt)
        contract.atomic_json(private_path, private_receipt)
        contract_pairs[case_id] = public_receipt, private_receipt
        public_episodes.append(
            {
                "case_id": case_id,
                "public_contract_relative_path": public_path.relative_to(output_dir).as_posix(),
                "public_contract_body_sha256": public_receipt["body_sha256"],
                "reference_image_relative_path": reference_path.relative_to(output_dir).as_posix(),
                "reference_image_sha256": sha256,
                "later_observations": [],
            }
        )
        private_episodes.append(
            {
                "case_id": case_id,
                "sequence": episode["sequence"],
                "private_identity_lock_relative_path": private_path.relative_to(output_dir).as_posix(),
                "private_identity_lock_body_sha256": private_receipt["body_sha256"],
                "selected_later_distractor_frame_count": episode["selected_later_distractor_frame_count"],
            }
        )
    _require(counters["later_gets"] == 0, "later observation was downloaded before identity-lock barrier")
    barrier = {
        "schema_version": "blindassist_c2_identity_lock_barrier_v1",
        "created_at_utc": _utc_now(),
        "episode_count": len(contract_pairs),
        "reference_image_gets": counters["reference_gets"],
        "later_image_gets": 0,
        "public_contract_body_sha256s": sorted(value[0]["body_sha256"] for value in contract_pairs.values()),
        "private_identity_lock_body_sha256s": sorted(value[1]["body_sha256"] for value in contract_pairs.values()),
    }
    barrier["body_sha256"] = _body_hash(barrier)
    _atomic_json(output_dir / "identity-lock-barrier.json", barrier)
    _atomic_json(journal_path, journal)

    public_by_id = {item["case_id"]: item for item in public_episodes}
    private_by_id = {item["case_id"]: item for item in private_episodes}
    for episode in roster["episodes"]:
        case_id = episode["case_id"]
        public_receipt, private_receipt = contract_pairs[case_id]
        truth_rows = []
        for later in episode["later_observations"]:
            later_path = output_dir / "later-images" / case_id / f"{later['observation_id']}.jpg"
            sha256, _, _ = _download_image(
                later["image_url"], later_path, "LATER_OBSERVATION", journal, journal_path, counters
            )
            counters["later_gets"] += 1
            _require(sha256 not in image_hashes, "image hash is duplicated across roles or episodes")
            image_hashes.add(sha256)
            public_by_id[case_id]["later_observations"].append(
                {
                    "observation_id": later["observation_id"],
                    "image_relative_path": later_path.relative_to(output_dir).as_posix(),
                    "image_sha256": sha256,
                }
            )
            truth_rows.append(
                {
                    "observation_id": later["observation_id"],
                    "frame_sha256": sha256,
                    "visibility": "VISIBLE",
                    "target_regions": [
                        {
                            "physical_instance_id": episode["private_physical_instance_id"],
                            "bbox_xyxy_normalized": later["bbox_xyxy_normalized"],
                        }
                    ],
                }
            )
        truth = {
            "schema_version": contract.TRUTH_SCHEMA,
            "contract_id": case_id,
            "public_contract_body_sha256": public_receipt["body_sha256"],
            "private_identity_lock_body_sha256": private_receipt["body_sha256"],
            "provider_access_to_truth": False,
            "truth_created_after_contract_freeze": True,
            "observations": truth_rows,
        }
        audit = contract.validate_observation_truth(public_receipt, private_receipt, truth)
        truth_path = output_dir / "private-truth" / f"{case_id}.json"
        audit_path = output_dir / "truth-audits" / f"{case_id}.json"
        contract.atomic_json(truth_path, truth)
        contract.atomic_json(audit_path, audit)
        private_by_id[case_id]["private_truth_relative_path"] = truth_path.relative_to(output_dir).as_posix()
        private_by_id[case_id]["truth_audit_relative_path"] = audit_path.relative_to(output_dir).as_posix()
        private_by_id[case_id]["truth_audit_body_sha256"] = audit["body_sha256"]
    _atomic_json(journal_path, journal)

    expected_later = len(roster["episodes"]) * LATER_PER_EPISODE
    _require(counters["later_gets"] == expected_later, "later image count mismatch")
    _require(len(image_hashes) == counters["image_gets"], "image hash uniqueness mismatch")
    _require(len({item["sequence"] for item in private_episodes}) == len(private_episodes), "episode sources are not disjoint")
    _require(all(len(item["later_observations"]) == LATER_PER_EPISODE for item in public_episodes), "public observation count mismatch")

    public_manifest = {
        "schema_version": "blindassist_c2_public_reference_roster_v1",
        "protocol_id": PROTOCOL_ID,
        "episode_count": len(public_episodes),
        "episodes": public_episodes,
        "provider_calls": 0,
        "baseline_authorized": False,
        "algorithm_authorized": False,
    }
    public_manifest["body_sha256"] = _body_hash(public_manifest)
    private_manifest = {
        "schema_version": "blindassist_c2_private_evidence_manifest_v1",
        "protocol_id": PROTOCOL_ID,
        "episodes": private_episodes,
        "identity_lock_barrier_body_sha256": barrier["body_sha256"],
    }
    private_manifest["body_sha256"] = _body_hash(private_manifest)
    _atomic_json(output_dir / "public-manifest.json", public_manifest)
    _atomic_json(output_dir / "private-evidence-manifest.json", private_manifest)

    report = {
        "schema_version": "blindassist_c2_small_roster_result_v1",
        "protocol_id": PROTOCOL_ID,
        "roster_body_sha256": roster["body_sha256"],
        "public_manifest_body_sha256": public_manifest["body_sha256"],
        "private_manifest_body_sha256": private_manifest["body_sha256"],
        "identity_lock_barrier_body_sha256": barrier["body_sha256"],
        "episode_count": len(public_episodes),
        "source_disjoint_episode_count": len({item["sequence"] for item in private_episodes}),
        "reference_image_unique_contract_count": len(public_episodes),
        "later_observation_count": expected_later,
        "real_viewpoint_gate_pass_count": expected_later,
        "truth_binding_pass_count": expected_later,
        "same_class_distractor_episode_count": sum(item["selected_later_distractor_frame_count"] > 0 for item in private_episodes),
        "image_gets": counters["image_gets"],
        "image_bytes": counters["image_bytes"],
        "unique_image_sha256_count": len(image_hashes),
        "provider_calls": 0,
        "teacher_calls": 0,
        "detector_calls": 0,
        "matcher_calls": 0,
        "baseline_calls": 0,
        "passive_baseline_authorized": False,
        "algorithm_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "SMALL_ROSTER_MATERIALIZABLE",
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(output_dir / "final-report.json", report)
    return report


def materialize(
    roster_path: Path, roster_file_sha256: str, protocol_path: Path, output_dir: Path
) -> dict[str, Any]:
    _require(not output_dir.exists(), "formal materialization root already exists")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    _verify_roster(roster, roster_path, roster_file_sha256, protocol_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _materialize_inner(roster, output_dir)
    except Exception as error:
        failure = {
            "schema_version": "blindassist_c2_small_roster_result_v1",
            "protocol_id": PROTOCOL_ID,
            "roster_body_sha256": roster["body_sha256"],
            "error_class": type(error).__name__,
            "error": str(error),
            "provider_calls": 0,
            "teacher_calls": 0,
            "baseline_calls": 0,
            "passive_baseline_authorized": False,
            "algorithm_authorized": False,
            "claim_ceiling": CLAIM_CEILING,
            "terminal": "C2_NOT_EVALUABLE_TRANSPORT_OR_MATERIALIZATION",
        }
        failure["body_sha256"] = _body_hash(failure)
        _atomic_json(output_dir / "final-report.json", failure)
        return failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--protocol", type=Path, required=True)
    materialize_parser.add_argument("--roster", type=Path, required=True)
    materialize_parser.add_argument("--roster-file-sha256", required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze(args.protocol, args.output)
    else:
        result = materialize(args.roster, args.roster_file_sha256, args.protocol, args.output_dir)
    print(json.dumps({"terminal": result["terminal"], "body_sha256": result["body_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
