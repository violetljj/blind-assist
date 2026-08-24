"""Fresh-source hard-error unary verifier probe over T-LESS/BOP19.

The protocol is intentionally two-stage. T-LESS metadata freezes a complete
candidate pool before RGB or model access. The unchanged DINOv2-S local scorer
then defines a baseline-hard stratum. Only after that stratum and matched
controls are sealed may the single predeclared PDM challenger be executed.

Public score inputs never contain the target slot, native object IDs, baseline
outcome, or evaluator stratum. T-LESS object IDs and slot mappings remain in a
private manifest until raw scores have been written and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_local_appearance_probe as local,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    oracle_competing_identity_probe as oracle,
)


SCHEMA_VERSION = "blindassist_pdm_hard_error_unary_probe_v0"
PROTOCOL_ID = "FRESH_SOURCE_DISJOINT_HARD_ERROR_UNARY_VERIFIER_V0"
MODE = "REVERSIBLE_EXPLORATION_SOURCE_NEW_DEVELOPMENT_STANDARD"

DATASET_REPO_ID = "bop-benchmark/tless"
DATASET_REVISION = "5fd309a04476a842d93abfb584fba9ee7caecdf1"
BASE_ARCHIVE = "tless_base.zip"
TRAIN_ARCHIVE = "tless_train_primesense.zip"
TEST_ARCHIVE = "tless_test_primesense_bop19.zip"
ARCHIVE_SHA256 = {
    BASE_ARCHIVE: "dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7",
    TRAIN_ARCHIVE: "7262fdf0e9de09cf5d051ca108d860c7c5fd563ee58496751e970458c6897ba",
    TEST_ARCHIVE: "1a18f6bbfb5ac4ced8529f7a35225adfed88c0f62ef38067933e2b541ef1d00",
}
ARCHIVE_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET_REPO_ID}/resolve/{DATASET_REVISION}/{{archive}}"
)

TEST_WIDTH = 1280
TEST_HEIGHT = 1024
TARGET_COUNT = 10
OBSERVATIONS_PER_TARGET = 3
EXPECTED_PAIR_COUNT = TARGET_COUNT * OBSERVATIONS_PER_TARGET
MIN_VISIBLE_FRACTION = 0.75
MIN_VISIBLE_AREA_FRACTION = 0.01
LOW_MARGIN_CORRECT_FRACTION = 0.10
MIN_HARD_PAIRS = 4
MIN_HARD_TARGETS = 3
CONTROL_RETENTION_GATE = 0.80
CLAIM_CEILING = (
    "SOURCE_NEW_TLESS_BOP19_ORACLE_CANDIDATE_UNARY_IDENTITY_RANKING_DEVELOPMENT_ONLY_"
    "NO_NATIVE_SAME_CLASS_CLAIM_NO_CANDIDATE_GENERATION_NONE_CALIBRATION_TRACKING_"
    "BELIEF_ACTIVE_SEARCH_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM"
)

FORBIDDEN_PUBLIC_TOKENS = (
    "target_slot",
    "distractor_slot",
    "target_native_object_id",
    "distractor_native_object_id",
    "physical_instance_id",
    "baseline_evaluation",
    "baseline_target_margin",
    "hard_error",
    "control",
    "stratum",
)


class PdmHardErrorProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PdmHardErrorProbeError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return local._sha256_file(path)


def _body_hash(value: Mapping[str, Any]) -> str:
    return local._body_hash(value)


def _atomic_json(path: Path, value: Any) -> None:
    local._atomic_json(path, value)


def _load_json(path: Path) -> Any:
    return local._load_json(path)


def _verify_body_hash(value: Mapping[str, Any], name: str) -> None:
    _require(value.get("body_sha256") == _body_hash(value), f"{name} body SHA mismatch")


def _assert_public_blind(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered_key = str(key).lower()
            for token in FORBIDDEN_PUBLIC_TOKENS:
                _require(token not in lowered_key, f"public score input leaks private field: {key}")
            _assert_public_blind(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public_blind(child)


def _archive_url(archive: str) -> str:
    _require(archive in ARCHIVE_SHA256, f"unlocked archive: {archive}")
    return ARCHIVE_URL.format(archive=archive)


def _json_from_zip(archive: Any, name: str) -> tuple[Any, str, int]:
    payload = archive.read(name)
    return json.loads(payload), _sha256_bytes(payload), len(payload)


def _visible_bbox(info: Mapping[str, Any]) -> tuple[list[int], float, float] | None:
    bbox = [int(value) for value in info["bbox_visib"]]
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None
    visible_fraction = float(info["visib_fract"])
    area_fraction = float(width * height) / float(TEST_WIDTH * TEST_HEIGHT)
    if visible_fraction < MIN_VISIBLE_FRACTION or area_fraction < MIN_VISIBLE_AREA_FRACTION:
        return None
    return [x, y, width, height], visible_fraction, area_fraction


def _frame_candidates(
    scene_id: int,
    image_id: int,
    gt_rows: Sequence[Mapping[str, Any]],
    info_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(len(gt_rows) == len(info_rows), "T-LESS gt/info row count mismatch")
    candidates = []
    for gt_index, (gt_row, info_row) in enumerate(zip(gt_rows, info_rows)):
        visible = _visible_bbox(info_row)
        if visible is None:
            continue
        bbox, visible_fraction, area_fraction = visible
        candidates.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "gt_index": gt_index,
                "native_object_id": int(gt_row["obj_id"]),
                "bbox_visib_xywh": bbox,
                "visible_fraction": visible_fraction,
                "visible_area_fraction": area_fraction,
            }
        )
    return candidates


def _spread_indices(length: int, count: int) -> list[int]:
    _require(length >= count >= 1, "cannot spread-select requested items")
    if count == 1:
        return [0]
    raw = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    _require(len(set(raw)) == count, "spread selection produced duplicate indices")
    return raw


def _target_slot(pair_index: int) -> str:
    return "A" if pair_index % 2 == 0 else "B"


def _select_reference(
    object_id: int,
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    info: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    candidates = []
    for image_key in sorted(gt, key=int):
        rows = gt[image_key]
        infos = info[image_key]
        _require(len(rows) == len(infos) == 1, "T-LESS train scene must isolate one object")
        _require(int(rows[0]["obj_id"]) == object_id, "T-LESS train object/scene mismatch")
        bbox = [int(value) for value in infos[0]["bbox_visib"]]
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        candidates.append(
            {
                "scene_id": object_id,
                "image_id": int(image_key),
                "gt_index": 0,
                "bbox_visib_xywh": bbox,
                "visible_fraction": float(infos[0]["visib_fract"]),
                "bbox_area_pixels": int(bbox[2] * bbox[3]),
            }
        )
    _require(candidates, f"no T-LESS train reference for object {object_id}")
    candidates.sort(
        key=lambda row: (
            -row["visible_fraction"],
            -row["bbox_area_pixels"],
            row["image_id"],
        )
    )
    return candidates[0]


def _choose_distractor(candidates: Sequence[Mapping[str, Any]], target_id: int) -> dict[str, Any]:
    others = [dict(row) for row in candidates if int(row["native_object_id"]) != target_id]
    _require(others, "competition frame lacks an eligible near-instance distractor")
    others.sort(
        key=lambda row: (
            -float(row["visible_area_fraction"]),
            -float(row["visible_fraction"]),
            int(row["native_object_id"]),
            int(row["gt_index"]),
        )
    )
    return others[0]


def _select_roster(
    test_scenes: Sequence[Mapping[str, Any]],
    train_by_object: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_object: dict[int, list[dict[str, Any]]] = {}
    qualified_frames: list[dict[str, Any]] = []
    for frame in test_scenes:
        candidates = _frame_candidates(
            int(frame["scene_id"]),
            int(frame["image_id"]),
            frame["gt_rows"],
            frame["info_rows"],
        )
        if len(candidates) < 2:
            continue
        qualified = {
            "scene_id": int(frame["scene_id"]),
            "image_id": int(frame["image_id"]),
            "candidates": candidates,
        }
        qualified_frames.append(qualified)
        for target in candidates:
            by_object.setdefault(int(target["native_object_id"]), []).append(
                {**qualified, "target": dict(target)}
            )
    ranked_objects = sorted(
        by_object,
        key=lambda object_id: (-len(by_object[object_id]), object_id),
    )
    _require(len(ranked_objects) >= TARGET_COUNT, "insufficient T-LESS target objects")
    used_competition_frames: set[tuple[int, int]] = set()
    episodes = []
    pair_index = 0
    for object_id in ranked_objects:
        if len(episodes) >= TARGET_COUNT:
            break
        eligible = [
            row
            for row in sorted(
                by_object[object_id], key=lambda value: (value["scene_id"], value["image_id"])
            )
            if (row["scene_id"], row["image_id"]) not in used_competition_frames
        ]
        if len(eligible) < OBSERVATIONS_PER_TARGET:
            continue
        selected = [eligible[index] for index in _spread_indices(len(eligible), OBSERVATIONS_PER_TARGET)]
        reference = dict(train_by_object[object_id])
        observations = []
        for observation_index, row in enumerate(selected, start=1):
            frame_key = (row["scene_id"], row["image_id"])
            _require(frame_key not in used_competition_frames, "competition frame reused")
            used_competition_frames.add(frame_key)
            distractor = _choose_distractor(row["candidates"], object_id)
            target_slot = _target_slot(pair_index)
            pair_index += 1
            observations.append(
                {
                    "pair_id": f"tless-{len(episodes) + 1:03d}-pair-{observation_index:02d}",
                    "scene_id": row["scene_id"],
                    "image_id": row["image_id"],
                    "target_slot": target_slot,
                    "distractor_slot": "B" if target_slot == "A" else "A",
                    "target": dict(row["target"]),
                    "distractor": distractor,
                }
            )
        episodes.append(
            {
                "case_id": f"tless-{len(episodes) + 1:03d}",
                "private_physical_instance_id": f"tless:object:{object_id:06d}",
                "native_object_id": object_id,
                "reference": reference,
                "eligible_competition_frame_count": len(by_object[object_id]),
                "observations": observations,
            }
        )
    _require(len(episodes) == TARGET_COUNT, "complete T-LESS target roster was not found")

    used_absence_frames: set[tuple[int, int]] = set()
    target_ids = {int(episode["native_object_id"]) for episode in episodes}
    for episode in episodes:
        target_id = int(episode["native_object_id"])
        absent = []
        for frame in qualified_frames:
            frame_key = (frame["scene_id"], frame["image_id"])
            present_ids = {int(row["native_object_id"]) for row in frame["candidates"]}
            if (
                target_id in present_ids
                or frame_key in used_competition_frames
                or frame_key in used_absence_frames
            ):
                continue
            candidate = sorted(
                frame["candidates"],
                key=lambda row: (
                    -float(row["visible_area_fraction"]),
                    int(row["native_object_id"]),
                    int(row["gt_index"]),
                ),
            )[0]
            absent.append((frame_key, candidate))
        _require(absent, f"no target-absent candidate for T-LESS object {target_id}")
        frame_key, candidate = absent[0]
        used_absence_frames.add(frame_key)
        episode["absence"] = {
            "absence_id": f"{episode['case_id']}-absence-01",
            "scene_id": frame_key[0],
            "image_id": frame_key[1],
            "candidate": dict(candidate),
            "target_native_object_id": target_id,
            "target_present": False,
        }
    dispositions = [
        {
            "native_object_id": object_id,
            "eligible_competition_frame_count": len(by_object[object_id]),
            "status": (
                "SELECTED"
                if object_id in target_ids
                else "NOT_SELECTED_AFTER_DETERMINISTIC_CAP"
            ),
        }
        for object_id in ranked_objects
    ]
    return episodes, dispositions


def _remote_zip_class() -> Any:
    try:
        from remotezip import RemoteZip
    except ImportError as error:  # pragma: no cover - environment preflight
        raise PdmHardErrorProbeError("remotezip runtime dependency is unavailable") from error
    return RemoteZip


def freeze(output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), f"frozen roster already exists: {output_path}")
    RemoteZip = _remote_zip_class()
    metadata_receipts: list[dict[str, Any]] = []
    with RemoteZip(_archive_url(BASE_ARCHIVE)) as archive:
        camera, digest, size = _json_from_zip(archive, "tless/camera_primesense.json")
        metadata_receipts.append(
            {"archive": BASE_ARCHIVE, "member": "tless/camera_primesense.json", "sha256": digest, "bytes": size}
        )
    _require(int(camera["width"]) == TEST_WIDTH and int(camera["height"]) == TEST_HEIGHT, "T-LESS camera size drifted")

    train_by_object: dict[int, dict[str, Any]] = {}
    with RemoteZip(_archive_url(TRAIN_ARCHIVE)) as archive:
        names = set(archive.namelist())
        for object_id in range(1, 31):
            prefix = f"train_primesense/{object_id:06d}"
            gt_name = f"{prefix}/scene_gt.json"
            info_name = f"{prefix}/scene_gt_info.json"
            _require(gt_name in names and info_name in names, "T-LESS train metadata member missing")
            gt, gt_sha, gt_size = _json_from_zip(archive, gt_name)
            info, info_sha, info_size = _json_from_zip(archive, info_name)
            metadata_receipts.extend(
                [
                    {"archive": TRAIN_ARCHIVE, "member": gt_name, "sha256": gt_sha, "bytes": gt_size},
                    {"archive": TRAIN_ARCHIVE, "member": info_name, "sha256": info_sha, "bytes": info_size},
                ]
            )
            train_by_object[object_id] = _select_reference(object_id, gt, info)

    test_frames = []
    with RemoteZip(_archive_url(TEST_ARCHIVE)) as archive:
        names = set(archive.namelist())
        for scene_id in range(1, 21):
            prefix = f"test_primesense/{scene_id:06d}"
            gt_name = f"{prefix}/scene_gt.json"
            info_name = f"{prefix}/scene_gt_info.json"
            _require(gt_name in names and info_name in names, "T-LESS test metadata member missing")
            gt, gt_sha, gt_size = _json_from_zip(archive, gt_name)
            info, info_sha, info_size = _json_from_zip(archive, info_name)
            metadata_receipts.extend(
                [
                    {"archive": TEST_ARCHIVE, "member": gt_name, "sha256": gt_sha, "bytes": gt_size},
                    {"archive": TEST_ARCHIVE, "member": info_name, "sha256": info_sha, "bytes": info_size},
                ]
            )
            _require(set(gt) == set(info), "T-LESS test gt/info image keys drifted")
            test_frames.extend(
                {
                    "scene_id": scene_id,
                    "image_id": int(image_key),
                    "gt_rows": gt[image_key],
                    "info_rows": info[image_key],
                }
                for image_key in sorted(gt, key=int)
            )

    episodes, dispositions = _select_roster(test_frames, train_by_object)
    roster = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "frozen_at_utc": _utc_now(),
        "data_role": "SOURCE_NEW_TLESS_BOP19_DEVELOPMENT_NATIVE_INSTANCE_AND_VISIBLE_MASK_TRUTH",
        "dataset_lock": {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_urls": {name: _archive_url(name) for name in ARCHIVE_SHA256},
        },
        "selection_contract": {
            "selected_before_rgb_access": True,
            "selected_before_baseline_scores": True,
            "selected_before_challenger_scores": True,
            "test_camera_size": [TEST_WIDTH, TEST_HEIGHT],
            "target_count": TARGET_COUNT,
            "observations_per_target": OBSERVATIONS_PER_TARGET,
            "minimum_visible_fraction": MIN_VISIBLE_FRACTION,
            "minimum_visible_area_fraction": MIN_VISIBLE_AREA_FRACTION,
            "target_order": "MOST_ELIGIBLE_COMPETITION_FRAMES_THEN_NATIVE_OBJECT_ID",
            "observation_selection": "EVENLY_SPACED_OVER_SCENE_ID_THEN_IMAGE_ID_WITH_GLOBAL_FRAME_UNIQUENESS",
            "distractor_selection": "LARGEST_VISIBLE_ELIGIBLE_OTHER_TLESS_INSTANCE_THEN_OBJECT_ID",
            "reference_selection": "MAX_VISIBLE_FRACTION_THEN_VISIBLE_BBOX_AREA_THEN_IMAGE_ID",
            "absence_selection": "FIRST_UNIQUE_QUALIFIED_FRAME_WITH_TARGET_OBJECT_ID_NOT_PRESENT",
            "candidate_truth_role": "ORACLE_NATIVE_TLESS_VISIBLE_BBOX",
            "native_same_class_claim": False,
            "near_instance_family": "TLESS_TEXTURELESS_INDUSTRIAL_OBJECTS",
        },
        "baseline_and_challenger_contract": {
            "baseline": "EXACT_EXISTING_DINOV2_V0_SYMMETRIC_LOCAL_SCORE",
            "challenger": "OFFICIAL_PDM_PERMIR_RETRIEVAL_SCORE_MINIMAL_RUNTIME_ADAPTER",
            "low_margin_correct_rule": "BOTTOM_CEIL_10_PERCENT_OF_ALL_BASELINE_PAIRS_AMONG_CORRECT_ROWS",
            "hard_rule": "ALL_BASELINE_NON_TARGET_ROWS_UNION_PREDECLARED_LOW_MARGIN_CORRECT_ROWS",
            "control_rule": "EQUAL_COUNT_REMAINING_CORRECT_ROWS_LOWEST_MARGIN_FIRST",
            "minimum_hard_pairs": MIN_HARD_PAIRS,
            "minimum_distinct_hard_targets": MIN_HARD_TARGETS,
            "success_gate": "RESCUE_GT_COLLATERAL_AND_CONTROL_RETENTION_GE_0P80",
            "threshold": None,
            "training": False,
            "fusion": False,
            "sweep": False,
        },
        "metadata_receipts": metadata_receipts,
        "source_dispositions": dispositions,
        "episode_count": len(episodes),
        "pair_count": sum(len(episode["observations"]) for episode in episodes),
        "absence_count": len(episodes),
        "episodes": episodes,
        "budgets": {
            "baseline_model_runs": 1,
            "challenger_model_runs": 1,
            "challenger_variants": 1,
            "threshold_or_score_sweeps": 0,
        },
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "TLESS_HARD_ERROR_ROSTER_FROZEN_READY",
    }
    _require(roster["pair_count"] == EXPECTED_PAIR_COUNT, "frozen T-LESS pair count drifted")
    roster["body_sha256"] = _body_hash(roster)
    _atomic_json(output_path, roster)
    return roster


def _member_paths(split: str, row: Mapping[str, Any]) -> tuple[str, str]:
    prefix = f"{split}/{int(row['scene_id']):06d}"
    image_id = int(row["image_id"])
    gt_index = int(row["gt_index"])
    return (
        f"{prefix}/rgb/{image_id:06d}.png",
        f"{prefix}/mask_visib/{image_id:06d}_{gt_index:06d}.png",
    )


def _extract_member(archive: Any, member: str, output_path: Path) -> dict[str, Any]:
    payload = archive.read(member)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    with Image.open(output_path) as image:
        width, height = image.size
        image.verify()
    return {
        "member": member,
        "relative_path": output_path.name,
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "width": width,
        "height": height,
    }


def _xywh_to_normalized(bbox: Sequence[int], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = [float(value) for value in bbox]
    _require(width > 0 and height > 0 and box_width > 0 and box_height > 0, "invalid bbox/image size")
    result = [x / width, y / height, (x + box_width) / width, (y + box_height) / height]
    _require(all(0.0 <= value <= 1.0 for value in result), "bbox falls outside image")
    return result


def _prior_image_hashes(search_root: Path) -> set[str]:
    hashes = set()
    patterns = (
        "public-identifiable-referent-c2-v1/formal/public-manifest.json",
        "public-identifiable-referent-dinov2-two-reference-matched-v0/materialized/public-manifest.json",
    )
    for relative in patterns:
        path = search_root / relative
        if not path.exists():
            continue
        payload = _load_json(path)
        for episode in payload.get("episodes", []):
            if "reference" in episode:
                hashes.add(episode["reference"]["image_sha256"])
            for reference in episode.get("references", []):
                hashes.add(reference["image_sha256"])
            for observation in episode.get("later_observations", episode.get("observations", [])):
                hashes.add(observation["image_sha256"])
    return hashes


def materialize(roster_path: Path, output_dir: Path, prior_evidence_root: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"materialization directory already exists: {output_dir}")
    roster = _load_json(roster_path)
    _verify_body_hash(roster, "T-LESS frozen roster")
    _require(roster["terminal"] == "TLESS_HARD_ERROR_ROSTER_FROZEN_READY", "roster is not executable")
    output_dir.mkdir(parents=True, exist_ok=False)
    RemoteZip = _remote_zip_class()
    prior_hashes = _prior_image_hashes(prior_evidence_root)
    seen_hashes: set[str] = set()
    public_episodes = []
    private_episodes = []
    receipts = []
    try:
        with RemoteZip(_archive_url(TRAIN_ARCHIVE)) as train_zip, RemoteZip(
            _archive_url(TEST_ARCHIVE)
        ) as test_zip:
            for episode in roster["episodes"]:
                case_id = episode["case_id"]
                reference_image_member, reference_mask_member = _member_paths(
                    "train_primesense", episode["reference"]
                )
                reference_dir = output_dir / "references" / case_id
                reference_image_path = reference_dir / "image.png"
                reference_mask_path = reference_dir / "mask_visib.png"
                image_receipt = _extract_member(train_zip, reference_image_member, reference_image_path)
                mask_receipt = _extract_member(train_zip, reference_mask_member, reference_mask_path)
                for receipt in (image_receipt, mask_receipt):
                    receipt.update({"case_id": case_id, "role": "REFERENCE"})
                    receipts.append(receipt)
                _require(image_receipt["sha256"] not in prior_hashes, "reference overlaps prior image hash")
                _require(image_receipt["sha256"] not in seen_hashes, "reference image hash duplicated")
                seen_hashes.add(image_receipt["sha256"])
                reference_bbox = _xywh_to_normalized(
                    episode["reference"]["bbox_visib_xywh"],
                    image_receipt["width"],
                    image_receipt["height"],
                )
                public_episode = {
                    "case_id": case_id,
                    "reference": {
                        "image_relative_path": reference_image_path.relative_to(output_dir).as_posix(),
                        "image_sha256": image_receipt["sha256"],
                        "mask_relative_path": reference_mask_path.relative_to(output_dir).as_posix(),
                        "mask_sha256": mask_receipt["sha256"],
                        "width": image_receipt["width"],
                        "height": image_receipt["height"],
                        "object_region_xyxy_normalized": reference_bbox,
                    },
                    "pairs": [],
                    "absence_candidates": [],
                }
                private_episode = {
                    "case_id": case_id,
                    "private_physical_instance_id": episode["private_physical_instance_id"],
                    "target_native_object_id": episode["native_object_id"],
                    "pairs": [],
                    "absence_candidates": [],
                }
                for observation in episode["observations"]:
                    image_member, _ = _member_paths("test_primesense", observation["target"])
                    pair_dir = output_dir / "pairs" / observation["pair_id"]
                    image_path = pair_dir / "image.png"
                    receipt = _extract_member(test_zip, image_member, image_path)
                    receipt.update({"case_id": case_id, "pair_id": observation["pair_id"], "role": "COMPETITION"})
                    receipts.append(receipt)
                    _require(receipt["sha256"] not in prior_hashes, "competition overlaps prior image hash")
                    _require(receipt["sha256"] not in seen_hashes, "competition image hash duplicated")
                    seen_hashes.add(receipt["sha256"])
                    candidate_regions = {
                        observation["target_slot"]: _xywh_to_normalized(
                            observation["target"]["bbox_visib_xywh"], receipt["width"], receipt["height"]
                        ),
                        observation["distractor_slot"]: _xywh_to_normalized(
                            observation["distractor"]["bbox_visib_xywh"], receipt["width"], receipt["height"]
                        ),
                    }
                    public_episode["pairs"].append(
                        {
                            "pair_id": observation["pair_id"],
                            "image_relative_path": image_path.relative_to(output_dir).as_posix(),
                            "image_sha256": receipt["sha256"],
                            "width": receipt["width"],
                            "height": receipt["height"],
                            "candidate_regions_xyxy_normalized": candidate_regions,
                        }
                    )
                    private_episode["pairs"].append(
                        {
                            "pair_id": observation["pair_id"],
                            "target_slot": observation["target_slot"],
                            "distractor_slot": observation["distractor_slot"],
                            "target_native_object_id": observation["target"]["native_object_id"],
                            "distractor_native_object_id": observation["distractor"]["native_object_id"],
                        }
                    )
                absence = episode["absence"]
                absence_image_member, _ = _member_paths("test_primesense", absence["candidate"])
                absence_dir = output_dir / "absence" / absence["absence_id"]
                absence_image_path = absence_dir / "image.png"
                receipt = _extract_member(test_zip, absence_image_member, absence_image_path)
                receipt.update({"case_id": case_id, "absence_id": absence["absence_id"], "role": "ABSENCE"})
                receipts.append(receipt)
                _require(receipt["sha256"] not in prior_hashes, "absence overlaps prior image hash")
                _require(receipt["sha256"] not in seen_hashes, "absence image hash duplicated")
                seen_hashes.add(receipt["sha256"])
                public_episode["absence_candidates"].append(
                    {
                        "absence_id": absence["absence_id"],
                        "image_relative_path": absence_image_path.relative_to(output_dir).as_posix(),
                        "image_sha256": receipt["sha256"],
                        "width": receipt["width"],
                        "height": receipt["height"],
                        "candidate_region_xyxy_normalized": _xywh_to_normalized(
                            absence["candidate"]["bbox_visib_xywh"], receipt["width"], receipt["height"]
                        ),
                    }
                )
                private_episode["absence_candidates"].append(
                    {
                        "absence_id": absence["absence_id"],
                        "target_present": False,
                        "target_native_object_id": absence["target_native_object_id"],
                        "candidate_native_object_id": absence["candidate"]["native_object_id"],
                    }
                )
                public_episodes.append(public_episode)
                private_episodes.append(private_episode)
        public = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "mode": MODE,
            "roster_body_sha256": roster["body_sha256"],
            "episode_count": len(public_episodes),
            "pair_count": sum(len(episode["pairs"]) for episode in public_episodes),
            "absence_count": sum(len(episode["absence_candidates"]) for episode in public_episodes),
            "episodes": public_episodes,
            "claim_ceiling": CLAIM_CEILING,
        }
        _assert_public_blind(public)
        public["body_sha256"] = _body_hash(public)
        private = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "roster_body_sha256": roster["body_sha256"],
            "public_manifest_body_sha256": public["body_sha256"],
            "episodes": private_episodes,
            "claim_ceiling": CLAIM_CEILING,
        }
        private["body_sha256"] = _body_hash(private)
        _atomic_json(output_dir / "public-manifest.json", public)
        _atomic_json(output_dir / "private-evidence-manifest.json", private)
        report = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": _utc_now(),
            "roster_body_sha256": roster["body_sha256"],
            "public_manifest_body_sha256": public["body_sha256"],
            "private_manifest_body_sha256": private["body_sha256"],
            "unique_image_count": len(seen_hashes),
            "prior_image_hash_overlap_count": 0,
            "receipts": receipts,
            "terminal": "TLESS_HARD_ERROR_MATERIALIZATION_COMPLETE",
            "claim_ceiling": CLAIM_CEILING,
        }
        report["body_sha256"] = _body_hash(report)
        _atomic_json(output_dir / "materialization-report.json", report)
        return report
    except Exception as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": _utc_now(),
            "error_class": type(error).__name__,
            "error": str(error),
            "model_calls": 0,
            "terminal": "TLESS_HARD_ERROR_NOT_EVALUABLE_MATERIALIZATION",
            "claim_ceiling": CLAIM_CEILING,
        }
        failure["body_sha256"] = _body_hash(failure)
        _atomic_json(output_dir / "materialization-report.json", failure)
        raise


def _crop_contract(image_path: Path, bbox: Sequence[float]) -> dict[str, Any]:
    crop = oracle._square_crop_bounds(bbox)
    return {
        "image_path": str(image_path.resolve()),
        "image_sha256": _sha256_file(image_path),
        "crop_bbox_xyxy_normalized": crop,
        "object_bbox_within_crop_xyxy_normalized": local._relative_bbox(bbox, crop),
    }


def prepare_baseline(public_manifest_path: Path, run_dir: Path, model: Mapping[str, Any]) -> dict[str, Any]:
    _require(not run_dir.exists(), f"baseline run directory already exists: {run_dir}")
    public = _load_json(public_manifest_path)
    _verify_body_hash(public, "T-LESS public manifest")
    _assert_public_blind(public)
    root = public_manifest_path.parent
    pairs = []
    for episode in public["episodes"]:
        reference = episode["reference"]
        reference_path = root / reference["image_relative_path"]
        _require(_sha256_file(reference_path) == reference["image_sha256"], "reference image SHA drifted")
        reference_crop = _crop_contract(reference_path, reference["object_region_xyxy_normalized"])
        for pair in episode["pairs"]:
            image_path = root / pair["image_relative_path"]
            _require(_sha256_file(image_path) == pair["image_sha256"], "competition image SHA drifted")
            pairs.append(
                {
                    "pair_id": pair["pair_id"],
                    "case_id": episode["case_id"],
                    "reference": reference_crop,
                    "candidates": {
                        slot: _crop_contract(image_path, pair["candidate_regions_xyxy_normalized"][slot])
                        for slot in ("A", "B")
                    },
                }
            )
    _require(len(pairs) == EXPECTED_PAIR_COUNT, "baseline prepared pair count drifted")
    config = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "BASELINE",
        "created_at_utc": _utc_now(),
        "source_code_sha256": _sha256_file(Path(__file__)),
        "local_scorer_sha256": _sha256_file(Path(local.__file__)),
        "public_manifest_body_sha256": public["body_sha256"],
        "model": dict(model),
        "score_contract": "EXACT_V0_SYMMETRIC_LOCAL_SCORE_STRICT_GREATER_THAN_NO_THRESHOLD",
        "pairs": pairs,
        "claim_ceiling": CLAIM_CEILING,
    }
    _assert_public_blind(config)
    config["body_sha256"] = _body_hash(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "run-config.json", config)
    return config


def execute_baseline(model_dir: Path, run_dir: Path, device: str) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    _verify_body_hash(config, "baseline run config")
    _assert_public_blind(config)
    tensors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for pair in config["pairs"]:
        for item in (pair["reference"], pair["candidates"]["A"], pair["candidates"]["B"]):
            tensors.append(local._crop_tensor(Path(item["image_path"]), item["crop_bbox_xyxy_normalized"]))
            masks.append(local._patch_mask(item["object_bbox_within_crop_xyxy_normalized"]))
    encoder = local.DenseEncoder(model_dir, device)
    features = encoder.encode(tensors)
    rows = []
    offset = 0
    for pair in config["pairs"]:
        reference_feature, feature_a, feature_b = features[offset : offset + 3]
        reference_mask, mask_a, mask_b = masks[offset : offset + 3]
        offset += 3
        score_a = local.symmetric_local_score(reference_feature, feature_a, reference_mask, mask_a)
        score_b = local.symmetric_local_score(reference_feature, feature_b, reference_mask, mask_b)
        scalar_a = float(score_a["symmetric_score"])
        scalar_b = float(score_b["symmetric_score"])
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "candidate_scores": {"A": score_a, "B": score_b},
                "winner_slot": local._winner(scalar_a, scalar_b),
                "slot_margin_a_minus_b": scalar_a - scalar_b,
            }
        )
    _require(offset == len(features), "baseline encoded crop accounting drifted")
    raw = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "BASELINE",
        "created_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "encoded_crop_count": encoder.encoded_crops,
        "forward_batch_count": encoder.forward_batches,
        "rows": rows,
        "claim_ceiling": CLAIM_CEILING,
    }
    raw["body_sha256"] = _body_hash(raw)
    _atomic_json(run_dir / "raw-scores.json", raw)
    return raw


def _evaluation(target_score: float, distractor_score: float) -> str:
    if target_score > distractor_score:
        return "TARGET_OUTRANKS"
    if distractor_score > target_score:
        return "DISTRACTOR_OUTRANKS"
    return "TIE"


def evaluate_baseline(run_dir: Path, private_manifest_path: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    raw = _load_json(run_dir / "raw-scores.json")
    private = _load_json(private_manifest_path)
    for value, name in ((config, "baseline config"), (raw, "baseline raw"), (private, "private manifest")):
        _verify_body_hash(value, name)
    truth = {
        pair["pair_id"]: {**pair, "case_id": episode["case_id"]}
        for episode in private["episodes"]
        for pair in episode["pairs"]
    }
    _require(set(truth) == {row["pair_id"] for row in raw["rows"]}, "baseline raw/private pair mismatch")
    rows = []
    for row in raw["rows"]:
        private_row = truth[row["pair_id"]]
        target_slot = private_row["target_slot"]
        distractor_slot = private_row["distractor_slot"]
        target_score = float(row["candidate_scores"][target_slot]["symmetric_score"])
        distractor_score = float(row["candidate_scores"][distractor_slot]["symmetric_score"])
        rows.append(
            {
                **row,
                "target_slot": target_slot,
                "distractor_slot": distractor_slot,
                "target_native_object_id": private_row["target_native_object_id"],
                "distractor_native_object_id": private_row["distractor_native_object_id"],
                "target_score": target_score,
                "distractor_score": distractor_score,
                "target_margin": target_score - distractor_score,
                "evaluation": _evaluation(target_score, distractor_score),
            }
        )
    metrics = {
        "pair_count": len(rows),
        "distinct_target_count": len({row["case_id"] for row in rows}),
        "target_outranks_count": sum(row["evaluation"] == "TARGET_OUTRANKS" for row in rows),
        "distractor_outranks_count": sum(row["evaluation"] == "DISTRACTOR_OUTRANKS" for row in rows),
        "tie_count": sum(row["evaluation"] == "TIE" for row in rows),
        "margin_median": statistics.median(row["target_margin"] for row in rows),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "BASELINE",
        "evaluated_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "raw_scores_body_sha256": raw["body_sha256"],
        "private_manifest_body_sha256": private["body_sha256"],
        "model": config["model"],
        "metrics": metrics,
        "rows": rows,
        "protocol_status": "VALID",
        "terminal": "TLESS_DINOV2_BASELINE_COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _select_challenger_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    incorrect = [row for row in rows if row["evaluation"] != "TARGET_OUTRANKS"]
    correct = sorted(
        (row for row in rows if row["evaluation"] == "TARGET_OUTRANKS"),
        key=lambda row: (float(row["target_margin"]), row["pair_id"]),
    )
    low_margin_count = math.ceil(len(rows) * LOW_MARGIN_CORRECT_FRACTION)
    low_margin = correct[:low_margin_count]
    hard = sorted(incorrect + low_margin, key=lambda row: row["pair_id"])
    hard_ids = {row["pair_id"] for row in hard}
    remaining_correct = [row for row in correct if row["pair_id"] not in hard_ids]
    controls = remaining_correct[: len(hard)]
    _require(len(controls) == len(hard), "insufficient matched baseline-correct controls")
    return [row["pair_id"] for row in hard], [row["pair_id"] for row in controls]


def freeze_challenger(
    baseline_report_path: Path,
    public_manifest_path: Path,
    private_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _require(not output_dir.exists(), f"challenger freeze directory already exists: {output_dir}")
    baseline = _load_json(baseline_report_path)
    public = _load_json(public_manifest_path)
    private = _load_json(private_manifest_path)
    for value, name in ((baseline, "baseline report"), (public, "public manifest"), (private, "private manifest")):
        _verify_body_hash(value, name)
    _require(baseline["terminal"] == "TLESS_DINOV2_BASELINE_COMPLETE", "baseline is not terminal")
    hard_ids, control_ids = _select_challenger_rows(baseline["rows"])
    baseline_by_id = {row["pair_id"]: row for row in baseline["rows"]}
    hard_target_ids = {baseline_by_id[pair_id]["case_id"] for pair_id in hard_ids}
    eligible = len(hard_ids) >= MIN_HARD_PAIRS and len(hard_target_ids) >= MIN_HARD_TARGETS
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_ids = set(hard_ids) | set(control_ids)
    public_episodes = []
    private_episodes = []
    private_by_case = {episode["case_id"]: episode for episode in private["episodes"]}
    for episode in public["episodes"]:
        selected_pairs = [pair for pair in episode["pairs"] if pair["pair_id"] in selected_ids]
        if not selected_pairs:
            continue
        public_episodes.append(
            {
                "case_id": episode["case_id"],
                "reference": episode["reference"],
                "pairs": selected_pairs,
                "absence_candidates": episode["absence_candidates"],
            }
        )
        private_source = private_by_case[episode["case_id"]]
        truth_by_id = {pair["pair_id"]: pair for pair in private_source["pairs"]}
        private_episodes.append(
            {
                "case_id": episode["case_id"],
                "private_physical_instance_id": private_source["private_physical_instance_id"],
                "target_native_object_id": private_source["target_native_object_id"],
                "pairs": [
                    {
                        **truth_by_id[pair["pair_id"]],
                        "baseline_evaluation": baseline_by_id[pair["pair_id"]]["evaluation"],
                        "baseline_target_margin": baseline_by_id[pair["pair_id"]]["target_margin"],
                        "stratum": "HARD" if pair["pair_id"] in hard_ids else "CONTROL",
                    }
                    for pair in selected_pairs
                ],
                "absence_candidates": private_source["absence_candidates"],
            }
        )
    challenger_public = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "PDM_CHALLENGER",
        "baseline_report_body_sha256": baseline["body_sha256"],
        "source_public_manifest_body_sha256": public["body_sha256"],
        "episode_count": len(public_episodes),
        "pair_count": sum(len(episode["pairs"]) for episode in public_episodes),
        "absence_count": sum(len(episode["absence_candidates"]) for episode in public_episodes),
        "episodes": public_episodes,
        "score_contract": "OFFICIAL_PDM_PERMIR_TOP1_DIFT_WEIGHTED_QK_COSINE_MEAN_NO_THRESHOLD",
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "PDM_CHALLENGER_PUBLIC_COHORT_READY" if eligible else "PDM_CHALLENGER_NOT_EVALUABLE_HARD_DENOMINATOR",
    }
    _assert_public_blind(challenger_public)
    challenger_public["body_sha256"] = _body_hash(challenger_public)
    challenger_private = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "PDM_CHALLENGER",
        "baseline_report_body_sha256": baseline["body_sha256"],
        "challenger_public_manifest_body_sha256": challenger_public["body_sha256"],
        "hard_pair_ids": hard_ids,
        "control_pair_ids": control_ids,
        "hard_pair_count": len(hard_ids),
        "control_pair_count": len(control_ids),
        "distinct_hard_target_count": len(hard_target_ids),
        "episodes": private_episodes,
        "eligible": eligible,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "PDM_CHALLENGER_PRIVATE_COHORT_FROZEN" if eligible else "PDM_CHALLENGER_NOT_EVALUABLE_HARD_DENOMINATOR",
    }
    challenger_private["body_sha256"] = _body_hash(challenger_private)
    _atomic_json(output_dir / "challenger-public-manifest.json", challenger_public)
    _atomic_json(output_dir / "challenger-private-manifest.json", challenger_private)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "baseline_report_body_sha256": baseline["body_sha256"],
        "hard_pair_count": len(hard_ids),
        "control_pair_count": len(control_ids),
        "distinct_hard_target_count": len(hard_target_ids),
        "absence_count": challenger_public["absence_count"],
        "challenger_authorized": eligible,
        "terminal": "PDM_CHALLENGER_COHORT_FROZEN_READY" if eligible else "PDM_CHALLENGER_NOT_EVALUABLE_HARD_DENOMINATOR",
        "claim_ceiling": CLAIM_CEILING,
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(output_dir / "challenger-freeze-report.json", report)
    return report


def prepare_challenger_inputs(public_manifest_path: Path) -> dict[str, Any]:
    public = _load_json(public_manifest_path)
    _verify_body_hash(public, "challenger public manifest")
    _assert_public_blind(public)
    _require(public["terminal"] == "PDM_CHALLENGER_PUBLIC_COHORT_READY", "challenger cohort is not executable")
    materialized_root = public_manifest_path.parent.parent / "materialized"
    pairs = []
    absences = []
    for episode in public["episodes"]:
        reference = episode["reference"]
        reference_path = materialized_root / reference["image_relative_path"]
        mask_path = materialized_root / reference["mask_relative_path"]
        _require(_sha256_file(reference_path) == reference["image_sha256"], "PDM reference image SHA drifted")
        _require(_sha256_file(mask_path) == reference["mask_sha256"], "PDM reference mask SHA drifted")
        reference_input = {
            **_crop_contract(reference_path, reference["object_region_xyxy_normalized"]),
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": reference["mask_sha256"],
        }
        for pair in episode["pairs"]:
            image_path = materialized_root / pair["image_relative_path"]
            candidates = {
                slot: _crop_contract(image_path, pair["candidate_regions_xyxy_normalized"][slot])
                for slot in ("A", "B")
            }
            pairs.append(
                {
                    "pair_id": pair["pair_id"],
                    "case_id": episode["case_id"],
                    "reference": reference_input,
                    "candidates": candidates,
                }
            )
        for absence in episode["absence_candidates"]:
            image_path = materialized_root / absence["image_relative_path"]
            absences.append(
                {
                    "absence_id": absence["absence_id"],
                    "case_id": episode["case_id"],
                    "reference": reference_input,
                    "candidate": _crop_contract(image_path, absence["candidate_region_xyxy_normalized"]),
                }
            )
    result = {"pairs": pairs, "absences": absences}
    _assert_public_blind(result)
    return result


def evaluate_challenger(
    run_dir: Path,
    private_manifest_path: Path,
) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    raw = _load_json(run_dir / "raw-scores.json")
    private = _load_json(private_manifest_path)
    for value, name in ((config, "PDM config"), (raw, "PDM raw"), (private, "PDM private manifest")):
        _verify_body_hash(value, name)
    _require(raw["run_config_body_sha256"] == config["body_sha256"], "PDM raw/config binding drifted")
    truth = {
        pair["pair_id"]: {**pair, "case_id": episode["case_id"]}
        for episode in private["episodes"]
        for pair in episode["pairs"]
    }
    raw_pairs = {row["pair_id"]: row for row in raw["pairs"]}
    _require(set(truth) == set(raw_pairs), "PDM raw/private pair mismatch")
    rows = []
    for pair_id in sorted(raw_pairs):
        raw_row = raw_pairs[pair_id]
        private_row = truth[pair_id]
        target_slot = private_row["target_slot"]
        distractor_slot = private_row["distractor_slot"]
        target_score = float(raw_row["candidate_scores"][target_slot])
        distractor_score = float(raw_row["candidate_scores"][distractor_slot])
        pdm_evaluation = _evaluation(target_score, distractor_score)
        baseline_target = private_row["baseline_evaluation"] == "TARGET_OUTRANKS"
        pdm_target = pdm_evaluation == "TARGET_OUTRANKS"
        transition = (
            "BASELINE_NON_TARGET_TO_PDM_TARGET"
            if not baseline_target and pdm_target
            else "BASELINE_TARGET_TO_PDM_NON_TARGET"
            if baseline_target and not pdm_target
            else "BASELINE_TARGET_TO_PDM_TARGET"
            if baseline_target
            else "BASELINE_NON_TARGET_TO_PDM_NON_TARGET"
        )
        rows.append(
            {
                **raw_row,
                "target_slot": target_slot,
                "distractor_slot": distractor_slot,
                "target_native_object_id": private_row["target_native_object_id"],
                "distractor_native_object_id": private_row["distractor_native_object_id"],
                "stratum": private_row["stratum"],
                "baseline_evaluation": private_row["baseline_evaluation"],
                "baseline_target_margin": private_row["baseline_target_margin"],
                "pdm_target_score": target_score,
                "pdm_distractor_score": distractor_score,
                "pdm_target_margin": target_score - distractor_score,
                "pdm_evaluation": pdm_evaluation,
                "transition": transition,
            }
        )
    hard_rows = [row for row in rows if row["stratum"] == "HARD"]
    control_rows = [row for row in rows if row["stratum"] == "CONTROL"]
    rescues = sum(row["transition"] == "BASELINE_NON_TARGET_TO_PDM_TARGET" for row in hard_rows)
    collateral = sum(row["transition"] == "BASELINE_TARGET_TO_PDM_NON_TARGET" for row in rows)
    control_retained = sum(row["pdm_evaluation"] == "TARGET_OUTRANKS" for row in control_rows)
    control_retention = control_retained / len(control_rows)
    gate_pass = rescues > collateral and control_retention >= CONTROL_RETENTION_GATE
    absence_truth = {
        row["absence_id"]: row
        for episode in private["episodes"]
        for row in episode["absence_candidates"]
    }
    raw_absences = {row["absence_id"]: row for row in raw["absences"]}
    _require(set(absence_truth) == set(raw_absences), "PDM absence raw/private mismatch")
    absence_rows = [
        {
            **raw_absences[absence_id],
            "target_present": False,
            "target_native_object_id": absence_truth[absence_id]["target_native_object_id"],
            "candidate_native_object_id": absence_truth[absence_id]["candidate_native_object_id"],
            "decision": "NOT_EVALUABLE_NO_FROZEN_NONE_THRESHOLD",
        }
        for absence_id in sorted(raw_absences)
    ]
    if gate_pass:
        outcome = "PDM_UNARY_HARD_ERROR_SIGNAL_SUPPORTED_DEVELOPMENT"
    elif rescues > 0:
        outcome = "PDM_UNARY_MIXED_RESCUE_WITH_COLLATERAL_DEVELOPMENT"
    else:
        outcome = "PDM_UNARY_HARD_ERROR_RESCUE_NOT_SUPPORTED_DEVELOPMENT"
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "PDM_CHALLENGER",
        "evaluated_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "raw_scores_body_sha256": raw["body_sha256"],
        "private_manifest_body_sha256": private["body_sha256"],
        "provider_lock": config["provider_lock"],
        "metrics": {
            "hard_pair_count": len(hard_rows),
            "control_pair_count": len(control_rows),
            "rescue_count": rescues,
            "collateral_count": collateral,
            "control_target_retained_count": control_retained,
            "control_retention": control_retention,
            "success_gate_pass": gate_pass,
            "absence_candidate_count": len(absence_rows),
            "absence_decision_count": 0,
        },
        "rows": rows,
        "absence_rows": absence_rows,
        "scientific_outcome": outcome,
        "protocol_status": "VALID",
        "success_gate": "RESCUE_GT_COLLATERAL_AND_CONTROL_RETENTION_GE_0P80",
        "target_absence_terminal": "TARGET_ABSENCE_NOT_EVALUABLE_NO_FROZEN_NONE_THRESHOLD",
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "PDM_HARD_ERROR_UNARY_CHALLENGER_COMPLETE",
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--roster", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--prior-evidence-root", type=Path, required=True)

    baseline_prepare = subparsers.add_parser("prepare-baseline")
    baseline_prepare.add_argument("--public-manifest", type=Path, required=True)
    baseline_prepare.add_argument("--run-dir", type=Path, required=True)
    baseline_prepare.add_argument("--model-dir", type=Path, required=True)
    baseline_prepare.add_argument("--device", default="cuda")

    baseline_run = subparsers.add_parser("run-baseline")
    baseline_run.add_argument("--model-dir", type=Path, required=True)
    baseline_run.add_argument("--run-dir", type=Path, required=True)
    baseline_run.add_argument("--device", default="cuda")

    baseline_eval = subparsers.add_parser("evaluate-baseline")
    baseline_eval.add_argument("--run-dir", type=Path, required=True)
    baseline_eval.add_argument("--private-manifest", type=Path, required=True)

    challenger_freeze = subparsers.add_parser("freeze-challenger")
    challenger_freeze.add_argument("--baseline-report", type=Path, required=True)
    challenger_freeze.add_argument("--public-manifest", type=Path, required=True)
    challenger_freeze.add_argument("--private-manifest", type=Path, required=True)
    challenger_freeze.add_argument("--output-dir", type=Path, required=True)

    challenger_eval = subparsers.add_parser("evaluate-challenger")
    challenger_eval.add_argument("--run-dir", type=Path, required=True)
    challenger_eval.add_argument("--private-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze(args.output)
    elif args.command == "materialize":
        result = materialize(args.roster, args.output_dir, args.prior_evidence_root)
    elif args.command == "prepare-baseline":
        model = local._validate_model(args.model_dir.resolve(), args.device)
        result = prepare_baseline(args.public_manifest, args.run_dir, model)
    elif args.command == "run-baseline":
        result = execute_baseline(args.model_dir, args.run_dir, args.device)
    elif args.command == "evaluate-baseline":
        result = evaluate_baseline(args.run_dir, args.private_manifest)
    elif args.command == "freeze-challenger":
        result = freeze_challenger(
            args.baseline_report,
            args.public_manifest,
            args.private_manifest,
            args.output_dir,
        )
    elif args.command == "evaluate-challenger":
        result = evaluate_challenger(args.run_dir, args.private_manifest)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
