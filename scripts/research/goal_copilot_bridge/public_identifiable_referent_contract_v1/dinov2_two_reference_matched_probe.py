"""Matched single- versus two-reference DINOv2-S appearance probe.

The data roster is frozen from SUN3D metadata before any new RGB is fetched.
Scoring sees two public reference regions and two oracle candidate regions, but
never the evaluator-private target slot or native identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from PIL import Image

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    c2_small_roster as c2,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    contract,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_local_appearance_probe as local,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    oracle_competing_identity_probe as oracle,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0 import (
    run_sun3d_door_approach_v0 as sun3d,
)


SCHEMA_VERSION = "blindassist_dinov2_two_reference_matched_probe_v0"
PROTOCOL_ID = "DINOV2_TWO_REFERENCE_MATCHED_INFORMATION_GAIN_V0"
MODE = "REVERSIBLE_EXPLORATION_DEVELOPMENT_STANDARD"
EXPECTED_EPISODES = 5
MIN_OBSERVATIONS_PER_EPISODE = 2
MAX_OBSERVATIONS_PER_EPISODE = 3
EXPECTED_PAIRS = 14
MAX_METADATA_GETS = 15
MAX_IMAGE_GETS = EXPECTED_EPISODES * 2 + EXPECTED_PAIRS
MAX_IMAGE_BYTES = 20_000_000
CLAIM_CEILING = (
    "TARGET_AND_FRAME_DISJOINT_SOURCE_CAPTURE_REUSED_SUN3D_DEVELOPMENT_MATCHED_"
    "SINGLE_VS_TWO_REFERENCE_LOCAL_APPEARANCE_ONLY_NO_CONFIRMATION_CANDIDATE_"
    "GENERATION_THRESHOLD_FUSION_BELIEF_TRACKING_ACTIVE_SEARCH_SAFETY_OR_PRODUCT_CLAIM"
)
FORBIDDEN_SCORE_CONFIG_TOKENS = (
    "target_slot",
    "target_position",
    "target_native_object_id",
    "distractor_native_object_id",
    "physical_instance_id",
    "private_identity",
    "baseline_identity_outcome",
    "history_category",
    "evaluation",
)


class TwoReferenceProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TwoReferenceProbeError(message)


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


def _assert_score_config_blind(value: Any) -> None:
    lowered = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    for token in FORBIDDEN_SCORE_CONFIG_TOKENS:
        _require(token not in lowered, f"score config leaks private token: {token}")


def _implementation_lock() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "c2_metadata_adapter": Path(c2.__file__).resolve(),
        "local_scorer": Path(local.__file__).resolve(),
    }
    return {
        f"{name}_path": str(path)
        for name, path in paths.items()
    } | {
        f"{name}_sha256": _sha256_file(path)
        for name, path in paths.items()
    }


def _used_parent_units(parent_episode: Mapping[str, Any]) -> tuple[set[str], set[int]]:
    filenames = {parent_episode["reference"]["source_filename"]}
    excluded_ids = {int(parent_episode["native_object_id"])}
    for observation in parent_episode["later_observations"]:
        filenames.add(observation["source_filename"])
        excluded_ids.update(int(value) for value in observation["same_class_distractor_object_ids"])
    return filenames, excluded_ids


def _eligible_distractors(
    frame: Mapping[str, Any],
    annotation: Mapping[str, Any],
    target_id: int,
    normalized_label: str,
    excluded_ids: set[int],
) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for polygon in frame.get("polygon", []):
        object_id = int(polygon["object"])
        if object_id == target_id or object_id in excluded_ids or object_id >= len(annotation["objects"]):
            continue
        item = annotation["objects"][object_id]
        clipped = c2._clipped_bbox(polygon)
        if (
            item is None
            or clipped is None
            or clipped[1] < c2.DISTRACTOR_MIN_AREA
            or c2._normalize_label(item["name"]) != normalized_label
        ):
            continue
        candidate = {
            "native_object_id": object_id,
            "bbox_xyxy_normalized": clipped[0],
            "bbox_area_fraction": clipped[1],
        }
        previous = found.get(object_id)
        if previous is None or candidate["bbox_area_fraction"] > previous["bbox_area_fraction"]:
            found[object_id] = candidate
    return sorted(found.values(), key=lambda item: (-item["bbox_area_fraction"], item["native_object_id"]))


def _view_changed(
    first: Mapping[str, Any], second: Mapping[str, Any], target_world: Sequence[float]
) -> tuple[bool, float, float]:
    translation = c2._distance(first["camera_center_world"], second["camera_center_world"])
    angle = c2._view_ray_angle(first["camera_center_world"], second["camera_center_world"], target_world)
    return translation >= c2.MIN_TRANSLATION_M or angle >= c2.MIN_VIEW_RAY_DEG, translation, angle


def _target_candidate(
    sequence: str,
    object_id: int,
    annotation: Mapping[str, Any],
    matrices: Sequence[Sequence[float]],
    used_filenames: set[str],
    excluded_ids: set[int],
) -> dict[str, Any] | None:
    if object_id in excluded_ids or object_id >= len(annotation["objects"]):
        return None
    item = annotation["objects"][object_id]
    if item is None:
        return None
    label = c2._normalize_label(item["name"])
    if not label or label in c2.STRUCTURAL_LABELS or label.startswith("wall"):
        return None
    records, world_points = c2._valid_object_records(object_id, annotation, matrices)
    records = [record for record in records if record["source_filename"] not in used_filenames]
    if len(records) < 5 or not world_points:
        return None
    target_world = [statistics.median(point[axis] for point in world_points) for axis in range(3)]
    selected_primary = None
    selected_secondary = None
    selected_qualified: list[dict[str, Any]] = []
    selected_competition: list[dict[str, Any]] = []
    selected_reference_translation = 0.0
    selected_reference_angle = 0.0
    for primary in records:
        if not c2.REFERENCE_MIN_AREA <= primary["bbox_area_fraction"] <= c2.REFERENCE_MAX_AREA:
            continue
        qualified = []
        for observation in records:
            if observation["source_frame_id"] - primary["source_frame_id"] < c2.MIN_SOURCE_FRAME_GAP:
                continue
            if observation["bbox_area_fraction"] < c2.LATER_MIN_AREA:
                continue
            changed, translation, angle = _view_changed(primary, observation, target_world)
            if not changed:
                continue
            qualified.append(
                {
                    **observation,
                    "translation_from_primary_m": translation,
                    "view_ray_from_primary_deg": angle,
                }
            )
        reference_candidates = [
            observation
            for observation in qualified
            if c2.REFERENCE_MIN_AREA
            <= observation["bbox_area_fraction"]
            <= c2.REFERENCE_MAX_AREA
        ]
        if not reference_candidates:
            continue
        reference_candidates.sort(
            key=lambda observation: (
                -observation["view_ray_from_primary_deg"],
                -observation["translation_from_primary_m"],
                -observation["source_frame_id"],
            )
        )
        secondary = reference_candidates[0]
        competition = []
        for observation in qualified:
            if observation["source_filename"] == secondary["source_filename"]:
                continue
            distractors = _eligible_distractors(
                annotation["frames"][observation["frame_id"]],
                annotation,
                object_id,
                label,
                excluded_ids,
            )
            if distractors:
                competition.append({**observation, "distractor": distractors[0]})
        if len(competition) < MIN_OBSERVATIONS_PER_EPISODE:
            continue
        selected_primary = primary
        selected_secondary = secondary
        selected_qualified = qualified
        selected_competition = competition
        selected_reference_translation = secondary["translation_from_primary_m"]
        selected_reference_angle = secondary["view_ray_from_primary_deg"]
        break
    if selected_primary is None or selected_secondary is None:
        return None
    observations = [dict(value) for value in selected_competition[:MAX_OBSERVATIONS_PER_EPISODE]]
    return {
        "sequence": sequence,
        "native_object_id": object_id,
        "private_physical_instance_id": f"sun3d:{sequence}:object:{object_id}",
        "private_normalized_label": label,
        "target_world_xyz_map_derived": target_world,
        "references": [dict(selected_primary), dict(selected_secondary)],
        "reference_translation_m": selected_reference_translation,
        "reference_view_ray_deg": selected_reference_angle,
        "qualifying_later_view_count": len(selected_qualified),
        "eligible_competition_frame_count": len(selected_competition),
        "later_observations": observations,
    }


def _episode_from_metadata(
    sequence: str,
    annotation: Mapping[str, Any],
    extrinsic_text: str,
    used_filenames: set[str],
    excluded_ids: set[int],
) -> dict[str, Any] | None:
    matrices = c2._parse_matrices(extrinsic_text)
    candidates = [
        candidate
        for object_id in range(len(annotation["objects"]))
        if (
            candidate := _target_candidate(
                sequence, object_id, annotation, matrices, used_filenames, excluded_ids
            )
        )
        is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item["eligible_competition_frame_count"],
            -item["qualifying_later_view_count"],
            item["private_normalized_label"],
            item["native_object_id"],
        )
    )
    return candidates[0]


def _target_slot(pair_index: int) -> str:
    return "A" if pair_index % 2 == 0 else "B"


def freeze(
    parent_roster_path: Path,
    parent_public_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require(not output_path.exists(), f"frozen roster already exists: {output_path}")
    parent_roster = _load_json(parent_roster_path)
    parent_public = _load_json(parent_public_manifest_path)
    _verify_body_hash(parent_roster, "parent C2 roster")
    _verify_body_hash(parent_public, "parent C2 public manifest")
    _require(parent_roster["protocol_id"] == c2.PROTOCOL_ID, "parent roster protocol drifted")
    _require(parent_roster["episode_count"] == c2.EXPECTED_SOURCE_COUNT, "parent episode count drifted")
    official_list = sun3d._fetch(sun3d.OFFICIAL_LIST_URL)
    _require(_sha256_bytes(official_list) == parent_roster["official_list_sha256"], "official list SHA drifted")
    metadata_gets = 1
    episodes = []
    dispositions = []
    pair_index = 0
    for parent_episode in parent_roster["episodes"]:
        sequence = parent_episode["sequence"]
        used_filenames, excluded_ids = _used_parent_units(parent_episode)
        base_url = f"{sun3d.DATA_ROOT}/{sequence}"
        annotation_bytes = sun3d._fetch(f"{base_url}/annotation/index.json")
        annotation = json.loads(annotation_bytes)
        extrinsics_name = annotation["extrinsics"]
        extrinsic_bytes = sun3d._fetch(f"{base_url}/extrinsics/{extrinsics_name}")
        metadata_gets += 2
        _require(
            _sha256_bytes(annotation_bytes) == parent_episode["source"]["annotation_sha256"],
            f"annotation SHA drifted for {sequence}",
        )
        _require(
            _sha256_bytes(extrinsic_bytes) == parent_episode["source"]["extrinsics_sha256"],
            f"extrinsics SHA drifted for {sequence}",
        )
        episode = _episode_from_metadata(
            sequence,
            annotation,
            extrinsic_bytes.decode("utf-8"),
            used_filenames,
            excluded_ids,
        )
        if episode is None:
            dispositions.append({"sequence": sequence, "status": "NO_DISJOINT_DUAL_REFERENCE_COMPETITION_EPISODE"})
            continue
        case_id = f"two-ref-{len(episodes) + 1:03d}"
        episode["case_id"] = case_id
        episode["source"] = {
            "annotation_url": f"{base_url}/annotation/index.json",
            "annotation_sha256": _sha256_bytes(annotation_bytes),
            "extrinsics_url": f"{base_url}/extrinsics/{extrinsics_name}",
            "extrinsics_sha256": _sha256_bytes(extrinsic_bytes),
        }
        for reference_index, reference in enumerate(episode["references"], start=1):
            reference["reference_id"] = f"R{reference_index}"
            reference["image_url"] = f"{base_url}/image/{reference['source_filename']}"
        for observation_index, observation in enumerate(episode["later_observations"], start=1):
            observation_id = f"{case_id}-later-{observation_index:02d}"
            observation["observation_id"] = observation_id
            observation["image_url"] = f"{base_url}/image/{observation['source_filename']}"
            observation["target_slot"] = _target_slot(pair_index)
            pair_index += 1
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
    terminal = (
        "TWO_REFERENCE_ROSTER_FROZEN_READY"
        if len(episodes) == EXPECTED_EPISODES
        else "TWO_REFERENCE_NOT_EVALUABLE_COMPLETE_CASE_ROSTER_NE_5"
    )
    roster = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "frozen_at_utc": _utc_now(),
        "data_role": "TARGET_AND_FRAME_DISJOINT_SOURCE_CAPTURE_REUSED_SUN3D_DEVELOPMENT",
        "parent_c2_roster_path": str(parent_roster_path.resolve()),
        "parent_c2_roster_file_sha256": _sha256_file(parent_roster_path),
        "parent_c2_roster_body_sha256": parent_roster["body_sha256"],
        "parent_c2_public_manifest_path": str(parent_public_manifest_path.resolve()),
        "parent_c2_public_manifest_file_sha256": _sha256_file(parent_public_manifest_path),
        "parent_c2_public_manifest_body_sha256": parent_public["body_sha256"],
        "selection_contract": {
            "selected_before_new_rgb_gets": True,
            "selected_before_model_scores": True,
            "selected_before_private_evaluation": True,
            "model_or_score_used_for_selection": False,
            "prior_aggregate_outcome_informed_hypothesis_only": True,
            "excluded_parent_reference_and_candidate_filenames": True,
            "excluded_parent_target_and_all_listed_same_class_distractor_ids": True,
            "reference_pair_view_change": (
                f"SOURCE_FRAME_GAP_GE_{c2.MIN_SOURCE_FRAME_GAP}_AND_"
                f"TRANSLATION_GE_{c2.MIN_TRANSLATION_M}_M_OR_VIEW_RAY_GE_{c2.MIN_VIEW_RAY_DEG}_DEG"
            ),
            "primary_reference_selection": "EARLIEST_REFERENCE_SUPPORTING_SECONDARY_AND_AT_LEAST_2_COMPETITION_FRAMES",
            "secondary_reference_selection": "MAX_VIEW_RAY_THEN_TRANSLATION_THEN_SOURCE_FRAME_ID",
            "within_sequence_order": "MOST_COMPETITION_FRAMES_THEN_ALL_QUALIFIED_VIEWS_THEN_LABEL_THEN_OBJECT_ID",
            "later_selection": "FIRST_UP_TO_3_METADATA_QUALIFIED_COMPETITION_FRAMES",
            "distractor_selection": "LARGEST_SAME_CLASS_REGION_THEN_NATIVE_OBJECT_ID",
            "complete_case_required": True,
        },
        "aggregation_contract": {
            "single_reference_arm": "R1_EXACT_V0_SYMMETRIC_LOCAL_SCORE",
            "two_reference_arm": "MAX_OF_R1_R2_EXACT_V0_SYMMETRIC_LOCAL_SCORES_PER_CANDIDATE",
            "candidate_scoring": "INDEPENDENT_PER_CANDIDATE",
            "comparison": "STRICT_GREATER_THAN_WITH_EXACT_TIE",
            "threshold": None,
            "training": False,
            "augmentation": False,
            "fusion_search": False,
        },
        "implementation_lock": _implementation_lock(),
        "source_dispositions": dispositions,
        "episode_count": len(episodes),
        "pair_count": sum(len(episode["later_observations"]) for episode in episodes),
        "episodes": episodes,
        "budgets": {
            "metadata_gets_used": metadata_gets,
            "image_gets_max": MAX_IMAGE_GETS,
            "image_bytes_max": MAX_IMAGE_BYTES,
            "model_runs": 1,
            "candidate_variants": 1,
        },
        "claim_ceiling": CLAIM_CEILING,
        "terminal": terminal,
    }
    roster["body_sha256"] = _body_hash(roster)
    _atomic_json(output_path, roster)
    return roster


def _verify_frozen_roster(
    roster: Mapping[str, Any], parent_public_manifest_path: Path
) -> Mapping[str, Any]:
    _verify_body_hash(roster, "two-reference roster")
    _require(roster["terminal"] == "TWO_REFERENCE_ROSTER_FROZEN_READY", "roster is not executable")
    _require(roster["episode_count"] == EXPECTED_EPISODES, "frozen episode count drifted")
    _require(roster["pair_count"] == EXPECTED_PAIRS, "frozen pair count drifted")
    _require(roster["implementation_lock"] == _implementation_lock(), "implementation lock drifted")
    _require(
        _sha256_file(parent_public_manifest_path) == roster["parent_c2_public_manifest_file_sha256"],
        "parent public manifest file SHA drifted",
    )
    parent_public = _load_json(parent_public_manifest_path)
    _verify_body_hash(parent_public, "parent C2 public manifest")
    _require(
        parent_public["body_sha256"] == roster["parent_c2_public_manifest_body_sha256"],
        "parent public manifest body SHA drifted",
    )
    return parent_public


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
    _require((width, height) == (c2.IMAGE_WIDTH, c2.IMAGE_HEIGHT), "SUN3D image dimensions drifted")
    sha256 = _sha256_bytes(payload)
    journal.append(
        {
            "event": "complete",
            "url": url,
            "role": role,
            "sha256": sha256,
            "bytes": len(payload),
            "at_utc": _utc_now(),
        }
    )
    _atomic_json(journal_path, journal)
    return sha256, width, height


def materialize(
    roster_path: Path,
    parent_public_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _require(not output_dir.exists(), f"materialization directory already exists: {output_dir}")
    roster = _load_json(roster_path)
    parent_public = _verify_frozen_roster(roster, parent_public_manifest_path)
    parent_hashes = set()
    for episode in parent_public["episodes"]:
        parent_hashes.add(episode["reference_image_sha256"])
        for observation in episode["later_observations"]:
            parent_hashes.add(observation["image_sha256"])
    output_dir.mkdir(parents=True, exist_ok=False)
    counters = {"image_gets": 0, "image_bytes": 0, "reference_gets": 0, "later_gets": 0}
    journal: list[dict[str, Any]] = []
    journal_path = output_dir / "download-journal.json"
    new_hashes: set[str] = set()
    public_episodes = []
    private_episodes = []
    try:
        for episode in roster["episodes"]:
            public_references = []
            private_references = []
            for reference in episode["references"]:
                relative_path = Path("reference-images") / episode["case_id"] / f"{reference['reference_id']}.jpg"
                image_path = output_dir / relative_path
                sha256, width, height = _download_image(
                    reference["image_url"],
                    image_path,
                    f"REFERENCE_{reference['reference_id']}",
                    journal,
                    journal_path,
                    counters,
                )
                counters["reference_gets"] += 1
                _require(sha256 not in parent_hashes, "reference image overlaps parent C2 image hash")
                _require(sha256 not in new_hashes, "reference image hash is duplicated")
                new_hashes.add(sha256)
                public_references.append(
                    {
                        "reference_id": reference["reference_id"],
                        "image_relative_path": relative_path.as_posix(),
                        "image_sha256": sha256,
                        "width": width,
                        "height": height,
                        "object_region_xyxy_normalized": reference["bbox_xyxy_normalized"],
                    }
                )
                private_references.append(
                    {
                        "reference_id": reference["reference_id"],
                        "source_filename": reference["source_filename"],
                        "source_frame_id": reference["source_frame_id"],
                        "image_sha256": sha256,
                    }
                )
            public_episodes.append(
                {
                    "case_id": episode["case_id"],
                    "reference_set_id": f"{episode['case_id']}-reference-set",
                    "references": public_references,
                    "observations": [],
                }
            )
            private_episodes.append(
                {
                    "case_id": episode["case_id"],
                    "sequence": episode["sequence"],
                    "reference_set_id": f"{episode['case_id']}-reference-set",
                    "references": private_references,
                    "private_physical_instance_id": episode["private_physical_instance_id"],
                    "target_native_object_id": episode["native_object_id"],
                    "private_normalized_label": episode["private_normalized_label"],
                    "private_identity_source_sha256": episode["private_identity_source_sha256"],
                    "observations": [],
                }
            )
        _require(counters["later_gets"] == 0, "later image downloaded before reference barrier")
        barrier = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "roster_body_sha256": roster["body_sha256"],
            "episode_count": len(public_episodes),
            "reference_image_gets": counters["reference_gets"],
            "later_image_gets": 0,
            "reference_sha256s": sorted(new_hashes),
        }
        barrier["body_sha256"] = _body_hash(barrier)
        _atomic_json(output_dir / "reference-lock-barrier.json", barrier)
        public_by_id = {episode["case_id"]: episode for episode in public_episodes}
        private_by_id = {episode["case_id"]: episode for episode in private_episodes}
        for episode in roster["episodes"]:
            for observation in episode["later_observations"]:
                relative_path = Path("later-images") / episode["case_id"] / f"{observation['observation_id']}.jpg"
                image_path = output_dir / relative_path
                sha256, width, height = _download_image(
                    observation["image_url"],
                    image_path,
                    "LATER_OBSERVATION",
                    journal,
                    journal_path,
                    counters,
                )
                counters["later_gets"] += 1
                _require(sha256 not in parent_hashes, "later image overlaps parent C2 image hash")
                _require(sha256 not in new_hashes, "new image hash is duplicated")
                new_hashes.add(sha256)
                target_slot = observation["target_slot"]
                distractor_slot = "B" if target_slot == "A" else "A"
                candidates = {
                    target_slot: observation["bbox_xyxy_normalized"],
                    distractor_slot: observation["distractor"]["bbox_xyxy_normalized"],
                }
                public_by_id[episode["case_id"]]["observations"].append(
                    {
                        "pair_id": observation["observation_id"],
                        "observation_id": observation["observation_id"],
                        "image_relative_path": relative_path.as_posix(),
                        "image_sha256": sha256,
                        "width": width,
                        "height": height,
                        "candidate_regions_xyxy_normalized": candidates,
                    }
                )
                private_by_id[episode["case_id"]]["observations"].append(
                    {
                        "pair_id": observation["observation_id"],
                        "observation_id": observation["observation_id"],
                        "image_sha256": sha256,
                        "target_slot": target_slot,
                        "distractor_slot": distractor_slot,
                        "target_native_object_id": episode["native_object_id"],
                        "distractor_native_object_id": observation["distractor"]["native_object_id"],
                    }
                )
        _require(counters["image_gets"] == MAX_IMAGE_GETS, "image GET count drifted")
        _require(counters["later_gets"] == EXPECTED_PAIRS, "later observation count drifted")
        _require(len(new_hashes) == MAX_IMAGE_GETS, "new image hash uniqueness drifted")
        public_manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "mode": MODE,
            "roster_body_sha256": roster["body_sha256"],
            "reference_lock_barrier_body_sha256": barrier["body_sha256"],
            "episode_count": len(public_episodes),
            "pair_count": EXPECTED_PAIRS,
            "episodes": public_episodes,
            "aggregation_contract": roster["aggregation_contract"],
            "claim_ceiling": CLAIM_CEILING,
        }
        public_manifest["body_sha256"] = _body_hash(public_manifest)
        private_manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "roster_body_sha256": roster["body_sha256"],
            "public_manifest_body_sha256": public_manifest["body_sha256"],
            "episodes": private_episodes,
            "claim_ceiling": CLAIM_CEILING,
        }
        private_manifest["body_sha256"] = _body_hash(private_manifest)
        _atomic_json(output_dir / "public-manifest.json", public_manifest)
        _atomic_json(output_dir / "private-evidence-manifest.json", private_manifest)
        report = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": _utc_now(),
            "roster_body_sha256": roster["body_sha256"],
            "public_manifest_body_sha256": public_manifest["body_sha256"],
            "private_manifest_body_sha256": private_manifest["body_sha256"],
            "episode_count": len(public_episodes),
            "pair_count": EXPECTED_PAIRS,
            "image_gets": counters["image_gets"],
            "image_bytes": counters["image_bytes"],
            "unique_new_image_sha256_count": len(new_hashes),
            "parent_image_hash_overlap_count": 0,
            "model_calls": 0,
            "claim_ceiling": CLAIM_CEILING,
            "terminal": "TWO_REFERENCE_MATERIALIZATION_COMPLETE",
        }
        report["body_sha256"] = _body_hash(report)
        _atomic_json(output_dir / "materialization-report.json", report)
        return report
    except Exception as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": _utc_now(),
            "roster_body_sha256": roster.get("body_sha256"),
            "error_class": type(error).__name__,
            "error": str(error),
            "model_calls": 0,
            "claim_ceiling": CLAIM_CEILING,
            "terminal": "TWO_REFERENCE_NOT_EVALUABLE_MATERIALIZATION",
        }
        failure["body_sha256"] = _body_hash(failure)
        _atomic_json(output_dir / "materialization-report.json", failure)
        raise


def prepare_score_run(
    public_manifest_path: Path,
    model: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    _require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    public = _load_json(public_manifest_path)
    _verify_body_hash(public, "public manifest")
    _require(public["protocol_id"] == PROTOCOL_ID, "public manifest protocol drifted")
    _require(public["episode_count"] == EXPECTED_EPISODES, "public episode count drifted")
    _require(public["pair_count"] == EXPECTED_PAIRS, "public pair count drifted")
    root = public_manifest_path.parent
    pairs = []
    for episode in public["episodes"]:
        _require(len(episode["references"]) == 2, "reference set must contain exactly two references")
        reference_ids = [reference["reference_id"] for reference in episode["references"]]
        _require(reference_ids == ["R1", "R2"], "reference IDs must be R1 then R2")
        references = []
        reference_hashes = set()
        for reference in episode["references"]:
            image_path = root / reference["image_relative_path"]
            _require(_sha256_file(image_path) == reference["image_sha256"], "reference image SHA drifted")
            crop = oracle._square_crop_bounds(reference["object_region_xyxy_normalized"])
            references.append(
                {
                    "reference_id": reference["reference_id"],
                    "image_path": str(image_path),
                    "image_sha256": reference["image_sha256"],
                    "crop_bbox_xyxy_normalized": crop,
                    "object_bbox_within_crop_xyxy_normalized": local._relative_bbox(
                        reference["object_region_xyxy_normalized"], crop
                    ),
                }
            )
            reference_hashes.add(reference["image_sha256"])
        _require(len(reference_hashes) == 2, "reference image hashes must be distinct")
        for observation in episode["observations"]:
            later_path = root / observation["image_relative_path"]
            _require(_sha256_file(later_path) == observation["image_sha256"], "later image SHA drifted")
            _require(observation["image_sha256"] not in reference_hashes, "later image duplicates a reference")
            candidates = {}
            for slot in ("A", "B"):
                bbox = observation["candidate_regions_xyxy_normalized"][slot]
                crop = oracle._square_crop_bounds(bbox)
                candidates[slot] = {
                    "crop_bbox_xyxy_normalized": crop,
                    "object_bbox_within_crop_xyxy_normalized": local._relative_bbox(bbox, crop),
                }
            pairs.append(
                {
                    "pair_id": observation["pair_id"],
                    "case_id": episode["case_id"],
                    "observation_id": observation["observation_id"],
                    "reference_set_id": episode["reference_set_id"],
                    "references": references,
                    "later_image_path": str(later_path),
                    "later_image_sha256": observation["image_sha256"],
                    "candidates": candidates,
                }
            )
    _require(len(pairs) == EXPECTED_PAIRS, "prepared pair count drifted")
    config = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "created_at_utc": _utc_now(),
        "source_code_sha256": _sha256_file(Path(__file__)),
        "local_scorer_sha256": _sha256_file(Path(local.__file__)),
        "public_manifest_path": str(public_manifest_path),
        "public_manifest_body_sha256": public["body_sha256"],
        "model": dict(model),
        "crop_contract": {
            "square_context_fraction_per_side": oracle.CROP_CONTEXT_FRACTION,
            "resize": [local.INPUT_SIZE, local.INPUT_SIZE],
            "interpolation": "opencv_inter_cubic",
            "normalization": "imagenet_mean_std",
            "annotations_rendered": False,
        },
        "score_contract": public["aggregation_contract"],
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "pairs": pairs,
    }
    _assert_score_config_blind(config)
    config["body_sha256"] = _body_hash(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "run-config.json", config)
    return config


def _aggregate_reference_scores(scores: Mapping[str, float]) -> tuple[float, list[str]]:
    _require(set(scores) == {"R1", "R2"}, "two-reference aggregation requires R1 and R2")
    values = {key: float(value) for key, value in scores.items()}
    _require(all(math.isfinite(value) for value in values.values()), "reference score is not finite")
    maximum = max(values.values())
    return maximum, sorted(key for key, value in values.items() if value == maximum)


def execute_score(model_dir: Path, run_dir: Path, device: str) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    _verify_body_hash(config, "run config")
    _assert_score_config_blind(config)
    tensors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for pair in config["pairs"]:
        for reference in pair["references"]:
            tensors.append(local._crop_tensor(Path(reference["image_path"]), reference["crop_bbox_xyxy_normalized"]))
            masks.append(local._patch_mask(reference["object_bbox_within_crop_xyxy_normalized"]))
        for slot in ("A", "B"):
            candidate = pair["candidates"][slot]
            tensors.append(local._crop_tensor(Path(pair["later_image_path"]), candidate["crop_bbox_xyxy_normalized"]))
            masks.append(local._patch_mask(candidate["object_bbox_within_crop_xyxy_normalized"]))
    encoder = local.DenseEncoder(model_dir, device)
    features = encoder.encode(tensors)
    rows = []
    offset = 0
    for pair in config["pairs"]:
        reference_features = {"R1": features[offset], "R2": features[offset + 1]}
        reference_masks = {"R1": masks[offset], "R2": masks[offset + 1]}
        offset += 2
        candidate_scores = {}
        for slot in ("A", "B"):
            candidate_feature = features[offset]
            candidate_mask = masks[offset]
            offset += 1
            per_reference = {
                reference_id: local.symmetric_local_score(
                    reference_features[reference_id],
                    candidate_feature,
                    reference_masks[reference_id],
                    candidate_mask,
                )
                for reference_id in ("R1", "R2")
            }
            scalar_scores = {
                reference_id: float(value["symmetric_score"])
                for reference_id, value in per_reference.items()
            }
            two_reference_score, maximizing_ids = _aggregate_reference_scores(scalar_scores)
            candidate_scores[slot] = {
                "per_reference": per_reference,
                "single_reference_score": scalar_scores["R1"],
                "two_reference_score": two_reference_score,
                "maximizing_reference_ids": maximizing_ids,
            }
        single_a = float(candidate_scores["A"]["single_reference_score"])
        single_b = float(candidate_scores["B"]["single_reference_score"])
        two_a = float(candidate_scores["A"]["two_reference_score"])
        two_b = float(candidate_scores["B"]["two_reference_score"])
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "observation_id": pair["observation_id"],
                "reference_set_id": pair["reference_set_id"],
                "candidate_scores": candidate_scores,
                "arms": {
                    "single_reference": {
                        "winner_slot": local._winner(single_a, single_b),
                        "slot_margin_a_minus_b": single_a - single_b,
                    },
                    "two_reference": {
                        "winner_slot": local._winner(two_a, two_b),
                        "slot_margin_a_minus_b": two_a - two_b,
                    },
                },
            }
        )
    _require(offset == len(features), "encoded crop accounting drifted")
    raw = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
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


def _arm_evaluation(target_score: float, distractor_score: float) -> str:
    return (
        "TARGET_OUTRANKS"
        if target_score > distractor_score
        else "DISTRACTOR_OUTRANKS"
        if distractor_score > target_score
        else "TIE"
    )


def _transition(single_evaluation: str, two_evaluation: str) -> str:
    single = "TARGET" if single_evaluation == "TARGET_OUTRANKS" else "NON_TARGET"
    two = "TARGET" if two_evaluation == "TARGET_OUTRANKS" else "NON_TARGET"
    return f"SINGLE_{single}_TO_TWO_{two}"


def _arm_metric(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, int]:
    evaluations = [row["arms"][arm]["evaluation"] for row in rows]
    return {
        "pair_count": len(rows),
        "target_outranks_count": sum(value == "TARGET_OUTRANKS" for value in evaluations),
        "distractor_outranks_count": sum(value == "DISTRACTOR_OUTRANKS" for value in evaluations),
        "tie_count": sum(value == "TIE" for value in evaluations),
    }


def evaluate(run_dir: Path, private_manifest_path: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    raw = _load_json(run_dir / "raw-scores.json")
    private = _load_json(private_manifest_path)
    _verify_body_hash(config, "run config")
    _verify_body_hash(raw, "raw scores")
    _verify_body_hash(private, "private manifest")
    _require(raw["run_config_body_sha256"] == config["body_sha256"], "raw/config binding drifted")
    _require(private["protocol_id"] == PROTOCOL_ID, "private manifest protocol drifted")
    _require(
        private["public_manifest_body_sha256"] == config["public_manifest_body_sha256"],
        "private/public manifest binding drifted",
    )
    private_rows = {
        observation["pair_id"]: observation
        for episode in private["episodes"]
        for observation in episode["observations"]
    }
    raw_ids = [row["pair_id"] for row in raw["rows"]]
    _require(len(raw_ids) == len(set(raw_ids)) == EXPECTED_PAIRS, "raw pair identity drifted")
    _require(set(raw_ids) == set(private_rows), "raw/private pair set mismatch")
    rows = []
    for raw_row in raw["rows"]:
        truth = private_rows[raw_row["pair_id"]]
        target_slot = truth["target_slot"]
        distractor_slot = truth["distractor_slot"]
        arms = {}
        for arm, score_key in (
            ("single_reference", "single_reference_score"),
            ("two_reference", "two_reference_score"),
        ):
            target_score = float(raw_row["candidate_scores"][target_slot][score_key])
            distractor_score = float(raw_row["candidate_scores"][distractor_slot][score_key])
            _require(math.isfinite(target_score) and math.isfinite(distractor_score), "evaluated score is not finite")
            arms[arm] = {
                **raw_row["arms"][arm],
                "target_score": target_score,
                "distractor_score": distractor_score,
                "target_margin": target_score - distractor_score,
                "evaluation": _arm_evaluation(target_score, distractor_score),
            }
        rows.append(
            {
                **raw_row,
                "target_slot": target_slot,
                "distractor_slot": distractor_slot,
                "target_native_object_id": truth["target_native_object_id"],
                "distractor_native_object_id": truth["distractor_native_object_id"],
                "arms": arms,
                "transition": _transition(
                    arms["single_reference"]["evaluation"], arms["two_reference"]["evaluation"]
                ),
                "target_margin_delta_two_minus_single": (
                    arms["two_reference"]["target_margin"] - arms["single_reference"]["target_margin"]
                ),
            }
        )
    transitions = {
        name: sum(row["transition"] == name for row in rows)
        for name in (
            "SINGLE_TARGET_TO_TWO_TARGET",
            "SINGLE_TARGET_TO_TWO_NON_TARGET",
            "SINGLE_NON_TARGET_TO_TWO_TARGET",
            "SINGLE_NON_TARGET_TO_TWO_NON_TARGET",
        )
    }
    rescues = transitions["SINGLE_NON_TARGET_TO_TWO_TARGET"]
    collateral = transitions["SINGLE_TARGET_TO_TWO_NON_TARGET"]
    single_metric = _arm_metric(rows, "single_reference")
    two_metric = _arm_metric(rows, "two_reference")
    deltas = [float(row["target_margin_delta_two_minus_single"]) for row in rows]
    if rescues > 0 and collateral == 0:
        scientific_outcome = "TWO_REFERENCE_MONOTONIC_HEADROOM_OBSERVED_DEVELOPMENT"
    elif rescues > 0 and two_metric["target_outranks_count"] > single_metric["target_outranks_count"]:
        scientific_outcome = "TWO_REFERENCE_NET_HEADROOM_WITH_COLLATERAL_DEVELOPMENT"
    elif rescues > 0:
        scientific_outcome = "TWO_REFERENCE_MIXED_COMPLEMENTARY_SIGNAL_DEVELOPMENT"
    elif collateral > 0:
        scientific_outcome = "TWO_REFERENCE_COLLATERAL_WITHOUT_RESCUE_DEVELOPMENT"
    else:
        scientific_outcome = "TWO_REFERENCE_NO_RANK_INCREMENT_OBSERVED_DEVELOPMENT"
    metrics = {
        "single_reference": single_metric,
        "two_reference": two_metric,
        "transitions": transitions,
        "rescued_pair_count": rescues,
        "collateral_pair_count": collateral,
        "net_target_outrank_delta": (
            two_metric["target_outranks_count"] - single_metric["target_outranks_count"]
        ),
        "single_correct_retained_count": transitions["SINGLE_TARGET_TO_TWO_TARGET"],
        "single_correct_count": single_metric["target_outranks_count"],
        "target_margin_delta": {
            "positive_count": sum(value > 0.0 for value in deltas),
            "zero_count": sum(value == 0.0 for value in deltas),
            "negative_count": sum(value < 0.0 for value in deltas),
            "mean": statistics.fmean(deltas),
            "median": statistics.median(deltas),
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evaluated_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "raw_scores_body_sha256": raw["body_sha256"],
        "private_manifest_body_sha256": private["body_sha256"],
        "model": config["model"],
        "metrics": metrics,
        "rows": rows,
        "scientific_outcome": scientific_outcome,
        "protocol_status": "VALID",
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "TWO_REFERENCE_MATCHED_DEVELOPMENT_COMPLETE",
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--parent-roster", type=Path, required=True)
    freeze_parser.add_argument("--parent-public-manifest", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--roster", type=Path, required=True)
    materialize_parser.add_argument("--parent-public-manifest", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--public-manifest", type=Path, required=True)
    score_parser.add_argument("--model-dir", type=Path, required=True)
    score_parser.add_argument("--run-dir", type=Path, required=True)
    score_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--private-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze(
            args.parent_roster.resolve(),
            args.parent_public_manifest.resolve(),
            args.output.resolve(),
        )
        print(json.dumps({"episode_count": result["episode_count"], "terminal": result["terminal"]}), flush=True)
        return 0 if result["terminal"] == "TWO_REFERENCE_ROSTER_FROZEN_READY" else 2
    if args.command == "materialize":
        result = materialize(
            args.roster.resolve(),
            args.parent_public_manifest.resolve(),
            args.output_dir.resolve(),
        )
        print(json.dumps({"pair_count": result["pair_count"], "terminal": result["terminal"]}), flush=True)
        return 0
    if args.command == "score":
        model_dir = args.model_dir.resolve()
        model = local._validate_model(model_dir, args.device)
        prepare_score_run(args.public_manifest.resolve(), model, args.run_dir.resolve())
        raw = execute_score(model_dir, args.run_dir.resolve(), args.device)
        print(json.dumps({"encoded_crop_count": raw["encoded_crop_count"], "pair_count": len(raw["rows"])}), flush=True)
        return 0
    report = evaluate(args.run_dir.resolve(), args.private_manifest.resolve())
    print(json.dumps({"metrics": report["metrics"], "scientific_outcome": report["scientific_outcome"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
