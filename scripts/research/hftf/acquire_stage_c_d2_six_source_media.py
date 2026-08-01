#!/usr/bin/env python3
"""Acquire the frozen D2 six-source media cohort through short paths.

The formal mode accepts one tracked and pushed mechanics execution contract.
Before the first network request it validates the exact metadata qualification,
the six source bindings, implementation and test receipts, retry count,
authorization firewall, all planned final/staging/downloader-temporary paths,
and then creates a durable global attempt marker.

This acquirer downloads bytes and verifies receipts.  After verifying the full
pose CSV it parses only the selected rows into independent, hash-bound pose
slices.  It does not use poses for a candidate or truth, decode RGB/mask/depth,
run a geometry teacher, create truth, or evaluate an effect.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_sanpo_sequence_evalset import (  # noqa: E402
    DATASET_PAGE,
    DATASET_REPO,
    GCS_PREFIX,
    LICENSE_NAME,
    LICENSE_URL,
    SANPO_CITATION,
    download,
    get_gcs_object,
    media_url,
)


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_six_source_media_"
    "acquisition_execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_D2_METADATA_QUALIFICATION_"
    "BEFORE_D2_MEDIA_OR_MECHANICS"
)
METADATA_SCHEMA = (
    "blindassist_hftf_stage_c_d2_official_train_metadata_qualification"
)
METADATA_READY = "D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED"
D2_DESIGN_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_transport_d2"
)
D2_DESIGN_STATUS = "FROZEN_BEFORE_D2_METADATA_SCAN_OR_SOURCE_OUTCOME"
D2_1_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_"
    "transport_clarification_d2_1"
)
D2_1_STATUS = (
    "FROZEN_AFTER_METADATA_ONLY_COHORT_LOCK_BEFORE_ANY_"
    "D2_MEDIA_POSE_CONTENT_OR_MECHANICS_OUTCOME"
)
TRACKED_METADATA_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_official_train_"
    "metadata_qualification_result"
)
T0_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_t0_consumed_development_transport_result"
)
T0_RESULT_TERMINAL = "T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT"
REPLAY_SCHEMA = "blindassist_sanpo_synthetic_replay_v1"
PREFLIGHT_READY = "D2_SIX_SOURCE_MEDIA_PATH_PREFLIGHT_READY"
ACQUISITION_READY = "D2_SIX_SOURCE_SHORT_PATH_MEDIA_COHORT_ACQUIRED"
NOT_EVALUABLE = (
    "D2_MEDIA_ACQUISITION_NOT_EVALUABLE_NO_RETRY_"
    "NO_SOURCE_REPLACEMENT"
)
SOURCE_INDEX_SCHEMA = (
    "blindassist_hftf_stage_c_d2_per_frame_acquisition_index"
)
SOURCE_INDEX_TERMINAL = "D2_SIX_SOURCE_PER_FRAME_MEDIA_POSE_SLICES_READY"
ACQUIRER_RELATIVE_PATH = (
    "scripts/research/hftf/acquire_stage_c_d2_six_source_media.py"
)
TEST_RELATIVE_PATH = (
    "scripts/research/hftf/test_acquire_stage_c_d2_six_source_media.py"
)
TRANSPORT_DEPENDENCY_RELATIVE_PATH = (
    "scripts/build_sanpo_sequence_evalset.py"
)
CANONICAL_RELATIVE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d2-six-source-media-acquisition-20260802"
)
MAX_CONTENT_PATH_EXCLUSIVE = 240
SESSION_ID_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ID = "sanpo_synthetic_v0"
DATASET_NAME = "SANPO-Synthetic v0"


class AcquisitionError(ValueError):
    """A frozen D2 media acquisition input or receipt is inadmissible."""


@dataclass(frozen=True)
class MetadataApi:
    """The only metadata lookup available after the global attempt opens."""

    get_object: Callable[[str, int], dict[str, Any]]


DEFAULT_API = MetadataApi(get_object=get_gcs_object)


@dataclass(frozen=True)
class CohortLayout:
    acquisition_root: Path
    staging_root: Path
    final_root: Path
    attempt_path: Path
    failure_path: Path
    relative_content_paths: tuple[Path, ...]
    source_tokens: dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AcquisitionError(f"JSON object required: {path}")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_contract_path(
    contract_path: Path,
    raw_value: Any,
) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        raise AcquisitionError("non-empty contract path required")
    raw = Path(raw_value)
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] in {"artifacts.local", "docs", "scripts"}:
        return (_repo_root() / raw).resolve()
    return (contract_path.parent / raw).resolve()


def _require_tracked_clean(path: Path, label: str) -> None:
    repo = _repo_root()
    try:
        relative = path.resolve().relative_to(repo).as_posix()
    except ValueError as error:
        raise AcquisitionError(f"{label} must stay in repository") from error

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AcquisitionError(
                f"{label} must be tracked and clean: "
                + (result.stderr.strip() or "git verification failed")
            )
        return result.stdout.strip()

    git("ls-files", "--error-unmatch", "--", relative)
    git("diff", "--quiet", "--", relative)
    git("diff", "--cached", "--quiet", "--", relative)


def _require_pushed_state(
    contract_path: Path,
    acquirer_path: Path,
    test_path: Path,
    transport_dependency_path: Path,
) -> None:
    repo = _repo_root()
    try:
        relative = contract_path.resolve().relative_to(repo)
    except ValueError as error:
        raise AcquisitionError(
            "D2 media contract must stay in repository"
        ) from error
    if (
        relative.parts[:3] != ("docs", "research", "hftf")
        or contract_path.suffix.lower() != ".json"
    ):
        raise AcquisitionError("D2 media contract must be an HFTF JSON")
    for path, label in (
        (contract_path, "D2 media execution contract"),
        (acquirer_path, "D2 media acquirer"),
        (test_path, "D2 media acquirer test"),
        (
            transport_dependency_path,
            "D2 media network transport dependency",
        ),
    ):
        _require_tracked_clean(path, label)

    def rev(name: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", name],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AcquisitionError(f"cannot resolve Git revision {name}")
        return result.stdout.strip()

    if rev("HEAD") != rev("origin/master"):
        raise AcquisitionError("HEAD must equal origin/master before media open")


def _receipt(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcquisitionError(f"{label} receipt missing")
    try:
        size = int(value.get("size", -1))
    except (TypeError, ValueError) as error:
        raise AcquisitionError(f"{label} size invalid") from error
    result = {
        "name": value.get("name"),
        "generation": str(value.get("generation") or ""),
        "metageneration": str(value.get("metageneration") or ""),
        "size": size,
        "md5_base64": value.get("md5_base64", value.get("md5Hash")),
        "crc32c_base64": value.get(
            "crc32c_base64", value.get("crc32c")
        ),
    }
    if (
        not isinstance(result["name"], str)
        or not result["name"]
        or not result["generation"]
        or result["size"] <= 0
        or not isinstance(result["md5_base64"], str)
        or not result["md5_base64"]
    ):
        raise AcquisitionError(
            f"{label} requires name, generation, positive size, and MD5"
        )
    return result


def _same_receipt(left: Any, right: Any, label: str) -> dict[str, Any]:
    a = _receipt(left, f"{label} frozen")
    b = _receipt(right, f"{label} observed")
    keys = (
        "name",
        "generation",
        "size",
        "md5_base64",
        "crc32c_base64",
    )
    if any(a.get(key) != b.get(key) for key in keys):
        raise AcquisitionError(f"{label} receipt drift")
    return a


def _expected_frames(source_fps: float) -> list[int]:
    if source_fps == 5.0:
        return list(range(13))
    if source_fps == 20.0:
        return list(range(0, 49, 4))
    raise AcquisitionError("source fps must be exactly 5 or 20")


def _validate_metadata_source(source: dict[str, Any]) -> None:
    session_id = source.get("session_id")
    rank = source.get("metadata_eligible_rank")
    fps = float(source.get("source_fps", -1))
    if (
        not isinstance(session_id, str)
        or SESSION_ID_RE.fullmatch(session_id) is None
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank < 1
        or source.get("official_split") != "train"
        or source.get("metadata_eligible") is not True
        or source.get("role")
        != "one_shot_thesis_development_mechanics_evaluation"
        or source.get("selected_source_frames") != _expected_frames(fps)
        or source.get("camera_pose_content_read") is not False
        or source.get("rgb_mask_depth_bytes_read") is not False
    ):
        raise AcquisitionError("metadata-qualified source contract mismatch")
    _receipt(source.get("description_object"), "description")
    _receipt(source.get("camera_pose_object_receipt"), "camera pose")
    modalities = source.get("media_object_listing_receipts")
    if not isinstance(modalities, dict):
        raise AcquisitionError("media object listing receipts missing")
    for key, suffix in (
        ("rgb", ".png"),
        ("mask", ".png"),
        ("depth", ".float16.gz"),
    ):
        value = modalities.get(key)
        if (
            not isinstance(value, dict)
            or value.get("required_frame_count") != 50
            or value.get("required_frame_indices") != [0, 49]
            or not isinstance(value.get("required_frame_receipts"), list)
            or len(value["required_frame_receipts"]) != 50
        ):
            raise AcquisitionError(f"{key} qualification receipt mismatch")
        normalized: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(
            value["required_frame_receipts"]
        ):
            receipt = _receipt(frame, f"{key} frame {frame_index}")
            if (
                frame.get("frame_index") != frame_index
                or not str(receipt["name"]).endswith(
                    f"{frame_index:06d}{suffix}"
                )
            ):
                raise AcquisitionError(f"{key} frame receipt order mismatch")
            normalized.append(
                {
                    "frame_index": frame_index,
                    "name": receipt["name"],
                    "generation": receipt["generation"],
                    "size": receipt["size"],
                    "md5_base64": receipt["md5_base64"],
                }
            )
        if value.get("required_frame_receipts_sha256") != (
            _canonical_json_sha256(normalized)
        ):
            raise AcquisitionError(f"{key} frame receipt hash mismatch")


def _validate_metadata_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        result.get("schema") != METADATA_SCHEMA
        or result.get("terminal") != METADATA_READY
        or result.get("workflow_profile") != "THESIS_DEVELOPMENT"
        or result.get("qualified_parent_count") != 6
        or result.get("requested_parent_count") != 6
        or result.get("official_train_split", {}).get("selection_order")
        != "ascending_session_id"
        or result.get("official_train_split", {}).get(
            "input_split_order_used_for_selection"
        )
        is not False
        or result.get("exclusions", {}).get("excluded_parent_count") != 78
    ):
        raise AcquisitionError("metadata qualification result mismatch")
    firewall = result.get("firewall", {})
    if any(
        firewall.get(key) is not False
        for key in (
            "rgb_bytes_read",
            "panoptic_mask_bytes_read",
            "metric_depth_bytes_read",
            "camera_pose_content_read",
            "geometry_teacher_outcome_read",
            "student_outcome_read",
            "reserved_official_test_opened",
        )
    ):
        raise AcquisitionError("metadata qualification firewall drift")
    sources = result.get("qualified_parents")
    if not isinstance(sources, list) or len(sources) != 6:
        raise AcquisitionError("exact six metadata-qualified parents required")
    for source in sources:
        if not isinstance(source, dict):
            raise AcquisitionError("metadata source must be an object")
        _validate_metadata_source(source)
    if [source["metadata_eligible_rank"] for source in sources] != list(
        range(1, 7)
    ):
        raise AcquisitionError("metadata source ranks must be 1..6")
    ids = [str(source["session_id"]) for source in sources]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise AcquisitionError("metadata sources must be unique ascending IDs")
    _receipt(
        result.get("official_train_split", {}).get("object_receipt"),
        "official train split",
    )
    return sources


def _test_definition_count(path: Path) -> int:
    return len(
        re.findall(
            r"^\s+def test_",
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def _bound_parent(
    contract_path: Path,
    contract: dict[str, Any],
    key: str,
    *,
    expected_schema: str,
    required_field: str,
    required_value: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = contract.get("parents", {}).get(key)
    if not isinstance(receipt, dict):
        raise AcquisitionError(f"D2 media parent missing: {key}")
    path = _resolve_contract_path(contract_path, receipt.get("path"))
    if (
        receipt.get("sha256") != _sha256(path)
        or receipt.get(f"required_{required_field}") != required_value
    ):
        raise AcquisitionError(f"D2 media parent receipt mismatch: {key}")
    value = _load_json(path)
    if (
        value.get("schema") != expected_schema
        or value.get(required_field) != required_value
    ):
        raise AcquisitionError(f"D2 media parent identity mismatch: {key}")
    return path, value


def validate_execution_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("workflow_profile") != "THESIS_DEVELOPMENT"
    ):
        raise AcquisitionError("D2 media execution contract identity mismatch")
    acquirer_path = (_repo_root() / ACQUIRER_RELATIVE_PATH).resolve()
    test_path = (_repo_root() / TEST_RELATIVE_PATH).resolve()
    transport_dependency_path = (
        _repo_root() / TRANSPORT_DEPENDENCY_RELATIVE_PATH
    ).resolve()
    implementation = contract.get("implementations", {}).get(
        "media_acquirer", {}
    )
    transport_dependency = contract.get("implementations", {}).get(
        "sanpo_network_transport_dependency", {}
    )
    test_receipt = contract.get("implementation_tests", {}).get(
        "media_acquirer_test", {}
    )
    actual_test_count = _test_definition_count(test_path)
    if (
        Path(str(implementation.get("path", ""))).as_posix()
        != ACQUIRER_RELATIVE_PATH
        or implementation.get("network_execution_authorized") is not True
        or implementation.get("sha256") != _sha256(acquirer_path)
        or Path(str(transport_dependency.get("path", ""))).as_posix()
        != TRANSPORT_DEPENDENCY_RELATIVE_PATH
        or transport_dependency.get("network_transport_authorized")
        is not True
        or transport_dependency.get("sha256")
        != _sha256(transport_dependency_path)
        or Path(str(test_receipt.get("path", ""))).as_posix()
        != TEST_RELATIVE_PATH
        or test_receipt.get("sha256") != _sha256(test_path)
        or contract.get("implementation_tests", {}).get("test_count")
        != actual_test_count
        or contract.get("implementation_tests", {}).get("tests_passed")
        != actual_test_count
    ):
        raise AcquisitionError("D2 media implementation receipt mismatch")
    if verify_git:
        _require_pushed_state(
            contract_path,
            acquirer_path,
            test_path,
            transport_dependency_path,
        )

    d2_design_path, d2_design = _bound_parent(
        contract_path,
        contract,
        "d2_design",
        expected_schema=D2_DESIGN_SCHEMA,
        required_field="status",
        required_value=D2_DESIGN_STATUS,
    )
    clarification_path, clarification = _bound_parent(
        contract_path,
        contract,
        "d2_1_clarification",
        expected_schema=D2_1_SCHEMA,
        required_field="status",
        required_value=D2_1_STATUS,
    )
    tracked_result_path, tracked_result = _bound_parent(
        contract_path,
        contract,
        "metadata_qualification_result",
        expected_schema=TRACKED_METADATA_RESULT_SCHEMA,
        required_field="terminal",
        required_value=METADATA_READY,
    )
    t0_result_path, _ = _bound_parent(
        contract_path,
        contract,
        "t0_result",
        expected_schema=T0_RESULT_SCHEMA,
        required_field="terminal",
        required_value=T0_RESULT_TERMINAL,
    )
    if (
        d2_design.get("workflow_profile") != "THESIS_DEVELOPMENT"
        or d2_design.get("authorization", {}).get(
            "open_any_new_d2_media_now"
        )
        is not False
        or clarification.get("authorization", {}).get(
            "execute_d2_media_acquisition_now"
        )
        is not False
        or clarification.get("outcome_firewall_at_freeze", {}).get(
            "rgb_bytes_open"
        )
        is not False
        or tracked_result.get("authorization", {}).get(
            "freeze_d2_media_and_mechanics_implementation_contract"
        )
        is not True
        or tracked_result.get("authorization", {}).get(
            "execute_d2_media_acquisition_now"
        )
        is not False
    ):
        raise AcquisitionError(
            "D2 media parent authorization or outcome firewall mismatch"
        )

    parent = contract.get("parents", {}).get("metadata_qualification")
    if not isinstance(parent, dict):
        raise AcquisitionError("metadata qualification parent missing")
    metadata_path = _resolve_contract_path(contract_path, parent.get("path"))
    if (
        parent.get("sha256") != _sha256(metadata_path)
        or parent.get("required_terminal") != METADATA_READY
    ):
        raise AcquisitionError("metadata qualification parent receipt mismatch")
    metadata = _load_json(metadata_path)
    sources = _validate_metadata_result(metadata)
    tracked_qualification = tracked_result.get(
        "durable_evidence", {}
    ).get("qualification", {})
    tracked_qualification_path = _resolve_contract_path(
        tracked_result_path,
        tracked_qualification.get("path"),
    )
    if (
        tracked_qualification_path != metadata_path
        or tracked_qualification.get("sha256") != _sha256(metadata_path)
        or tracked_qualification.get("required_terminal") != METADATA_READY
    ):
        raise AcquisitionError(
            "tracked metadata result does not bind qualification bytes"
        )
    frozen_sources = contract.get("source_cohort", {}).get("sources")
    if (
        contract.get("source_cohort", {}).get("parent_count") != 6
        or contract.get("source_cohort", {}).get("order")
        != "metadata_eligible_rank"
        or not isinstance(frozen_sources, list)
        or len(frozen_sources) != 6
    ):
        raise AcquisitionError("D2 media source cohort contract mismatch")
    expected_sources = [
        {
            "session_id": source["session_id"],
            "metadata_eligible_rank": source["metadata_eligible_rank"],
            "source_fps": float(source["source_fps"]),
            "selected_source_frames": source["selected_source_frames"],
            "metadata_qualified_parent_sha256": _canonical_json_sha256(
                source
            ),
        }
        for source in sources
    ]
    if frozen_sources != expected_sources:
        raise AcquisitionError("D2 media exact source bindings mismatch")

    acquisition_root = _resolve_contract_path(
        contract_path,
        contract.get("canonical_artifacts", {}).get(
            "media_acquisition_root"
        ),
    )
    expected_root = (_repo_root() / CANONICAL_RELATIVE_ROOT).resolve()
    if acquisition_root != expected_root:
        raise AcquisitionError("D2 media canonical root mismatch")
    policy = contract.get("failure_policy", {})
    if (
        policy.get("internal_retries_per_request") != 3
        or policy.get(
            "durable_attempt_file_flush_and_fsync_before_network"
        )
        is not True
        or policy.get("do_not_rerun_same_acquisition") is not True
        or policy.get("do_not_replace_or_append_sources") is not True
        or policy.get("preserve_global_staging_on_failure") is not True
        or policy.get("failure_terminal") != NOT_EVALUABLE
    ):
        raise AcquisitionError("D2 media failure policy mismatch")
    authorization = contract.get("authorization", {})
    required_false = (
        "future_blind_preprocessor_execution_authorized",
        "geometry_teacher_execution_authorized",
        "effect_evaluation_authorized",
        "student_execution_authorized",
        "reserved_official_test_open_authorized",
        "research_mainline_changed",
        "default_app_changed",
        "android_changed",
        "production_authorized",
        "safety_claim_authorized",
    )
    if (
        authorization.get("six_source_media_acquisition_authorized")
        is not True
        or authorization.get("pose_slice_materialization_authorized")
        is not True
        or authorization.get(
            "freeze_future_blind_preprocessor_contract_on_success"
        )
        is not True
        or any(authorization.get(key) is not False for key in required_false)
    ):
        raise AcquisitionError("D2 media authorization firewall mismatch")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": _sha256(contract_path),
        "metadata_path": metadata_path,
        "metadata_sha256": _sha256(metadata_path),
        "metadata": metadata,
        "sources": sources,
        "acquisition_root": acquisition_root,
        "retries": 3,
        "bindings": {
            "execution_contract": {
                "path": str(contract_path),
                "sha256": _sha256(contract_path),
            },
            "metadata_qualification": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
            "d2_design": {
                "path": str(d2_design_path),
                "sha256": _sha256(d2_design_path),
            },
            "d2_1_clarification": {
                "path": str(clarification_path),
                "sha256": _sha256(clarification_path),
            },
            "metadata_qualification_result": {
                "path": str(tracked_result_path),
                "sha256": _sha256(tracked_result_path),
            },
            "t0_result": {
                "path": str(t0_result_path),
                "sha256": _sha256(t0_result_path),
            },
            "media_acquirer": {
                "path": ACQUIRER_RELATIVE_PATH,
                "sha256": _sha256(acquirer_path),
            },
            "media_acquirer_test": {
                "path": TEST_RELATIVE_PATH,
                "sha256": _sha256(test_path),
            },
            "sanpo_network_transport_dependency": {
                "path": TRANSPORT_DEPENDENCY_RELATIVE_PATH,
                "sha256": _sha256(transport_dependency_path),
            },
        },
    }


def _source_token(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("ascii")).hexdigest()[:12]


def _source_relative_root(source: dict[str, Any]) -> Path:
    return (
        Path("s")
        / f"{int(source['metadata_eligible_rank']):02d}-{_source_token(str(source['session_id']))}"
    )


def plan_layout(
    acquisition_root: Path,
    sources: list[dict[str, Any]],
) -> CohortLayout:
    tokens = {
        str(source["session_id"]): _source_token(str(source["session_id"]))
        for source in sources
    }
    if len(set(tokens.values())) != len(tokens):
        raise AcquisitionError("short source token collision")
    relatives: list[Path] = [
        Path("cohort_manifest.json"),
        Path("acquisition_receipt.json"),
        Path("per_frame_acquisition_index.json"),
        Path("global/official_train_split.txt"),
        Path("global/source_labelmap.json"),
    ]
    for source in sources:
        root = _source_relative_root(source)
        relatives.extend(
            root / relative
            for relative in (
                Path("manifest.replay.jsonl"),
                Path("dataset_spec.json"),
                Path("source_licenses.md"),
                Path("qa/replay_validation.json"),
                Path("qa/source_transport_receipt.json"),
                Path("source_metadata/source_session_description.json"),
                Path("source_metadata/source_labelmap.json"),
                Path("source_metadata/source_annotation_types.json"),
                Path("source_metadata/camera_poses.csv"),
                Path("source_metadata/official_split_session_ids.txt"),
            )
        )
        for timeline_index in range(len(source["selected_source_frames"])):
            alias = f"{timeline_index:02x}"
            relatives.extend(
                (
                    root / "pose" / f"{alias}.json",
                    root / "i/train" / f"{alias}.png",
                    root / "m/train" / f"{alias}.png",
                    root / "d/train" / f"{alias}.f16.gz",
                )
            )
    root = acquisition_root.resolve()
    return CohortLayout(
        acquisition_root=root,
        staging_root=root / "staging",
        final_root=root / "cohort",
        attempt_path=root / "attempt.json",
        failure_path=root / "failure.json",
        relative_content_paths=tuple(relatives),
        source_tokens=tokens,
    )


def _tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def path_preflight(layout: CohortLayout) -> dict[str, Any]:
    paths: list[Path] = [
        layout.attempt_path,
        layout.failure_path,
    ]
    for root in (layout.staging_root, layout.final_root):
        for relative in layout.relative_content_paths:
            path = root / relative
            paths.extend((path, _tmp_path(path)))
    rendered = [str(path.resolve()) for path in paths]
    violations = [
        {"path": value, "length": len(value)}
        for value in rendered
        if len(value) >= MAX_CONTENT_PATH_EXCLUSIVE
    ]
    if violations:
        worst = max(violations, key=lambda value: int(value["length"]))
        raise AcquisitionError(
            "D2 media path budget exceeded: "
            f"{worst['length']} >= {MAX_CONTENT_PATH_EXCLUSIVE}"
        )
    ids = list(layout.source_tokens)
    return {
        "schema": (
            "blindassist_hftf_stage_c_d2_six_source_"
            "media_path_preflight"
        ),
        "terminal": PREFLIGHT_READY,
        "maximum_content_path_length": max(map(len, rendered), default=0),
        "content_path_count_including_tmp": len(rendered),
        "limit_exclusive": MAX_CONTENT_PATH_EXCLUSIVE,
        "all_content_paths_under_limit": True,
        "session_id_present_in_any_content_path": any(
            session_id in path for session_id in ids for path in rendered
        ),
        "source_count": len(ids),
        "source_tokens": layout.source_tokens,
        "network_opened": False,
        "media_opened": False,
        "output_created": False,
        "authorization": {
            "media_acquisition_authorized": False,
            "future_blind_preprocessor_execution_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "effect_evaluation_authorized": False,
        },
    }


def _verify_local(path: Path, receipt: dict[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt["size"])
        or _md5_base64(path) != receipt["md5_base64"]
    ):
        raise AcquisitionError(f"{label} local size or MD5 mismatch")


def _download_verified(
    receipt_value: Any,
    target: Path,
    retries: int,
    *,
    downloader: Callable[[str, Path, int], None] = download,
) -> dict[str, Any]:
    receipt = _receipt(receipt_value, str(target))
    if target.exists() or _tmp_path(target).exists():
        raise AcquisitionError(f"refusing to reuse download target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    downloader(
        media_url(str(receipt["name"]), str(receipt["generation"])),
        target,
        retries,
    )
    _verify_local(target, receipt, str(target))
    return receipt


def _copy_verified(
    source: Path,
    target: Path,
    receipt_value: Any,
) -> dict[str, Any]:
    receipt = _receipt(receipt_value, str(target))
    _verify_local(source, receipt, str(source))
    if target.exists() or _tmp_path(target).exists():
        raise AcquisitionError(f"refusing to reuse copy target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp_path(target)
    shutil.copyfile(source, temporary)
    _verify_local(temporary, receipt, str(temporary))
    temporary.replace(target)
    return receipt


def _selected_receipts(
    source: dict[str, Any],
    modality: str,
) -> list[dict[str, Any]]:
    values = source["media_object_listing_receipts"][modality][
        "required_frame_receipts"
    ]
    by_index = {int(value["frame_index"]): value for value in values}
    return [
        _receipt(by_index[index], f"{modality} frame {index}")
        for index in source["selected_source_frames"]
    ]


def _live_receipt(
    api: MetadataApi,
    object_name: str,
    retries: int,
    label: str,
) -> dict[str, Any]:
    value = api.get_object(object_name, retries)
    receipt = _receipt(value, label)
    if receipt["name"] != object_name:
        raise AcquisitionError(f"{label} object name mismatch")
    return receipt


def _source_manifest_row(
    source: dict[str, Any],
    source_root: Path,
    timeline_index: int,
    source_frame_index: int,
    receipts: dict[str, dict[str, Any]],
    relative_paths: dict[str, Path],
    pose_sha256: str,
) -> dict[str, Any]:
    sequence_id = (
        f"sanpo_synthetic_{source['session_id']}_camera_chest_left_"
        f"000000_5fps"
    )
    sample_id = f"{sequence_id}_{timeline_index:06d}"
    return {
        "id": sample_id,
        "image_path": relative_paths["rgb"].as_posix(),
        "image_sha256": _sha256(source_root / relative_paths["rgb"]),
        "source_mask_path": relative_paths["mask"].as_posix(),
        "source_mask_sha256": _sha256(source_root / relative_paths["mask"]),
        "source_depth_path": relative_paths["depth"].as_posix(),
        "source_depth_sha256": _sha256(source_root / relative_paths["depth"]),
        "width": int(source["camera"]["image_width"]),
        "height": int(source["camera"]["image_height"]),
        "session_id": source["session_id"],
        "sequence_id": sequence_id,
        "frame_index": timeline_index,
        "source_frame_index": source_frame_index,
        "source_timestamp_ms": int(
            round(source_frame_index * 1000.0 / float(source["source_fps"]))
        ),
        "source_annotation_quality": "UNKNOWN_NOT_INTERPRETED_BY_ACQUIRER",
        "label_authority": "official_panoptic_d2_development_candidate",
        "event_truth": None,
        "source": {
            "source_id": SOURCE_ID,
            "dataset": DATASET_NAME,
            "dataset_page": DATASET_PAGE,
            "repository": DATASET_REPO,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "official_split": "train",
            "session_id": source["session_id"],
            "camera": "camera_chest",
            "lens": "left",
            "privacy_status": "synthetic_source",
        },
        "modalities": {
            "rgb": receipts["rgb"],
            "panoptic_mask": receipts["mask"],
            "metric_depth": receipts["depth"],
            "camera_poses": {
                "path": "source_metadata/camera_poses.csv",
                "sha256": pose_sha256,
            },
            "imu": {
                "status": "not_present_in_published_session_inventory",
                "usable_for_replay": False,
            },
        },
        "authorization": {
            "offline_replay": True,
            "d2_development_media_candidate": True,
            "future_blind_preprocessor_candidate": True,
            "geometry_teacher_execution": False,
            "effect_evaluation": False,
            "student_training": False,
            "production_model_replacement": False,
        },
    }


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_source_package(
    source: dict[str, Any],
    source_root: Path,
    *,
    description_receipt: dict[str, Any],
    labelmap_receipt: dict[str, Any],
    annotation_receipt: dict[str, Any],
    pose_receipt: dict[str, Any],
    split_receipt: dict[str, Any],
    media_receipts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pose_path = source_root / "source_metadata/camera_poses.csv"
    pose_sha = _sha256(pose_path)
    with pose_path.open("r", encoding="utf-8-sig", newline="") as handle:
        pose_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for timeline_index, source_frame_index in enumerate(
        source["selected_source_frames"]
    ):
        alias = f"{timeline_index:02x}"
        paths = {
            "rgb": Path("i/train") / f"{alias}.png",
            "mask": Path("m/train") / f"{alias}.png",
            "depth": Path("d/train") / f"{alias}.f16.gz",
        }
        receipts = {
            key: media_receipts[key][timeline_index]
            for key in ("rgb", "mask", "depth")
        }
        rows.append(
            _source_manifest_row(
                source,
                source_root,
                timeline_index,
                source_frame_index,
                receipts,
                paths,
                pose_sha,
            )
        )
    frame_index: list[dict[str, Any]] = []
    source_relative_root = _source_relative_root(source)
    for normalized_index, (source_frame_index, row) in enumerate(
        zip(source["selected_source_frames"], rows)
    ):
        if (
            not isinstance(source_frame_index, int)
            or isinstance(source_frame_index, bool)
            or source_frame_index < 0
            or source_frame_index >= len(pose_rows)
        ):
            raise AcquisitionError("selected pose row index is invalid")
        pose_row = pose_rows[source_frame_index]
        try:
            position = [
                float(pose_row[key])
                for key in ("pos_x", "pos_y", "pos_z")
            ]
            quaternion = [
                float(pose_row[key])
                for key in ("q_x", "q_y", "q_z", "q_w")
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise AcquisitionError(
                "selected pose row values are invalid"
            ) from error
        norm = math.sqrt(sum(value * value for value in quaternion))
        if (
            pose_row.get("tracking_state") != "TrackingState.READY"
            or not all(math.isfinite(value) for value in position)
            or not all(math.isfinite(value) for value in quaternion)
            or abs(norm - 1.0) > 1e-3
        ):
            raise AcquisitionError(
                "selected pose row fails frozen authority checks"
            )
        alias = f"{normalized_index:02x}"
        pose_relative = Path("pose") / f"{alias}.json"
        pose_slice = {
            "schema": (
                "blindassist_hftf_stage_c_d2_single_pose_slice"
            ),
            "normalized_index": normalized_index,
            "timeline_index": normalized_index,
            "source_frame_index": source_frame_index,
            "tracking_state": pose_row.get("tracking_state"),
            "source_pose_csv_sha256": pose_sha,
            "binding": {
                "position_m": position,
                "quaternion_xyzw": quaternion,
            },
        }
        pose_slice_path = source_root / pose_relative
        _write_json_exclusive(pose_slice_path, pose_slice)
        depth_relative = Path("d/train") / f"{alias}.f16.gz"
        mask_relative = Path("m/train") / f"{alias}.png"
        frame_index.append(
            {
                "normalized_index": normalized_index,
                "source_frame_index": source_frame_index,
                "manifest_id": row["id"],
                "pose_slice": {
                    "path": (
                        source_relative_root / pose_relative
                    ).as_posix(),
                    "sha256": _sha256(pose_slice_path),
                },
                "depth": {
                    "path": (
                        source_relative_root / depth_relative
                    ).as_posix(),
                    "sha256": _sha256(
                        source_root / depth_relative
                    ),
                },
                "mask": {
                    "path": (
                        source_relative_root / mask_relative
                    ).as_posix(),
                    "sha256": _sha256(
                        source_root / mask_relative
                    ),
                },
            }
        )
    manifest_path = source_root / "manifest.replay.jsonl"
    _write_text_exclusive(
        manifest_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    spec = {
        "schema": REPLAY_SCHEMA,
        "purpose": (
            "D2 future-blind preprocessing candidate intake; "
            "no teacher or effect execution"
        ),
        "source": {
            "source_id": SOURCE_ID,
            "dataset": DATASET_NAME,
            "official_split": "train",
            "session_id": source["session_id"],
        },
        "sampling": {
            "source_fps": float(source["source_fps"]),
            "target_fps": 5.0,
            "selected_source_frames": source["selected_source_frames"],
        },
        "camera": source["camera"],
        "source_inventory": {
            "description": description_receipt,
            "labelmap": labelmap_receipt,
            "annotation_types": annotation_receipt,
            "camera_poses": pose_receipt,
            "official_split_receipt": split_receipt,
            "rgb": media_receipts["rgb"],
            "masks": media_receipts["mask"],
            "depth": media_receipts["depth"],
        },
        "prohibited_execution": [
            "geometry_teacher",
            "effect_evaluation",
            "student_training",
            "Android_runtime",
            "production_replacement",
            "safety_claim",
        ],
    }
    spec_path = source_root / "dataset_spec.json"
    _write_json_exclusive(spec_path, spec)
    _write_text_exclusive(
        source_root / "source_licenses.md",
        (
            "# SANPO-Synthetic source\n\n"
            f"- License: [{LICENSE_NAME}]({LICENSE_URL})\n"
            f"- Attribution: {SANPO_CITATION}\n"
            "- Boundary: D2 Development media candidate only.\n"
        ),
    )
    qa = {
        "ok": True,
        "frame_count": len(rows),
        "all_downloads_generation_bound": True,
        "all_downloads_size_and_md5_verified": True,
        "rgb_mask_depth_decoded": False,
        "camera_pose_content_interpreted_for_candidate_or_truth": False,
        "camera_pose_content_read_for_exact_frame_slicing": True,
        "camera_pose_used_for_candidate_or_truth": False,
        "geometry_teacher_executed": False,
        "effect_evaluated": False,
    }
    qa_path = source_root / "qa/replay_validation.json"
    _write_json_exclusive(qa_path, qa)
    receipt = {
        "schema": (
            "blindassist_hftf_stage_c_d2_source_short_path_media_receipt"
        ),
        "terminal": "D2_SOURCE_MEDIA_PACKAGE_READY",
        "session_id": source["session_id"],
        "metadata_eligible_rank": source["metadata_eligible_rank"],
        "source_frame_indices": source["selected_source_frames"],
        "manifest_sha256": _sha256(manifest_path),
        "dataset_spec_sha256": _sha256(spec_path),
        "replay_validation_sha256": _sha256(qa_path),
        "future_blind_preprocessor_candidate": True,
        "geometry_teacher_executed": False,
        "effect_evaluated": False,
    }
    receipt_path = source_root / "qa/source_transport_receipt.json"
    _write_json_exclusive(receipt_path, receipt)
    return {
        **receipt,
        "source_relative_root": _source_relative_root(source).as_posix(),
        "camera": source["camera"],
        "frames": frame_index,
        "source_transport_receipt_sha256": _sha256(receipt_path),
    }


def open_attempt(
    layout: CohortLayout,
    frozen: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    layout.acquisition_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        layout.attempt_path,
        {
            "schema": (
                "blindassist_hftf_stage_c_d2_six_source_"
                "media_acquisition_attempt"
            ),
            "status": "ATTEMPT_OPENED_BEFORE_FIRST_NETWORK_REQUEST",
            "bindings": frozen["bindings"],
            "source_count": 6,
            "source_order": [
                source["session_id"] for source in frozen["sources"]
            ],
            "path_preflight": preflight,
            "internal_retries_per_request": 3,
            "rerun_authorized": False,
            "source_replacement_authorized": False,
            "future_blind_preprocessor_execution_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "effect_evaluation_authorized": False,
        },
    )


def acquire(
    frozen: dict[str, Any],
    layout: CohortLayout,
    *,
    api: MetadataApi = DEFAULT_API,
    downloader: Callable[[str, Path, int], None] = download,
) -> dict[str, Any]:
    retries = int(frozen["retries"])
    staging = layout.staging_root
    staging.mkdir(parents=True, exist_ok=False)
    metadata = frozen["metadata"]
    split_receipt = _receipt(
        metadata["official_train_split"]["object_receipt"],
        "official train split",
    )
    split_path = staging / "global/official_train_split.txt"
    _download_verified(
        split_receipt, split_path, retries, downloader=downloader
    )
    split_text = split_path.read_text(encoding="utf-8-sig")
    if (
        hashlib.sha256(split_text.encode("utf-8")).hexdigest()
        != metadata["official_train_split"]["text_sha256"]
    ):
        raise AcquisitionError("official train split content SHA drift")
    split_ids = {line.strip() for line in split_text.splitlines() if line.strip()}
    if any(source["session_id"] not in split_ids for source in frozen["sources"]):
        raise AcquisitionError("qualified source missing from official train split")

    labelmap_name = f"{GCS_PREFIX}/labelmap.json"
    labelmap_receipt = _live_receipt(
        api, labelmap_name, retries, "labelmap"
    )
    labelmap_path = staging / "global/source_labelmap.json"
    _download_verified(
        labelmap_receipt, labelmap_path, retries, downloader=downloader
    )

    source_results: list[dict[str, Any]] = []
    for source in frozen["sources"]:
        session_id = str(source["session_id"])
        source_root = staging / _source_relative_root(source)
        metadata_root = source_root / "source_metadata"
        prefix = f"{GCS_PREFIX}/sanpo-synthetic/{session_id}"
        annotation_name = (
            f"{prefix}/camera_chest/left/"
            "frame_segmentation_annotation_type.json"
        )
        annotation_receipt = _live_receipt(
            api, annotation_name, retries, f"{session_id}:annotation"
        )
        description_receipt = _receipt(
            source["description_object"], f"{session_id}:description"
        )
        pose_receipt = _receipt(
            source["camera_pose_object_receipt"], f"{session_id}:pose"
        )
        description_path = (
            metadata_root / "source_session_description.json"
        )
        pose_path = metadata_root / "camera_poses.csv"
        annotation_path = metadata_root / "source_annotation_types.json"
        _download_verified(
            description_receipt,
            description_path,
            retries,
            downloader=downloader,
        )
        _download_verified(
            pose_receipt, pose_path, retries, downloader=downloader
        )
        _download_verified(
            annotation_receipt,
            annotation_path,
            retries,
            downloader=downloader,
        )
        _copy_verified(
            labelmap_path,
            metadata_root / "source_labelmap.json",
            labelmap_receipt,
        )
        _copy_verified(
            split_path,
            metadata_root / "official_split_session_ids.txt",
            split_receipt,
        )
        description = _load_json(description_path)
        if description.get("session_type") != "synthetic":
            raise AcquisitionError("downloaded description is not synthetic")

        selected = {
            key: _selected_receipts(source, key)
            for key in ("rgb", "mask", "depth")
        }
        for timeline_index in range(len(source["selected_source_frames"])):
            alias = f"{timeline_index:02x}"
            for key, relative in (
                ("rgb", Path("i/train") / f"{alias}.png"),
                ("mask", Path("m/train") / f"{alias}.png"),
                ("depth", Path("d/train") / f"{alias}.f16.gz"),
            ):
                _download_verified(
                    selected[key][timeline_index],
                    source_root / relative,
                    retries,
                    downloader=downloader,
                )
        source_results.append(
            _write_source_package(
                source,
                source_root,
                description_receipt=description_receipt,
                labelmap_receipt=labelmap_receipt,
                annotation_receipt=annotation_receipt,
                pose_receipt=pose_receipt,
                split_receipt=split_receipt,
                media_receipts=selected,
            )
        )

    manifest = {
        "schema": (
            "blindassist_hftf_stage_c_d2_six_source_media_cohort"
        ),
        "terminal": ACQUISITION_READY,
        "bindings": frozen["bindings"],
        "source_count": 6,
        "source_order": [
            source["session_id"] for source in frozen["sources"]
        ],
        "sources": source_results,
        "all_downloads_generation_bound": True,
        "all_downloads_size_and_md5_verified": True,
        "rgb_mask_depth_decoded": False,
        "camera_pose_content_interpreted_for_candidate_or_truth": False,
        "camera_pose_content_read_for_exact_frame_slicing": True,
        "camera_pose_used_for_candidate_or_truth": False,
        "geometry_teacher_executed": False,
        "effect_evaluated": False,
    }
    manifest_path = staging / "cohort_manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    source_index = {
        "schema": SOURCE_INDEX_SCHEMA,
        "terminal": SOURCE_INDEX_TERMINAL,
        "bindings": frozen["bindings"],
        "source_count": 6,
        "source_order": manifest["source_order"],
        "sources": source_results,
        "pose_csv_read_only_for_exact_selected_frame_slicing": True,
        "candidate_or_truth_executed": False,
        "future_blind_preprocessor_execution_authorized": False,
    }
    source_index_path = staging / "per_frame_acquisition_index.json"
    _write_json_exclusive(source_index_path, source_index)
    cohort_receipt = {
        "schema": (
            "blindassist_hftf_stage_c_d2_six_source_"
            "media_acquisition_receipt"
        ),
        "terminal": ACQUISITION_READY,
        "bindings": frozen["bindings"],
        "cohort_manifest_sha256": _sha256(manifest_path),
        "per_frame_acquisition_index_sha256": _sha256(
            source_index_path
        ),
        "source_count": 6,
        "source_order": manifest["source_order"],
        "all_downloads_generation_bound": True,
        "all_downloads_size_and_md5_verified": True,
        "authorization": {
            "freeze_future_blind_preprocessor_contract": True,
            "future_blind_preprocessor_execution_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "effect_evaluation_authorized": False,
            "student_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }
    receipt_path = staging / "acquisition_receipt.json"
    _write_json_exclusive(receipt_path, cohort_receipt)
    if layout.final_root.exists():
        raise AcquisitionError("final cohort root appeared during acquisition")
    staging.replace(layout.final_root)
    final_receipt = layout.final_root / "acquisition_receipt.json"
    return {
        **cohort_receipt,
        "cohort_root": str(layout.final_root),
        "acquisition_receipt_path": str(final_receipt),
        "acquisition_receipt_sha256": _sha256(final_receipt),
        "per_frame_acquisition_index_path": str(
            layout.final_root / "per_frame_acquisition_index.json"
        ),
        "per_frame_acquisition_index_sha256": _sha256(
            layout.final_root / "per_frame_acquisition_index.json"
        ),
    }


def failure_report(
    frozen: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema": (
            "blindassist_hftf_stage_c_d2_six_source_media_"
            "acquisition_failure"
        ),
        "terminal": NOT_EVALUABLE,
        "bindings": frozen["bindings"],
        "error": {"type": type(error).__name__, "message": str(error)},
        "global_staging_preserved": True,
        "rerun_authorized": False,
        "source_replacement_authorized": False,
        "partial_completion_authorized": False,
        "authorization": {
            "future_blind_preprocessor_execution_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "effect_evaluation_authorized": False,
            "student_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }


def _require_preflight_output(path: Path) -> Path:
    root = (_repo_root() / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AcquisitionError("preflight output must stay under artifacts.local") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    frozen: dict[str, Any] | None = None
    layout: CohortLayout | None = None
    attempt_opened = False
    try:
        frozen = validate_execution_contract(
            args.execution_contract,
            verify_git=not args.preflight_only,
        )
        if args.retries != 3:
            raise AcquisitionError("retries must equal frozen value 3")
        layout = plan_layout(
            frozen["acquisition_root"], frozen["sources"]
        )
        preflight = path_preflight(layout)
        if preflight["session_id_present_in_any_content_path"] is not False:
            raise AcquisitionError("session identity leaked into short path")
        if args.preflight_only:
            if args.preflight_output is not None:
                output = _require_preflight_output(args.preflight_output)
                if output.exists():
                    raise AcquisitionError("refusing to overwrite preflight")
                _write_json_exclusive(output, preflight)
            print(json.dumps(preflight, ensure_ascii=False))
            return 0
        if args.preflight_output is not None:
            raise AcquisitionError(
                "--preflight-output is only valid with --preflight-only"
            )
        if layout.acquisition_root.exists():
            raise AcquisitionError(
                "D2 media acquisition root already exists; rerun forbidden"
            )
        open_attempt(layout, frozen, preflight)
        attempt_opened = True
        result = acquire(frozen, layout)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        KeyboardInterrupt,
        json.JSONDecodeError,
    ) as error:
        terminal: str | None = None
        if (
            attempt_opened
            and frozen is not None
            and layout is not None
            and not layout.failure_path.exists()
            and not layout.final_root.exists()
        ):
            report = failure_report(frozen, error)
            _write_json_exclusive(layout.failure_path, report)
            terminal = NOT_EVALUABLE
        print(
            json.dumps(
                {
                    "ok": False,
                    "terminal": terminal,
                    "error": str(error),
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
