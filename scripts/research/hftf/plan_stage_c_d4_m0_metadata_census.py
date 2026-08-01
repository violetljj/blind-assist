#!/usr/bin/env python3
"""Run the one-shot HFTF D4-M0 metadata census and frozen allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d2_official_train_metadata import (  # noqa: E402
    DEFAULT_API,
    CandidateIneligible,
    GCS_PREFIX,
    MetadataApi,
    _receipt,
    _timeline,
    _validate_intrinsics,
    camera_metadata,
    media_url,
)
from plan_stage_c_d3_q0_metadata_roster import (  # noqa: E402
    derive_exclusions as derive_prior_84,
    selected_modality_receipt,
)


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d4_m0_metadata_census_execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_D4_COUNT_CORRECTION_BEFORE_M0_METADATA_CENSUS"
)
ATTEMPT_SCHEMA = "blindassist_hftf_stage_c_d4_m0_scan_attempt"
ATTEMPT_STATUS = "ATTEMPT_FSYNCED_BEFORE_FIRST_NETWORK_REQUEST"
CENSUS_SCHEMA = "blindassist_hftf_stage_c_d4_m0_metadata_census"
POOL_SCHEMA = "blindassist_hftf_stage_c_d4_m0_five_hz_pool_manifest"
ALLOCATION_ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d4_m0_allocation_attempt"
)
SEED_SCHEMA = "blindassist_hftf_stage_c_d4_m0_random_seed_receipt"
RESULT_SCHEMA = "blindassist_hftf_stage_c_d4_m0_recruitability_pool_result"
FAILURE_SCHEMA = "blindassist_hftf_stage_c_d4_m0_failure"

M0_LOCKED = "D4_M0_FRESH_METADATA_RECRUITABILITY_POOL_CENSUS_LOCKED"
M0_INSUFFICIENT = (
    "D4_M0_FRESH_5HZ_METADATA_RECRUITABILITY_POOL_INSUFFICIENT_STOP"
)
M0_INVALID = "D4_M0_FRESH_METADATA_RECRUITABILITY_POOL_INVALID_STOP"

CANONICAL_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d4-m0-metadata-census-20260802"
)
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D4_M0_METADATA_CENSUS_EXECUTION_CONTRACT_"
    "2026-08-02.json"
)
PLANNER_RELATIVE_PATH = Path(
    "scripts/research/hftf/plan_stage_c_d4_m0_metadata_census.py"
)
TEST_RELATIVE_PATH = Path(
    "scripts/research/hftf/test_plan_stage_c_d4_m0_metadata_census.py"
)
FILENAMES = {
    "attempt": "attempt.json",
    "preflight": "preflight.json",
    "census": "census.json",
    "pool": "pool.json",
    "allocation_attempt": "alloc-attempt.json",
    "seed": "seed.json",
    "result": "result.json",
    "failure": "failure.json",
}
SESSION_ID_RE = re.compile(r"^[0-9a-f]{64}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
RANK_DOMAIN = b"HFTF_D4_R0_ALLOC|"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_bound(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return candidate.resolve()


def test_definition_count(path: Path) -> int:
    return len(re.findall(r"^\s+def test_", path.read_text(), re.MULTILINE))


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_tracked_clean(path: Path, label: str) -> None:
    relative = path.resolve().relative_to(repo_root()).as_posix()
    if not git("ls-files", "--error-unmatch", "--", relative):
        raise ValueError(f"{label} is not tracked")
    if git("status", "--porcelain", "--", relative):
        raise ValueError(f"{label} is not clean")


def require_pushed_state(paths: list[tuple[Path, str]]) -> None:
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/master"):
        raise ValueError("HEAD differs from origin/master")
    for path, label in paths:
        require_tracked_clean(path, label)


def bound_parent(
    binding: dict[str, Any],
    *,
    status_key: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = resolve_bound(str(binding["path"]))
    if sha256(path) != str(binding["sha256"]):
        raise ValueError(f"Parent hash drift: {path}")
    value = load_json(path)
    if status_key:
        expected = str(binding[f"required_{status_key}"])
        if str(value.get(status_key)) != expected:
            raise ValueError(f"Parent {status_key} drift: {path}")
    return path, value


def exclusion_manifest_bytes(ids: set[str]) -> bytes:
    return "".join(f"{item}\n" for item in sorted(ids)).encode("utf-8")


def derive_exclusions(
    d2_qualification: dict[str, Any],
    q0_roster: dict[str, Any],
    design: dict[str, Any],
) -> tuple[set[str], set[str], dict[str, list[str]]]:
    prior_categories, prior = derive_prior_84(d2_qualification)
    q0_rows = q0_roster.get("locked_slots")
    if not isinstance(q0_rows, list) or len(q0_rows) != 40:
        raise ValueError("Q0 roster is not exact 40")
    q0 = {str(row.get("session_id", "")) for row in q0_rows}
    if len(q0) != 40 or not all(SESSION_ID_RE.fullmatch(x) for x in q0):
        raise ValueError("Q0 roster identities invalid")
    if prior & q0:
        raise ValueError("Prior 84 and Q0 40 overlap")
    union = prior | q0
    manifest = exclusion_manifest_bytes(union)
    target = design["target_source_universe_before_metadata_eligibility"]
    if len(union) != int(target["complete_exclusion_union_count"]):
        raise ValueError("Global exclusion count drift")
    if len(manifest) != int(target["complete_exclusion_union_bytes"]):
        raise ValueError("Global exclusion byte count drift")
    if sha256_bytes(manifest) != str(
        target["complete_exclusion_union_sha256"]
    ):
        raise ValueError("Global exclusion hash drift")
    categories = {
        key: sorted(str(x) for x in values)
        for key, values in prior_categories.items()
    }
    categories["d3_q0_locked_roster"] = sorted(q0)
    return prior, q0, categories


def audit_post_q0_local_evidence(
    union: set[str],
    roots: list[dict[str, Any]],
) -> dict[str, Any]:
    observed: set[str] = set()
    files = 0
    per_root: list[dict[str, Any]] = []
    for specification in roots:
        root = resolve_bound(str(specification["path"]))
        if not root.is_dir():
            raise ValueError(f"Required Q0 evidence root missing: {root}")
        paths = sorted(root.glob("slot-*/attempt.json"))
        expected_count = int(specification["expected_attempt_count"])
        if len(paths) != expected_count:
            raise ValueError(
                f"Unexpected slot attempt count under {root}: {len(paths)}"
            )
        allowed_schemas = set(specification["allowed_attempt_schemas"])
        root_observed: set[str] = set()
        for path in paths:
            files += 1
            value = load_json(path)
            if value.get("schema") not in allowed_schemas:
                raise ValueError(f"Unexpected slot attempt schema: {path}")
            if value.get("status") != (
                "D3_Q0_SLOT_ATTEMPT_FSYNCED_BEFORE_FIRST_POSE_OR_MEDIA_REQUEST"
            ):
                raise ValueError(f"Unexpected slot attempt status: {path}")
            session_id = str(value.get("session_id", ""))
            if not SESSION_ID_RE.fullmatch(session_id):
                raise ValueError(f"Invalid slot attempt session ID: {path}")
            root_observed.add(session_id)
        if len(root_observed) != expected_count:
            raise ValueError(f"Duplicate slot attempt session ID under {root}")
        observed |= root_observed
        per_root.append(
            {
                "path": str(root),
                "attempt_file_count": len(paths),
                "observed_session_id_count": len(root_observed),
            }
        )
    outside = observed - union
    if outside:
        raise ValueError(
            "Post-Q0 attempted/opened parent outside frozen 124: "
            + ",".join(sorted(outside))
        )
    return {
        "json_file_count": files,
        "observed_session_id_count": len(observed),
        "all_observed_session_ids_within_frozen_124": True,
        "allowlisted_receipts_only": True,
        "sealed_payload_selector_truth_or_frame_json_read": False,
        "roots": per_root,
    }


def validate_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected D4-M0 contract schema")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("Unexpected D4-M0 contract status")
    parents = contract["parents"]
    design_path, design = bound_parent(
        parents["d4_design"], status_key="status"
    )
    d2_path, d2 = bound_parent(
        parents["d2_metadata_qualification"], status_key="terminal"
    )
    q0_path, q0 = bound_parent(
        parents["q0_metadata_roster"], status_key="terminal"
    )
    prior, q0_ids, categories = derive_exclusions(d2, q0, design)
    union = prior | q0_ids
    implementation_paths: list[tuple[Path, str]] = []
    for label, binding in contract["implementations"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"Implementation hash drift: {label}")
        implementation_paths.append((path, label))
    test_binding = contract["implementation_tests"]["planner_test"]
    test_path = resolve_bound(str(test_binding["path"]))
    if sha256(test_path) != str(test_binding["sha256"]):
        raise ValueError("Planner test hash drift")
    expected_tests = int(contract["implementation_tests"]["test_count"])
    if test_definition_count(test_path) != expected_tests:
        raise ValueError("Planner test count drift")
    if int(contract["network"]["retries"]) != 3:
        raise ValueError("Frozen retry count is not three")
    if not contract["authorization"]["metadata_census_authorized"]:
        raise ValueError("Metadata census not authorized")
    if verify_git:
        require_pushed_state(
            [
                (contract_path, "contract"),
                (design_path, "design"),
                (d2_path, "D2 qualification"),
                (q0_path, "Q0 roster"),
                *implementation_paths,
                (test_path, "planner test"),
            ]
        )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "design": design,
        "design_path": design_path,
        "d2_path": d2_path,
        "q0_path": q0_path,
        "prior": prior,
        "q0_ids": q0_ids,
        "excluded": union,
        "exclusion_categories": categories,
        "local_evidence_audit_roots": list(
            contract["pre_network_local_evidence_audit"]["roots"]
        ),
        "split_generation": str(
            contract["official_train_split"]["generation"]
        ),
        "split_sha256": str(
            contract["official_train_split"]["text_sha256"]
        ),
        "retries": 3,
    }


def parse_split_ids(text: str) -> list[str]:
    ids = [line.strip() for line in text.splitlines() if line.strip()]
    if len(ids) != 1560:
        raise ValueError("Official train split must contain exact 1560 IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Official train split contains duplicate IDs")
    if not all(SESSION_ID_RE.fullmatch(item) for item in ids):
        raise ValueError("Official train split contains invalid ID")
    return ids


def qualify_candidate(
    session_id: str,
    retries: int,
    api: MetadataApi,
) -> dict[str, Any]:
    prefix = f"{GCS_PREFIX}/sanpo-synthetic/{session_id}"
    description_name = f"{prefix}/description.json"
    pose_name = f"{prefix}/camera_chest/camera_poses.csv"
    try:
        description_receipt = _receipt(
            api.get_object(description_name, retries),
            description_name,
            "description",
        )
        pose_receipt = _receipt(
            api.get_object(pose_name, retries),
            pose_name,
            "camera pose",
        )
        description = api.fetch_json(
            media_url(description_name, description_receipt["generation"]),
            retries,
        )
        if description.get("session_type") != "synthetic":
            raise CandidateIneligible("description is not synthetic")
        source_fps, dimensions = camera_metadata(
            description, "camera_chest", "left"
        )
        if source_fps not in (5.0, 20.0):
            raise CandidateIneligible("source fps is not exact 5 or 20")
        camera = _validate_intrinsics(dimensions)
        frames = _timeline(source_fps)
        modalities: dict[str, Any] = {}
        for label, folder, suffix in (
            ("mask", "segmentation_masks", ".png"),
            ("depth", "depth_maps", ".float16.gz"),
        ):
            objects = api.list_objects(
                f"{prefix}/camera_chest/left/{folder}/", retries
            )
            modalities[label] = selected_modality_receipt(
                objects, suffix, label, frames
            )
        return {
            "session_id": session_id,
            "classification": (
                "metadata_eligible_5hz"
                if source_fps == 5.0
                else "metadata_eligible_20hz"
            ),
            "source_fps": source_fps,
            "normalized_target_fps": 5.0,
            "selected_source_frames": frames,
            "description_object": description_receipt,
            "camera_pose_object_receipt": pose_receipt,
            "camera": camera,
            "mask_depth_object_listing_receipts": modalities,
            "description_bytes_read": True,
            "pose_content_read": False,
            "rgb_listing_or_bytes_read": False,
            "mask_depth_bytes_read": False,
            "support_truth_or_effect_read": False,
        }
    except CandidateIneligible:
        raise
    except HTTPError as error:
        if error.code == 404:
            raise CandidateIneligible(
                f"metadata object deterministically absent after "
                f"{retries} retries: HTTP 404"
            ) from error
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CandidateIneligible(
            f"metadata schema invalid after {retries} retries: "
            f"{error}"
        ) from error


def scan_all(
    ordered_ids: list[str],
    excluded: set[str],
    qualifier: Callable[[str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ledger: list[dict[str, Any]] = []
    five: list[dict[str, Any]] = []
    twenty: list[dict[str, Any]] = []
    calls = 0
    for split_index, session_id in enumerate(ordered_ids, 1):
        if session_id in excluded:
            ledger.append(
                {
                    "official_split_index": split_index,
                    "session_id": session_id,
                    "classification": "excluded_before_candidate_request",
                    "candidate_metadata_request_started": False,
                }
            )
            continue
        calls += 1
        try:
            row = qualifier(session_id)
        except CandidateIneligible as error:
            ledger.append(
                {
                    "official_split_index": split_index,
                    "candidate_attempt_index": calls,
                    "session_id": session_id,
                    "classification": "metadata_ineligible_closed",
                    "reason": str(error),
                    "candidate_metadata_request_started": True,
                }
            )
            continue
        row = dict(row)
        row["official_split_index"] = split_index
        row["candidate_attempt_index"] = calls
        ledger.append(row)
        if row["classification"] == "metadata_eligible_5hz":
            five.append(row)
        elif row["classification"] == "metadata_eligible_20hz":
            twenty.append(row)
        else:
            raise ValueError("Qualifier returned unknown classification")
    if len(ledger) != 1560:
        raise ValueError("Census ledger is not exact 1560")
    return ledger, five, twenty


def rank_pool(
    ids: list[str],
    seed: bytes,
    *,
    digest: Callable[[bytes], str] | None = None,
) -> list[dict[str, Any]]:
    digest = digest or (lambda value: hashlib.sha256(value).hexdigest())
    rows = [
        {
            "session_id": session_id,
            "rank_digest": digest(
                RANK_DOMAIN
                + seed
                + b"|"
                + session_id.lower().encode("ascii")
            ),
        }
        for session_id in ids
    ]
    if not all(HEX64_RE.fullmatch(str(row["rank_digest"])) for row in rows):
        raise ValueError("Rank digest is not lowercase SHA-256 hex")
    digests = [str(row["rank_digest"]) for row in rows]
    if len(set(digests)) != len(digests):
        raise ValueError("Rank digest collision")
    rows.sort(key=lambda row: str(row["rank_digest"]))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def allocation_parameters(n_five: int) -> dict[str, int]:
    total = min(n_five, 128)
    ecology = (3 * total) // 8
    return {
        "N_five_hz": n_five,
        "C": total,
        "n_ecology": ecology,
        "B_effect_reserve": total - ecology,
    }


def fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    # Windows does not expose a supported directory fsync equivalent:
    # FlushFileBuffers rejects directory handles.  Each artifact itself is
    # flushed and then reopened/byte-verified below; that is the frozen
    # Windows durability barrier used by this one-shot workflow.
    if not path.is_dir():
        raise NotADirectoryError(path)


def write_json_exclusive_fsync(path: Path, value: dict[str, Any]) -> None:
    data = canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    if path.read_bytes() != data:
        raise OSError(f"Durable JSON reopen verification failed: {path}")
    fsync_directory(path.parent)


def require_canonical_root(path: Path) -> Path:
    expected = (repo_root() / CANONICAL_ROOT).resolve()
    requested = path.resolve(strict=False)
    if requested != expected:
        raise ValueError("Noncanonical D4-M0 output root")
    current = repo_root().resolve()
    for part in CANONICAL_ROOT.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("Symlink is forbidden in canonical output path")
    for filename in FILENAMES.values():
        for suffix in ("", ".tmp", ".orphan"):
            if len(str(expected / f"{filename}{suffix}")) >= 240:
                raise ValueError("D4-M0 artifact path reaches 240 characters")
    return expected


def artifact_state(root: Path) -> set[str]:
    if not root.exists():
        return set()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Canonical D4-M0 root is not a real directory")
    return {path.name for path in root.iterdir()}


def validate_existing_terminal(root: Path, names: set[str]) -> bool:
    result_name = FILENAMES["result"]
    failure_name = FILENAMES["failure"]
    if result_name in names and failure_name in names:
        return False
    if result_name in names:
        try:
            result = load_json(root / result_name)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if result.get("schema") != RESULT_SCHEMA:
            return False
        terminal = result.get("terminal")
        if terminal == M0_LOCKED:
            expected = {
                FILENAMES[key]
                for key in (
                    "attempt",
                    "preflight",
                    "census",
                    "pool",
                    "allocation_attempt",
                    "seed",
                    "result",
                )
            }
            hashes = {
                "attempt_sha256": FILENAMES["attempt"],
                "preflight_sha256": FILENAMES["preflight"],
                "census_sha256": FILENAMES["census"],
                "pool_sha256": FILENAMES["pool"],
                "allocation_attempt_sha256": FILENAMES[
                    "allocation_attempt"
                ],
                "seed_receipt_sha256": FILENAMES["seed"],
            }
            bindings = result.get("bindings", {})
        elif terminal == M0_INSUFFICIENT:
            expected = {
                FILENAMES[key]
                for key in ("attempt", "preflight", "census", "pool", "result")
            }
            hashes = {
                "attempt_sha256": FILENAMES["attempt"],
                "preflight_sha256": FILENAMES["preflight"],
                "census_sha256": FILENAMES["census"],
                "pool_sha256": FILENAMES["pool"],
            }
            bindings = result.get("bindings", {})
        else:
            return False
        if names != expected or not isinstance(bindings, dict):
            return False
        return all(
            bindings.get(field) == sha256(root / filename)
            for field, filename in hashes.items()
        )
    if failure_name in names:
        try:
            failure = load_json(root / failure_name)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        known = set(FILENAMES.values())
        return (
            names <= known
            and failure.get("schema") == FAILURE_SCHEMA
            and failure.get("terminal") == M0_INVALID
            and result_name not in names
        )
    return False


def freeze_existing_partial(root: Path, names: set[str]) -> int:
    if FILENAMES["failure"] in names:
        raise ValueError("D4-M0 frozen failure already exists")
    write_json_exclusive_fsync(
        root / FILENAMES["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": M0_INVALID,
            "reason": "preexisting_partial_or_unknown_canonical_root",
            "observed_artifact_names": sorted(names),
            "network_requests_started_by_this_freeze": False,
            "seed_generated_by_this_freeze": False,
            "resume_or_rerun_authorized": False,
            "fresh_content_read": False,
        },
    )
    return 2


def execute(
    contract_path: Path,
    root: Path,
    retries: int,
    *,
    api: MetadataApi = DEFAULT_API,
    token_bytes: Callable[[int], bytes] = secrets.token_bytes,
    verify_git: bool = True,
) -> dict[str, Any]:
    context = validate_contract(contract_path, verify_git=verify_git)
    if retries != context["retries"]:
        raise ValueError("Retries differ from frozen D4-M0 contract")
    if root.exists():
        raise FileExistsError("Canonical D4-M0 root already exists")
    root.mkdir(parents=False, exist_ok=False)
    fsync_directory(root.parent)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": ATTEMPT_STATUS,
        "execution_contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256(contract_path),
        },
        "d4_design_sha256": sha256(context["design_path"]),
        "retries": retries,
        "first_network_request_started": False,
        "seed_generated": False,
        "fresh_content_read": False,
        "resume_or_rerun_authorized": False,
        "pre_network_local_evidence_audit_pending": True,
    }
    write_json_exclusive_fsync(root / FILENAMES["attempt"], attempt)
    local_audit = audit_post_q0_local_evidence(
        context["excluded"], context["local_evidence_audit_roots"]
    )
    write_json_exclusive_fsync(
        root / FILENAMES["preflight"],
        {
            "schema": "blindassist_hftf_stage_c_d4_m0_pre_network_preflight",
            "status": "PREFLIGHT_FSYNCED_BEFORE_FIRST_NETWORK_REQUEST",
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "local_evidence_audit": local_audit,
            "network_request_started": False,
            "seed_generated": False,
            "fresh_content_read": False,
        },
    )

    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/train_session_ids.txt"
    )
    split_object = _receipt(
        api.get_object(split_name, retries), split_name, "official train split"
    )
    if str(split_object["generation"]) != context["split_generation"]:
        raise ValueError("Official train split generation drift")
    split_text = api.fetch_text(
        media_url(split_name, split_object["generation"]), retries
    )
    if sha256_text(split_text) != context["split_sha256"]:
        raise ValueError("Official train split SHA-256 drift")
    ordered_ids = parse_split_ids(split_text)
    in_split_excluded = context["excluded"] & set(ordered_ids)
    out_of_split = context["excluded"] - set(ordered_ids)
    if len(in_split_excluded) != 118 or len(out_of_split) != 6:
        raise ValueError("Frozen 124 exclusion train projection drift")
    frozen_out = {
        item
        for values in context["design"][
            "target_source_universe_before_metadata_eligibility"
        ]["outside_official_train_exclusions"].values()
        for item in values
    }
    if out_of_split != frozen_out:
        raise ValueError("Out-of-split exclusion identity drift")
    ledger, five, twenty = scan_all(
        ordered_ids,
        context["excluded"],
        lambda session_id: qualify_candidate(session_id, retries, api),
    )
    candidate_attempts = len(ordered_ids) - len(in_split_excluded)
    if candidate_attempts != 1442:
        raise ValueError("Candidate attempt count is not exact 1442")
    ineligible = candidate_attempts - len(five) - len(twenty)
    census = {
        "schema": CENSUS_SCHEMA,
        "status": "COMPLETE_METADATA_CENSUS_FSYNCED",
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "execution_contract_sha256": sha256(contract_path),
            "d4_design_sha256": sha256(context["design_path"]),
        },
        "official_train_split": {
            "generation": split_object["generation"],
            "text_sha256": sha256_text(split_text),
            "session_count": len(ordered_ids),
            "order_preserved": True,
        },
        "exclusion_projection": {
            "global_count": len(context["excluded"]),
            "in_split_count": len(in_split_excluded),
            "outside_split_count": len(out_of_split),
            "manifest_sha256": sha256_bytes(
                exclusion_manifest_bytes(context["excluded"])
            ),
        },
        "candidate_attempt_count": candidate_attempts,
        "metadata_eligible_5hz_count": len(five),
        "metadata_eligible_20hz_count": len(twenty),
        "metadata_ineligible_count": ineligible,
        "scan_ledger": ledger,
        "firewall": {
            "pose_content_read": False,
            "rgb_listing_or_bytes_read": False,
            "mask_depth_bytes_read": False,
            "support_truth_effect_or_sealed_payload_read": False,
        },
    }
    write_json_exclusive_fsync(root / FILENAMES["census"], census)

    five_ids = [str(row["session_id"]) for row in five]
    pool = {
        "schema": POOL_SCHEMA,
        "status": "FIVE_HZ_ELIGIBLE_POOL_MANIFEST_FSYNCED",
        "census_sha256": sha256(root / FILENAMES["census"]),
        "N_five_hz": len(five_ids),
        "session_ids_in_official_split_order": five_ids,
        "session_ids_sha256": sha256_bytes(
            "".join(f"{item}\n" for item in five_ids).encode("utf-8")
        ),
        "fresh_content_read": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["pool"], pool)
    params = allocation_parameters(len(five_ids))
    if len(five_ids) < 64:
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": M0_INSUFFICIENT,
            "bindings": {
                "attempt_sha256": sha256(root / FILENAMES["attempt"]),
                "preflight_sha256": sha256(root / FILENAMES["preflight"]),
                "census_sha256": sha256(root / FILENAMES["census"]),
                "pool_sha256": sha256(root / FILENAMES["pool"]),
            },
            "allocation_parameters": params,
            "allocation_attempt_created": False,
            "seed_generated": False,
            "fresh_content_read": False,
            "ecology_or_effect_authorized": False,
        }
        write_json_exclusive_fsync(root / FILENAMES["result"], result)
        return result

    allocation_attempt = {
        "schema": ALLOCATION_ATTEMPT_SCHEMA,
        "status": "ALLOCATION_ATTEMPT_FSYNCED_BEFORE_OS_CSPRNG_CALL",
        "pool_sha256": sha256(root / FILENAMES["pool"]),
        "N_five_hz": len(five_ids),
        "seed_generated": False,
        "fresh_content_read": False,
        "resume_or_rerun_authorized": False,
    }
    write_json_exclusive_fsync(
        root / FILENAMES["allocation_attempt"], allocation_attempt
    )
    seed = token_bytes(32)
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError("OS CSPRNG did not return exact 32 bytes")
    seed_receipt = {
        "schema": SEED_SCHEMA,
        "status": "ONE_SHOT_OS_CSPRNG_SEED_FSYNCED",
        "eligible_pool_manifest_sha256": sha256(root / FILENAMES["pool"]),
        "allocation_attempt_sha256": sha256(
            root / FILENAMES["allocation_attempt"]
        ),
        "seed_hex": seed.hex(),
        "os_csprng_source": "python_secrets_token_bytes_32",
        "generated_after_pool_manifest_fsync": True,
        "fresh_content_read": False,
        "rerun_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["seed"], seed_receipt)
    ranked = rank_pool(five_ids, seed)
    n = params["n_ecology"]
    total = params["C"]
    ecology = ranked[:n]
    effect = ranked[n:total]
    unassigned = ranked[total:]
    if (
        {row["session_id"] for row in ecology}
        & {row["session_id"] for row in effect}
    ):
        raise ValueError("Ecology/effect allocation overlap")
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": M0_LOCKED,
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "census_sha256": sha256(root / FILENAMES["census"]),
            "pool_sha256": sha256(root / FILENAMES["pool"]),
            "allocation_attempt_sha256": sha256(
                root / FILENAMES["allocation_attempt"]
            ),
            "seed_receipt_sha256": sha256(root / FILENAMES["seed"]),
        },
        "allocation_parameters": params,
        "randomized_five_hz_parent_rank": ranked,
        "ecology_parent_ids": [row["session_id"] for row in ecology],
        "effect_reserve_parent_ids": [
            row["session_id"] for row in effect
        ],
        "unassigned_parent_ids": [
            row["session_id"] for row in unassigned
        ],
        "sets_are_disjoint_and_exhaust_five_hz_pool": True,
        "fresh_content_read": False,
        "ecology_or_effect_execution_authorized": False,
        "next_authority": "freeze_new_ecology_execution_contract_only",
    }
    write_json_exclusive_fsync(root / FILENAMES["result"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-contract",
        type=Path,
        default=repo_root() / CONTRACT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / CANONICAL_ROOT,
    )
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    root = require_canonical_root(args.output_root)
    names = artifact_state(root)
    if names:
        if validate_existing_terminal(root, names):
            raise ValueError("D4-M0 validated terminal already exists")
        return freeze_existing_partial(root, names)
    result = execute_with_failure_closure(
        args.execution_contract.resolve(),
        root,
        args.retries,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def execute_with_failure_closure(
    contract_path: Path,
    root: Path,
    retries: int,
    *,
    api: MetadataApi = DEFAULT_API,
    token_bytes: Callable[[int], bytes] = secrets.token_bytes,
    verify_git: bool = True,
) -> dict[str, Any]:
    try:
        return execute(
            contract_path,
            root,
            retries,
            api=api,
            token_bytes=token_bytes,
            verify_git=verify_git,
        )
    except BaseException as error:
        if root.exists() and FILENAMES["failure"] not in artifact_state(root):
            try:
                write_json_exclusive_fsync(
                    root / FILENAMES["failure"],
                    {
                        "schema": FAILURE_SCHEMA,
                        "terminal": M0_INVALID,
                        "reason": f"{type(error).__name__}: {error}",
                        "observed_artifact_names": sorted(
                            artifact_state(root)
                        ),
                        "resume_or_rerun_authorized": False,
                        "fresh_content_read": False,
                    },
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
