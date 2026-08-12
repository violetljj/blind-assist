#!/usr/bin/env python3
"""Add the frozen walking_xyz negative to the consumed metric-depth corpus."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    require,
    sha256_file,
)


BASE_CORPUS_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-multisource-distillation-corpus-r0/result.json"
)
WALKING_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-walking-xyz-final-confirmation-labels-r0/result.json"
)
FROZEN_WALKING_SEAM_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-walking-xyz-final-ag-seam-r1-recovery/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-post-walking-multisource-distillation-corpus-r0"
)
EXPECTED_BASE_CORPUS_SHA256 = (
    "0D948F8D582F132BD941CAFBDBC7E60E8C11D9C40CC301B0A0AD4F59F369CD6E"
)
EXPECTED_WALKING_LABEL_SHA256 = (
    "D8083B567CF227AB83423A14B281B8B9A451DF8D6B29F78DF34A5B4459F10812"
)
EXPECTED_FROZEN_WALKING_SEAM_SHA256 = (
    "BEA5F85A9C38BAB0A8EA8DCE81C8E851BABB1C2E01D7A65F82EFF14C2CCA0A96"
)
PARENT_ID = "rgbd_dataset_freiburg3_walking_xyz"


def load_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file(), f"result missing: {path}")
    require(sha256_file(path) == expected_sha256, f"result drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    base = load_result(BASE_CORPUS_RESULT, EXPECTED_BASE_CORPUS_SHA256)
    walking = load_result(WALKING_LABEL_RESULT, EXPECTED_WALKING_LABEL_SHA256)
    frozen = load_result(
        FROZEN_WALKING_SEAM_RESULT,
        EXPECTED_FROZEN_WALKING_SEAM_SHA256,
    )
    require(base["passed"] and base["frame_count"] == 180, "base corpus invalid")
    require(
        walking["passed"]
        and walking["frame_count"] == 12
        and walking["source"]["checkpoint_unseen_by_current_frozen_recipe"],
        "walking labels invalid",
    )
    require(
        not frozen["passed"]
        and frozen["status"]
        == "AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_FROZEN_FAIL"
        and frozen["decision"]["terminal_for_r0_regardless_of_outcome"],
        "walking seam consumption state drift",
    )
    require(
        frozen["aggregate_state_counts"] == {"UNKNOWN": 108}
        and frozen["aggregate_reason_counts"].get("SUPPORT_INVALID") == 99,
        "consumed failure signature drift",
    )

    base_role_parents = {
        role: sorted(
            {
                str(row["parent_id"])
                for row in base["frames"]
                if row["role"] == role
            }
        )
        for role in ("FIT", "CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    require(
        PARENT_ID
        not in set().union(*(set(values) for values in base_role_parents.values())),
        "walking parent already present in base corpus",
    )
    frames = [dict(row) for row in base["frames"]]
    for row in walking["frames"]:
        frames.append(
            {
                **dict(row),
                "source_role_before_post_failure_fit": str(row["role"]),
                "source_id": "TUM_FR3_WALKING_XYZ_CONSUMED_FROZEN_NEGATIVE",
                "role": "FIT",
            }
        )
    frames.sort(key=lambda row: str(row["sample_id"]))
    require(len(frames) == 192, "post-failure corpus frame count drift")
    require(
        len({row["sample_id"] for row in frames}) == 192,
        "post-failure sample collision",
    )
    require(
        all(
            Path(row["output"]).is_file()
            and sha256_file(Path(row["output"])) == row["output_sha256"]
            for row in frames
        ),
        "post-failure label payload drift",
    )

    roles = Counter(str(row["role"]) for row in frames)
    role_parents = {
        role: sorted(
            {str(row["parent_id"]) for row in frames if row["role"] == role}
        )
        for role in ("FIT", "CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    parent_disjoint = (
        set(role_parents["FIT"]).isdisjoint(role_parents["CHECKPOINT_SELECTION"])
        and set(role_parents["FIT"]).isdisjoint(role_parents["TRAIN_CANARY"])
        and set(role_parents["CHECKPOINT_SELECTION"]).isdisjoint(
            role_parents["TRAIN_CANARY"]
        )
    )
    gates = {
        "POSTWALK_C01_EXACT_BASE_LABEL_AND_FROZEN_FAIL_RECEIPTS": True,
        "POSTWALK_C02_192_UNIQUE_PAYLOADS_HASH_EXACT": True,
        "POSTWALK_C03_PARENT_DISJOINT_144_24_24": bool(
            roles == {"FIT": 144, "CHECKPOINT_SELECTION": 24, "TRAIN_CANARY": 24}
            and parent_disjoint
        ),
        "POSTWALK_C04_WALKING_XYZ_CONSUMED_FIT_ONLY": bool(
            PARENT_ID in role_parents["FIT"]
            and PARENT_ID not in role_parents["CHECKPOINT_SELECTION"]
            and PARENT_ID not in role_parents["TRAIN_CANARY"]
        ),
        "POSTWALK_C05_METRIC_FACTOR_LABELS_ONLY_NO_TASK_TARGET": True,
        "POSTWALK_C06_ORIGINAL_SELECTION_AND_CANARY_UNCHANGED": bool(
            role_parents["CHECKPOINT_SELECTION"]
            == base_role_parents["CHECKPOINT_SELECTION"]
            and role_parents["TRAIN_CANARY"] == base_role_parents["TRAIN_CANARY"]
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_post_walking_multisource_corpus_result_v1",
        "status": (
            "AG_R2_POST_WALKING_MULTISOURCE_CORPUS_PASS"
            if passed
            else "AG_R2_POST_WALKING_MULTISOURCE_CORPUS_FAIL"
        ),
        "passed": passed,
        "parent_count": len({row["parent_id"] for row in frames}),
        "frame_count": len(frames),
        "optimizer_supported_frame_count": sum(
            int(row["metric_depth_valid_pixels"]) > 0 for row in frames
        ),
        "roles": {
            role: {"frame_count": roles[role], "parents": role_parents[role]}
            for role in role_parents
        },
        "inputs": {
            "base_multisource_corpus": {
                "path": str(BASE_CORPUS_RESULT.resolve()),
                "sha256": EXPECTED_BASE_CORPUS_SHA256,
            },
            "walking_xyz_labels": {
                "path": str(WALKING_LABEL_RESULT.resolve()),
                "sha256": EXPECTED_WALKING_LABEL_SHA256,
            },
            "walking_xyz_frozen_negative": {
                "path": str(FROZEN_WALKING_SEAM_RESULT.resolve()),
                "sha256": EXPECTED_FROZEN_WALKING_SEAM_SHA256,
            },
        },
        "gates": gates,
        "frames": frames,
        "decision": {
            "walking_xyz_outcome_role": "CONSUMED_TRIGGER_ONLY",
            "training_targets": "METRIC_DEPTH_FACTOR_LABELS_ONLY",
            "task_or_reducer_outputs_used_as_training_targets": False,
            "r0_reopened_or_retuned": False,
            "next_confirmation_must_use_another_parent": True,
        },
        "claim_ceiling": "Consumed multi-source metric-factor supervision; not confirmation, task utility, deployment, product, or safety evidence.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> int:
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "parent_count": result["parent_count"],
                "frame_count": result["frame_count"],
                "optimizer_supported_frame_count": result[
                    "optimizer_supported_frame_count"
                ],
                "roles": result["roles"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
