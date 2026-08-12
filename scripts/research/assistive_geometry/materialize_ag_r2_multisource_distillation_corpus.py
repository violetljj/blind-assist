#!/usr/bin/env python3
"""Bind TUM13 plus consumed ICL/TUM labels into a multi-source depth corpus."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_factor_learnability_attempt03 import require, sha256_file  # noqa: E402


BASE_CORPUS_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-superteacher-distillation-corpus-tum13-r0/result.json"
)
ICL_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-icl-fresh-confirmation-labels-r0/result.json"
)
TUM_REAL_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-real-fresh-confirmation-labels-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-multisource-distillation-corpus-r0"
)
EXPECTED_BASE_CORPUS_SHA256 = (
    "1D94379A61E26C5D7457A25BCDE1E90128678DC96603980A23F0F1263F1D7A42"
)
EXPECTED_ICL_LABEL_SHA256 = (
    "E3A8F7FF73BD30AD9701D090F5D8959F4C93F45BB70944C85BA01D0AE3CAFBB1"
)
EXPECTED_TUM_REAL_LABEL_SHA256 = (
    "04C10167E2C94010D4680510A30F0F05B284D822EF4858ED772E83B0390F4ABB"
)


def load_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file(), f"label result missing: {path}")
    require(sha256_file(path) == expected_sha256, f"label result drift: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result["passed"], f"label prerequisite failed: {path}")
    return result


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    base = load_result(BASE_CORPUS_RESULT, EXPECTED_BASE_CORPUS_SHA256)
    icl = load_result(ICL_LABEL_RESULT, EXPECTED_ICL_LABEL_SHA256)
    tum = load_result(TUM_REAL_LABEL_RESULT, EXPECTED_TUM_REAL_LABEL_SHA256)
    require(base["frame_count"] == 156, "base corpus frame count drift")
    require(icl["frame_count"] == 12 and tum["frame_count"] == 12, "adaptation label count drift")
    require(
        icl["decision"]["current_student_or_reducer_output_opened_during_materialization"] is False
        and tum["decision"]["current_student_or_reducer_output_opened_during_materialization"] is False,
        "label firewall drift",
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
    frames = [dict(row) for row in base["frames"]]
    for source_id, result in (("ICL_TRAJECTORY1_CONSUMED", icl), ("TUM_FR3_SITTING_STATIC_CONSUMED", tum)):
        for row in result["frames"]:
            frames.append(
                {
                    **dict(row),
                    "source_role_before_multisource_fit": str(row["role"]),
                    "source_id": source_id,
                    "role": "FIT",
                }
            )
    frames.sort(key=lambda row: str(row["sample_id"]))
    require(len(frames) == 180, "combined corpus frame count drift")
    require(len({row["sample_id"] for row in frames}) == 180, "combined sample identity collision")
    require(
        all(
            Path(row["output"]).is_file()
            and sha256_file(Path(row["output"])) == row["output_sha256"]
            for row in frames
        ),
        "combined label payload drift",
    )
    roles = Counter(str(row["role"]) for row in frames)
    role_parents = {
        role: sorted({str(row["parent_id"]) for row in frames if row["role"] == role})
        for role in ("FIT", "CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    parent_disjoint = (
        set(role_parents["FIT"]).isdisjoint(role_parents["CHECKPOINT_SELECTION"])
        and set(role_parents["FIT"]).isdisjoint(role_parents["TRAIN_CANARY"])
        and set(role_parents["CHECKPOINT_SELECTION"]).isdisjoint(role_parents["TRAIN_CANARY"])
    )
    gates = {
        "MSCORPUS_C01_EXACT_THREE_LABEL_RECEIPTS": True,
        "MSCORPUS_C02_180_UNIQUE_PAYLOADS_HASH_EXACT": True,
        "MSCORPUS_C03_PARENT_DISJOINT_132_24_24": bool(
            roles == {"FIT": 132, "CHECKPOINT_SELECTION": 24, "TRAIN_CANARY": 24}
            and parent_disjoint
        ),
        "MSCORPUS_C04_TWO_CONSUMED_SOURCES_FIT_ONLY": bool(
            {"icl_living_room_kt1", "rgbd_dataset_freiburg3_sitting_static"}
            .issubset(role_parents["FIT"])
        ),
        "MSCORPUS_C05_LABEL_ONLY_NO_TASK_OR_REDUCER_OUTPUT": True,
        "MSCORPUS_C06_ORIGINAL_SELECTION_AND_CANARY_UNCHANGED": bool(
            role_parents["CHECKPOINT_SELECTION"]
            == base_role_parents["CHECKPOINT_SELECTION"]
            and role_parents["TRAIN_CANARY"]
            == base_role_parents["TRAIN_CANARY"]
        ),
    }
    passed = all(gates.values())
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "blindassist_ag_r2_multisource_distillation_corpus_result_v1",
        "status": "AG_R2_MULTISOURCE_DISTILLATION_CORPUS_PASS"
        if passed
        else "AG_R2_MULTISOURCE_DISTILLATION_CORPUS_FAIL",
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
            "base_tum13_corpus": {
                "path": str(BASE_CORPUS_RESULT.resolve()),
                "sha256": EXPECTED_BASE_CORPUS_SHA256,
            },
            "consumed_icl_labels": {
                "path": str(ICL_LABEL_RESULT.resolve()),
                "sha256": EXPECTED_ICL_LABEL_SHA256,
            },
            "consumed_tum_real_labels": {
                "path": str(TUM_REAL_LABEL_RESULT.resolve()),
                "sha256": EXPECTED_TUM_REAL_LABEL_SHA256,
            },
        },
        "gates": gates,
        "frames": frames,
        "decision": {
            "icl_and_tum_seam_outcomes_used_to_trigger_development": True,
            "task_or_reducer_outputs_used_as_training_targets": False,
            "training_targets": "metric depth factor labels only",
            "next_action": "Train one multi-source metric-depth student; keep a third real parent outside all fitting and selection.",
        },
        "claim_ceiling": "Consumed multi-source factor supervision corpus; not fresh confirmation, task utility, deployment, product, or safety evidence.",
    }
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
