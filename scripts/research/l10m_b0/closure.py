"""Freeze and close L10M-B0 without adding a B0-E protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .b0c_precedence import run_matrix_b0c
from .b0d_arrival_orthogonality import (
    B0C_FROZEN_VERDICT,
    B0D_CONFIRMED_VERDICT,
    run_matrix_b0d,
)
from .evaluation import Action, Arm, Evidence, Hazard, ProgressStatus, Truth, run_episode


PROTOCOL_ID = "L10M-B0-CLOSURE-FREEZE-V1"
VERDICT = "B0_CLOSED_B0_E_NOT_REQUIRED"
EXPECTED_SOURCE_SHA256 = {
    "scripts/research/l10m_b0/evaluation.py": "98628dc63ad0d58f96ec45cc7be0a72b0020cbe567934e8617f64b5ae453ce47",
    "scripts/research/l10m_b0/scenario_matrix.py": "7f2942f72a980502a352b83c6faeb392a65b82cbb0616c0ac5f716d6e3e8abb8",
    "scripts/research/l10m_b0/b0c_precedence.py": "35211316892ead6206c9e007ad5336285e6f8cf0ee215991cf128153efe36297",
    "scripts/research/l10m_b0/b0d_arrival_orthogonality.py": "5ba38d5754eaa0e53359c3be34135481dce2d33aa6626be378f063fca8acedd9",
}
EXPECTED_B0C_MATRIX_SHA256 = "5674fd8834c095b8a8331993138b0a77dac1a8a9863af0d735957045172bba34"
EXPECTED_B0D_MATRIX_SHA256 = "b2de9a8613ba4194b50013dd4a619d459ce1ecf8d1c23ab7ec0325180ddf7ba5"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unknown_probe() -> dict[str, object]:
    evidence = [
        Evidence("b0-closure-unknown", 0, 0.0, Hazard.LOW, 0.95, progress_signal=0.0),
        Evidence("b0-closure-unknown", 1, 0.0, Hazard.LOW, 0.40, stale=True, progress_signal=None),
        Evidence("b0-closure-unknown", 2, 0.0, Hazard.LOW, 0.40, conflict=True, progress_signal=None),
    ]
    truth = [Truth("b0-closure-unknown", index, progress=0.0) for index in range(3)]
    result = run_episode(Arm.STATEFUL, evidence, truth)
    return {
        "progress_states": [row.progress_status.value for row in evidence],
        "stuck_detection_step": result.stuck_detection_step,
        "success": result.success,
        "actions": [action.value for action in result.actions],
    }


def build_closure_manifest() -> dict[str, object]:
    root = _repository_root()
    actual_hashes = {path: _sha256(root / path) for path in EXPECTED_SOURCE_SHA256}
    if actual_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("B0 frozen source identity changed")

    b0c = run_matrix_b0c()
    b0d = run_matrix_b0d()
    if b0c["matrix_sha256"] != EXPECTED_B0C_MATRIX_SHA256:
        raise RuntimeError("B0-C frozen result changed")
    if b0d["matrix_sha256"] != EXPECTED_B0D_MATRIX_SHA256:
        raise RuntimeError("B0-D frozen result changed")
    if b0d["verdict"] != B0D_CONFIRMED_VERDICT:
        raise RuntimeError("B0-D terminal invariant is no longer confirmed")
    if b0d["b0c_frozen_observation"]["verdict"] != B0C_FROZEN_VERDICT:
        raise RuntimeError("B0-C frozen verdict changed")

    unknown_probe = _unknown_probe()
    if unknown_probe["stuck_detection_step"] is not None or unknown_probe["success"]:
        raise RuntimeError("UNKNOWN fabricated stuck or arrival state")

    return {
        "protocol_id": PROTOCOL_ID,
        "verdict": VERDICT,
        "b0_e_required": False,
        "claim_ceiling": "synthetic controlled-evidence policy mechanics only; no end-to-end, device, user, or safety-effect claim",
        "frozen_source_sha256": actual_hashes,
        "frozen_result_sha256": {
            "b0c_matrix": b0c["matrix_sha256"],
            "b0d_matrix": b0d["matrix_sha256"],
        },
        "frozen_verdicts": {
            "b0c": B0C_FROZEN_VERDICT,
            "b0d": B0D_CONFIRMED_VERDICT,
        },
        "semantic_locks": {
            "progress_states": [status.value for status in ProgressStatus],
            "stuck_evidence_update": "only observable CONFIRMED_NO_PROGRESS after FORWARD increments; POSITIVE_PROGRESS resets; UNKNOWN_PROGRESS never increments",
            "recovery_enter": "stuck_count >= 2",
            "recovery_exit": "credible POSITIVE_PROGRESS resets stuck_count before ordinary action selection",
            "terminal_truth": "confirmed arrival implies terminal success independent of parent action, stuck_count, or recovery attempts",
            "unknown_truth": "UNKNOWN_PROGRESS cannot fabricate stuck evidence or arrival",
            "unsafe_definition": "action == FORWARD and truth.unsafe_forward",
            "hard_safety_invariant": "the frozen shield may override candidate policy actions but candidate search may not edit the shield",
            "terminal_actions": [Action.STOP.value, Action.FORWARD.value],
        },
        "probe_receipts": {
            "unknown_does_not_fabricate_state": unknown_probe,
            "b0d_invariants": b0d["invariants"],
        },
        "successor": {
            "protocol": "L10M-B1-STRUCTURED-SEARCHABILITY-MATCHED-V1",
            "question": "Does a structured exposure of the same mutable policy space improve searchability?",
            "b0_semantics_mutable": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_closure_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
