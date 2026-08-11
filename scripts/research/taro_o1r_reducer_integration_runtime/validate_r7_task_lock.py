from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_BYTES = 4666
EXPECTED_SHA256 = "A3396E9CE2A50D9B5D54122068CAA836D740707309AF19D9D8D995DA964F0C74"
EXPECTED_LOCK_ID = "TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_TASK_LOCK"
EXPECTED_SUCCESSOR = (
    "TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_IMPLEMENTATION_LOCK"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(lock_path: Path) -> dict[str, Any]:
    raw = lock_path.read_bytes()
    _require(len(raw) == EXPECTED_BYTES, "R7 task-lock byte count mismatch")
    _require(
        hashlib.sha256(raw).hexdigest().upper() == EXPECTED_SHA256,
        "R7 task-lock SHA-256 mismatch",
    )
    payload = json.loads(raw.decode("utf-8"))
    _require(payload["lock_id"] == EXPECTED_LOCK_ID, "unexpected R7 lock_id")
    _require(payload["status"] == "FROZEN", "R7 task lock is not frozen")
    _require(
        payload["predecessor"]["terminal"]
        == "TARO_O1R_R6_REDUCER_INTEGRATION_NOT_EVALUABLE_ALL_UNKNOWN",
        "R7 predecessor terminal drifted",
    )

    roles = payload["data_roles"]
    fit = roles["algorithm_design_and_nested_parent_holdout"]
    _require(fit["role"] == "ADAPTER_FIT", "R7 design role must be ADAPTER_FIT")
    _require(fit["parent_count"] == 8 and fit["frame_count"] == 211, "R7 fit roster drifted")
    diagnostic = roles["observed_eval_diagnostic_only"]
    _require(diagnostic["promotion_allowed"] is False, "observed eval parents cannot promote R7")
    _require(
        roles["future_promotion"]["requires_new_parent_disjoint_untouched_cohort"] is True,
        "R7 must require a new untouched promotion cohort",
    )

    canary = payload["fit_only_canary"]
    _require(canary["outer_validation"] == "LEAVE_ONE_PARENT_OUT_8_FOLDS", "R7 holdout drifted")
    _require(canary["no_model_training"] is True and canary["cpu_only"] is True, "R7 must be CPU/no-training")
    gates = canary["held-parent_gates"]
    _require(gates["false_clear_count"] == 0, "R7 false-clear gate must remain zero")
    _require(gates["unknown_is_negative"] is False, "UNKNOWN cannot become a negative label")

    runtime = payload["runtime_contract"]
    _require(runtime["source_phase_has_faro_or_label_input"] is False, "source phase leaked labels")
    _require(runtime["source_phase_outputs_sealed_before_label_join"] is True, "source phase must seal first")
    _require(runtime["occupied_requires_positive_evidence"] is True, "occupied must remain positive-evidence only")
    _require(runtime["missing_nonfinite_or_unbound_inputs_return_unknown"] is True, "missing evidence must remain UNKNOWN")
    _require(payload["unique_successor"] == EXPECTED_SUCCESSOR, "unexpected R7 successor")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock_path", type=Path)
    args = parser.parse_args()
    validate(args.lock_path)
    print("TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_TASK_LOCK_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
