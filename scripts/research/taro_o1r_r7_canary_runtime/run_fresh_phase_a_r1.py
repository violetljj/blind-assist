#!/usr/bin/env python3
"""Environment-repaired one-shot entrypoint for TARO R7 fresh Phase A."""

from __future__ import annotations

from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as base


base.LOCK_SCHEMA = "blindassist.taro.o1r.r7_fresh_phase_a_execution_lock.r1.v1"
base.LOCK_ID = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_SOURCE_AND_MODEL_R1_ONE_SHOT_EXECUTION_LOCK"
base.OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-phase-a-r1"
base.PASS_TERMINAL = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_SEALED_PASS_R1"
base.FAIL_TERMINAL = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_R1_EXECUTION_INVALID"
base.EXPECTED_BINDINGS = {
    **base.EXPECTED_BINDINGS,
    "R0_FAILURE": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-phase-a-r0/failure.json",
    "R0_MANIFEST": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-phase-a-r0/manifest.json",
    "PHASE_A_R1_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a_r1.py",
}


if __name__ == "__main__":
    raise SystemExit(base.main())
