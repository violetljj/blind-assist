#!/usr/bin/env python3
"""Run the stable, dependency-free research contract regression suite.

This is the shared local/CI entry point for contracts that must not depend on a
GPU, Android device, model download, or optional scientific Python packages.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    "scripts/test_validate_research_protocol.py",
    "scripts/test_validate_explicit_route_intent_episode.py",
    "scripts/test_validate_route_conditioned_event_truth.py",
    "scripts/test_generate_sanpo_counterfactual_capture_plan.py",
    "scripts/test_validate_ustrf_sc_route_conditioned_event_pilot.py",
    "scripts/test_run_ustrf_sc_u0_candidate_bundle.py",
    "scripts/test_run_ustrf_sc_u0_android_baseline_adapter.py",
    "scripts/test_run_ustrf_sc_u0_android_bbox_route_adapter.py",
    "scripts/test_generate_ustrf_sc_u0_dense_teacher_loso_artifact.py",
    "scripts/test_validate_ustrf_sc_u0_prediction_bundle.py",
    "scripts/test_evaluate_ustrf_sc_u0_teacher_upper_bound.py",
    "scripts/test_validate_ustrf_sc_device_metric_geometry.py",
    "scripts/test_report_ustrf_sc_research_benchmark.py",
)


def main() -> int:
    results: list[dict[str, object]] = []
    for relative in TEST_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            results.append({"test": relative, "returncode": 2, "error": "missing test file"})
            continue
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=REPO_ROOT,
            check=False,
        )
        results.append({"test": relative, "returncode": completed.returncode})

    failures = [result for result in results if result["returncode"] != 0]
    print(json.dumps({
        "suite": "blindassist_research_contracts_v1",
        "test_file_count": len(results),
        "failure_count": len(failures),
        "results": results,
    }, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
