"""Evaluation entrypoint for the frozen R0 development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearance_student_mobile_r0 import require_finite_metrics


def validate_report(metrics: dict[str, float]) -> None:
    require_finite_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    candidate = report.get("candidate", {}).get("task", {})
    validate_report({key: float(candidate[key]) for key in ("clearance_mae_m", "collision_agreement", "false_clear_rate_all_known_decisions", "temporal_clearance_delta_mae_m")})
    print(json.dumps({"terminal": report.get("terminal"), "engineering_noninferiority_passed": report.get("engineering_noninferiority_passed"), "candidate_task": candidate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
