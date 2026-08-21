"""Audit whether consumed Development parents can support P0-D2 fitting."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.research.goal_copilot_bridge.p0_d2_calibration import core
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    cohorts = [json.loads(path.read_text(encoding="utf-8")) for path in args.cohort]
    report = core.audit_data_frontdoor(cohorts)
    report["input_cohort_report_sha256s"] = [str(value["report_sha256"]) for value in cohorts]
    report["report_sha256"] = materializer.content_sha256(report)
    materializer.write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
