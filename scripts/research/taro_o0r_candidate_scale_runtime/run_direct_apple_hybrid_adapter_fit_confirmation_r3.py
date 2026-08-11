#!/usr/bin/env python3
"""Run the one-shot R5 Phase-B normalized-camera-hash repair replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import run_direct_apple_hybrid_adapter_fit_confirmation_r2 as phase_b
from scripts.research.taro_o0r_candidate_scale_runtime.validate_r5_camera_hash_repair_execution_lock import validate_execution_lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = phase_b.execute(args.execution_lock, lock_validator=validate_execution_lock)
    except Exception as error:
        print(json.dumps({"status": "EXECUTION_NOT_STARTED", "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result.get("execution_valid", False)}, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
