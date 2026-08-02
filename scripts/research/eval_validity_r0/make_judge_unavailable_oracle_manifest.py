"""Write an explicit unavailable-oracle receipt after a burned pilot hold.

This is not a trace and contains no invented metrics.  It lets the audit
report separate primitive/pair findings from the missing native and
system-chain oracle inputs without turning absence into negative evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import PROTOCOL_ID, read_json
from .judge_audit import ORACLE_ARMS, ORACLE_OPPORTUNITY_FIELDS, ORACLE_SCHEMA


def build(*, contract: dict, reason: str, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite oracle receipt: {output}")
    if not reason.strip():
        raise ValueError("oracle unavailability reason must be non-empty")
    opportunity = {
        "eligible_event_ids": [],
        "eligible_for_native_task": False,
        "eligible_for_system_chain": False,
        "required_inputs": [],
        "expected_improvement_dimension": [],
        "not_evaluable_reason": reason,
    }
    system = {
        "current_yolo": {"available": False, "not_evaluable_reason": reason},
        **{
            arm: {"available": False, "not_evaluable_reason": reason, "opportunity": opportunity}
            for arm in ORACLE_ARMS
        },
    }
    native = {
        arm: {"available": False, "not_evaluable_reason": reason, "opportunity": opportunity}
        for arm in ORACLE_ARMS
    }
    result = {
        "schema_version": ORACLE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "shared_execution": contract["shared_execution"],
        "status": "ORACLE_INPUTS_NOT_GENERATED_NOT_EVALUABLE",
        "system_chain": system,
        "native_information_ceiling": native,
        "not_evaluable_reason": reason,
        "metrics_present": False,
        "required_opportunity_fields": list(ORACLE_OPPORTUNITY_FIELDS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(contract=read_json(args.contract), reason=args.reason, output=args.output)
    print(f"status={result['status']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
