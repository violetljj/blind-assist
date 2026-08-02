"""Create a calibration-only RGB staging root without a materialized manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .materialize_screening_inputs import PLAN_SCHEMA


class RGBStageError(ValueError):
    """Raised when a calibration RGB staging root cannot be created safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGBStageError(message)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def stage(*, source_root: Path, plan: dict[str, Any], event_ids: list[str], output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite RGB staging root: {output}")
    _require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("protocol_id") == PROTOCOL_ID, "native plan schema/protocol mismatch")
    _require(plan.get("candidate_outputs_opened") is False, "native plan is output-contaminated")
    plan_items = {item.get("screening_event_id"): item for item in plan.get("items", []) if isinstance(item, dict)}
    _require(len(event_ids) == len(set(event_ids)) and event_ids, "event IDs must be unique and non-empty")
    _require(set(event_ids) <= set(plan_items), "requested event is not in native plan")
    staged = output.with_name(f".{output.name}.staging")
    try:
        receipts: list[dict[str, Any]] = []
        for event_id in event_ids:
            item = plan_items[event_id]
            frames = item.get("frames")
            _require(isinstance(frames, list) and frames, f"{event_id}: plan frames missing")
            event_receipt = {"event_id": event_id, "frame_count": len(frames), "frames": []}
            for ordinal, frame in enumerate(frames):
                source = source_root / "events" / event_id / "rgb" / f"{ordinal:03d}.png"
                target = staged / "events" / event_id / "rgb" / f"{ordinal:03d}.png"
                _require(source.is_file(), f"{event_id}: missing source RGB frame {ordinal}")
                _link_or_copy(source, target)
                event_receipt["frames"].append({"ordinal": ordinal, "source_frame_index": frame.get("source_frame_index"), "sha256": sha256_file(source)})
            receipts.append(event_receipt)
        receipt = {
            "schema_version": "blindassist.eval_validity_r0.judge_rgb_subset_stage.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "CALIBRATION_RGB_ONLY_STAGING_NO_MATERIALIZED_MANIFEST",
            "source_root": str(source_root),
            "native_plan_sha256": sha256_json(plan),
            "event_ids": event_ids,
            "candidate_outputs_opened": False,
            "source_mask_staged": False,
            "model_oracle_outputs_staged": False,
            "formal_denominator_inclusion": False,
            "items": receipts,
        }
        (staged / "calibration-rgb-stage-receipt.json").parent.mkdir(parents=True, exist_ok=True)
        (staged / "calibration-rgb-stage-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, output)
        return receipt
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--native-plan", type=Path, required=True)
    parser.add_argument("--event-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = stage(
        source_root=args.source_root,
        plan=read_json(args.native_plan),
        event_ids=args.event_ids,
        output=args.output,
    )
    print(f"status={result['status']} event_count={len(result['event_ids'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
