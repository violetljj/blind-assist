#!/usr/bin/env python3
"""Create and verify auditable LLM reviews for SANPO sequence drafts.

This tool intentionally separates *model scene review* from dense semantic
annotation.  A review may admit or reject a continuous sequence for the v3
annotation queue, but it cannot promote a draft to training or benchmark data:
four-class masks and the v3 gate remain required for that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_BUCKETS = {
    "parallel_boundary",
    "step_curb",
    "center_obstacle",
    "lateral_pedestrian_or_ebike",
    "low_light",
    "tactile_paving_occupied",
}
ALLOWED_DECISIONS = {"accept_for_dense_annotation", "reject", "needs_recapture"}
ALLOWED_ALERT_OUTCOMES = {"alert", "no_alert"}
PROMPT_VERSION = "sanpo-sequence-model-review-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_for(draft_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sequence_ids = sorted({str(row["sequence_id"]) for row in rows})
    if len(sequence_ids) != 1:
        raise ValueError("a draft root must contain exactly one sequence")
    indexes = sorted(int(row["frame_index"]) for row in rows)
    evidence_indexes = sorted({indexes[0], indexes[len(indexes) // 2], indexes[-1]})
    evidence = []
    for index in evidence_indexes:
        row = next(row for row in rows if int(row["frame_index"]) == index)
        evidence.append({
            "frame_index": index,
            "image_path": row["image_path"],
            "image_sha256": row["source"]["sha256"],
            "qa_overlay_path": f"qa/boxed/{row['id']}.jpg",
        })
    return {
        "format": "blindassist_sanpo_model_review_request_v1",
        "prompt_version": PROMPT_VERSION,
        "draft_manifest": "manifest.draft.jsonl",
        "draft_manifest_sha256": sha256(draft_root / "manifest.draft.jsonl"),
        "sequence_id": sequence_ids[0],
        "frame_count": len(rows),
        "evidence_frames": evidence,
        "instructions": [
            "Review only the supplied local images and overlays. Do not infer unseen frames.",
            "Classify the primary scene bucket, whether a relevant event enters the walking corridor, and the expected alert outcome.",
            "A curb or stairs outside the corridor is boundary evidence, not a free-standing obstacle alert.",
            "Use reject or needs_recapture when evidence is ambiguous. Do not invent pixel masks.",
            "Return reviewer type=model, model identifier, model version/date, decision, confidence 0..1, evidence frame indexes, and concise rationale.",
        ],
        "response_schema": {
            "reviewer": {"type": "model", "model": "string", "version_or_date": "string"},
            "decision": "accept_for_dense_annotation | reject | needs_recapture",
            "primary_scene_bucket": "one allowed bucket",
            "corridor_event_present": "boolean",
            "expected_alert_outcome": "alert | no_alert",
            "confidence": "number 0..1",
            "evidence_frame_indexes": "array matching supplied evidence frames",
            "rationale": "string",
            "limitations": "string",
        },
        "promotion_rule": "This response only selects a draft for dense annotation; it does not make v3 data benchmark-ready.",
    }


def validate_response(request: dict[str, Any], response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reviewer = response.get("reviewer") if isinstance(response.get("reviewer"), dict) else {}
    if reviewer.get("type") != "model":
        errors.append("reviewer.type must be model")
    if not str(reviewer.get("model", "")).strip() or not str(reviewer.get("version_or_date", "")).strip():
        errors.append("model reviewer identifier and version_or_date are required")
    if response.get("decision") not in ALLOWED_DECISIONS:
        errors.append("unsupported decision")
    if response.get("primary_scene_bucket") not in ALLOWED_BUCKETS:
        errors.append("unsupported primary_scene_bucket")
    if not isinstance(response.get("corridor_event_present"), bool):
        errors.append("corridor_event_present must be boolean")
    if response.get("expected_alert_outcome") not in ALLOWED_ALERT_OUTCOMES:
        errors.append("expected_alert_outcome must be alert or no_alert")
    confidence = response.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be in [0, 1]")
    expected = {item["frame_index"] for item in request["evidence_frames"]}
    actual = set(response.get("evidence_frame_indexes", []))
    if actual != expected:
        errors.append("evidence_frame_indexes must match all supplied evidence frames")
    for field in ("rationale", "limitations"):
        if not str(response.get(field, "")).strip():
            errors.append(f"missing {field}")
    if response.get("decision") == "accept_for_dense_annotation":
        if float(confidence or 0) < 0.85:
            errors.append("accepted model review requires confidence >= 0.85")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--response", type=Path, help="Model response JSON to validate and record.")
    parser.add_argument("--request-output", type=Path, help="Defaults to qa/model_review_request.json")
    parser.add_argument("--result-output", type=Path, help="Defaults to qa/model_review_result.json")
    args = parser.parse_args()
    root = args.draft_root.resolve()
    manifest = root / "manifest.draft.jsonl"
    rows = load_jsonl(manifest)
    request = request_for(root, rows)
    request_output = (args.request_output or root / "qa" / "model_review_request.json").resolve()
    write_json(request_output, request)
    if not args.response:
        print(f"model_review_request={request_output}")
        return 0
    response = json.loads(args.response.read_text(encoding="utf-8"))
    errors = validate_response(request, response)
    result = {
        "format": "blindassist_sanpo_model_review_result_v1",
        "request_sha256": sha256(request_output),
        "draft_manifest_sha256": request["draft_manifest_sha256"],
        "response": response,
        "ok": not errors,
        "errors": errors,
        "promotion": "dense_annotation_queue" if not errors and response["decision"] == "accept_for_dense_annotation" else "not_promoted",
    }
    output = (args.result_output or root / "qa" / "model_review_result.json").resolve()
    write_json(output, result)
    print(json.dumps({"ok": result["ok"], "promotion": result["promotion"], "output": str(output)}, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
