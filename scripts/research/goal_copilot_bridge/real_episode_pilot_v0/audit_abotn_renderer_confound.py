"""Bind the sealed WebGL failure to the zero-model official-render comparison.

This is a read-only attribution audit.  It never invokes a renderer, teacher,
provider, or baseline and cannot predict how the provider would behave on the
official pixels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_abotn_renderer_fidelity_confound_audit_v0"
EXPECTED_POSES = (0, 1, 2)
WEBGL_FRAME_INDICES = (2, 7, 12)
FIDELITY_TERMS = ("obscured", "indistinctly", "blurred")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(
    *,
    official_audit_path: Path,
    webgl_receipt_path: Path,
    webgl_frames_dir: Path,
    sealed_receipt_path: Path,
    provider_calls_dir: Path,
) -> dict[str, Any]:
    official = _read_json(official_audit_path)
    webgl = _read_json(webgl_receipt_path)
    sealed = _read_json(sealed_receipt_path)

    if official.get("terminal") != "ABOTN_OFFICIAL_RENDER_CANARY_LOCAL_AUDIT_PASS":
        raise ValueError("official-render local audit is not terminal PASS")
    if webgl.get("terminal") != "ABOTN_WEBGL_ACTION_GRAPH_PIXELS_PASS":
        raise ValueError("WebGL pixel receipt is not terminal PASS")
    if sealed.get("terminal") != "ABOTN_V0_CLOSED_LOOP_ENGINEERING_RUN_COMPLETE":
        raise ValueError("sealed closed-loop receipt is not terminal")
    if sealed.get("rerun_authorized") is not False:
        raise ValueError("sealed receipt no-rerun boundary drift")
    if sealed["episode"].get("failure_class") != "CURRENT_FRAME_GROUNDING_BOTTLENECK":
        raise ValueError("sealed failure class drift")

    official_frames = official.get("frames", [])
    if tuple(row.get("pose_index") for row in official_frames) != EXPECTED_POSES:
        raise ValueError("official pose roster drift")
    official_rows = []
    for row in official_frames:
        frame_path = Path(row["path"])
        if _sha256(frame_path) != row["sha256"]:
            raise ValueError(f"official frame hash mismatch: {frame_path.name}")
        official_rows.append({
            "pose_index": row["pose_index"],
            "path": str(frame_path.resolve()),
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
        })

    by_index = {row["observation_index"]: row for row in webgl.get("frames", [])}
    webgl_rows = []
    for pose_index, frame_index in zip(EXPECTED_POSES, WEBGL_FRAME_INDICES, strict=True):
        row = by_index.get(frame_index)
        expected_id = f"abotn-20260227163550-traj-0-p{pose_index:03d}-yaw-z"
        if row is None or row.get("observation_id") != expected_id:
            raise ValueError(f"WebGL pose binding mismatch at pose {pose_index}")
        frame_path = webgl_frames_dir / f"frame-{frame_index:03d}.png"
        if _sha256(frame_path) != row["sha256"]:
            raise ValueError(f"WebGL frame hash mismatch: {frame_path.name}")
        webgl_rows.append({
            "pose_index": pose_index,
            "observation_id": expected_id,
            "path": str(frame_path.resolve()),
            "sha256": row["sha256"],
            "width": row["pixel_stats"]["width"],
            "height": row["pixel_stats"]["height"],
        })

    trajectory = sealed.get("action_state_trajectory", [])
    if tuple(row.get("pose_index") for row in trajectory) != EXPECTED_POSES:
        raise ValueError("sealed episode pose trajectory drift")
    if any(row.get("viewport_yaw_index") != 0 for row in trajectory):
        raise ValueError("sealed episode did not use the bound yaw-z views")

    provider_rows = []
    for sequence, action_row in enumerate(trajectory, start=1):
        observation_id = action_row["observation_id"]
        message_path = provider_calls_dir / observation_id / "attempt-1-last-message.json"
        message = _read_json(message_path)
        decisions = message.get("decisions", [])
        if len(decisions) != 1 or decisions[0].get("action") != "ABSTAIN":
            raise ValueError(f"provider decision drift: {observation_id}")
        rationale = str(decisions[0].get("rationale", ""))
        matched_terms = [term for term in FIDELITY_TERMS if term in rationale.lower()]
        if not matched_terms:
            raise ValueError(f"no recognized input-fidelity term in rationale: {observation_id}")
        provider_rows.append({
            "sequence": sequence,
            "observation_id": observation_id,
            "path": str(message_path.resolve()),
            "sha256": _sha256(message_path),
            "action": "ABSTAIN",
            "rationale": rationale,
            "matched_fidelity_terms": matched_terms,
        })

    dimensions_differ = any(
        (official_row["width"], official_row["height"])
        != (webgl_row["width"], webgl_row["height"])
        for official_row, webgl_row in zip(official_rows, webgl_rows, strict=True)
    )
    if not dimensions_differ:
        raise ValueError("expected official/WebGL camera envelope difference was absent")
    if webgl.get("renderer", {}).get("kind") != "UNOFFICIAL_WEBGL_MECHANICS_CANARY":
        raise ValueError("WebGL renderer authority drift")

    return {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "terminal": "ABOTN_SEALED_FAILURE_RENDERER_CONFOUNDED",
        "official_audit": {
            "path": str(official_audit_path.resolve()),
            "sha256": _sha256(official_audit_path),
            "frames": official_rows,
            "renderer_authority": "PINNED_OFFICIAL_RENDERER",
        },
        "sealed_webgl": {
            "receipt_path": str(webgl_receipt_path.resolve()),
            "receipt_sha256": _sha256(webgl_receipt_path),
            "frames": webgl_rows,
            "renderer_authority": webgl["renderer"]["kind"],
        },
        "sealed_provider_rationales": provider_rows,
        "bindings": {
            "same_scene": "20260227163550",
            "same_trajectory_pose_indices": list(EXPECTED_POSES),
            "same_front_view_semantics": True,
            "renderer_implementations_differ": True,
            "camera_envelopes_differ": dimensions_differ,
            "all_sealed_abstentions_cite_input_fidelity": True,
        },
        "teacher_calls": 0,
        "provider_calls": 0,
        "baseline_calls": 0,
        "render_calls": 0,
        "sealed_episode_reruns": 0,
        "supported_conclusion": (
            "The sealed CURRENT_FRAME_GROUNDING_BOTTLENECK is renderer-confounded and "
            "cannot be attributed solely to provider semantic selection."
        ),
        "unsupported_conclusions": [
            "provider success on official pixels",
            "functional entrance-region truth",
            "PROPOSAL_MISS versus REFERENT_SELECTION",
            "algorithm accuracy",
        ],
        "claim_ceiling": "READ_ONLY_RENDERER_CONFOUND_ATTRIBUTION_NO_MODEL_REPLAY",
        "next_action": "REQUIRE_A_FRESH_PROSPECTIVE_OFFICIAL_PIXEL_EPISODE_NOT_A_RERUN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-audit", type=Path, required=True)
    parser.add_argument("--webgl-receipt", type=Path, required=True)
    parser.add_argument("--webgl-frames-dir", type=Path, required=True)
    parser.add_argument("--sealed-receipt", type=Path, required=True)
    parser.add_argument("--provider-calls-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        official_audit_path=args.official_audit.resolve(),
        webgl_receipt_path=args.webgl_receipt.resolve(),
        webgl_frames_dir=args.webgl_frames_dir.resolve(),
        sealed_receipt_path=args.sealed_receipt.resolve(),
        provider_calls_dir=args.provider_calls_dir.resolve(),
    )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "terminal": result["terminal"],
        "bound_poses": result["bindings"]["same_trajectory_pose_indices"],
        "provider_calls": result["provider_calls"],
        "render_calls": result["render_calls"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
