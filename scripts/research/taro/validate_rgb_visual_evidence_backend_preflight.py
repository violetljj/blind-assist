from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "blindassist.taro.rgb_pair_frozen_visual_evidence_backend_preflight.v1"
EXPECTED_STATUS = "ONE_ADMISSIBLE_BACKEND_SELECTED_POSITIVE_ONLY_SHADOW_PROTOCOL_LOCKED"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate(repo_root: Path, protocol_path: Path) -> dict[str, Any]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(payload["schema"] == EXPECTED_SCHEMA, "schema mismatch")
    require(payload["status"] == EXPECTED_STATUS, "status mismatch")
    audit = payload["backend_audit"]
    require(audit["candidate_count"] == 2, "candidate census drift")
    require(audit["admissible_candidate_count"] == 1, "exactly one backend must be admissible")
    selected = audit["selected"]
    require(selected["admission"] == "PASS", "selected backend is not admitted")
    for path_key, size_key, hash_key in (
        ("model_asset", "model_size_bytes", "model_sha256"),
        ("labels_asset", "labels_size_bytes", "labels_sha256"),
    ):
        asset = repo_root / selected[path_key]
        require(asset.is_file(), f"missing asset: {selected[path_key]}")
        require(asset.stat().st_size == selected[size_key], f"asset size drift: {selected[path_key]}")
        require(sha256(asset) == selected[hash_key], f"asset hash drift: {selected[path_key]}")
    rejected = audit["rejected"]
    require(len(rejected) == 1 and rejected[0]["admission"] == "REJECT", "rejection census drift")
    protocol = payload["frozen_shadow_protocol"]
    require(protocol["required_distinct_scene_parents"] >= 4, "scene denominator is too small")
    require(protocol["minimum_evaluable_references_total"] >= 120, "reference denominator is too small")
    require(protocol["same_extra_frame_budget"] is True, "arm budgets must match")
    require(protocol["history_window_ns"] == {"minimum": 150000000, "maximum": 1000000000}, "history window drift")
    require(protocol["gates"]["pose_parent_macro_must_exceed_passive"] is True, "primary comparison gate missing")
    require(protocol["gates"]["zero_source_identity_mismatches"] is True, "identity gate missing")
    unknown = payload["unknown_policy"]
    require(unknown["absence_is_safe"] is False, "absence must never become safe")
    boundary = payload["read_and_mutation_boundary"]
    require(boundary["live_model_runs_before_this_lock"] == 0, "pre-lock model read boundary violated")
    require(boundary["new_training"] is False, "training is forbidden")
    require(boundary["default_app_changes"] is False, "default app mutation is forbidden")
    return {
        "schema": "blindassist_taro_rgb_visual_backend_preflight_validation_v1",
        "status": "PASS",
        "protocol": str(protocol_path.relative_to(repo_root)).replace("\\", "/"),
        "model_sha256": selected["model_sha256"],
        "labels_sha256": selected["labels_sha256"],
        "admissible_candidate_count": audit["admissible_candidate_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="docs/research/taro/TARO_RGB_PAIR_FROZEN_VISUAL_EVIDENCE_BACKEND_PREFLIGHT_2026-08-14.json",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol_path = (repo_root / args.protocol).resolve()
    print(json.dumps(validate(repo_root, protocol_path), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
