#!/usr/bin/env python3
"""Compare inference-only edge event decisions with hash-bound GPT/VLM silver labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import SilverLabelError, load_json, validate


class ComparisonError(ValueError):
    """The edge report cannot be compared safely to a silver manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(silver_manifest: dict[str, Any], edge_report: dict[str, Any], *, silver_path: Path, source_manifest_path: Path) -> dict[str, Any]:
    validate(silver_manifest, source_manifest_path=source_manifest_path)
    if edge_report.get("schema") != "blindassist_public_video_edge_event_report_v1":
        raise ComparisonError("unexpected edge event report schema")
    if edge_report.get("silver_manifest_sha256") != sha256_file(silver_path):
        raise ComparisonError("edge report is not bound to this silver manifest")
    if edge_report.get("human_event_truth_present") is not False:
        raise ComparisonError("edge report must not claim human event truth")
    if edge_report.get("production_model_replacement_authorized") is not False:
        raise ComparisonError("edge report must not authorize production replacement")
    risk_config = edge_report.get("risk_config")
    if not isinstance(risk_config, str) or not risk_config.strip():
        raise ComparisonError("edge report must contain a non-empty risk_config")
    rows = edge_report.get("episodes")
    if not isinstance(rows, list):
        raise ComparisonError("edge report episodes must be a list")
    edge_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("episode_id"), str):
            raise ComparisonError(f"edge report episodes[{index}] has no episode_id")
        if row["episode_id"] in edge_by_id:
            raise ComparisonError(f"duplicate edge episode_id: {row['episode_id']}")
        if not isinstance(row.get("edge_should_alert"), bool):
            raise ComparisonError(f"edge report {row['episode_id']} must contain boolean edge_should_alert")
        edge_by_id[row["episode_id"]] = row
    frames = edge_report.get("frames")
    diagnostics_by_episode: dict[str, dict[str, Any]] = {}
    if frames is not None:
        if not isinstance(frames, list):
            raise ComparisonError("edge report frames must be a list when present")
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("episode_id"), str):
                raise ComparisonError(f"edge report frames[{index}] has no episode_id")
            episode = diagnostics_by_episode.setdefault(
                frame["episode_id"], {"trigger_count": 0, "event_ids": [], "untracked_trigger_count": 0}
            )
            if frame.get("edge_should_alert") is True:
                event_id = frame.get("risk_event_id")
                if not isinstance(event_id, str) or not event_id:
                    episode["untracked_trigger_count"] += 1
                else:
                    episode["event_ids"].append(event_id)
                episode["trigger_count"] += 1
    result_rows: list[dict[str, Any]] = []
    for silver in silver_manifest["episodes"]:
        episode_id = silver["episode_id"]
        edge = edge_by_id.get(episode_id)
        if edge is None:
            raise ComparisonError(f"missing edge result for silver episode: {episode_id}")
        verdict = silver["silver_should_alert"]
        comparable = verdict != "abstain"
        agreement = None if not comparable else edge["edge_should_alert"] == (verdict == "candidate_alert")
        diagnostics = diagnostics_by_episode.get(episode_id)
        trigger_count = diagnostics["trigger_count"] if diagnostics is not None else None
        unique_event_count = len(set(diagnostics["event_ids"])) if diagnostics is not None else None
        untracked_trigger_count = diagnostics["untracked_trigger_count"] if diagnostics is not None else None
        result_rows.append({
            "episode_id": episode_id,
            "silver_should_alert": verdict,
            "edge_should_alert": edge["edge_should_alert"],
            "comparable": comparable,
            "agreement": agreement,
            "edge_event_ids": edge.get("edge_event_ids", []),
            "edge_trigger_count": trigger_count,
            "edge_unique_event_count": unique_event_count,
            "edge_untracked_trigger_count": untracked_trigger_count,
            "edge_duplicate_event_trigger_count": (
                max(0, trigger_count - unique_event_count - untracked_trigger_count)
                if trigger_count is not None and unique_event_count is not None and untracked_trigger_count is not None else None
            ),
        })
    comparable = [row for row in result_rows if row["comparable"]]
    agreements = sum(row["agreement"] is True for row in comparable)
    return {
        "schema": "blindassist_public_video_silver_edge_comparison_v1",
        "silver_manifest_sha256": sha256_file(silver_path),
        "risk_config": risk_config,
        "comparison_rows": result_rows,
        "comparable_episode_count": len(comparable),
        "silver_abstain_count": len(result_rows) - len(comparable),
        "candidate_agreement_count": agreements,
        "candidate_agreement_rate": agreements / len(comparable) if comparable else None,
        "metric_interpretation": "Candidate agreement with model-produced silver labels; event counts diagnose repeat-trigger behavior only. Untracked triggers use an older non-event feedback path. Neither is human-verified accuracy, recall, or false-alert rate.",
        "production_model_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--edge-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ComparisonError(f"refusing to overwrite output: {args.output}")
        report = compare(load_json(args.silver_manifest), load_json(args.edge_report), silver_path=args.silver_manifest, source_manifest_path=args.source_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, **{key: report[key] for key in ("comparable_episode_count", "silver_abstain_count", "candidate_agreement_rate")}}, ensure_ascii=False))
    except (ComparisonError, SilverLabelError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
