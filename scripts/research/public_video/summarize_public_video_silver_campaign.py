#!/usr/bin/env python3
"""Aggregate hash-bound public-video silver/edge comparisons without upgrading their evidence status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class CampaignSummaryError(ValueError):
    """A supplied comparison is not a safe campaign input."""


COMPARISON_SCHEMA = "blindassist_public_video_silver_edge_comparison_v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignSummaryError(f"JSON root must be an object: {path}")
    return value


def summarize(comparisons: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if not comparisons:
        raise CampaignSummaryError("at least one comparison is required")
    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    tracked_trigger_count = 0
    tracked_event_count = 0
    duplicate_event_trigger_count = 0
    untracked_trigger_count = 0
    campaign_risk_config: str | None = None
    for path, comparison in comparisons:
        if comparison.get("schema") != COMPARISON_SCHEMA:
            raise CampaignSummaryError(f"unexpected comparison schema: {path}")
        if comparison.get("production_model_replacement_authorized") is not False:
            raise CampaignSummaryError(f"comparison must not authorize production replacement: {path}")
        risk_config = comparison.get("risk_config")
        if not isinstance(risk_config, str) or not risk_config.strip():
            raise CampaignSummaryError(f"comparison must contain a non-empty risk_config: {path}")
        if campaign_risk_config is None:
            campaign_risk_config = risk_config
        elif risk_config != campaign_risk_config:
            raise CampaignSummaryError(
                f"cannot mix risk_config values in one campaign: {campaign_risk_config} versus {risk_config}: {path}"
            )
        comparison_rows = comparison.get("comparison_rows")
        if not isinstance(comparison_rows, list):
            raise CampaignSummaryError(f"comparison_rows must be a list: {path}")
        comparable_count = 0
        agreement_count = 0
        abstain_count = 0
        for index, row in enumerate(comparison_rows):
            if not isinstance(row, dict) or not isinstance(row.get("episode_id"), str):
                raise CampaignSummaryError(f"invalid comparison row {index}: {path}")
            comparable = row.get("comparable")
            agreement = row.get("agreement")
            if not isinstance(comparable, bool):
                raise CampaignSummaryError(f"comparison row missing boolean comparable: {path}")
            if comparable:
                if not isinstance(agreement, bool):
                    raise CampaignSummaryError(f"comparable row missing boolean agreement: {path}")
                comparable_count += 1
                agreement_count += int(agreement)
            elif agreement is not None:
                raise CampaignSummaryError(f"abstained row must have null agreement: {path}")
            else:
                abstain_count += 1
            for field in (
                "edge_trigger_count",
                "edge_unique_event_count",
                "edge_duplicate_event_trigger_count",
                "edge_untracked_trigger_count",
            ):
                if field in row and (not isinstance(row[field], int) or row[field] < 0):
                    raise CampaignSummaryError(f"comparison row has invalid {field}: {path}")
            if isinstance(row.get("edge_trigger_count"), int):
                untracked = row.get("edge_untracked_trigger_count", 0)
                unique = row.get("edge_unique_event_count", 0)
                duplicate = row.get("edge_duplicate_event_trigger_count", 0)
                if not all(isinstance(value, int) for value in (untracked, unique, duplicate)):
                    raise CampaignSummaryError(f"event diagnostics must be complete when trigger count is present: {path}")
                if unique + untracked + duplicate > row["edge_trigger_count"]:
                    raise CampaignSummaryError(f"event diagnostics exceed trigger count: {path}")
                tracked_trigger_count += row["edge_trigger_count"] - untracked
                tracked_event_count += unique
                duplicate_event_trigger_count += duplicate
                untracked_trigger_count += untracked
            rows.append({"comparison_path": str(path), **row})
        if comparison.get("comparable_episode_count") != comparable_count or comparison.get("candidate_agreement_count") != agreement_count or comparison.get("silver_abstain_count") != abstain_count:
            raise CampaignSummaryError(f"comparison aggregate counts do not match rows: {path}")
        reports.append({
            "comparison_path": str(path),
            "silver_manifest_sha256": comparison.get("silver_manifest_sha256"),
            "risk_config": risk_config,
            "episode_count": len(comparison_rows),
            "comparable_episode_count": comparable_count,
            "candidate_agreement_count": agreement_count,
            "silver_abstain_count": abstain_count,
        })
    candidate_count = sum(report["comparable_episode_count"] for report in reports)
    agreement_count = sum(report["candidate_agreement_count"] for report in reports)
    abstain_count = sum(report["silver_abstain_count"] for report in reports)
    return {
        "schema": "blindassist_public_video_silver_campaign_summary_v1",
        "risk_config": campaign_risk_config,
        "report_count": len(reports),
        "episode_count": len(rows),
        "candidate_episode_count": candidate_count,
        "candidate_agreement_count": agreement_count,
        "candidate_agreement_rate": agreement_count / candidate_count if candidate_count else None,
        "silver_abstain_count": abstain_count,
        "event_diagnostics": {
            "tracked_trigger_count": tracked_trigger_count,
            "tracked_unique_event_count": tracked_event_count,
            "tracked_duplicate_event_trigger_count": duplicate_event_trigger_count,
            "untracked_trigger_count": untracked_trigger_count,
            "important_limit": "Diagnostic only: zero duplicate tracked-event triggers does not establish human-verified safety or authorize production promotion."
        },
        "reports": reports,
        "comparison_rows": rows,
        "metric_interpretation": "Aggregate candidate agreement with model-produced silver labels; not human-verified accuracy, recall, false-alert rate, calibration evidence, or a production-promotion gate.",
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise CampaignSummaryError(f"refusing to overwrite output: {args.output}")
        result = summarize([(path, load_json(path)) for path in args.comparison])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, **{key: result[key] for key in ("report_count", "candidate_episode_count", "candidate_agreement_rate", "silver_abstain_count")}}, ensure_ascii=False))
    except (CampaignSummaryError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
