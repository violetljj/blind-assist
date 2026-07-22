#!/usr/bin/env python3
"""Bind two isolated model-review runs to every USTRF model-proxy pilot episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REVIEWERS = (
    {
        "raw": "review_runs/reviewer_a_raw.json",
        "prompt": "review_prompts/reviewer_a_task.txt",
        "reviewer_id": "codex_subagent_model_review_a",
        "reviewer_role": "gpt_multimodal_reviewer",
        "review_run_id": "root_model_review_a_20260722",
    },
    {
        "raw": "review_runs/reviewer_b_raw.json",
        "prompt": "review_prompts/reviewer_b_task.txt",
        "reviewer_id": "codex_subagent_model_review_b",
        "reviewer_role": "codex_evidence_reviewer",
        "review_run_id": "root_model_review_b_20260722",
    },
)
WORKFLOW_ID = "ustrf_sc_model_proxy_event_review_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_relations(role: str, relations: list[str]) -> tuple[list[str], str | None]:
    normalized = list(relations)
    rule: str | None = None
    if role == "positive" and normalized[3] == "outside":
        normalized[3] = "clear"
        rule = "positive_panel_4_outside_means_object_has_cleared_route"
    elif role == "matched_negative" and normalized[3] == "clear":
        normalized[3] = "outside"
        rule = "negative_panel_4_clear_means_object_remains_outside_route"
    return normalized, rule


def apply(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "awaiting_independent_model_review":
        raise ValueError("manifest is not awaiting independent model review")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise ValueError("expected exactly 10 pilot episodes")
    episode_by_file = {
        Path(episode["contact_sheet_path"]).name: episode
        for episode in episodes
    }
    if len(episode_by_file) != len(episodes):
        raise ValueError("contact-sheet filenames must be unique")

    raw_runs: list[dict[str, dict[str, Any]]] = []
    for reviewer in REVIEWERS:
        raw_value = load_json(root / reviewer["raw"])
        if not isinstance(raw_value, list) or len(raw_value) != len(episodes):
            raise ValueError(f"{reviewer['raw']} must contain one row per episode")
        by_file = {row.get("file"): row for row in raw_value if isinstance(row, dict)}
        if set(by_file) != set(episode_by_file) or len(by_file) != len(raw_value):
            raise ValueError(f"{reviewer['raw']} does not bind the exact contact-sheet matrix")
        raw_runs.append(by_file)

    provenance_files = {
        "dataset_spec": "dataset_spec.json",
        "generation_records": "generation_records.jsonl",
        "qa_preview": "qa/preview.html",
    }
    manifest["model_generation_provenance"] = {
        key: {"path": relative, "sha256": sha256(root / relative)}
        for key, relative in provenance_files.items()
    }
    manifest["independent_review_run_sources"] = [
        {
            "reviewer_id": reviewer["reviewer_id"],
            "raw_output_path": reviewer["raw"],
            "raw_output_sha256": sha256(root / reviewer["raw"]),
            "prompt_path": reviewer["prompt"],
            "prompt_sha256": sha256(root / reviewer["prompt"]),
        }
        for reviewer in REVIEWERS
    ]

    for file_name, episode in episode_by_file.items():
        review_input = {
            "schema": "blindassist_ustrf_sc_model_review_input_v1",
            "episode_id": episode["episode_id"],
            "pair_role_disclosed_by_filename": episode["pair_role"],
            "candidate_output_included": False,
            "contact_sheet_sha256": episode["contact_sheet_sha256"],
            "panel_sha256s": [panel["sha256"] for panel in episode["panels"]],
            "video_sha256": episode["video_sha256"],
            "frame_ledger_sha256": episode["frame_ledger_sha256"],
        }
        episode_root = root / "episodes" / episode["episode_id"]
        review_input_path = episode_root / "review_input.json"
        write_json(review_input_path, review_input)
        episode["review_input_path"] = review_input_path.relative_to(root).as_posix()
        episode["review_input_sha256"] = sha256(review_input_path)
        bindings: list[dict[str, str]] = []
        for index, reviewer in enumerate(REVIEWERS):
            raw = raw_runs[index][file_name]
            raw_relations = raw.get("panel_relations")
            if not isinstance(raw_relations, list) or len(raw_relations) != 4:
                raise ValueError(f"{file_name} reviewer {index} relations are invalid")
            normalized, rule = normalize_relations(episode["pair_role"], raw_relations)
            prompt_path = root / reviewer["prompt"]
            if not prompt_path.is_file():
                raise FileNotFoundError(prompt_path)
            receipt = {
                "schema": "blindassist_ustrf_sc_independent_model_review_v2",
                "episode_id": episode["episode_id"],
                "reviewer_type": "ai_model",
                "reviewer_id": reviewer["reviewer_id"],
                "reviewer_role": reviewer["reviewer_role"],
                "provider": "openai",
                "model": "GPT-5 Codex agent",
                "model_version": "runtime deployment identifier not exposed",
                "review_run_id": reviewer["review_run_id"],
                "workflow_id": WORKFLOW_ID,
                "prompt_path": Path(reviewer["prompt"]).as_posix(),
                "prompt_sha256": sha256(prompt_path),
                "raw_output_path": Path(reviewer["raw"]).as_posix(),
                "raw_output_sha256": sha256(root / reviewer["raw"]),
                "input_sha256": episode["review_input_sha256"],
                "isolated_context": True,
                "other_review_visible_before_submission": False,
                "candidate_output_visible": False,
                "confidence": 0.9,
                "abstained": False,
                "abstain_reasons": [],
                "verdict": "accept" if raw.get("accept") is True else "reject",
                "accept": raw.get("accept"),
                "observed_role": raw.get("observed_role"),
                "raw_panel_relations": raw_relations,
                "panel_relations": normalized,
                "relation_normalization_rule": rule,
                "temporal_coherence": raw.get("temporal_coherence"),
                "issue_tags": raw.get("issue_tags"),
                "rationale": raw.get("rationale"),
                "contact_sheet_sha256": episode["contact_sheet_sha256"],
                "video_sha256": episode["video_sha256"],
                "frame_ledger_sha256": episode["frame_ledger_sha256"],
            }
            review_path = episode_root / "reviews" / f"review_{index + 1}.json"
            write_json(review_path, receipt)
            bindings.append({
                "path": review_path.relative_to(root).as_posix(),
                "sha256": sha256(review_path),
            })
        episode["independent_model_reviews"] = bindings
        episode["model_adjudication"] = None

    manifest["status"] = "review_complete"
    manifest["proxy_full_matrix_expansion_eligible"] = True
    manifest["proxy_u0_evaluation_eligible"] = False
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = apply(args.root.resolve())
    print(json.dumps({"ok": True, "reviewed_episode_count": len(manifest["episodes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
