"""Oracle-candidate diagnostic for competing physical-instance hypotheses."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import visible_identity_probe as base


SCHEMA_VERSION = "blindassist_oracle_competing_identity_probe_v0"
EXPECTED_WRONG_PAIRS = 4
EXPECTED_CONTROL_PAIRS = 13
CROP_SIZE = 384
CROP_CONTEXT_FRACTION = 0.20
CLAIM_CEILING = (
    "ORACLE_CANDIDATE_CONSUMED_VISIBLE_ONLY_IDENTITY_DISCRIMINATION_DIAGNOSTIC_"
    "NO_CANDIDATE_GENERATION_BELIEF_TRACKING_ACTIVE_SEARCH_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM"
)
OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["A", "B", "CONTESTED", "NEITHER"]},
    },
    "required": ["decision"],
    "additionalProperties": False,
}
PROMPT = """You are an identity verifier, not an object detector. Exactly four images are attached in this order:
1. the clean reference scene;
2. the same reference scene with a red rectangle marking the reference target;
3. current candidate A;
4. current candidate B.

Candidate A and candidate B are real objects of the same category in one later view. A green rectangle marks the candidate object inside each crop; green is not part of its appearance. Decide which candidate is the exact same physical object instance as the marked reference target.

Return A or B only when the visual evidence supports that exact physical identity, not merely the same category. Return CONTESTED when both remain plausible or the evidence cannot distinguish them. Return NEITHER when neither candidate matches. Do not assume that one candidate must be correct. Do not inspect files, call tools, browse, or use information beyond the four attached images. Return only the JSON object required by the output schema.
"""


def _square_crop_bounds(bbox: Sequence[float], context_fraction: float = CROP_CONTEXT_FRACTION) -> list[float]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width = x1 - x0
    height = y1 - y0
    base._require(width > 0.0 and height > 0.0, "candidate bbox is degenerate")
    side = min(1.0, max(width, height) * (1.0 + 2.0 * context_fraction))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    crop_x0 = min(max(0.0, center_x - side / 2.0), 1.0 - side)
    crop_y0 = min(max(0.0, center_y - side / 2.0), 1.0 - side)
    return [crop_x0, crop_y0, crop_x0 + side, crop_y0 + side]


def _render_candidate(image_path: Path, instance_bbox: Sequence[float], output_path: Path) -> list[float]:
    crop_bbox = _square_crop_bounds(instance_bbox)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    left = int(round(crop_bbox[0] * width))
    top = int(round(crop_bbox[1] * height))
    right = int(round(crop_bbox[2] * width))
    bottom = int(round(crop_bbox[3] * height))
    crop = image.crop((left, top, right, bottom)).resize(
        (CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(crop)
    crop_width = max(1e-9, crop_bbox[2] - crop_bbox[0])
    crop_height = max(1e-9, crop_bbox[3] - crop_bbox[1])
    marked = [
        (float(instance_bbox[0]) - crop_bbox[0]) / crop_width * CROP_SIZE,
        (float(instance_bbox[1]) - crop_bbox[1]) / crop_height * CROP_SIZE,
        (float(instance_bbox[2]) - crop_bbox[0]) / crop_width * CROP_SIZE,
        (float(instance_bbox[3]) - crop_bbox[1]) / crop_height * CROP_SIZE,
    ]
    for offset in range(4):
        draw.rectangle(
            (
                int(round(marked[0])) - offset,
                int(round(marked[1])) - offset,
                int(round(marked[2])) + offset,
                int(round(marked[3])) + offset,
            ),
            outline=(0, 255, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="PNG")
    return crop_bbox


def _same_class_distractors(private_input: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in private_input["native_instances"]
        if item["physical_instance_id"] != private_input["target_physical_instance_id"]
        and item["normalized_label"] == private_input["target_normalized_label"]
        and float(item["bbox_area_fraction"]) >= base.c2.DISTRACTOR_MIN_AREA
    ]


def _select_competitor(row: Mapping[str, Any], distractors: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if row["identity_outcome"] == "SAME_CLASS_DISTRACTOR":
        assigned_id = int(row["assigned_instance"]["native_object_id"])
        return next(item for item in distractors if int(item["native_object_id"]) == assigned_id)
    return sorted(
        distractors,
        key=lambda item: (-float(item["bbox_area_fraction"]), int(item["native_object_id"])),
    )[0]


def _target_position(stratum_index: int) -> str:
    return "A" if stratum_index % 2 == 0 else "B"


def _swapped_private_pair(
    source_private: Mapping[str, Any], pair_id: str, source_pair_id: str
) -> dict[str, Any]:
    return {
        **source_private,
        "pair_id": pair_id,
        "source_pair_id": source_pair_id,
        "target_position": "B" if source_private["target_position"] == "A" else "A",
        "candidate_a_physical_instance_id": source_private["candidate_b_physical_instance_id"],
        "candidate_b_physical_instance_id": source_private["candidate_a_physical_instance_id"],
        "candidate_a_native_object_id": source_private["candidate_b_native_object_id"],
        "candidate_b_native_object_id": source_private["candidate_a_native_object_id"],
        "crop_bounds_xyxy_normalized": {
            "A": source_private["crop_bounds_xyxy_normalized"]["B"],
            "B": source_private["crop_bounds_xyxy_normalized"]["A"],
        },
        "counterbalance": "A_B_IMAGES_SWAPPED_FROM_PARENT_PAIR",
    }


def prepare_run(parent_run: Path, run_dir: Path, provider: Mapping[str, Any]) -> list[dict[str, Any]]:
    base._require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    parent_report = base._load_json(parent_run / "final-report.json")
    base._verify_body_hash(parent_report, "parent visible identity report")
    base._require(parent_report["model"] == base.MODEL, "parent model drifted")
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    excluded_correct_without_distractor = 0
    for row in parent_report["rows"]:
        if row["identity_outcome"] not in {"SAME_INSTANCE", "SAME_CLASS_DISTRACTOR"}:
            continue
        private_input = base._load_json(
            parent_run / "evaluator-private" / "observations" / f"{row['observation_id']}.json"
        )
        distractors = _same_class_distractors(private_input)
        if not distractors:
            if row["identity_outcome"] == "SAME_INSTANCE":
                excluded_correct_without_distractor += 1
            continue
        target = next(
            item
            for item in private_input["native_instances"]
            if item["physical_instance_id"] == private_input["target_physical_instance_id"]
        )
        selected.append((row, private_input, target, _select_competitor(row, distractors)))
    wrong_count = sum(item[0]["identity_outcome"] == "SAME_CLASS_DISTRACTOR" for item in selected)
    control_count = sum(item[0]["identity_outcome"] == "SAME_INSTANCE" for item in selected)
    base._require(wrong_count == EXPECTED_WRONG_PAIRS, "historical wrong-pair count drifted")
    base._require(control_count == EXPECTED_CONTROL_PAIRS, "eligible control-pair count drifted")
    base._require(excluded_correct_without_distractor == 3, "no-distractor correct count drifted")
    run_dir.mkdir(parents=True, exist_ok=False)
    public_root = run_dir / "provider-public"
    private_root = run_dir / "evaluator-private"
    stratum_indices = {"HISTORICAL_WRONG": 0, "BASELINE_CORRECT_CONTROL": 0}
    pairs = []
    for pair_number, (row, private_input, target, competitor) in enumerate(selected, start=1):
        pair_id = f"pair-{pair_number:03d}"
        stratum = "HISTORICAL_WRONG" if row["identity_outcome"] == "SAME_CLASS_DISTRACTOR" else "BASELINE_CORRECT_CONTROL"
        target_position = _target_position(stratum_indices[stratum])
        stratum_indices[stratum] += 1
        candidates = {target_position: target, "B" if target_position == "A" else "A": competitor}
        parent_workspace = parent_run / "provider-public" / "cases" / row["observation_id"]
        workspace = public_root / "pairs" / pair_id
        workspace.mkdir(parents=True, exist_ok=False)
        shutil.copy2(parent_workspace / "01-reference.jpg", workspace / "01-reference.jpg")
        shutil.copy2(parent_workspace / "02-reference-target.png", workspace / "02-reference-target.png")
        crop_bounds = {}
        for position in ("A", "B"):
            crop_bounds[position] = _render_candidate(
                parent_workspace / "03-later.jpg",
                candidates[position]["bbox_xyxy_normalized"],
                workspace / f"0{3 if position == 'A' else 4}-candidate-{position.lower()}.png",
            )
        base._atomic_text(workspace / "prompt.txt", PROMPT)
        base._atomic_json(workspace / "output-schema.json", OUTPUT_SCHEMA)
        base._atomic_json(
            workspace / "public-input.json",
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "roles": ["CLEAN_REFERENCE", "PUBLIC_TARGET_OVERLAY", "CANDIDATE_A", "CANDIDATE_B"],
                "candidate_category_relation": "SAME_CLASS",
            },
        )
        base._assert_provider_firewall(workspace)
        private_pair = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "case_id": row["case_id"],
            "observation_id": row["observation_id"],
            "stratum": stratum,
            "baseline_identity_outcome": row["identity_outcome"],
            "target_position": target_position,
            "candidate_a_physical_instance_id": candidates["A"]["physical_instance_id"],
            "candidate_b_physical_instance_id": candidates["B"]["physical_instance_id"],
            "candidate_a_native_object_id": candidates["A"]["native_object_id"],
            "candidate_b_native_object_id": candidates["B"]["native_object_id"],
            "crop_bounds_xyxy_normalized": crop_bounds,
        }
        private_path = private_root / "pairs" / f"{pair_id}.json"
        base._atomic_json(private_path, private_pair)
        pairs.append(
            {
                "pair_id": pair_id,
                "workspace_relative_path": workspace.relative_to(run_dir).as_posix(),
                "private_input_relative_path": private_path.relative_to(run_dir).as_posix(),
            }
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": base._utc_now(),
        "mode": "REVERSIBLE_EXPLORATION_ORACLE_CANDIDATE_DIAGNOSTIC",
        "provider": dict(provider),
        "parent_run": str(parent_run),
        "parent_report_body_sha256": parent_report["body_sha256"],
        "pair_selection": {
            "historical_wrong": "EXACT_BASELINE_COMMITTED_SAME_CLASS_INSTANCE",
            "baseline_correct_control": "LARGEST_AREA_SAME_CLASS_DISTRACTOR",
            "minimum_distractor_area_fraction": base.c2.DISTRACTOR_MIN_AREA,
        },
        "candidate_representation": {
            "square_crop_pixels": CROP_SIZE,
            "context_fraction_per_side": CROP_CONTEXT_FRACTION,
            "candidate_region_marker": "GREEN_RECTANGLE",
        },
        "target_position_rule": "ALTERNATE_A_B_INDEPENDENTLY_WITHIN_EACH_STRATUM",
        "historical_wrong_pair_count": wrong_count,
        "baseline_correct_control_pair_count": control_count,
        "baseline_correct_total_before_oracle_filter": 16,
        "baseline_correct_excluded_no_same_class_candidate_count": excluded_correct_without_distractor,
        "retry_policy": "NO_AUTOMATIC_RETRY_DISPATCHED_OR_COMPLETED_CALLS",
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "pairs": pairs,
    }
    config["body_sha256"] = base._body_hash(config)
    base._atomic_json(run_dir / "run-config.json", config)
    return pairs


def prepare_order_counterbalance(parent_oracle_run: Path, run_dir: Path, provider: Mapping[str, Any]) -> list[dict[str, Any]]:
    base._require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    parent_report = base._load_json(parent_oracle_run / "final-report.json")
    base._verify_body_hash(parent_report, "parent oracle report")
    source_rows = [row for row in parent_report["rows"] if row["stratum"] == "HISTORICAL_WRONG"]
    base._require(len(source_rows) == EXPECTED_WRONG_PAIRS, "counterbalance source count drifted")
    run_dir.mkdir(parents=True, exist_ok=False)
    public_root = run_dir / "provider-public"
    private_root = run_dir / "evaluator-private"
    pairs = []
    for index, row in enumerate(source_rows, start=1):
        source_pair_id = row["pair_id"]
        pair_id = f"swap-{index:03d}"
        source_workspace = parent_oracle_run / "provider-public" / "pairs" / source_pair_id
        source_private = base._load_json(parent_oracle_run / "evaluator-private" / "pairs" / f"{source_pair_id}.json")
        workspace = public_root / "pairs" / pair_id
        workspace.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_workspace / "01-reference.jpg", workspace / "01-reference.jpg")
        shutil.copy2(source_workspace / "02-reference-target.png", workspace / "02-reference-target.png")
        shutil.copy2(source_workspace / "04-candidate-b.png", workspace / "03-candidate-a.png")
        shutil.copy2(source_workspace / "03-candidate-a.png", workspace / "04-candidate-b.png")
        base._atomic_text(workspace / "prompt.txt", PROMPT)
        base._atomic_json(workspace / "output-schema.json", OUTPUT_SCHEMA)
        base._atomic_json(
            workspace / "public-input.json",
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "roles": ["CLEAN_REFERENCE", "PUBLIC_TARGET_OVERLAY", "CANDIDATE_A", "CANDIDATE_B"],
                "candidate_category_relation": "SAME_CLASS",
                "counterbalance": "A_B_IMAGES_SWAPPED",
            },
        )
        base._assert_provider_firewall(workspace)
        private_pair = _swapped_private_pair(source_private, pair_id, source_pair_id)
        private_path = private_root / "pairs" / f"{pair_id}.json"
        base._atomic_json(private_path, private_pair)
        pairs.append(
            {
                "pair_id": pair_id,
                "workspace_relative_path": workspace.relative_to(run_dir).as_posix(),
                "private_input_relative_path": private_path.relative_to(run_dir).as_posix(),
            }
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": base._utc_now(),
        "mode": "POSTHOC_ORDER_COUNTERBALANCE_DIAGNOSTIC",
        "provider": dict(provider),
        "parent_run": str(parent_oracle_run),
        "parent_report_body_sha256": parent_report["body_sha256"],
        "pair_selection": "ALL_FOUR_HISTORICAL_WRONG_PAIRS_FROM_PARENT_ORACLE_RUN",
        "intervention": "SWAP_CANDIDATE_A_AND_B_IMAGES_ONLY",
        "historical_wrong_pair_count": len(pairs),
        "baseline_correct_control_pair_count": 0,
        "baseline_correct_total_before_oracle_filter": 0,
        "baseline_correct_excluded_no_same_class_candidate_count": 0,
        "retry_policy": "NO_AUTOMATIC_RETRY_DISPATCHED_OR_COMPLETED_CALLS",
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "pairs": pairs,
    }
    config["body_sha256"] = base._body_hash(config)
    base._atomic_json(run_dir / "run-config.json", config)
    return pairs


def _provider_command(codex_exe: Path, workspace: Path) -> list[str]:
    return [
        str(codex_exe),
        "exec",
        "--model",
        base.MODEL,
        "--config",
        f'model_reasoning_effort="{base.REASONING_EFFORT}"',
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "--cd",
        str(workspace),
        "--output-schema",
        str(workspace / "output-schema.json"),
        "--output-last-message",
        str(workspace / "last-message.json"),
        PROMPT,
        "--image",
        str(workspace / "01-reference.jpg"),
        "--image",
        str(workspace / "02-reference-target.png"),
        "--image",
        str(workspace / "03-candidate-a.png"),
        "--image",
        str(workspace / "04-candidate-b.png"),
    ]


def _parse_decision(path: Path) -> str:
    value = base._load_json(path)
    base._require(set(value) == {"decision"}, "provider output fields are invalid")
    base._require(value["decision"] in {"A", "B", "CONTESTED", "NEITHER"}, "provider decision is invalid")
    return str(value["decision"])


def _assert_no_tool_calls(stdout: str) -> None:
    for line in stdout.splitlines():
        event = json.loads(line)
        item = event.get("item")
        if item is not None:
            base._require(item.get("type") == "agent_message", "provider used a non-message tool item")


def _run_pair(codex_exe: Path, run_dir: Path, pair: Mapping[str, Any], timeout_s: int) -> dict[str, Any]:
    workspace = run_dir / pair["workspace_relative_path"]
    dispatch = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair["pair_id"],
        "state": "DISPATCHED",
        "dispatched_at_utc": base._utc_now(),
    }
    base._require(not (workspace / "dispatch.json").exists(), f"{pair['pair_id']} was already dispatched")
    base._atomic_json(workspace / "dispatch.json", dispatch)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            _provider_command(codex_exe, workspace),
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            shell=False,
        )
        base._atomic_text(workspace / "stdout.jsonl", completed.stdout)
        base._atomic_text(workspace / "stderr.txt", completed.stderr)
        state = "COMPLETED" if completed.returncode == 0 else "PROVIDER_ERROR"
        if state == "COMPLETED":
            try:
                _assert_no_tool_calls(completed.stdout)
                _parse_decision(workspace / "last-message.json")
            except (ValueError, json.JSONDecodeError, OSError, base.ProbeError):
                state = "INVALID_OUTPUT_OR_TOOL_USE"
        result = {
            **dispatch,
            "state": state,
            "completed_at_utc": base._utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "returncode": completed.returncode,
        }
    except subprocess.TimeoutExpired as error:
        base._atomic_text(workspace / "stdout.jsonl", error.stdout or "")
        base._atomic_text(workspace / "stderr.txt", error.stderr or "")
        result = {
            **dispatch,
            "state": "IN_DOUBT_TIMEOUT_NO_RETRY",
            "completed_at_utc": base._utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "returncode": None,
        }
    base._atomic_json(workspace / "provider-result.json", result)
    return result


def execute(codex_exe: Path, run_dir: Path, pairs: Sequence[Mapping[str, Any]], workers: int, timeout_s: int) -> list[dict[str, Any]]:
    first = _run_pair(codex_exe, run_dir, pairs[0], timeout_s)
    print(f"[1/{len(pairs)}] {pairs[0]['pair_id']} {first['state']} {first['duration_seconds']:.1f}s", flush=True)
    base._require(first["state"] == "COMPLETED", "first pair failed transport qualification")
    results = [first]
    completed_count = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_pair, codex_exe, run_dir, pair, timeout_s): pair for pair in pairs[1:]}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count += 1
            print(
                f"[{completed_count}/{len(pairs)}] {result['pair_id']} {result['state']} {result['duration_seconds']:.1f}s",
                flush=True,
            )
    return results


def summarize(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    def counts(stratum: str) -> dict[str, int]:
        subset = [row for row in rows if row["stratum"] == stratum]
        return {
            "pair_count": len(subset),
            "target_selected_count": sum(row["evaluation"] == "TARGET_SELECTED" for row in subset),
            "distractor_selected_count": sum(row["evaluation"] == "DISTRACTOR_SELECTED" for row in subset),
            "contested_count": sum(row["evaluation"] == "CONTESTED" for row in subset),
            "neither_count": sum(row["evaluation"] == "NEITHER" for row in subset),
            "provider_failure_count": sum(row["evaluation"] == "PROVIDER_FAILURE" for row in subset),
        }

    return {
        "oracle_pair_count": len(rows),
        "historical_wrong": counts("HISTORICAL_WRONG"),
        "baseline_correct_control": counts("BASELINE_CORRECT_CONTROL"),
        "baseline_correct_total_before_oracle_filter": config["baseline_correct_total_before_oracle_filter"],
        "baseline_correct_excluded_no_same_class_candidate_count": config[
            "baseline_correct_excluded_no_same_class_candidate_count"
        ],
        "target_position_a_count": sum(row["target_position"] == "A" for row in rows),
        "target_position_b_count": sum(row["target_position"] == "B" for row in rows),
        "provider_decision_a_count": sum(row.get("provider_decision") == "A" for row in rows),
        "provider_decision_b_count": sum(row.get("provider_decision") == "B" for row in rows),
    }


def evaluate(run_dir: Path) -> dict[str, Any]:
    config = base._load_json(run_dir / "run-config.json")
    base._verify_body_hash(config, "oracle run config")
    rows = []
    for pair in config["pairs"]:
        workspace = run_dir / pair["workspace_relative_path"]
        private_input = base._load_json(run_dir / pair["private_input_relative_path"])
        provider_result = base._load_json(workspace / "provider-result.json")
        row = {
            "pair_id": pair["pair_id"],
            "case_id": private_input["case_id"],
            "observation_id": private_input["observation_id"],
            "stratum": private_input["stratum"],
            "target_position": private_input["target_position"],
            "provider_state": provider_result["state"],
            "duration_seconds": provider_result["duration_seconds"],
        }
        if provider_result["state"] != "COMPLETED":
            row["evaluation"] = "PROVIDER_FAILURE"
        else:
            decision = _parse_decision(workspace / "last-message.json")
            row["provider_decision"] = decision
            if decision in {"A", "B"}:
                row["evaluation"] = "TARGET_SELECTED" if decision == private_input["target_position"] else "DISTRACTOR_SELECTED"
            else:
                row["evaluation"] = decision
        rows.append(row)
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": base._utc_now(),
        "mode": config["mode"],
        "run_config_body_sha256": config["body_sha256"],
        "parent_report_body_sha256": config["parent_report_body_sha256"],
        "model": config["provider"]["model"],
        "reasoning_effort": config["provider"]["reasoning_effort"],
        "metrics": summarize(rows, config),
        "rows": rows,
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": (
            "ORACLE_COMPETING_IDENTITY_ORDER_COUNTERBALANCE_OBSERVED"
            if config["mode"] == "POSTHOC_ORDER_COUNTERBALANCE_DIAGNOSTIC"
            else "ORACLE_COMPETING_IDENTITY_DIAGNOSTIC_OBSERVED"
        ),
    }
    report["body_sha256"] = base._body_hash(report)
    base._atomic_json(run_dir / "final-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parent_group = parser.add_mutually_exclusive_group(required=True)
    parent_group.add_argument("--parent-run", type=Path)
    parent_group.add_argument("--counterbalance-parent-run", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    args = parser.parse_args(argv)
    base._require(1 <= args.workers <= 3, "workers must be between 1 and 3")
    provider = base.preflight_codex(args.codex_exe.resolve(), base.MODEL)
    parent_path = (args.parent_run or args.counterbalance_parent_run).resolve()
    parent_config = base._load_json(parent_path / "run-config.json")
    base._require(parent_config["provider"] == provider, "provider identity drifted from parent probe")
    pairs = (
        prepare_order_counterbalance(parent_path, args.run_dir.resolve(), provider)
        if args.counterbalance_parent_run is not None
        else prepare_run(parent_path, args.run_dir.resolve(), provider)
    )
    results = execute(args.codex_exe.resolve(), args.run_dir.resolve(), pairs, args.workers, args.timeout_seconds)
    report = evaluate(args.run_dir.resolve())
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if all(result["state"] == "COMPLETED" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
