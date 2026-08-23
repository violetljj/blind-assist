"""Run the frozen P0 proposal and single-Brain baseline on native CMP door truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as brain


SCHEMA_VERSION = "cmp_facade_native_door_89_run_v0"
IOU_THRESHOLD = 0.5
CLAIM_CEILING = "CMP_FACADE_CURRENT_FRAME_DOOR_ONLY_NO_TRAJECTORY_NO_RANGE_NO_LOST"


class RunError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def truth_box(item: Mapping[str, Any]) -> list[float]:
    door = item["native_xml_door"]
    return [min(door["x"]), min(door["y"]), max(door["x"]), max(door["y"])]


def build_episode(item: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    frame_id = Path(item["rgb_path"]).stem
    candidates = []
    for rank, proposal in enumerate(proposals, start=1):
        x0, y0, x1, y1 = proposal["bbox_xyxy"]
        candidates.append({
            "candidate_id": f"gdino-{frame_id}-{rank:03d}",
            "region": {
                "frame_id": frame_id,
                "coordinate_space": "NORMALIZED_XYXY",
                "x_min": x0 / item["image_width"],
                "y_min": y0 / item["image_height"],
                "x_max": x1 / item["image_width"],
                "y_max": y1 / item["image_height"],
            },
            "category_label": proposal["label"],
            "proposal_score": proposal["score"],
            "provider_rank": rank,
        })
    return {
        "episode_id": f"cmp-door-{index:03d}",
        "goal_text": "the door",
        "image_path": item["absolute_rgb_path"],
        "candidates": candidates,
        "evaluator_episode": {
            "goal_spec": {"target_name": "the door"},
            "observation_window": {"frame_ids": [frame_id], "start_timestamp_ms": index, "end_timestamp_ms": index},
        },
    }


def run_brain(
    *, episodes: Sequence[Mapping[str, Any]], run_dir: Path, executable: Path, model: str,
    reasoning_effort: str, batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schema_path = run_dir / "brain-output-schema.json"
    atomic_json(schema_path, brain._schema(brain.POLICY_ID))
    aliases = [(f"case-{index:03d}", episode) for index, episode in enumerate(episodes, start=1)]
    input_dir = run_dir / "brain-inputs"
    rendered = {}
    for case_id, episode in aliases:
        path = input_dir / f"{case_id}.jpg"
        brain._render_input(episode, case_id, path)
        rendered[case_id] = path

    decisions = []
    receipts = []
    for offset in range(0, len(aliases), batch_size):
        batch = aliases[offset:offset + batch_size]
        batch_id = f"batch-{offset // batch_size + 1:03d}"
        batch_dir = run_dir / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        prompt = brain._prompt(batch, brain.POLICY_ID)
        (batch_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        raw_path = batch_dir / "last-message.json"
        atomic_json(batch_dir / "dispatch.json", {
            "status": "DISPATCH_STARTED", "attempt": 1,
            "case_ids": [case_id for case_id, _ in batch],
        })
        command = [
            str(executable), "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
            "--json", "--color", "never", "--sandbox", "read-only", "--model", model,
            "-c", f'model_reasoning_effort="{reasoning_effort}"', "--output-schema", str(schema_path),
            "--output-last-message", str(raw_path),
        ]
        for case_id, _ in batch:
            command.extend(["--image", str(rendered[case_id])])
        command.extend(["--", prompt])
        result = subprocess.run(
            command, cwd=batch_dir, shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900,
        )
        (batch_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (batch_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0 or not raw_path.is_file():
            atomic_json(batch_dir / "completion.json", {"status": "IN_DOUBT", "returncode": result.returncode})
            raise RunError(f"{batch_id} provider call is in_doubt; no retry permitted")
        try:
            parsed = json.loads(raw_path.read_text(encoding="utf-8"))
            validated = brain._validate_raw(parsed, batch, brain.POLICY_ID)
        except (json.JSONDecodeError, brain.BrainRunError) as error:
            atomic_json(batch_dir / "completion.json", {"status": "INVALID_OUTPUT", "error": str(error)})
            raise RunError(f"{batch_id} invalid provider output; no retry permitted") from error
        decisions.extend(validated)
        receipt = {
            "status": "RUN_SUCCESS", "attempt": 1, "episode_count": len(batch),
            "response_sha256": materializer.content_sha256(parsed),
        }
        atomic_json(batch_dir / "completion.json", receipt)
        receipts.append({"batch_id": batch_id, **receipt})
        print(f"{batch_id} complete {len(batch)}", flush=True)
    return decisions, receipts


def evaluate(
    roster: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decision_by_episode = {item["episode_id"]: item for item in decisions}
    rows = []
    for truth, episode in zip(roster, episodes):
        native = truth_box(truth)
        candidate_ious = {
            candidate["candidate_id"]: iou(
                [candidate["region"][key] for key in ("x_min", "y_min", "x_max", "y_max")], native,
            )
            for candidate in episode["candidates"]
        }
        correct = [candidate_id for candidate_id, value in candidate_ious.items() if value >= IOU_THRESHOLD]
        decision = decision_by_episode[episode["episode_id"]]
        selected = decision["selected_candidate_ids"][0] if decision["selected_candidate_ids"] else None
        if not correct:
            failure = "PROPOSAL_MISS"
        elif decision["action"] != "SELECT":
            failure = "REFERENT_SELECTION_ABSTAINED_WITH_USABLE_PROPOSAL"
        elif selected not in correct:
            failure = "WRONG_CONFIDENT_GUIDANCE"
        else:
            failure = "CORRECT_GROUNDING"
        rows.append({
            "episode_id": episode["episode_id"], "rgb_sha256": truth["rgb_sha256"],
            "proposal_count": len(candidate_ious), "best_proposal_iou": max(candidate_ious.values(), default=0.0),
            "correct_candidate_ids": correct, "brain_action": decision["action"],
            "selected_candidate_id": selected, "outcome": failure,
        })
    counts = dict(sorted(Counter(item["outcome"] for item in rows).items()))
    available = [item for item in rows if item["correct_candidate_ids"]]
    correct = [item for item in available if item["outcome"] == "CORRECT_GROUNDING"]
    return {
        "iou_threshold": IOU_THRESHOLD,
        "observation_count": len(rows),
        "proposal_availability": {"numerator": len(available), "denominator": len(rows), "value": len(available) / len(rows)},
        "selection_accuracy_given_usable_proposal": {
            "numerator": len(correct), "denominator": len(available),
            "value": len(correct) / len(available) if available else None,
        },
        "outcome_counts": counts,
        "observations": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    if sha256_file(roster_path) != args.roster_sha256:
        raise RunError("frozen roster hash mismatch")
    roster_doc = json.loads(roster_path.read_text(encoding="utf-8"))
    roster = roster_doc["observations"]
    if roster_doc.get("selected_count") != 89 or len(roster) != 89 or roster_doc.get("provider_calls") != 0:
        raise RunError("frozen roster is not an untouched 89-observation cohort")
    provider_lock = provider_adapter.preflight_provider(codex_exe=args.codex_exe, model_dir=args.model_dir)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise RunError("formal run directory already exists; refusing replay")
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "provider-lock.json", provider_lock)

    public_rows = []
    metadata = []
    for index, item in enumerate(roster, start=1):
        rgb = (roster_path.parent / item["rgb_path"]).resolve()
        if sha256_file(rgb) != item["rgb_sha256"]:
            raise RunError(f"RGB hash mismatch at observation {index}")
        public = dict(item, absolute_rgb_path=str(rgb))
        public_rows.append(public)
        metadata.append({"id": Path(item["rgb_path"]).stem, "path": str(rgb), "image_sha256": item["rgb_sha256"]})

    proposals, runtime = dino.run_inference(args.model_dir.resolve(), metadata)
    atomic_json(run_dir / "proposal-provider-output.json", {"runtime": runtime, "outputs": proposals})
    episodes = [
        build_episode(item, result["proposals"], index)
        for index, (item, result) in enumerate(zip(public_rows, proposals), start=1)
    ]
    atomic_json(run_dir / "public-provider-input.json", {"episodes": episodes})
    decisions, receipts = run_brain(
        episodes=episodes, run_dir=run_dir / "brain", executable=args.codex_exe.resolve(),
        model=provider_adapter.CODEX_MODEL, reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
        batch_size=args.batch_size,
    )
    evaluation = evaluate(roster, episodes, decisions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "roster_sha256": args.roster_sha256,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "provider_lock": provider_lock,
        "provider_calls": len(receipts),
        "provider_attempts": len(receipts),
        "provider_in_doubt": 0,
        "teacher_calls": 0,
        "baseline_reruns": 0,
        "brain_batch_receipts": receipts,
        "raw_brain_decisions": decisions,
        "evaluation": evaluation,
        "claim_ceiling": CLAIM_CEILING,
        "lost_status": "NOT_EVALUABLE_NO_EPISODES",
        "range_bearing_status": "NOT_EVALUABLE_NO_METRIC_TRAJECTORY",
    }
    report["report_sha256"] = materializer.content_sha256(report)
    atomic_json(run_dir / "final-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({
        "report_sha256": report["report_sha256"], "evaluation": report["evaluation"],
        "claim_ceiling": report["claim_ceiling"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
