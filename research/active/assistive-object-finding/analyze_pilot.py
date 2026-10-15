#!/usr/bin/env python3
"""Evaluate a small language-guided object-finding Development pilot.

No third-party dependencies. Manifest and prediction rows join on
(sequence_id, frame_idx), never file order. Scores/boxes are observations;
ground truth is used only by evaluation and explicitly named oracles.

Example:
  python analyze_pilot.py --manifest manifest.jsonl \
    --predictions predictions.jsonl --output-dir analysis
  python analyze_pilot.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def ratio(n: int | float, d: int | float) -> dict[str, Any]:
    return {"n": n, "d": d, "value": n / d if d else None}


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (aa + ab - intersection) if aa + ab > intersection else 0.0


def valid_box(box: Any, context: str) -> list[float] | None:
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{context}: expected four-element pixel xyxy box")
    out = [float(v) for v in box]
    if not all(math.isfinite(v) for v in out) or out[2] <= out[0] or out[3] <= out[1]:
        raise ValueError(f"{context}: invalid xyxy box {box}")
    return out


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["sequence_id"]), int(row["frame_idx"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen = set()
    queries: dict[str, str] = {}
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate manifest key {key}")
        seen.add(key)
        if not isinstance(row["gt_present"], bool):
            raise ValueError(f"{key}: gt_present must be JSON boolean")
        row["gt_bbox_xyxy"] = valid_box(row.get("gt_bbox_xyxy"), str(key))
        if row["gt_present"] != (row["gt_bbox_xyxy"] is not None):
            raise ValueError(f"{key}: presence/box mismatch")
        for field in ("width", "height"):
            if int(row[field]) <= 0:
                raise ValueError(f"{key}: invalid {field}")
        row["timestamp_s"] = float(row["timestamp_s"]) if row.get("timestamp_s") is not None else None
        if row["timestamp_s"] is not None and not math.isfinite(row["timestamp_s"]):
            raise ValueError(f"{key}: timestamp_s must be finite")
        query = str(row["query"])
        if key[0] in queries and queries[key[0]] != query:
            raise ValueError(f"{key}: sequence_id must identify one query")
        queries[key[0]] = query
    rows.sort(key=lambda row: (str(row["sequence_id"]), int(row["frame_idx"])))
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    for sequence_id, seq in by_sequence.items():
        if any(b["timestamp_s"] < a["timestamp_s"] for a, b in zip(seq, seq[1:])
               if a["timestamp_s"] is not None and b["timestamp_s"] is not None):
            raise ValueError(f"{sequence_id}: timestamps decrease with frame_idx")
    return rows


def prediction_files(paths: list[str]) -> list[Path]:
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(path.rglob("*predictions*.jsonl"))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(p.resolve() for p in found))
    if not unique:
        raise ValueError("no prediction files found (directory search: *predictions*.jsonl)")
    return unique


def load_predictions(paths: list[Path]) -> tuple[dict, dict]:
    observations = {}
    duplicates = 0
    for path in paths:
        for row in read_jsonl(path):
            key = row_key(row)
            candidates = []
            for candidate in row.get("candidates", []):
                item = dict(candidate)
                item["bbox_xyxy"] = valid_box(item.get("bbox_xyxy"), str(key))
                if item["bbox_xyxy"] is None:
                    raise ValueError(f"{key}: candidate box may not be null")
                item["score"] = float(item["score"])
                if not math.isfinite(item["score"]):
                    raise ValueError(f"{key}: nonfinite candidate score")
                item["ranking_score"] = float(item.get("ranking_score", item["score"]))
                if not math.isfinite(item["ranking_score"]):
                    raise ValueError(f"{key}: nonfinite ranking score")
                candidates.append(item)
            # Exact score ties use a published, stable lexicographic box rule.
            candidates.sort(key=lambda c: (-c["ranking_score"], -c["score"], tuple(c["bbox_xyxy"])))
            presence_score = max((c["score"] for c in candidates), default=0.0)
            if row.get("presence_score") is not None and not math.isclose(
                    float(row["presence_score"]), presence_score, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"{key}: presence_score differs from max candidate observation score")
            item = {"query": str(row["query"]), "candidates": candidates,
                    "presence_score": presence_score,
                    "selection_method": row.get("selection_method"),
                    "score_semantics": row.get("score_semantics", "model_confidence"),
                    "detector_called": row.get("detector_called"),
                    "action_reason": row.get("action_reason"),
                    "relocation_source": row.get("relocation_source"),
                    "elapsed_s": row.get("elapsed_s")}
            if key in observations:
                if observations[key] != item:
                    raise ValueError(f"conflicting duplicate prediction {key}; separate model runs")
                duplicates += 1
            else:
                observations[key] = item
    return observations, {"files": [str(p) for p in paths], "identical_duplicate_rows": duplicates}


def top_box(candidates: list[dict], threshold: float | None) -> list[float] | None:
    if not candidates or (threshold is not None and max(c["score"] for c in candidates) < threshold):
        return None
    return candidates[0]["bbox_xyxy"]


def paired_outputs(rows: list[dict], baseline: dict, trial: dict, iou_threshold: float) -> dict:
    counts = Counter()
    for row in rows:
        key = row_key(row)
        if row["gt_present"]:
            old = iou(baseline.get(key), row["gt_bbox_xyxy"]) >= iou_threshold
            new = iou(trial.get(key), row["gt_bbox_xyxy"]) >= iou_threshold
            counts["visible_frames"] += 1
            counts["visible_saved"] += new and not old
            counts["visible_harmed"] += old and not new
        else:
            old = baseline.get(key) is None
            new = trial.get(key) is None
            counts["absent_frames"] += 1
            counts["absent_false_output_removed"] += new and not old
            counts["absent_false_output_added"] += old and not new
    return {key: counts[key] for key in (
        "visible_frames", "visible_saved", "visible_harmed", "absent_frames",
        "absent_false_output_removed", "absent_false_output_added")}


def evaluate_outputs(rows: list[dict], outputs: dict, iou_threshold: float) -> dict:
    visible = [row for row in rows if row["gt_present"]]
    absent = [row for row in rows if not row["gt_present"]]
    overlaps = [iou(outputs.get(row_key(row)), row["gt_bbox_xyxy"]) for row in visible]
    tp = sum(outputs.get(row_key(row)) is not None for row in visible)
    fp = sum(outputs.get(row_key(row)) is not None for row in absent)
    loc = sum(overlap >= iou_threshold for overlap in overlaps)
    return {
        "frames": len(rows), "visible_frames": len(visible), "absent_frames": len(absent),
        "visible_localization_success": ratio(loc, len(visible)),
        "mean_iou_visible_missing_as_zero": ratio(sum(overlaps), len(visible)),
        "presence_recall": ratio(tp, len(visible)),
        "presence_precision": ratio(tp, tp + fp),
        "absent_false_output": ratio(fp, len(absent)),
        "joint_frame_correct": ratio(loc + len(absent) - fp, len(rows)),
    }


def presence_curve(rows: list[dict], observations: dict, iou_threshold: float) -> list[dict]:
    # Prespecified descriptive points; the evaluator never selects an operating point.
    results = []
    for threshold in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        outputs = {row_key(row): top_box(observations.get(row_key(row), {}).get("candidates", []),
                                        threshold) for row in rows}
        results.append({"threshold": threshold, **evaluate_outputs(rows, outputs, iou_threshold)})
    return results


def candidate_headroom(rows: list[dict], observations: dict, iou_threshold: float,
                       score_threshold: float = 0.25) -> dict:
    visible = [row for row in rows if row["gt_present"]]
    oracle, eligible_oracle = {}, {}
    for k in (1, 3, 5, 10, None):
        count, eligible_count = 0, 0
        for row in visible:
            candidates = observations.get(row_key(row), {}).get("candidates", [])
            subset = candidates if k is None else candidates[:k]
            count += any(iou(c["bbox_xyxy"], row["gt_bbox_xyxy"]) >= iou_threshold for c in subset)
            eligible = [c for c in candidates if c["score"] >= score_threshold]
            eligible_subset = eligible if k is None else eligible[:k]
            eligible_count += any(iou(c["bbox_xyxy"], row["gt_bbox_xyxy"]) >= iou_threshold
                                  for c in eligible_subset)
        oracle[str(k) if k is not None else "all"] = ratio(count, len(visible))
        eligible_oracle[str(k) if k is not None else "all"] = ratio(eligible_count, len(visible))
    top1 = oracle["1"]["n"]
    all_candidates = oracle["all"]["n"]
    return {
        "label": "EVALUATOR-ONLY ORACLE; candidate coverage, not a method result",
        "top_k_oracle_success_visible": oracle,
        "top_k_oracle_after_fixed_score_threshold": eligible_oracle,
        "fixed_score_threshold_for_eligible_candidate_pool": score_threshold,
        "top1_correct": top1,
        "candidate_present_but_top1_wrong": all_candidates - top1,
        "no_correct_candidate_in_cache": len(visible) - all_candidates,
        "visible_denominator": len(visible),
        "reranking_maximum_recoverable_fraction_of_top1_errors": ratio(
            all_candidates - top1, len(visible) - top1),
    }


def simulate(rows: list[dict], observations: dict, threshold: float) -> dict[str, dict]:
    """Causal cached replay. No GT passed to action/selection rules.

    Hold policies access candidate rows only when a re-grounding call is made.
    Cached-IoU association deliberately reads candidates every frame and is a
    surrogate, not a measured cheap tracker or an achieved inference speedup.
    """
    strategies = {}
    specifications = [
        ("every_frame", 1, "hold"),
        ("fixed2_last_box_hold", 2, "hold"),
        ("fixed4_last_box_hold", 4, "hold"),
        ("adaptive_last_margin_hold", 4, "adaptive_hold"),
        ("fixed2_cached_iou_association", 2, "cached_iou"),
        ("fixed4_cached_iou_association", 4, "cached_iou"),
        ("every_frame_temporal_iou_rerank", 1, "rerank"),
    ]
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    for name, period, mode in specifications:
        outputs, calls, reads, call_keys = {}, 0, 0, []
        for sequence in by_sequence.values():
            last_box, last_score, last_margin, last_call_ordinal = None, 0.0, 0.0, -1000
            for ordinal, row in enumerate(sequence):
                key = row_key(row)
                due = ordinal % period == 0
                if mode == "adaptive_hold":
                    # Only prior observed score/margin and sample age trigger a call.
                    due = (ordinal == 0 or ordinal - last_call_ordinal >= period
                           or last_score < threshold or last_margin < 0.05)
                if due or mode in ("cached_iou", "rerank"):
                    candidates = observations.get(key, {}).get("candidates", [])
                    reads += 1
                else:
                    candidates = []
                if due:
                    calls += 1
                    call_keys.append([key[0], key[1]])
                    last_call_ordinal = ordinal
                    last_score = max((c["score"] for c in candidates), default=0.0)
                    last_margin = (candidates[0].get("ranking_score", candidates[0]["score"])
                                   - candidates[1].get("ranking_score", candidates[1]["score"])
                                   if len(candidates) >= 2 else last_score)
                    if mode != "rerank":
                        last_box = top_box(candidates, threshold)
                if mode == "cached_iou" and not due and last_box is not None:
                    eligible = [c for c in candidates if c["score"] >= threshold]
                    eligible.sort(key=lambda c: (-iou(last_box, c["bbox_xyxy"]),
                                                 -c["score"], tuple(c["bbox_xyxy"])))
                    last_box = (eligible[0]["bbox_xyxy"] if eligible and
                                iou(last_box, eligible[0]["bbox_xyxy"]) >= 0.1 else None)
                if mode == "rerank":
                    eligible = [c for c in candidates if c["score"] >= threshold]
                    eligible.sort(key=lambda c: (-(c["score"] + 0.2 * iou(last_box, c["bbox_xyxy"])),
                                                 -c["score"], tuple(c["bbox_xyxy"])))
                    last_box = eligible[0]["bbox_xyxy"] if eligible else None
                outputs[key] = last_box
        strategies[name] = {
            "outputs": outputs, "regrounding_calls": calls,
            "regrounding_call_ratio": ratio(calls, len(rows)),
            "candidate_cache_reads": reads, "call_keys": call_keys,
            "surrogate_reads_candidates_every_frame": mode == "cached_iou",
        }
    return strategies


def segment_events(rows: list[dict], outputs: dict, iou_threshold: float) -> list[dict]:
    """Time to first correct sample per presence/absence segment; failures censored.

    Initial visible/absent segments are recorded separately from 0->1 / 1->0
    transitions. An unobserved interval does not create an inferred onset.
    """
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    events = []
    for sequence_id, seq in by_sequence.items():
        start = 0
        for end in range(1, len(seq) + 1):
            if end < len(seq) and seq[end]["gt_present"] == seq[start]["gt_present"]:
                continue
            segment = seq[start:end]
            present = seq[start]["gt_present"]
            kind = ("initial_present" if present else "initial_absent") if start == 0 else (
                "reappearance" if present else "disappearance")
            correct = []
            for offset, row in enumerate(segment):
                box = outputs.get(row_key(row))
                good = iou(box, row["gt_bbox_xyxy"]) >= iou_threshold if present else box is None
                if good:
                    correct.append((offset, row))
            first = correct[0] if correct else None
            events.append({
                "sequence_id": sequence_id, "event_type": kind, "gt_present": present,
                "start_frame_idx": seq[start]["frame_idx"],
                "start_timestamp_s": seq[start]["timestamp_s"],
                "last_observed_timestamp_s": segment[-1]["timestamp_s"],
                "segment_samples": len(segment), "censored": first is None,
                "first_correct_frame_idx": first[1]["frame_idx"] if first else None,
                "delay_observed_samples": first[0] if first else None,
                "delay_frame_idx_units": first[1]["frame_idx"] - seq[start]["frame_idx"] if first else None,
                "delay_s": (first[1]["timestamp_s"] - seq[start]["timestamp_s"]
                            if first and first[1]["timestamp_s"] is not None
                            and seq[start]["timestamp_s"] is not None else None),
                "observed_censor_duration_s": (segment[-1]["timestamp_s"] - seq[start]["timestamp_s"]
                                               if segment[-1]["timestamp_s"] is not None
                                               and seq[start]["timestamp_s"] is not None else None),
                "correct_samples": len(correct),
            })
            start = end
    return events


def summarize_events(events: list[dict]) -> dict:
    result = {}
    for kind in ("initial_present", "reappearance", "initial_absent", "disappearance"):
        subset = [event for event in events if event["event_type"] == kind]
        recovered = [event for event in subset if not event["censored"]]
        observed = [event["delay_s"] for event in recovered if event["delay_s"] is not None]
        sample_delays = [event["delay_observed_samples"] for event in recovered]
        frame_delays = [event["delay_frame_idx_units"] for event in recovered]
        result[kind] = {
            "events": len(subset), "recovered": len(recovered),
            "censored": len(subset) - len(recovered),
            "recovery_fraction": ratio(len(recovered), len(subset)),
            "recovered_events_with_seconds_available": len(observed),
            "mean_delay_observed_samples_among_recovered_only": sum(sample_delays) / len(sample_delays) if sample_delays else None,
            "mean_delay_frame_idx_units_among_recovered_only": sum(frame_delays) / len(frame_delays) if frame_delays else None,
            "mean_delay_s_among_recovered_only": sum(observed) / len(observed) if observed else None,
        }
    return result


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    previous = {}
    for row in rows:
        seq = str(row["sequence_id"])
        present = row["gt_present"]
        groups[f"presence/{'visible' if present else 'absent'}"].append(row)
        groups[f"dataset/{row.get('dataset', 'unknown')}"].append(row)
        groups[f"split/{row.get('split', 'unknown')}"].append(row)
        if "selection_reason" in row:
            groups[f"selection_reason/{row['selection_reason']}"].append(row)
        if seq in previous and previous[seq] != present:
            groups[f"transition/{'reappearance_sample' if present else 'disappearance_sample'}"].append(row)
        previous[seq] = present
        if present:
            box = row["gt_bbox_xyxy"]
            area = (box[2] - box[0]) * (box[3] - box[1]) / (row["width"] * row["height"])
            area_group = "small_lt_1pct" if area < 0.01 else (
                "medium_1_to_10pct" if area < 0.1 else "large_ge_10pct")
            groups[f"area/{area_group}"].append(row)
        if "is_multi" in row:
            groups[f"is_multi/{row['is_multi']}"].append(row)
    return dict(groups)


def analyze(rows: list[dict], observations: dict, threshold: float = 0.25,
            iou_threshold: float = 0.5) -> dict:
    keys = {row_key(row) for row in rows}
    for row in rows:
        pred = observations.get(row_key(row))
        if pred is not None and pred["query"] != str(row["query"]):
            raise ValueError(f"query mismatch at {row_key(row)}")
    raw = {row_key(row): top_box(observations.get(row_key(row), {}).get("candidates", []), None)
           for row in rows}
    confidence_available = all(observations[key].get("score_semantics", "model_confidence") == "model_confidence"
                               for key in keys & observations.keys())
    strategies = simulate(rows, observations, threshold)
    if not confidence_available:
        # Native empty/nonempty masks supply no scores for adaptive confidence policies.
        strategies = {"every_frame": strategies["every_frame"]}
    per_frame = []
    for row in rows:
        key = row_key(row)
        candidates = observations.get(key, {}).get("candidates", [])
        overlap = iou(raw[key], row["gt_bbox_xyxy"])
        best = max((iou(c["bbox_xyxy"], row["gt_bbox_xyxy"]) for c in candidates), default=0.0)
        per_frame.append({
            "sequence_id": key[0], "frame_idx": key[1], "timestamp_s": row["timestamp_s"],
            "gt_present": row["gt_present"], "prediction_row_present": key in observations,
            "observed_detector_called": observations.get(key, {}).get("detector_called"),
            "observed_action_reason": observations.get(key, {}).get("action_reason"),
            "candidate_count": len(candidates),
            "top1_score": candidates[0]["score"] if candidates else None,
            "presence_score": max((c["score"] for c in candidates), default=0.0),
            "top1_ranking_score": candidates[0].get("ranking_score", candidates[0]["score"]) if candidates else None,
            "top1_margin": candidates[0]["score"] - candidates[1]["score"] if len(candidates) > 1 else None,
            "top1_ranking_margin": (candidates[0].get("ranking_score", candidates[0]["score"])
                                    - candidates[1].get("ranking_score", candidates[1]["score"]) if len(candidates) > 1 else None),
            "top_score_tie_count": sum(c.get("ranking_score", c["score"]) == candidates[0].get("ranking_score", candidates[0]["score"])
                                       for c in candidates) if candidates else 0,
            "top1_iou": overlap if row["gt_present"] else None,
            "candidate_oracle_iou": best if row["gt_present"] else None,
            "failure_type": ("absent" if not row["gt_present"] else "top1_correct" if overlap >= iou_threshold
                             else "candidate_present_but_top1_wrong" if best >= iou_threshold
                             else "candidate_missing"),
        })
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    policy_results, events = {}, []
    for name, strategy in strategies.items():
        policy_events = segment_events(rows, strategy["outputs"], iou_threshold)
        events.extend({"strategy": name, **event} for event in policy_events)
        policy_results[name] = {
            **evaluate_outputs(rows, strategy["outputs"], iou_threshold),
            **{key: value for key, value in strategy.items() if key not in ("outputs", "call_keys")},
            "events": summarize_events(policy_events),
            "paired_vs_every_frame": paired_outputs(
                rows, strategies["every_frame"]["outputs"], strategy["outputs"], iou_threshold),
        }
        for item, row in zip(per_frame, rows):
            box = strategy["outputs"].get(row_key(row))
            item[f"{name}_output_box_xyxy"] = box
            item[f"{name}_correct"] = (iou(box, row["gt_bbox_xyxy"]) >= iou_threshold
                                       if row["gt_present"] else box is None)
    baseline_outputs = strategies["every_frame"]["outputs"]
    elapsed = [float(observations[key]["elapsed_s"]) for key in keys & observations.keys()
               if observations[key].get("elapsed_s") is not None]
    action_keys = [key for key in keys & observations.keys()
                   if isinstance(observations[key].get("detector_called"), bool)]
    intervals = [b["timestamp_s"] - a["timestamp_s"] for seq in by_sequence.values()
                 for a, b in zip(seq, seq[1:])
                 if a["timestamp_s"] is not None and b["timestamp_s"] is not None]
    headroom = candidate_headroom(rows, observations, iou_threshold, threshold)
    if not confidence_available:
        headroom.pop("top_k_oracle_after_fixed_score_threshold")
        headroom.pop("fixed_score_threshold_for_eligible_candidate_pool")
    return {
        "contract": {
            "join_key": ["sequence_id", "frame_idx"],
            "score_threshold": threshold if confidence_available else None,
            "iou_threshold": iou_threshold,
            "tie_rule": "ranking_score (default score) descending, score descending, box xyxy lexicographic ascending",
            "presence_score_rule": "max original candidate score; reranking does not recalibrate presence",
            "confidence_scores_available": confidence_available,
            "native_presence_rule_if_no_confidence": "nonempty official mask is present; score=1 is only a binary output indicator",
            "lane": "EXPLORE / Development pilot",
            "presence_curve_is_descriptive_no_evaluation_selected_threshold": True,
            "oracle_not_method_result": True,
            "simulation_limit": "Cached replay; hold is not a tracker; IoU association uses current cached detector candidates. Call ratios do not measure real speedup.",
            "recovery_limit": "Observed annotation samples only; initial segments separate; unrecovered segments censored; recovered-only means exclude censored events.",
            "area_bins_fraction_of_image": [0.01, 0.1],
            "fixed_policy_period_unit": "manifest sample ordinal per sequence, not raw-video frame_idx",
            "fixed_trial_constants": {"adaptive_prior_margin_threshold": 0.05,
                                      "association_min_iou": 0.1, "temporal_rerank_iou_weight": 0.2},
        },
        "coverage": {
            "manifest_frames": len(rows), "sequences": len(by_sequence),
            "source_ids": len({str(row.get("source_id", row["sequence_id"])) for row in rows}),
            "prediction_rows_matched": len(keys & observations.keys()),
            "missing_prediction_rows_counted_as_failure_or_no_output": len(keys - observations.keys()),
            "extra_prediction_rows_ignored": len(observations.keys() - keys),
            "top_score_tied_frames": sum(row["top_score_tie_count"] > 1 for row in per_frame),
            "annotation_sample_intervals_s": sorted(set(intervals)),
            "rows_with_known_timestamp_s": sum(row.get("timestamp_s") is not None for row in rows),
            "measured_model_elapsed_s_sum": sum(elapsed) if elapsed and not action_keys else None,
            "measured_model_elapsed_s_mean": sum(elapsed) / len(elapsed) if elapsed and not action_keys else None,
            "observed_row_elapsed_s_sum": sum(elapsed) if elapsed else None,
            "observed_row_elapsed_s_mean": sum(elapsed) / len(elapsed) if elapsed else None,
            "elapsed_rows": len(elapsed),
            "elapsed_s_scope": ("row-level CPU tracker/replay execution; cached detector not executed here"
                                if action_keys else "model runner provided elapsed_s; see runner receipt"),
        },
        "observed_actions": {
            "rows_with_detector_call_field": len(action_keys),
            "detector_call_ratio": ratio(sum(observations[key]["detector_called"] for key in action_keys),
                                         len(action_keys)),
            "action_reason_counts": dict(Counter(observations[key].get("action_reason") for key in action_keys)),
            "by_sequence_detector_calls": {seq: sum(observations[row_key(row)].get("detector_called") is True
                                                     for row in subset if row_key(row) in observations)
                                           for seq, subset in by_sequence.items()} if action_keys else {},
            "limit": "These are input-recorded actions. Cached_replay_strategies below are separate hypothetical policies applied to the input outputs.",
        },
        "baseline_raw_top1": evaluate_outputs(rows, raw, iou_threshold),
        "baseline_thresholded": evaluate_outputs(rows, baseline_outputs, iou_threshold),
        "candidate_oracle_headroom": headroom,
        "presence_curve": presence_curve(rows, observations, iou_threshold) if confidence_available else [],
        "cached_replay_strategies": policy_results,
        "per_sequence": {seq: evaluate_outputs(subset, baseline_outputs, iou_threshold)
                         for seq, subset in by_sequence.items()},
        "groups": {group: evaluate_outputs(subset, baseline_outputs, iou_threshold)
                   for group, subset in sorted(group_rows(rows).items())},
        "per_frame": per_frame, "segment_events": events,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                         else value for key, value in row.items()} for row in rows)


def metric_text(metric: dict) -> str:
    return f"{metric['n']}/{metric['d']} ({metric['value']:.1%})" if metric["value"] is not None else "NA (denominator 0)"


def write_results(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("per_frame", "segment_events", "presence_curve"):
        write_csv(out / f"{name}.csv", result[name])
    write_csv(out / "per_sequence.csv", [{"sequence_id": key, **value} for key, value in result["per_sequence"].items()])
    write_csv(out / "groups.csv", [{"group": key, **value} for key, value in result["groups"].items()])
    write_csv(out / "cached_replay.csv", [{"strategy": key, **value} for key, value in result["cached_replay_strategies"].items()])
    coverage, raw, baseline, oracle = (result[key] for key in (
        "coverage", "baseline_raw_top1", "baseline_thresholded", "candidate_oracle_headroom"))
    lines = [
        "# Language-guided object-finding Development pilot",
        f"Frames: {coverage['manifest_frames']}; sequences: {coverage['sequences']}; source IDs: {coverage['source_ids']}.",
        f"Prediction coverage: {coverage['prediction_rows_matched']}/{coverage['manifest_frames']}; missing rows remain in denominators.",
        f"Raw top-1 localization IoU >= {result['contract']['iou_threshold']}: {metric_text(raw['visible_localization_success'])} visible samples.",
        ((f"At prespecified score threshold {result['contract']['score_threshold']}" if result['contract']['confidence_scores_available'] else
          "At native empty/nonempty-mask output state (confidence unavailable; indicator score is not calibrated)") +
         f": localization {metric_text(baseline['visible_localization_success'])}; absent false output {metric_text(baseline['absent_false_output'])}."),
        f"EVALUATOR-ONLY candidate oracle: {metric_text(oracle['top_k_oracle_success_visible']['all'])}; wrong top-1 with a correct cached candidate {oracle['candidate_present_but_top1_wrong']}; candidate missing {oracle['no_correct_candidate_in_cache']}.",
        "", "Cached replay (accuracy/call-count diagnostic; no measured tracker or end-to-end speed claim):",
    ]
    for name, value in result["cached_replay_strategies"].items():
        lines.append(f"- {name}: localization {metric_text(value['visible_localization_success'])}; absent false output {metric_text(value['absent_false_output'])}; reground calls {metric_text(value['regrounding_call_ratio'])}; candidate-cache reads {value['candidate_cache_reads']}.")
    lines.extend(["", "Recovery/clearance is evaluated only at annotation timestamps. Censored events are retained; no millisecond-scale recovery claims are supported by sparse labels.",
                  "Candidate oracle measures available boxes, not deployable reranking improvement. Trials use fixed constants, no evaluation-selected tuning. No training or actual low-cost tracker is evaluated here."])
    (out / "SUMMARY.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    # Material checks: shuffled identities, missing prediction denominator,
    # censored recovery and initial absence, ties, and GT-independent policies.
    rows = []
    for seq, pattern in (("a", [False, True, True, False, True, True]),
                         ("b", [True, True])):
        for ordinal, present in enumerate(pattern):
            rows.append({"sequence_id": seq, "source_id": seq, "frame_idx": ordinal * 5,
                         "timestamp_s": ordinal * 0.5, "query": "cup", "gt_present": present,
                         "gt_bbox_xyxy": [0, 0, 10, 10] if present else None,
                         "width": 100, "height": 100, "split": "dev", "dataset": "synthetic"})
    observations = {}
    for row in reversed(rows):
        key = row_key(row)
        if key == ("b", 5):
            continue
        boxes = []
        if key == ("a", 5):
            boxes = [{"bbox_xyxy": [20, 20, 30, 30], "score": 0.9},
                     {"bbox_xyxy": [0, 0, 10, 10], "score": 0.8}]
        elif key in (("a", 10), ("b", 0)):
            boxes = [{"bbox_xyxy": [0, 0, 10, 10], "score": 0.9}]
        observations[key] = {"query": "cup", "candidates": boxes, "elapsed_s": 0.01}
    result = analyze(rows, observations)
    assert result["baseline_raw_top1"]["visible_localization_success"] == ratio(2, 6)
    assert result["coverage"]["missing_prediction_rows_counted_as_failure_or_no_output"] == 1
    assert result["candidate_oracle_headroom"]["candidate_present_but_top1_wrong"] == 1
    assert result["candidate_oracle_headroom"]["no_correct_candidate_in_cache"] == 3
    events = result["cached_replay_strategies"]["every_frame"]["events"]
    assert events["reappearance"]["events"] == 2
    assert events["reappearance"]["censored"] == 1
    assert events["reappearance"]["mean_delay_s_among_recovered_only"] == 0.5
    assert events["initial_absent"]["events"] == 1
    other_truth = [{**row, "gt_present": False, "gt_bbox_xyxy": None} for row in rows]
    first, second = simulate(rows, observations, 0.25), simulate(other_truth, observations, 0.25)
    assert first == second, "policy consumed GT"
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou(None, [0, 0, 10, 10]) == 0
    assert result["per_sequence"]["b"]["visible_localization_success"] == ratio(1, 2)
    unknown_times = [{**row, "timestamp_s": None} for row in rows]
    unknown_result = analyze(unknown_times, observations)
    unknown_events = unknown_result["cached_replay_strategies"]["every_frame"]["events"]["reappearance"]
    assert unknown_events["recovered"] == 1 and unknown_events["censored"] == 1
    assert unknown_events["mean_delay_s_among_recovered_only"] is None
    assert unknown_events["mean_delay_observed_samples_among_recovered_only"] == 1
    assert unknown_events["mean_delay_frame_idx_units_among_recovered_only"] == 5
    native_observations = {key: {**value, "score_semantics": "binary_nonempty_mask_indicator_not_confidence"}
                           for key, value in observations.items()}
    native_result = analyze(rows, native_observations)
    assert native_result["contract"]["score_threshold"] is None
    assert native_result["presence_curve"] == []
    assert list(native_result["cached_replay_strategies"]) == ["every_frame"]
    assert top_box([{"bbox_xyxy": [0, 0, 1, 1], "score": 0.3},
                    {"bbox_xyxy": [1, 1, 2, 2], "score": 0.9}], 0.5) == [0, 0, 1, 1]
    print("PASS: synthetic metric/join/missing-frame/censoring/causal-policy/unknown-FPS/native-state checks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.manifest is None:
            return
    if args.manifest is None or not args.predictions or args.output_dir is None:
        parser.error("--manifest, --predictions and --output-dir are required")
    rows = load_manifest(args.manifest)
    files = prediction_files(args.predictions)
    observations, provenance = load_predictions(files)
    result = analyze(rows, observations, args.threshold, args.iou_threshold)
    result["inputs"] = {"manifest": str(args.manifest.resolve()),
                        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                        **provenance,
                        "prediction_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in files}}
    write_results(result, args.output_dir)
    print(json.dumps({"output": str(args.output_dir.resolve()), "coverage": result["coverage"],
                      "baseline": result["baseline_thresholded"],
                      "oracle": result["candidate_oracle_headroom"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
