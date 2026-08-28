from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import artvideo_ocr_replay as l10_text


VIEW_OFFSETS = (0, 3, 6)
REQUIRED_MATCHES = 2
SEMANTIC_GATE = 0.58
MIN_PRECISION_UPLIFT = 0.05
MIN_WRONG_REDUCTION = 0.50
MIN_CORRECT_COVERAGE = 0.50
MIN_CORRECT_TRACK_ACQUISITION = 0.60
MAX_WRONG_TRACK_ACQUISITION = 0.10


def normalized(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def precision(correct: int, wrong: int) -> float | None:
    total = correct + wrong
    return round(correct / total, 4) if total else None


def rate(value: int, total: int) -> float | None:
    return round(value / total, 4) if total else None


def evaluate(observations: list[dict[str, Any]]) -> dict[str, Any]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_track[str(row["track_id"])].append(row)

    baseline = Counter()
    consensus = Counter()
    tracks_with_output: set[str] = set()
    tracks_with_correct: set[str] = set()
    tracks_with_wrong: set[str] = set()
    rows: list[dict[str, Any]] = []
    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda row: int(row["frame_id"]))
        for index, current in enumerate(track_rows):
            current_text = normalized(str(current["recognized_text"]))
            if current_text:
                baseline["accepted"] += 1
                if float(current["semantic_lexical"]) >= SEMANTIC_GATE:
                    baseline["correct"] += 1
                else:
                    baseline["wrong"] += 1
            if index < max(VIEW_OFFSETS):
                continue

            source_rows = [track_rows[index - offset] for offset in VIEW_OFFSETS]
            source_texts = [normalized(str(row["recognized_text"])) for row in source_rows]
            counts = Counter(text for text in source_texts if text)
            chosen, matches = counts.most_common(1)[0] if counts else ("", 0)
            if matches < REQUIRED_MATCHES:
                consensus["unknown"] += 1
                continue

            semantic_score = float(l10_text.lexical(str(current["transcription"]), chosen))
            correct = semantic_score >= SEMANTIC_GATE
            consensus["accepted"] += 1
            consensus["correct" if correct else "wrong"] += 1
            tracks_with_output.add(track_id)
            if correct:
                tracks_with_correct.add(track_id)
            else:
                tracks_with_wrong.add(track_id)
            rows.append(
                {
                    "track_id": track_id,
                    "frame_id": current["frame_id"],
                    "transcription": current["transcription"],
                    "source_frame_ids": [row["frame_id"] for row in source_rows],
                    "source_texts": source_texts,
                    "chosen_text": chosen,
                    "matching_views": matches,
                    "semantic_lexical": round(semantic_score, 6),
                    "correct": correct,
                }
            )

    track_count = len(by_track)
    baseline_precision = precision(baseline["correct"], baseline["wrong"])
    consensus_precision = precision(consensus["correct"], consensus["wrong"])
    precision_uplift = (
        round(float(consensus_precision) - float(baseline_precision), 4)
        if baseline_precision is not None and consensus_precision is not None
        else None
    )
    wrong_reduction = (
        round(1.0 - consensus["wrong"] / baseline["wrong"], 4)
        if baseline["wrong"]
        else None
    )
    correct_coverage = rate(consensus["correct"], baseline["correct"])
    correct_track_acquisition = rate(len(tracks_with_correct), track_count)
    wrong_track_acquisition = rate(len(tracks_with_wrong), track_count)
    checks = {
        "precision_uplift_pass": precision_uplift is not None and precision_uplift >= MIN_PRECISION_UPLIFT,
        "wrong_reduction_pass": wrong_reduction is not None and wrong_reduction >= MIN_WRONG_REDUCTION,
        "correct_coverage_pass": correct_coverage is not None and correct_coverage >= MIN_CORRECT_COVERAGE,
        "correct_track_acquisition_pass": (
            correct_track_acquisition is not None
            and correct_track_acquisition >= MIN_CORRECT_TRACK_ACQUISITION
        ),
        "wrong_track_acquisition_pass": (
            wrong_track_acquisition is not None
            and wrong_track_acquisition <= MAX_WRONG_TRACK_ACQUISITION
        ),
    }
    return {
        "baseline": {
            **dict(baseline),
            "precision": baseline_precision,
        },
        "triview_consensus": {
            **dict(consensus),
            "precision": consensus_precision,
            "precision_uplift": precision_uplift,
            "wrong_reduction": wrong_reduction,
            "correct_coverage_vs_baseline": correct_coverage,
            "tracks": track_count,
            "tracks_with_any_consensus": len(tracks_with_output),
            "tracks_with_correct_consensus": len(tracks_with_correct),
            "tracks_with_wrong_consensus": len(tracks_with_wrong),
            "correct_track_acquisition_rate": correct_track_acquisition,
            "wrong_track_acquisition_rate": wrong_track_acquisition,
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.observations_result.read_text(encoding="utf-8"))
    evaluation = evaluate(source["observations"])
    decision = (
        "SC17_TRIVIEW_SEMANTIC_CONSENSUS_DEVELOPMENT_SIGNAL"
        if evaluation["gate"]["passed"]
        else "SC17_TRIVIEW_SEMANTIC_CONSENSUS_DEVELOPMENT_GATE_NOT_MET"
    )
    result = {
        "schema_version": 1,
        "experiment": "l10_sc17_triview_semantic_consensus_development_v0",
        "decision": decision,
        "source_experiment": source.get("experiment"),
        "frozen_rule": {
            "view_offsets": list(VIEW_OFFSETS),
            "required_exact_normalized_matches": REQUIRED_MATCHES,
            "semantic_gate": SEMANTIC_GATE,
            "no_confidence_or_quality_thresholds": True,
            "sweeps": [],
        },
        **evaluation,
        "claim_boundary": (
            "Consumed SC16 Development observations only. This can justify a fresh source-disjoint test of "
            "cross-view semantic consensus, not live active-view causality, exact-instance identity, metric "
            "arrival, product benefit, user benefit, or safety."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "baseline": result["baseline"],
                "triview_consensus": result["triview_consensus"],
                "gate": result["gate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
