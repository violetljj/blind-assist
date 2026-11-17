"""Attribute replayed turn exits to changes in the two logged angular terms."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import median

from mobilio_turn_confirmation import (
    BASELINES,
    COHORTS,
    HOLD_SECONDS,
    TOLERANCE_DEGREES,
    WINDOW_SECONDS,
    load_trials,
    replay,
    trial_path,
    wrap_degrees,
    write_json,
)


LINEAR = next(rule for rule in BASELINES if rule.name == "linear_projection")


def audit_trigger(trial, heading, decision):
    if decision["time_s"] is None:
        return None
    time, error = trial.time_s, trial.error_deg
    bearing = tuple(wrap_degrees(u + e) for u, e in zip(heading, error, strict=True))
    if not all(math.isfinite(point) for point in bearing):
        raise ValueError(f"Nonfinite heading: {trial.person}/{trial.number}")
    trigger = bisect_left(time, decision["time_s"])
    future = bisect_left(time, time[trigger] + HOLD_SECONDS)
    past = bisect_right(time, time[trigger] - LINEAR.lookback_s) - 1
    duration = time[trigger] - time[past]
    exit_index = next(
        (j for j in range(trigger + 1, future + 1) if abs(error[j]) > TOLERANCE_DEGREES), None
    )
    if decision["correct"] != (exit_index is None):
        raise ValueError(f"Exit must agree with the original replay: {trial.person}/{trial.number}")
    result = {
        "person": trial.person,
        "trial": trial.number,
        "cue": trial.cue,
        "initial_deg": error[0],
        "trigger_s": time[trigger],
        "trigger_deg": error[trigger],
        "relative_rate_deg_s": wrap_degrees(error[trigger] - error[past]) / duration,
        "heading_rate_deg_s": wrap_degrees(heading[trigger] - heading[past]) / duration,
        "bearing_rate_deg_s": wrap_degrees(bearing[trigger] - bearing[past]) / duration,
        "correct": decision["correct"],
        "attribution": None,
        "after_trigger_exit_s": None,
        "exit_deg": None,
        "toward_exit_from_heading_deg": None,
        "toward_exit_from_bearing_deg": None,
        "peak_abs_deg_in_horizon": max(map(abs, error[trigger:future + 1])),
        "reentered_in_horizon": None,
        "after_exit_next_stable_s": None,
        "after_exit_max_stable_hold_s": None,
    }
    if exit_index is None:
        return result

    direction = math.copysign(1, error[exit_index])
    from_heading = -direction * wrap_degrees(heading[exit_index] - heading[trigger])
    from_bearing = direction * wrap_degrees(bearing[exit_index] - bearing[trigger])
    observed = direction * (error[exit_index] - error[trigger])
    if not math.isclose(from_heading + from_bearing, observed, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"Unresolved wrap at crossing: {trial.person}/{trial.number}")
    distance_to_boundary = TOLERANCE_DEGREES - direction * error[trigger]
    heading_alone = from_heading > distance_to_boundary
    bearing_alone = from_bearing > distance_to_boundary
    attribution = {
        (True, False): "heading_alone_sufficient",
        (False, True): "bearing_alone_sufficient",
        (True, True): "both_individually_sufficient",
        (False, False): "both_required",
    }[heading_alone, bearing_alone]

    end = bisect_right(time, WINDOW_SECONDS)
    next_stable = None
    max_stable_hold = 0.0
    in_band_since = None
    for j in range(exit_index + 1, end):
        if abs(error[j]) > TOLERANCE_DEGREES:
            in_band_since = None
            continue
        if in_band_since is None:
            in_band_since = time[j]
        elapsed = time[j] - in_band_since
        max_stable_hold = max(max_stable_hold, elapsed)
        if next_stable is None and time[j] >= in_band_since + HOLD_SECONDS:
            next_stable = in_band_since - time[exit_index]
    result.update(
        attribution=attribution,
        after_trigger_exit_s=time[exit_index] - time[trigger],
        exit_deg=error[exit_index],
        toward_exit_from_heading_deg=from_heading,
        toward_exit_from_bearing_deg=from_bearing,
        reentered_in_horizon=any(abs(e) <= TOLERANCE_DEGREES for e in error[exit_index + 1:future + 1]),
        after_exit_next_stable_s=next_stable,
        after_exit_max_stable_hold_s=max_stable_hold,
    )
    return result


def describe(rows):
    correct = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    result = {"correct": len(correct), "wrong": len(wrong)}
    for name, group in (("correct", correct), ("wrong", wrong)):
        if group:
            result[f"{name}_median_absolute_prefix"] = {
                field: median(abs(row[field]) for row in group)
                for field in ("trigger_deg", "relative_rate_deg_s", "heading_rate_deg_s", "bearing_rate_deg_s")
            }
    later = [row["after_exit_next_stable_s"] for row in wrong if row["after_exit_next_stable_s"] is not None]
    result.update(
        attribution=dict(Counter(row["attribution"] for row in wrong)),
        reentered_in_horizon=sum(row["reentered_in_horizon"] for row in wrong),
        later_evaluable_stable_window=len(later),
        median_exit_to_later_stable_s=median(later) if later else None,
        median_trigger_to_exit_s=median(row["after_trigger_exit_s"] for row in wrong) if wrong else None,
        peak_above_12_deg=sum(row["peak_abs_deg_in_horizon"] > 12 for row in wrong),
        peak_above_15_deg=sum(row["peak_abs_deg_in_horizon"] > 15 for row in wrong),
        exit_across_initial_side=sum(row["initial_deg"] * row["exit_deg"] < 0 for row in wrong),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    all_rows = []
    summary = {}
    for cohort, members in COHORTS.items():
        trials = load_trials(args.source_dir, cohort)
        rows = []
        for trial in trials:
            decision = replay(trial, (LINEAR,))["decisions"][LINEAR.name]
            if decision["time_s"] is not None:
                log = json.loads(trial_path(args.source_dir, trial.person, trial.number).read_bytes())
                rows.append(dict(cohort=cohort, **audit_trigger(trial, log["user"], decision)))
        with (args.source_dir / cohort / "trials.csv").open(newline="", encoding="utf-8") as file:
            original = {
                (row["person"], int(row["trial"])): (
                    float(row["linear_projection_time_s"]), row["linear_projection_correct"] == "True"
                )
                for row in csv.DictReader(file) if row["linear_projection_time_s"]
            }
        observed = {(row["person"], row["trial"]): (row["trigger_s"], row["correct"]) for row in rows}
        if observed != original:
            raise ValueError(f"Audit did not reproduce saved triggers for {cohort}")
        summary[cohort] = {
            "total_trials": len(trials),
            **describe(rows),
            "by_person": {person: describe([row for row in rows if row["person"] == person]) for person in members},
        }
        all_rows.extend(rows)
    summary["all"] = describe(all_rows)
    write_json(args.output_dir / "exit_audit.json", summary)
    with (args.output_dir / "exit_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    for cohort, result in summary.items():
        print(cohort, {key: value for key, value in result.items() if key != "by_person"})


if __name__ == "__main__":
    main()
