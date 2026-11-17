"""Replay the original confirmation rules on Mobilio's separate calibration phase."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from mobilio_turn_confirmation import (
    BASELINES,
    DEVELOPMENT,
    HOLDOUT,
    REVISION,
    TOLERANCE_DEGREES,
    WINDOW_SECONDS,
    Trial,
    compare_rules,
    replay,
    score_rule,
    wrap_degrees,
    write_json,
)
from mobilio_turn_exit_audit import LINEAR, audit_trigger, describe


CALIBRATION_TRIALS = 200
RULES = BASELINES[:3]
REOPEN_SECONDS = 0.2
SUSTAINED_SECONDS = 0.5


def load_calibration(archive):
    manifest = []
    records = []
    motion = []
    runs = []
    for person in sorted(DEVELOPMENT + HOLDOUT):
        root = f"Mobilio-{REVISION}/data/{person}/data"
        cma_body = archive.read(f"{root}/CMA.json")
        cma = json.loads(cma_body)
        manifest.append({"path": f"{root}/CMA.json", "bytes": len(cma_body), "sha256": hashlib.sha256(cma_body).hexdigest()})
        last_start = None
        for number in range(CALIBRATION_TRIALS):
            path = f"{root}/data_{number}.json"
            body = archive.read(path)
            log = json.loads(body)
            params = tuple(log["params"])
            time_s = tuple(t - log["time"][0] for t in log["time"])
            error_deg = tuple(map(wrap_degrees, log["target"]))
            headings = log["user"]
            if (
                not (len(time_s) == len(error_deg) == len(headings) == len(log["localX"]) == len(log["localZ"]))
                or log["filename"] != f"HEADING_{number}"
                or len(params) != 3
                or not all(map(math.isfinite, (*time_s, *error_deg, *headings, *params)))
                or not 59.99 <= abs(error_deg[0]) <= 120.01
                or any(b <= a for a, b in zip(time_s, time_s[1:]))
                or (last_start is not None and last_start >= log["time"][0])
                or not all(
                    math.isclose(param, scale * normalized, abs_tol=1e-7)
                    for param, scale, normalized in zip(params, (11, 1, 2), cma["inputs"][number], strict=True)
                )
            ):
                raise ValueError(f"Calibration identity or time/parameter alignment failed: {path}")
            last_start = log["time"][0]
            trial = Trial(person, number, "adaptive", time_s, error_deg, params)
            record = replay(trial, RULES)
            manifest.append({"path": path, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
            records.append(record)
            decision = record["decisions"][LINEAR.name]
            if decision["time_s"] is None:
                continue
            motion.append(audit_trigger(trial, headings, decision))
            runs.extend(exit_runs(trial, decision))
    return manifest, records, motion, runs


def exit_runs(trial, decision):
    """Enumerate out-of-band runs after the original cue, retaining right censoring."""
    time, error = trial.time_s, trial.error_deg
    trigger = bisect_left(time, decision["time_s"])
    end = bisect_right(time, WINDOW_SECONDS)
    index = trigger + 1
    while index < end:
        if abs(error[index]) <= TOLERANCE_DEGREES:
            index += 1
            continue
        first = index
        side = math.copysign(1, error[first])
        while index < end and side * error[index] > TOLERANCE_DEGREES:
            index += 1
        update = bisect_left(time, time[first] + REOPEN_SECONDS, lo=first, hi=index)
        yield {
            "person": trial.person,
            "trial": trial.number,
            "original_correct": decision["correct"],
            "exit_time_s": time[first],
            "side": int(side),
            "delay_to_debounced_update_s": time[update] - time[first] if update < index else None,
            "sustained_observed": time[index - 1] >= time[first] + SUSTAINED_SECONDS,
            "right_censored": index == end,
            "last_observed_duration_s": time[index - 1] - time[first],
            "first_sample_after_run_s": time[index] if index < end else None,
        }


def summarize(records, motion, runs):
    update_runs = [run for run in runs if run["delay_to_debounced_update_s"] is not None]
    wrong_sustained = {
        (run["person"], run["trial"])
        for run in runs if not run["original_correct"] and run["sustained_observed"]
    }
    return {
        "shorter_than_5s": sum(record["observed_seconds"] < WINDOW_SECONDS for record in records),
        "rules": {rule.name: score_rule(records, rule.name) for rule in RULES},
        "linear_vs_first_entry": compare_rules(records, "first_entry", LINEAR.name),
        "first_exit": describe(motion),
        "follow_through": {
            "exit_runs": len(runs),
            "trials_with_exit": len({(run["person"], run["trial"]) for run in runs}),
            "sustained_exit_runs": sum(run["sustained_observed"] for run in runs),
            "wrong_confirmations_with_sustained_exit": len(wrong_sustained),
            "delayed_updates": len(update_runs),
            "delayed_updates_sustained": sum(run["sustained_observed"] for run in update_runs),
            "delayed_updates_right_censored_without_sustain": sum(
                run["right_censored"] and not run["sustained_observed"] for run in update_runs
            ),
            "median_update_delay_s": median(run["delay_to_debounced_update_s"] for run in update_runs)
            if update_runs else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        if archive.comment.decode("ascii") != REVISION:
            raise ValueError("Archive's embedded commit comment does not match the pinned revision")
        files, records, motion, runs = load_calibration(archive)
    summary = summarize(records, motion, runs)
    summary["by_person"] = {
        person: summarize(
            [record for record in records if record["person"] == person],
            [row for row in motion if row["person"] == person],
            [row for row in runs if row["person"] == person],
        )
        for person in sorted(DEVELOPMENT + HOLDOUT)
    }
    summary["exit_attribution_counts"] = dict(Counter(row["attribution"] for row in motion if not row["correct"]))
    with args.archive.open("rb") as archive_file:
        archive_hash = hashlib.file_digest(archive_file, "sha256").hexdigest()
    write_json(args.output_dir / "sources.json", {
        "revision": REVISION,
        "source_url": f"https://codeload.github.com/Harvard-Slade-Lab/Mobilio/zip/{REVISION}",
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_hash,
        "files": files,
    })
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "trials.json", records)
    for name, rows in (("linear_triggers.csv", motion), ("exit_runs.csv", runs)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "by_person"}, indent=2))


if __name__ == "__main__":
    main()
