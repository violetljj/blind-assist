"""Fetch the published Mobilio turn-validation logs and replay confirmation rules."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from urllib.error import URLError
from urllib.request import urlopen


REVISION = "9dd54b80e95a48a95bde9d405ce6e632d66fbff9"
SOURCE = f"https://raw.githubusercontent.com/Harvard-Slade-Lab/Mobilio/{REVISION}/data"
DEVELOPMENT = ("VI01", "VI05", "VI10", "VI15")
HOLDOUT = ("VI02", "VI03", "VI04", "VI06", "VI07", "VI08", "VI09", "VI11", "VI12", "VI14")
COHORTS = {"development": DEVELOPMENT, "holdout": HOLDOUT}
TRIAL_NUMBERS = range(200, 264)
WINDOW_SECONDS = 5.0
TOLERANCE_DEGREES = 10.0
HOLD_SECONDS = 0.5
DEFAULT_CUE = (3.0, 0.5, 1.0)
DEVELOPMENT_WRONG_BUDGET = 34


def source_trial_number(person, number):
    # VI03's participant directory places the 64 validation logs after the field logs.
    return number + 76 if person == "VI03" else number


def trial_path(run_dir, person, number):
    return run_dir / "raw" / person / f"data_{source_trial_number(person, number)}.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_trials(run_dir, cohort):
    def fetch_one(item):
        person, number = item
        path = trial_path(run_dir, person, number)
        url = f"{SOURCE}/{person}/data/data_{source_trial_number(person, number)}.json"
        if path.exists():
            body = path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=12) as response:
                        body = response.read()
                    break
                except (URLError, TimeoutError) as error:
                    if attempt == 2 or getattr(error, "code", None) == 404:
                        raise
            log = json.loads(body)
            if not {"time", "user", "target", "params"}.issubset(log):
                raise ValueError(f"Unexpected source schema at {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(body)
            temporary.replace(path)
        return {
            "person": person,
            "trial": number,
            "source_trial": source_trial_number(person, number),
            "url": url,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }

    members = [(person, number) for person in COHORTS[cohort] for number in TRIAL_NUMBERS]
    with ThreadPoolExecutor(max_workers=10) as executor:
        files = list(executor.map(fetch_one, members))
    manifest = {"revision": REVISION, "cohort": cohort, "files": files}
    write_json(run_dir / f"{cohort}_sources.json", manifest)
    print(f"{cohort}: {len(files)} logs, {sum(file['bytes'] for file in files)} bytes")


def wrap_degrees(value):
    return (value + 180) % 360 - 180


@dataclass(frozen=True)
class Trial:
    person: str
    number: int
    cue: str
    time_s: tuple
    error_deg: tuple
    params: tuple


def load_trials(run_dir, cohort):
    manifest = json.loads((run_dir / f"{cohort}_sources.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    expected = {(person, number) for person in COHORTS[cohort] for number in TRIAL_NUMBERS}
    if manifest["revision"] != REVISION or manifest["cohort"] != cohort:
        raise ValueError("Source manifest has a different revision or cohort")
    if len(files) != len(expected) or {(f["person"], f["trial"]) for f in files} != expected:
        raise ValueError("Source manifest does not cover exactly the requested validation logs")
    trials = []
    for file in files:
        person, number = file["person"], file["trial"]
        raw = trial_path(run_dir, person, number).read_bytes()
        if hashlib.sha256(raw).hexdigest() != file["sha256"]:
            raise ValueError(f"Source hash changed: {person}/{number}")
        log = json.loads(raw)
        if log["filename"] != f"HEADING_{number}":
            raise ValueError(f"Embedded trial ID does not match {person}/{number}")
        time_s = tuple(t - log["time"][0] for t in log["time"])
        error_deg = tuple(map(wrap_degrees, log["target"]))
        params = tuple(log["params"])
        if (
            len(time_s) != len(error_deg) or len(time_s) != len(log["user"])
            or time_s[-1] < HOLD_SECONDS or len(params) != 3
            or not all(map(math.isfinite, time_s + error_deg + params))
            or any(b <= a for a, b in zip(time_s, time_s[1:]))
            or not 59.99 <= abs(error_deg[0]) <= 120.01
        ):
            raise ValueError(f"Inconsistent validation trace: {person}/{number}")
        generic = all(abs(a - b) < 1e-4 for a, b in zip(params, DEFAULT_CUE))
        trials.append(Trial(person, number, "default" if generic else "personalized", time_s, error_deg, params))
    cue_counts = Counter((trial.person, trial.cue) for trial in trials)
    if any(cue_counts[person, cue] != 32 for person in COHORTS[cohort] for cue in ("default", "personalized")):
        raise ValueError("Expected 32 validation trials of each recorded cue for each participant")
    return trials


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    lookback_s: float = 0.0
    current_deg: float = TOLERANCE_DEGREES
    projected_deg: float = TOLERANCE_DEGREES
    history_deg: float | None = None


BASELINES = (
    Rule("first_entry", "entry"),
    Rule("dwell_500ms", "dwell", lookback_s=0.5),
    Rule("linear_projection", "linear", lookback_s=0.2),
    Rule("current_margin_8", "linear", lookback_s=0.2, current_deg=8),
)


def development_rules():
    rules = list(BASELINES)
    for current in range(1, 11):
        rules.append(Rule(f"entry_{current}", "entry", current_deg=current))
        for history in (0.1, 0.25, 0.3, 0.5, 0.75, 1.0):
            rules.append(Rule(f"dwell_{history}_{current}", "dwell", history, current))
            if current < 10:
                rules.append(Rule(
                    f"dwell_{history}_{current}_wide_history", "dwell", history, current, history_deg=10,
                ))
    for history in (0.1, 0.2, 0.3, 0.4):
        for projected in (*range(4, 17), 18, 20, 24, 30):
            rules.append(Rule(f"linear_{history}_{projected}", "linear", history, 10, projected))
            for current in range(6, 10):
                rules.append(Rule(f"margin_{history}_{current}_{projected}", "linear", history, current, projected))
    return rules


def replay(trial, rules, tolerance=TOLERANCE_DEGREES, hold=HOLD_SECONDS):
    time, error = trial.time_s, trial.error_deg
    end = bisect_right(time, WINDOW_SECONDS)
    first_stable = None
    decisions = {rule.name: {"time_s": None, "correct": None} for rule in rules}
    pending = list(rules)
    for index, now in enumerate(time[:end]):
        future = bisect_left(time, now + hold, lo=index)
        if future >= end:
            break
        if abs(error[index]) > tolerance:
            continue
        # These labels are used only after a decision and to locate the earliest evaluable opportunity.
        correct = all(abs(value) <= tolerance for value in error[index:future + 1])
        if correct and first_stable is None:
            first_stable = now

        past = {}
        for rule in pending.copy():
            current_limit = min(tolerance, rule.current_deg)
            if abs(error[index]) > current_limit:
                continue
            if rule.kind != "entry":
                if rule.lookback_s not in past:
                    past[rule.lookback_s] = bisect_right(time, now - rule.lookback_s, hi=index) - 1
                earlier = past[rule.lookback_s]
                if earlier < 0:
                    continue
                if rule.kind == "dwell":
                    history_limit = current_limit if rule.history_deg is None else min(tolerance, rule.history_deg)
                    if any(abs(value) > history_limit for value in error[earlier:index]):
                        continue
                else:
                    rate = wrap_degrees(error[index] - error[earlier]) / (now - time[earlier])
                    if abs(error[index] + hold * rate) > rule.projected_deg:
                        continue
            decisions[rule.name] = {"time_s": now, "correct": correct}
            pending.remove(rule)
    return {
        "person": trial.person,
        "trial": trial.number,
        "cue": trial.cue,
        "initial_deg": error[0],
        "mean_normalized_error": (
            mean(abs(e) for t, e in zip(time, error) if t < WINDOW_SECONDS) / abs(error[0])
            if time[-1] >= WINDOW_SECONDS else None
        ),
        "observed_seconds": min(time[-1], WINDOW_SECONDS),
        "first_stable_s": first_stable,
        "decisions": decisions,
    }


def score_rule(records, name):
    decisions = [record["decisions"][name] for record in records]
    opportunities = [record["first_stable_s"] is not None for record in records]
    correct_times = [d["time_s"] for d in decisions if d["correct"] is True]
    return {
        "trials": len(records),
        "available": sum(opportunities),
        "correct": len(correct_times),
        "wrong": sum(d["correct"] is False for d in decisions),
        "silent": sum(d["correct"] is None for d in decisions),
        "silent_with_opportunity": sum(
            d["correct"] is None and available for d, available in zip(decisions, opportunities)
        ),
        "median_correct_s": median(correct_times) if correct_times else None,
    }


def compare_rules(records, baseline, candidate):
    pairs = 0
    time_differences = []
    gains = losses = 0
    for record in records:
        a, b = (record["decisions"][name] for name in (baseline, candidate))
        a_correct = a["correct"] is True
        b_correct = b["correct"] is True
        if a_correct and b_correct:
            pairs += 1
            time_differences.append(b["time_s"] - a["time_s"])
        else:
            gains += b_correct
            losses += a_correct
    return {
        "common_correct": pairs,
        "median_paired_delta_s": median(time_differences) if pairs else None,
        "gained_correct": gains,
        "lost_correct": losses,
    }


def cue_summary(records):
    summary = {}
    for cue in ("default", "personalized"):
        rows = [r for r in records if r["cue"] == cue]
        times = [r["first_stable_s"] for r in rows if r["first_stable_s"] is not None]
        full_window_errors = [r["mean_normalized_error"] for r in rows if r["mean_normalized_error"] is not None]
        angle_counts = Counter(round(abs(r["initial_deg"])) for r in rows)
        summary[cue] = {
            "trials": len(rows),
            "stable": len(times),
            "normalized_error_trials": len(full_window_errors),
            "mean_normalized_error": mean(full_window_errors),
            "median_when_stable_s": median(times) if times else None,
            "mean_time_capped_at_5s": mean(
                r["first_stable_s"] if r["first_stable_s"] is not None else WINDOW_SECONDS for r in rows
            ),
            "absolute_turn_counts": dict(sorted(angle_counts.items())),
        }
    return summary


def mark_examples(run_dir, trials, records, rules):
    by_id = {(r["person"], r["trial"]): r for r in records}
    trial_by_id = {(t.person, t.number): t for t in trials}
    examples = []
    for kind, key in (("transient", ("VI10", 200)), ("stable", ("VI05", 201))):
        row = by_id[key]
        trial = trial_by_id[key]
        end = bisect_right(trial.time_s, WINDOW_SECONDS)
        examples.append({
            "kind": kind,
            "person": trial.person,
            "trial": trial.number,
            "cue": trial.cue,
            "time_s": trial.time_s[:end],
            "error_deg": trial.error_deg[:end],
            "first_stable_s": row["first_stable_s"],
            "decisions": {rule.name: row["decisions"][rule.name] for rule in rules},
        })
    write_json(run_dir / "examples.json", {
        "window_seconds": WINDOW_SECONDS,
        "tolerance_deg": TOLERANCE_DEGREES,
        "hold_seconds": HOLD_SECONDS,
        "examples": examples,
    })


def select_development_rules(rules, results):
    families = defaultdict(list)
    for rule in rules[len(BASELINES):]:
        family = "margin" if rule.name.startswith("margin_") else rule.kind
        if results[rule.name]["wrong"] <= DEVELOPMENT_WRONG_BUDGET:
            families[family].append(rule)

    def rank(rule):
        result = results[rule.name]
        time = result["median_correct_s"]
        return (-result["correct"], result["wrong"], time if time is not None else WINDOW_SECONDS)

    return {family: asdict(min(options, key=rank)) for family, options in families.items()}


def evaluate(run_dir, cohort, comparison):
    trials = load_trials(run_dir, cohort)
    rules = development_rules() if comparison is None else [
        Rule(**row) for row in json.loads(comparison.read_text(encoding="utf-8"))["rules"]
    ]
    records = [replay(trial, rules) for trial in trials]
    overall = {rule.name: score_rule(records, rule.name) for rule in rules}
    selected = select_development_rules(rules, overall) if comparison is None else {}
    shown = tuple(BASELINES) + tuple(Rule(**row) for row in selected.values()) if selected else rules
    cohort_dir = run_dir / cohort
    by_person = {person: [r for r in records if r["person"] == person] for person in COHORTS[cohort]}
    report = {
        "cohort": cohort,
        "configuration": [asdict(rule) for rule in shown],
        "selected_by_family": selected,
        "settings": {
            "window_seconds": WINDOW_SECONDS,
            "tolerance_deg": TOLERANCE_DEGREES,
            "hold_seconds": HOLD_SECONDS,
        },
        "overall": {rule.name: overall[rule.name] for rule in shown},
        "by_person": {
            person: {rule.name: score_rule(rows, rule.name) for rule in shown}
            for person, rows in by_person.items()
        },
        "cues_by_person": {person: cue_summary(rows) for person, rows in by_person.items()},
        "cues_overall": cue_summary(records),
        "comparisons": {
            f"{a.name}--{b.name}": {
                "overall": compare_rules(records, a.name, b.name),
                "by_person": {person: compare_rules(rows, a.name, b.name) for person, rows in by_person.items()},
            }
            for i, a in enumerate(shown) for b in shown[i+1:]
        },
    }
    write_json(cohort_dir / "summary.json", report)
    if selected:
        write_json(cohort_dir / "rule_sweep.json", [
            {"rule": asdict(rule), **overall[rule.name]} for rule in rules
        ])
        mark_examples(cohort_dir, trials, records, shown)
    with (cohort_dir / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        rows = []
        for record in records:
            flat = {key: value for key, value in record.items() if key != "decisions"}
            for rule in shown:
                decision = record["decisions"][rule.name]
                flat.update({f"{rule.name}_{key}": value for key, value in decision.items()})
            rows.append(flat)
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cohort": cohort, "overall": report["overall"], "selected": selected}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(COHORTS), required=True)
    parser.add_argument("--comparison", type=Path, help="Fixed rule configuration for the holdout.")
    args = parser.parse_args()
    if args.action == "fetch":
        fetch_trials(args.run_dir, args.cohort)
    else:
        if args.cohort == "holdout" and args.comparison is None:
            parser.error("Holdout evaluation requires a fixed --comparison")
        evaluate(args.run_dir, args.cohort, args.comparison)


if __name__ == "__main__":
    main()
