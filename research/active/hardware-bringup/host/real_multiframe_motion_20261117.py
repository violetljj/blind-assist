"""Require causal depth evidence before applying the fixed short person history."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from real_multiframe_mixed_20261117 import read_bundle, restrict
from real_multiframe_person_20261117 import PERSON_HISTORY_S
from real_multiframe_pilot_20261117 import HISTORY_S, follows, replay, save


METHODS = ("single", "memory", "fixed_person", "motion_checked")
WITNESS_FRAMES = 2


def confirm_motion(frames):
    """Store when a second distinct frame witnesses the old position as free."""
    bounds = [0, *(i for i in range(1, len(frames)) if not follows(frames[i - 1], frames[i])), len(frames)]
    marked, evidence = [], []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        segment = frames[start:end]
        times = np.array([frame.time for frame in segment])
        for index, source in enumerate(segment):
            candidates = np.flatnonzero(source.person)
            points = source.cloud[candidates]
            hits = np.zeros(len(points), dtype=np.int16)
            confirmed_at = np.full(len(source.cloud), np.inf)
            eps = 1e-8
            lo = np.searchsorted(times, source.time - HISTORY_S - eps)
            hi = np.searchsorted(times, source.time + HISTORY_S + eps, side="right")
            for neighbor in range(lo, hi):
                if neighbor == index or not len(points):
                    continue
                observed = segment[neighbor]
                free = observed.witnessed_free(points)
                hits[free] += 1
                confirmed_at[candidates[free & (hits == WITNESS_FRAMES)]] = max(source.time, observed.time)
            marked.append(replace(source, person_motion_at=confirmed_at))
            evidence.append(dict(row=source.row, mask_points=len(points), one_witness=int((hits == 1).sum()),
                                 confirmed=int((hits >= WITNESS_FRAMES).sum())))
    return marked, evidence


def compare(fixed, marked):
    baseline = replay(fixed, keep_trace=True, person_history_s=PERSON_HISTORY_S)
    checked = replay(marked, keep_trace=True, person_history_s=PERSON_HISTORY_S)
    if baseline.keys() != checked.keys():
        raise AssertionError("Geometry changed replay coverage")
    for row, item in baseline.items():
        new = checked[row]
        for key in ("raw", "alarm"):
            if item[key][:3] != new[key][:3]:
                raise AssertionError(f"Geometry changed a common control at row {row}")
            old_person, motion, memory = (np.array(arm) for arm in (item[key][3], new[key][3], item[key][1]))
            if np.any(old_person & ~motion) or np.any(motion & ~memory):
                raise AssertionError(f"Motion history outside fixed filter and original memory at row {row}")
            item[key] = [item[key][0], item[key][1], item[key][3], new[key][3]]
    return baseline


def verify_old(actual, old):
    if {str(row) for row in actual} != old.keys():
        raise AssertionError("Window differs from the prior experiment")
    for row, item in actual.items():
        prior = old[str(row)]
        if item["time"] != prior["time"] or item["history_frames"] != prior["history_frames"]:
            raise AssertionError(f"Timing differs at row {row}")
        for key in ("raw", "alarm"):
            if item[key][:3] != [prior[key][i] for i in (0, 1, 3)]:
                raise AssertionError(f"An earlier replay arm changed at row {row}")


def active(signal, rows):
    return np.array([signal[row]["alarm"] for row in rows], dtype=bool).any(axis=-1)


def mixed_results(signals, previous):
    results = []
    for event in previous["events"]:
        rows = event["rows"]
        times = np.array(event["times"])
        people = active(signals["people"], rows)
        target = active(signals["obstacle"], rows)
        full = active(signals["full"], rows)
        hidden = event.get("hidden_obstacle", {}).get("memory_only_rows", [])
        metrics = {}
        for column, method in enumerate(METHODS):
            quiet = np.flatnonzero(~people[:, column])
            metrics[method] = dict(
                tail_s=float(times[quiet[0]] - times[0]) if len(quiet) else None,
                person_alarm_frames=int(people[:, column].sum()),
                static_alarm_frames=int(target[:, column].sum()),
                full_alarm_frames=int(full[:, column].sum()),
                hidden_static_alarm_frames=sum(bool(target[rows.index(row), column]) for row in hidden),
            )
        results.append(dict(rows=rows, times_s=times.tolist(), hidden_static_rows=hidden, metrics=metrics))
    return results


def static_results(signals, source_indices, selected):
    rows = [row for row, signal in signals["obstacle"].items()
            if str(source_indices[row]) in selected and any(signal["alarm"][1]) and not any(signal["alarm"][2])]
    return [dict(row=row, source_index=int(source_indices[row]), time_s=signals["obstacle"][row]["time"],
                 static={m: bool(v) for m, v in zip(METHODS, active(signals["obstacle"], [row])[0], strict=True)},
                 full={m: bool(v) for m, v in zip(METHODS, active(signals["full"], [row])[0], strict=True)})
            for row in rows]


def run_case(case, snapshot):
    source = Path(case["source"])
    cache = Path(case["cached_masks"]) if "cached_masks" in case else None
    radius, _, expanded, human, obstacle = read_bundle(source, snapshot, cache)
    previous = json.loads((source / case["previous"]).read_text(encoding="utf-8"))
    marked, evidence = confirm_motion(expanded)
    # Reproduce a prefix while withholding the later observations from the motion calculation.
    stop = len(expanded) // 2 + 1
    prefix, _ = confirm_motion(expanded[:stop])
    prefix_cutoff = prefix[-1].time
    for truncated, whole in zip(prefix, marked[:stop], strict=True):
        if not np.array_equal(truncated.person_motion_at <= prefix_cutoff,
                              whole.person_motion_at <= prefix_cutoff):
            raise AssertionError(f"A later depth frame leaked into row {truncated.row}")
    moving = compare(expanded, marked)
    prefix_replay = compare(expanded[:stop], prefix)
    if any(item != moving[row] for row, item in prefix_replay.items()):
        raise AssertionError("Prefix changed full alarm output")

    mixed = "events" in previous
    signals = dict(full=moving, obstacle=compare(restrict(expanded, obstacle), restrict(marked, obstacle)))
    verify_old(signals["full"], previous["full"])
    verify_old(signals["obstacle"], previous["obstacle" if mixed else "table"])
    if mixed:
        signals["people"] = compare(restrict(expanded, human), restrict(marked, human))
        verify_old(signals["people"], previous["people"])
        summary = mixed_results(signals, previous)
    else:
        with np.load(source / "frames.npz", allow_pickle=False) as bundle:
            summary = static_results(signals, bundle["source_indices"], previous["selected"])
    return dict(source=str(source), mask_dilation_px=radius, evidence=evidence, summary=summary, signals=signals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {}
    for name, case in cases.items():
        result[name] = run_case(case, args.snapshot)
        print(name, json.dumps(result[name]["summary"], ensure_ascii=False), flush=True)
    save(args.output, result)


if __name__ == "__main__":
    main()
