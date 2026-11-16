"""Score new real departure windows with the frozen depth-checked history rule."""

import argparse
import json
from pathlib import Path

import numpy as np

from real_multiframe_mixed_20261117 import read_bundle, restrict, sees_target
from real_multiframe_motion_20261117 import METHODS, compare, confirm_motion
from real_multiframe_pilot_20261117 import save


BANDS = ("head", "body")


def alarms(signal, rows):
    return np.asarray([signal[row]["alarm"] for row in rows], dtype=bool)


def band_tails(people, rows, times):
    active = alarms(people, rows)
    tails = {}
    for band, label in enumerate(BANDS):
        signal = active[:, :, band]
        metrics = {}
        for i, name in enumerate(METHODS):
            first_quiet = np.flatnonzero(~signal[:, i])
            metrics[name] = dict(
                first_quiet_s=float(times[first_quiet[0]] - times[0]) if first_quiet.size else None,
                observed_s=float(times[-1] - times[0]),
                alarm_frames=int(signal[:, i].sum()),
            )
        tails[label] = dict(
            methods=metrics,
            added_rows=[row for row, changed in zip(rows, signal[:, 3] & ~signal[:, 2], strict=True) if changed],
        )
    return tails


def score_event(event, frames, signals, person_present, static_present):
    start, exit_row, end = (event[key] for key in ("start", "exit", "end"))
    if not 0 <= start < exit_row < end < len(frames):
        raise ValueError("Invalid raw-annotated departure window")
    rows = list(range(exit_row, end + 1))
    if not set(rows).issubset(signals["full"]):
        raise ValueError("Counting window includes a missing frame or lacks the 2/3 warmup")
    if any(person_present[row] for row in rows) or not any(person_present[row] for row in range(start, exit_row)):
        raise ValueError("The actor must reach the corridor and be absent throughout the scored departure window")

    hidden = set(event.get("hidden_rows", ()))
    observed_static = {row for row in rows if static_present[row]}
    static_in_corridor = set(event.get("static_rows", observed_static | hidden))
    if not hidden.isdisjoint(observed_static) or not hidden.issubset(static_in_corridor):
        raise ValueError("A hidden static query must have no current target support and be labeled in the corridor")
    if not static_in_corridor.issubset(rows):
        raise ValueError("Static queries must lie in the post-departure window")

    target = alarms(signals["obstacle"], rows).any(axis=-1)
    eligible = [i for i, row in enumerate(rows) if row in static_in_corridor and target[i, 1] and not target[i, 0]]
    hidden_eligible = [i for i in eligible if rows[i] in hidden]
    full = alarms(signals["full"], rows)
    full_added = full[:, 3] & ~full[:, 2]
    full_any = full.any(axis=-1)
    times = np.array([frames[row].time for row in rows])
    return dict(
        episode=event,
        rows=rows,
        times_s=times.tolist(),
        prior_coverage_s=float(times[0] - frames[0].time),
        person=band_tails(signals["people"], rows, times),
        static=dict(
            in_corridor_rows=sorted(static_in_corridor),
            observed_rows=sorted(observed_static),
            memory_required_rows=[rows[i] for i in eligible],
            no_current_depth_rows=[rows[i] for i in hidden_eligible],
            memory_confirmed={name: [rows[i] for i in eligible if target[i, j]] for j, name in enumerate(METHODS)},
        ),
        full_band_added_rows={label: [row for row, added in zip(rows, full_added[:, i], strict=True) if added]
                              for i, label in enumerate(BANDS)},
        full_any_added_rows=[row for row, added in zip(rows, full_any[:, 3] & ~full_any[:, 2], strict=True) if added],
    )


def run_case(case, snapshot):
    source = Path(case["source"])
    cache = Path(case["cached_masks"]) if "cached_masks" in case else None
    radius, _, expanded, human, obstacle = read_bundle(source, snapshot, cache)
    for frame, person, target in zip(expanded, human, obstacle, strict=True):
        if person.shape != frame.weights.shape or target.shape != person.shape or np.any(person & target):
            raise ValueError(f"Reference rays disagree or overlap in source frame {frame.row}")
    events = json.loads((source / "episodes.json").read_text(encoding="utf-8"))
    marked, evidence = confirm_motion(expanded)
    full = compare(expanded, marked)
    moving = restrict(expanded, human)
    target = restrict(expanded, obstacle)
    signals = dict(full=full,
                   people=compare(moving, restrict(marked, human)),
                   obstacle=compare(target, restrict(marked, obstacle)))
    person_present = [sees_target(frame) for frame in moving]
    static_present = [sees_target(frame) for frame in target]
    summary = [score_event(event, expanded, signals, person_present, static_present) for event in events]

    # Test the cutoff where the actor first leaves, with all subsequent depth removed.
    for event in events:
        stop = event["exit"] + 1
        prefix, _ = confirm_motion(expanded[:stop])
        result = compare(expanded[:stop], prefix)
        if any(item != full[row] for row, item in result.items()):
            raise AssertionError(f"Later observations changed the departure prefix at row {stop - 1}")
    return dict(source=str(source), mask_dilation_px=radius, summary=summary, evidence=evidence, signals=signals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = {}
    for name, case in cases.items():
        results[name] = run_case(case, args.snapshot)
        print(name, json.dumps(results[name]["summary"], ensure_ascii=False), flush=True)
    save(args.output, results)


if __name__ == "__main__":
    main()
