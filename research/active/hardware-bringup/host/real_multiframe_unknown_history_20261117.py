"""Prove an unknown actor's historical body vote before comparing the two body filters."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real_multiframe_foreground_20261117 import confirm_foreground, observations
from real_multiframe_mixed_20261117 import read_bundle, restrict, sees_target
from real_multiframe_motion_transfer_20261117 import BANDS, band_tails, score_event
from real_multiframe_person_20261117 import PERSON_HISTORY_S
from real_multiframe_pilot_20261117 import (
    BODY_BAND, HISTORY_S, MIN_SOLID_ANGLE, corridor_bands, follows, replay, save,
)
from real_multiframe_static_history_20261117 import candidate, corroborated_at, verify_candidate


METHODS = ("single", "memory", "cleared", "motion_veto")


def pose_coverage(pose, depth, rgb):
    bonn = pose.suffix == ".npz"
    if bonn:
        with np.load(pose, allow_pickle=False) as raw:
            timestamps = raw["raw_pose"][:, 0]
        max_pose, max_pair = .08, .04  # Original Bonn acquisition limits.
    else:
        timestamps = np.loadtxt(pose, usecols=0)
        max_pose, max_pair = .020, 0.  # Original ETH3D synchronized inputs.
    if np.any(np.diff(depth) <= 0.) or np.any(np.diff(rgb) <= 0.) or np.any(abs(rgb - depth) > max_pair):
        raise ValueError("Depth and RGB are not the original distinct paired observations")
    right = np.searchsorted(timestamps, depth)
    if np.any((right == 0) | (right == len(timestamps))):
        raise ValueError("A replay frame lacks bracketing motion capture")
    spans = timestamps[right] - timestamps[right - 1]
    exceeds_limit = spans >= max_pose if bonn else spans > max_pose
    if np.any(spans <= 0.) or np.any(exceeds_limit):
        raise ValueError("A bracketing pose gap exceeds the original dataset limit")
    return dict(max_pose_ms=float(1000 * spans.max()), max_rgb_depth_ms=float(1000 * abs(rgb-depth).max()),
                spacing_ms=(1000 * np.diff(depth)).tolist())


def load(case, snapshot):
    folder = Path(case["source"])
    stride, _, frames, human, obstacle, masks = read_bundle(folder, snapshot, include_pixel_masks=True)
    with np.load(folder / "frames.npz", allow_pickle=False) as data:
        coverage = pose_coverage(Path(case["pose"]), data["depth_timestamp"], data["rgb_timestamp"])
        intrinsics = data["intrinsics"]
        views = list(observations(data, frames, masks, stride))
        ids = data["source_indices"].tolist() if "source_indices" in data else list(range(len(frames)))
    if any(not follows(left, right) for left, right in zip(frames[:-1], frames[1:], strict=True)):
        raise ValueError("The scored window contains a discontinuity")
    gate = {f.row for f in frames if f.weights[corridor_bands(f.cloud, f.origin, f.plane)[BODY_BAND]].sum()
            < MIN_SOLID_ANGLE}
    marked, _ = confirm_foreground(frames, views, intrinsics)
    ready = corroborated_at(frames, views, intrinsics)
    return marked, ready, gate, human, obstacle, stride, ids, coverage


def motion_veto(frames, gate):
    def allowed(past, query):
        return past.person_motion_at > query.time if query.row in gate else None

    return replay(frames, keep_trace=True, person_history_s=PERSON_HISTORY_S, body_history=allowed)


def replay_arms(frames, ready, gate):
    control = motion_veto(frames, gate)
    tested = candidate(frames, ready, gate)
    verify_candidate(control, tested, gate)
    return control, tested


def source_weights(frames, ready, actor, row):
    """Remove just the annotated actor's unknown old points from each source's motion-veto vote."""
    query = frames[row]
    sources, masks = [], {}
    for past in reversed(frames[:row]):
        if query.time - past.time > HISTORY_S + 1e-8:
            break
        body = corridor_bands(past.cloud, query.origin, query.plane)[BODY_BAND]
        kept = body & (past.person_motion_at > query.time)
        unknown_actor = kept & (ready[past.row] >= query.time) & actor[past.row]
        if not kept.any():
            continue
        sources.append(dict(
            row=past.row, time_s=past.time, total_sr=float(past.weights[kept].sum()),
            unknown_actor_n=int(unknown_actor.sum()),
            unknown_actor_sr=float(past.weights[unknown_actor].sum()),
            without_unknown_actor_sr=float(past.weights[kept & ~unknown_actor].sum()),
        ))
        masks[past.row] = unknown_actor
    return sources, masks


def screen(frames, ready, gate, actor, obstacle, episode):
    signal = motion_veto(frames, gate)
    people_frames, obstacle_frames = restrict(frames, actor), restrict(frames, obstacle)
    attribution = {"full": signal, "people": motion_veto(people_frames, gate),
                   "obstacle": motion_veto(obstacle_frames, gate)}
    event = score_event(episode, frames, attribution, [sees_target(f) for f in people_frames],
                        [sees_target(f) for f in obstacle_frames], methods=METHODS)
    eligible_rows = event["rows"]
    support = {}
    for row in eligible_rows:
        if row not in gate or not signal[row]["raw"][-1][BODY_BAND]:
            continue
        weights, masks = source_weights(frames, ready, actor, row)
        if not any(s["total_sr"] >= MIN_SOLID_ANGLE for s in weights):
            raise AssertionError("Per-source raw vote does not reproduce the motion-veto arm")
        without = any(s["without_unknown_actor_sr"] >= MIN_SOLID_ANGLE for s in weights)
        sufficient = [s for s in weights if s["unknown_actor_sr"] >= MIN_SOLID_ANGLE]
        support[row] = (weights, masks, without, sufficient)
    proofs, overlays = [], []
    for confirm_row in eligible_rows:
        if confirm_row not in gate or not signal[confirm_row]["alarm"][-1][BODY_BAND]:
            continue
        slots = range(confirm_row - 2, confirm_row + 1)
        raw = [signal[row]["raw"][-1][BODY_BAND] for row in slots]
        decisive = [row for row in slots if row in support and not support[row][2]]
        without = [value and row not in decisive for row, value in zip(slots, raw, strict=True)]
        sufficient_rows = [row for row in decisive if support[row][3]]
        if sum(without) >= 2 or not sufficient_rows:
            continue
        restored = [value or row in sufficient_rows for row, value in zip(slots, without, strict=True)]
        if sum(restored) < 2:
            continue
        proofs.append(dict(confirm_row=confirm_row, vote_rows=list(slots), raw=raw,
                           without_unknown_actor=without, restored_with_unknown_actor=restored,
                           removed_votes=decisive))
        overlays.extend((row, s["row"], support[row][1][s["row"]])
                        for row in sufficient_rows for s in support[row][3])
    return dict(proofs=proofs, exit_row=episode["exit"], rows=eligible_rows,
                source_support={row: value[0] for row, value in support.items()},
                signal=attribution), overlays


def write_overlay(folder, file, overlays, stride, frames, ids):
    tiles = []
    with np.load(folder / "frames.npz", allow_pickle=False) as data:
        colors, scale = data["bgr"], data["depth_scale"].item()
        for query, source, points in overlays:
            observed, after = colors[source].copy(), colors[query].copy()
            depth = data["depth_png"][source, ::stride, ::stride].astype(np.float32) / scale
            y, x = np.nonzero((depth >= .35) & (depth <= 5.46))
            if len(points) != len(x):
                raise AssertionError("Attribution points do not match original sampled pixels")
            for px, py in zip(x[points], y[points], strict=True):
                cv2.circle(observed, (int(px * stride), int(py * stride)), 2, (0, 200, 255), -1)
            for tile, row, stage in ((observed, source, "Source of unknown old points"),
                                     (after, query, "Post-departure query")):
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 48), (18, 18, 18), -1)
                cv2.putText(tile, stage, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(tile, f"Row {ids[row]} / {frames[row].time:.3f} s", (8, 39),
                            cv2.FONT_HERSHEY_SIMPLEX, .49, (0, 210, 255), 1, cv2.LINE_AA)
            tiles.append(np.hstack((observed, after)))
    if not cv2.imwrite(str(file), np.vstack(tiles)):
        raise OSError(f"Could not save {file}")


def by_band(signals, rows, t, method):
    metrics = band_tails(signals, rows, t, methods=(*METHODS[:3], method))
    for i, band in enumerate(BANDS):
        entry = metrics[band]["methods"][method]
        entry["confirmed_rows"] = [row for row in rows if signals[row]["alarm"][-1][i]]
        metrics[band] = entry
    return metrics


def compare(frames, ready, gate, human, obstacle, proof, table_row=None):
    full_before, full_after = replay_arms(frames, ready, gate)
    references = {"people": human} if proof else {"obstacle": obstacle}
    inputs = {"full": (full_before, full_after)}
    for label, flags in references.items():
        ready_subset = {f.row: ready[f.row][mask] for f, mask in zip(frames, flags, strict=True)}
        inputs[label] = replay_arms(restrict(frames, flags), ready_subset, gate)
    if proof:
        rows = proof["rows"]
        if any({str(row): item for row, item in before.items()} != proof["signal"][label]
               for label, (before, _) in inputs.items()):
            raise AssertionError("Comparison no longer reproduces the admitted unknown support")
        times = np.array([frames[row].time for row in rows])
        scores = {method: {label: by_band(pair[i], rows, times, method) for label, pair in inputs.items()}
                  for i, method in enumerate(("motion_veto", "static_body_history"))}
    else:
        scores = {method: {label: pair[i][table_row]["alarm"][-1] for label, pair in inputs.items()}
                  for i, method in enumerate(("motion_veto", "static_body_history"))}
    return dict(scores=scores, signals=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "compare"))
    parser.add_argument("case")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cases = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    case = cases[args.case]
    source = Path(case["source"])
    screened = args.output / f"{args.case}_screen.json"
    if args.stage == "screen":
        frames, ready, gate, human, obstacle, stride, ids, coverage = load(case, args.snapshot)
        episode, = json.loads((source / "episodes.json").read_text(encoding="utf-8"))
        proof, overlays = screen(frames, ready, gate, human, obstacle, episode)
        save(screened, dict(source=str(source), times_s=[f.time for f in frames], coverage=coverage, **proof))
        if overlays:
            distinct = {(query, row): points for query, row, points in overlays}
            write_overlay(source, args.output / f"{args.case}_unknown_points.jpg",
                          [(q, row, pts) for (q, row), pts in distinct.items()], stride, frames, ids)
        print(json.dumps(dict(case=args.case, proofs=proof["proofs"], coverage=coverage)), flush=True)
        return
    proof = json.loads(screened.read_text(encoding="utf-8"))
    if not proof["proofs"] or proof["source"] != str(source):
        raise ValueError("The requested window has no demonstrated unknown body-tail support")
    for name in (args.case, "table"):
        selected = cases[name]
        frames, ready, gate, human, obstacle, _, ids, coverage = load(selected, args.snapshot)
        if name == "table" and ids[selected["score_row"]] != 1101:
            raise ValueError("The static control is no longer the original score frame")
        result = compare(frames, ready, gate, human, obstacle, proof if name == args.case else None,
                         table_row=selected.get("score_row"))
        save(args.output / f"{name}_comparison.json", dict(coverage=coverage, **result))
        print(name, json.dumps(result["scores"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
