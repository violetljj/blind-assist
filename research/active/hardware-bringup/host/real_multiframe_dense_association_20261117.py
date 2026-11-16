"""Fill missing 5 Hz foreground links with native RGB-D, replay only original alarm frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real_multiframe_foreground_20261117 import (
    METHODS, confirm_foreground, direct_links, observations,
)
from real_multiframe_mixed_20261117 import read_bundle, restrict, sees_target
from real_multiframe_motion_20261117 import compare
from real_multiframe_motion_transfer_20261117 import score_event
from real_multiframe_person_20261117 import PERSON_HISTORY_S, dilate_person, infer_person
from real_multiframe_pilot_20261117 import corridor_bands, follows, save
from real_multiframe_tum_20261117 import FR3_RGB, tum_frames


FRAME_ARRAYS = ("time", "depth_timestamp", "rgb_timestamp", "depth_png", "bgr", "locations", "rotations")


def load_native(path, sparse, sparse_masks, start, end, model, radius):
    with np.load(path, allow_pickle=False) as source:
        anchors = source["anchor_native_indices"]
        data = {}
        for key in FRAME_ARRAYS:
            native = source[key]
            if not np.array_equal(native[anchors], sparse[key]):
                raise ValueError(f"Original alarm frame data changed for {key}")
            data[key] = native[anchors[start]:anchors[end] + 1]
    maps = anchors[start:end+1] - anchors[start]
    if len(np.unique(data["rgb_timestamp"])) != len(data["time"]):
        raise ValueError("A native track interval repeated a camera exposure")
    intermediate = np.ones(len(data["time"]), dtype=bool)
    intermediate[maps] = False
    masks = np.empty(data["bgr"].shape[:3], dtype=bool)
    masks[maps] = sparse_masks[start:end+1]
    if intermediate.any():
        masks[intermediate] = dilate_person(infer_person(data["bgr"][intermediate], model), radius)
    return data, masks, maps


def bridge_paths(native_links, start, end):
    """Compose complete one-to-one paths across native frames inside one alarm interval."""
    paths = {source: ([source, target], matched, ready)
             for source, target, matched, _, ready in native_links.get(start + 1, ())}
    for i in range(start + 2, end + 1):
        step = {source: (target, matched, ready) for source, target, matched, _, ready in native_links.get(i, ())}
        advanced = {}
        for original, (groups, support, time) in paths.items():
            if groups[-1] in step:
                group, matched, observed = step[groups[-1]]
                advanced[original] = ([*groups, group], min(support, matched), max(time, observed))
        paths = advanced
    return paths


def extend_links(frames, views, intrinsics, masks, data, model, radius, case, direct):
    first, last = case["native_anchor_range"]
    raw, pixel_masks, anchors = load_native(case["native"], data, masks, first, last, model, radius)
    native_frames = list(tum_frames(raw, pixel_masks, intrinsics=intrinsics, stride=radius))
    seen = list(observations(raw, native_frames, pixel_masks, radius))
    for original, i in zip(views[first:last + 1], anchors, strict=True):
        if not np.array_equal(original.labels, seen[i].labels):
            raise AssertionError("Native associations changed the component identities on an alarm frame")
    steps = {i: direct_links(seen[i - 1], seen[i], intrinsics)
             for i in range(1, len(seen)) if follows(native_frames[i - 1], native_frames[i])}
    links = {i: list(edges) for i, edges in direct.items()}
    details = []
    for right, (start, end) in enumerate(zip(anchors[:-1], anchors[1:], strict=True), start=first + 1):
        if end == start + 1 or not follows(frames[right - 1], frames[right]):
            continue
        used_source = {link[0] for link in links[right]}
        used_target = {link[1] for link in links[right]}
        for source, (groups, support, ready) in bridge_paths(steps, start, end).items():
            target = groups[-1]
            accepted = source not in used_source and target not in used_target
            if accepted:
                # Native links carry identity only, without introducing a 30 Hz motion witness.
                links[right].append((source, target, support, 0, ready))
            details.append(dict(start_anchor=right - 1, end_anchor=right, components=groups,
                                accepted=accepted, support=support, available_at_s=ready))
    return links, details, dict(native_frames=len(seen), extra_frames=len(seen)-len(anchors),
                                max_gap_s=float(np.diff(raw["time"]).max()))


def verify_previous(signals, case):
    earlier = json.loads(Path(case["previous"]).read_text(encoding="utf-8"))
    if "previous_key" in case:
        earlier = earlier[case["previous_key"]]
    for kind, output in earlier["signals"].items():
        if {str(key): value for key, value in signals[kind].items()} != output:
            raise AssertionError(f"The original 5 Hz foreground replay changed for {kind}")


def support_by_component(frames, views, actor, marked, source, query):
    past, at = frames[source], frames[query]
    _, body = corridor_bands(past.cloud, at.origin, at.plane)
    samples = views[source].sampled_labels
    body &= actor[source]
    shortened = marked[source].person_motion_at <= at.time
    if at.time - past.time <= PERSON_HISTORY_S + 1e-8:
        shortened[:] = False
    results = []
    for group in np.unique(samples[body]):
        mask = body & (samples == group)
        results.append(dict(component=int(group), points=int(mask.sum()), expired=int((mask & shortened).sum()),
                            remaining_sr=float(past.weights[mask & ~shortened].sum())))
    return results


def measure(case, model):
    folder = Path(case["source"])
    cache = Path(case["cached_masks"]) if "cached_masks" in case else None
    radius, _, frames, human, static, masks = read_bundle(folder, model, cache, include_pixel_masks=True)
    with np.load(folder / "frames.npz", allow_pickle=False) as data:
        intrinsics = data["intrinsics"] if "intrinsics" in data else FR3_RGB
        views = list(observations(data, frames, masks, radius))
        direct = {i: direct_links(views[i - 1], views[i], intrinsics)
                  for i in range(1, len(frames)) if follows(frames[i - 1], frames[i])}
        links, paths, cadence = direct, [], None
        if "native" in case:
            links, paths, cadence = extend_links(frames, views, intrinsics, masks, data, model, radius, case, direct)
    sparse, old_trace = confirm_foreground(frames, views, intrinsics, links_by_end=direct)
    dense, new_trace = confirm_foreground(frames, views, intrinsics, links_by_end=links)
    kinds = {"full": (frames, sparse, dense)}
    for name, mask in (("people", human), ("obstacle", static)):
        kinds[name] = tuple(restrict(arm, mask) for arm in (frames, sparse, dense))
    before = {label: compare(base, old) for label, (base, old, _) in kinds.items()}
    after = {label: compare(base, new) for label, (base, _, new) in kinds.items()}
    verify_previous(before, case)
    for label in before:
        for row, prior in before[label].items():
            if any(prior[k][:3] != after[label][row][k][:3] for k in ("raw", "alarm")):
                raise AssertionError("An original 5 Hz alarm arm changed after adding association frames")
    stop = case.get("exit", len(frames) // 2) + 1
    prefix, _ = confirm_foreground(frames[:stop], views[:stop], intrinsics, links_by_end=links)
    partial = compare(frames[:stop], prefix)
    if any(item != after["full"][row] for row, item in partial.items()):
        raise AssertionError("Association observations after the cutoff changed its replay")

    result = dict(alarm_frames=len(frames), cadence=cadence, paths=paths, before=before, after=after,
                  sparse_links=old_trace, supplemented_links=new_trace)
    if "exit" in case:
        events = json.loads((folder / "episodes.json").read_text(encoding="utf-8"))
        person = [sees_target(frame) for frame in kinds["people"][0]]
        targets = [sees_target(frame) for frame in kinds["obstacle"][0]]
        result["scores"] = {
            key: [score_event(event, frames, sig, person, targets, methods=METHODS) for event in events]
            for key, sig in (("before", before), ("after", after))
        }
        if "probe_source" in case:
            result["probe"] = {key: support_by_component(frames, views, human, arm,
                                                          case["probe_source"], case["exit"])
                               for key, arm in (("before", sparse), ("after", dense))}
    else:
        result["scores"] = {key: {name: signal[case["score_row"]]["alarm"][3] for name, signal in seq.items()}
                            for key, seq in (("before", before), ("after", after))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    results = {}
    for name, case in json.loads(args.manifest.read_text(encoding="utf-8")).items():
        results[name] = measure(case, args.snapshot)
        print(name, json.dumps(results[name]["scores"], ensure_ascii=False), flush=True)
    save(args.output, results)


if __name__ == "__main__":
    main()
