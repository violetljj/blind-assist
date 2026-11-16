"""EXPLORE replay with shorter history for points projected into a person mask."""

import argparse
from collections import defaultdict
from itertools import takewhile
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from real_multiframe_pilot_20261117 import (
    ARMS, HISTORY_S, MAX_GAP_S, MIN_SOLID_ANGLE, corridor_bands, lamar_frames, replay, save,
)
from real_multiframe_tum_20261117 import contact_sheet, score, tum_frames


PERSON_HISTORY_S = .25
PERSON_ARMS = (*ARMS, "person_history")


@torch.inference_mode()
def infer_person(images, snapshot):
    """Accept OpenCV BGR images in native depth-aligned or camera pixels."""
    torch.set_num_threads(4)
    processor = SegformerImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(snapshot, local_files_only=True).eval()
    person_class, = (i for i, label in model.config.id2label.items() if label == "person")
    masks = []
    for start in range(0, len(images), 4):
        batch = [image[:, :, ::-1].copy() for image in images[start:start + 4]]
        inputs = processor(batch, return_tensors="pt")
        logits = model(**inputs).logits
        for image, logit in zip(batch, logits, strict=True):
            scores = interpolate(logit[None], size=image.shape[:2], mode="bilinear", align_corners=False)
            masks.append((scores[0].argmax(dim=0) == person_class).numpy())
    return np.stack(masks)


def check_legacy(predictions, old):
    for row, item in predictions.items():
        prior = old[str(row)]
        if (item["time"] != prior["time"] or item["history_frames"] != prior["history_frames"]
                or any(item[key][:len(ARMS)] != prior[key] for key in ("raw", "alarm"))):
            raise AssertionError(f"Original replay changed at row {row}")
        single, memory, _, candidate = np.array(item["alarm"])
        if np.any(single & ~candidate) or np.any(candidate & ~memory):
            raise AssertionError(f"Confirmed person history outside single/memory at row {row}")


def history_support(frames, query_rows, visible=None):
    """Trace how much each eligible past frame contributes to each height band."""
    support = []
    for index, frame in enumerate(frames):
        if frame.row not in query_rows:
            continue
        newer = frame
        for past in reversed(frames[:index]):
            age = frame.time - past.time
            if (age > HISTORY_S + 1e-8 or past.component != newer.component or past.row + 1 != newer.row
                    or not 0 < newer.time - past.time <= MAX_GAP_S):
                break
            newer = past
            in_view = visible[past.row] if visible is not None else np.ones(len(past.cloud), dtype=bool)
            kept = ~past.person if age > PERSON_HISTORY_S + 1e-8 else np.ones(len(past.cloud), dtype=bool)
            for height, band in zip(("head", "body"), corridor_bands(past.cloud, frame.origin, frame.plane),
                                    strict=True):
                total = float(past.weights[band].sum())
                if total < MIN_SOLID_ANGLE:
                    continue
                retained = float(past.weights[band & kept].sum())
                support.append(dict(query_row=frame.row, source_row=past.row, band=height, age_s=age,
                                    total_sr=total, retained_sr=retained,
                                    in_view_retained_sr=float(past.weights[band & kept & in_view].sum())))
    return support


def run_tum(work, previous, snapshot):
    intervals = json.loads((previous / "tum_reference.json").read_text(encoding="utf-8-sig"))
    old = json.loads((previous / "tum_predictions.json").read_text(encoding="utf-8"))
    with np.load(work / "tum_frames.npz", allow_pickle=False) as data:
        if not np.all(data["rgb_timestamp"][:-1] <= data["depth_timestamp"][1:]):
            raise ValueError("A history mask uses RGB captured after its query frame")
        bgr, times = data["bgr"], data["time"]
        cache = work / "tum_person_masks.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as source:
                if (source["snapshot"].item() != snapshot.name
                        or not np.array_equal(source["rgb_timestamp"], data["rgb_timestamp"])):
                    raise ValueError("Cached masks do not match the model snapshot and RGB frames")
                masks = source["mask"]
        else:
            masks = infer_person(bgr, snapshot)
            with cache.open("xb") as stream:
                np.savez_compressed(stream, mask=masks, snapshot=snapshot.name, rgb_timestamp=data["rgb_timestamp"])
        frames = list(tum_frames(data, masks))

    predictions = replay(frames, keep_trace=True, person_history_s=PERSON_HISTORY_S)
    check_legacy(predictions, old)
    if len(predictions) != len(old):
        raise AssertionError("TUM coverage changed")
    unshortened = replay(frames, keep_trace=True, person_history_s=HISTORY_S)
    if any(item[key][3] != item[key][1] for item in unshortened.values() for key in ("raw", "alarm")):
        raise AssertionError("Person arm with the original duration differs from unfiltered memory")
    for interval in intervals:
        end = interval["end"]
        if interval["kind"] == "after_exit":
            prefix = replay(frames[:end + 1], keep_trace=True, person_history_s=PERSON_HISTORY_S)
            if prefix != {i: item for i, item in predictions.items() if i <= end}:
                raise AssertionError(f"TUM results depend on frames after row {end}")

    example_rows = [i for e in intervals for i in range(e["start"], e["end"] + 1)]
    for i in example_rows:
        pixels = bgr[i, masks[i]].astype(np.float32)
        bgr[i, masks[i]] = (.5 * pixels + [17, 110, 126]).astype(np.uint8)
    contact_sheet(bgr, times, work / "tum_person_overlay.jpg", example_rows)
    comparison = score(predictions, intervals, PERSON_ARMS)
    exit_rows = {i for e in intervals if e["kind"] == "after_exit" for i in range(e["start"], e["end"] + 1)}
    result = dict(person_history_s=PERSON_HISTORY_S, model_snapshot=snapshot.name,
                  legacy_matches=len(predictions), no_shortening_matches=len(unshortened),
                  frames_with_person_mask=int(np.count_nonzero(masks.any(axis=(1, 2)))),
                  samples_marked_person={str(i): int(np.count_nonzero(frames[i].person)) for i in example_rows},
                  history_support=history_support(frames, exit_rows),
                  intervals=comparison)
    save(work / "tum_person_predictions.json", predictions)
    save(work / "tum_person_comparison.json", result)
    for item in comparison:
        print({key: item[key] for key in ("start", "end")},
              {arm: (value["alarms"], value["initial_tail_s"], value["first_s"])
               for arm, value in item["counts"].items()})


def project_to_gray(cloud, camera, view):
    pose = np.array(view["T_local_from_gray"])
    points = (cloud - pose[:3, 3]) @ pose[:3, :3]
    z = points[:, 2]
    pixels = (points / np.where(z[:, None] > 0, z[:, None], 1.)) @ np.array(camera["K"]).T
    x, y = np.rint(pixels[:, :2]).astype(int).T
    in_image = (z > 0) & (x >= 0) & (x < camera["width"]) & (y >= 0) & (y < camera["height"])
    return in_image, x, y


def run_lamar(work, previous, snapshot, lamar):
    gray = work / "lamar-gray"
    manifest = json.loads((gray / "manifest.json").read_text(encoding="utf-8"))
    session = manifest["session"]
    source = lamar / "hololens" / "CAB"
    target = 12
    rows = {d["depth_row"]: d for d in manifest["matches"]}
    if set(rows) != set(range(6, target + 1)):
        raise ValueError("Wrong LaMAR history window")
    pairs = [(i, camera, view) for i, row in rows.items() for camera, view in row["gray"].items()]
    bgr = [cv2.imdecode(np.frombuffer((gray / view["relative_path"]).read_bytes(), dtype=np.uint8),
                        cv2.IMREAD_COLOR) for _, _, view in pairs]
    masks = infer_person(bgr, snapshot)
    views = defaultdict(list)
    for (i, camera, view), image, mask in zip(pairs, bgr, masks, strict=True):
        intrinsics = manifest["cameras"][camera]
        if (intrinsics["model"] != "PINHOLE" or view["T_local_from_gray"] is None
                or view["track_component"] != rows[i]["depth_track_component"]
                or not 0 <= rows[i]["depth_timestamp_us"] - view["gray_timestamp_us"] <= 65_000
                or image.shape[:2] != (intrinsics["height"], intrinsics["width"])):
            raise ValueError(f"Unsuitable gray association at LaMAR row {i}, {camera}")
        views[i].append((intrinsics, view, mask))

    visible, point_counts = {}, {}

    def mask_for_cloud(row, stamp, cloud):
        if row not in views:
            return None
        if stamp != rows[row]["depth_timestamp_us"]:
            raise ValueError(f"Gray manifest has the wrong depth timestamp, row {row}")
        person = np.zeros(len(cloud), dtype=bool)
        in_any_image = person.copy()
        for camera, view, mask in views[row]:
            in_image, x, y = project_to_gray(cloud, camera, view)
            person[in_image] |= mask[y[in_image], x[in_image]]
            in_any_image |= in_image
        visible[row] = in_any_image
        point_counts[row] = dict(points=len(cloud), in_image=int(in_any_image.sum()), person=int(person.sum()))
        return person

    native = source / "native_depth" / f"{session}.npz"
    prepared = source / "prepared" / native.name
    frames = [frame for frame in takewhile(lambda f: f.row <= target,
                                           lamar_frames(prepared, native, mask_for_cloud)) if frame.row in rows]
    if ([f.row for f in frames] != list(range(6, target + 1))
            or any(f.component != frames[0].component or f.person is None for f in frames)):
        raise AssertionError("LaMAR rescue window lost its annotated history")
    predictions = replay(frames, keep_trace=True, person_history_s=PERSON_HISTORY_S)
    old = json.loads((previous / "lamar_predictions.json").read_text(encoding="utf-8"))[session]
    # The first complete vote with a full 0.8 s horizon for its three raw votes is row 12.
    check_legacy({target: predictions[target]}, old)
    for i in range(10, target + 1):
        if predictions[i]["raw"][:len(ARMS)] != old[str(i)]["raw"]:
            raise AssertionError(f"Original LaMAR raw vote changed at row {i}")

    labels = json.loads((lamar / "screen-hololens" / "summary.json").read_text(encoding="utf-8"))
    visit, = (visit for visit in labels["visits"] if visit["visit"] == session)
    reference, = (row for row in visit["rows"] if row["depth_row"] == target)
    result = dict(session=session, target_row=target, reference_travel=reference["travel"],
                  person_history_s=PERSON_HISTORY_S, model_snapshot=snapshot.name,
                  point_counts=point_counts, history_support=history_support(frames, range(10, target + 1), visible),
                  predictions=predictions)
    save(work / "lamar_person_comparison.json", result)
    print(json.dumps({"target": predictions[target], "point_counts": point_counts,
                      "history_support": result["history_support"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("previous", type=Path, help="Original real-depth replay outputs")
    parser.add_argument("snapshot", type=Path, help="Local SegFormer B0 ADE20K snapshot")
    parser.add_argument("--lamar", type=Path, help="Replay only the original LaMAR rescue window")
    args = parser.parse_args()
    if args.lamar:
        run_lamar(args.work, args.previous, args.snapshot, args.lamar)
    else:
        run_tum(args.work, args.previous, args.snapshot)


if __name__ == "__main__":
    main()
