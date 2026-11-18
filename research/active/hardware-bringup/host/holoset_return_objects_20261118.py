"""Trace the two HoloSet return alarms to annotated surfaces in the original exposures."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import holoset_return_time_20261118 as replay


ROOT = replay.WORK / "holoset-return-20261118"
OUT = ROOT / "objects"
MAX_M = replay.A.MAX_MM * .001


def input_with_depth(frame, depth):
    v, u = np.indices(depth.shape, dtype=np.float32) * replay.original.STEP
    cx, cy = frame["center"]
    ray = np.stack([(u-cx) / replay.original.FOCAL, (v-cy) / replay.original.FOCAL, np.ones_like(v)], axis=-1)
    ok = (depth > 0) & (depth < MAX_M)
    pose = frame["pose"]
    points = ((ray[ok] * depth[ok, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return {**frame, "depth": depth, "points": points}


def coordinates(points, ref_pose, floor):
    forward = ref_pose[:3, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    delta = points - ref_pose[:3, 3]
    lateral = delta @ np.cross(forward, [0., 0., 1.])
    longitudinal = delta @ forward
    height = points[:, 2] - floor
    grid = ((abs(lateral) <= replay.V.X_HALF) & (longitudinal >= 0)
            & (longitudinal < replay.V.NF*replay.V.CELL) & (height >= 0) & (height < replay.V.NH*replay.V.CELL))
    body = ((abs(lateral) <= .25) & (longitudinal >= .30) & (longitudinal <= 2) & (height >= .60) & (height <= 1.90))
    return grid, body, lateral, longitudinal, height


def spatial_stats(frame, labels, target, floor):
    ok = (frame["depth"] > 0) & (frame["depth"] < MAX_M)
    pred = coordinates(frame["points"], target, floor)
    ref = coordinates(input_with_depth(frame, np.where(ok, frame["ref"], 0))["points"], target, floor)
    measured_good = (frame["ref"][ok] > 0) & (frame["ref"][ok] < MAX_M)
    paired = labels[ok][measured_good]
    result = {}
    for label in np.unique(paired):
        selected = paired == label
        count = selected & pred[0][measured_good]
        entry = dict(paired_rays=int(selected.sum()), predicted_grid=int(count.sum()),
                     predicted_body=int((selected & pred[1][measured_good]).sum()),
                     measured_grid=int((selected & ref[0]).sum()), measured_body=int((selected & ref[1]).sum()))
        if count.any():
            entry["depth_medians_on_predicted_grid"] = dict(
                mono=float(np.median(frame["depth"][ok][measured_good][count])),
                reference=float(np.median(frame["ref"][ok][measured_good][count])),
                reference_abs_lateral_m=float(np.median(abs(ref[2][count]))),
                reference_forward_m=float(np.median(ref[3][count])),
                reference_height_m=float(np.median(ref[4][count])))
        result[int(label)] = entry
    return result


def run_case(case, rows, cap, net, groups, annotation_dir="labels"):
    lookup = {r["depth_ts"]:r for r in rows}
    inputs = [lookup[r["ts"]] for r in case["arms"]["outer"]["frames"]]
    frames, labels, reference = [], [], []
    edge_regions = {}
    for r in inputs:
        raw, clean = replay.remove_tube(cap, r)
        rt = r["rgb_ts"]
        tag = cv2.imread(str(ROOT / "scene" / annotation_dir / f"{rt}.png"), cv2.IMREAD_UNCHANGED)
        if tag is None or tag.shape != (cap.height, cap.width) or tag.dtype != np.uint8:
            raise ValueError(f"Invalid labels for {rt}")
        if annotation_dir == "labels":
            combined = np.isin(tag, [14, 15, 17]).astype(np.uint8)
            for radius in (4, 8):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
                inside = cv2.erode(combined, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1)
                for part, native in (("interior", inside), ("rim", combined-inside)):
                    grid = cv2.remap(native, *raw["maps"], cv2.INTER_NEAREST)
                    edge_regions.setdefault(f"combined_{part}_{radius}px", []).append(grid != 0)
        tag = cv2.remap(tag, *raw["maps"], cv2.INTER_NEAREST)
        tube = cv2.imread(str(replay.TUBE / "masks/outer" / replay.SEQ / f"{rt}.png"), cv2.IMREAD_GRAYSCALE)
        tube_mask = cv2.remap(tube, *raw["maps"], cv2.INTER_NEAREST) > 0
        if annotation_dir == "labels" and not np.array_equal(tag == 1, tube_mask):
            raise ValueError("The original tube mask differs from the semantic map")
        tag[tube_mask] = 1
        labels.append(tag)
        frames.append(clean)
        reference.append(input_with_depth(clean, np.where(tag == 1, 0, raw["ref"])))
    floor = inputs[-1]["floor_y"]
    def infer(history):
        return replay.score(net, replay.original.features(history, floor, replay.V))
    original = infer(frames)
    if abs(original - case["arms"]["outer"]["scores"]["all"]) > .0001:
        raise ValueError("Cannot reproduce tube-removed score")
    if not np.array_equal(replay.original.features(frames, floor, replay.V),
                          replay.original.features([input_with_depth(f, f["depth"]) for f in frames], floor, replay.V)):
        raise ValueError("No-op depth replacement changed a model input")
    scores = {}
    selections = {name:[np.isin(tag, ids) for tag in labels] for name, ids in groups.items()}
    selections.update(edge_regions)
    for name, selection in selections.items():
        unknown, keep, corrected, restored = [], [], [], []
        for f, ref, region in zip(frames, reference, selection):
            z, measured = f["depth"], ref["depth"]
            unknown.append(input_with_depth(f, np.where(region, 0, z)))
            keep.append(input_with_depth(f, np.where(region, z, 0)))
            corrected.append(input_with_depth(f, np.where(region, measured, z)))
            restored.append(input_with_depth(f, np.where(region, z, measured)))
        scores[name] = dict(ids=groups.get(name, [14, 15, 17]), drop=infer(unknown), keep=infer(keep),
                            correct=infer(corrected), restore_to_reference=infer(restored),
                            drop_current=infer([*frames[:-1], unknown[-1]]),
                            drop_past=infer([*unknown[:-1], frames[-1]]),
                            correct_past=infer([*corrected[:-1], frames[-1]]))
    stats = {r["rgb_ts"]:spatial_stats(f, label, frames[-1]["pose"], floor)
             for r, f, label in zip(inputs, frames, labels)}
    return dict(case=case["case"], baseline=original, reference=infer(reference), groups=scores,
                measured_stats_by_rgb=stats)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    archive = json.loads((replay.TUBE / "d8/attribution.json").read_bytes())
    cases = [r for r in archive["results"] if r["case"] in (7, 8)]
    net = replay.I.D.make_head(8)
    weights = replay.WORK / "ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    if replay.original.file_digest(weights) != archive["model_sha256"]:
        raise ValueError("Changed D8 weights")
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    cap = replay.original.Recording(replay.HEAD / replay.SEQ)
    rows = replay.original.records(cap.directory / "geometry.jsonl")
    manifest = ROOT / "scene/manifest.json"
    classes = json.loads(manifest.read_bytes())["classes"]
    groups = {f"id_{i}":[int(i)] for i in classes if int(i) != 1}
    groups.update(right_wall_band=[8,15], equipment_stack=[10,11,16], right_hardware=[10,16],
                  band_hardware=[10,15,16], right_wall_equipment_band=[8,10,11,15,16],
                  floor_baseboards=[6,7,9], left_walls=[2,5], band_and_area_above=[15,17],
                  right_wall_band_and_above=[8,15,17], distant_corridor_and_above=[14,17],
                  distant_corridor_band_above=[14,15,17],
                  right_wall_distant_band_above=[8,14,15,17])
    core_groups = {k:groups[k] for k in ("id_8","id_14","id_15","id_17","band_and_area_above",
                                         "right_wall_band_and_above","distant_corridor_band_above")}
    result = dict(classes=classes, model_sha256=archive["model_sha256"],
                  annotation_sha256=replay.original.file_digest(manifest),
                  cases=[run_case(c, rows, cap, net, groups) for c in cases],
                  core_cases=[run_case(c, rows, cap, net, core_groups, "cores") for c in cases])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic.json").write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
