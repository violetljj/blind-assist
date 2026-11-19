"""Softly weight historical occupancy on the originally frozen HoloSet cafeteria controls."""
import json

import cv2
import numpy as np
import torch

import holoset_contradiction_static_20261119 as controls
from holoset_rgb_contradiction_20261119 import holoset_views
from holoset_rgb_exclude_near_20261119 import RGBNearExcluder
from holoset_soft_history_20261119 import weighted_scores


d8 = controls.d8
OUT = d8.WORK / "holoset-soft-history-20261119/furniture.json"


def checked_furniture_at_onset(history, masks, cap, truth):
    """Independently register rays to the existing tabletop and chair polygons with registered range."""
    cur = history[-1]
    outline = np.zeros((cap.height, cap.width), np.uint8)
    for polygon in truth["polygons"]:
        cv2.fillPoly(outline, [np.int32(polygon)], 1)
    roi = cv2.remap(outline, *cur["maps"], cv2.INTER_NEAREST) > 0
    height, width = roi.shape
    tested, tagged = [], []
    for i, frame in enumerate(history):
        local = (frame["measured"] - cur["pose"][:3, 3]) @ cur["pose"][:3, :3]
        z = local[:, 2]
        uv = d8.original.FOCAL * local[:, :2] / np.maximum(z[:, None], .05) + cur["center"]
        u, v = np.rint(uv / d8.original.STEP).astype(int).T
        visible = (z>.05) & (u>=0) & (u<width) & (v>=0) & (v<height)
        u, v = u.clip(0, width-1), v.clip(0, height-1)
        match = visible & roi[v, u] & (np.abs(z - cur["ref"][v, u]) <= truth["range_tolerance_m"])
        tested.append(int(match.sum()))
        if i < len(masks):
            valid = (frame["depth"] > 0) & (frame["depth"] < 8.)
            tagged.append(int(np.count_nonzero(match & masks[i][valid])))
    if tested != [r["rays"] for r in truth["verified_rays"]]:
        raise ValueError("Surveyed furniture rays differ from the original control")
    return dict(verified_by_exposure=tested, marked_history_by_exposure=tagged,
                original_remove_verified_score=truth["erased"], original_only_verified_score=truth["only"])


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    archive_path = controls.OUT / "near-exclusion/result.json"
    archive = json.loads(archive_path.read_bytes())
    seq = controls.CONTROLS[0][0]
    prior = next(c for c in archive["controls"] if c["seq"] == seq)
    if len(prior["rows"]) != 10:
        raise ValueError("The original furniture window changed")
    weights = d8.WORK / "ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    if d8.original.file_digest(weights) != archive["weights_sha256"]:
        raise ValueError("D8 weights changed")
    net = d8.I.D.make_head(8)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    cap = d8.original.Recording(d8.HEAD / seq)
    verifier = RGBNearExcluder(holoset_views(cap, controls.OUT / "extra" / seq))
    geometry = [r for r in d8.original.records(cap.directory / "geometry.jsonl")
                if r["tag"] not in ("no_depth_pose", "no_rgb_match")
                and d8.original.rigid(cap.rig[r["depth_ts"]]) and d8.original.rigid(cap.pv[r["rgb_ts"]][2])]
    cache, output = {}, []
    furniture = None
    for row in prior["rows"]:
        ts = row["dt"]
        support = [r for r in geometry if ts-d8.original.WINDOW <= r["depth_ts"] <= ts]
        if ([support[-1]["depth_ts"], support[-1]["rgb_ts"], len(support)-1]
                != [ts, row["rgb_ts"], row["historical_exposures"]]):
            raise ValueError("Target exposure or history differs")
        history, masks = [], []
        validity_count = 0
        for source in support:
            dt, rt = source["depth_ts"], source["rgb_ts"]
            if dt not in cache:
                path = d8.SCORES / seq / "predicted" / f"{dt}.npy"
                cache[dt] = (d8.replay.cached_frame(cap, source, path, 8.), np.load(path))
            frame, mono = cache[dt]
            history.append(frame)
            if dt != ts:
                cutoff = min(ts, row["rgb_ts"])
                mask = (verifier.classify(rt, cutoff, frame["maps"], mono)[0] if rt < cutoff
                        else np.zeros(frame["depth"].shape, bool))
                masks.append(mask)
                validity_count += int((mask & (frame["depth"] > 0) & (frame["depth"] < 8.)).sum())
        scores, grid_count = weighted_scores(history, masks, support[-1]["floor_y"], net)
        baseline_key = next(k for k in scores if float(k) == 1.)
        if ({float(k) for k in scores} != {1., .9, .75, .5, .25} or abs(scores[baseline_key] - row["baseline"])>1e-4
                or (validity_count, grid_count) != (row["counts"]["marked"], row["counts"]["marked_pred_grid"])):
            raise ValueError("Weights, baseline or geometry mismatch")
        output.append(dict(depth_ts=ts, rgb_ts=row["rgb_ts"], time_s=row["time_s"], scores=scores,
                           marked=validity_count, marked_grid=int(grid_count)))
        if row["baseline"] > archive["threshold"]:
            if furniture is not None:
                raise ValueError("Archive has an unexpected second alert")
            furniture = checked_furniture_at_onset(history, masks, cap, prior["furniture_source"])
    summary = {}
    for key in output[0]["scores"]:
        alert = [r for r in output if r["scores"][key] > archive["threshold"]]
        summary[key] = dict(alerts=len(alert), first_depth_ts=alert[0]["depth_ts"] if alert else None,
                            first_s=alert[0]["time_s"] if alert else None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(seq=seq, source_sha256=d8.original.file_digest(archive_path),
                                  threshold=archive["threshold"], timing=summary, rows=output,
                                  furniture=furniture, skipped=prior["skipped"]), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
