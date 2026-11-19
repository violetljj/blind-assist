"""Downweight flagged historical endpoint votes in D8; all rays remain observed."""
import json

import cv2
import numpy as np
import torch

import holoset_return_time_20261118 as d8
import holoset_contradiction_replay_20261119 as previous
from holoset_return_objects_20261118 import coordinates


WEIGHTS = (1., .90, .75, .50, .25)
FROZEN = previous.OUT.parent / "near-exclusion/replay"
OUT = d8.WORK / "holoset-soft-history-20261119"


def weighted_scores(history, masks, floor, net):
    """Return D8 scores and the number of marked historical endpoints in the grid."""
    if not history or len(masks) != len(history)-1:
        raise ValueError("Expected one flag map per past exposure")
    V = d8.V
    target = history[-1]
    base = d8.original.features(history, floor, V)
    total = np.zeros((V.NH,V.NF), np.int64)
    marked = np.zeros_like(total)
    for frame, mask in zip(history[:-1], masks):
        depth = frame["depth"]
        if mask.dtype != np.bool_ or mask.shape != depth.shape:
            raise ValueError("Invalid flag map")
        valid = (depth>0) & (depth<previous.MAX_M)
        grid, _, _, distance, height = coordinates(frame["points"], target["pose"], floor)
        cells = (height[grid]//V.CELL).astype(int)*V.NF + (distance[grid]//V.CELL).astype(int)
        flagged = mask[valid][grid]
        total += np.bincount(cells,minlength=total.size).reshape(total.shape)
        marked += np.bincount(cells[flagged],minlength=total.size).reshape(total.shape)
    if not np.array_equal(np.log1p(total).astype(base.dtype),base[1]):
        raise ValueError("Unweighted historical votes do not reproduce the frozen grid")
    scores = {}
    for weight in WEIGHTS:
        features = base.copy()
        votes = total - marked + weight*marked
        if np.any((votes>0)!=(total>0)):
            raise ValueError("Occupied cell lost or introduced")
        features[1] = np.log1p(votes)
        scores[f"{weight:.2f}"] = d8.score(net,features)
    return scores,int(marked.sum())


def replay_sequence(seq, frozen, net):
    cap = d8.original.Recording(d8.HEAD/seq)
    geo = d8.original.records(cap.directory/"geometry.jsonl")
    rows = {r["depth_ts"]:r for r in geo}
    cache = {}
    results = []
    for target in frozen["reports"]:
        dt, rt = target["depth_ts"],target["rgb_ts"]
        exposure_times = target["source_rgb_ts"]
        sources = [r for r in geo if dt-d8.original.WINDOW<=r["depth_ts"]<=dt
                   and r.get("rgb_ts") in exposure_times and r["tag"] not in ("no_depth_pose","no_rgb_match")]
        sources.sort(key=lambda r:r["depth_ts"])
        if [r["rgb_ts"] for r in sources]!=exposure_times or sources[-1]["depth_ts"]!=dt:
            raise ValueError("Historical source clocks differ")
        history = []
        for row in sources:
            ts = row["depth_ts"]
            if ts not in cache:
                raw = d8.replay.cached_frame(cap,row,d8.SCORES/seq/"predicted"/f"{ts}.npy",previous.MAX_M)
                cache[ts] = {**raw,"rgb_ts":row["rgb_ts"]}
            history.append(cache[ts])
        floor = rows[dt-round(target["floor_age_s"]*1e7)]["floor_y"]
        with np.load(FROZEN/"masks"/seq/f"{rt}.npz") as archived:
            if set(archived.files)!={str(t) for t in exposure_times[:-1]}:
                raise ValueError("Wrong historical mask identities")
            masks = [archived[str(t)] for t in exposure_times[:-1]]
        scores, n = weighted_scores(history,masks,floor,net)
        if abs(scores["1.00"]-target["baseline"])>1e-5 or n!=sum(c["grid"] for c in target["changes"]):
            raise ValueError(f"Baseline or marked endpoint count mismatch for {seq} {rt}")
        row = dict(depth_ts=dt,rgb_ts=rt,offset_s=target["offset_s"],floor_age_s=target["floor_age_s"],
                   scores=scores,marked_grid=n)
        if seq==d8.SEQ and target.get("case") in (7,8):
            # Reproduce the previous attribution condition separately from the normal scores.
            no_tube = [previous.remove(f,previous.tube_mask(cap,f)) for f in history]
            reduced,marked = weighted_scores(no_tube,masks,floor,net)
            if abs(reduced["1.00"]-target["tube_removed"])>1e-5:
                raise ValueError("Tube-excluded baseline mismatch")
            row.update(case=target["case"],tube_excluded=reduced,tube_excluded_marked_grid=marked)
        results.append(row)
    return results


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    src = FROZEN/"result.json"
    old = json.loads(src.read_bytes())
    weights = d8.WORK/"ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    if d8.original.file_digest(weights)!=old["model_sha256"] or old["threshold"]!=d8.I.THETA:
        raise ValueError("Frozen model identity or threshold differs")
    net = d8.I.D.make_head(8)
    net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    net.eval()
    result = dict(source_sha256=d8.original.file_digest(src),model_sha256=old["model_sha256"],
                  threshold=d8.I.THETA,weight_grid=WEIGHTS,
                  sequences={s:replay_sequence(s,v,net) for s,v in old["sequences"].items()})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"passages.json").write_text(json.dumps(result,indent=1,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
