"""Replay temporal subsets of the two HoloSet return alarms with all tube evidence removed."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import real_wall_holoset_attribution_20261118 as replay
import real_wall_holoset_score_20261118 as original
from spires_insert_dev_20261117 import A, I, V


WORK = Path("artifacts.local/work")
HEAD = WORK / "real-wall-search-20261118/head"
SCORES = WORK / "real-wall-search-20261118/holoset-d8/remote/out"
TUBE = WORK / "holoset-tube-20261118"
OUTPUT = WORK / "holoset-return-20261118/time"
SEQ = "research-center-seq4"
CHANNELS = ["current_hits", "history_hits", "current_free", "history_free", "current_visible", "history_visible"]


def remove_tube(cap, row):
    path = SCORES / SEQ / "predicted" / f"{row['depth_ts']}.npy"
    frame = replay.cached_frame(cap, row, path, A.MAX_MM * .001)
    outer = cv2.imread(str(TUBE / "masks/outer" / SEQ / f"{row['rgb_ts']}.png"), cv2.IMREAD_GRAYSCALE)
    if outer is None or outer.shape != (cap.height, cap.width):
        raise ValueError("Missing tube outline")
    mask = cv2.remap(outer, *frame["maps"], cv2.INTER_NEAREST) > 0
    valid = (frame["depth"] > 0) & (frame["depth"] < A.MAX_MM * .001)
    clean = {**frame, "depth": np.where(mask, 0., frame["depth"]), "points": frame["points"][~mask[valid]]}
    return frame, clean


def score(net, features):
    with torch.inference_mode():
        x = torch.from_numpy(features[None]).float()
        return float(net(V.pad(x)).sigmoid().item())


def subset(history, bits, floor):
    kept = [frame for i, frame in enumerate(history) if bits & (1 << i)]
    if not bits & (1 << (len(history)-1)):
        target = history[-1]
        kept.append({**target, "depth": np.zeros_like(target["depth"]), "points": target["points"][:0]})
    return original.features(kept, floor, V)


def branch_subsets(net, features):
    values = []
    for bits in range(1 << len(CHANNELS)):
        changed = features.copy()
        for i in range(len(CHANNELS)):
            if not bits & (1 << i):
                changed[i] = 0
        values.append(score(net, changed))
    return values


def split_exposure_effects(net, baseline, removed):
    """Restore the original context and change only the channels affected by omission."""
    arms = {"occupancy": [0, 1], "free": [2, 3], "visible": [4, 5],
            "free_visible": [2, 3, 4, 5], "all": list(range(6))}
    scores = {}
    for name, ids in arms.items():
        x = baseline.copy()
        x[ids] = removed[ids]
        scores[name] = score(net, x)
    return scores


def score_case(cap, case, rows, net):
    target = case["depth_ts"]
    support = [r for r in rows if 0 <= target-r["depth_ts"] <= original.WINDOW]
    expected = case["arms"]["outer"]["frames"]
    if [r["depth_ts"] for r in support] != [r["ts"] for r in expected]:
        raise ValueError("History differs from the frozen baseline")
    frames = [remove_tube(cap, r) for r in support]
    raw, clean = zip(*frames)
    floor = support[-1]["floor_y"]
    x = original.features(clean, floor, V)
    clean_score = score(net, x)
    raw_score = score(net, original.features(raw, floor, V))
    if abs(clean_score - case["arms"]["outer"]["scores"]["all"]) > .0001 or abs(raw_score - case["archived"]) > .0001:
        raise ValueError("Score reproduction failed")
    total = (1 << len(clean))-1
    current = 1 << (len(clean)-1)
    history = current-1
    priority = {0, total, current, history}
    for i in range(len(clean)):
        priority.update([1 << i, total ^ (1 << i)])
    values = [None] * (total+1)
    for bits in sorted(priority):
        values[bits] = score(net, subset(clean, bits, floor))
    if values[total] != clean_score:
        raise ValueError("Full subset differs from complete tube-removed input")
    print("case", case["case"], "all/current/history/blank", [values[b] for b in [total, current, history, 0]],
          "single", [values[1<<i] for i in range(len(clean))],
          "leave-one-out", [values[total ^ (1<<i)] for i in range(len(clean))], flush=True)
    branches = branch_subsets(net, x)
    for bits in range(total+1):
        if values[bits] is None:
            values[bits] = score(net, subset(clean, bits, floor))
    if branches[21] != values[current] or branches[42] != values[history] or branches[0] != values[0]:
        raise ValueError("Feature- and exposure-level time roles disagree")
    omitted_groups = [[i] for i in range(len(clean))]
    omitted_groups.append([len(clean)-3, len(clean)-2])
    split = []
    for omitted in omitted_groups:
        keep = total ^ sum(1 << i for i in omitted)
        scores = split_exposure_effects(net, x, subset(clean, keep, floor))
        if scores["all"] != values[keep]:
            raise ValueError("Restored channel split must reproduce exposure ablation")
        split.append(dict(dropped_indices=omitted, scores=scores))
    return dict(case=case["case"], window=[dict(index=i, depth_ts=r["depth_ts"], rgb_ts=r["rgb_ts"],
                                                depth_offset_s=(r["depth_ts"]-target)/1e7,
                                                rgb_offset_s=(r["rgb_ts"]-support[-1]["rgb_ts"])/1e7)
                                           for i,r in enumerate(support)], raw_score=raw_score, tube_removed=clean_score,
                only_current=values[current], only_history=values[history], no_evidence=values[0],
                latest_two_exposures=values[current | (current >> 1)],
                temporal_scores_by_retained_bits=values, branch_scores_by_retained_bits=branches,
                dropped_exposure_branch_effects=split)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    archive_path = TUBE / "d8/attribution.json"
    archive = json.loads(archive_path.read_bytes())
    weights = WORK / "ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    if original.file_digest(weights) != archive["model_sha256"] or original.file_digest(TUBE / "masks/manifest.json") != archive["mask_manifest_sha256"]:
        raise ValueError("Masked model baseline identity changed")
    net = I.D.make_head(8)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    cap = original.Recording(HEAD / SEQ)
    rows = [r for r in original.records(cap.directory / "geometry.jsonl") if
            r["tag"] not in ("no_depth_pose", "no_rgb_match") and
            original.rigid(cap.rig[r["depth_ts"]]) and original.rigid(cap.pv[r["rgb_ts"]][2])]
    rows.sort(key=lambda r: r["depth_ts"])
    selected = [c for c in archive["results"] if c["seq"] == SEQ]
    if [c["case"] for c in selected] != [7,8]:
        raise ValueError("Unexpected return alarm timestamps")
    results = [score_case(cap, c, rows, net) for c in selected]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = dict(threshold=archive["threshold"], branch_bit_order=CHANNELS,
               time_bit_order="oldest exposure at bit 0, target at highest bit",
               source_attribution_sha256=original.file_digest(archive_path),
               model_sha256=archive["model_sha256"], results=results)
    (OUTPUT / "temporal.json").write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
