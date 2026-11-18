"""Replay causal RGB contradiction masks on the frozen HoloSet approach clocks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import holoset_return_time_20261118 as source
from holoset_tube_attribution_20261118 import inspect_points


OUT = source.WORK / "holoset-contradiction-20261119/replay"
TICKS = 10_000_000
ARMS = {"research-center-seq3": (-3.01, 3.01), "research-center-seq4": (-2.41, .21)}
MAX_M = source.A.MAX_MM * .001


def remove(frame, mask):
    valid = (frame["depth"] > 0) & (frame["depth"] < MAX_M)
    return {**frame, "depth": np.where(mask, 0., frame["depth"]), "points": frame["points"][~mask[valid]]}


def tube_mask(cap, frame):
    source_mask = source.TUBE / "masks/outer" / cap.directory.name / f"{frame['rgb_ts']}.png"
    mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (cap.height, cap.width):
        raise ValueError(f"Missing native tube annotation: {source_mask}")
    return cv2.remap(mask, *frame["maps"], cv2.INTER_NEAREST) > 0


def isolate_dropout(cap, history, masks, floor, score):
    marked, rest, sources = [], [], []
    for f,m in zip(history[:-1], masks):
        path = source.TUBE/"masks/outer"/cap.directory.name/f"{f['rgb_ts']}.png"
        region = tube_mask(cap, f) if path.is_file() else np.zeros_like(m)
        marked.append(remove(f,m & region))
        rest.append(remove(f,m & ~region))
        valid = (f["depth"] > 0) & (f["depth"] < MAX_M)
        pose = history[-1]["pose"]
        inside = (m & region)[valid]
        other = (m & ~region)[valid]
        sources.append(dict(rgb_ts=f["rgb_ts"],has_tube_annotation=path.is_file(),
                            marked_grid_body=inspect_points(f["points"][inside],pose,floor,source.V),
                            other_grid_body=inspect_points(f["points"][other],pose,floor,source.V),
                            marked_measured_grid_body=inspect_points(f["measured"][inside],pose,floor,source.V)))
    return dict(remove_only_annotated_tube=score([*marked,history[-1]]),
                remove_only_other=score([*rest,history[-1]]), sources=sources)


def summary(scored, primary):
    rows = [r for r in scored if not primary or r["floor_age_s"] == 0]
    result = {arm: dict(alarms=sum(r[arm] > source.I.THETA for r in rows),
                      first_depth_ts=next((r["depth_ts"] for r in rows if r[arm] > source.I.THETA), None),
                      first_offset_s=next((r["offset_s"] for r in rows if r[arm] > source.I.THETA), None),
                      samples=len(rows)) for arm in rows[0] if arm in ("baseline","filtered")}
    if "filtered" in result:
        result["lost_offsets_s"] = [r["offset_s"] for r in rows if r["baseline"] > source.I.THETA >= r["filtered"]]
        result["gained_offsets_s"] = [r["offset_s"] for r in rows if r["filtered"] > source.I.THETA >= r["baseline"]]
        first = [result[arm]["first_depth_ts"] for arm in ("baseline", "filtered")]
        result["first_delay_s"] = (first[1] - first[0]) / TICKS if all(t is not None for t in first) else None
    return result


def scalar_info(values):
    if isinstance(values, np.generic):
        return values.item()
    raise TypeError(f"Unexpected classifier diagnostics type {type(values)}")


def evaluate(seq, net, archive, extra_root, baseline_only, destination, exclude_near):
    cap = source.original.Recording(source.HEAD / seq)
    cases = [r for r in archive["results"] if r["seq"] == seq]
    by_ts = {r["depth_ts"]:r for r in cases}
    zero = cases[0]["depth_ts"]
    lo, hi = ARMS[seq]
    old = json.loads((source.SCORES / seq / "scores.json").read_bytes())
    if old["cohort_sha256"] != archive["cohort_sha256"]:
        raise ValueError("Archived scoring cohort changed")
    stored = {r["depth_ts"]:r for r in old["scores"]}
    all_rows = sorted(source.original.records(cap.directory/"geometry.jsonl"), key=lambda r:r["depth_ts"])
    bad = set(old["invalid_history_pose_ids"])
    pool = {r["depth_ts"]:r for r in all_rows if r["tag"] not in ("no_depth_pose","no_rgb_match")
            and r["depth_ts"] not in bad}
    cache = {}
    def frame(r):
        ts = r["depth_ts"]
        if ts not in cache:
            p = source.SCORES / seq / "predicted" / f"{ts}.npy"
            f = source.replay.cached_frame(cap, r, p, MAX_M)
            cache[ts] = {**f, "rgb_ts":r["rgb_ts"], "raw_mono":np.load(p)}
        return cache[ts]
    if not baseline_only:
        from holoset_rgb_contradiction_20261119 import RGBVerifier, holoset_views
        views = holoset_views(cap, extra_root / seq)
        if exclude_near:
            from holoset_rgb_exclude_near_20261119 import RGBNearExcluder
            verifier = RGBNearExcluder(views)
        else:
            verifier = RGBVerifier(views)
    reports, excluded = [], []
    for row in all_rows:
        dt = row["depth_ts"]
        age = (dt-zero) / TICKS
        if not lo <= age <= hi:
            continue
        if dt not in pool:
            excluded.append(dict(depth_ts=dt,offset_s=age,reason=row["tag"] if dt not in bad else "invalid_pose"))
            continue
        previous = [r for r in pool.values() if 0 <= dt-r["depth_ts"] <= 6_500_000 and r.get("floor_y") is not None]
        if not previous:
            excluded.append(dict(depth_ts=dt,offset_s=age,reason="no_current_or_recent_floor"))
            continue
        floor = previous[-1]["floor_y"]
        support = [r for t,r in pool.items() if 0 <= dt-t <= source.original.WINDOW]
        missing = [r["depth_ts"] for r in support if not (source.SCORES/seq/"predicted"/f"{r['depth_ts']}.npy").is_file()]
        if missing:
            excluded.append(dict(depth_ts=dt,offset_s=age,reason="incomplete_original_cache",missing=missing))
            continue
        history = [frame(r) for r in support]
        if stored.get(dt, {}).get("history", len(history)) != len(history) or history[-1]["ts"] != dt:
            raise ValueError("Replay history differs from stored clocks")
        target = history[-1]
        rt = target["rgb_ts"]
        cutoff = min(dt, rt)
        def score(frames):
            return source.score(net, source.original.features(frames, floor, source.V))
        base = score(history)
        record = dict(depth_ts=dt, rgb_ts=rt, cutoff=cutoff, offset_s=age, baseline=base, history=len(history),
                      floor_age_s=(dt-previous[-1]["depth_ts"])/TICKS, archived=stored.get(dt,{}).get("score"),
                      source_rgb_ts=[r["rgb_ts"] for r in support])
        if dt in stored and abs(base - stored[dt]["score"]) > .0001:
            raise ValueError(f"Archived raw score differs at {dt}")
        is_case = dt in by_ts
        if is_case:
            tubes = [tube_mask(cap, f) for f in history]
            record["case"] = by_ts[dt]["case"]
            record["tube_removed"] = score([remove(f, m) for f,m in zip(history,tubes)])
            if abs(record["tube_removed"]-by_ts[dt]["arms"]["outer"]["scores"]["all"]) > .0001:
                raise ValueError(f"Archived tube-removed score differs at {dt}")
        if not baseline_only:
            masks, entries = {}, []
            for i, f in enumerate(history[:-1]):
                ts = f["rgb_ts"]
                if ts >= cutoff:
                    raise ValueError("Unordered RGB exposure")
                mask, info = verifier.classify(ts, cutoff, f["maps"], f["raw_mono"])
                if mask.dtype != np.bool_ or mask.shape != f["depth"].shape:
                    raise ValueError("Expected a boolean mask on the original grid")
                witnesses = info["witnesses"]
                if any(t >= cutoff for t in witnesses):
                    raise ValueError("A verification image reaches or exceeds the decision clock")
                if mask.any() and not any(t > ts for t in witnesses):
                    raise ValueError("A withdrawal lacks a subsequent pre-decision witness")
                masks[str(ts)] = mask
                info = dict(info)
                arrays = {name:v for name,v in info.items() if isinstance(v,np.ndarray)}
                if is_case and seq.endswith("4"):
                    p = destination / "evidence" / seq / str(rt)
                    p.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(p/f"{ts}.npz", **arrays)
                diagnostics = {k:v for k,v in info.items() if k not in arrays}
                valid = (f["depth"] > 0) & (f["depth"] < MAX_M)
                removed = mask[valid]
                grid, body = inspect_points(f["points"][removed],target["pose"],floor,source.V)
                entries.append(dict(source_rgb_ts=ts, withdrawn=int(removed.sum()), grid=grid, body=body,
                                    tube_rays_withdrawn=int((mask & valid & tubes[i]).sum()) if is_case else None,
                                    diagnostics=diagnostics))
            p = destination / "masks" / seq
            p.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p/f"{rt}.npz", **masks)
            past = [masks[str(f["rgb_ts"])] for f in history[:-1]]
            record["filtered"] = score([*(remove(f,m) for f,m in zip(history[:-1],past)),target])
            if seq.endswith("3") and base > source.I.THETA >= record["filtered"]:
                record["dropout_support"] = isolate_dropout(cap, history, past, floor, score)
            if is_case:
                interventions = [m | t for m,t in zip(past,tubes[:-1])] + [tubes[-1]]
                record["tube_removed_filtered"] = score([remove(f,m) for f,m in zip(history,interventions)])
            record["changes"] = entries
        reports.append(record)
        print(seq, round(age,2), round(base,4), round(record.get("filtered",base),4), flush=True)
    return dict(zero_depth_ts=zero, reports=reports, excluded=excluded,
                original_floor_clock=summary(reports,True), completed_floor_clock=summary(reports,False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra-root", type=Path, default=OUT.parent / "rgb")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--exclude-near",action="store_true",
                    help="Test the absence of a near match while allowing ambiguities at larger depths.")
    ap.add_argument("--baseline-only",action="store_true")
    ap.add_argument("--sequence",choices=list(ARMS))
    args=ap.parse_args()
    near_out = OUT.parent/"near-exclusion/replay"
    if args.out is None:
        args.out = near_out if args.exclude_near else OUT
    elif args.exclude_near and args.out.resolve() != near_out.resolve():
        ap.error(f"The near-exclusion arm writes only to {near_out}")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    archive = json.loads((source.TUBE / "d8/attribution.json").read_bytes())
    weights = source.WORK / "ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    if source.original.file_digest(weights) != archive["model_sha256"]:
        raise ValueError("D8 weights changed")
    net = source.I.D.make_head(8)
    net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    net.eval()
    args.out.mkdir(parents=True,exist_ok=True)
    result = dict(threshold=source.I.THETA, model_sha256=archive["model_sha256"],
                  verifier="near_exclusion" if args.exclude_near else "metric_match", sequences={})
    for seq in ([args.sequence] if args.sequence else reversed(ARMS)):
        result["sequences"][seq] = evaluate(seq,net,archive,args.extra_root,args.baseline_only,args.out,args.exclude_near)
        if not args.sequence and not args.baseline_only:
            completed = {**result, "sequences":{seq:result["sequences"][seq]}}
            (args.out/f"result-{seq}.json").write_text(
                json.dumps(completed,indent=1,allow_nan=False,default=scalar_info), encoding="utf8")
    name = "baseline" if args.baseline_only else "result"
    if args.sequence:
        name += "-"+args.sequence
    (args.out/f"{name}.json").write_text(json.dumps(result,indent=1,allow_nan=False,default=scalar_info),encoding="utf8")


if __name__ == "__main__":
    main()
