"""Paired RGB-ground removal replay of the consumed chest hybrid_out development sample.

Prepare reproduces ground/axes solely for the original evaluator. The candidate receives
only the current RGB frame. Because removing finite depth values cannot create a positive
post-opening area, we infer on previously alerting rows and their causal scale updates.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict

import sanpo_wall_attribution_20261118 as prior
from sanpo_wall_attribution_20261118 import ARM, ART, RUN, core, cv2, frame, np, read

OUT = ART / "work/ground-alert-20261118/sanpo"


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")


def plan(rows):
    targets, anchors, last = [], set(), {}
    for i, row in enumerate(rows):
        source = row["source"][ARM]
        if source in ("tof", "floor"):
            last[row["seq"]] = i
        if any(row["alert"][ARM].values()):
            assert row["evaluable"]
            targets.append(i)
            if source == "last":
                anchors.add(last[row["seq"]])
    return targets, sorted(anchors.difference(targets))


def advance(rng, f):
    z = f.ref_depth
    ys = (np.arange(z.shape[0], dtype=float)[:, None] - f.K[1, 2]) / f.K[1, 1]
    below = np.count_nonzero((z > .3) & (z < 10) & (ys * z > .5))
    if below >= 2000:
        n = min(30000, below)
        rng.choice(below, n, replace=False)
        for _ in range(300):
            rng.choice(n, 3, replace=False)


def prepare(rows):
    targets, anchors = plan(rows)
    need = set(targets)
    existing = read(prior.OUT / "ground.json")["planes"]
    rng = np.random.default_rng(20261117)
    k = max(1, len(rows) // 20)
    samples = [frame(rows[i])[0] for i in list(range(0, len(rows), k))[:20]]
    check = core.self_check(samples, rng)
    original = read(RUN / "report.json")["sanpo_chest_seg"]["self_check"]
    assert np.isclose(check["sensor_height_m"], original["sensor_height_m"])
    del samples
    planes = {}
    for i, row in enumerate(rows):
        f, _ = frame(row)
        if i in need and str(i) not in existing:
            reg = core.regions(f, rng)
            assert {b: reg["truth_" + b] for b in core.BANDS} == row["truth"], i
            n, h = reg["ground"]
            planes[i] = [n.tolist(), h]
        else:
            advance(rng, f)
            if i in need:
                planes[i] = existing[str(i)]
        if (i + 1) % 400 == 0:
            print("reference rows", i + 1, flush=True)
    assert set(planes) == need
    write("ground.json", {"targets": targets, "anchors": anchors, "planes": planes})
    print("targets", len(targets), "extra scale sources", len(anchors), flush=True)


def evaluator(plane):
    n, h = np.asarray(plane[0]), plane[1]
    down = -n
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    return {"ground": (n, h), "axes": (down, forward, np.cross(down, forward))}


def opened_low(pred, f, reg, reference=False):
    shape = pred.shape
    z = pred.ravel()
    valid = np.isfinite(z) & ((z > .3) & (z < 10) if reference else z > .05)
    pts = core.rays(f.K, *shape) * np.where(valid, z, 0)[:, None]
    above, _, corridor = core._corridor(pts, valid, reg["axes"], reg["ground"][1])
    lo, hi = core.BANDS["LOW"]
    opened = core._open(corridor & (above >= lo) & (above < hi), *shape)
    return opened, valid


def low_overlap(pred, mask, f, reg):
    shape = pred.shape
    z = pred.ravel()
    opened, valid = opened_low(pred, f, reg)
    area = core.pixel_area(f.K, *shape, np.where(valid, z, 0))
    crossed = opened & mask.ravel()
    return dict(opened_pixels=int(opened.sum()), covered_pixels=int(crossed.sum()),
                opened_area=float(area[opened].sum()), covered_area=float(area[crossed].sum()))


def loss_image(i, f, reg, depth, mask):
    def outline(img, binary, color):
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 3)

    rgb = f.image
    panels = [rgb.copy() for _ in range(3)]
    panels[1][mask] = np.uint8(.70 * panels[1][mask] + .30 * np.array([235, 215, 20]))
    outline(panels[1], opened_low(depth, f, reg)[0].reshape(mask.shape), (0, 70, 250))
    outline(panels[2], opened_low(f.ref_depth, f, reg, reference=True)[0].reshape(mask.shape), (0, 240, 245))
    labels = ("RGB: " + str(i), "RGB ground (cyan) / baseline LOW (red)", "Stereo reference LOW (yellow)")
    for j, panel in enumerate(panels):
        panel = cv2.resize(panel, (640, 360))
        cv2.rectangle(panel, (0, 0), (445, 24), (0, 0, 0), -1)
        cv2.putText(panel, labels[j], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .50, (255, 255, 255), 1, cv2.LINE_AA)
        panels[j] = panel
    path = OUT / f"isolated_low_loss_{i:04d}.jpg"
    assert cv2.imwrite(str(path), np.hstack(panels))


def score(rows, device):
    from rgb_ground_mask_20261118 import GroundMask
    prep = read(OUT / "ground.json")
    targets, anchors = set(prep["targets"]), set(prep["anchors"])
    report = read(RUN / "report.json")["sanpo_chest_seg"]
    candidate = GroundMask(device)
    model = prior.Methods()
    model.set_unidepth_pixels(315560)
    result = []
    for i in sorted(targets | anchors):
        row = rows[i]
        f, _ = frame(row, True)
        f.eye_height = report["eye_heights"][f.seq]
        depth = model.hybrid_out(f)
        assert model.source[ARM] == row["source"][ARM], (i, model.source[ARM], row["source"][ARM])
        if i in anchors:
            continue
        reg = evaluator(prep["planes"][str(i)])
        base = core.band_alert(depth, f, reg)
        assert base == row["alert"][ARM], (i, base, row["alert"][ARM])
        mask = candidate.predict(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB))
        assert mask.shape == depth.shape and mask.dtype == np.bool_
        cleaned = depth.copy()
        cleaned[mask] = np.nan
        arm = core.band_alert(cleaned, f, reg)
        assert all(not arm[b] or base[b] for b in base), (i, base, arm)
        if (row["truth"] == dict(LOW=1, BODY=0, HEAD=0)
                and ((base["LOW"] and not arm["LOW"]) or (any(base.values()) and not any(arm.values())))):
            loss_image(i, f, reg, depth, mask)
        overlap = low_overlap(depth, mask, f, reg) if base["LOW"] else None
        z = f.ref_depth.ravel()
        near = core.depth_metrics(depth, f.ref_depth, np.isfinite(z) & (z >= .5) & (z <= 3.))
        stored = row["depth"][ARM]["near"]
        diff = (abs(near["absrel"] - stored["absrel"]) if near and stored and
                near["absrel"] is not None and stored["absrel"] is not None else None)
        result.append(dict(row=i, seq=f.seq, frame=f.index, source=model.source[ARM],
                           base=base, candidate=arm, low_support=overlap,
                           mask_fraction=float(mask.mean()), near_absrel_diff=diff))
        if len(result) % 50 == 0:
            print("paired alarm rows", len(result), flush=True)
    write("predictions.json", result)


def paired(indices, values, band):
    def hit(alert):
        return any(alert.values()) if band == "ANY" else alert[band]
    cells = Counter()
    for i in indices:
        a, b = (hit(alert) for alert in values[i])
        cells[(a, b)] += 1
    on, lost, gained, off = (cells[pair] for pair in [(True, True), (True, False), (False, True), (False, False)])
    return dict(paired_n=len(indices), base_on=on + lost, candidate_on=on + gained,
                both_on=on, base_on_candidate_off=lost, base_off_candidate_on=gained, both_off=off,
                base_off=off + gained, candidate_off=off + lost)


def summarize(rows):
    predictions = read(OUT / "predictions.json")
    changed = {r["row"]: r for r in predictions}
    targets, anchors = plan(rows)
    assert set(changed) == set(targets)
    oracle = {r["row"]: r["removed"] for r in read(prior.OUT / "attribution.json") if r["match"]}
    values = {}
    all_seqs = sorted({r["seq"] for r in rows})
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        a = row["alert"][ARM]
        if not row["evaluable"] or any(v is None for v in a.values()):
            continue
        arm = changed[i]["candidate"] if i in changed else a
        values[i] = a, arm
        t = row["truth"]
        strict = all(v == 0 for v in t.values())
        flags = dict(STRICT_EMPTY=strict, ANY_POS=core.obstacle_truth(row) == 1,
                     AMBIGUOUS_ANY=core.obstacle_truth(row) == -1,
                     LOW_POS=t["LOW"] == 1, LOW_STRICT_EMPTY=t["LOW"] == 0,
                     LOW_ONLY=core.obstacle_truth(row, "LOW_ONLY") == 1,
                     STRICT_LOW=(t == dict(LOW=1, BODY=0, HEAD=0)))
        flags["STRICT_LOW_NEAR"] = flags["STRICT_LOW"] and row["truth_dist"]["LOW"] <= 1.5
        if i in oracle:
            assert strict and any(a.values())
            semantics = oracle[i]["walkable_semantic"]
            height = oracle[i]["reference_below_low"]
            flags.update(PRIOR_STRICT_FP=True, WALKABLE_95=semantics,
                         REF_BELOW_LOW_101=height, BOTH_SURFACE_94=height and semantics)
        for key, selected in flags.items():
            if selected:
                groups[key].append(i)
    assert (len(groups["BOTH_SURFACE_94"]), len(groups["WALKABLE_95"]),
            len(groups["REF_BELOW_LOW_101"]), len(groups["PRIOR_STRICT_FP"])) == (94, 95, 101, 123)
    bands = ("ANY", "LOW", "BODY", "HEAD")
    per_seq = {s: {} for s in all_seqs}
    totals = {}
    for group, ix in groups.items():
        seq_rows = defaultdict(list)
        for i in ix:
            seq_rows[rows[i]["seq"]].append(i)
        for seq, selected in seq_rows.items():
            per_seq[seq][group] = {b: paired(selected, values, b) for b in bands}
        totals[group] = {}
        for band in bands:
            entry = paired(ix, values, band)
            blocks = [per_seq[s][group][band] for s in seq_rows]
            entry.update(sequences=len(seq_rows),
                         sequences_with_reduced_alarms=sum(b["base_on"] > b["candidate_on"] for b in blocks),
                         macro_base=sum(b["base_on"] / b["paired_n"] for b in blocks) / len(blocks),
                         macro_candidate=sum(b["candidate_on"] / b["paired_n"] for b in blocks) / len(blocks))
            totals[group][band] = entry
    low_support = {}
    for group in ["STRICT_EMPTY", "ANY_POS", "LOW_POS", "STRICT_LOW", "STRICT_LOW_NEAR"]:
        supported = [changed[i]["low_support"] for i in groups[group]
                     if i in changed and changed[i]["low_support"] is not None]
        low_support[group] = dict(baseline_low_alarms=len(supported),
                                  rgb_mask_touches_opened_low=sum(v["covered_pixels"] > 0 for v in supported),
                                  rgb_mask_covers_over_half_opened_low_area=sum(
                                      v["covered_area"] > .5 * v["opened_area"] for v in supported))
    lost = []
    for i in groups["ANY_POS"]:
        a, b = values[i]
        dropped = [v for v in a if a[v] and not b[v]]
        if any(a.values()) and not any(b.values()):
            dropped.append("ANY")
        if not dropped:
            continue
        r = rows[i]
        lost.append(dict(row=i, seq=r["seq"], time_s=r["t"], truth=r["truth"], distance=r["truth_dist"],
                         disappeared=dropped, base=a, candidate=b, file=(f"isolated_low_loss_{i:04d}.jpg"
                         if i in groups["STRICT_LOW"] and ("LOW" in dropped or "ANY" in dropped) else None)))
    diffs = [r["near_absrel_diff"] for r in predictions if r["near_absrel_diff"] is not None]
    fp_by_source = {}
    for src in sorted({r["source"][ARM] for r in rows}):
        ix = [i for i in groups["STRICT_EMPTY"] if rows[i]["source"][ARM] == src]
        fp_by_source[src] = paired(ix, values, "ANY")
    spec = prior.HOST / "rgb_ground_mask_20261118.py"
    summary = dict(rows=len(rows), sequences=len(all_seqs), evaluable=sum(r["evaluable"] for r in rows),
                   sample="SANPO real chest development replay; one sample every 2 seconds",
                   original_run=str(RUN.relative_to(prior.ROOT)), baseline=ARM,
                   rgb_candidate=spec.name, rgb_candidate_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
                   provenance=dict(candidate="RGB only; removed depth pixels before the unchanged reader",
                                   baseline="Emulated ToF zones from stereo reference; sequence sensor height from reference planes",
                                   scoring="Human sky mask cleans stereo reference; reference ground and axes for readout and occupancy labels"),
                   paired_rows=len(values), original_alert_rows=len(targets), extra_anchor_rows=len(anchors),
                   original_reproduced_band_count=len(predictions) * len(core.BANDS),
                   near_absrel_checked=len(diffs), near_absrel_max_difference=max(diffs),
                   paired=totals, low_pixel_overlap=low_support, strict_fp_by_scale=fp_by_source)
    write("per_sequence.json", per_seq)
    write("lost_bands_on_reference_positive.json", lost)
    write("summary.json", summary)
    print(json.dumps({k: {b: [v[b]["paired_n"], v[b]["base_on"], v[b]["candidate_on"]]
                           for b in ("ANY", "LOW")} for k, v in totals.items()}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "score", "summary"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    rows = read(RUN / "rows_sanpo_chest_seg.json")
    if args.mode == "prepare":
        prepare(rows)
    elif args.mode == "score":
        score(rows, args.device)
    else:
        summarize(rows)
