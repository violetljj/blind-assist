"""Geometry protection vs stricter semantic thresholds on the consumed SANPO chest sample.

Only differing old endpoints require inference. Their removal masks bound all candidates,
and opening plus positive area weighting preserve set inclusion.
"""
import json

import sanpo_ground_alert_replay_20261118 as previous
import sanpo_wall_attribution_20261118 as source
from ground_geometry_filter_20261118 import CONTROL_THRESHOLDS, GroundEvidence, protection
from sanpo_wall_attribution_20261118 import ARM, ART, RUN, core, cv2, frame, np, read

OUT = ART / "work/ground-geometry-20261118/sanpo"
CASES = {0, 1111, 1467}


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, allow_nan=False), encoding="utf-8")


def key(threshold):
    return f"{threshold:.3f}" if threshold in (.995, .999) else f"{threshold:.2f}"


def plan(rows, old):
    targets = {i for i, r in old.items() if r["base"] != r["candidate"]}
    last, origins = {}, {}
    for i, r in enumerate(rows):
        source_name = r["source"][ARM]
        if source_name in ("floor", "tof"):
            last[r["seq"]] = i
        if i in targets and source_name == "last":
            origins[i] = last[r["seq"]]
    return targets, set(origins.values()) - targets, origins


def band_regions(f, depth, reg):
    z = depth.ravel()
    ok = np.isfinite(z) & (z > .05)
    z = np.where(ok, z, 0)
    pts = core.rays(f.K, *depth.shape) * z[:, None]
    above, _, planar = core._corridor(pts, ok, reg["axes"], reg["ground"][1])
    area = core.pixel_area(f.K, *depth.shape, z)
    return {b: planar & (above >= lo) & (above < hi) for b, (lo, hi) in core.BANDS.items()}, area


def read_bands(raw, area, remove):
    return {b: bool(area[core._open(support & ~remove.ravel(), *remove.shape)].sum() >= core.MIN_AREA)
            for b, support in raw.items()}


def render_case(i, f, reg, prob, protected, raw, b):
    def mark(panel, m, color):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, color, 3)
    shape = prob.shape
    sem = prob >= .8
    outline = core._open(raw[b], *shape).reshape(shape)
    surviving = core._open(raw[b] & ~(sem & ~protected).ravel(), *shape).reshape(shape)
    panels = [f.image.copy() for _ in range(4)]
    panels[1][sem] = .64 * panels[1][sem] + .36 * np.array([225, 200, 30])
    mark(panels[1], outline, (0, 60, 250))
    panels[2][protected] = .50 * panels[2][protected] + .50 * np.array([15, 240, 20])
    mark(panels[2], surviving, (0, 60, 250))
    reflow = previous.opened_low(f.ref_depth, f, reg, reference=True)[0]
    mark(panels[3], reflow.reshape(shape), (0, 235, 255))
    labels = [f"RGB: row {i}", f"sem80 (cyan), baseline {b} (red)", f"protected (green), geometry {b} (red)",
              "stereo reference LOW (yellow)"]
    for j in range(4):
        panels[j] = cv2.resize(panels[j], (640, 360))
        cv2.rectangle(panels[j], (0, 0), (335, 22), (0, 0, 0), -1)
        cv2.putText(panels[j], labels[j], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .46, (255,255,255), 1, cv2.LINE_AA)
    name = f"case_{i:04d}_{b}.jpg"
    assert cv2.imwrite(str(OUT / name), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    return name


def run():
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(RUN / "rows_sanpo_chest_seg.json")
    old = {r["row"]: r for r in read(previous.OUT / "predictions.json")}
    ground = {r["row"]: r["removed"] for r in read(source.OUT / "attribution.json") if r["match"]}
    planes = read(previous.OUT / "ground.json")["planes"]
    heights = read(RUN / "report.json")["sanpo_chest_seg"]["eye_heights"]
    targets, anchors, origins = plan(rows, old)
    assert len(targets) == 68 and len(anchors) == 35
    print("target frames", len(targets), "scale sources", len(anchors), flush=True)
    model, network = source.Methods(), GroundEvidence("cuda")
    model.set_unidepth_pixels(315560)
    measured, case_details, errors = {}, [], []
    for i in sorted(targets | anchors):
        r = rows[i]
        f, _ = frame(r, True)
        f.eye_height = heights[f.seq]
        depth = model.hybrid_out(f)
        assert model.source[ARM] == r["source"][ARM], i
        if i in anchors:
            continue
        rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
        p = network.probability(rgb)
        assert p.shape == depth.shape and p.dtype == np.float32 and np.isfinite(p).all()
        protected = protection(depth, f.K, p)
        sem = p >= .80
        assert protected.dtype == np.bool_ and protected.shape == depth.shape and not (protected & ~sem).any()
        reg = previous.evaluator(planes[str(i)])
        raw, area = band_regions(f, depth, reg)
        controls = {key(t): read_bands(raw, area, p >= t) for t in CONTROL_THRESHOLDS}
        geom = read_bands(raw, area, sem & ~protected)
        a, s = r["alert"][ARM], old[i]["candidate"]
        assert controls["1.01"] == a == old[i]["base"] and controls["0.80"] == s, i
        for b in core.BANDS:
            chain = [controls[key(t)][b] for t in CONTROL_THRESHOLDS]
            assert chain == sorted(chain) and s[b] <= geom[b] <= a[b], (i,b)
        z = f.ref_depth.ravel()
        near = core.depth_metrics(depth, f.ref_depth, np.isfinite(z) & (z >= .5) & (z <= 3.))
        stored = r["depth"][ARM]["near"]
        if near and stored and near["absrel"] is not None and stored["absrel"] is not None:
            errors.append(abs(near["absrel"] - stored["absrel"]))
        result = dict(geom=geom, thresholds=controls, mask80_px=int(sem.sum()), restored_px=int(protected.sum()))
        measured[i] = result
        if i in CASES or (core.obstacle_truth(r) == 1 and geom != s):
            b = next(b for b in core.BANDS if a[b] != s[b])
            opened = core._open(raw[b], *p.shape)
            details = {}
            for name, mask in [("base",np.zeros_like(sem)), ("sem80",sem), ("geom",sem & ~protected)]:
                support = core._open(raw[b] & ~mask.ravel(), *p.shape)
                details[name + "_area_m2"] = float(area[support].sum())
            details["restored_pixels_on_original_opened_band"] = int((opened & protected.ravel()).sum())
            details["sem80_deleted_pixels_on_original_opened_band"] = int((opened & sem.ravel()).sum())
            case_details.append(dict(row_index=i, seq=r["seq"], t=r["t"], truth=r["truth"], truth_dist=r["truth_dist"],
                                     affected_band=b, baseline=a, sem80=s, **result, **details,
                                     figure=render_case(i, f, reg, p, protected, raw, b)))
        if len(measured) % 15 == 0:
            print("paired changed rows", len(measured), flush=True)
    records = []
    for i,r in enumerate(rows):
        if not r["evaluable"]:
            continue
        a = r["alert"][ARM]
        s = old[i]["candidate"] if i in old else a
        method = "replayed" if i in targets else "monotone"
        if i in targets:
            result = measured[i]
        else:
            assert a == s
            result = dict(geom=a, thresholds={key(t):a for t in CONTROL_THRESHOLDS}, mask80_px=None, restored_px=None)
        oracle = ground.get(i, {})
        low_ground = oracle.get("reference_below_low", False)
        walkable = oracle.get("walkable_semantic", False)
        records.append(dict(row_index=i, seq=r["seq"], t=r["t"], i=int(round(15*r["t"])), truth=r["truth"],
                            truth_dist=r["truth_dist"], low_dist=r["truth_dist"]["LOW"], floor_case=walkable and low_ground,
                            walkable_case=walkable, below_low_case=low_ground, source=r["source"][ARM],
                            baseline=a, sem80=s, method=method, **result))
    assert len(records) == 1951 and sum(r["floor_case"] for r in records) == 94 and len(measured) == len(targets)
    save("rows.json", records)
    save("cases.json", case_details)
    save("replay_checks.json", dict(paired=len(records), sequences=len({r["seq"] for r in records}),
         changed=len(targets), anchors=sorted(anchors), anchor_origins=origins, endpoint_band_checks=6*len(measured),
         diff_checked=len(errors), max_absrel_diff=max(errors), monotone_completed=len(records)-len(measured),
         gpu_peak_reserved_MiB=round(torch.cuda.max_memory_reserved()/2**20, 1)))
    print("done", len(records), len(case_details), max(errors), flush=True)


if __name__ == "__main__":
    run()
