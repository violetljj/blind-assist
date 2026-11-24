"""Paired common-ray error, aggregated within recordings and then equally across sites."""
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import scipy
from scipy.ndimage import distance_transform_edt

from depth_transfer import ARMS, CAM, HERE, NATIVE, OUT, SITES, camera, cohort, name, persist, project_pinhole, pull, sha


def read(path):
    return json.loads(path.read_text())


def pool(rows, region):
    items = [r["regions"][region] for r in rows]
    n = sum(x["pixels"] for x in items)
    return dict(pixels=n, mae={a:sum(x["error_sum"][a] for x in items)/n if n else None for a in ARMS})


def score():
    refs = read(OUT/"reference_receipt.json")
    preds = read(OUT/"prediction_receipt.json")
    assert refs["cohort"] == preds["cohort"] == sha(OUT/"cohort.json")
    assert preds["input_receipt"] == sha(OUT/"input_receipt.json")
    locations, ok = project_pinhole(CAM, NATIVE)
    rows = []
    for row in cohort():
        key = name(row)
        ref = OUT/"reference"/(key+".png")
        pred = OUT/"predictions"/(key+".npz")
        assert sha(ref) == refs["sites"][row["site"]]["reference"][key]
        assert sha(pred) == preds["files"][key]
        native = cv2.imread(str(ref),cv2.IMREAD_UNCHANGED).astype(np.float32)*.001
        gt, in_view = pull(native,locations,ok,nearest=True)
        with np.load(pred) as bundle:
            valid = bundle["valid"]
            outputs = {a:bundle[a] for a in ARMS}
        assert all(np.all(np.isfinite(v)) for v in outputs.values())
        support = valid & in_view & (gt >= .3) & (gt <= 12.)
        dist = distance_transform_edt(valid)
        regions = dict(all=support, near=support & (gt <= 4.), far=support & (gt > 4.),
                       edge=support & (dist <= 16.), interior=support & (dist >= 64.))
        stats = {}
        for region, pixels in regions.items():
            err = {a:float(np.sum(np.abs(outputs[a][pixels]-gt[pixels]),dtype=float)) for a in ARMS}
            stats[region] = dict(pixels=int(pixels.sum()),error_sum=err)
        rows.append(dict(site=row["site"], seq=row["seq"], key=key, t=row["t"],
                         valid=int(valid.sum()), regions=stats))
    regions = list(rows[0]["regions"])
    seqs = {r["seq"]:r["site"] for r in rows}
    result = {}
    for reg in regions:
        seq = {s:pool([r for r in rows if r["seq"] == s],reg) for s in seqs}
        sites = {}
        for site in SITES:
            group = [v for s,v in seq.items() if seqs[s] == site and v["pixels"]]
            sites[site] = dict(sequences=len(group), mae={
                a:float(np.mean([v["mae"][a] for v in group])) if group else None for a in ARMS})
        available = [s["mae"] for s in sites.values() if s["sequences"]]
        result[reg] = dict(sequences=seq, sites=sites, pooled=pool(rows,reg),
                           site_equal={a:float(np.mean([v[a] for v in available])) if available else None for a in ARMS})
    primary = result["all"]
    if any(v["pixels"] == 0 for v in primary["sequences"].values()):
        decision = "incomplete: an entire sequence has no reference rays"
    else:
        passed = all(v["nearest"] < v["black"] for v in
                     (primary["site_equal"], primary["sites"]["blenheim-palace"]["mae"],
                      primary["sites"]["observatory-quarter"]["mae"]))
        decision = "retain for depth development" if passed else "stop universal improvement interpretation"
    report = dict(rows=rows, summaries=result, decision=decision,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                               scipy=scipy.__version__, onnxruntime=ort.__version__, provider='CPUExecutionProvider'),
                  receipts={f:sha(OUT/f) for f in ("cohort.json","input_receipt.json","reference_receipt.json","prediction_receipt.json")},
                  code={p.name:sha(p) for p in [HERE/n for n in ("cohort.py","depth_transfer.py","score_transfer.py")] +
                        [HERE.parent/"atlas.py", HERE.parent/"eval_empirical.py", Path(camera.__file__), Path(camera.S.__file__)]})
    persist(OUT/"score.json", report)
    print(json.dumps(dict(primary=primary, regions={r:v["site_equal"] for r,v in result.items()},
                          decision=decision),indent=1))
