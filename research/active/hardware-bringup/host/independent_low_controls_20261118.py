"""Blind visibility review and paired frame counts for the fixed SANPO negative windows."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[4] / "artifacts.local/work/independent-low-20261118"
PLAN = ROOT / "sanpo-new/controls.json"
OUT = ROOT / "controls"
ARMS = ("baseline", "sem80", "geom")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_index(bundle):
    return {(r["seq"], r["i"]): r for r in read(bundle)["rows"]}


def visibility(seq, i, ref, dataset):
    base = PLAN.parent / "data" / seq / "camera_chest/left"
    name = f"{i:06d}"
    picture = cv2.imread(str(base / "video_frames" / (name + ".png")))
    jpeg = cv2.resize(picture, (148, 84), interpolation=cv2.INTER_AREA)
    if not ref["ground"]:
        return [0.] * 3, [0.] * 3, jpeg
    payload = gzip.decompress((base / "depth_maps" / (name + ".float16.gz")).read_bytes())
    raw = np.frombuffer(payload, "<f2")
    h, w = map(int, raw[:2])
    stride = 4
    z = raw[2:].reshape(h, w)[::stride, ::stride].astype(np.float32)
    sky_key = hashlib.sha256(f"{dataset}/{seq}/{i}".encode()).hexdigest()[:22]
    sky = cv2.imread(str(ROOT / "replay/reference_sky" / (sky_key + ".png")), cv2.IMREAD_GRAYSCALE)
    if sky is None or sky.shape != (h, w):
        raise ValueError(f"Missing registered sky mask for {seq}/{i}")
    valid = (sky[::stride, ::stride] == 0) & np.isfinite(z) & (z > .3) & (z < 10)
    z = np.where(valid, z, 0)
    k = np.asarray(ref["K"])
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    rx, ry = (u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1]
    down = -np.asarray(ref["ground"][0], float)
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    outline = []
    for depth, side in ((1., -.35), (2.5, -.35), (2.5, .35), (1., .35)):
        xy = k @ (ref["ground"][1] * down + depth * forward + side * right)
        outline.append(np.rint(xy[:2] / xy[2] * [148 / w, 84 / h]).astype(np.int32))
    cv2.polylines(jpeg, [np.array(outline)], True, (0, 255, 255), 1)
    def project(axis):
        return (rx * axis[0] + ry * axis[1] + axis[2]) * z
    along, lateral = project(forward), project(right)
    floor = valid & (np.abs(ref["ground"][1] - project(down)) <= .10) & (along >= .30)
    area = stride**2 * z**2 / (k[0, 0] * k[1, 1] * np.sqrt(1 + rx**2 + ry**2))
    def strips(limit):
        edges = np.linspace(-.35, .35, 4)
        return [round(float(area[floor & (along <= limit) & (lateral >= lo) & (lateral < hi)].sum()), 6)
                for lo, hi in zip(edges, edges[1:])]
    return strips(2.50), strips(1.50), jpeg


def freeze(reference_path):
    spec = read(PLAN)
    reference = reference_index(reference_path)
    rows, sheets = [], []
    for seq, indices in spec["sessions"].items():
        thumbs = []
        for i in indices:
            ref = reference[seq, i]
            full, near, thumb = visibility(seq, i, ref, spec["dataset"])
            cv2.putText(thumb, f"{seq[:5]} {i}", (3, 11), cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(thumb, f"{seq[:5]} {i}", (3, 11), cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(thumb)
            rows.append(dict(seq=seq, i=i, t=ref["t"], strict_empty=ref["strict_empty"],
                             full_ground_m2=full, near_ground_m2=near))
        sheets.append(cv2.hconcat(thumbs))
    for i in range(0, len(sheets), 6):
        if not cv2.imwrite(str(OUT / f"blind-{i//6}.jpg"), cv2.vconcat(sheets[i:i+6])):
            raise OSError("Could not write blind contact sheet")
    save_new(OUT / "reference_controls.json", dict(plan_sha256=digest(PLAN),
             reference_sha256=digest(reference_path), rows=rows))


def save_new(path, data):
    value = (json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Refusing to overwrite different evidence in {path}")
    else:
        with path.open("xb") as dest:
            dest.write(value)


def score(predictions_path, visual):
    ref_path = OUT / "reference_controls.json"
    refs = read(ref_path)["rows"]
    rejects = {(r["seq"], r["i"]) for r in read(visual)["unresolved"]}
    assert rejects <= {(r["seq"], r["i"]) for r in refs}
    preds = reference_index(predictions_path)
    summary = {}
    for key in ("strict", "visible", "visible_near"):
        chosen = [r for r in refs if r["strict_empty"] and (key == "strict" or
                  ((r["seq"], r["i"]) not in rejects and min(r["near_ground_m2" if
                   key == "visible_near" else "full_ground_m2"]) >= .005))]
        alerts = {a: set() for a in ARMS}
        by_seq = defaultdict(list)
        for r in chosen:
            idx = r["seq"], r["i"]
            ev = preds[idx]["evidence"]
            for a in ARMS:
                assert all(isinstance(v, bool) for v in ev[a].values())
                if any(ev[a].values()):
                    alerts[a].add(idx)
            by_seq[idx[0]].append(idx)
        summary[key] = dict(frames=len(chosen), sessions=len(by_seq),
            false_frames={a:len(hits) for a, hits in alerts.items()},
            sessions_with_any_false={a:len({s for s, i in hits}) for a, hits in alerts.items()},
            paired_geom_added=sorted(alerts["geom"] - alerts["sem80"]),
            paired_geom_removed=sorted(alerts["sem80"] - alerts["geom"]),
            by_recording={s:dict(n=len(ix), **{a:len(alerts[a].intersection(ix)) for a in ARMS})
                          for s, ix in by_seq.items()})
    save_new(OUT / "score.json", dict(plan_sha256=digest(PLAN), predictions_sha256=digest(predictions_path),
             reference_controls_sha256=digest(ref_path), visual_sha256=digest(visual), summary=summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("reference", "score"))
    p.add_argument("bundle", type=Path, help="Reference or predictions from the frozen replay")
    p.add_argument("--visual", type=Path, help="Blinded visual review with any unresolved seq/i pairs")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.stage == "reference":
        freeze(a.bundle)
    elif a.visual:
        score(a.bundle, a.visual)
    else:
        p.error("score requires --visual")
