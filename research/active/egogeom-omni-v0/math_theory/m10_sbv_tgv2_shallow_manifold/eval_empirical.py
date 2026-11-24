"""CPU development replay. Predict every manifest row before opening scoring footprints."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from tgv import CELL, SHAPE, solve

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROUTE = HERE.parents[1]
HOST = REPO / "research/active/hardware-bringup/host"
ART = REPO / "artifacts.local/work"
OLD = ART / "shallow-alerts-20261118"
PAIR = ART / "paired-low-sites-20261118"
DEST = ART / "egogeom-omni-v0/math_theory" / HERE.name
MODEL = ART / "egogeom-omni-v0/egogeom_omni_v41_best_pure_rgb_fp32.onnx"
sys.path[:0] = [str(HOST), str(HOST.parent / "evalkit"), str(ROUTE)]
import core
import shallow_event_comparison_20261118 as scoring
from ground_geometry_filter_20261118 import down_from_ground
from shallow_surface_evidence_20261118 import _points, shallow_masks
from shallow_valid_view_20261118 import valid_view
from egogeom_omni_v41_model import analytical_host_focal_from_K

SOURCES = {"egowalk-union": "newsource/egowalk/locked_union", "lin-yellow": "lamar/locked/lin_yellow",
           "oxford": "oxford/locked"}
ARMS = ("old_tgv", "v41_tgv", "v41_tgv_normals", "v41_local")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=1), encoding="utf8")
    tmp.replace(path)


def axes(depth, k, prob):
    h, w = depth.shape
    step = max(1, max(h, w) // 256)
    vv, uu = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step]
    valid = np.isfinite(z) & (z >= .25) & (z <= 6) & (prob[::step, ::step] >= .8) & (vv >= h * .5)
    down = down_from_ground(_points(uu[valid], vv[valid], z[valid], k))
    if down is None:
        return None
    right = np.cross(down, np.array([0, 0, 1], dtype=np.float32))
    right /= np.linalg.norm(right)
    return np.stack((right, down, np.cross(right, down)))


def infer(session, image, k, rotation):
    # Preserve the entire locked field with a centered aspect-safe letterbox.
    h, w = image.shape[:2]
    factor = min(384 / w, 288 / h)
    nw, nh = round(w * factor), round(h * factor)
    x0, y0 = (384 - nw) // 2, (288 - nh) // 2
    norm = np.zeros((288, 384, 3), np.float32)
    crop = np.s_[y0:y0 + nh, x0:x0 + nw]
    rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_AREA)
    norm[crop] = (rgb / 255. - [.485, .456, .406]) / [.229, .224, .225]
    fx, fy = k[0, 0] * nw / w, k[1, 1] * nh / h
    cx, cy = k[0, 2] * nw / w + x0, k[1, 2] * nh / h + y0
    v, u = np.indices((288, 384), dtype=np.float32)
    rays = np.stack(((u + .5 - cx) / fx, (v + .5 - cy) / fy, np.ones_like(v)))
    rays /= np.linalg.norm(rays, axis=0)
    rays = (rotation @ rays.reshape(3, -1)).reshape(1, 3, 288, 384).astype(np.float32)
    focal = analytical_host_focal_from_K(fx, fy, cx, cy)
    depth, n = session.run(["depth_m", "normal_grav"], {
        "rgb": norm.transpose(2, 0, 1)[None], "grav_rays": rays,
        "host_focal": np.full((1, 1, 1, 1), focal, np.float32)})
    return (cv2.resize(depth[0, 0][crop], (w, h)),
            cv2.resize(n[0].transpose(1, 2, 0)[crop], (w, h)))


def cell_medians(keys, values):
    out = np.full(np.prod(SHAPE), np.nan, np.float32)
    order = np.lexsort((values, keys))
    ids, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    enough = counts >= 3
    out[ids[enough]] = values[order[starts[enough] + counts[enough] // 2]]
    return out.reshape(SHAPE)


def grid_fields(z, k, prob, rotation, normals):
    v, u = np.nonzero(np.isfinite(z) & (z >= .05) & (z < 8.))
    xyz = _points(u, v, z[v, u], k) @ rotation.T
    column = np.floor((xyz[:, 0] + 1.5) / CELL).astype(np.int32)
    row = np.floor((xyz[:, 2] - .3) / CELL).astype(np.int32)
    inside = (column >= 0) & (column < SHAPE[1]) & (row >= 0) & (row < SHAPE[0])
    v, u, xyz = v[inside], u[inside], xyz[inside]
    keys = row[inside] * SHAPE[1] + column[inside]
    h = cell_medians(keys, -xyz[:, 1])
    valid = np.isfinite(h)
    if valid.any():
        h -= np.median(h[valid])
    slopes = slope_valid = None
    if normals is not None:
        n = normals[v, u]
        ok = (abs(n[:, 1]) >= .35) & (prob[v, u] >= .8)
        slopes = np.stack([cell_medians(keys[ok], CELL * n[ok, i] / n[ok, 1]) for i in (0, 2)])
        slope_valid = np.isfinite(slopes).all(axis=0)
        slopes = np.nan_to_num(slopes)
    corridor = (abs(xyz[:, 0]) <= .35) & (xyz[:, 2] <= 1.50)
    return h, valid, slopes, slope_valid, (v[corridor], u[corridor], keys[corridor])


def store_support(packet, name, mask, area):
    indices = np.flatnonzero(mask)
    packet[name + "_idx"] = indices.astype(np.int32)
    packet[name + "_area"] = area.ravel()[indices].astype(np.float64)


def distribution(x):
    return dict(zip(("p50", "p95"), map(float, np.percentile(x, [50, 95])))) if len(x) else None


def predict(session, packet_path, rgb_path, source, row, frame, attribution=False):
    timings = {}
    with np.load(packet_path, allow_pickle=False) as p:
        z, k, prob = p["z"], p["K"], p["probability"]
    view = valid_view(source, row, z.shape)
    z[~view] = np.nan
    prob[~view] = 0.
    h, w = z.shape
    rotation = axes(z, k, prob)
    out = {"shape": z.shape}
    diag = {"axis_available": rotation is not None, "timings_ms": timings}
    if rotation is None:
        for arm in ARMS:
            out[arm + "_idx"] = np.empty(0, np.int32)
            out[arm + "_area"] = np.empty(0, np.float64)
        return out, diag
    image = cv2.imread(str(rgb_path))
    if image is None:
        raise FileNotFoundError(rgb_path)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    image[~view] = 0
    start = perf_counter()
    z41, n41 = infer(session, image, k, rotation)
    z41[~view] = np.nan
    timings["v41_cpu_inference_and_resampling"] = 1000 * (perf_counter() - start)
    for label, dz, normals in (("old", z, None), ("v41", z41, n41)):
        area = core.pixel_area(k, h, w, dz.ravel()).reshape(h, w)
        start = perf_counter()
        height, valid, slopes, slope_valid, locations = grid_fields(dz, k, prob, rotation, normals)
        timings[label + "_grid"] = 1000 * (perf_counter() - start)
        diag[label + "_valid_cells"] = int(valid.sum())
        v, u, keys = locations
        for normals_on in ([False] if normals is None else [False, True]):
            name = label + "_tgv" + ("_normals" if normals_on else "")
            start = perf_counter()
            surface, _, _, jump = solve(height, valid, slopes if normals_on else None, slope_valid)
            timings[name + "_solver"] = 1000 * (perf_counter() - start)
            mask = np.zeros((h, w), bool)
            j = jump.ravel()[keys]
            mask[v, u] = valid.ravel()[keys] & (j >= .05) & (j < .15)
            store_support(out, name, core._open(mask, h, w).reshape(h, w), area)
            if attribution and label == "old":
                _, local, _ = shallow_masks(dz, k, prob)
                eligible = local[v, u] & valid.ravel()[keys]
                jj = jump.ravel()[keys[eligible]]
                mass = area[local].sum()
                local_valid_area = area[v[eligible], u[eligible]]
                diag["frame3400"] = {
                    "local_m2": float(mass), "gridded_local_m2": float(local_valid_area.sum()),
                    "above_5cm_j_local_m2": float(local_valid_area[jj >= .05].sum()),
                    "local_j_median_p95_m": distribution(jj),
                    "field_energy_m2": dict(surface_centered=float(np.sum((surface[valid] - np.median(surface[valid]))**2)),
                                            jump=float(np.sum(jump[valid]**2)),
                                            data_residual=float(np.sum((height[valid] - surface[valid] - jump[valid])**2)))}
                # Spectrum proxy on observed adjacent triples; no fill across missing cells.
                for tag, field in (("input", height), ("surface", surface), ("jump", jump)):
                    second = []
                    for f, m in ((field, valid), (field.T, valid.T)):
                        mask3 = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
                        second.extend(np.abs(np.diff(f, 2, axis=1)[mask3]).tolist())
                    diag["frame3400"][tag + "_second_diff_m"] = distribution(second)
        if label == "v41":
            _, local, _ = shallow_masks(dz, k, prob)
            store_support(out, "v41_local", local, area)
    return out, diag


def run_predictions():
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])
    paths = [Path(__file__), HERE / "tgv.py", MODEL, HOST / "shallow_surface_evidence_20261118.py",
             HOST / "shallow_valid_view_20261118.py", HOST / "ground_geometry_filter_20261118.py"]

    paths += [HOST.parent / "evalkit/core.py", ROUTE / "egogeom_omni_v41_model.py", HOST / "spires_insert_dev_20261117.py"]
    code = {p.name: sha(p) for p in paths}
    records = {}
    digests = {}
    for source, subdir in SOURCES.items():
        manifest = PAIR / subdir / "manifest.json"
        source_hash = sha(manifest)
        cache = OLD / "cache" / source_hash[:16]
        inventory = read(cache / "index.json")
        assert inventory["manifest_sha256"] == source_hash
        spec = read(manifest)
        by_key = {(r["seq"], r["i"]): r for r in spec["frames"]}
        digests[source] = {"manifest_sha256": source_hash, "index_sha256": sha(cache / "index.json")}
        for n, entry in enumerate(inventory["rows"]):
            seq, i = entry["seq"], entry["i"]
            row = by_key[seq, i]
            packet_path = cache / entry["file"]
            rgb_path = manifest.parent / spec["data_root"] / row["rgb"]
            input_ids = {"rgb_sha256": sha(rgb_path), "packet_sha256": sha(packet_path)}
            folder = DEST / "predictions" / source_hash[:16]
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / entry["file"]
            receipt = output.with_suffix(".json")
            identity = hashlib.sha256(json.dumps([input_ids, source_hash, code, ort.__version__], sort_keys=True).encode()).hexdigest()
            if receipt.exists() and output.exists() and read(receipt)["identity"] == identity:
                data = read(receipt)
                if sha(output) != data["output_sha256"]:
                    raise ValueError(f"Prediction bytes changed at {output}")
            else:
                fields, diag = predict(session, packet_path, rgb_path, source, row, entry,
                                      attribution=(source == "oxford" and i == 3400 and "christ" in seq))
                with output.with_suffix(".partial").open("wb") as handle:
                    np.savez_compressed(handle, **fields)
                output.with_suffix(".partial").replace(output)
                data = dict(identity=identity, **input_ids, output_sha256=sha(output), diagnostics=diag)
                save(receipt, data)
            key = (entry["t"], i)
            if key in records:
                raise ValueError(f"Duplicated time/index: {key}")
            records[key] = (output, data, seq)
            if (n+1) % 32 == 0:
                print(source, n+1, "/", len(inventory["rows"]), flush=True)
    return records, code, digests


def evaluate(records, code, digests):
    areas = read(OLD / "areas-camera-valid.json")
    baselines = scoring.compare(areas, read(scoring.ROSTER), read(OLD / "areas.json"))
    archived = read(OLD / "comparison-camera-valid.json")
    for category in ("matched", "leave_site_out", "raw"):
        assert baselines[category] == archived[category]
    lookup = {}
    labels_hashes = {}
    for name in SOURCES:
        path = PAIR / "scoring/labels" / (name + ".json")
        labels_hashes[name] = sha(path)
        for c in read(path)["cases"]:
            for o in c["observations"]:
                if o.get("near", True):
                    lookup[c["encounter_id"], c["object_id"], o["seq"], o["i"]] = (path.parent / o["reference_footprint"], o["reference_mask_sha256"])
    def populate(row, event=None):
        path, _, seq = records[row["t"], row["i"]]
        with np.load(path, allow_pickle=False) as p:
            roi = None
            if event is not None:
                name, digest = lookup[event, row["object_id"], seq, row["i"]]
                assert sha(name) == digest
                a = cv2.imread(str(name), cv2.IMREAD_GRAYSCALE)
                roi = cv2.resize(a, tuple(p["shape"][::-1]), interpolation=cv2.INTER_NEAREST).ravel() > 0
            for arm in ARMS:
                area = p[arm + "_area"]
                row["total"][arm] = float(area.sum())
                if roi is not None:
                    row["target"][arm] = float(area[roi[p[arm + "_idx"]]].sum())
    for e in areas["events"]:
        for r in e["observations"]:
            populate(r, e["id"])
    windows = areas["windows"]
    for w in windows:
        for r in w["rows"]:
            populate(r)
    sites = sorted({w["site"] for w in windows})
    results = {x: {name: baselines[x][name] for name in scoring.METHODS} for x in ("matched", "leave_site_out","raw")}
    for arm in ARMS:
        cuts = {s: scoring.match_threshold([w for w in windows if w["site"] != s], arm) for s in sites}
        for scenario, thresholds in (("leave_site_out", cuts), ("raw", dict.fromkeys(sites, scoring.MIN_AREA)),
                                     ("matched", dict.fromkeys(sites, scoring.match_threshold(windows, arm)))):
            results[scenario][arm] = scoring.score(areas, arm, thresholds)
    all_times = defaultdict(list)
    no_axis = []
    frame3400 = None
    for (t,i), (_, d, seq) in records.items():
        diag = d["diagnostics"]
        if not diag["axis_available"]:
            no_axis.append([seq, i])
        for name, val in diag["timings_ms"].items():
            all_times[name].append(val)
        if "frame3400" in diag:
            frame3400 = diag["frame3400"]
    assert frame3400 and abs(frame3400["local_m2"]-.09846) < .00001
    evaluator = {p.name:sha(p) for p in [OLD / "areas-camera-valid.json", OLD / "areas.json",
                                          OLD / "comparison-camera-valid.json", scoring.ROSTER, Path(scoring.__file__)]}
    output = dict(arms=results, input_code_sha256=code, sources_sha256=digests, evaluator_sha256=evaluator,
                  labels_sha256=labels_hashes, total_frames=len(records), no_axis_frames=no_axis,
                  per_frame_receipts={f"{seq}/{i}": {k:v for k,v in d.items() if k != "diagnostics"}
                                      for (_,i),(_,d,seq) in records.items()},
                  frame3400=frame3400, cpu_ms={k:dict(n=len(v), **distribution(v)) for k,v in all_times.items()},
                  metrics="Shallow support, fixed 5 silent approaches / 8 windows / 55 timestamps. No path length or distance target in this protocol.")
    save(DEST / "v5_empirical_report.json", output)
    print({k:(s["shallow"]["first_detections"],s["controls"]) for k,s in results["leave_site_out"].items()})


if __name__ == "__main__":
    cv2.setNumThreads(1)
    DEST.mkdir(parents=True, exist_ok=True)
    evaluate(*run_predictions())
