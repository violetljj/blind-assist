"""Four-arm cache-only Development probe. No model or simulator execution."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from near_field import Camera, NearFieldEncoder
from ground_anchor import fit_ground, arm_inputs
from boundary_retention import diagnose_boundary
from diagnose_depth_loss import contained, distribution, sha

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from research_backend import torch_observation

ARMS = ("raw", "scale_only", "ground_only", "scale_and_ground")
RECEIPT_SHA = "e23186a10e6f5021017c7336b8c9490617e6b86ebf7ce27b5225cec673f343c1"


def read_verified(path, expected, identities):
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError("Receipt hash mismatch: " + str(path))
    identities[str(path)] = actual
    return data


def compare(alerts, states, reference):
    unknown = np.asarray(states) == "UNKNOWN"
    return dict(tp=int((alerts & reference).sum()), fp=int((alerts & ~reference).sum()),
                fn=int((~alerts & reference).sum()), tn=int((~alerts & ~reference).sum()),
                unknown=int(unknown.sum()), unknown_positive=int((unknown & reference).sum()))


def execute(run, output):
    started = time.perf_counter()
    identities = {}
    prior = json.loads(read_verified(run/"result.json", RECEIPT_SHA, identities))
    replay = prior["replay"]
    source = Path(replay["source"]["model_dir"]).resolve(strict=True)
    manifest = json.loads(read_verified(source/"manifest.json", replay["source"]["manifest_sha256"], identities))
    frames, records = manifest["episodes"][0]["frames"], replay["rows"]
    if len(frames) != 11 or len(records) != 11:
        raise ValueError("Require complete eleven-frame source")
    loaded = []
    for i, (frame, record) in enumerate(zip(frames, records)):
        if frame["sample_index"] != i or record["sample_index"] != i:
            raise ValueError("Source order changed")
        native_path = contained(source, frame["depth_path"])
        prediction_path = Path(record["predicted_path"]).resolve(strict=True)
        if not prediction_path.is_relative_to(run):
            raise ValueError("Prediction escapes original run")
        # Decode the exact bytes hashed, avoiding separate hash/load races.
        native = np.load(io.BytesIO(read_verified(native_path, record["native_depth_sha256"], identities)), allow_pickle=False)
        predicted = np.load(io.BytesIO(read_verified(prediction_path, record["predicted_sha256"], identities)), allow_pickle=False)
        read_verified(contained(source, frame["rgb_path"]), record["rgb_sha256"], identities)
        camera = Camera(**record["camera"])
        calibration = manifest["calibration"]
        cam, wearer = frame["camera_transform"], frame["wearer_transform"]
        if (camera.width != calibration["width"] or camera.height != calibration["height"]
            or camera.hfov_deg != calibration["horizontal_fov_degrees"]
            or not np.isclose(camera.camera_height_m, cam["z"]-wearer["z"])
            or camera.pitch_deg != cam["pitch"] or cam["roll"] != 0
            or wearer["roll"] != 0 or cam["yaw"] != wearer["yaw"]):
            raise ValueError("Calibration disagrees with receipt")
        if any(a.dtype != np.float32 or a.shape != (camera.height,camera.width) for a in (native,predicted)):
            raise ValueError("Depth raster contract changed")
        loaded.append((record, camera, native, predicted))
    load_s = time.perf_counter()-started
    if not torch.cuda.is_available():
        raise RuntimeError("Prior equivalent CUDA placement unavailable; no silent fallback")
    encoders = {camera: NearFieldEncoder(camera, device="cuda:0") for _,camera,_,_ in loaded}
    first = loaded[0]
    encoders[first[1]].encode(first[3])  # one cache-only warmup, excluded from arm times
    totals = {arm: dict(tp=0,fp=0,fn=0,tn=0,unknown=0,unknown_positive=0) for arm in ARMS}
    times = {arm: [] for arm in ARMS}
    fit_times, boundary_times, rows = [], [], []
    for record, camera, native, predicted in loaded:
        encoder = encoders[camera]
        reference = np.asarray(record["native"]["near_field"]["alerts"], dtype=bool)
        np.testing.assert_array_equal(encoder.encode(native).alerts(), reference)
        tick = time.perf_counter()
        fit = fit_ground(predicted, camera)
        fit_times.append((time.perf_counter()-tick)*1000)
        row = {"sample_index": record["sample_index"], "time_s": record["time_s"],
               "camera": asdict(camera), "fit": fit.summary(), "arms": {}}
        for arm in ARMS:
            tick = time.perf_counter()
            inputs = arm_inputs(predicted, fit, arm)
            if inputs is None:
                alerts = np.zeros((3,3), bool)
                states = [["UNKNOWN"]*3 for _ in range(3)]
                evidence = None
            else:
                evidence = encoder.encode(inputs[0], ground_plane=inputs[1])
                alerts, states = evidence.alerts(), evidence.region_state
            times[arm].append((time.perf_counter()-tick)*1000)
            if arm == "raw":
                np.testing.assert_array_equal(alerts, record["rgb"]["near_field"]["alerts"])
            counts = compare(alerts, states, reference)
            for key, count in counts.items():
                totals[arm][key] += count
            row["arms"][arm] = dict(alerts=alerts.tolist(), states=states, counts=counts,
                                    evidence=evidence.summary() if evidence else None)
        tick = time.perf_counter()
        row["boundary"] = diagnose_boundary(native, predicted, camera)
        boundary_times.append((time.perf_counter()-tick)*1000)
        rows.append(row)
    counts = {key: sum(row["boundary"][key] for row in rows) for key in (
        "native_edges", "matched_native_edges", "predicted_edges_in_neighborhood", "extraneous_predicted_edges")}
    counts["existence_recall"] = counts["matched_native_edges"]/counts["native_edges"] if counts["native_edges"] else None
    counts["by_native_height"] = {height: {key: sum(r["boundary"]["by_native_height"][height][key] for r in rows)
                                        for key in ("native_edges", "matched_native_edges")}
                                  for height in ("low", "body", "head")}
    if totals["raw"]["tp"] != 0 or totals["raw"]["fn"] != 26:
        raise AssertionError("Frozen raw baseline changed")
    return dict(schema="nearfield-ground-anchor-v1", status="COMPLETED", phase="EXPLORE",
                frames=11, arm_evaluations=44, model_inference_calls=0, simulator_launches=0,
                timestamp_waits=0, source_duration_s=records[-1]["time_s"]-records[0]["time_s"],
                verified_inputs=identities, totals=totals, rows=rows, boundary=counts,
                fitting_failures=sum(r["fit"]["status"] != "ACCEPTED" for r in rows),
                reference="fixed prior native supported direction-height observations, not independent object truth",
                backend=dict(encoder=asdict(torch_observation(output=encoders[first[1]].ray_x)),
                             reused_placement=prior["replay_representation_backend"],
                             fit="CPU TASK_NOT_GPU_SUITABLE: bounded three-variable solves",
                             boundary="CPU GPU_BACKEND_UNAVAILABLE: NumPy/OpenCV implementation"),
                timing=dict(load_hash_decode_s=load_s, fit_ms=distribution(fit_times),
                            arm_ms={arm: distribution(v) for arm,v in times.items()},
                            boundary_ms=distribution(boundary_times),
                            total_processing_s=time.perf_counter()-started,
                            excludes_from_total="Python/import startup, final receipt writing; no online inference measured"),
                thin_or_shape_instance_recall="NOT_EVALUABLE", registration="PENDING",
                promotion=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(strict=True), args.output.resolve()
    if not output.is_relative_to((ROOT/"artifacts.local").resolve()) or output.is_relative_to(run):
        raise ValueError("Use a fresh canonical artifact directory outside consumed input")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = Path(__file__).with_name("GROUND_ANCHOR_20260907.md")
    receipt = dict(code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
                   protocol_sha256=sha(protocol), python=sys.executable, torch=torch.__version__, numpy=np.__version__)
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    (output/"started.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    try:
        result = {**receipt, **execute(run, output)}
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("totals","fitting_failures","boundary","timing")}))
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps({**receipt,"error":repr(exc)},indent=2)+"\n",encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
