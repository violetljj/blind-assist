"""Bounded, reproducible depth-representation Development comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from research_backend import BackendCandidate, Workload, select_backend, torch_observation
from near_field import Camera, NearFieldEncoder, quantile_baseline
from probe_source import generate_cases


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentiles(samples):
    return {"n": len(samples), "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95))}


def choose_device(camera, depth, output, name):
    encoders = {"cpu": NearFieldEncoder(camera)}
    if torch.cuda.is_available():
        encoders["cuda"] = NearFieldEncoder(camera, device="cuda")
        cpu_result, gpu_result = (encoders[d].encode(depth) for d in ("cpu", "cuda"))
        np.testing.assert_array_equal(cpu_result.alerts(), gpu_result.alerts())
        np.testing.assert_allclose(cpu_result.region_distance_m, gpu_result.region_distance_m,
                                   atol=1e-5, rtol=1e-5)
    candidates = {
        name: BackendCandidate(
            name, name, lambda e=encoder: e.encode(depth),
            lambda _, e=encoder: torch_observation(output=e.ray_x),
            torch.cuda.synchronize if name == "cuda" else lambda: None,
        ) for name, encoder in encoders.items()
    }
    record = select_backend(
        Workload.BATCH_TENSOR, cpu=candidates["cpu"], gpu=candidates.get("cuda"),
        cpu_reason=None if "cuda" in candidates else "ACCELERATOR_UNAVAILABLE",
        record_path=output / f"backend-{name}.json", warmups=1, repeats=3,
    )
    return record["selected_device_type"], record


def score(prediction, expected):
    return {"tp": int((prediction & expected).sum()),
            "fp": int((prediction & ~expected).sum()),
            "fn": int((~prediction & expected).sum()),
            "tn": int((~prediction & ~expected).sum()),
            "false_positive_frame": int((prediction & ~expected).any())}


def aggregate(rows):
    aggregate = {}
    for arm in ("STRIDE4_QUANTILE", "DENSE_QUANTILE", "SUPPORTED_TILES"):
        families = defaultdict(lambda: {"frames": 0, "tp": 0, "fp": 0, "fn": 0,
                                       "tn": 0, "false_positive_frame": 0})
        latency = []
        for row in rows:
            for group in ("ALL", row["family"]):
                record = families[group]
                record["frames"] += 1
                for key, value in row["arms"][arm]["score"].items():
                    record[key] += value
            latency.append(row["arms"][arm]["milliseconds"])
        for values in families.values():
            values["recall"] = values["tp"]/(values["tp"]+values["fn"]) \
                if values["tp"]+values["fn"] else None
        aggregate[arm] = {"families": dict(families), "latency": percentiles(latency)}
    return aggregate


def run_procedural(output):
    cases = list(generate_cases())
    if not 1 <= len(cases) <= 120:
        raise ValueError("procedural source exceeds declared budget")
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names):
        raise ValueError("duplicate source identity")
    camera = Camera(**cases[0]["camera"])
    device, backend = choose_device(camera, cases[-1]["depth"], output, "320x240")
    encoder = NearFieldEncoder(camera, device=device)
    rows, digests = [], []
    for i, case in enumerate(cases):
        depth, expected = case["depth"], case["expected"]
        source_hash = hashlib.sha256(depth.tobytes()).hexdigest()
        digests.append({"name": case["name"], "depth_sha256": source_hash,
                        "expected": expected.astype(int).tolist(), "camera": case["camera"],
                        "note": case["note"]})
        arms = {}
        # Rotate execution order to reduce a fixed cold/warm order advantage.
        arm_names = ["STRIDE4_QUANTILE", "DENSE_QUANTILE", "SUPPORTED_TILES"]
        for arm in arm_names[i % 3:] + arm_names[:i % 3]:
            started = time.perf_counter()
            if arm == "SUPPORTED_TILES":
                evidence = encoder.encode(depth)
                distances = evidence.region_distance_m
            else:
                distances = quantile_baseline(depth, camera,
                                              stride=4 if arm == "STRIDE4_QUANTILE" else 1)
            elapsed = (time.perf_counter()-started)*1000
            prediction = distances <= 3.
            arms[arm] = {"score": score(prediction, expected),
                         "alerts": prediction.astype(int).tolist(), "milliseconds": elapsed}
            if arm == "SUPPORTED_TILES":
                arms[arm]["evidence"] = evidence.summary()
                arms[arm]["raw_depth_bytes"] = depth.nbytes
        if hashlib.sha256(depth.tobytes()).hexdigest() != source_hash:
            raise AssertionError("input mutated by an arm")
        if case.get("expected_unknown"):
            if any(s != "UNKNOWN" for row in evidence.region_state for s in row):
                raise AssertionError("invalid frame became observed absence")
        rows.append({"name": case["name"], "family": case["family"],
                     "expected": expected.astype(int).tolist(), "arms": arms})
    (output / "procedural-inputs.json").write_text(json.dumps(digests, indent=2)+"\n")
    result = {"source": "ANALYTIC_DEPTH_SPACE_DEVELOPMENT", "frames": len(rows),
              "backend": backend, "summary": aggregate(rows), "rows": rows,
              "numeric_payload_note": "arrays only; excludes Python objects and temporary work buffers",
              "raw_depth_bytes_per_frame": cases[0]["depth"].nbytes,
              "token_bytes_per_frame": rows[0]["arms"]["SUPPORTED_TILES"]["evidence"]["token_bytes"]}
    (output / "procedural.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--metric-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists() or (args.output / "procedural.json").exists():
        raise SystemExit("refusing to overwrite a consumed diagnostic")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    code_hashes = {str(p.relative_to(ROOT)): sha256(p)
                   for p in Path(__file__).parent.glob("*.py")}
    receipt = {"schema": "nearfield-representation-probe-v1", "phase": "EXPLORE",
               "python": sys.executable, "platform": platform.platform(),
               "torch": torch.__version__, "numpy": np.__version__,
               "cpu_threads": torch.get_num_threads(), "code_sha256": code_hashes,
               "attention_forward_m": 3., "algorithm_promotion": False,
               "real_camera_validation": False}
    (args.output / "started.json").write_text(json.dumps(receipt, indent=2)+"\n")
    try:
        procedural = run_procedural(args.output)
        manifest = json.loads((args.model_dir/"manifest.json").read_text(encoding="utf-8"))
        calibration = manifest["calibration"]
        frames = [f for e in manifest["episodes"] for f in e["frames"]]
        if len(frames) != 11:
            raise ValueError("fixed Willow source must contain exactly eleven frames")
        frame = frames[0]
        camera = Camera(calibration["width"], calibration["height"],
                        calibration["horizontal_fov_degrees"],
                        frame["camera_transform"]["z"]-frame["wearer_transform"]["z"],
                        frame["camera_transform"]["pitch"])
        device, backend = choose_device(camera, np.load(args.model_dir/frame["depth_path"]),
                                        args.output, "640x360")
        from rgb_replay import run_replay
        replay = run_replay(model_dir=args.model_dir, metric_source=args.metric_source,
                            weights=args.weights, output=args.output/"rgb-replay", device=device)
        receipt.update(status="COMPLETED", elapsed_seconds=time.perf_counter()-started,
                       procedural_summary=procedural["summary"], replay=replay,
                       replay_representation_backend=backend,
                       source_manifest_sha256=sha256(args.model_dir/"manifest.json"))
        (args.output/"result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"status": receipt["status"], "output": str(args.output),
                          "procedural_summary": procedural["summary"]}), flush=True)
    except BaseException as exc:
        receipt.update(status="MECHANICAL_FAILURE", error=repr(exc),
                       elapsed_seconds=time.perf_counter()-started)
        (args.output/"failure.json").write_text(json.dumps(receipt, indent=2)+"\n")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
