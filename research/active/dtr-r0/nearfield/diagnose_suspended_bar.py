"""Frozen NF-G3 center/head cache attribution; native depth is evaluator-only."""
import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

# Establish the repository tools import path before loading backend consumers.
from run_ground_anchor import ROOT, read_verified
from run_surface_support import tensors, attribution
from near_field import Camera, NearFieldEncoder, surface_support
from diagnose_depth_loss import distribution, sha
from research_backend import torch_observation

EVALUATION_SHA = "e332b251bdd0267897855ad41fbafdf5cb40f817037e9c08b00d4de8a8b8174a"
PREDICTION_SHA = "fa29b6ba9e2fecf94caffd0ca05732c579f17fe43ac2ef3b35dfa96057e010ec"


def stats(values):
    finite = torch.isfinite(values)
    return {**distribution(values[finite].detach().cpu().numpy()),
            "nonfinite": int((~finite).sum())}


@torch.inference_mode()
def execute(source):
    started = time.perf_counter()
    identities = {}
    evaluation = json.loads(read_verified(source / "evaluation/result.json", EVALUATION_SHA, identities))
    predictions = json.loads(read_verified(source / "predictions/result.json", PREDICTION_SHA, identities))
    assert evaluation["prediction_receipt_sha256"] == PREDICTION_SHA
    rows = [r for r in evaluation["rows"] if r["case_name"] == "bar_near"]
    records = [r for r in predictions["rows"] if r["case_name"] == "bar_near"]
    assert len(rows) == len(records) == 1
    row, record = rows[0], records[0]
    arrays = {}
    for kind in ("native", "predicted"):
        path = Path(record[kind + "_path"])
        expected = record[kind + "_sha256"]
        assert evaluation["verified_inputs"][str(path)] == expected
        arrays[kind] = np.load(io.BytesIO(read_verified(path, expected, identities)), allow_pickle=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    encoder = NearFieldEncoder(Camera(**record["camera"]), device="cuda:0")
    native, predicted = arrays["native"], arrays["predicted"]
    reference = encoder.encode(native)
    np.testing.assert_array_equal(reference.alerts(), row["native_alerts"])
    assert reference.alerts()[1, 2]
    _, nx, nh, nb, _, _, ns = tensors(encoder, native, None)
    mask = (encoder.direction == 1) & ns & (nb == 2) & (nx <= 3)
    assert row["fit"]["status"] == "ACCEPTED"
    arms = {}
    for name, plane in (("raw", None), ("ground_only", row["fit"]["plane"])):
        evidence = encoder.encode(predicted, ground_plane=plane)
        np.testing.assert_array_equal(evidence.alerts(), row["arms"][name]["alerts"])
        assert evidence.region_state == row["arms"][name]["states"]
        misses = attribution(encoder, native, predicted, plane, reference.alerts(), evidence)
        result = next(r for r in misses if r["direction"] == "center" and r["height"] == "head")
        d, x, height, band, valid, eligible, support = tensors(encoder, predicted, plane)
        candidate = (encoder.direction == 1) & eligible & (band == 2)
        near = candidate & (x <= 3)
        # Constant depth makes every triple compatible, exposing eligibility alone.
        triples = surface_support(torch.zeros_like(d), height, eligible)
        weak = near & ~support
        split = {"no_eligible_triple": int((weak & ~triples).sum()),
                 "eligible_triple_depth_inconsistent": int((weak & triples).sum())}
        assert sum(split.values()) == int(weak.sum())
        finite = torch.isfinite(d)
        invalid_split = {
            "nonfinite": int((mask & ~finite).sum()),
            "depth_at_or_below_0p08": int((mask & finite & (d <= .08)).sum()),
            "depth_at_or_above_12": int((mask & finite & (d >= 12)).sum()),
            "nonforward_valid_depth": int((mask & finite & (d > .08) & (d < 12) & (encoder.ray_x <= 0)).sum())}
        assert sum(invalid_split.values()) == result["buckets"]["invalid"]
        arms[name] = {**result, "invalid_partition": invalid_split,
            "weak_near_candidates": int(weak.sum()), "weak_support_partition": split,
            "reference_all_finite_forward_m": stats(x[mask]),
            "reference_all_finite_height_m": stats(height[mask]),
            "candidate_forward_m": stats(x[candidate]),
            "weak_forward_m": stats(x[weak]), "weak_height_m": stats(height[weak]),
            "reference_overlapping_range_loss": int((mask & finite & (x > 3)).sum()),
            "reference_overlapping_height_outside_head": int((mask & finite & ((height < 1.4) | (height >= 1.85))).sum())}
    torch.cuda.synchronize()
    return dict(schema="nearfield-bar-diagnosis-v1", status="COMPLETED", phase="EXPLORE",
        case_name="bar_near", sample_index=record["sample_index"], verified_inputs=identities,
        saved_fit=row["fit"], native_supported_center_head_near_pixels=int(mask.sum()),
        native_forward_m=stats(nx[mask]), native_height_m=stats(nh[mask]), arms=arms,
        backend=asdict(torch_observation(output=encoder.ray_x)),
        model_inference_calls=0, simulator_launches=0, fitting_calls=0, threshold_sweeps=0,
        metadata_placement="CPU: receipt hashing, decoding and small quantile summaries; dense masks CUDA",
        timing=dict(total_processing_s=time.perf_counter()-started,
                    exclusions="Python/import startup and receipt writing; cache diagnosis, not online latency"),
        limitation="Consumed single synthetic view. Native center/head supported pixels are geometric cell reference, not an instance mask or independently labeled obstacle accuracy. Overlapping losses are not additive; ordered buckets partition the reference.",
        promotion=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts.local").resolve()):
        raise ValueError("Canonical artifacts required")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    code_paths = [Path(__file__), Path(__file__).with_name("run_surface_support.py"),
                  Path(__file__).with_name("near_field.py"), Path(__file__).with_name("run_ground_anchor.py")]
    code_hashes = {str(p): sha(p) for p in code_paths}
    try:
        result = execute(args.source.resolve(strict=True))
        result.update(code_sha256=code_hashes, python=sys.executable, torch=torch.__version__)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
