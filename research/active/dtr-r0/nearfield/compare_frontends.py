"""NF-G5 fixed-cache comparison; native geometry is evaluator-only."""
import argparse
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from run_ground_anchor import ROOT, read_verified
from run_evidence_fusion import BranchEvidence, run_rows
from run_surface_support import tensors
from near_field import Camera, NearFieldEncoder
from diagnose_suspended_bar import stats, EVALUATION_SHA, PREDICTION_SHA
from diagnose_depth_loss import sha


def execute(baseline, candidates):
    tick = time.perf_counter()
    identities = {}
    configurations = {}
    source_identity = None
    baseline_alerts = None
    for name, prediction_dir, evaluation_dir in candidates:
        pp, ep = prediction_dir / "result.json", evaluation_dir / "result.json"
        predictions = json.loads(read_verified(pp, PREDICTION_SHA if name == "baseline" else sha(pp), identities))
        evaluation = json.loads(read_verified(ep, EVALUATION_SHA if name == "baseline" else sha(ep), identities))
        assert evaluation["prediction_receipt_sha256"] == sha(pp)
        assert len(predictions["rows"]) == len(evaluation["rows"]) == 18
        identity = [(r["sample_index"], r["case_name"], r["rgb_sha256"], r["native_sha256"], r["camera"]) for r in predictions["rows"]]
        if source_identity is None:
            source_identity = identity
        assert identity == source_identity, "Source, calibration, or order changed"
        branches = []
        bar = None
        for record, row in zip(predictions["rows"], evaluation["rows"]):
            assert record["case_name"] == row["case_name"]
            arrays = {}
            for kind in ("predicted", "native", "rgb"):
                path = Path(record[kind + "_path"])
                digest = record[kind + "_sha256"]
                assert evaluation["verified_inputs"][str(path)] == digest
                payload = read_verified(path, digest, identities)
                if kind != "rgb":
                    arrays[kind] = np.load(io.BytesIO(payload), allow_pickle=False)
            raw, ground = row["arms"]["raw"], row["arms"]["ground_only"]
            branches.append((row["case_name"], BranchEvidence(raw["alerts"], raw["states"]),
                BranchEvidence(ground["alerts"], ground["states"]) if row["fit"]["status"] == "ACCEPTED" else None,
                np.asarray(row["native_alerts"], bool)))
            if row["case_name"] == "bar_near":
                encoder = NearFieldEncoder(Camera(**record["camera"]), device="cuda:0")
                _, nx, _, nb, _, _, ns = tensors(encoder, arrays["native"], None)
                mask = (encoder.direction == 1) & ns & (nb == 2) & (nx <= 3)
                assert int(mask.sum()) == 1054
                bar = dict(native_pixels=int(mask.sum()), native_forward_m=stats(nx[mask]), arms={})
                for arm, plane in (("raw", None), ("ground_only", row["fit"]["plane"] if row["fit"]["status"] == "ACCEPTED" else None)):
                    if arm == "ground_only" and row["fit"]["status"] != "ACCEPTED":
                        bar["arms"][arm] = dict(status="UNAVAILABLE")
                        continue
                    d, x, h, b, valid, eligible, support = tensors(encoder, arrays["predicted"], plane)
                    near_head = (encoder.direction == 1) & (b == 2) & eligible & (x <= 3)
                    bar["arms"][arm] = dict(forward_m=stats(x[mask]),
                        reference_valid_pixels=int((mask & valid).sum()),
                        reference_near_supported_head_pixels=int((mask & near_head & support).sum()),
                        whole_cell_near_supported_head_pixels=int((near_head & support).sum()))
        fusion = run_rows(branches)
        alerts = np.asarray([r["fusion"]["union"]["diagnostic_alerts"] for r in fusion["rows"]], bool)
        if baseline_alerts is None:
            baseline_alerts = alerts
        regressions = [dict(case_name=r["name"], lost_cells=np.argwhere(before & ~after).tolist(),
                            added_cells=np.argwhere(~before & after).tolist())
                       for r, before, after in zip(fusion["rows"], baseline_alerts, alerts)]
        bar_row = next(r for r in fusion["rows"] if r["name"] == "bar_near")
        bar["union_center_head_alert"] = bool(bar_row["fusion"]["union"]["diagnostic_alerts"][1][2])
        configurations[name] = dict(model=predictions["model"], timing=predictions["timing"],
            fitting_failures=evaluation["fitting_failures"], fusion=fusion, bar=bar, changes_from_baseline=regressions,
            meets_predefined_retention=bar["union_center_head_alert"] and fusion["totals"]["union"]["legacy_cells"]["fp"] == 0)
    return dict(schema="nearfield-frontend-comparison-v1", status="COMPLETED", phase="EXPLORE",
        configurations=configurations, verified_inputs=identities, model_inference_calls=0, simulator_launches=0,
        cache_comparison_s=time.perf_counter()-tick, backend="CUDA dense masks; CPU small-cell fusion and metadata",
        limitation="Consumed static synthetic views; cell geometry, not instance labels or first-warning accuracy", promotion=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts.local").resolve()):
        raise ValueError("Canonical artifacts required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        candidates = [("baseline", args.baseline / "predictions", args.baseline / "evaluation")]
        candidates += [(name, args.candidates / (name + "-predictions"), args.candidates / (name + "-evaluation")) for name in ("vkitti", "hypersim1036")]
        result = execute(args.baseline, candidates)
        result["code_sha256"] = sha(Path(__file__))
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({k: dict(union=v["fusion"]["totals"]["union"], bar=v["bar"], retained=v["meets_predefined_retention"]) for k,v in result["configurations"].items()}))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
