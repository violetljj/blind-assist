"""Separate count-model tests, v4 simulator transfer, physical H3 and real-image I/O."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import sys

import numpy as np
import torch
from poisson_cnh import OUT, ROUTE, WORK, SlidingPoisson, experiment, front_from_atoms
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(ROUTE))
from egogeom_losses_and_physics import (
    compute_gravity_aligned_ray_map,
    simulate_tof_and_cnh_prompt,
)


def v4_fields():
    from egogeom_omni_v4_model import EgoGeomOmniV4Net, EgoGeomV4Config
    from eval_cross_glasses_zeroshot import (
        H_IN,
        IM_MEAN,
        IM_STD,
        W_IN,
        decode_isotropic_frame,
    )

    ck = torch.load(
        WORK / "egogeom_omni_v4_unified_best.pt", map_location="cpu", weights_only=False
    )
    model = EgoGeomOmniV4Net(
        EgoGeomV4Config(**ck["cfg"]), pretrained_backbone=False
    ).eval()
    model.load_state_dict(ck["ema"], strict=True)
    scenes = {}
    for folder in ("aria_g2_20clips", "coda_zeroshot_shard000", "arkit_val30_shard00"):
        path = min((WORK / "zeroshot_eval" / folder).rglob("*.npz"))
        with np.load(path, allow_pickle=False) as d:
            off = d["rgb_offsets"][:2]
            rgb, k = decode_isotropic_frame(
                d["rgb_blob"][off[0] : off[1]].tobytes(), d["orig_hw"], d["K_orig"]
            )
            gravity = d["poses_R"][0].T @ np.array([0, 0, -1], dtype=np.float32)
            x = torch.tensor(((rgb - IM_MEAN) / IM_STD).transpose(2, 0, 1))[None]
            rays = compute_gravity_aligned_ray_map(
                torch.tensor(k)[None],
                torch.tensor(gravity, dtype=torch.float32)[None],
                H_IN,
                W_IN,
            )
            with torch.inference_mode():
                depth = model(x, rays, None, None, return_features=False)["depth_m"]
            scenes[folder] = {
                "clip": path.stem,
                "frame": 0,
                "pack_keys": sorted(d.files),
                "median_v4_depth_m": float(depth.median()),
                "model_output_shape": list(depth.shape),
            }
    path = WORK / "qnn_v75_v4_final" / "tof_cnh.raw"
    prompt = np.fromfile(path, dtype="<f4").reshape(20, 72, 96)
    mask = prompt[18] > 0.5
    h = np.expm1(prompt[2:18, mask].T) / 10
    # Replicated zones from a single model-input fixture, not independent photons.
    decoded = {
        "valid_pixels": int(mask.sum()),
        "distinct_profiles": len(np.unique(h, axis=0)),
        "median_decoded_sum": float(np.median(h.sum(1))),
        "median_bin0_share": float(np.median(h[:, 0] / h.sum(1))),
    }
    hist = json.loads((WORK / "v4_history.json").read_text())[-1]
    return {
        "real_frame_io": scenes,
        "frozen_qnn_input": decoded,
        "archived_glassrecon_full_model": {
            k: v
            for k, v in hist["eval"].items()
            if k.startswith("glassrecon") and "with_cnh" in k
        },
    }


def measured_h3():
    p = (
        WORK.parent
        / "real-pilot-calibration-replay-dev-20261011"
        / "relaxed-pilot"
        / "1.npz"
    )
    if not p.is_file():
        return {"status": "archived_pilot_unavailable"}
    with np.load(p) as d:
        h = d["H"]
        return {
            "consumed_development": "relaxed-pilot/1.npz",
            "shape": list(h.shape),
            "samples": h.size,
            "fraction_negative": float((h < 0).mean()),
            "fraction_noninteger": float((h != np.rint(h)).mean()),
            "minimum": float(h.min()),
            "maximum": float(h.max()),
            "raw_photon_likelihood_applicable": False,
        }


def counts_sample(n, seed):
    rng = np.random.default_rng(seed)
    counts, z1, labels = [], [], []
    for t in range(n):
        fg = rng.uniform(0.7, 1.9)
        gap = rng.uniform(0.45, 1.40)
        # Each class has the same background-depth and total signal distributions.
        r = rng.uniform(0.15, 0.35) if t % 2 else 0
        photons = rng.uniform(200, 1000)
        mean, _ = experiment(fg, gap, photons, r, 0.74, sbr=rng.uniform(0.5, 2))
        counts.append(rng.poisson(mean))
        z1.append(fg)
        labels.append(t % 2)
    return np.array(counts), np.array(z1), np.array(labels)


def simulator_sample(n, seed):
    """One zone per synthetic plane; select measured bin-0 fractions symmetrically."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    counts, z1, labels = [], [], []
    totals = [0, 0]
    for k in range(200):
        b = 64
        fg = rng.uniform(0.7, 1.9, b).astype("f")
        lab = np.arange(b) % 2
        dep = torch.from_numpy(fg)[:, None, None, None].expand(-1, 1, 64, 64)
        glass = torch.tensor(lab, dtype=torch.float32)[:, None, None, None].expand_as(
            dep
        )
        prompt, _ = simulate_tof_and_cnh_prompt(
            dep,
            torch.ones_like(dep),
            torch.eye(3).repeat(b, 1, 1),
            glass,
            mode="cnh",
            realistic_hardware=True,
        )
        prob = np.expm1(prompt[:, 2:18].permute(0, 2, 3, 1).numpy()) / 10
        valid = prompt[:, 18].numpy() > 0.5
        for j in range(b):
            if totals[lab[j]] >= n // 2:
                continue
            prof = np.unique(prob[j, valid[j]], axis=0)
            prof = prof[(prof[:, 0] >= 0.715) & (prof[:, 0] <= 0.765)]
            if len(prof):
                counts.append(prof[rng.integers(len(prof))])
                labels.append(lab[j])
                z1.append(fg[j])
                totals[lab[j]] += 1
        if len(counts) == n:
            break
    if len(counts) != n:
        raise RuntimeError(f"Insufficient strong-crosstalk zones {totals=}")
    return np.array(counts), np.array(z1), np.array(labels)


def v4_features(y):
    y = y / y.sum(axis=1, keepdims=True)
    raw = np.log1p(10 * y)
    tail = raw[:, 1:] / (1e-4 + raw[:, 1:].sum(axis=1, keepdims=True))
    return np.column_stack([raw, tail, np.diff(tail, axis=1)])


def compare(train, test):
    x, _, lab = train
    yt, z1, labels = test
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(24,), alpha=0.2, max_iter=350, random_state=1717
        ),
    )
    mlp.fit(v4_features(x), lab)
    solver = SlidingPoisson()
    scores, mae_l2, mae_argmax, mae_sfw = [], [], [], []
    for y, fg, positive in zip(yt, z1, labels):
        loc, a, _ = solver.fit(y)
        pred, score = front_from_atoms(loc, a, y)
        scores.append(score)
        if positive:
            mae_sfw.append(abs(fg - pred))
            mae_argmax.append(abs(fg - (1 + np.argmax(y[1:])) * 0.25))
            mae_l2.append(abs(fg - solver.l2(y)))
    sfw, l2, arg = map(np.mean, (mae_sfw, mae_l2, mae_argmax))
    return {
        "train": len(x),
        "test": len(yt),
        "positive": int(labels.sum()),
        "SFW_MAE_m": sfw,
        "L2_MAE_m": l2,
        "argmax_MAE_m": arg,
        "improvement_vs_argmax": 1 - sfw / arg,
        "improvement_vs_L2": 1 - sfw / l2,
        "SFW_AUROC": roc_auc_score(labels, scores),
        "static_feature_MLP_AUROC": roc_auc_score(
            labels, mlp.predict_proba(v4_features(yt))[:, 1]
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = v4_fields()
        result["measured_H3"] = measured_h3()
        result["poisson_model"] = compare(
            counts_sample(2048, 801), counts_sample(256, 802)
        )
        result["v4_realistic_hardware_synthetic_planes"] = compare(
            simulator_sample(512, 141), simulator_sample(256, 142)
        )
    result["measured_glass_front_MAE_and_detection"] = (
        "NOT_EVALUABLE: paired photon-count frames + independent front/background distances and material labels unavailable in these inputs"
    )
    (OUT / "v4_empirical_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
