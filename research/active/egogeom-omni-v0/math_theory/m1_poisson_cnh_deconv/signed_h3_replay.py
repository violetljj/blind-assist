"""Replay consumed H3 captures in decoded signal units, including the measured cabinet."""

import hashlib
import json
import sys
from collections import Counter

import numpy as np
from poisson_cnh import OUT, ROUTE, WORK
from scipy.optimize import nnls
from scipy.stats import norm

REPO = ROUTE.parents[2]
BRINGUP = REPO / "artifacts.local" / "hardware-bringup"
CAPTURE = BRINGUP / "captures" / "simple-20260927T164234Z-cc4d0e"
PILOT = WORK.parent / "real-pilot-calibration-replay-dev-20261011" / "relaxed-pilot"
sys.path.insert(0, str(ROUTE.parent / "dtr-r0" / "nearfield"))
from cnh_h3_recording_diagnostics import load_recording


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_raw(path):
    decoded = load_recording(path)
    hdr = Counter()
    scale = Counter()
    configs = []
    for line in path.read_bytes().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        if row.get("type") == "config":
            configs.append(
                {
                    k: row[k]
                    for k in (
                        "zones",
                        "bins",
                        "start_bin",
                        "sub_sample",
                        "base_bin_mm",
                        "cnh_bytes",
                        "integration_ms",
                        "frequency_hz",
                    )
                    if k in row
                }
            )
        if row.get("type") == "cnh_frame" and np.shape(row.get("hist_scaler")) == (
            64,
            16,
        ):
            c = row.get("cnh_header", [])
            if len(c) == 5:
                hdr.update(
                    [
                        (
                            len(c),
                            (c[1] >> 16) & 255,
                            (c[1] >> 24) & 1,
                            c[4] & 65535,
                            c[4] >> 16,
                        )
                    ]
                )
            unique, counts = np.unique(row["hist_scaler"], return_counts=True)
            scale.update(dict(zip(unique.tolist(), counts.tolist())))
    receipt = {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": sha(path),
        "frames": len(decoded["seq"]),
        "bad_lines": decoded["quality"]["bad_lines"],
        "missing_seq": decoded["quality"]["missing_seq"],
        "header_formats": [
            {
                "words": k[0],
                "n_buffers": k[1],
                "no_variance": k[2],
                "zones": k[3],
                "bins": k[4],
                "frames": v,
            }
            for k, v in hdr.items()
        ],
        "hist_scaler_counts": dict(scale),
        "observed_config_examples": configs[:1],
    }
    return decoded, receipt


def innovation_samples(data, lo=0, hi=None):
    # Each quartet uses ambient in the first two frames, then a disjoint
    # adjacent-pair innovation. Break gaps, resets, or gaps over 0.5 seconds.
    ix = np.arange(len(data["seq"]))[lo:hi]
    split = (
        np.flatnonzero(
            (np.diff(data["seq"][ix]) != 1)
            | (np.diff(data["ms"][ix]) <= 0)
            | (np.diff(data["ms"][ix]) > 500)
        )
        + 1
    )
    quartets = [
        run[: len(run) // 4 * 4].reshape(-1, 4)
        for run in np.split(ix, split)
        if len(run) >= 4
    ]
    q = np.concatenate(quartets)
    ambient = (data["A"][q[:, 0]] + data["A"][q[:, 1]]) / 2
    diff = (data["H"][q[:, 3], :, 8:14] - data["H"][q[:, 2], :, 8:14]) / np.sqrt(2)
    return ambient.reshape(-1).clip(0), diff.reshape(-1, 6)


class SignedNoise:
    """Per-bin signal-unit variance over ambient and pooled innovation correlation."""

    def __init__(self, recordings):
        samples = [innovation_samples(*args) for args in recordings]
        amb = np.concatenate([s[0] for s in samples])
        delta = np.concatenate([s[1] for s in samples])
        design = np.column_stack([np.ones_like(amb), amb])
        self.coeffs = np.column_stack([nnls(design, col**2)[0] for col in delta.T])
        self.floor = max(1e-9, 0.001 * np.median(delta**2))
        scales = np.sqrt(np.maximum(design @ self.coeffs, self.floor))
        self.corr = 0.9 * np.corrcoef((delta / scales).T) + 0.1 * np.eye(6)
        assert np.linalg.eigvalsh(self.corr).min() > 0

    def evaluate(self, data, lo=0, hi=None):
        amb, diff = innovation_samples(data, lo, hi)
        scales = np.sqrt(
            np.maximum(
                np.column_stack([np.ones_like(amb), amb]) @ self.coeffs, self.floor
            )
        )
        std = diff / scales
        result = {
            "quartets": len(amb) // 64,
            "zone_vectors": len(amb),
            "far_bins": [8, 9, 10, 11, 12, 13],
            "ambient_median": float(np.median(amb)),
            "diagonal_95pct_coverage": float((np.abs(std) <= norm.ppf(0.975)).mean()),
        }
        for tag, r in [("independent", np.eye(6)), ("correlated", self.corr)]:
            # Compare two joint forecasts with the exact same marginals.
            var_sum = np.einsum("ni,ij,nj->n", scales, r, scales)
            energy = diff.sum(axis=1) ** 2
            quad = np.einsum("ni,ij,nj->n", std, np.linalg.inv(r), std)
            logdet = np.linalg.slogdet(r)[1] + 2 * np.log(scales).sum(axis=1)
            score = 0.5 * (6 * np.log(2 * np.pi) + logdet + quad)
            result[tag] = {
                "per_vector_gaussian_NLL": float(score.mean()),
                "sum_observed_over_forecast_second_moment": float(
                    energy.sum() / var_sum.sum()
                ),
                "sum_95pct_coverage": float(
                    (energy <= norm.ppf(0.975) ** 2 * var_sum).mean()
                ),
            }
        return result


def change_score(train, h):
    # Fit one two-bin model per zone to initial cabinet background only.
    # Foreground changes should increase bin1 and occlude the cabinet in bin2.
    mean = train[:, :, 1:3].mean(axis=0)
    sd = np.maximum(train[:, :, 1:3].std(axis=0, ddof=1), 1e-6)
    whitened = (h[:, :, 1:3] - mean) / sd
    front = whitened[:, :, 0]
    joint = np.minimum(front, -whitened[:, :, 1])
    return {
        "front_bin_only": front.max(axis=1),
        "signed_near_up_cabinet_down": joint.max(axis=1),
    }


def cabinet_diagnostic(records):
    background = records["initial"]["H"]
    nfit = len(background) // 2
    thresholds = {
        key: float(np.quantile(score, 0.95, method="higher"))
        for key, score in change_score(background[:nfit], background[nfit:]).items()
    }
    report = {
        "fit_initial_frames": nfit,
        "threshold_initial_later_frames": len(background) - nfit,
        "frame_thresholds": thresholds,
        "by_segment": {},
    }
    for name, d in records.items():
        start = nfit if name == "initial" else 0
        scores = change_score(background[:nfit], d["H"][start:])
        valid = (d["S"] == 5) & (d["N"] > 0) & (d["D"] > 150)
        center = [27, 28, 35, 36]
        dv = d["D"][:, center][valid[:, center]]
        center_hist = d["H"][:, center, :].mean(axis=(0, 1))
        report["by_segment"][name] = {
            "frames_scored": len(d["H"]) - start,
            "firmware_valid_center_zone_frames": len(dv),
            "firmware_center_distance_median_m": float(np.median(dv) / 1000)
            if len(dv)
            else None,
            "mean_center_histogram": center_hist.tolist(),
            "strongest_bin1to4": int(np.argmax(center_hist[1:5]) + 1),
            "above_chosen_threshold": {
                k: int((v > thresholds[k]).sum()) for k, v in scores.items()
            },
            "p50_frame_score": {k: float(np.median(v)) for k, v in scores.items()},
        }
    return report


def run():
    captured = {}
    receipt = {}
    for label, folder in [
        ("initial", "01-background-attempt1"),
        ("object_segment", "02-object-attempt2"),
        ("movement_segment", "03-movement-attempt3"),
        ("restored", "04-restored-attempt4"),
    ]:
        captured[label], receipt[label] = decode_raw(
            CAPTURE / folder / "tof" / "raw.bin"
        )
    inventory = {
        r["id"]: r
        for r in json.loads((PILOT / "inventory.json").read_text(encoding="utf-8"))
    }
    for name, idx in [("bright", "4"), ("dark", "4b"), ("bright_restart", "9")]:
        d, rr = decode_raw(REPO / inventory[idx]["raw_path"])
        if rr["sha256"] != inventory[idx]["raw_sha256"]:
            raise ValueError(f"raw SHA changed: {name}")
        with np.load(PILOT / (idx + ".npz"), allow_pickle=False) as old:
            for key in ["H", "A", "D", "S", "N", "seq", "ms"]:
                if not np.array_equal(d[key], old[key]):
                    raise ValueError(f"decoder mismatch: {name} {key}")
        captured[name] = d
        receipt[name] = rr
    cabinet = {
        k: captured[k]
        for k in ["initial", "object_segment", "movement_segment", "restored"]
    }
    result = {
        "raw_receipts": receipt,
        "signed_change": cabinet_diagnostic(cabinet),
        "noise": {},
    }
    training_sets = {
        "cabinet_initial": [("initial", 0, 76)],
        "corridor_bright_dark": [("bright", 0, 76), ("dark", 0, 76)],
    }
    for setting, parts in training_sets.items():
        noise = SignedNoise([(captured[name], lo, hi) for name, lo, hi in parts])
        result["noise"][setting] = {
            "coefficients_intercept_and_ambient": noise.coeffs.tolist(),
            "corr": noise.corr.tolist(),
            "variance_floor": noise.floor,
            "eval": {},
        }
        for name, lo in [
            ("initial", 76),
            ("restored", 0),
            ("bright", 76),
            ("dark", 76),
            ("bright_restart", 0),
        ]:
            result["noise"][setting]["eval"][name] = noise.evaluate(captured[name], lo)
    src = BRINGUP / "sketches" / "tof_cnh_h3_wifi" / "vl53lmz_plugin_cnh.c"
    vendor = json.loads(
        (ROUTE.parent / "hardware-bringup" / "vendor-lock.json").read_text(
            encoding="utf-8"
        )
    )
    expected = next(
        v["sha256"] for v in vendor["files"] if v["path"] == "src/vl53lmz_plugin_cnh.c"
    )
    assert (
        sha(src)
        == sha(BRINGUP / "sketches" / "tof_cnh_h3" / "vl53lmz_plugin_cnh.c")
        == expected
    )
    result["coarse_cabinet_reference"] = {
        "distance_m": [0.63, 0.64],
        "starting_point": "unconfirmed",
        "notes_sha256": sha(CAPTURE / "notes.md"),
    }
    assert all(
        rec["frames"] == sum(header["frames"] for header in rec["header_formats"])
        for rec in receipt.values()
    )
    result["locked_plugin_sha256"] = expected
    return result


if __name__ == "__main__":
    out = run()
    output = OUT / "signed_h3_replay"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(out, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        {
            tag: {
                k: (
                    round(
                        x["independent"]["sum_observed_over_forecast_second_moment"], 2
                    ),
                    round(
                        x["correlated"]["sum_observed_over_forecast_second_moment"], 2
                    ),
                )
                for k, x in arm["eval"].items()
            }
            for tag, arm in out["noise"].items()
        }
    )
