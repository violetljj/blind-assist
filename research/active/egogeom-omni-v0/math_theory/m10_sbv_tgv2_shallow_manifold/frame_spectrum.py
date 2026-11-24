"""Eight-cell valid transects on the frozen offending frame; no padding across missing cells."""
import json
from pathlib import Path

import cv2
import numpy as np

from eval_empirical import ARMS, OLD, PAIR, SOURCES, DEST, read, save, sha, valid_view, axes, grid_fields
from tgv import CELL, solve


def spectrum(fields, valid):
    # 40 cm lines along either grid axis. First remove constant and linear components.
    xx = np.arange(8) - 3.5
    taper = np.hanning(8)
    energy = np.zeros(5)
    count = 0
    for f, v in ((fields, valid), (fields.T, valid.T)):
        complete = np.lib.stride_tricks.sliding_window_view(v, 8, axis=1).all(-1)
        lines = np.lib.stride_tricks.sliding_window_view(f, 8, axis=1)[complete]
        if not len(lines):
            continue
        detrended = lines - lines.mean(axis=-1, keepdims=True) - (lines @ xx / (xx @ xx))[:, None] * xx
        energy += (abs(np.fft.rfft(taper * detrended, axis=-1))**2).sum(axis=0)
        count += len(lines)
    if not count:
        return {"n":0, "power_m2":None}
    # Parseval: sum across bins equals detrended taper-weighted mean-square height.
    energy[1:-1] *= 2
    return {"n":count, "power_m2":(energy / (8 * count * np.sum(taper**2))).tolist()}


def target_peaks():
    """Fixed-denominator per-object maxima from prediction masks and evaluator footprints."""
    maxima = {}
    for source in read(OLD / "areas-camera-valid.json")["sources"]:
        short = source["manifest_sha256"][:16]
        index = read(OLD / "cache" / short / "index.json")
        names = {(r["seq"], r["i"]):r["file"] for r in index["rows"]}
        labels = PAIR / "scoring/labels" / (source["source"] + ".json")
        assert sha(labels) == source["labels_sha256"]
        for case in read(labels)["cases"]:
            if case["height_band"] != "shallow":
                continue
            peaks = dict.fromkeys(ARMS, 0.)
            for obs in case["observations"]:
                if not obs.get("near",True):
                    continue
                footprint = labels.parent / obs["reference_footprint"]
                assert sha(footprint) == obs["reference_mask_sha256"]
                image = cv2.imread(str(footprint), cv2.IMREAD_GRAYSCALE)
                with np.load(DEST / "predictions" / short / names[obs["seq"],obs["i"]]) as packet:
                    roi = cv2.resize(image,tuple(packet["shape"][::-1]),interpolation=cv2.INTER_NEAREST).ravel()>0
                    for arm in ARMS:
                        area = packet[arm + "_area"][roi[packet[arm + "_idx"]]].sum()
                        peaks[arm] = max(peaks[arm], float(area))
            maxima[case["encounter_id"] + "/" + case["object_id"]] = peaks
    return maxima


def main():
    manifest = PAIR / SOURCES["oxford"] / "manifest.json"
    cache = OLD / "cache" / sha(manifest)[:16]
    row = next(r for r in read(manifest)["frames"] if "christ-church" in r["seq"] and r["i"] == 3400)
    entry = next(r for r in read(cache / "index.json")["rows"] if (r["seq"], r["i"]) == (row["seq"], row["i"]))
    path = cache / entry["file"]
    with np.load(path) as p:
        z, k, prob = p["z"], p["K"], p["probability"]
    view = valid_view("oxford", row, z.shape)
    z[~view] = np.nan
    prob[~view] = 0
    r = axes(z,k,prob)
    height, valid, _, _, _ = grid_fields(z,k,prob,r,None)
    u, _, _, j = solve(height, valid)
    results = {k: spectrum(f,valid) for k,f in (("h",height),("u",u),("j",j),("residual",height-u-j))}
    report = {"seq":row["seq"],"frame":3400,"source_sha256":sha(path),
              "code_sha256":{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name("eval_empirical.py"),
                                                       Path(__file__).with_name("tgv.py")]},
              "frequencies_cycles_per_m":np.fft.rfftfreq(8,CELL).tolist(),"fields":results,
              "shallow_object_peak_areas_m2":target_peaks(),
              "coincident_energy_warning":"Power of a sum includes cross-spectra. Fractions are not causal attribution."}
    save(DEST / "frame3400_spectrum.json", report)
    print(json.dumps(report,indent=1))


if __name__ == "__main__":
    main()
