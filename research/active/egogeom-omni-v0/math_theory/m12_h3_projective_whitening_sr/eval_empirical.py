"""Frozen H3 returns only. Cabinet truth / firmware range never enter blind scores."""
import json
import sys
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from projective_h3 import (BlindBoxBank, HERE, REPO, WORK, ar_cov, contrast,
                           features, save, sha)

M1 = HERE.parent / "m1_poisson_cnh_deconv"
sys.path.insert(0, str(M1))
import signed_h3_replay as baseline


def summaries(x):
    return dict(zip(("p50", "p95"), map(float, np.quantile(x, [.5, .95]))))


def analyze(h, w, bank, raw_bank):
    # Ratios of RSS gains to squared whitened norms. Thresholds have no FAR meaning.
    # No firmware range, site metadata, labels, prior or per-record fitting is used.
    eye_w = contrast(np.eye(16))
    timings, core_times, blind, raw = [], [], [], []
    maximum_difference = 0.
    degenerate = 0
    for i, frame in enumerate(h):
        t0 = perf_counter_ns()
        y, q = features(frame, w)
        t1 = perf_counter_ns()
        s = bank.score(q)
        t2 = perf_counter_ns()
        timings.append((t2-t0)/1e6)
        core_times.append((t1-t0)/1e6)
        blind.append(float(s.max()))
        raw.append(float(raw_bank.score(features(frame, eye_w)[1]).max()))
        degenerate += int((np.linalg.norm(y, axis=-1)==0).sum())
        # Probe nuisance algebra on the same measured vector, scaling its noise too.
        if i % 8 == 0:
            transformed = 3.7 * frame + 85.
            transformed[:, 0] += 100.
            transformed[:, 1] -= 30.
            q2 = features(transformed, w)[1]
            maximum_difference = max(maximum_difference, float(abs(q-q2).max()))
    return {"frames": len(h), "zone_frames": len(h)*64, "undefined_ray_zone_frames": degenerate,
            "frame_max_fraction_extra_second_box_fixed_cov": summaries(blind),
            "frame_max_fraction_extra_second_box_diagonal_cov": summaries(raw),
            "exceed_fixed_0p25_fraction": int(np.sum(np.asarray(blind) > .25)),
            "exceed_fixed_0p25_diagonal": int(np.sum(np.asarray(raw) > .25)),
            "actual_transformed_record_max_abs_q_error": maximum_difference}, core_times, timings


def run():
    captured = {}
    receipts = {}
    for label, folder in (("initial","01-background-attempt1"),("object","02-object-attempt2"),
                          ("movement","03-movement-attempt3"),("restored","04-restored-attempt4")):
        cap, receipt = baseline.decode_raw(baseline.CAPTURE/folder/"tof"/"raw.bin")
        captured[label], receipts[label] = cap, receipt
    inventory_file = baseline.PILOT/"inventory.json"
    inventory = {str(r["id"]):r for r in json.loads(inventory_file.read_text(encoding="utf-8"))}
    for label, idx in (("bright","4"),("dark","4b"),("restart","9")):
        cap, receipt = baseline.decode_raw(REPO/inventory[idx]["raw_path"])
        assert receipt["sha256"] == inventory[idx]["raw_sha256"]
        captured[label], receipts[label] = cap, receipt
    cabinet = {k:captured[k] for k in ("initial","object","movement","restored")}
    old = baseline.cabinet_diagnostic(cabinet)
    cov = baseline.SignedNoise([(captured["initial"],0,76)])
    noise_result = cov.evaluate(captured["restored"])
    restore = old["by_segment"]["restored"]["above_chosen_threshold"]
    assert tuple(restore.values()) == (84,26)
    assert round(noise_result["correlated"]["sum_95pct_coverage"],2)==.94
    assert sum(len(d["H"]) for d in captured.values()) == 1209
    w = contrast(ar_cov())
    bank = BlindBoxBank(w)
    raw_bank = BlindBoxBank(contrast(np.eye(16)))
    all_core, all_time = [], []
    def apply(h):
        res, core, full = analyze(h, w, bank, raw_bank)
        all_core.extend(core)
        all_time.extend(full)
        return res
    recorded = {name: apply(data["H"]) for name, data in captured.items()}
    phone = WORK/"phone-new-capture-20261012"
    source_list = phone/"import_receipt.json"
    sessions = json.loads(source_list.read_text(encoding="utf-8"))["sessions"]
    mobile = {}
    for name in sessions:
        path = phone/"inputs"/(name+".npz")
        manifest = phone/"raw"/name/"manifest.json"
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as arr:
            h = arr["H"]
            mobile[name] = {**apply(h), "npz_sha256": sha(path), "manifest_sha256":sha(manifest),
                            "raw_inventory_ToF_count":meta["tof_count"], "npz_keys":arr.files}
    assert len(mobile)==18 and sum(v["frames"] for v in mobile.values())==1818
    return {"development": "previously consumed sensor waveforms; no independent glass labels",
            "sources_seven": receipts, "inventory_sha256":sha(inventory_file),
            "phone_import_sha256":sha(source_list), "cabinet_reference_sha256":sha(baseline.CAPTURE/"notes.md"),
            "m1_restore_counts":restore, "m1_restore_correlated_noise":noise_result,
            "m1_cabinet":old, "unfitted_blind_two_box":recorded, "phone_frozen_sources":mobile,
            "scored_denominators":{"seven":1209,"phone":1818,"total":3027},
            "per_frame_cpu_ms":{"mode":"single BLAS thread, NumPy host, 64 zones, IO excluded",
                                "linear_and_unit_contrasts":summaries(all_core),
                                "with_two_box_energy_score":summaries(all_time)},
            "cabinet_known_box2_prior":{"frames":sum(len(x["H"]) for x in cabinet.values()),
                                        "recoverable_front_intervals":0, "ideal_GLRT_identically":0},
            "physical_radial_alignment":"NOT_EVALUABLE: no independently measured z-axis displacement synchronized to ToF",
            "glass_sensitivity_range_MAE_and_FA_per_km":"NOT_EVALUABLE: no glass depth, glass presence/absence, clear-walk exposure or kilometer labels",
            "code_sha256":{p.name:sha(p) for p in (HERE/"eval_empirical.py",HERE/"projective_h3.py",
                                                  M1/"poisson_cnh.py",M1/"signed_h3_replay.py",
                    REPO/"research/active/dtr-r0/nearfield/cnh_h3_recording_diagnostics.py")}}


if __name__=="__main__":
    with threadpool_limits(limits=1):
        report = run()
    save("v5_empirical_report.json",report)
    print(report["scored_denominators"], report["per_frame_cpu_ms"], report["m1_restore_counts"])
