"""Read-only acceptance analysis for the corrected 10k-B readout adaptation.

This consumes saved probabilities only.  In particular, range events are
recomputed by convolving the three per-cell capped-count distributions rather
than treating a complement of the six-cell alert as a far-range score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from body_query_data import read, sha, truth, write
from city_dev_selection import select_threshold


TRAINED_10K_SHA = "db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0"
ROLES = ("train", "dev", "eval")
HEADS = ("BODY", "HEAD")


def confusion(score, label, threshold):
    predicted = np.asarray(score, dtype=np.float64) >= threshold
    positive = np.asarray(label, dtype=bool)
    return dict(TP=int((predicted & positive).sum()), FP=int((predicted & ~positive).sum()),
                FN=int((~predicted & positive).sum()), TN=int((~predicted & ~positive).sum()))


def range_event_from_counts(probabilities):
    """P(sum of exactly one three-cell range is >=3), via count convolution."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (12, 4):
        raise ValueError("Expected saved Bx12x4 count probabilities")
    p = probabilities.reshape(-1, 2, 2, 3, 4)
    below = np.zeros(p.shape[:3] + (3,), dtype=np.float64)
    below[..., 0] = 1.0
    for cell in range(3):
        next_below = np.zeros_like(below)
        for total in range(3):
            for count in range(total + 1):
                next_below[..., total] += below[..., total-count] * p[..., cell, count]
        below = next_below
    return 1.0 - below.sum(-1)


def native_range_events(counts):
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 12:
        raise ValueError("Expected Bx12 native capped count labels")
    return counts.reshape(-1, 2, 2, 3).sum(-1) >= 3


def groups(decisions, labels, records):
    result = {}
    for group_id in dict.fromkeys(row["group_id"] for row in records):
        ids = np.array([i for i, row in enumerate(records) if row["group_id"] == group_id])
        result[group_id] = bool((decisions[ids] == labels[ids]).all())
    return dict(correct=int(sum(result.values())), total=len(result), rows=result)


def verify_invariants(cache, run):
    protocol = read(run / "protocol.json")
    preflight = read(run / "preflight.json")
    fit = read(run / "fit-complete.json")
    receipt = read(run / "receipt.json")
    assert protocol["steps"] == fit["steps"] == receipt["steps"] == 2000
    assert receipt["fits"] == 1
    assert protocol["initial_sha256"] == protocol["old_sha256"] == TRAINED_10K_SHA
    assert preflight["initial_sha256"] == preflight["baseline_sha256"] == TRAINED_10K_SHA
    assert preflight["schedule_identical"] is True and preflight["train_prediction_parity"] == 0.0
    assert preflight["trainable"] == ["query_readout.weight", "query_readout.bias"]
    assert fit["trainable"] == "query_readout only"
    assert fit["frozen_parameters_verified"] is True and fit["buffers_verified"] is True
    assert sha(cache / "manifest.json") == protocol["cache_sha256"]
    assert sha(run / "schedule.npy") == protocol["schedule_sha256"]
    baseline = cache.parent / 'run-v1'
    assert sha(baseline / 'NEW-step2000.pt') == TRAINED_10K_SHA
    assert np.array_equal(np.load(run / "schedule.npy", allow_pickle=False),
                          np.load(baseline / "schedule.npy", allow_pickle=False))
    old_state = torch.load(baseline / 'NEW-step2000.pt', map_location='cpu', weights_only=True)
    new_state = torch.load(run / 'NEW-step2000.pt', map_location='cpu', weights_only=True)
    changed = [key for key in old_state if not torch.equal(old_state[key], new_state[key])]
    assert set(changed) == {'query_readout.weight', 'query_readout.bias'}, changed
    assert sha(run / "NEW-step2000.pt") == fit["checkpoint_sha256"]
    assert sha(run / "result.json") == receipt["result_sha256"]
    assert sha(run / "selection.json") == receipt["selection_sha256"]
    train_source = Path(__file__).with_name("body_query_10000_readout_train.py")
    assert sha(train_source) == protocol["source_sha256"]
    for name, digest in protocol["dependency_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    return protocol, fit, receipt


def run_analysis(cache, run, output):
    cache, run, output = Path(cache), Path(run), Path(output)
    if run.resolve() != output.resolve():
        raise ValueError("Readout analysis must be written beside the immutable run-v2 evidence")
    protocol, fit, receipt = verify_invariants(cache, run)
    saved_selection = read(run / "selection.json")
    saved_result = read(run / "result.json")
    summary = {"schema": "body-query-10000-readout-analysis-v1", "status": "PASS",
               "scope": "Consumed shared-site Development; one readout-only adaptation, no new inference or fitting",
               "arms": {}, "gates": {}}
    recomputed = {}
    wrong_far = {}
    head_only_body_fp = {}
    for role in ROLES:
        record, target = truth(cache, role, training=(role == "train"))
        labels = target["near"].astype(bool)
        native_events = native_range_events(target["counts"])
        near_only_head = native_events[:, 1, 0] & ~native_events[:, 1, 1]
        head_only = np.array([row['condition'] == "HEAD_ONLY" for row in record["records"]])
        assert int(head_only.sum()) == {'train':1000,'dev':400,'eval':600}[role]
        summary[role] = {"native_HEAD_near_only_frames": int(near_only_head.sum()), "arms": {}}
        recomputed[role] = {}
        wrong_far[role] = {}
        head_only_body_fp[role] = {}
        for arm in ("OLD", "NEW"):
            prediction = dict(np.load(run / f"{arm}-{role}.npz", allow_pickle=False))
            if set(prediction) != {"near", "support", "counts"}:
                raise ValueError("Saved prediction payload differs from training output")
            if prediction["near"].shape != labels.shape or prediction["counts"].shape != (len(labels), 12, 4):
                raise ValueError("Saved prediction shape mismatch")
            if not np.isfinite(prediction["near"]).all() or not np.isfinite(prediction["counts"]).all():
                raise ValueError("Nonfinite saved prediction")
            if not np.allclose(prediction["counts"].sum(-1), 1.0, rtol=0, atol=2e-6):
                raise ValueError("Saved count distributions are not normalized")
            thresholds = np.asarray(saved_selection[arm]["thresholds"], dtype=np.float64)
            decisions = prediction["near"].astype(np.float64) >= thresholds
            head_confusions = {name: confusion(prediction["near"][:, h], labels[:, h], thresholds[h])
                               for h, name in enumerate(HEADS)}
            saved_heads = saved_result["arms"][arm][role]["heads"]
            for name, value in head_confusions.items():
                assert all(value[key] == saved_heads[name]["selected"][key] for key in value)
            range_scores = range_event_from_counts(prediction["counts"])
            range_confusions = {
                f"{head}_{distance}": confusion(range_scores[:, h, d], native_events[:, h, d], .5)
                for h, head in enumerate(HEADS) for d, distance in enumerate(("near", "far"))
            }
            query_scores = 1.0 - prediction["counts"][:, :, 0]
            query_labels = target["counts"] > 0
            query_confusions = {}
            for h, head in enumerate(HEADS):
                for d, distance in enumerate(("near", "far")):
                    cells = slice(h * 6 + d * 3, h * 6 + (d + 1) * 3)
                    query_confusions[f"{head}_{distance}"] = confusion(
                        query_scores[:, cells].reshape(-1), query_labels[:, cells].reshape(-1), .5)
            wrong = confusion(range_scores[near_only_head, 1, 1],
                              np.zeros(int(near_only_head.sum()), dtype=bool), .5)
            body_fp = int((decisions[:, 0] & head_only & ~labels[:, 0]).sum())
            assert body_fp == saved_result['arms'][arm][role]['conditions']['HEAD_ONLY']['false_positives'][0]
            group_summary = groups(decisions, labels, record["records"])
            assert group_summary["correct"] == saved_result["arms"][arm][role]["selected_groups"]["correct"]
            assert group_summary["total"] == saved_result["arms"][arm][role]["selected_groups"]["total"]
            summary[role]["arms"][arm] = dict(thresholds=thresholds.tolist(), heads=head_confusions,
                                                complete_groups=group_summary,
                                                native_query_cells_at_0_5=query_confusions,
                                                native_range_events_at_0_5=range_confusions,
                                                native_HEAD_near_only_wrong_far_at_0_5=wrong,
                                                HEAD_ONLY_BODY_FP=body_fp)
            wrong_far[role][arm] = wrong["FP"]
            head_only_body_fp[role][arm] = body_fp
            if role == "dev":
                selected = [select_threshold(prediction["near"][:, h], labels[:, h], min_count=48)
                            for h in range(2)]
                recomputed[arm] = selected
                assert np.allclose([item["threshold"] for item in selected], thresholds, rtol=0, atol=0)
                for h, name in enumerate(HEADS):
                    assert all(selected[h][key] == saved_selection[arm]["heads"][h][key]
                               for key in ("threshold", "TP", "FP", "FN", "TN"))
    candidate = summary["eval"]["arms"]["NEW"]
    gates = dict(
        HEAD_near_TP_at_least_294=candidate["native_query_cells_at_0_5"]["HEAD_near"]["TP"] >= 294,
        native_HEAD_near_only_wrong_far_lower_than_baseline=wrong_far["eval"]["NEW"] < wrong_far["eval"]["OLD"],
        HEAD_ONLY_BODY_FP_below_38=head_only_body_fp["eval"]["NEW"] < 38,
        complete_groups_at_least_508=candidate["complete_groups"]["correct"] >= 508,
        BODY_TP_at_least_1187=candidate["heads"]["BODY"]["TP"] >= 1187,
        BODY_FP_at_most_69=candidate["heads"]["BODY"]["FP"] <= 69,
        HEAD_TP_at_least_1150=candidate["heads"]["HEAD"]["TP"] >= 1150,
        HEAD_FP_at_most_35=candidate["heads"]["HEAD"]["FP"] <= 35,
    )
    summary["dev_threshold_recomputation"] = recomputed
    summary["gates"] = dict(values=gates, all_pass=all(gates.values()))
    summary["invariants"] = dict(protocol_sha256=sha(run / "protocol.json"), fit_sha256=sha(run / "fit-complete.json"),
                                 receipt_sha256=sha(run / "receipt.json"), trained_10k_sha256=TRAINED_10K_SHA,
                                 train_source_sha256=sha(Path(__file__).with_name("body_query_10000_readout_train.py")),
                                 checkpoint_sha256=fit["checkpoint_sha256"])
    write(output / "readout-analysis.json", summary)
    write(output / "validation.json", dict(status="PASS", no_training_or_inference=True, fits=0, optimizer_steps=0,
          checks=["Trained-10k state, schedule, source and frozen-state receipts", "DEV threshold recomputation",
                  "All split head confusions and complete-group totals", "Three-cell count-distribution convolution for native near-only HEAD wrong-far events"],
          limits="Consumed Development and one readout-only adaptation; any gain is not unique causal proof."))
    print(summary["gates"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_analysis(args.cache, args.run, args.output or args.run)
