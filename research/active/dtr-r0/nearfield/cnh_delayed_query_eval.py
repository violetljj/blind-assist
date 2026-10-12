"""Evaluate delayed-query raw scores on the consumed aligned Development cohort.

No model inference, sampling, protected access or promotion. Each candidate uses
one shared HEAD/BODY threshold selected on this same cohort's joint-clear slots.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_fusion_probe_dev as F


SHAPE = (492, 4, 13, 2)
HEIGHTS = ("HEAD", "BODY")
FUSION_POINT = "k5_unbounded_below"


def independent_check(flags, category, summary, timely, baseline=None, paired=None):
    """Separate numpy formulas for exposure costs and paired transition counts."""
    clear = np.flatnonzero(np.all(category == "clear", axis=1))
    joint = np.max(flags[clear].astype(np.int8), axis=3)
    starts = np.diff(np.pad(joint, ((0, 0), (0, 0), (1, 0))), axis=2) == 1
    expected = {
        "clear_slots": int(np.count_nonzero(joint)),
        "clear_denominator": int(joint.size),
        "clear_segments": int(np.count_nonzero(starts)),
        "clear_clips": int(np.count_nonzero(np.max(joint, axis=2))),
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(("Independent clear-cost mismatch", key, summary[key], value))
    independent_timely = (np.sum(flags[:, :, :11, :], axis=2) > 0)
    independent_timely &= (category == "contact")[:, None, :]
    np.testing.assert_array_equal(timely, independent_timely)
    if baseline is not None:
        for q, comparison in enumerate(paired):
            mask = category[:, q] == "contact"
            before = baseline[mask, :, q].astype(np.int8)
            after = independent_timely[mask, :, q].astype(np.int8)
            delta = after - before
            check = dict(denominator=delta.size, baseline=int(before.sum()),
                         candidate=int(after.sum()), gain=int((delta == 1).sum()),
                         loss=int((delta == -1).sum()), net=int(delta.sum()))
            for key, value in check.items():
                if comparison[key] != value:
                    raise ValueError(("Independent paired mismatch", q, key))
    return expected


def grouped(flags, timely, baseline, category, rows):
    groups = {}
    for family in S.FAMILIES:
        groups[family] = np.array([r["family"] == family for r in rows])
    groups["horizontal_dark4"] = np.array([
        r["family"] == "horizontal" and r["rho"] == .25
        and "thick0.04" in r["variant"] for r in rows])
    result = {}
    for name, mask in groups.items():
        summary, subgroup_timely = F.metrics(flags[mask], category[mask])
        np.testing.assert_array_equal(subgroup_timely, timely[mask])
        result[name] = dict(metrics=summary,
            paired_minus_M3=F.compare(baseline[mask], timely[mask], category[mask],
                                     np.flatnonzero(mask)))
    for q in (0, 1):
        if sum(result[f]["paired_minus_M3"][q]["denominator"]
               for f in S.FAMILIES) != 688:
            raise ValueError("Family contact denominators do not conserve full batch")
    return result


def first_frame(flags, i, k, q, end=13):
    positions = np.flatnonzero(flags[i, k, :end, q])
    return int(positions[0] + 3) if positions.size else ""


def evaluate(output, scores_path):
    began = time.monotonic()
    output = Path(output).resolve()
    scores_path = Path(scores_path).resolve()
    artifact_root = (B.ROOT / "artifacts.local").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("Evaluation output must remain under canonical artifacts.local")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "metrics.json", output / "event_ledger.csv"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Preserve prior evaluation outputs; choose a new evaluation directory")

    plan_path = S.OUT / "PLAN.json"
    geometry_path = S.OUT / "geometry.npz"
    physical_path = S.OUT / "physical.npz"
    evaluated_path = S.OUT / "evaluated.npz"
    fusion_path = F.OUT / "scores.npz"
    rows = json.loads(plan_path.read_text(encoding="utf-8"))["scene_rows"]
    if len(rows) != 492 or [r["id"] for r in rows] != list(range(492)):
        raise ValueError("Full original aligned scene ordering required")
    with np.load(geometry_path, allow_pickle=False) as data:
        category = data["category"]
    if category.shape != (492, 2) or not np.isin(category, ["contact", "pass", "clear"]).all():
        raise ValueError("Original full scene/query categories required")
    if not np.array_equal((category == "contact").sum(0) * 4, [688, 688]):
        raise ValueError("Frozen per-height contact denominators changed")
    with np.load(physical_path, allow_pickle=False) as data:
        m3 = F.smooth(data["raw"])
    with np.load(evaluated_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(category, data["category"])
        np.testing.assert_allclose(m3, data["scores"], rtol=1e-13, atol=1e-13)
    if m3.shape != SHAPE or not np.isfinite(m3).all():
        raise ValueError("Finite complete M3 raw scores required")
    m3_flags = m3 >= B.THETA
    m3_summary, m3_timely = F.metrics(m3_flags, category)
    if (m3_summary["counts"] != [517, 406] or m3_summary["clear_slots"] != 46
            or m3_summary["clear_denominator"] != 4576):
        raise ValueError("Frozen M3 baseline parity failed")
    independent_check(m3_flags, category, m3_summary, m3_timely)

    with np.load(fusion_path, allow_pickle=False) as data:
        names = data["workpoints"].tolist()
        if names.count(FUSION_POINT) != 1:
            raise ValueError("Unique frozen 5-slot fusion point required")
        fusion_flags = data["fusion_alarm"][names.index(FUSION_POINT)]
        np.testing.assert_array_equal(data["original_M3_alarm"], m3_flags)
    if fusion_flags.shape != SHAPE or fusion_flags.dtype != np.bool_:
        raise ValueError("Complete frozen fusion alarm cache required")
    fusion_summary, fusion_timely = F.metrics(fusion_flags, category)
    fusion_paired = F.compare(m3_timely, fusion_timely, category)
    independent_check(fusion_flags, category, fusion_summary, fusion_timely,
                      m3_timely, fusion_paired)
    if fusion_summary["counts"] != [543, 461] or fusion_summary["clear_slots"] != 46:
        raise ValueError("Frozen 5-slot fusion parity failed")

    with np.load(scores_path, allow_pickle=False) as data:
        arm_names = data["arm_names"]
        raw = data["raw"]
    if (arm_names.ndim != 1 or arm_names.dtype.kind != "U" or not len(arm_names)
            or len(set(arm_names.tolist())) != len(arm_names)
            or any(not n or n in ("M3", "frozen_5slot_fusion") for n in arm_names)):
        raise ValueError("Unique nonempty Unicode arm_names required; baseline names reserved")
    if raw.shape != (len(arm_names), *SHAPE) or not np.isfinite(raw).all():
        raise ValueError("Finite raw scores [A,492,4,13,2] required")

    result = dict(status="COMPLETE", baseline=dict(threshold=B.THETA, metrics=m3_summary),
        frozen_5slot_fusion=dict(workpoint=FUSION_POINT, metrics=fusion_summary,
            paired_minus_M3=fusion_paired), arms={}, descriptive_contrasts={},
        cohort=dict(scenes=492, replicas=4, output_frames=list(range(3, 16)),
            timely_frames=list(range(3, 14)), contact_events_per_height=688,
            joint_clear_slots=4576, joint_clear_clips=352),
        evidence_scope="Consumed simulated Development; candidate thresholds selected on the same full cohort. "
            "One training seed per new arm; no unseen-scene, hardware, population or safety/generalization claim. "
            "K4 are correlated scene noise repeats. M3/fixed fusion remain retained baselines.",
        cost_rule="Shared HEAD/BODY threshold, whole-score ties, nearest 46 joint-clear slots; "
            "higher threshold on equal absolute residual. Nonzero residual is not equal cost. "
            "Segments, clips and pass are reported independently.",
        inputs=dict(scores=str(scores_path), shapes_plan=B.logical_path(plan_path),
            category=B.logical_path(geometry_path), m3_raw=B.logical_path(physical_path),
            m3_evaluated=B.logical_path(evaluated_path), fusion=B.logical_path(fusion_path)),
        score_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        source_sha256=B.sha(__file__), independent_checks="PASS")
    clear = (category == "clear").all(1)
    flags_by_arm = {"M3": m3_flags, "frozen_5slot_fusion": fusion_flags}
    timely_by_arm = {"M3": m3_timely, "frozen_5slot_fusion": fusion_timely}
    for arm, values in zip(arm_names.tolist(), raw):
        smoothed = F.smooth(values)
        threshold, cost = F.nearest(smoothed[clear].max(-1), 46)
        flags = smoothed >= threshold
        summary, timely = F.metrics(flags, category)
        paired = F.compare(m3_timely, timely, category)
        independent_check(flags, category, summary, timely, m3_timely, paired)
        if summary["clear_slots"] != cost:
            raise ValueError("Calibrated joint-clear cost mismatch")
        flags_by_arm[arm] = flags
        timely_by_arm[arm] = timely
        result["arms"][arm] = dict(threshold=float(threshold) if np.isfinite(threshold) else None,
            threshold_is_positive_infinity=bool(np.isposinf(threshold)), metrics=summary,
            clear_cost_residual=cost - 46, exactly_matched_clear_cost=cost == 46,
            paired_minus_M3=paired,
            paired_minus_frozen_5slot_fusion=F.compare(fusion_timely, timely, category),
            groups=grouped(flags, timely, m3_timely, category, rows))

    for before, after in (("center_bce", "extent_bce"), ("center_pair", "extent_pair"),
                          ("center_bce", "center_pair"), ("extent_bce", "extent_pair")):
        if before in timely_by_arm and after in timely_by_arm:
            result["descriptive_contrasts"][after + "_minus_" + before] = dict(
                paired=F.compare(timely_by_arm[before], timely_by_arm[after], category),
                scope="Descriptive comparison at each arm's selected clear cost; no causal/statistical effect claim")

    fields = ["arm", "scene", "replica", "height", "family", "placement", "context", "variant",
              "rho", "horizontal_dark4", "M3_timely", "fusion_timely", "candidate_timely",
              "gain_vs_M3", "loss_vs_M3", "first_alarm_frame", "first_timely_alarm_frame"]
    ledger_rows = 0
    with targets[1].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm, flags in flags_by_arm.items():
            timely = timely_by_arm[arm]
            for i, k, q in np.argwhere(np.broadcast_to((category == "contact")[:, None, :], timely.shape)):
                r = rows[i]
                before, after = bool(m3_timely[i, k, q]), bool(timely[i, k, q])
                writer.writerow(dict(arm=arm, scene=int(i), replica=int(k), height=HEIGHTS[q],
                    family=r["family"], placement=r["placement"], context=r["context"],
                    variant=r["variant"], rho=r["rho"],
                    horizontal_dark4=int(r["family"] == "horizontal" and r["rho"] == .25
                                         and "thick0.04" in r["variant"]),
                    M3_timely=int(before), fusion_timely=int(fusion_timely[i, k, q]),
                    candidate_timely=int(after), gain_vs_M3=int(not before and after),
                    loss_vs_M3=int(before and not after), first_alarm_frame=first_frame(flags, i, k, q),
                    first_timely_alarm_frame=first_frame(flags, i, k, q, end=11)))
                ledger_rows += 1
    if ledger_rows != (len(arm_names) + 2) * 1376:
        raise ValueError("Full event ledger denominator mismatch")
    result["event_ledger_rows"] = ledger_rows
    result["seconds"] = time.monotonic() - began
    B.save(targets[0], result)
    print(json.dumps(dict(status="COMPLETE", arms=list(result["arms"]),
                          event_rows=ledger_rows, seconds=result["seconds"])))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.output, args.scores)
