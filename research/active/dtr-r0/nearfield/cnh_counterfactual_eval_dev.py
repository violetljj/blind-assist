"""Fixed-calibration evaluation for three-arm counterfactual-generation Explore.

Manifest: arm_names, seed_ids, ordinary_arm and datasets list. Each dataset has
split (cal/validation), branch (ideal/yaw_plus3), physics, scene_rows, baseline,
and scores: [{arm,seed,path}]. Job NPZs contain raw[N,K,13,2]; baseline NPZ has
m3_raw/local_raw with the same shape. Optional combined_scores uses
arm_names/seed_ids/raw[A,S,N,K,13,2]. All paths are relative to the manifest.
Only ideal calibration selects candidate thresholds. No validation retuning,
all-seed success gate, model selection, inference or training is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import cnh_bar_fusion_probe_dev as F

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts.local/work/cnh-counterfactual-dev-20261009"
M3_THETA = .8557642486787612
OLD_RAISED = .9404184587540165
OLD_LOCAL = 4.625390338985158
HEIGHTS = ("HEAD", "BODY")


def smooth(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-2:] != (13, 2):
        raise ValueError("Exactly output frames3..15 and two queries required")
    return F.smooth(raw)


def old_fusion(m3, local):
    return (m3 >= OLD_RAISED) | (local >= OLD_LOCAL)


def summary(flags, category):
    metrics, timely = F.metrics(flags, category)
    contact = category == "contact"
    clear = (category == "clear").all(1)
    passed = (category == "pass").any(1) & ~contact.any(1)
    replicas = flags.shape[1]
    metrics.update(contact_event_denominators=(contact.sum(0) * replicas).tolist(),
        physical_contact_denominator=int(contact.any(1).sum() * replicas),
        clear_clip_denominator=int(clear.sum() * replicas),
        pass_clip_denominator=int(passed.sum() * replicas))
    # Independent costs from integer transitions, separate from F.metrics.
    cc = np.max(flags[clear].astype(np.int8), axis=-1)
    expected = dict(clear_slots=int(np.count_nonzero(cc)), clear_denominator=cc.size,
        clear_segments=int((np.diff(np.pad(cc, ((0, 0), (0, 0), (1, 0))), axis=2) == 1).sum()),
        clear_clips=int(np.count_nonzero(cc.max(2))) if len(cc) else 0)
    for key, value in expected.items():
        if metrics[key] != value:
            raise ValueError(("Independent clear-cost mismatch", key))
    independent = (flags[:, :, :11].sum(2) > 0) & contact[:, None, :]
    np.testing.assert_array_equal(independent, timely)
    return metrics, timely


def paired(before, after, category, scene_ids):
    result = F.compare(before, after, category, scene_ids)
    for q, row in enumerate(result):
        keep = category[:, q] == "contact"
        a, b = before[keep, :, q].astype(np.int8), after[keep, :, q].astype(np.int8)
        difference = b - a
        expected = dict(denominator=difference.size, baseline=int(a.sum()),
            candidate=int(b.sum()), gain=int((difference == 1).sum()),
            loss=int((difference == -1).sum()), net=int(difference.sum()))
        for key, value in expected.items():
            if row[key] != value:
                raise ValueError(("Independent paired mismatch", q, key))
    return result


def calibrate(m3, local, candidate, category):
    clear = (category == "clear").all(1)
    if not clear.any():
        raise ValueError("NOT_EVALUABLE: no joint-clear calibration exposure")
    reference = int((m3[clear] >= M3_THETA).any(-1).sum())
    theta, cost = F.nearest(candidate[clear].max(-1), reference)
    old = old_fusion(m3, local)[clear].any(-1)
    eligible = candidate[clear].max(-1)[~old]
    if eligible.size:
        maximum = float(eligible.max())
        addition = float(np.nextafter(maximum, np.inf))
        if not np.isfinite(addition):
            raise ValueError("Finite nextafter threshold required")
        or_status = "CALIBRATED_ZERO_EXTRA_CAL_CLEAR"
    else:
        maximum, addition = None, np.inf
        or_status = "NOT_EVALUABLE_NO_UNALARMED_CAL_CLEAR_SLOTS"
    return dict(standalone=record_threshold(theta), standalone_target_clear_slots=reference,
        standalone_actual_clear_slots=cost, standalone_clear_cost_residual=cost-reference,
        shared_query_threshold=True, calibration_clear_denominator=int(candidate[clear].shape[0]
            * candidate.shape[1] * 13), old_fusion_cal_clear_slots=int(old.sum()),
        old_fusion_plus_candidate=record_threshold(addition), zero_extra_status=or_status,
        eligible_clear_slots=int(eligible.size), maximum_eligible_clear_score=maximum)


def record_threshold(value):
    return dict(value=float(value) if np.isfinite(value) else None,
                positive_infinity=bool(np.isposinf(value)))


def threshold(record):
    value = np.inf if record["positive_infinity"] else record["value"]
    if value is None:
        raise ValueError("Missing threshold")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def split_name(value):
    return "cal" if value == "calibration" else value


def branch_name(value):
    return "yaw_plus3" if value in ("yaw+3", "yaw3", "yaw_plus3") else value


def read_job(path, arm, seed, split, branch):
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw"]
        for key, expected in (("arm", arm), ("seed", seed)):
            if key in data and data[key].item() != expected:
                raise ValueError(("Job identity differs", path, key))
        for key, expected, normalize in (("split", split_name(split), split_name),
                                         ("branch", branch_name(branch), branch_name)):
            if key in data and normalize(data[key].item()) != expected:
                raise ValueError(("Job identity differs", path, key))
    return raw


def load_dataset(spec, base, arms, seeds):
    resolve = lambda name: (base / spec[name]).resolve()
    with np.load(resolve("physics"), allow_pickle=False) as data:
        category = data["category"]
        frame_category = data["frame_category"] if "frame_category" in data else None
        physics_ids = data["scene_ids"] if "scene_ids" in data else None
    rows = read_json(resolve("scene_rows"))
    if isinstance(rows, dict):
        options = [spec.get("rows_key"), "scene_rows", spec["split"], split_name(spec["split"])]
        if split_name(spec["split"]) == "cal":
            options.append("calibration")
        key = next((key for key in options if key is not None and key in rows), None)
        if key is None:
            raise ValueError("No matching split in scene_rows metadata")
        rows = rows[key]
    n = len(rows)
    if category.shape != (n, 2) or not np.isin(category, ["contact", "pass", "clear"]).all():
        raise ValueError("Evaluator categories [N,2] required; truth must be constant over retained frames")
    if frame_category is not None:
        np.testing.assert_array_equal(frame_category, np.broadcast_to(category[:, None, :], (n, 13, 2)))
    ids = np.array([r["scene_id"] if "scene_id" in r else r["id"] for r in rows], dtype=np.int64)
    if physics_ids is not None:
        np.testing.assert_array_equal(ids, physics_ids)
    if len(set(ids.tolist())) != n:
        raise ValueError("Unique scene IDs required")
    families = np.array([r["family"] if "family" in r else r["shape_family"] for r in rows], dtype=str)
    with np.load(resolve("baseline"), allow_pickle=False) as data:
        m3, local = data["m3_raw"], data["local_raw"]
    shape = m3.shape
    if (m3.ndim != 4 or shape[0] != n or shape[-2:] != (13, 2)
            or shape[1] < 1 or local.shape != shape or not np.isfinite(m3).all()
            or np.isnan(local).any() or np.isposinf(local).any()):
        raise ValueError("Finite M3 and local scores [N,K,13,2] required; local -inf is allowed for unobserved patches")
    if "combined_scores" in spec:
        with np.load(resolve("combined_scores"), allow_pickle=False) as data:
            if data["arm_names"].tolist() != arms or data["seed_ids"].tolist() != seeds:
                raise ValueError("Combined score arm/seed identities differ")
            raw = data["raw"]
    else:
        jobs = {}
        for entry in spec["scores"]:
            key = (entry["arm"], int(entry["seed"]))
            if key in jobs or key[0] not in arms or key[1] not in seeds:
                raise ValueError("Duplicate or unexpected score job")
            jobs[key] = read_job((base / entry["path"]).resolve(), *key, spec["split"], spec["branch"])
        expected = {(a, s) for a in arms for s in seeds}
        if set(jobs) != expected:
            raise ValueError("All declared paired arm/seed jobs required; no seed subset selection")
        raw = np.stack([np.stack([jobs[a, s] for s in seeds]) for a in arms])
    if raw.shape != (len(arms), len(seeds), *shape) or not np.isfinite(raw).all():
        raise ValueError("Complete finite candidate raw scores [A,S,N,K,13,2] required")
    result = dict(category=category, rows=rows, scene_ids=ids, families=families,
                  m3=smooth(m3), local=smooth(local), candidates=smooth(raw),
                  split=split_name(spec["split"]), branch=branch_name(spec["branch"]))
    result["groups"] = {"family:" + f: families == f for f in sorted(set(families.tolist()))}
    for field in ("background_family", "geometry_family"):
        if all(field in row for row in rows):
            values = np.array([str(row[field]) for row in rows])
            result["groups"].update({field + ":" + f: values == f for f in sorted(set(values.tolist()))})
    return result


def pose_identity(ideal, yaw):
    for key in ("category", "scene_ids", "families"):
        np.testing.assert_array_equal(ideal[key], yaw[key])
    if ideal["candidates"].shape != yaw["candidates"].shape or ideal["groups"].keys() != yaw["groups"].keys():
        raise ValueError("Yaw must preserve ideal cohort identity and score shape")
    for group in ideal["groups"]:
        np.testing.assert_array_equal(ideal["groups"][group], yaw["groups"][group])


def first_frame(flags, i, k, q, start=0, end=13):
    ix = np.flatnonzero(flags[i, k, start:end, q])
    return int(ix[0] + start + 3) if ix.size else ""


def describe(data, flags, before_flags, ordinary_flags, base_flags):
    category, ids = data["category"], data["scene_ids"]
    metrics, timely = summary(flags, category)
    references = {name: summary(value, category)[1] for name, value in before_flags.items()}
    references["ordinary_same_seed_same_policy"] = summary(ordinary_flags, category)[1]
    references["base_same_seed_same_policy"] = summary(base_flags, category)[1]
    comparisons = {name: paired(value, timely, category, ids) for name, value in references.items()}
    groups = {}
    for group, mask in data["groups"].items():
        gm, gt = summary(flags[mask], category[mask])
        groups[group] = dict(metrics=gm,
            paired={name: paired(value[mask], gt, category[mask], ids[mask])
                    for name, value in references.items()})
    return dict(metrics=metrics, paired=comparisons, groups=groups), timely, references


def scalar_row(split_branch, arm, seed, policy, stratum, report):
    metrics = report["metrics"]
    row = dict(dataset=split_branch, arm=arm, seed=int(seed), policy=policy, stratum=stratum,
        HEAD_timely=metrics["counts"][0], BODY_timely=metrics["counts"][1],
        HEAD_denominator=metrics["contact_event_denominators"][0],
        BODY_denominator=metrics["contact_event_denominators"][1],
        HEAD_late=metrics["late_counts"][0], BODY_late=metrics["late_counts"][1],
        clear_slots=metrics["clear_slots"], clear_denominator=metrics["clear_denominator"],
        clear_segments=metrics["clear_segments"], clear_clips=metrics["clear_clips"],
        clear_clip_denominator=metrics["clear_clip_denominator"],
        pass_clips=metrics["pass_clips"], pass_clip_denominator=metrics["pass_clip_denominator"],
        physical_contact_any_height=metrics["physical_contact_any_height"],
        physical_contact_denominator=metrics["physical_contact_denominator"])
    for name, comparisons in report["paired"].items():
        for height, comparison in zip(HEIGHTS, comparisons):
            for field in ("gain", "loss", "net"):
                row[height + "_" + field + "_vs_" + name] = comparison[field]
    return row


def evaluate(manifest_path, output=OUT / "evaluation"):
    began = time.monotonic()
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if not output.is_relative_to((ROOT / "artifacts.local").resolve()):
        raise ValueError("Output must remain under canonical artifacts.local")
    targets = [output / n for n in ("thresholds.json", "metrics.json", "event_ledger.csv", "summary.csv")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Preserve existing evaluation; use a new output directory")
    manifest = read_json(manifest_path)
    arms, seeds = manifest["arm_names"], [int(s) for s in manifest["seed_ids"]]
    ordinary = manifest.get("ordinary_arm", "ordinary")
    base_arm = manifest.get("base_arm", "base")
    if (not arms or len(set(arms)) != len(arms) or ordinary not in arms
            or base_arm not in arms
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("Unique arm/seed identities and ordinary control required")
    datasets = {}
    for spec in manifest["datasets"]:
        key = (split_name(spec["split"]), branch_name(spec["branch"]))
        if key in datasets or key[0] not in ("cal", "validation") or key[1] not in ("ideal", "yaw_plus3"):
            raise ValueError("Unique cal/validation x ideal/yaw_plus3 datasets required")
        datasets[key] = load_dataset(spec, manifest_path.parent, arms, seeds)
    if ("cal", "ideal") not in datasets or ("validation", "ideal") not in datasets:
        raise ValueError("Ideal calibration and validation are required")
    if manifest.get("require_disjoint_background_families", True):
        cal_rows = datasets["cal", "ideal"]["rows"]
        validation_rows = datasets["validation", "ideal"]["rows"]
        if all("background_family" in r for r in cal_rows + validation_rows):
            calibration_families = {r["background_family"] for r in cal_rows}
            validation_families = {r["background_family"] for r in validation_rows}
            if calibration_families & validation_families:
                raise ValueError("Calibration and validation background families overlap")
    for split in ("cal", "validation"):
        if (split, "yaw_plus3") in datasets:
            pose_identity(datasets[split, "ideal"], datasets[split, "yaw_plus3"])
    cal = datasets["cal", "ideal"]
    calibrations = {a: {str(seed): calibrate(cal["m3"], cal["local"],
        cal["candidates"][ai, si], cal["category"]) for si, seed in enumerate(seeds)}
        for ai, a in enumerate(arms)}
    result = dict(status="COMPLETE", arm_names=arms, seed_ids=seeds, datasets={},
        baseline_thresholds=dict(M3=M3_THETA, old_fusion_M3=OLD_RAISED, old_fusion_local=OLD_LOCAL),
        calibration_scope="Only ideal calibration. Standalone sharedH/B whole-tie threshold matches actual frozenM3 calibration clear rate; "
            "OR retains every fixed old-fusion alarm and nextafter threshold prevents additional calibration joint-clear slots. "
            "Validation and yaw reuse these ideal thresholds and may incur higher costs.",
        interpretation="Three paired training seeds are all reported, with no all-seed success gate. "
            "Ordinary BCE does not consume pair identity: CF-versus-ordinary evaluates counterfactual-generated training distribution, "
            "not a pair-aware objective. New simulator/background-family validation is Development, not hardware or safety confirmation.",
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        inference=0, training=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    fields = ["split", "branch", "arm", "seed", "policy", "scene", "replica", "height", "family",
        "category", "timely", "late_only", "first_alarm_frame", "first_timely_frame", "first_late_frame"]
    for reference in ("M3", "old_local_fusion", "ordinary", "base"):
        fields += [reference + "_timely", "gain_vs_" + reference, "loss_vs_" + reference,
                   reference + "_first_alarm_frame", reference + "_first_timely_frame"]
    count = 0
    with targets[2].open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (split, branch), data in datasets.items():
            category = data["category"]
            before = {"M3": data["m3"] >= M3_THETA,
                      "old_local_fusion": old_fusion(data["m3"], data["local"])}
            dataset_report = dict(scene_count=len(category), replicas=data["m3"].shape[1],
                baseline={name: summary(value, category)[0] for name, value in before.items()},
                baseline_groups={group: {name: summary(value[mask], category[mask])[0]
                    for name, value in before.items()} for group, mask in data["groups"].items()}, arms={})
            result["datasets"][split + "/" + branch] = dataset_report
            ordinary_index = arms.index(ordinary)
            base_index = arms.index(base_arm)
            for ai, arm in enumerate(arms):
                dataset_report["arms"][arm] = {}
                for si, seed in enumerate(seeds):
                    own, control = calibrations[arm][str(seed)], calibrations[ordinary][str(seed)]
                    base_control = calibrations[base_arm][str(seed)]
                    candidate = data["candidates"][ai, si]
                    ordinary_score = data["candidates"][ordinary_index, si]
                    base_score = data["candidates"][base_index, si]
                    reports = {}
                    for policy in ("standalone", "old_fusion_plus_candidate"):
                        theta, ordinary_theta = threshold(own[policy]), threshold(control[policy])
                        flags, control_flags = candidate >= theta, ordinary_score >= ordinary_theta
                        base_flags = base_score >= threshold(base_control[policy])
                        if policy == "old_fusion_plus_candidate":
                            flags |= before["old_local_fusion"]
                            control_flags |= before["old_local_fusion"]
                            base_flags |= before["old_local_fusion"]
                            if not np.all(~before["old_local_fusion"] | flags):
                                raise ValueError("OR lost a baseline alarm")
                        report, timely, references = describe(data, flags, before, control_flags, base_flags)
                        reference_flags = dict(before, ordinary=control_flags, base=base_flags)
                        report["threshold"] = own[policy]
                        report["threshold_origin"] = "ideal calibration fixed for every evaluation branch"
                        if policy == "old_fusion_plus_candidate":
                            report["calibration_status"] = own["zero_extra_status"]
                            if split == "cal" and branch == "ideal":
                                if report["metrics"]["clear_slots"] != dataset_report["baseline"]["old_local_fusion"]["clear_slots"]:
                                    raise ValueError("OR calibration added clear slots")
                        reports[policy] = report
                        for i, k, q in np.argwhere(np.broadcast_to((category == "contact")[:, None, :], timely.shape)):
                            r = dict(split=split, branch=branch, arm=arm, seed=seed, policy=policy,
                                scene=int(data["scene_ids"][i]), replica=int(k), height=HEIGHTS[q],
                                family=data["families"][i], category=category[i, q], timely=int(timely[i, k, q]),
                                late_only=int(not timely[i, k, q] and flags[i, k, 11:, q].any()),
                                first_alarm_frame=first_frame(flags, i, k, q),
                                first_timely_frame=first_frame(flags, i, k, q, end=11),
                                first_late_frame=first_frame(flags, i, k, q, start=11))
                            for name, key in (("M3", "M3"), ("old_local_fusion", "old_local_fusion"),
                                              ("ordinary", "ordinary_same_seed_same_policy"),
                                              ("base", "base_same_seed_same_policy")):
                                old = bool(references[key][i, k, q])
                                new = bool(timely[i, k, q])
                                r[name + "_timely"] = int(old)
                                r["gain_vs_" + name] = int(not old and new)
                                r["loss_vs_" + name] = int(old and not new)
                                r[name + "_first_alarm_frame"] = first_frame(reference_flags[name], i, k, q)
                                r[name + "_first_timely_frame"] = first_frame(reference_flags[name], i, k, q, end=11)
                            writer.writerow(r);count += 1
                    dataset_report["arms"][arm][str(seed)] = reports
    result["event_ledger_rows"] = count
    result["seconds"] = time.monotonic() - began
    result["independent_numpy_checks"] = "PASS"
    scalars = []
    for split_branch, dataset_report in result["datasets"].items():
        for arm, seed_reports in dataset_report["arms"].items():
            for seed, policies in seed_reports.items():
                for policy, report in policies.items():
                    scalars.append(scalar_row(split_branch, arm, seed, policy, "all", report))
                    for group, group_report in report["groups"].items():
                        scalars.append(scalar_row(split_branch, arm, seed, policy, group, group_report))
    with targets[3].open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalars[0]))
        writer.writeheader();writer.writerows(scalars)
    result["scalar_summary_rows"] = len(scalars)
    with targets[0].open("x", encoding="utf8") as handle:
        json.dump(dict(status="COMPLETE", only_source="cal/ideal", arms=calibrations), handle,
                  ensure_ascii=False, indent=2, allow_nan=False)
    with targets[1].open("x", encoding="utf8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status="COMPLETE", datasets=list(result["datasets"]),
        paired_seeds=seeds, event_rows=count, seconds=result["seconds"])))
    return result


def check():
    # Fixtures test costs/denominators, tie behavior and transport of a cal threshold.
    category = np.array([["contact", "clear"], ["clear", "contact"],
                         ["clear", "clear"], ["pass", "clear"]])
    flags = np.zeros((4, 2, 13, 2), bool)
    flags[0, 0, 10, 0] = True
    flags[1, 1, 11, 1] = True
    flags[2, 0, [0, 1, 3], 0] = True
    flags[3, 0, 0, 1] = True
    m, t = summary(flags, category)
    assert m["counts"] == [1, 0] and m["late_counts"] == [0, 1]
    assert (m["clear_slots"], m["clear_segments"], m["clear_clips"]) == (3, 2, 1)
    assert m["contact_event_denominators"] == [2, 2] and m["pass_clip_denominator"] == 2
    p = paired(np.zeros_like(t), t, category, np.array([101, 102, 103, 104]))
    assert p[0]["gain_scene_keys"] == [[101, 0]] and p[1]["gain"] == 0
    theta, cost = F.nearest([1., 1., 2., 3.], 3)
    assert theta == 2. and cost == 2
    m3 = np.full((1, 1, 13, 2), -10.)
    local = np.full_like(m3, -10.)
    candidate = np.arange(26, dtype=float).reshape(1, 1, 13, 2)
    cal = calibrate(m3, local, candidate, np.array([["clear", "clear"]]))
    zero = threshold(cal["old_fusion_plus_candidate"])
    assert np.isfinite(zero) and zero == np.nextafter(25., np.inf)
    assert not (candidate >= zero).any()
    assert (candidate + 1 >= zero).any()  # Fixed validation can add clear cost.
    full = calibrate(m3 + 20, local, candidate, np.array([["clear", "clear"]]))
    assert full["zero_extra_status"].startswith("NOT_EVALUABLE")
    assert threshold(full["old_fusion_plus_candidate"]) == np.inf
    np.testing.assert_allclose(smooth(np.arange(26.).reshape(1, 1, 13, 2))[0, 0, 4],
        (np.arange(26.).reshape(13, 2)[:5] * np.array([1, 2, 4, 8, 16])[:, None]).sum(0) / 31)
    check_loader()
    print("PASS fixed-calibration/OR/ties/exposure/paired-ID/timely/late/causal-smoothing/job-loader fixtures")


def check_loader():
    """In-memory loader fixture: no experiment input or temporary artifact writes."""
    original_load, original_json = np.load, read_json
    category = np.array([["contact", "clear"], ["clear", "clear"]])
    ids = np.array([11, 12])
    raw = np.zeros((2, 1, 13, 2))
    arms, seeds = ["base", "ordinary", "counterfactual"], [42]
    rows = {"cal": [dict(scene_id=int(i), shape_family="rod", background_family="cal-bg") for i in ids]}

    class Archive(dict):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_load(path, **_):
        name = Path(path).name
        if name == "physics.npz":
            return Archive(category=category, scene_ids=ids,
                frame_category=np.broadcast_to(category[:, None], (2, 13, 2)))
        if name == "baseline.npz":
            return Archive(m3_raw=raw, local_raw=raw)
        arm = name.removesuffix(".npz")
        return Archive(raw=raw, arm=np.array(arm), seed=np.array(42),
                       split=np.array("cal"), branch=np.array("yaw+3"))

    spec = dict(split="cal", branch="yaw_plus3", physics="physics.npz",
        scene_rows="scene_rows.json", baseline="baseline.npz",
        scores=[dict(arm=arm, seed=42, path=arm+".npz") for arm in arms])
    try:
        np.load = fake_load
        globals()["read_json"] = lambda _: rows
        data = load_dataset(spec, Path.cwd(), arms, seeds)
        assert data["candidates"].shape == (3, 1, 2, 1, 13, 2)
        np.testing.assert_array_equal(data["scene_ids"], ids)
        assert set(data["groups"]) == {"family:rod", "background_family:cal-bg"}
        assert data["branch"] == "yaw_plus3"
        pose_identity(data, data)
        bad = dict(spec, scores=spec["scores"][:-1])
        try:
            load_dataset(bad, Path.cwd(), arms, seeds)
        except ValueError as error:
            assert "All declared paired" in str(error)
        else:
            raise AssertionError("Missing paired arm was accepted")
    finally:
        np.load = original_load
        globals()["read_json"] = original_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUT / "evaluation")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.manifest:
        evaluate(args.manifest, args.output)
    else:
        parser.error("--manifest or --check required")
