"""Frozen offline notification replay and paired scene-cluster metrics.

No data loading, fitting, threshold selection, inference or protected-data access.
Call summarize(grade, category, rows) and compare(base, candidate, category,
rows) separately for each fixed model seed. Grades are [scene, replica, time,
HEAD/BODY] integers 0/1/2; truth is evaluator-only [scene, HEAD/BODY]. All
replicas stay inside their originating scene in bootstrap resamples. These
offline clip-reset counts do not establish App dispatch, hardware or safety.

Default timely window is the first 11 outputs (nominal f3..13). Delay is in
nominal output frames; no unverified conversion to real time is performed.
Run this file with --self-test for independent synthetic fixtures only.
"""
import argparse

import numpy as np

HEIGHTS = ("HEAD", "BODY")
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_SAMPLES = 2000


def replay_gap1(grade):
    """Emit first positive immediately and only strong upgrades within an episode.

    One intervening zero keeps episode memory. Two consecutive zeros reset it.
    Each scene/replica/query starts with empty state. Return emissions with the
    same shape; simultaneous HEAD/BODY notifications are joined only for cost.
    """
    grade = np.asarray(grade)
    if grade.ndim != 4 or grade.shape[-1] != 2 or min(grade.shape[:3]) < 1:
        raise ValueError("Expected nonempty grade[N,K,T,2]")
    if not np.isin(grade, (0, 1, 2)).all():
        raise ValueError("Grades must be integer-valued 0/1/2")
    result = np.zeros(grade.shape, dtype=np.int8)
    peak = np.zeros(grade.shape[:2] + (2,), dtype=np.int8)
    quiet = np.zeros_like(peak)
    for t in range(grade.shape[2]):
        current = grade[:, :, t, :]
        positive = current > 0
        quiet = np.where(positive, 0, np.minimum(quiet + 1, 2))
        peak = np.where(quiet >= 2, 0, peak)
        emit = positive & (current > peak)
        result[:, :, t, :] = np.where(emit, current, 0)
        peak = np.where(positive, np.maximum(peak, current), peak)
    return result


def _validate(grade, category, scene_rows, timely_frames, frame_start):
    grade = np.asarray(grade)
    category = np.asarray(category, dtype=str)
    if grade.ndim != 4 or grade.shape[-1] != 2 or min(grade.shape[:3]) < 1:
        raise ValueError("Expected nonempty grade[N,K,T,2]")
    if not np.isin(grade, (0, 1, 2)).all():
        raise ValueError("Grades must be integer-valued 0/1/2")
    if category.shape != (grade.shape[0], 2):
        raise ValueError("Category must be [N,2], aligned with grade")
    if not np.isin(category, ("contact", "pass", "clear")).all():
        raise ValueError("Unknown truth is not negative; only contact/pass/clear admitted")
    if len(scene_rows) != grade.shape[0]:
        raise ValueError("One aligned evaluator row per scene required")
    if not isinstance(timely_frames, int) or not 1 <= timely_frames <= grade.shape[2]:
        raise ValueError("timely_frames must be an integer within the recorded window")
    if not isinstance(frame_start, int):
        raise ValueError("frame_start must be an integer nominal index")
    ids, families = [], []
    for row in scene_rows:
        scene_id = row.get("scene_uid", row.get("scene_id", row.get("id")))
        family = row.get("shape_family", row.get("family"))
        if scene_id is None or family is None:
            raise ValueError("Each row needs scene_uid/scene_id/id and shape_family/family")
        ids.append(str(scene_id))
        families.append(str(family))
    if len(set(ids)) != len(ids):
        raise ValueError("Unique scene IDs required; aggregate correlated captures before calling")
    return grade, category, ids, np.asarray(families)


def _first(emitted, minimum_grade=1):
    flags = emitted >= minimum_grade
    return np.where(flags.any(axis=2), flags.argmax(axis=2), -1)


def _stats(values):
    values = np.asarray(values, dtype=float)
    return dict(n=int(values.size), min=float(values.min()) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                max=float(values.max()) if values.size else None,
                sum=float(values.sum()))


def _contact_masks(category, families, scene_rows):
    contact = category == "contact"
    head = np.zeros_like(contact)
    body = np.zeros_like(contact)
    head[:, 0] = contact[:, 0]
    body[:, 1] = contact[:, 1]
    thin = np.asarray([row.get("size_variant") == 0 for row in scene_rows])
    return dict(HEAD=head, BODY=body, all_contact_queries=contact,
                HEAD_horizontal_thin=head & ((families == "horizontal") & thin)[:, None],
                HEAD_horizontal=head & (families == "horizontal")[:, None],
                HEAD_sign_edge=head & (families == "sign_edge")[:, None])


def _contact_counts(first, first_strong, mask, timely_frames):
    truth = np.broadcast_to(mask[:, None, :], first.shape)
    selected = first[truth]
    strong = first_strong[truth]
    return dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
                timely=int(((selected >= 0) & (selected < timely_frames)).sum()),
                timely_strong=int(((strong >= 0) & (strong < timely_frames)).sum()),
                late=int((selected >= timely_frames).sum()), silent=int((selected < 0).sum()),
                any_notification=int((selected >= 0).sum()),
                first_index_histogram={str(int(i)): int((selected == i).sum())
                                       for i in np.unique(selected)},
                first_index_stats=_stats(selected[selected >= 0]),
                first_strong_index_histogram={str(int(i)): int((strong == i).sum())
                                              for i in np.unique(strong)},
                first_strong_index_stats=_stats(strong[strong >= 0]))


def _cost_counts(values, scene_mask):
    """Values [N,K,T]: joint max of emissions or one particular query."""
    subset = values[scene_mask]
    positive = subset > 0
    return dict(scene_denominator=int(scene_mask.sum()),
                clip_denominator=int(subset.shape[0] * subset.shape[1]),
                frame_denominator=int(subset.size),
                notifications=int(positive.sum()),
                light_notifications=int((subset == 1).sum()),
                strong_notifications=int((subset == 2).sum()),
                clips_with_any_notification=int(positive.any(-1).sum()),
                clips_with_light_notification=int((subset == 1).any(-1).sum()),
                clips_with_strong_notification=int((subset == 2).any(-1).sum()))


def summarize(grade, category, scene_rows, timely_frames=11, frame_start=3):
    """Replay grades and return counted numerators, denominators and scene rows.

    Contact benefit requires an emission on the actual contact height. Cost for
    joint-clear (both clear) and pure-pass (any pass, neither contact) counts the
    maximum emitted grade per frame, so two same-frame prompts count once.
    Per-query costs are additionally reported and never summed as joint cost.
    """
    grade, category, ids, families = _validate(
        grade, category, scene_rows, timely_frames, frame_start)
    emitted = replay_gap1(grade)
    first, first_strong = _first(emitted), _first(emitted, 2)
    contact_masks = _contact_masks(category, families, scene_rows)
    cost_masks = dict(clear=(category == "clear").all(1),
                      purepass=(category == "pass").any(1) & ~(category == "contact").any(1))
    joint = emitted.max(-1)
    contacts = {name: _contact_counts(first, first_strong, mask, timely_frames)
                for name, mask in contact_masks.items()}
    costs = {name: _cost_counts(joint, mask) for name, mask in cost_masks.items()}
    query_costs = {height: {kind: _cost_counts(emitted[..., q], category[:, q] == kind)
                            for kind in ("clear", "pass")}
                   for q, height in enumerate(HEIGHTS)}
    # Physical clips count only notifications in an actually contacting query.
    qualifying = ((first >= 0) & (first < timely_frames)
                  & (category == "contact")[:, None, :]).any(-1)
    physical_mask = (category == "contact").any(1)
    physical = dict(denominator=int(physical_mask.sum() * grade.shape[1]),
                    scene_denominator=int(physical_mask.sum()), timely=int(qualifying.sum()))
    scenes = []
    for n, scene_id in enumerate(ids):
        scenes.append(dict(scene_id=scene_id, shape_family=str(families[n]),
            contact_queries={name: _contact_counts(first[n:n+1], first_strong[n:n+1],
                                                   mask[n:n+1], timely_frames)
                             for name, mask in contact_masks.items()},
            costs={name: _cost_counts(joint[n:n+1], mask[n:n+1])
                   for name, mask in cost_masks.items()}))
    return dict(contract=dict(shape=list(grade.shape), heights=list(HEIGHTS),
        timely_frames=timely_frames, frame_start=frame_start,
        timely_nominal_frames=[frame_start, frame_start + timely_frames - 1],
        first_index_unit="zero-based output frame; -1 is silent",
        notification="gap1 first positive; same episode strong upgrade only; two zeros reset",
        independent_unit="scene; K replicas and model seeds are not independent samples",
        primary="HEAD contact, horizontal, size_variant==0; all horizontal is secondary",
        joint_cost="maximum emitted HEAD/BODY grade per same frame; no grade-sum cost",
        scope="offline clip-reset controlled replay; no hardware/App/safety claim"),
        contacts=contacts, physical_contact_clips=physical,
        costs=costs, query_costs=query_costs, scene_summaries=scenes)


def _bootstrap_ratio_difference(scene_num, scene_den, seed, samples):
    """Percentile paired cluster CI for summed numerator / summed denominator."""
    scene_num, scene_den = np.asarray(scene_num), np.asarray(scene_den)
    eligible = scene_den > 0
    scene_num, scene_den = scene_num[eligible], scene_den[eligible]
    n = len(scene_den)
    if not n:
        return dict(estimate=None, ci95=None, independent_scenes=0,
                    status="NOT_EVALUABLE_NO_DENOMINATOR")
    estimate = float(scene_num.sum() / scene_den.sum())
    if n < 2:
        return dict(estimate=estimate, ci95=None, independent_scenes=n,
                    status="NOT_EVALUABLE_SINGLE_SCENE")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(samples, n))
    ratios = scene_num[draws].sum(1) / scene_den[draws].sum(1)
    return dict(estimate=estimate, ci95=np.quantile(ratios, [.025, .975]).tolist(),
                independent_scenes=n, status="DESCRIPTIVE_SCENE_CLUSTER_PERCENTILE")


def compare(baseline, candidate, category, scene_rows, timely_frames=11, frame_start=3,
            bootstrap_seed=BOOTSTRAP_SEED, bootstrap_samples=BOOTSTRAP_SAMPLES):
    """Paired contact rescue/loss/timing and notification-cost differences.

    No operating point is selected here. CI applies within one fixed model seed,
    resampling scene IDs with all K replicas together. Replicas are counted in
    event numerators/denominators but never become independent CI samples.
    """
    baseline, category, ids, families = _validate(
        baseline, category, scene_rows, timely_frames, frame_start)
    candidate, _, candidate_ids, _ = _validate(
        candidate, category, scene_rows, timely_frames, frame_start)
    if baseline.shape != candidate.shape or ids != candidate_ids:
        raise ValueError("Paired grades must have identical scene/replica/time/query shape")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("Positive integer bootstrap_samples required")
    before, after = replay_gap1(baseline), replay_gap1(candidate)
    a, b = _first(before), _first(after)
    ast, bst = _first(before, 2), _first(after, 2)
    contacts, scene_contacts = {}, {scene_id: {} for scene_id in ids}
    for name, mask in _contact_masks(category, families, scene_rows).items():
        truth = np.broadcast_to(mask[:, None, :], a.shape)
        at = truth & (a >= 0) & (a < timely_frames)
        bt = truth & (b >= 0) & (b < timely_frames)
        common = truth & (a >= 0) & (b >= 0)
        common_timely = at & bt
        delta = b - a
        gain, loss = (~at & bt), (at & ~bt)
        denom = truth.sum(axis=(1, 2))
        net = gain.sum(axis=(1, 2)) - loss.sum(axis=(1, 2))
        row = dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
            baseline_timely=int(at.sum()), candidate_timely=int(bt.sum()),
            rescue=int(gain.sum()), loss=int(loss.sum()), net=int(net.sum()),
            baseline_timely_strong=int((truth & (ast >= 0) & (ast < timely_frames)).sum()),
            candidate_timely_strong=int((truth & (bst >= 0) & (bst < timely_frames)).sum()),
            common_any_notified=int(common.sum()), common_timely=int(common_timely.sum()),
            earlier_common=int((common & (delta < 0)).sum()),
            later_common=int((common & (delta > 0)).sum()),
            unchanged_common=int((common & (delta == 0)).sum()),
            delta_frames_common=_stats(delta[common]),
            delta_frames_common_timely=_stats(delta[common_timely]),
            timely_rate_difference_ci=_bootstrap_ratio_difference(net, denom,
                                                                  bootstrap_seed, bootstrap_samples))
        contacts[name] = row
        for n, scene_id in enumerate(ids):
            scene_contacts[scene_id][name] = dict(denominator=int(denom[n]),
                baseline_timely=int(at[n].sum()), candidate_timely=int(bt[n].sum()),
                rescue=int(gain[n].sum()), loss=int(loss[n].sum()), net=int(net[n]))
    cost_masks = dict(clear=(category == "clear").all(1),
                      purepass=(category == "pass").any(1) & ~(category == "contact").any(1))
    aj, bj = before.max(-1), after.max(-1)
    costs, scene_costs = {}, {scene_id: {} for scene_id in ids}
    for name, mask in cost_masks.items():
        den = mask.astype(np.int64) * baseline.shape[1]
        summaries = {}
        for metric, va, vb in (
            ("notifications", (aj > 0).sum((1, 2)), (bj > 0).sum((1, 2))),
            ("light_notifications", (aj == 1).sum((1, 2)), (bj == 1).sum((1, 2))),
            ("strong_notifications", (aj == 2).sum((1, 2)), (bj == 2).sum((1, 2))),
            ("clips_with_any_notification", (aj > 0).any(2).sum(1), (bj > 0).any(2).sum(1)),
            ("clips_with_light_notification", (aj == 1).any(2).sum(1), (bj == 1).any(2).sum(1)),
            ("clips_with_strong_notification", (aj == 2).any(2).sum(1), (bj == 2).any(2).sum(1))):
            va, vb = va * mask, vb * mask
            summaries[metric] = dict(baseline=int(va.sum()), candidate=int(vb.sum()),
                delta=int((vb - va).sum()), per_clip_difference_ci=
                _bootstrap_ratio_difference(vb - va, den, bootstrap_seed, bootstrap_samples))
            for n, scene_id in enumerate(ids):
                scene_costs[scene_id].setdefault(name, dict(clip_denominator=int(den[n])))
                scene_costs[scene_id][name][metric] = dict(baseline=int(va[n]),
                    candidate=int(vb[n]), delta=int(vb[n] - va[n]))
        costs[name] = dict(scene_denominator=int(mask.sum()), clip_denominator=int(den.sum()),
                           **summaries)
    return dict(contacts=contacts, costs=costs,
        bootstrap=dict(seed=int(bootstrap_seed), samples=bootstrap_samples, confidence=.95,
                       unit="paired scene; all K replicas retained; each model seed separate",
                       interpretation="descriptive controlled fresh-simulation uncertainty"),
        scene_summaries=[dict(scene_id=i, contacts=scene_contacts[i], costs=scene_costs[i]) for i in ids])


def self_test():
    # Hand-written expected notifications, independent of implementation state.
    sequences = [([1, 0, 1, 2, 1, 0, 0, 1], [1, 0, 0, 2, 0, 0, 0, 1]),
                 ([2, 0, 2, 1, 0, 0, 2, 2], [2, 0, 0, 0, 0, 0, 2, 0]),
                 ([0, 1, 1, 0, 2, 0, 0, 2], [0, 1, 0, 0, 2, 0, 0, 2])]
    for source, expected in sequences:
        grade = np.zeros((1, 1, len(source), 2), np.int8)
        grade[0, 0, :, 0] = source
        emitted = replay_gap1(grade)
        np.testing.assert_array_equal(emitted[0, 0, :, 0], expected)
        assert _first(grade)[0, 0, 0] == _first(emitted)[0, 0, 0]
        assert _first(grade, 2)[0, 0, 0] == _first(emitted, 2)[0, 0, 0]
    grade = np.zeros((3, 1, 5, 2), np.int8)
    grade[0, 0, 0, 1] = 2  # BODY false prompt on actual HEAD contact.
    grade[1, 0, 0, :] = [1, 2]  # Same-frame two-query cost is one strong.
    grade[2, 0, 0, :] = [1, 1]
    category = np.asarray([["contact", "clear"], ["clear", "clear"], ["pass", "clear"]])
    rows = [dict(scene_id=i, shape_family="horizontal", size_variant=0) for i in range(3)]
    result = summarize(grade, category, rows, timely_frames=3)
    assert result["contacts"]["HEAD_horizontal"]["timely"] == 0
    assert result["physical_contact_clips"]["timely"] == 0
    assert result["costs"]["clear"]["notifications"] == 1
    assert result["costs"]["clear"]["strong_notifications"] == 1
    assert result["costs"]["clear"]["light_notifications"] == 0
    assert result["costs"]["purepass"]["light_notifications"] == 1
    candidate = grade.copy()
    paired = compare(np.zeros_like(grade), candidate, category, rows, timely_frames=3,
                     bootstrap_samples=20)
    assert paired["contacts"]["HEAD_horizontal"]["rescue"] == 0
    candidate[0, 0, 1, 0] = 1
    paired = compare(grade, candidate, category, rows, timely_frames=3, bootstrap_samples=20)
    assert paired["contacts"]["HEAD_horizontal"]["rescue"] == 1
    assert paired["contacts"]["BODY"]["denominator"] == 0
    # Rescue, loss and delay each have a distinct hand-built contact scene.
    base = np.zeros((4, 2, 5, 2), np.int8)
    cand = np.zeros_like(base)
    base[1, :, 0, 0] = 1
    base[2, :, 0, 0] = cand[2, :, 2, 0] = 2
    cand[0, :, 0, 0] = 1
    cat = np.asarray([["contact", "clear"]] * 4)
    meta = [dict(scene_id=i, shape_family="horizontal" if i < 3 else "sign_edge",
                 size_variant=0 if i < 2 else 1) for i in range(4)]
    check = compare(base, cand, cat, meta, timely_frames=3, bootstrap_samples=100)
    head = check["contacts"]["HEAD_horizontal"]
    assert (head["rescue"], head["loss"], head["later_common"], head["denominator"]) == (2, 2, 2, 6)
    assert head["timely_rate_difference_ci"]["independent_scenes"] == 3
    assert head["delta_frames_common"]["median"] == 2
    assert check["contacts"]["HEAD_sign_edge"]["denominator"] == 2
    thin = check["contacts"]["HEAD_horizontal_thin"]
    assert (thin["rescue"], thin["loss"], thin["later_common"], thin["denominator"]) == (2, 2, 0, 4)
    assert check == compare(base, cand, cat, meta, timely_frames=3, bootstrap_samples=100)
    return "PASS: gap1/reset/upgrade/immediate onset, joint cost, actual-contact query, paired counts and scene bootstrap"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(self_test())
    else:
        parser.error("Import this helper or run --self-test; no file/data reader is exposed")
