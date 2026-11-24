"""Shared evaluator adapter for the two sequential research probes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M3 = HERE.parent / "m3_online_conformal_safety"
sys.path.insert(0, str(M3))
from eval_alert_operating_point import (read_clips, load_stream, confirm_score, fit_threshold,
                                       fit_range, ranges, summarize, OUT, LATE, CLEAR_WARN, KM)
from diagnose_clear_peak_ceiling import ceiling


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def materialize(version):
    clips = read_clips()
    stream = OUT.parent / (version+"_stream")
    if version != "v4":
        observations = load_stream(stream, OUT.parent.parent / "zeroshot_eval")
        for c, obs in zip(clips, observations, strict=True):
            assert (c.obs["name"], c.obs["domain"]) == (obs["name"], obs["domain"])
            c.obs = obs
            with np.load(stream / obs["domain"] / (obs["name"]+".npz")) as f:
                c.raw, c.bev_d = f["r_hyb_strict"], f["d_bev_strict"]
            c.confirmed = confirm_score(c.raw)
    return clips, stream


def score(counts):
    rec = summarize(counts)
    rec["timely"] = rec["avoidable"] - rec["late"]
    return rec


def crossfit(clips, feature):
    counts = []
    fits = []
    for fold in (0,1):
        train = [c for c in clips if c.fold != fold]
        test = [c for c in clips if c.fold == fold]
        assert {c.source for c in train}.isdisjoint(c.source for c in test)
        t, report = fit_threshold(train, feature, True)
        fits.append(dict(test_fold=fold, training=report, train_sources=sorted({c.source for c in train})))
        counts += [(c, c.count(getattr(c, feature) >= t)) for c in test]
    return {"fits":fits, "pooled":score(sum(x for _,x in counts)),
            "domains":{dom:score(sum(x for c,x in counts if c.obs["domain"]==dom))
                       for dom in sorted({c.obs["domain"] for c,_ in counts})}}


def baselines(clips):
    result = {"confirmed_shared":crossfit(clips, "confirmed")}
    device, lower, distance = [], [], []
    for fold in (0,1):
        train, test = [c for c in clips if c.fold != fold], [c for c in clips if c.fold == fold]
        q = fit_range(train)
        cuts = {d:fit_threshold([c for c in train if c.device==d], "confirmed", True)[0]
                for d in {c.device for c in train}}
        for c in test:
            device.append(c.count(c.confirmed>=cuts[c.device]))
            lower.append(c.count(ranges(c,q)[0]<=2.))
            distance.append(c.count(c.bev_d<=2.))
    result.update(device_cost=score(sum(device)), fixed_range_95_2m=score(sum(lower)),
                  bev_distance_2m=score(sum(distance)), confirmed_peak=ceiling(clips,"confirmed"),
                  raw_peak=ceiling(clips,"raw"))
    return result


def frontier(clips, field):
    # At a global threshold only clips with that exact score change.
    events = {}
    base = sum(c.count(np.zeros(len(c.raw), bool)) for c in clips)
    for c in clips:
        vals = getattr(c,field)
        prev = c.count(np.zeros(len(vals),bool))
        for cut in np.unique(vals)[::-1]:
            nxt = c.count(vals>=cut)
            events[float(cut)] = events.get(float(cut),0) + (nxt-prev)
            prev = nxt
    rows = []
    def point(t):
        x = score(base)
        return [t,x["timely"],x["fa_episodes"],x["repeated_clear_warnings"],x["known_clear_frames"],x["clear_alert_frames"]]
    rows.append(point(None))
    for cut in sorted(events,reverse=True):
        base += events[cut]
        rows.append(point(cut))
    # Exact non-dominated points for timely vs repeated clear cost on this cohort.
    costs = {}
    for row in rows:
        if row[3] not in costs or row[1] > costs[row[3]][1]:
            costs[row[3]] = row
    envelope = []
    for _, row in sorted(costs.items()):
        if not envelope or row[1] > envelope[-1][1]:
            envelope.append(row)
    budget = 5*base[KM]
    best = max(row[1] for row in envelope if row[3]<=budget)
    return dict(columns=["cut_ge_null_is_off","timely","original_false","clear_charges","clear_frames","alert_clear_frames"],
                walker_km=base[KM], distinct=len(events), curve=rows, pareto=envelope,
                exact_best_at_5_clear_per_km=best)
