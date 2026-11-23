"""A label-informed ceiling for monotone thresholds under the clear-warning cost.

One false-positive clip necessarily consumes six walker charges. This evaluation
bounds the family using a single order statistic and does not select a policy.
"""
import json
import math

import numpy as np
from controller import OUT
from eval_alert_operating_point import FA_BUDGET, KM, read_clips, summarize


def ceiling(clips, field):
    km = sum(c.count(np.zeros(len(c.raw), dtype=bool))[KM] for c in clips)
    walkers = len(clips[0].episodes)
    assert all(len(c.episodes) == walkers for c in clips)
    max_clear_clips = math.floor(FA_BUDGET * km / walkers)
    peaks = sorted((getattr(c, field)[c.clear].max() for c in clips if c.clear.any()), reverse=True)
    boundary = float(peaks[max_clear_clips]) if len(peaks) > max_clear_clips else None
    masks = [np.ones(len(c.raw), dtype=bool) if boundary is None else getattr(c, field) > boundary for c in clips]
    counts = sum(c.count(a) for c, a in zip(clips, masks))
    score = summarize(counts)
    return {"clips": len(clips), "clear_frames": sum(int(c.clear.sum()) for c in clips),
            "clear_positive_clips_allowed_at_most": max_clear_clips, "must_exceed_risk": boundary,
            "detected_at_most": score["avoidable"] - score["late"], "avoidable": score["avoidable"],
            "at_ceiling": score, "attains_cost_on_these_clips": score["clear_warnings_per_km"] <= FA_BUDGET}


def main():
    clips = read_clips()
    families = {"pooled_one_cut": clips}
    for dev in ("arkit", "aria", "coda"):
        families[dev] = [c for c in clips if c.device == dev]
    for domain in ("arkit_val30_day", "arkit_val30_dark018"):
        families[domain] = [c for c in clips if c.obs["domain"] == domain]
    result = {g: {field: ceiling(sub, field) for field in ("raw", "confirmed")} for g, sub in families.items()}
    path = OUT / "operating_point_v2" / "clear_peak_ceiling.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf8")
    for g, measures in result.items():
        x = measures["confirmed"]
        print(g, x["clear_positive_clips_allowed_at_most"], x["detected_at_most"], "/", x["avoidable"],
              "attainable:", x["attains_cost_on_these_clips"])


if __name__ == "__main__":
    main()
