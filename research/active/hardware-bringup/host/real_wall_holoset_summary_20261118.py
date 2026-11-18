"""Aggregate a frozen HoloSet cohort. Each eligible timestamp must have exactly one score."""
import argparse
import csv
import json
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import median


def rate(rows):
    alerts = sum(r["alert"] for r in rows)
    return {"alerts": alerts, "frames": len(rows),
            "rate": alerts / len(rows) if rows else None,
            "history_median": median(r["history"] for r in rows) if rows else None,
            "history_span_s_median": median(r["history_span_s"] for r in rows) if rows else None,
            "longest_gap_s_median": median(r["max_window_gap_s"] for r in rows) if rows else None,
            "valid_depth_fraction_median": median(r["valid_fraction"] for r in rows) if rows else None}


def summarize(cohort, results, audit=None):
    rows = [json.loads(line) for line in cohort.read_text().splitlines() if line]
    digest = sha256(cohort.read_bytes()).hexdigest()
    by_id = {}
    sequences = sorted({r["seq"] for r in rows})
    for seq in sequences:
        data = json.loads((results / seq / "scores.json").read_bytes())
        if data["cohort_sha256"] != digest or data["sequence"] != seq:
            raise ValueError(f"Wrong score receipt for {seq}")
        for s in data["scores"]:
            key = seq, s["depth_ts"]
            if key in by_id or not 0 <= s["score"] <= 1 or s["alert"] != (s["score"] > .5367):
                raise ValueError(f"Invalid score {key}")
            by_id[key] = s
    ids = [(r["seq"], r["depth_ts"]) for r in rows]
    if set(ids) != set(by_id) or len(ids) != len(by_id):
        raise ValueError("Score and cohort denominators differ")
    rows = [r | by_id[key] for key, r in zip(ids, rows)]
    groups = ("low", "mid", "high", "none")
    overall = {group: rate([r for r in rows if r["stratum"] == group]) for group in groups}
    by_seq = {seq: {g: rate([r for r in rows if r["seq"] == seq and r["stratum"] == g])
                    for g in groups} for seq in sequences}
    fractions = {g: Fraction(overall[g]["alerts"], overall[g]["frames"]) for g in ("low", "none", "high")}
    diff = fractions["low"] - fractions["none"]
    passed = all(overall[g]["frames"] >= 100 and sum(by_seq[s][g]["frames"] > 0 for s in sequences) >= 4
                 for g in ("low", "none"))
    if not passed:
        verdict = "insufficient coverage"
    elif fractions["low"] >= 2 * fractions["none"] and diff >= Fraction(5, 100):
        verdict = "increased alert rate"
    elif diff <= Fraction(2, 100):
        verdict = "no increased alert rate under the registered rule"
    else:
        verdict = "undetermined"
    paired = [v["low"]["rate"] - v["none"]["rate"] for v in by_seq.values()
              if v["low"]["frames"] and v["none"]["frames"]]
    result = dict(cohort_sha256=digest, groups=overall, per_sequence=by_seq, verdict=verdict,
                low_minus_none=float(diff), low_minus_high=float(fractions["low"] - fractions["high"]),
                equal_sequence_difference=sum(paired)/len(paired) if paired else None,
                shared_sequences=len(paired), positive_differences=sum(d > 0 for d in paired))
    if audit:
        low_ids = {(r["seq"], r["depth_ts"]) for r in rows if r["stratum"] == "low"}
        with audit.open(encoding="utf-8-sig", newline="") as handle:
            inspected = list(csv.DictReader(handle))
        verified = []
        for r in inspected:
            key = r["seq"], int(r["depth_ts"])
            if key not in low_ids:
                raise ValueError(f"Visual audit row outside the low stratum: {key}")
            if r["material"] == "opaque_plain_wall":
                verified.append(by_id[key])
        result["reviewed_wall_frames"] = rate(verified)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "results", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--audit", type=Path, help="Pre-score frame_reviews.csv from the wall identity audit")
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.cohort, args.results, args.audit), indent=2), encoding="utf-8")
