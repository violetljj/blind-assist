"""Posthoc rotation observation audit; no model inference, fitting or selection.

This explicitly audits an outcome-informed 180-degree reorientation diagnostic,
separate from the original main comparisons. It cannot establish confirmation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from audit_results import check_parent, seal_check, sha, write
from build_manifest import FIELDS
from score_fields import read_records

REVIEW_IMAGES = ["VizWiz_train_" + value + ".jpg" for value in ("00010954", "00014955", "00012312", "00001404", "00019585", "00012716")]


def compare(rows):
    lookup = {(r["model"], r["image"], r["field"]): r for r in rows}
    pairs = []
    for original in rows:
        if original["model"] != "image_only":
            continue
        rotated = lookup[("rotated_180", original["image"], original["field"])]
        assert original["status"] == rotated["status"]
        result = {"image": original["image"], "split": original["split"], "field": original["field"],
            "status": original["status"], "required": original["required"], "original_answer": original["answer"], "rotated_answer": rotated["answer"]}
        for criterion in ("strict", "relaxed"):
            a, b = original[criterion + "_error"], rotated[criterion + "_error"]
            result[criterion + "_transition"] = "NOT_EVALUABLE" if a is None else {(0, 0): "both_correct", (1, 0): "rescued", (0, 1): "harmed", (1, 1): "both_error"}[(a, b)]
        pairs.append(result)
    summary = {}
    for split in ("train", "dev_eval"):
        summary[split] = {}
        for field in ("all",) + FIELDS:
            valid = [r for r in pairs if r["split"] == split and r["status"] == "EVALUABLE" and (field == "all" or r["field"] == field)]
            summary[split][field] = {"evaluable_fields": len(valid)}
            for criterion in ("strict", "relaxed"):
                counts = Counter(r[criterion + "_transition"] for r in valid)
                original = counts["both_correct"] + counts["harmed"]
                rotated = counts["both_correct"] + counts["rescued"]
                summary[split][field][criterion] = {**{k: counts[k] for k in ("both_correct", "rescued", "harmed", "both_error")},
                    "original_correct": original, "rotated_correct": rotated, "net_correct_gain": rotated - original,
                    "oracle_correct_union": counts["both_correct"] + counts["harmed"] + counts["rescued"]}
    return pairs, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("truth", "original", "rotated", "features", "fitted", "compare", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    assert not args.out.exists(), "Preserve previous rotation audit"
    seal = seal_check(args.fitted)
    fit_receipt = json.loads((args.fitted / "fit_receipt.json").read_text(encoding="utf-8"))
    assert sha(args.features) == fit_receipt["features_sha256"]
    args.out.mkdir(parents=True)
    scorer_out = args.out / "score_fields"
    child = subprocess.run([sys.executable, str(Path(__file__).with_name("score_fields.py")), "--truth", str(args.truth),
        "--predictions", "image_only=" + str(args.original), "--predictions", "rotated_180=" + str(args.rotated), "--out", str(scorer_out)],
        text=True, capture_output=True, check=True, timeout=90)
    (args.out / "scorer_stdout.txt").write_text(child.stdout, encoding="utf-8")
    rows = read_records(scorer_out / "field_scores.jsonl")
    verification = check_parent("rotated_180", args.compare, rows)
    write(args.out / "parent_metrics_verification.json", verification)
    pairs, summary = compare(rows)
    write(args.out / "paired_field_changes.jsonl", pairs, lines=True)
    write(args.out / "paired_field_summary.json", {"analysis_role": "POSTHOC_OUTCOME_INFORMED_ROTATION_DIAGNOSTIC", "results": summary,
        "limitation": "Original main results unchanged. Oracle union uses truth to choose between two fixed outputs, not a deployable orientation selector or ceiling over future methods."})
    review = [r for r in pairs if r["image"] in REVIEW_IMAGES]
    assert len(review) == 18
    write(args.out / "six_reviewed_images_changes.jsonl", review, lines=True)
    receipt = {"status": "PASS", "analysis_role": "POSTHOC_OUTCOME_INFORMED_ROTATION_DIAGNOSTIC_NOT_CONFIRMATION",
        "truth_sha256": sha(args.truth), "original_prediction_sha256": sha(args.original), "rotated_prediction_sha256": sha(args.rotated),
        "script_sha256": sha(__file__), "fit_seal": seal, "sklearn_metric_checks": verification["metric_checks"],
        "largest_metric_abs_difference": verification["largest_abs_difference"], "six_review_images": REVIEW_IMAGES,
        "parent_process_cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - started,
        "limitation": "Rotation was chosen after seeing previous outcomes/images. No new inference, fit, selector, labels or tuning in this audit."}
    write(args.out / "audit_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    print(json.dumps(summary["dev_eval"], indent=2))


if __name__ == "__main__":
    main()
