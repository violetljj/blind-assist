"""Independent post-seal evaluation and diagnostic audit, no fitting or tuning.

Do not run until both fit receipts are sealed and the parent authorizes opening
dev_eval. CLI requires both fitted directories and verifies their hashes before
opening any truth. OCR transcript coverage is diagnostic, NOT field assignment
or hallucination-correctness accuracy. All outputs use the fixed v1 scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from build_manifest import FIELDS
from score_fields import read_records, score_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, lines=False):
    text = "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in value) if lines else json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def seal_check(directory):
    receipt = json.loads((directory / "fit_receipt.json").read_text(encoding="utf-8"))
    assert receipt["dev_eval_truth_opened"] is False
    assert sha(directory / "sealed_risks.jsonl") == receipt["sealed_risks_sha256"]
    assert sha(directory / "models.pkl") == receipt["models_sha256"]
    return {"fit_receipt_sha256": sha(directory / "fit_receipt.json"),
            "sealed_risks_sha256": receipt["sealed_risks_sha256"], "models_sha256": receipt["models_sha256"]}


def summarize_coverage(rows):
    summary = {}
    for split in sorted({r["split"] for r in rows}):
        summary[split] = {}
        for field in ("all",) + FIELDS:
            subset = [r for r in rows if r["split"] == split and (field == "all" or r["field"] == field)]
            valid = [r for r in subset if r["status"] == "EVALUABLE"]
            summary[split][field] = {"candidate_fields": len(subset), "evaluable_fields": len(valid),
                "not_evaluable_fields": len(subset) - len(valid),
                **{criterion + "_covered": sum(not r[criterion + "_error"] for r in valid) for criterion in ("strict", "relaxed")}}
    return summary


def ocr_diagnostic(truth, path, name):
    source = read_records(path)
    indexed = {r["image"]: r for r in source}
    assert len(indexed) == len(source)
    assert {r["image"] for r in truth} <= set(indexed)
    rows, variants = [], {}
    for policy in ("default_accepted", "all_retained"):
        selected = []
        for target in truth:
            record = indexed[target["image"]]
            if policy == "default_accepted":
                # The runner explicitly marks default >=0.5 accepted lines.
                text = "\n".join(str(line["text"]) for line in record["lines"] if line.get("accepted_at_default", False))
            else:
                text = "\n".join(str(line["text"]) for line in record["lines"])
            for field in FIELDS:
                row = {"ocr": name, "policy": policy, "image": target["image"], "split": target["split"],
                       "field": field, "ocr_status": record.get("status"),
                       **score_text(text, target["fields"][field])}
                selected.append(row)
        rows.extend(selected)
        variants[policy] = summarize_coverage(selected)
    return rows, {"prediction_sha256": sha(path), "image_statuses": dict(Counter(r.get("status", "missing") for r in source)),
                  "interpretation": "Required annotated text occurring anywhere in OCR transcript; diagnostic only, no field assignment or ownership correctness", "variants": variants}


def paired_flips(field_rows):
    lookup = {(r["model"], r["image"], r["field"]): r for r in field_rows}
    rows = []
    for old in field_rows:
        if old["model"] != "image_only":
            continue
        new = lookup[("image_ocr", old["image"], old["field"])]
        assert old["status"] == new["status"]
        row = {"image": old["image"], "split": old["split"], "field": old["field"], "status": old["status"],
               "image_only_answer": old["answer"], "image_ocr_answer": new["answer"], "required": old["required"]}
        for criterion in ("strict", "relaxed"):
            a, b = old[criterion + "_error"], new[criterion + "_error"]
            row[criterion + "_transition"] = "NOT_EVALUABLE" if a is None else {(0, 0): "both_correct", (1, 0): "rescued", (0, 1): "harmed", (1, 1): "both_error"}[(a, b)]
        rows.append(row)
    summary = {}
    for split in sorted({r["split"] for r in rows}):
        summary[split] = {}
        for field in ("all",) + FIELDS:
            subset = [r for r in rows if r["split"] == split and (field == "all" or r["field"] == field)]
            valid = [r for r in subset if r["status"] == "EVALUABLE"]
            summary[split][field] = {"evaluable_fields": len(valid), "candidate_fields": len(subset)}
            for criterion in ("strict", "relaxed"):
                counts = Counter(r[criterion + "_transition"] for r in valid)
                summary[split][field][criterion] = {k: counts[k] for k in ("both_correct", "rescued", "harmed", "both_error")}
                summary[split][field][criterion]["net_correct_gain"] = counts["rescued"] - counts["harmed"]
    return rows, summary


def check_parent(model, compare_path, scored_rows):
    from sklearn.metrics import average_precision_score, roc_auc_score
    parent = json.loads((compare_path / "metrics.json").read_text(encoding="utf-8"))
    rows = read_records(compare_path / "eval_fields.jsonl")
    lookup = {(r["image"], r["field"]): r for r in scored_rows if r["model"] == model and r["split"] == "dev_eval"}
    assert len(rows) == len(lookup) == 96
    assert len({(r["image"], r["field"]) for r in rows}) == len(rows)
    for row in rows:
        independent = lookup[(row["image"], row["field"])]
        for key in ("status", "strict_error", "relaxed_error"):
            assert row[key] == independent[key], (model, row["image"], row["field"], key)
    valid = [r for r in rows if r["status"] == "EVALUABLE"]
    assert len(valid) == parent["evaluable_fields"] == 76
    assert parent["not_evaluable_fields"] == 20
    checks, largest = [], 0.
    for field in ("all",) + FIELDS:
        subset = [r for r in valid if field == "all" or r["field"] == field]
        group = parent["groups"][field]
        assert group["evaluable_fields"] == len(subset)
        for arm in group["arms"]:
            risks = [r["risks"][arm] for r in subset]
            for criterion in ("strict", "relaxed"):
                y = [r[criterion + "_error"] for r in subset]
                expected = {"auroc_error_positive": roc_auc_score(y, risks) if 0 < sum(y) < len(y) else None,
                            "ap_error_positive": average_precision_score(y, risks) if sum(y) else None}
                recorded = group["arms"][arm][criterion]
                assert recorded["n"] == len(y) and recorded["errors"] == sum(y)
                for name, value in expected.items():
                    actual = recorded[name]
                    if value is None:
                        assert actual is None
                        difference = None
                    else:
                        difference = abs(float(value) - actual)
                        largest = max(largest, difference)
                        assert difference < 1e-12, (model, field, arm, criterion, name, difference)
                    checks.append({"field": field, "arm": arm, "criterion": criterion, "metric": name, "abs_difference": difference})
    return {"status": "PASS", "parent_metrics_sha256": sha(compare_path / "metrics.json"),
            "label_rows_checked": len(rows), "metric_checks": len(checks), "largest_abs_difference": largest, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("truth", "image-only", "image-ocr", "main-ocr", "english-ocr", "fitted-image-only", "fitted-image-ocr", "compare-image-only", "compare-image-ocr", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    if args.out.exists():
        raise FileExistsError("Preserve original audit; provide fresh output directory")
    seals = {"image_only": seal_check(args.fitted_image_only), "image_ocr": seal_check(args.fitted_image_ocr)}
    # Caller must have received the parent's explicit post-seal opening notice.
    truth = read_records(args.truth)
    assert len(truth) == 96 and sum(r["split"] == "train" for r in truth) == 64 and sum(r["split"] == "dev_eval" for r in truth) == 32
    args.out.mkdir(parents=True)
    scorer_out = args.out / "score_fields"
    child = subprocess.run([sys.executable, str(Path(__file__).with_name("score_fields.py")), "--truth", str(args.truth),
        "--predictions", "image_only=" + str(args.image_only), "--predictions", "image_ocr=" + str(args.image_ocr), "--out", str(scorer_out)],
        text=True, capture_output=True, check=True, timeout=90)
    (args.out / "scorer_stdout.txt").write_text(child.stdout, encoding="utf-8")
    scored = read_records(scorer_out / "field_scores.jsonl")
    flips, flip_summary = paired_flips(scored)
    write(args.out / "paired_field_flips.jsonl", flips, lines=True)
    write(args.out / "paired_field_flips_summary.json", flip_summary)
    ocr_rows, ocr_summaries = [], {}
    for name, path in (("main", args.main_ocr), ("english", args.english_ocr)):
        rows, summary = ocr_diagnostic(truth, path, name)
        ocr_rows.extend(rows)
        ocr_summaries[name] = summary
    write(args.out / "ocr_transcript_field_coverage.jsonl", ocr_rows, lines=True)
    write(args.out / "ocr_transcript_field_coverage_summary.json", ocr_summaries)
    verification = {"image_only": check_parent("image_only", args.compare_image_only, scored),
                    "image_ocr": check_parent("image_ocr", args.compare_image_ocr, scored)}
    write(args.out / "parent_metrics_verification.json", verification)
    # Reuse the exact prewritten score_fields hash sampling; enrich with the
    # opposite model answer and label-coverage transition for visual inspection.
    queue = read_records(scorer_out / "scoring_disputes_review_queue.jsonl")
    flips_by_key = {(r["image"], r["field"]): r for r in flips}
    enriched = []
    for row in queue:
        flip = flips_by_key[(row["image"], row["field"])]
        enriched.append(row | {k: flip[k] for k in ("image_only_answer", "image_ocr_answer", "strict_transition", "relaxed_transition")})
    write(args.out / "visual_review_queue.jsonl", enriched, lines=True)
    manifest = {"status": "PASS", "truth_sha256": sha(args.truth), "script_sha256": sha(__file__),
                "score_fields_sha256": sha(Path(__file__).with_name("score_fields.py")), "fit_seals": seals,
                "model_prediction_hashes": {"image_only": sha(args.image_only), "image_ocr": sha(args.image_ocr)},
                "model_statuses": {"image_only": dict(Counter(r.get("status", "missing") for r in read_records(args.image_only))),
                                   "image_ocr": dict(Counter(r.get("status", "missing") for r in read_records(args.image_ocr)))},
                "visual_review_queue_records": len(enriched), "parent_process_cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - started,
                "limitations": "Fixed lexical proxy only; OCR whole-transcript presence is not field assignment accuracy. No training, tuning, relabeling, or extra model inference."}
    write(args.out / "audit_receipt.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
