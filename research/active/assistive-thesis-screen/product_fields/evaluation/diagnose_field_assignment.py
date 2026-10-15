"""Posthoc diagnostic only: required words in assigned vs all model fields.

This does not replace the prewritten score or establish semantic correctness.
It quantifies failures where the model emitted all required text somewhere in
its structured response but not in the assigned field. No fitting/inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
from build_manifest import FIELDS
from score_fields import read_records, score_text


def answer(row, field):
    value = row["fields"][field]
    return str(value.get("text") or "") if isinstance(value, dict) else str(value or "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--predictions", action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows, summaries = [], {}
    truth = [r for r in read_records(args.truth) if r["split"] == "dev_eval"]
    assert len(truth) == 32
    for item in args.predictions:
        name, path = item.split("=", 1)
        predictions = {r["image"]: r for r in read_records(path)}
        model_rows = []
        for target in truth:
            prediction = predictions[target["image"]]
            merged = " ; ".join(answer(prediction, f) for f in FIELDS)
            for field in FIELDS:
                annotation = target["fields"][field]
                original = score_text(answer(prediction, field), annotation)
                union = score_text(merged, annotation)
                model_rows.append({"model": name, "image": target["image"], "field": field, "status": original["status"],
                    "required": annotation["required"], "assigned_answer": answer(prediction, field), "merged_answer": merged,
                    "assigned_strict_error": original["strict_error"], "merged_strict_error": union["strict_error"],
                    "required_text_elsewhere": original["strict_error"] == 1 and union["strict_error"] == 0})
        rows.extend(model_rows)
        summaries[name] = {}
        for field in ("all",) + FIELDS:
            valid = [r for r in model_rows if r["status"] == "EVALUABLE" and (field == "all" or r["field"] == field)]
            summaries[name][field] = {"evaluable_fields": len(valid), "assigned_errors": sum(r["assigned_strict_error"] for r in valid),
                "assigned_errors_with_required_text_elsewhere": sum(r["required_text_elsewhere"] for r in valid),
                "merged_text_covered": sum(not r["merged_strict_error"] for r in valid)}
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "field_assignment_posthoc.jsonl"
    assert not destination.exists(), "Preserve existing diagnostic"
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"analysis_role": "POSTHOC_DIAGNOSTIC_NOT_MAIN_SCORE", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation": "Word presence across fields does not establish correct field meaning or absence of hallucination. Original scores unchanged.", "models": summaries}
    (args.out / "field_assignment_posthoc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
