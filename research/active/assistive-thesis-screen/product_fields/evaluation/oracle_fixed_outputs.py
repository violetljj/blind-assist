"""Truth-assisted ceiling of selecting between TWO FIXED outputs, not deployable.

This is not an upper bound for future different models/prompts/passes. It only
answers how many fields these two recorded predictions can jointly cover.
"""
import argparse
import hashlib
import json
from pathlib import Path
from build_manifest import FIELDS
from score_fields import read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Preserve prior oracle diagnostic"
    rows = read_records(args.scores)
    lookup = {(r["model"], r["image"], r["field"]): r for r in rows}
    summary = {}
    for split in ("train", "dev_eval"):
        summary[split] = {}
        for field in ("all",) + FIELDS:
            eligible = [r for r in rows if r["model"] == "image_only" and r["split"] == split and r["status"] == "EVALUABLE" and (field == "all" or r["field"] == field)]
            summary[split][field] = {}
            for criterion in ("strict", "relaxed"):
                pairs = [(r[criterion + "_error"], lookup[("image_ocr", r["image"], r["field"])][criterion + "_error"]) for r in eligible]
                a, b = sum(x == 0 for x, y in pairs), sum(y == 0 for x, y in pairs)
                oracle = sum(x == 0 or y == 0 for x, y in pairs)
                summary[split][field][criterion] = {"evaluable_fields": len(pairs), "image_only_correct": a,
                    "image_ocr_correct": b, "oracle_correct_union": oracle,
                    "oracle_additional_vs_image_only": oracle - a, "oracle_additional_vs_image_ocr": oracle - b}
    output = {"analysis_role": "POSTHOC_TWO_FIXED_OUTPUTS_ORACLE_NOT_DEPLOYABLE",
        "limitation": "Evaluator truth chooses between the two recorded answers. No evidence for an implementable selector or an upper bound over future different models, prompts or passes.",
        "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(), "results": summary}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
