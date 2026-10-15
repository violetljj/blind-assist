"""Score structured field reads with fixed automatic proxies, not human truth.

Predictions are JSON/JSONL records: {image, fields: {product: {text, risk_scores},
brand: ..., variety: ...}}. Fields may instead be strings. risk_scores are
optional named numeric error scores (higher means more likely wrong), generated
without evaluator truth. Missing source annotations are NOT_EVALUABLE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from build_manifest import FIELDS, OUTPUT, tokens


def read_records(path):
    text = Path(path).read_text(encoding="utf-8")
    value = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    assert isinstance(value, list)
    return value


def has_span(sequence, span):
    return any(sequence[i:i + len(span)] == span for i in range(len(sequence) - len(span) + 1))


def score_text(answer, annotation):
    required = annotation["required"]
    if not required or annotation["status"] != "EVALUABLE":
        return {"status": "NOT_EVALUABLE", "strict_error": None, "relaxed_error": None}
    actual = tokens(answer)
    spans = [tokens(x) for x in required]
    # Required phrases may overlap (e.g. "sweet potato" and "potato fries").
    # Reusing their shared token is valid; a relaxed score must not be stricter
    # merely because annotations split overlapping phrases into two entries.
    target = Counter()
    for span in spans:
        target |= Counter(span)
    predicted = Counter(actual)
    overlap = sum((target & predicted).values())
    recall = overlap / sum(target.values())
    precision = overlap / len(actual) if actual else 0.
    strict = all(has_span(actual, span) for span in spans)
    relaxed = not (target - predicted)
    return {"status": "EVALUABLE", "strict_error": int(not strict), "relaxed_error": int(not relaxed),
            "required_span_coverage": sum(has_span(actual, span) for span in spans) / len(spans),
            "required_token_recall": recall, "required_token_precision": precision,
            "required_token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "exact_required_token_bag": predicted == target,
            "prediction_empty": not actual,
            "strict_relaxed_disagreement": strict != relaxed}


def ranking_metrics(labels, risks, keys):
    """Error-positive AUROC/AP; tie-expected AURC with low-risk acceptance."""
    n = len(labels)
    if not n:
        return {"n": 0}
    assert all(y in (0, 1) for y in labels)
    assert all(math.isfinite(float(s)) for s in risks)
    pos, neg = sum(labels), n - sum(labels)
    descending = sorted(range(n), key=lambda i: -risks[i])
    tp = fp = 0
    auc = ap = 0.
    j = 0
    while j < n:
        k = j + 1
        while k < n and risks[descending[k]] == risks[descending[j]]:
            k += 1
        dp = sum(labels[i] for i in descending[j:k])
        dn = k - j - dp
        tp += dp
        fp += dn
        auc += dp * (neg - fp + 0.5 * dn)
        if pos:
            ap += dp / pos * tp / (tp + fp)
        j = k
    # Hash is used only to make exact-count tie breaking reproducible, not as a feature.
    ascending = sorted(range(n), key=lambda i: (risks[i], hashlib.sha256(keys[i].encode()).hexdigest()))
    expected_cumulative = []
    previous_errors, j = 0, 0
    while j < n:
        k = j + 1
        while k < n and risks[ascending[k]] == risks[ascending[j]]:
            k += 1
        group_errors = sum(labels[i] for i in ascending[j:k])
        for step in range(1, k - j + 1):
            expected_cumulative.append(previous_errors + step * group_errors / (k - j))
        previous_errors += group_errors
        j = k
    aurc = sum(e / k for k, e in enumerate(expected_cumulative, 1)) / n
    oracle_aurc = sum(max(0, k - neg) / k for k in range(1, n + 1)) / n
    coverage = {}
    for fraction in (0.25, 0.5, 0.75, 0.9, 1.0):
        k = max(1, math.ceil(n * fraction))
        count = sum(labels[i] for i in ascending[:k])
        coverage[str(fraction)] = {"accepted_fields": k, "candidate_fields": n, "actual_coverage": k / n,
                                   "errors_hash_tiebreak": count, "risk_hash_tiebreak": count / k,
                                   "errors_expected_random_ties": expected_cumulative[k - 1],
                                   "risk_expected_random_ties": expected_cumulative[k - 1] / k}
    return {"n": n, "errors": pos, "error_prevalence": pos / n,
            "auroc_error_positive": auc / (pos * neg) if pos and neg else None,
            "ap_error_positive": ap if pos else None,
            "aurc_tie_expected": aurc, "oracle_aurc": oracle_aurc, "excess_aurc": aurc - oracle_aurc,
            "same_coverage": coverage}


def evaluate(truth, predictions, model):
    by_image = {p["image"]: p for p in predictions}
    assert len(by_image) == len(predictions), f"duplicate prediction image in {model}"
    expected = {r["image"] for r in truth}
    missing = expected - set(by_image)
    assert not missing, f"Missing {len(missing)} predictions in {model}; runtime gaps cannot silently change the denominator"
    rows = []
    for target in truth:
        prediction = by_image[target["image"]]
        fields = prediction.get("fields", prediction)
        for field in FIELDS:
            assert field in fields, f"Missing structured {field} in {target['image']} ({model})"
            value = fields[field]
            if isinstance(value, dict):
                answer = value.get("text", "")
                risks = value.get("risk_scores", {})
            else:
                answer, risks = value, {}
            answer = "" if answer is None else str(answer)
            row = {"model": model, "image": target["image"], "split": target["split"], "field": field,
                   "answer": answer, "required": target["fields"][field]["required"],
                   **score_text(answer, target["fields"][field]), "risk_scores": risks}
            rows.append(row)
    result = {"model": model, "prediction_images": len(predictions), "truth_images": len(truth),
              "extra_prediction_images_ignored": len(set(by_image) - expected), "splits": {}}
    for split in sorted({r["split"] for r in rows}):
        split_rows = [r for r in rows if r["split"] == split]
        groups = {}
        for field in ("all",) + FIELDS:
            selected = [r for r in split_rows if field == "all" or r["field"] == field]
            valid = [r for r in selected if r["status"] == "EVALUABLE"]
            group = {"candidate_fields": len(selected), "evaluable_fields": len(valid),
                     "not_evaluable_fields": len(selected) - len(valid), "scores": {}}
            for metric in ("strict", "relaxed"):
                errors = sum(r[f"{metric}_error"] for r in valid)
                group[f"{metric}_errors"] = errors
                group[f"{metric}_correct"] = len(valid) - errors
                group[f"{metric}_error_rate"] = errors / len(valid) if valid else None
            group["strict_relaxed_disagreements"] = sum(r["strict_relaxed_disagreement"] for r in valid)
            for name in sorted({name for r in valid for name in r["risk_scores"]}):
                scored = [r for r in valid if name in r["risk_scores"] and r["risk_scores"][name] is not None]
                group["scores"][name] = {"scored_fields": len(scored), "omitted_evaluable_fields": len(valid) - len(scored)}
                for metric in ("strict", "relaxed"):
                    group["scores"][name][metric] = ranking_metrics(
                        [r[f"{metric}_error"] for r in scored], [float(r["risk_scores"][name]) for r in scored],
                        [r["image"] + ":" + r["field"] for r in scored])
            groups[field] = group
        result["splits"][split] = groups
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=OUTPUT / "truth_manifest.jsonl")
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    truth = read_records(args.truth)
    assert len({r["image"] for r in truth}) == len(truth)
    all_rows, models = [], {}
    for item in args.predictions:
        name, path = item.split("=", 1)
        assert name not in models, "duplicate model name"
        rows, result = evaluate(truth, read_records(path), name)
        result["prediction_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        all_rows.extend(rows)
        models[name] = result
    output = {"contract_version": "product-fields-dev-v1", "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
              "positive_class": "required-annotation reading error (automatic lexical proxy)",
              "strict": "all required normalized token phrases appear contiguously in the corresponding field",
              "relaxed": "all required normalized tokens covered; max token multiplicity across required phrases, no phrase order or synonym expansion",
              "normalization": "Unicode NFKD/casefold, remove combining accents and apostrophes, ampersand->and, punctuation->token boundaries",
              "limitations": "Extra unsupported claims are not penalized by recall proxies. Valid synonyms may be false errors. This is not the source paper's human precision+recall coding. No scoring result authorizes tuning on dev_eval.",
              "models": models}
    (args.out / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "field_scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    # Fixed review queue: every strict/relaxed disagreement plus deterministic
    # lexical-error examples per model/split/field. Queue is for reporting disputes,
    # never for silently changing labels or retrospectively changing the scorer.
    queue = [dict(r, review_reason="strict_relaxed_disagreement") for r in all_rows if r.get("strict_relaxed_disagreement")]
    grouped = defaultdict(list)
    for r in all_rows:
        if r.get("strict_error") == 1:
            grouped[(r["model"], r["split"], r["field"])].append(r)
    for group in grouped.values():
        chosen = sorted(group, key=lambda r: hashlib.sha256((r["image"] + ":" + r["field"]).encode()).hexdigest())[:3]
        queue.extend(dict(r, review_reason="deterministic_lexical_error_sample") for r in chosen)
    (args.out / "scoring_disputes_review_queue.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in queue), encoding="utf-8")
    print(json.dumps({"models": list(models), "field_rows": len(all_rows), "review_queue_records": len(queue), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
