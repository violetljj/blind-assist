"""Build image-only inference inputs and separate evaluator-only annotations.

Uses the prior 96-image receipt order verbatim, never resamples by labels.
Standard library only. Existing outputs may be regenerated only identically.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import unicodedata
import re

FIELDS = ("product", "brand", "variety")
REPO = Path(__file__).resolve().parents[5]
SOURCE = REPO / "artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa"
OUTPUT = REPO / "artifacts.local/work/product-fields-dev-20261015/evaluation"


def tokens(text):
    text = unicodedata.normalize("NFKD", str(text)).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ").replace("'", "").replace("\u2019", "")
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value, lines=False):
    text = ("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in value) if lines
            else json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite changed existing evidence: {path}")
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    start = time.process_time()
    receipts = json.loads((args.source / "image_receipts.json").read_text(encoding="utf-8"))
    products = json.loads((args.source / "products.json").read_text(encoding="utf-8"))
    by_name = {p["file_name"]: p for p in products}
    assert len(by_name) == len(products), "duplicate source product image"
    assert len(receipts) == 96, "Original sample must contain exactly 96 images"
    assert len({r["image"] for r in receipts}) == 96, "duplicate receipt image"
    args.out.mkdir(parents=True, exist_ok=True)
    inference, truth = [], []
    denominators = {}
    for index, receipt in enumerate(receipts):
        assert not receipt.get("error"), receipt
        path = args.source / "images" / receipt["image"]
        assert digest(path) == receipt["sha256"], f"Image changed: {path}"
        assert path.stat().st_size == receipt["bytes"]
        source = by_name[receipt["image"]]
        fields = {}
        for field in FIELDS:
            entries = source[field]
            assert isinstance(entries, list)
            assert all(isinstance(e["text"], str) and isinstance(e["optional"], bool) for e in entries)
            required = [e["text"] for e in entries if not e["optional"] and tokens(e["text"])]
            fields[field] = {"required": required,
                             "optional": [e["text"] for e in entries if e["optional"]],
                             "status": "EVALUABLE" if required else "NOT_EVALUABLE",
                             "reason": None if required else "empty_or_optional_only_annotation"}
        # Deliberately only image identifier/path: no annotations, quality, or split.
        inference.append({"image": receipt["image"], "path": str(path.absolute())})
        truth.append({"image": receipt["image"], "source_id": source["id"],
                      "split": "train" if index < 64 else "dev_eval",
                      "receipt_index": index, "image_sha256": receipt["sha256"], "fields": fields})
    for split in ("train", "dev_eval"):
        subset = [r for r in truth if r["split"] == split]
        denominators[split] = {"images": len(subset), "candidate_fields": len(subset) * 3,
                              "fields": {f: dict(Counter(r["fields"][f]["status"] for r in subset)) for f in FIELDS}}
        denominators[split]["evaluable_fields"] = sum(r["fields"][f]["status"] == "EVALUABLE" for r in subset for f in FIELDS)
    train, evaluation = truth[:64], truth[64:]
    def signature(row, field):
        return " | ".join(" ".join(tokens(s)) for s in row["fields"][field]["required"])
    duplicates = {}
    for field in FIELDS + ("all_required_fields",):
        sig = (lambda r: " || ".join(signature(r, f) for f in FIELDS)) if field == "all_required_fields" else (lambda r: signature(r, field))
        train_signatures = {sig(r) for r in train if sig(r).strip(" |")}
        evaluation_signatures = [sig(r) for r in evaluation if sig(r).strip(" |")]
        duplicates[field] = {"train_distinct_nonempty": len(train_signatures),
                             "eval_nonempty_images": len(evaluation_signatures),
                             "eval_images_with_train_signature": sum(s in train_signatures for s in evaluation_signatures),
                             "shared_signature_count": len(train_signatures.intersection(evaluation_signatures))}
    audit = {"contract_version": "product-fields-dev-v1", "evidence": "reused Development; not independent confirmation",
             "selection": "existing image_receipts.json order, first 64 train and last 32 dev_eval, no reselection",
             "source_products_sha256": digest(args.source / "products.json"),
             "source_receipts_sha256": digest(args.source / "image_receipts.json"),
             "image_names_disjoint": not ({r["image"] for r in train} & {r["image"] for r in evaluation}),
             "image_hashes_disjoint": not ({r["image_sha256"] for r in train} & {r["image_sha256"] for r in evaluation}),
             "denominators": denominators, "cross_split_annotation_overlap": duplicates,
             "limitation": "No photographer or physical-product identities. Signature overlap is a leakage-risk diagnostic, not proof of duplicate physical instances. Empty annotations do not identify unanswerable fields."}
    assert audit["image_names_disjoint"] and audit["image_hashes_disjoint"]
    dump(args.out / "inference_manifest.jsonl", inference, lines=True)
    dump(args.out / "truth_manifest.jsonl", truth, lines=True)
    dump(args.out / "truth_train.jsonl", train, lines=True)
    dump(args.out / "truth_dev_eval.jsonl", evaluation, lines=True)
    dump(args.out / "manifest_audit.json", audit)
    print(json.dumps({"out": str(args.out), "denominators": denominators,
                      "cross_split_annotation_overlap": duplicates, "cpu_seconds": time.process_time() - start}, indent=2))


if __name__ == "__main__":
    main()
