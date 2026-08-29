"""Evaluate the frozen PB3 metadata-backed lexical identity proof and join."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from rapidocr import RapidOCR
from transformers import AutoImageProcessor, AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_facade_fingerprint as facade  # noqa: E402
import named_poi_place_identity_eval as pb2  # noqa: E402
import named_poi_text_identity_opportunity as textaudit  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label}_HASH_MISMATCH:expected={expected}:observed={observed}")


def _candidate_values(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [0.0, 1.0]
    result = {0.0, 1.0, *unique}
    result.update((left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False))
    return sorted(result)


def _proof(row: dict[str, Any], score_threshold: float, margin_threshold: float) -> str | None:
    if row["top_score"] >= score_threshold and row["margin"] >= margin_threshold:
        return row["top_entity"]
    return None


def _select_text_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: tuple[tuple[float, float, float, float, float], dict[str, Any]] | None = None
    for score_threshold in _candidate_values([row["top_score"] for row in rows]):
        for margin_threshold in _candidate_values([row["margin"] for row in rows]):
            proofs = [(_proof(row, score_threshold, margin_threshold), row) for row in rows]
            emitted = [(entity, row) for entity, row in proofs if entity is not None]
            correct = sum(entity == row["entity_id"] for entity, row in emitted)
            wrong = len(emitted) - correct
            precision = correct / len(emitted) if emitted else 0.0
            correct_coverage = correct / len(rows)
            identity_rows = [row for row in rows if row["facet"] == "identity-bearing"]
            identity_correct = sum(
                _proof(row, score_threshold, margin_threshold) == row["entity_id"]
                for row in identity_rows
            )
            identity_coverage = identity_correct / len(identity_rows)
            payload = {
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
                "proof_precision": precision,
                "correct_proof_coverage": correct_coverage,
                "identity_bearing_correct_proof_coverage": identity_coverage,
                "proofs": len(emitted),
                "correct_proofs": correct,
                "wrong_proofs": wrong,
            }
            key = (precision, correct_coverage, identity_coverage, score_threshold, margin_threshold)
            if best is None or key > best[0]:
                best = (key, payload)
    assert best is not None
    return best[1]


def _text_rows(
    queries: list[dict[str, Any]],
    manifest: dict[str, Any],
    engine: RapidOCR,
    canonical_only: bool,
    ocr_cache: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    entities_by_split = {
        split: [entity for entity in manifest["entities"] if entity["split"] == split]
        for split in ("development", "test")
    }
    rows = []
    cache = ocr_cache if ocr_cache is not None else {}
    for query in queries:
        texts = cache.get(query["key"])
        if texts is None:
            image = cv2.imread(str(query["path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(query["path"])
            output = engine(image)
            texts = [str(value) for value in (output.txts if output.txts is not None else [])]
            cache[query["key"]] = texts
        observations = textaudit._windows(texts)
        scores: dict[str, float] = {}
        for entity in entities_by_split[query["split"]]:
            aliases = [entity["name"]] if canonical_only else entity["aliases"]
            scores[entity["id"]] = max(
                (textaudit._similarity(alias, observations) for alias in aliases), default=0.0
            )
        ranked = sorted(scores, key=lambda entity_id: (-scores[entity_id], entity_id))
        top = ranked[0]
        second = ranked[1]
        rows.append(
            {
                **query,
                "ocr_texts": texts,
                "entity_scores": scores,
                "top_entity": top,
                "top_score": scores[top],
                "second_score": scores[second],
                "margin": scores[top] - scores[second],
            }
        )
    return rows


def _proof_metrics(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    records = []
    for row in rows:
        entity = _proof(row, thresholds["score_threshold"], thresholds["margin_threshold"])
        records.append(
            {
                "key": row["key"],
                "target": row["entity_id"],
                "facet": row["facet"],
                "proof": entity,
                "correct": entity == row["entity_id"] if entity is not None else None,
                "top_score": row["top_score"],
                "margin": row["margin"],
                "ocr_texts": row["ocr_texts"],
                "entity_scores": row["entity_scores"],
            }
        )
    proofs = [row for row in records if row["proof"] is not None]
    correct = sum(row["correct"] is True for row in proofs)
    wrong = len(proofs) - correct
    facets = {}
    for facet in sorted({row["facet"] for row in records}):
        selected = [row for row in records if row["facet"] == facet]
        facets[facet] = {
            "count": len(selected),
            "correct_proofs": sum(row["correct"] is True for row in selected),
            "wrong_proofs": sum(row["correct"] is False for row in selected),
            "correct_proof_coverage": sum(row["correct"] is True for row in selected) / len(selected),
        }
    return {
        "query_count": len(records),
        "proofs": len(proofs),
        "correct_proofs": correct,
        "wrong_proofs": wrong,
        "proof_precision": correct / len(proofs) if proofs else 0.0,
        "correct_proof_coverage": correct / len(records),
        "unknown": len(records) - len(proofs),
        "facets": facets,
        "records": records,
    }


def _join_metrics(
    baseline_records: list[dict[str, Any]], proof_records: list[dict[str, Any]]
) -> dict[str, Any]:
    proof_by_key = {row["key"]: row for row in proof_records}
    positive_accepts = 0
    wrong_accepts = 0
    wrong_total = 0
    records = []
    for baseline in baseline_records:
        proof = proof_by_key[baseline["key"]]["proof"]
        target = baseline["target"]
        if proof is None:
            positive = baseline["positive_accepted"]
            accepted_wrong = list(baseline["wrong_entities_accepted"])
            source = "APPEARANCE_FALLBACK"
        else:
            positive = proof == target
            accepted_wrong = [] if proof == target else [proof]
            source = "METADATA_ALIAS_PROOF"
        positive_accepts += int(positive)
        wrong_accepts += len(accepted_wrong)
        wrong_total += len(baseline["ranked_entities"]) - 1
        records.append(
            {
                "key": baseline["key"],
                "target": target,
                "facet": baseline["facet"],
                "source": source,
                "proof": proof,
                "positive_accepted": bool(positive),
                "wrong_entities_accepted": accepted_wrong,
            }
        )
    return {
        "query_count": len(records),
        "positive_acceptance": positive_accepts / len(records),
        "source_label_negative_false_confirmation": wrong_accepts / wrong_total,
        "state_counts": {
            "positive_accept": positive_accepts,
            "positive_reject": len(records) - positive_accepts,
            "wrong_accept": wrong_accepts,
            "wrong_reject": wrong_total - wrong_accepts,
        },
        "records": records,
    }


def run(protocol_path: Path, output_root: Path, models: Path, batch_size: int) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for label, source in protocol["sources"].items():
        _require_hash(ROOT / source["path"], source["sha256"], label.upper())
    manifest_path = ROOT / protocol["sources"]["dataset_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["inventory_sha256"] != protocol["sources"]["dataset_manifest"]["inventory_sha256"]:
        raise ValueError("DATASET_INVENTORY_HASH_MISMATCH")
    audit = json.loads((ROOT / protocol["sources"]["source_audit"]["path"]).read_text(encoding="utf-8"))
    if audit["checks"]["ocr_calls_before_freeze"] != 0 or audit["checks"]["embedding_calls_before_freeze"] != 0:
        raise ValueError("SOURCE_AUDIT_PRE_FREEZE_MODEL_CALLS_NOT_ZERO")
    for filename, expected in protocol["runtime"]["ocr_models"].items():
        _require_hash(models / filename, expected, f"OCR:{filename}")

    rows: list[dict[str, Any]] = []
    for entity in manifest["entities"]:
        for index, image in enumerate(entity["references"], start=1):
            path = Path(image["local_path"])
            _require_hash(path, image["sha256"], f"REFERENCE:{entity['id']}:{index}")
            rows.append({"key": f"ref:{entity['split']}:{entity['id']}:{index:02d}", "entity_id": entity["id"], "split": entity["split"], "role": "reference", "facet": None, "path": path})
        for image in entity["queries"]:
            path = Path(image["local_path"])
            _require_hash(path, image["sha256"], f"QUERY:{image['key']}")
            rows.append({"key": f"query:{image['key']}", "entity_id": entity["id"], "split": entity["split"], "role": "query", "facet": image["facet"], "path": path})
    split_entities = {split: {row["entity_id"] for row in rows if row["split"] == split} for split in ("development", "test")}
    if split_entities["development"] & split_entities["test"]:
        raise ValueError("BUILDING_SPLIT_OVERLAP")
    queries = {split: [row for row in rows if row["split"] == split and row["role"] == "query"] for split in ("development", "test")}

    clip_path = ROOT / protocol["runtime"]["appearance_models"]["clip"]
    dino_path = ROOT / protocol["runtime"]["appearance_models"]["dinov2"]
    _require_hash(clip_path / "pytorch_model.bin", protocol["runtime"]["appearance_models"]["clip_weights_sha256"], "CLIP")
    _require_hash(dino_path / "model.safetensors", protocol["runtime"]["appearance_models"]["dinov2_weights_sha256"], "DINOV2")
    output_root.mkdir(parents=True, exist_ok=True)
    representative_images = [Image.open(row["path"]).convert("RGB") for row in rows if row["split"] == "development"][:8]
    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative = (
        clip_processor(images=representative_images, return_tensors="pt")["pixel_values"],
        dino_processor(images=representative_images, return_tensors="pt")["pixel_values"],
    )
    backend, baseline_models = facade._select_backend(
        clip_path, dino_path, representative, output_root / "appearance_backend_receipt.json"
    )
    image_rows = [facade.ImageRow(row["key"], row["entity_id"], row["role"], row["path"], _sha256(row["path"]), "") for row in rows]
    encoded = facade._encode_images(image_rows, baseline_models, clip_processor, dino_processor, backend["selected_device_type"], patch_grid=9, batch_size=batch_size)
    descriptors = {
        "B0_CLIP": {key: value["clip"] for key, value in encoded.items()},
        "B1_DINOv2": {key: value["dino"] for key, value in encoded.items()},
    }
    del baseline_models, encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for split in ("development", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        entity_ids = sorted(split_entities[split])
        refs = {entity: [row["key"] for row in split_rows if row["entity_id"] == entity and row["role"] == "reference"] for entity in entity_ids}
        scores[split] = {"B0_CLIP": {}, "B1_DINOv2": {}, "B2_CLIP_DINO": {}}
        for query in queries[split]:
            for arm in ("B0_CLIP", "B1_DINOv2"):
                vector = descriptors[arm][query["key"]]
                scores[split][arm][query["key"]] = {entity: max(float(vector @ descriptors[arm][ref]) for ref in refs[entity]) for entity in entity_ids}
            clip = pb2._zscore(scores[split]["B0_CLIP"][query["key"]])
            dino = pb2._zscore(scores[split]["B1_DINOv2"][query["key"]])
            scores[split]["B2_CLIP_DINO"][query["key"]] = {entity: 0.5 * (clip[entity] + dino[entity]) for entity in entity_ids}

    thresholds = {}
    development_appearance = {}
    for arm in ("B0_CLIP", "B1_DINOv2", "B2_CLIP_DINO"):
        positive = [scores["development"][arm][row["key"]][row["entity_id"]] for row in queries["development"]]
        negative = [score for row in queries["development"] for entity, score in scores["development"][arm][row["key"]].items() if entity != row["entity_id"]]
        thresholds[arm] = pb2._threshold(positive, negative)
        development_appearance[arm] = pb2._metrics(queries["development"], scores["development"][arm], thresholds[arm]["threshold"])
    baseline = max(development_appearance, key=lambda arm: (pb2._selection_key(development_appearance[arm]), arm))
    test_appearance = pb2._metrics(queries["test"], scores["test"][baseline], thresholds[baseline]["threshold"])

    engine = RapidOCR(params={"Global.model_root_dir": str(models), "Global.log_level": "error", "EngineConfig.onnxruntime.intra_op_num_threads": 4, "EngineConfig.onnxruntime.inter_op_num_threads": 1})
    ocr_cache: dict[str, list[str]] = {}
    alias_rows = _text_rows(queries["development"] + queries["test"], manifest, engine, canonical_only=False, ocr_cache=ocr_cache)
    canonical_rows = _text_rows(queries["development"] + queries["test"], manifest, engine, canonical_only=True, ocr_cache=ocr_cache)
    selected_text = _select_text_thresholds([row for row in alias_rows if row["split"] == "development"])
    text_metrics = {
        "development": {
            "C0_CANONICAL_TEXT": _proof_metrics([row for row in canonical_rows if row["split"] == "development"], selected_text),
            "C1_METADATA_ALIAS_PROOF": _proof_metrics([row for row in alias_rows if row["split"] == "development"], selected_text),
        },
        "test": {
            "C0_CANONICAL_TEXT": _proof_metrics([row for row in canonical_rows if row["split"] == "test"], selected_text),
            "C1_METADATA_ALIAS_PROOF": _proof_metrics([row for row in alias_rows if row["split"] == "test"], selected_text),
        },
    }
    development_baseline_records = development_appearance[baseline]["records"]
    test_baseline_records = test_appearance["records"]
    join = {
        "development": _join_metrics(development_baseline_records, text_metrics["development"]["C1_METADATA_ALIAS_PROOF"]["records"]),
        "test": _join_metrics(test_baseline_records, text_metrics["test"]["C1_METADATA_ALIAS_PROOF"]["records"]),
    }
    baseline_test_fpr = test_appearance["wrong_building_false_confirmation"]
    join_test_fpr = join["test"]["source_label_negative_false_confirmation"]
    proof_test = text_metrics["test"]["C1_METADATA_ALIAS_PROOF"]
    proof_precision = proof_test["proof_precision"] >= 0.95 and proof_test["wrong_proofs"] == 0
    identity_coverage = proof_test["facets"]["identity-bearing"]["correct_proofs"] >= 3
    positive_guardrail = join["test"]["positive_acceptance"] + 1e-12 >= test_appearance["positive_acceptance"]
    relative_reduction = (baseline_test_fpr - join_test_fpr) / baseline_test_fpr if baseline_test_fpr > 0 else 0.0
    absolute_reduction = baseline_test_fpr - join_test_fpr
    false_effect = relative_reduction >= 0.50 - 1e-12 or absolute_reduction >= 0.10 - 1e-12
    passed = proof_precision and identity_coverage and positive_guardrail and false_effect
    decision = "L10_PB3_METADATA_BACKED_TEXT_IDENTITY_BRANCH_GATE_MET" if passed else "L10_PB3_METADATA_BACKED_TEXT_IDENTITY_BRANCH_GATE_NOT_MET"
    result = {
        "schema": "l10-named-poi-geolex-result-v1",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "inventory_sha256": manifest["inventory_sha256"],
        "selection": {"appearance_baseline": baseline, "appearance_threshold": thresholds[baseline], "text_thresholds": selected_text, "authority": "DEVELOPMENT_ONLY"},
        "development": {
            "appearance": {arm: pb2._compact(metrics) for arm, metrics in development_appearance.items()},
            "text": {arm: {key: value for key, value in metrics.items() if key != "records"} for arm, metrics in text_metrics["development"].items()},
            "join": {key: value for key, value in join["development"].items() if key != "records"},
        },
        "test": {
            "appearance": pb2._compact(test_appearance),
            "text": {arm: {key: value for key, value in metrics.items() if key != "records"} for arm, metrics in text_metrics["test"].items()},
            "join": {key: value for key, value in join["test"].items() if key != "records"},
        },
        "gate": {
            "proof_precision": proof_precision,
            "identity_bearing_coverage": identity_coverage,
            "positive_guardrail": positive_guardrail,
            "false_confirmation_effect": false_effect,
            "relative_false_confirmation_reduction": relative_reduction,
            "absolute_false_confirmation_reduction": absolute_reduction,
        },
        "backend": {"appearance": backend, "ocr": "CPUExecutionProvider"},
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
        "raw_records": {
            "development": {"appearance": development_baseline_records, "canonical_text": text_metrics["development"]["C0_CANONICAL_TEXT"]["records"], "alias_text": text_metrics["development"]["C1_METADATA_ALIAS_PROOF"]["records"], "join": join["development"]["records"]},
            "test": {"appearance": test_baseline_records, "canonical_text": text_metrics["test"]["C0_CANONICAL_TEXT"]["records"], "alias_text": text_metrics["test"]["C1_METADATA_ALIAS_PROOF"]["records"], "join": join["test"]["records"]},
        },
    }
    (output_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "named_poi_geolex_protocol_v1.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-geolex-v1")
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts.local/runtime/semantic-anchor-v1/models")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = run(args.protocol.resolve(), args.output_root.resolve(), args.models.resolve(), args.batch_size)
    print(json.dumps({"selection": result["selection"], "test": result["test"], "gate": result["gate"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
