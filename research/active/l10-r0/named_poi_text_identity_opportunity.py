"""Audit whether scene text carries named-POI identity on an opened cohort.

This script is deliberately diagnostic: it reports canonical-name similarity
and margins but does not select an acceptance threshold or claim a fresh gate.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
from rapidocr import RapidOCR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def _windows(lines: list[str], maximum: int = 4) -> list[str]:
    normalized = [_normalize(line) for line in lines]
    normalized = [line for line in normalized if line]
    values = set(normalized)
    for start in range(len(normalized)):
        for width in range(2, min(maximum, len(normalized) - start) + 1):
            values.add(" ".join(normalized[start : start + width]))
    return sorted(values)


def _similarity(alias: str, observations: list[str]) -> float:
    target = _normalize(alias)
    if not target:
        return 0.0
    return max((difflib.SequenceMatcher(None, target, row).ratio() for row in observations), default=0.0)


def audit(manifest_path: Path, output_path: Path, models: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    names = {str(entity["id"]): str(entity["name"]) for entity in manifest["entities"]}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for entity in manifest["entities"]:
        split_ids = [
            str(candidate["id"])
            for candidate in manifest["entities"]
            if candidate["split"] == entity["split"]
        ]
        for query in entity["queries"]:
            image = cv2.imread(str(query["local_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(query["local_path"])
            before = time.perf_counter()
            result = engine(image)
            texts = [str(value) for value in (result.txts if result.txts is not None else [])]
            scores = [float(value) for value in (result.scores if result.scores is not None else [])]
            observations = _windows(texts)
            entity_scores = {
                entity_id: _similarity(names[entity_id], observations) for entity_id in split_ids
            }
            ranked = sorted(entity_scores, key=lambda key: (-entity_scores[key], key))
            true_id = str(entity["id"])
            true_score = entity_scores[true_id]
            strongest_wrong = max(
                (score for candidate, score in entity_scores.items() if candidate != true_id),
                default=0.0,
            )
            rows.append(
                {
                    "key": query["key"],
                    "split": entity["split"],
                    "facet": query["facet"],
                    "true_entity": true_id,
                    "ocr_texts": texts,
                    "ocr_confidences": scores,
                    "entity_scores": entity_scores,
                    "ranked_entities": ranked,
                    "true_score": true_score,
                    "strongest_wrong_score": strongest_wrong,
                    "true_margin": true_score - strongest_wrong,
                    "true_rank": ranked.index(true_id) + 1,
                    "wall_s": time.perf_counter() - before,
                }
            )

    summaries: dict[str, Any] = {}
    for split in ("development", "test"):
        selected = [row for row in rows if row["split"] == split]
        summaries[split] = {
            "queries": len(selected),
            "text_detections": sum(bool(row["ocr_texts"]) for row in selected),
            "identity_rank1": sum(row["true_rank"] == 1 for row in selected),
            "positive_margin": sum(row["true_margin"] > 0 for row in selected),
            "true_score_ge_0_80": sum(row["true_score"] >= 0.80 for row in selected),
            "true_score_ge_0_90": sum(row["true_score"] >= 0.90 for row in selected),
            "wrong_score_ge_0_80": sum(row["strongest_wrong_score"] >= 0.80 for row in selected),
            "wrong_score_ge_0_90": sum(row["strongest_wrong_score"] >= 0.90 for row in selected),
        }
    payload = {
        "schema": "l10-named-poi-text-identity-opportunity-v1",
        "authority": "CONSUMED_PB2A_DIAGNOSTIC_ONLY",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "backend": "RapidOCR 3.9.2 / ONNX Runtime CPUExecutionProvider",
        "matching": "canonical English name vs OCR line/adjacent-line windows; SequenceMatcher ratio",
        "threshold_policy": "NONE; fixed score landmarks are descriptive only",
        "wall_s": time.perf_counter() - started,
        "summary": summaries,
        "rows": rows,
        "claim_boundary": (
            "Opportunity audit on the already-consumed PB2-A public-image cohort. "
            "It may justify a fresh source experiment but cannot select a threshold, "
            "promote an identity branch, or establish portal, navigation, or arrival authority."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.manifest, args.output, args.models)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
