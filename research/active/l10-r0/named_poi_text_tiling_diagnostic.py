"""Target-blind multi-scale OCR diagnostic on the consumed PB3 cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
from rapidocr import RapidOCR


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_geolex_eval as geolex  # noqa: E402
import named_poi_text_identity_opportunity as textaudit  # noqa: E402


def _target_blind_views(image: Any) -> list[Any]:
    height, width = image.shape[:2]
    tile_width = max(1, int(round(width * 0.60)))
    tile_height = max(1, int(round(height * 0.60)))
    views = [image]
    for y in (0, height - tile_height):
        for x in (0, width - tile_width):
            views.append(image[y : y + tile_height, x : x + tile_width])
    return views


def run(manifest_path: Path, formal_result_path: Path, output_path: Path, models: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal = json.loads(formal_result_path.read_text(encoding="utf-8"))
    thresholds = formal["selection"]["text_thresholds"]
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    entities_by_split = {
        split: [entity for entity in manifest["entities"] if entity["split"] == split]
        for split in ("development", "test")
    }
    rows = []
    for entity in manifest["entities"]:
        for query in entity["queries"]:
            image = cv2.imread(str(query["local_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(query["local_path"])
            texts: list[str] = []
            per_view = []
            for view_index, view in enumerate(_target_blind_views(image)):
                output = engine(view)
                current = [str(value) for value in (output.txts if output.txts is not None else [])]
                per_view.append({"view": view_index, "texts": current})
                texts.extend(current)
            texts = list(dict.fromkeys(texts))
            observations = textaudit._windows(texts)
            scores = {
                candidate["id"]: max(
                    (textaudit._similarity(alias, observations) for alias in candidate["aliases"]),
                    default=0.0,
                )
                for candidate in entities_by_split[entity["split"]]
            }
            ranked = sorted(scores, key=lambda entity_id: (-scores[entity_id], entity_id))
            rows.append(
                {
                    "key": f"query:{query['key']}",
                    "entity_id": entity["id"],
                    "split": entity["split"],
                    "facet": query["facet"],
                    "path": query["local_path"],
                    "ocr_texts": texts,
                    "per_view": per_view,
                    "entity_scores": scores,
                    "top_entity": ranked[0],
                    "top_score": scores[ranked[0]],
                    "second_score": scores[ranked[1]],
                    "margin": scores[ranked[0]] - scores[ranked[1]],
                }
            )
    metrics = {
        split: geolex._proof_metrics([row for row in rows if row["split"] == split], thresholds)
        for split in ("development", "test")
    }
    single = formal["test"]["text"]["C1_METADATA_ALIAS_PROOF"]
    tiled = metrics["test"]
    improved = tiled["correct_proofs"] > single["correct_proofs"] and tiled["wrong_proofs"] <= single["wrong_proofs"]
    payload = {
        "schema": "l10-named-poi-text-tiling-diagnostic-v1",
        "authority": "CONSUMED_PB3_DIAGNOSTIC_ONLY",
        "formal_result": str(formal_result_path.resolve()),
        "formal_result_sha256": geolex._sha256(formal_result_path),
        "frozen_from_formal": {
            "aliases": "unchanged",
            "score_threshold": thresholds["score_threshold"],
            "margin_threshold": thresholds["margin_threshold"],
            "ocr_models": "unchanged"
        },
        "only_change": "Target-blind full frame plus four fixed 60%-width/height corner tiles; no target-conditioned crop.",
        "development": {key: value for key, value in metrics["development"].items() if key != "records"},
        "test": {key: value for key, value in tiled.items() if key != "records"},
        "test_delta": {
            "correct_proofs": tiled["correct_proofs"] - single["correct_proofs"],
            "wrong_proofs": tiled["wrong_proofs"] - single["wrong_proofs"],
            "identity_bearing_correct_proofs": tiled["facets"]["identity-bearing"]["correct_proofs"] - single["facets"]["identity-bearing"]["correct_proofs"]
        },
        "decision": "PB3_FIXED_TILING_OBSERVABILITY_SIGNAL" if improved else "PB3_FIXED_TILING_OBSERVABILITY_GATE_NOT_MET",
        "records": {split: metrics[split]["records"] for split in ("development", "test")},
        "claim_boundary": "Post-result diagnostic on the consumed PB3 cohort. Fixed digital tiling is not a new physical view, commanded action, fresh confirmation, portal identity, navigation, or arrival evidence."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts.local/knowledge/named-poi-geolex-v1/dataset_manifest.json")
    parser.add_argument("--formal-result", type=Path, default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-geolex-v1/result.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-geolex-v1/tiling_diagnostic.json")
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts.local/runtime/semantic-anchor-v1/models")
    args = parser.parse_args()
    result = run(args.manifest.resolve(), args.formal_result.resolve(), args.output.resolve(), args.models.resolve())
    print(json.dumps({"development": result["development"], "test": result["test"], "test_delta": result["test_delta"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
