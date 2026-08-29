"""Evaluate PB4 Script-Contrastive Identity Lattice (SCIL) on a frozen cohort."""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from rapidocr import RapidOCR
from transformers import AutoImageProcessor, AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_facade_fingerprint as facade  # noqa: E402
import named_poi_geolex_eval as pb3  # noqa: E402
import named_poi_place_identity_eval as pb2  # noqa: E402


HAN_FOLD = str.maketrans(
    {
        "懲": "惩",
        "視": "视",
        "覺": "觉",
        "藝": "艺",
        "專": "专",
        "設": "设",
        "計": "计",
        "學": "学",
        "戲": "戏",
        "濕": "湿",
        "國": "国",
        "際": "际",
        "覽": "览",
        "館": "馆",
        "會": "会",
        "創": "创",
        "馬": "马",
        "賽": "赛",
        "區": "区",
        "處": "处",
        "號": "号",
        "臺": "台",
        "廣": "广",
        "門": "门",
        "業": "业",
    }
)


def _latin_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", normalized.lower())


def _latin_key(value: str) -> str:
    return " ".join(_latin_tokens(value))


def _is_han(character: str) -> bool:
    code = ord(character)
    return 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF


def _han_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(HAN_FOLD)
    return "".join(character for character in normalized if _is_han(character))


def _windows(lines: list[str], maximum: int = 4) -> list[str]:
    cleaned = [unicodedata.normalize("NFKC", line).strip() for line in lines]
    cleaned = [line for line in cleaned if line]
    values = set(cleaned)
    for start in range(len(cleaned)):
        for width in range(2, min(maximum, len(cleaned) - start) + 1):
            values.add(" ".join(cleaned[start : start + width]))
    return sorted(values)


def _sequence_score(alias: str, observations: list[str], script: str) -> float:
    normalize = _han_key if script == "han" else _latin_key
    target = normalize(alias)
    if not target:
        return 0.0
    return max(
        (difflib.SequenceMatcher(None, target, normalize(row)).ratio() for row in observations),
        default=0.0,
    )


def _han_atoms(value: str) -> dict[str, float]:
    key = _han_key(value)
    atoms = {f"u:{character}": 0.35 for character in key}
    atoms.update({f"b:{key[index:index + 2]}": 1.0 for index in range(len(key) - 1)})
    return atoms


def _latin_atoms(value: str) -> dict[str, float]:
    return {
        f"w:{token}": 0.65 + min(len(token), 10) / 10.0
        for token in _latin_tokens(value)
        if len(token) >= 2
    }


def _mark_atoms(value: str) -> dict[str, float]:
    key = "".join(_latin_tokens(value))
    return {f"m:{key}": 2.0} if len(key) >= 2 else {}


def _observed_atoms(observations: list[str]) -> set[str]:
    atoms: set[str] = set()
    for observation in observations:
        atoms.update(_latin_atoms(observation))
        atoms.update(_han_atoms(observation))
        compact = "".join(_latin_tokens(observation))
        if len(compact) >= 2:
            atoms.add(f"m:{compact}")
        for token in _latin_tokens(observation):
            if len(token) >= 2:
                atoms.add(f"m:{token}")
    return atoms


def _alias_atoms(alias: str, script: str) -> dict[str, float]:
    if script == "han":
        return _han_atoms(alias)
    if script == "mark":
        return _mark_atoms(alias)
    return _latin_atoms(alias)


def _profiles(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    atom_documents: Counter[str] = Counter()
    raw: dict[str, dict[str, Any]] = {}
    for entity in entities:
        scripts: dict[str, list[dict[str, float]]] = {}
        document_atoms: set[str] = set()
        for script in ("latin", "han", "marks"):
            canonical_script = "mark" if script == "marks" else script
            script_aliases = []
            for alias in entity["aliases"][script]:
                atoms = _alias_atoms(alias, canonical_script)
                if atoms:
                    script_aliases.append(atoms)
                    document_atoms.update(atoms)
            scripts[canonical_script] = script_aliases
        raw[entity["id"]] = {"entity": entity, "atoms": scripts}
        atom_documents.update(document_atoms)

    count = len(entities)
    for profile in raw.values():
        weighted: dict[str, list[dict[str, float]]] = {}
        for script, aliases in profile["atoms"].items():
            weighted[script] = []
            for alias in aliases:
                weighted[script].append(
                    {
                        atom: base * (1.0 + math.log((count + 1.0) / (atom_documents[atom] + 0.5)))
                        for atom, base in alias.items()
                    }
                )
        profile["atoms"] = weighted
    return raw


def _coverage(weighted_atoms: dict[str, float], observed: set[str]) -> float:
    total = sum(weighted_atoms.values())
    if total <= 0:
        return 0.0
    return sum(weight for atom, weight in weighted_atoms.items() if atom in observed) / total


def _carrier_score(
    aliases: list[str],
    weighted_aliases: list[dict[str, float]],
    observations: list[str],
    observed_atoms: set[str],
    script: str,
) -> float:
    best = 0.0
    for alias, atoms in zip(aliases, weighted_aliases, strict=True):
        sequence = _sequence_score(alias, observations, "latin" if script == "mark" else script)
        atom_coverage = _coverage(atoms, observed_atoms)
        if script == "mark":
            score = 1.0 if atom_coverage >= 1.0 else 0.35 * sequence
        else:
            score = 0.45 * sequence + 0.55 * atom_coverage
        best = max(best, score)
    return min(1.0, best)


def _flat_score(entity: dict[str, Any], observations: list[str], canonical_only: bool) -> float:
    if canonical_only:
        return _sequence_score(entity["name"], observations, "latin")
    scores = [
        _sequence_score(alias, observations, "latin")
        for alias in entity["aliases"]["latin"]
    ]
    scores.extend(_sequence_score(alias, observations, "han") for alias in entity["aliases"]["han"])
    scores.extend(_sequence_score(alias, observations, "latin") for alias in entity["aliases"]["marks"])
    return max(scores, default=0.0)


def _lattice_scores(
    entities: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    observations: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], bool]:
    observed_atoms = _observed_atoms(observations)
    carrier_scores: dict[str, dict[str, float]] = {}
    fused: dict[str, float] = {}
    for entity in entities:
        profile = profiles[entity["id"]]
        carriers = {
            "LATIN": _carrier_score(
                entity["aliases"]["latin"], profile["atoms"]["latin"], observations, observed_atoms, "latin"
            ),
            "HAN": _carrier_score(
                entity["aliases"]["han"], profile["atoms"]["han"], observations, observed_atoms, "han"
            ),
            "MARK": _carrier_score(
                entity["aliases"]["marks"], profile["atoms"]["mark"], observations, observed_atoms, "mark"
            ),
        }
        ranked = sorted(carriers.values(), reverse=True)
        fused[entity["id"]] = min(1.0, ranked[0] + 0.10 * ranked[1])
        carrier_scores[entity["id"]] = carriers

    strong_leaders = []
    for carrier in ("LATIN", "HAN", "MARK"):
        leader = max(entities, key=lambda entity: (carrier_scores[entity["id"]][carrier], entity["id"]))
        score = carrier_scores[leader["id"]][carrier]
        if score >= 0.72:
            strong_leaders.append(leader["id"])
    conflict = len(set(strong_leaders)) > 1
    if conflict:
        fused = {entity_id: 0.0 for entity_id in fused}
    return fused, carrier_scores, conflict


def _text_rows(
    queries: list[dict[str, Any]],
    manifest: dict[str, Any],
    engine: RapidOCR,
    arm: str,
    ocr_cache: dict[str, list[str]],
) -> list[dict[str, Any]]:
    entities_by_split = {
        split: [entity for entity in manifest["entities"] if entity["split"] == split]
        for split in ("development", "test")
    }
    profiles = {split: _profiles(entities) for split, entities in entities_by_split.items()}
    rows = []
    for query in queries:
        texts = ocr_cache.get(query["key"])
        if texts is None:
            image = cv2.imread(str(query["path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(query["path"])
            output = engine(image)
            texts = [str(value) for value in (output.txts if output.txts is not None else [])]
            ocr_cache[query["key"]] = texts
        observations = _windows(texts)
        entities = entities_by_split[query["split"]]
        carrier_scores: dict[str, dict[str, float]] = {}
        conflict = False
        if arm == "C2_SCIL":
            scores, carrier_scores, conflict = _lattice_scores(entities, profiles[query["split"]], observations)
        else:
            scores = {
                entity["id"]: _flat_score(entity, observations, canonical_only=arm == "C0_ENGLISH_CANONICAL")
                for entity in entities
            }
        ranked = sorted(scores, key=lambda entity_id: (-scores[entity_id], entity_id))
        top, second = ranked[:2]
        top_carrier = None
        strong_script_count = 0
        if carrier_scores:
            top_carrier = max(carrier_scores[top], key=lambda carrier: (carrier_scores[top][carrier], carrier))
            strong_script_count = sum(score >= 0.45 for score in carrier_scores[top].values())
        rows.append(
            {
                **query,
                "ocr_texts": texts,
                "entity_scores": scores,
                "carrier_scores": carrier_scores,
                "carrier_conflict": conflict,
                "top_carrier": top_carrier,
                "strong_script_count": strong_script_count,
                "top_entity": top,
                "top_score": scores[top],
                "second_score": scores[second],
                "margin": scores[top] - scores[second],
            }
        )
    return rows


def _proof_metrics(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    result = pb3._proof_metrics(rows, thresholds)
    by_key = {row["key"]: row for row in rows}
    for record in result["records"]:
        source = by_key[record["key"]]
        record["carrier_scores"] = source["carrier_scores"]
        record["carrier_conflict"] = source["carrier_conflict"]
        record["top_carrier"] = source["top_carrier"]
        record["strong_script_count"] = source["strong_script_count"]
    result["conflicts"] = sum(row["carrier_conflict"] for row in rows)
    result["proofs_with_two_or_more_scripts"] = sum(
        record["proof"] is not None and record["strong_script_count"] >= 2
        for record in result["records"]
    )
    return result


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in manifest["entities"]:
        for index, image in enumerate(entity["references"], start=1):
            path = Path(image["local_path"])
            pb3._require_hash(path, image["sha256"], f"REFERENCE:{entity['id']}:{index}")
            rows.append(
                {
                    "key": f"ref:{entity['split']}:{entity['id']}:{index:02d}",
                    "entity_id": entity["id"],
                    "split": entity["split"],
                    "role": "reference",
                    "facet": None,
                    "path": path,
                }
            )
        for image in entity["queries"]:
            path = Path(image["local_path"])
            pb3._require_hash(path, image["sha256"], f"QUERY:{image['key']}")
            rows.append(
                {
                    "key": f"query:{image['key']}",
                    "entity_id": entity["id"],
                    "split": entity["split"],
                    "role": "query",
                    "facet": image["facet"],
                    "path": path,
                }
            )
    return rows


def _appearance(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    output_root: Path,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_entities = {
        split: {row["entity_id"] for row in rows if row["split"] == split}
        for split in ("development", "test")
    }
    if split_entities["development"] & split_entities["test"]:
        raise ValueError("ENTITY_SPLIT_OVERLAP")
    queries = {
        split: [row for row in rows if row["split"] == split and row["role"] == "query"]
        for split in ("development", "test")
    }
    clip_path = ROOT / protocol["runtime"]["appearance_models"]["clip"]
    dino_path = ROOT / protocol["runtime"]["appearance_models"]["dinov2"]
    pb3._require_hash(
        clip_path / "pytorch_model.bin",
        protocol["runtime"]["appearance_models"]["clip_weights_sha256"],
        "CLIP",
    )
    pb3._require_hash(
        dino_path / "model.safetensors",
        protocol["runtime"]["appearance_models"]["dinov2_weights_sha256"],
        "DINOV2",
    )
    representative_images = [Image.open(row["path"]).convert("RGB") for row in rows if row["split"] == "development"][:8]
    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative = (
        clip_processor(images=representative_images, return_tensors="pt")["pixel_values"],
        dino_processor(images=representative_images, return_tensors="pt")["pixel_values"],
    )
    backend, models = facade._select_backend(
        clip_path, dino_path, representative, output_root / "appearance_backend_receipt.json"
    )
    image_rows = [
        facade.ImageRow(row["key"], row["entity_id"], row["role"], row["path"], pb3._sha256(row["path"]), "")
        for row in rows
    ]
    encoded = facade._encode_images(
        image_rows,
        models,
        clip_processor,
        dino_processor,
        backend["selected_device_type"],
        patch_grid=9,
        batch_size=batch_size,
    )
    descriptors = {
        "B0_CLIP": {key: value["clip"] for key, value in encoded.items()},
        "B1_DINOv2": {key: value["dino"] for key, value in encoded.items()},
    }
    del models, encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for split in ("development", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        entity_ids = sorted(split_entities[split])
        references = {
            entity: [row["key"] for row in split_rows if row["entity_id"] == entity and row["role"] == "reference"]
            for entity in entity_ids
        }
        scores[split] = {arm: {} for arm in ("B0_CLIP", "B1_DINOv2", "B2_CLIP_DINO")}
        for query in queries[split]:
            for arm in ("B0_CLIP", "B1_DINOv2"):
                vector = descriptors[arm][query["key"]]
                scores[split][arm][query["key"]] = {
                    entity: max(float(vector @ descriptors[arm][reference]) for reference in references[entity])
                    for entity in entity_ids
                }
            clip = pb2._zscore(scores[split]["B0_CLIP"][query["key"]])
            dino = pb2._zscore(scores[split]["B1_DINOv2"][query["key"]])
            scores[split]["B2_CLIP_DINO"][query["key"]] = {
                entity: 0.5 * (clip[entity] + dino[entity]) for entity in entity_ids
            }

    thresholds: dict[str, dict[str, Any]] = {}
    development: dict[str, dict[str, Any]] = {}
    for arm in ("B0_CLIP", "B1_DINOv2", "B2_CLIP_DINO"):
        positive = [scores["development"][arm][row["key"]][row["entity_id"]] for row in queries["development"]]
        negative = [
            score
            for row in queries["development"]
            for entity, score in scores["development"][arm][row["key"]].items()
            if entity != row["entity_id"]
        ]
        thresholds[arm] = pb2._threshold(positive, negative)
        development[arm] = pb2._metrics(
            queries["development"], scores["development"][arm], thresholds[arm]["threshold"]
        )
    baseline = max(development, key=lambda arm: (pb2._selection_key(development[arm]), arm))
    test = pb2._metrics(
        queries["test"], scores["test"][baseline], thresholds[baseline]["threshold"]
    )
    return (
        {
            "baseline": baseline,
            "threshold": thresholds[baseline],
            "development": development,
            "test": test,
        },
        backend,
        queries,
    )


def run(protocol_path: Path, output_root: Path, models: Path, batch_size: int) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for label, source in protocol["sources"].items():
        pb3._require_hash(ROOT / source["path"], source["sha256"], label.upper())
    if protocol["evaluator"]["sha256"] != pb3._sha256(Path(__file__)):
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    manifest_path = ROOT / protocol["sources"]["dataset_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["inventory_sha256"] != protocol["sources"]["dataset_manifest"]["inventory_sha256"]:
        raise ValueError("DATASET_INVENTORY_HASH_MISMATCH")
    audit = json.loads((ROOT / protocol["sources"]["source_audit"]["path"]).read_text(encoding="utf-8"))
    if audit["checks"]["ocr_calls_before_freeze"] != 0 or audit["checks"]["embedding_calls_before_freeze"] != 0:
        raise ValueError("SOURCE_AUDIT_PRE_FREEZE_MODEL_CALLS_NOT_ZERO")
    for filename, expected in protocol["runtime"]["ocr_models"].items():
        pb3._require_hash(models / filename, expected, f"OCR:{filename}")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = _rows(manifest)
    appearance, backend, queries = _appearance(rows, protocol, output_root, batch_size)

    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    all_queries = queries["development"] + queries["test"]
    ocr_cache: dict[str, list[str]] = {}
    arms = ("C0_ENGLISH_CANONICAL", "C1_FLAT_BISCRIPT_ALIAS", "C2_SCIL")
    text_rows = {arm: _text_rows(all_queries, manifest, engine, arm, ocr_cache) for arm in arms}
    thresholds = {
        arm: pb3._select_text_thresholds([row for row in text_rows[arm] if row["split"] == "development"])
        for arm in arms
    }
    text_metrics = {
        split: {
            arm: _proof_metrics([row for row in text_rows[arm] if row["split"] == split], thresholds[arm])
            for arm in arms
        }
        for split in ("development", "test")
    }

    join = {
        "development": pb3._join_metrics(
            appearance["development"][appearance["baseline"]]["records"],
            text_metrics["development"]["C2_SCIL"]["records"],
        ),
        "test": pb3._join_metrics(
            appearance["test"]["records"], text_metrics["test"]["C2_SCIL"]["records"]
        ),
    }
    lattice_test = text_metrics["test"]["C2_SCIL"]
    flat_test = text_metrics["test"]["C1_FLAT_BISCRIPT_ALIAS"]
    canonical_test = text_metrics["test"]["C0_ENGLISH_CANONICAL"]
    baseline_fpr = appearance["test"]["wrong_building_false_confirmation"]
    join_fpr = join["test"]["source_label_negative_false_confirmation"]
    relative_reduction = (baseline_fpr - join_fpr) / baseline_fpr if baseline_fpr > 0 else 0.0
    absolute_reduction = baseline_fpr - join_fpr
    correct_lattice = [row for row in lattice_test["records"] if row["correct"] is True]
    script_participation = sum(
        row["carrier_scores"].get(row["proof"], {}).get("HAN", 0.0) >= 0.45
        or row["carrier_scores"].get(row["proof"], {}).get("MARK", 0.0) >= 0.45
        for row in correct_lattice
    )
    proof_precision = lattice_test["proof_precision"] >= 0.95 and lattice_test["wrong_proofs"] == 0
    identity_coverage = lattice_test["facets"]["identity-bearing"]["correct_proofs"] >= 3
    positive_guardrail = join["test"]["positive_acceptance"] + 1e-12 >= appearance["test"]["positive_acceptance"]
    false_effect = relative_reduction >= 0.50 - 1e-12 or absolute_reduction >= 0.10 - 1e-12
    flat_pareto = (
        lattice_test["correct_proofs"] >= flat_test["correct_proofs"]
        and lattice_test["wrong_proofs"] <= flat_test["wrong_proofs"]
    )
    script_utility = script_participation >= 1
    passed = proof_precision and identity_coverage and positive_guardrail and false_effect and flat_pareto and script_utility
    decision = (
        "L10_PB4_SCRIPT_CONTRASTIVE_IDENTITY_LATTICE_GATE_MET"
        if passed
        else "L10_PB4_SCRIPT_CONTRASTIVE_IDENTITY_LATTICE_GATE_NOT_MET"
    )

    result = {
        "schema": "l10-named-poi-biscript-result-v1",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": pb3._sha256(protocol_path),
        "dataset_manifest_sha256": pb3._sha256(manifest_path),
        "inventory_sha256": manifest["inventory_sha256"],
        "selection": {
            "appearance_baseline": appearance["baseline"],
            "appearance_threshold": appearance["threshold"],
            "text_thresholds": thresholds,
            "authority": "DEVELOPMENT_ONLY",
        },
        "development": {
            "appearance": {arm: pb2._compact(metrics) for arm, metrics in appearance["development"].items()},
            "text": {arm: {key: value for key, value in metrics.items() if key != "records"} for arm, metrics in text_metrics["development"].items()},
            "join": {key: value for key, value in join["development"].items() if key != "records"},
        },
        "test": {
            "appearance": pb2._compact(appearance["test"]),
            "text": {arm: {key: value for key, value in metrics.items() if key != "records"} for arm, metrics in text_metrics["test"].items()},
            "join": {key: value for key, value in join["test"].items() if key != "records"},
            "script_participating_correct_proofs": script_participation,
        },
        "gate": {
            "proof_precision": proof_precision,
            "identity_bearing_coverage": identity_coverage,
            "positive_guardrail": positive_guardrail,
            "false_confirmation_effect": false_effect,
            "flat_alias_pareto": flat_pareto,
            "han_or_mark_script_utility": script_utility,
            "relative_false_confirmation_reduction": relative_reduction,
            "absolute_false_confirmation_reduction": absolute_reduction,
        },
        "backend": {"appearance": backend, "ocr": "CPUExecutionProvider"},
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
        "raw_records": {
            split: {
                "appearance": appearance["development"][appearance["baseline"]]["records"] if split == "development" else appearance["test"]["records"],
                **{arm: text_metrics[split][arm]["records"] for arm in arms},
                "join": join[split]["records"],
            }
            for split in ("development", "test")
        },
        "comparisons": {
            "canonical_correct_proofs": canonical_test["correct_proofs"],
            "flat_biscript_correct_proofs": flat_test["correct_proofs"],
            "scil_correct_proofs": lattice_test["correct_proofs"],
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "named_poi_biscript_protocol_v1.json"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-biscript-v1",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts.local/runtime/semantic-anchor-v1/models",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = run(args.protocol.resolve(), args.output_root.resolve(), args.models.resolve(), args.batch_size)
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "test": result["test"],
                "gate": result["gate"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
