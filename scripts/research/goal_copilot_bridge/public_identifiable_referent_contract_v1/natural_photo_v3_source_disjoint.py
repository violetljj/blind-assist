"""Freeze and evaluate the natural-photo source-disjoint SAGE-R V3-A cohort.

The freeze command materializes deterministic views and seals every input hash
before OCR or any model is opened.  The evaluate command verifies the seal,
runs RapidOCR once per observation, and evaluates exactly four frozen arms:
substring FSM, V2 heuristic, V3 no-relation, and V3 full typed relations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from .semantic_anchor_graph_and_belief_v2 import (
    Box,
    Candidate,
    Frame,
    ReferentBelief,
    SubstringFsmBaseline,
    TargetGraph,
    Token,
    graph_candidate_scores,
    normalize_text,
)
from .semantic_anchor_graph_and_belief_v2_1_real_ocr import (
    RapidOcrPolygonAdapter,
    _measured_blur,
    _split_line_tokens,
)
from .typed_semantic_referent_graph_v3 import (
    TypedSemanticReferentGraphNet,
    predict_scores,
)


EXPERIMENT_ID = "SAGE_R_V3_A_NATURAL_PHOTO_SOURCE_DISJOINT"
SCHEMA_VERSION = "blindassist_sage_r_v3_a_natural_photo_source_disjoint_v1"
FROZEN_V3_COMMIT = "188b4b2744f134bcccf43efd3e0443ce7581afb5"
FROZEN_V2_SHA256 = "c6ecb7625c996982b04134c2824c8efe6026dab7da49b4c13c3a4893a28b95cb"
FROZEN_MODEL_SHA256 = {
    "v3_no_relation": "4fb110b8518dec4800e9a7b113e84aaa56a5a6a065685105742a7c985c8cf5a2",
    "v3_full": "926cc835b9a33499fac210272723b2c986aab19aabfc0b08e52f154f47a8b454",
}
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
CLAIM_CEILING = (
    "NINE_NATURAL_COMMONS_SOURCES_TWELVE_CORRELATED_QUERIES_THIRTY_SIX_DERIVED_"
    "OBSERVATIONS_SOURCE_DISJOINT_DEVELOPMENT_NO_POPULATION_GENERALIZATION_"
    "LEARNED_BELIEF_ACTIVE_PERCEPTION_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _base_homography(width: int, height: int) -> np.ndarray:
    scale = min(CANVAS_WIDTH / width, CANVAS_HEIGHT / height)
    offset_x = (CANVAS_WIDTH - width * scale) / 2
    offset_y = (CANVAS_HEIGHT - height * scale) / 2
    return np.array([[scale, 0.0, offset_x], [0.0, scale, offset_y], [0.0, 0.0, 1.0]], dtype=np.float32)


def _view_homography(width: int, height: int, view: Mapping[str, Any]) -> np.ndarray:
    base = _base_homography(width, height)
    scale = float(view["scale"])
    if scale != 1.0:
        shift_x = CANVAS_WIDTH * (1.0 - scale) / 2
        shift_y = CANVAS_HEIGHT * (1.0 - scale) / 2
        base = np.array(
            [[scale, 0.0, shift_x], [0.0, scale, shift_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ) @ base
    perspective = float(view["perspective"])
    if perspective <= 0:
        return base
    skew = CANVAS_WIDTH * 0.09 * perspective / 0.18
    source = np.float32(
        [[0, 0], [CANVAS_WIDTH - 1, 0], [CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1], [0, CANVAS_HEIGHT - 1]]
    )
    target = np.float32(
        [[skew, 12], [CANVAS_WIDTH - 1 - skew, 0], [CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1], [0, CANVAS_HEIGHT - 25]]
    )
    return cv2.getPerspectiveTransform(source, target) @ base


def _transform_box(box: Sequence[float], width: int, height: int, homography: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = map(float, box)
    corners = np.float32([[[x0 * width, y0 * height], [x1 * width, y0 * height], [x1 * width, y1 * height], [x0 * width, y1 * height]]])
    mapped = cv2.perspectiveTransform(corners, homography)[0]
    return [
        float(np.clip(mapped[:, 0].min() / CANVAS_WIDTH, 0.0, 1.0)),
        float(np.clip(mapped[:, 1].min() / CANVAS_HEIGHT, 0.0, 1.0)),
        float(np.clip(mapped[:, 0].max() / CANVAS_WIDTH, 0.0, 1.0)),
        float(np.clip(mapped[:, 1].max() / CANVAS_HEIGHT, 0.0, 1.0)),
    ]


def _materialize_view(image: np.ndarray, view: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    homography = _view_homography(width, height, view)
    output = cv2.warpPerspective(
        image,
        homography,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_AREA,
        borderValue=(196, 196, 196),
    )
    sigma = float(view["blur_sigma"])
    if sigma > 0:
        kernel = max(3, int(round(sigma * 4)) | 1)
        output = cv2.GaussianBlur(output, (kernel, kernel), sigma)
    return output, homography


def freeze_cohort(spec_path: Path, source_dir: Path, run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite frozen cohort: {run_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["frozen_v3_commit"] != FROZEN_V3_COMMIT:
        raise ValueError("cohort V3 commit drift")
    run_dir.mkdir(parents=True)
    source_output = run_dir / "sources"
    observation_output = run_dir / "observations"
    source_output.mkdir()
    observation_output.mkdir()
    sources: dict[str, dict[str, Any]] = {}
    for item in spec["sources"]:
        original = source_dir / item["local_name"]
        if not original.is_file():
            raise FileNotFoundError(original)
        image = cv2.imread(str(original), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"not a readable image: {original}")
        destination = source_output / item["local_name"]
        shutil.copy2(original, destination)
        height, width = image.shape[:2]
        sources[item["source_id"]] = {
            **item,
            "path": str(destination.resolve()),
            "sha256": _sha256_file(destination),
            "width": width,
            "height": height,
        }

    observations = []
    for query in spec["queries"]:
        source = sources[query["source_id"]]
        image = cv2.imread(source["path"], cv2.IMREAD_COLOR)
        for frame_index, view in enumerate(spec["views"]):
            output, homography = _materialize_view(image, view)
            name = f"{query['query_id']}-{frame_index:02d}-{view['view_id']}.jpg"
            path = observation_output / name
            if not cv2.imwrite(str(path), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"failed to write {path}")
            boxes = {
                candidate_id: _transform_box(box, source["width"], source["height"], homography)
                for candidate_id, box in source["candidate_boxes"].items()
            }
            observations.append(
                {
                    **query,
                    "frame_index": frame_index,
                    "view": dict(view),
                    "image_path": str(path.resolve()),
                    "image_sha256": _sha256_file(path),
                    "candidate_boxes": boxes,
                }
            )

    module_path = Path(inspect.getsourcefile(freeze_cohort) or "").resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": _utc_now(),
        "frozen_before_ocr_or_model_open": True,
        "repository_head_at_freeze": _head(),
        "frozen_v3_commit": FROZEN_V3_COMMIT,
        "cohort_spec_path": str(spec_path.resolve()),
        "cohort_spec_sha256": _sha256_file(spec_path),
        "freeze_and_evaluate_module_path": str(module_path),
        "freeze_and_evaluate_module_sha256": _sha256_file(module_path),
        "source_count": len(sources),
        "query_count": len(spec["queries"]),
        "observation_count": len(observations),
        "sources": list(sources.values()),
        "observations": observations,
        "data_role": spec["data_role"],
        "claim_boundary": spec["claim_boundary"],
    }
    manifest_path = run_dir / "cohort-manifest.json"
    _atomic_json(manifest_path, manifest)
    seal = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "observation_hash_count": len(observations),
        "ocr_calls_at_freeze": 0,
        "model_calls_at_freeze": 0,
    }
    _atomic_json(run_dir / "cohort-seal.json", seal)
    return seal


def _verify_manifest(cohort_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = cohort_dir / "cohort-manifest.json"
    seal_path = cohort_dir / "cohort-seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if _sha256_file(manifest_path) != seal["manifest_sha256"]:
        raise ValueError("cohort manifest seal mismatch")
    for source in manifest["sources"]:
        if _sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"source hash drift: {source['source_id']}")
    for row in manifest["observations"]:
        if _sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"observation hash drift: {row['query_id']} {row['frame_index']}")
    return manifest, seal


def _target_tokens_from_ocr(raw_lines: Sequence[Mapping[str, Any]], target: TargetGraph) -> tuple[Token, ...]:
    tokens = []
    target_tokens = tuple(normalize_text(value) for value in target.tokens)
    for raw in raw_lines:
        for text, box, confidence in _split_line_tokens(raw):
            normalized = normalize_text(text)
            adapted = text
            prefixes = [value for value in target_tokens if len(normalized) >= 2 and value.startswith(normalized) and value != normalized]
            if len(prefixes) == 1:
                adapted = normalized + "?" * (len(prefixes[0]) - len(normalized))
            tokens.append(Token(adapted, box, confidence))
    return tuple(tokens)


def _load_model(path: Path, expected_sha256: str, device: torch.device) -> TypedSemanticReferentGraphNet:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"frozen model hash mismatch for {path.name}: {actual}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TypedSemanticReferentGraphNet(hidden=int(checkpoint["hidden"]), layers=int(checkpoint["layers"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _is_correct(row: Mapping[str, Any], arm: str) -> bool:
    decision = row[arm]
    if row["truth"] == "NONE":
        return decision["state"] == "NONE"
    return decision["state"] == "TARGET" and decision["selected_candidate"] == row["truth"]


def _metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    targets = [row for row in rows if row["truth"] != "NONE"]
    evaluable = [row for row in targets if row["expected_state"] != "UNKNOWN"]
    absent = [row for row in rows if row["truth"] == "NONE"]
    final_rows = [row for row in rows if row["frame_index"] == 2]
    final_evaluable = [row for row in final_rows if row["truth"] != "NONE" and row["expected_state"] != "UNKNOWN"]
    final_absent = [row for row in final_rows if row["truth"] == "NONE"]
    wrong = lambda row: row[arm]["state"] == "TARGET" and row[arm]["selected_candidate"] != row["truth"]
    target_correct = sum(_is_correct(row, arm) for row in evaluable)
    none_correct = sum(_is_correct(row, arm) for row in absent)
    return {
        "observations": len(rows),
        "correct_terminal_observations": sum(_is_correct(row, arm) for row in rows),
        "evaluable_target_correct": target_correct,
        "evaluable_target_observations": len(evaluable),
        "target_recall_evaluable": target_correct / max(1, len(evaluable)),
        "none_correct": none_correct,
        "absent_observations": len(absent),
        "none_accuracy": none_correct / max(1, len(absent)),
        "wrong_locks": sum(wrong(row) for row in rows),
        "unknown_preserved": sum(row["expected_state"] == "UNKNOWN" and row[arm]["state"] == "UNKNOWN" for row in rows),
        "unknown_observations": sum(row["expected_state"] == "UNKNOWN" for row in rows),
        "directory_false_bindings": sum(row["scenario"] == "directory_trap" and wrong(row) for row in rows),
        "directory_correct": sum(row["scenario"] == "directory_trap" and _is_correct(row, arm) for row in rows),
        "hard_neighbor_wrong_locks": sum("hard_neighbor" in row["scenario"] and wrong(row) for row in rows),
        "final_query_correct": sum(_is_correct(row, arm) for row in final_rows),
        "final_queries": len(final_rows),
        "final_target_recall_evaluable": sum(_is_correct(row, arm) for row in final_evaluable) / max(1, len(final_evaluable)),
        "final_none_accuracy": sum(_is_correct(row, arm) for row in final_absent) / max(1, len(final_absent)),
    }


def evaluate_cohort(
    cohort_dir: Path,
    run_dir: Path,
    runtime_root: Path,
    no_relation_path: Path,
    full_path: Path,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite evaluation: {run_dir}")
    manifest, seal = _verify_manifest(cohort_dir)
    module_path = Path(inspect.getsourcefile(evaluate_cohort) or "").resolve()
    if _sha256_file(module_path) != manifest["freeze_and_evaluate_module_sha256"]:
        raise ValueError("freeze/evaluate module changed after cohort seal")
    v2_path = Path(inspect.getsourcefile(graph_candidate_scores) or "").resolve()
    if _sha256_file(v2_path) != FROZEN_V2_SHA256:
        raise ValueError("frozen V2 scorer/belief source hash drift")
    run_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_relation = _load_model(no_relation_path, FROZEN_MODEL_SHA256["v3_no_relation"], device)
    full = _load_model(full_path, FROZEN_MODEL_SHA256["v3_full"], device)
    ocr = RapidOcrPolygonAdapter(runtime_root)
    rows = []
    permutation_errors = {"v3_no_relation": [], "v3_full": []}
    query_ids = list(dict.fromkeys(row["query_id"] for row in manifest["observations"]))
    for query_id in query_ids:
        episode = [row for row in manifest["observations"] if row["query_id"] == query_id]
        target = TargetGraph(tuple(episode[0]["target_tokens"]))
        baseline = SubstringFsmBaseline(target)
        beliefs = {
            "v2_heuristic": ReferentBelief(("A", "B", "C")),
            "v3_no_relation": ReferentBelief(("A", "B", "C")),
            "v3_full": ReferentBelief(("A", "B", "C")),
        }
        for observation in episode:
            image = cv2.imread(observation["image_path"], cv2.IMREAD_COLOR)
            raw_ocr = ocr.read(image)
            tokens = _target_tokens_from_ocr(raw_ocr, target)
            blur, laplacian_variance = _measured_blur(image)
            candidates = tuple(
                Candidate(candidate_id, Box(*box))
                for candidate_id, box in sorted(observation["candidate_boxes"].items())
            )
            frame = Frame(
                query_id,
                int(observation["frame_index"]),
                str(observation["view"]["view_id"]),
                candidates,
                tokens,
                blur,
                float(observation["view"]["perspective"]),
                str(observation["truth"]),
                str(observation["expected_state"]),
                str(observation["note"]),
            )
            v2_scores = graph_candidate_scores(target, frame)
            no_scores, no_diagnostics = predict_scores(no_relation, target, frame, include_relations=False, device=device)
            full_scores, full_diagnostics = predict_scores(full, target, frame, include_relations=True, device=device)
            reversed_frame = Frame(
                frame.episode_id,
                frame.frame_index,
                frame.viewpoint,
                tuple(reversed(frame.candidates)),
                frame.tokens,
                frame.blur,
                frame.perspective,
                frame.truth,
                frame.expected_state,
                frame.note,
            )
            for name, model, include_relations, scores in (
                ("v3_no_relation", no_relation, False, no_scores),
                ("v3_full", full, True, full_scores),
            ):
                permuted, _ = predict_scores(model, target, reversed_frame, include_relations=include_relations, device=device)
                permutation_errors[name].append(max(abs(scores[key]["score"] - permuted[key]["score"]) for key in scores))
            rows.append(
                {
                    "query_id": query_id,
                    "source_id": observation["source_id"],
                    "frame_index": frame.frame_index,
                    "viewpoint": frame.viewpoint,
                    "target_tokens": list(target.tokens),
                    "truth": frame.truth,
                    "expected_state": frame.expected_state,
                    "scenario": observation["scenario"],
                    "note": frame.note,
                    "image_path": observation["image_path"],
                    "image_sha256": observation["image_sha256"],
                    "ocr_raw": raw_ocr,
                    "ocr_line_count": len(raw_ocr),
                    "laplacian_variance": laplacian_variance,
                    "frame": asdict(frame),
                    "substring_fsm": baseline.update(frame),
                    "v2_heuristic": beliefs["v2_heuristic"].update(frame, v2_scores),
                    "v3_no_relation": beliefs["v3_no_relation"].update(frame, no_scores),
                    "v3_full": beliefs["v3_full"].update(frame, full_scores),
                    "diagnostic_heads_not_used": {"v3_no_relation": no_diagnostics, "v3_full": full_diagnostics},
                }
            )

    metrics = {name: _metrics(rows, name) for name in ("substring_fsm", "v2_heuristic", "v3_no_relation", "v3_full")}
    for name in ("v3_no_relation", "v3_full"):
        errors = permutation_errors[name]
        metrics[name]["candidate_permutation_max_abs_error"] = max(errors, default=0.0)
        metrics[name]["candidate_permutation_invariant_observations"] = sum(error <= 1e-5 for error in errors)
    raw = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "cohort_manifest_sha256": seal["manifest_sha256"],
        "rows": rows,
    }
    raw_path = run_dir / "raw-decisions.json"
    _atomic_json(raw_path, raw)
    full_metrics = metrics["v3_full"]
    v2_metrics = metrics["v2_heuristic"]
    no_metrics = metrics["v3_no_relation"]
    promotion_checks = {
        "correct_observation_uplift_at_least_4": full_metrics["correct_terminal_observations"] >= v2_metrics["correct_terminal_observations"] + 4,
        "evaluable_target_uplift_at_least_3": full_metrics["evaluable_target_correct"] >= v2_metrics["evaluable_target_correct"] + 3,
        "wrong_locks_at_most_1": full_metrics["wrong_locks"] <= 1,
        "relation_target_uplift_at_least_6": full_metrics["evaluable_target_correct"] >= no_metrics["evaluable_target_correct"] + 6,
        "directory_false_binding_zero": full_metrics["directory_false_bindings"] == 0,
        "permutation_invariant_all": full_metrics["candidate_permutation_invariant_observations"] == len(rows),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "NATURAL_PHOTO_SOURCE_DISJOINT_DEVELOPMENT",
        "frozen_inputs": {
            "cohort_manifest_sha256": seal["manifest_sha256"],
            "frozen_v3_commit": FROZEN_V3_COMMIT,
            "v2_source_path": str(v2_path),
            "v2_source_sha256": FROZEN_V2_SHA256,
            "v3_no_relation_path": str(no_relation_path.resolve()),
            "v3_no_relation_sha256": FROZEN_MODEL_SHA256["v3_no_relation"],
            "v3_full_path": str(full_path.resolve()),
            "v3_full_sha256": FROZEN_MODEL_SHA256["v3_full"],
        },
        "cohort": {
            "independent_source_photos": manifest["source_count"],
            "target_queries": manifest["query_count"],
            "derived_observations": manifest["observation_count"],
            "query_source_correlation_is_not_independence": True,
            "source_licenses": [
                {key: source[key] for key in ("source_id", "page_url", "artist", "license", "license_url", "sha256")}
                for source in manifest["sources"]
            ],
        },
        "runtime": {"device": str(device), "torch": torch.__version__, "ocr": ocr.receipt()},
        "metrics": metrics,
        "decision": {
            "v3_full_vs_v2_correct_observation_delta": full_metrics["correct_terminal_observations"] - v2_metrics["correct_terminal_observations"],
            "v3_full_vs_v2_evaluable_target_delta": full_metrics["evaluable_target_correct"] - v2_metrics["evaluable_target_correct"],
            "relation_ablation_evaluable_target_delta": full_metrics["evaluable_target_correct"] - no_metrics["evaluable_target_correct"],
            "promotion_checks": promotion_checks,
            "v3_b_status": "ENTER_V3_B" if all(promotion_checks.values()) else "DO_NOT_ENTER_V3_B",
        },
        "interpretation_boundary": [
            "All source, query, truth, candidate-region, transformation, and image hashes were sealed before RapidOCR or model evaluation.",
            "The nine source photographs are naturally occurring Wikimedia Commons photographs and are source-disjoint from synthetic V3 training and generated-pixel V2.1 Development.",
            "The thirty-six observations are deterministic transformations of nine sources; they are not thirty-six independent photographs, and repeated queries on a source are correlated.",
            "Candidate boxes are predeclared physical destination regions, not detector outputs; this isolates semantic-referent scoring from proposal generation.",
            "No scorer, model weight, generator, V2 belief rule, novelty rule, NONE construction, or terminal threshold was changed.",
            "Only four arms ran. Learned diagnostic reliability/NONE heads did not drive decisions. V3-B and V4 did not run.",
            "This small curated Development cohort is mechanism evidence, not a population-level natural-photo generalization, calibration, navigation, safety, or product claim.",
        ],
        "claim_ceiling": CLAIM_CEILING,
        "raw_decisions_sha256": _sha256_file(raw_path),
    }
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--source-dir", type=Path, required=True)
    freeze.add_argument("--run-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cohort-dir", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--runtime-root", type=Path, required=True)
    evaluate.add_argument("--no-relation-model", type=Path, required=True)
    evaluate.add_argument("--full-model", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze_cohort(args.spec.resolve(), args.source_dir.resolve(), args.run_dir.resolve())
    else:
        result = evaluate_cohort(
            args.cohort_dir.resolve(),
            args.run_dir.resolve(),
            args.runtime_root.resolve(),
            args.no_relation_model.resolve(),
            args.full_model.resolve(),
        )
        result = {"metrics": result["metrics"], "decision": result["decision"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
