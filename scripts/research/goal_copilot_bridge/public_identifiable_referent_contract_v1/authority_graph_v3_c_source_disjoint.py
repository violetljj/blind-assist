"""Freeze and evaluate the fresh natural-photo SAGE-R V3-C cohort."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import torch

from .authority_typed_sign_destination_graph_v3_c import (
    SignCarrier,
    _atomic_json,
    _sha256_file,
    _utc_now,
    load_checkpoint,
    score_candidates,
)
from .natural_photo_v3_source_disjoint import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    FROZEN_MODEL_SHA256,
    _load_model,
    _materialize_view,
    _metrics,
    _target_tokens_from_ocr,
    _transform_box,
    _view_homography,
)
from .semantic_anchor_graph_and_belief_v2 import Box, Candidate, Frame, ReferentBelief, TargetGraph, graph_candidate_scores
from .semantic_anchor_graph_and_belief_v2_1_real_ocr import RapidOcrPolygonAdapter, _measured_blur
from .typed_semantic_referent_graph_v3 import predict_scores


EXPERIMENT_ID = "SAGE_R_V3_C_FRESH_NATURAL_SOURCE_DISJOINT"
SCHEMA_VERSION = "blindassist_sage_r_v3_c_fresh_natural_source_disjoint_v1"


def freeze_cohort(spec_path: Path, source_dir: Path, run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite cohort: {run_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True)
    (run_dir / "sources").mkdir()
    (run_dir / "observations").mkdir()
    sources = {}
    for source in spec["sources"]:
        original = source_dir / source["local_name"]
        image = cv2.imread(str(original), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable source: {original}")
        destination = run_dir / "sources" / source["local_name"]
        shutil.copy2(original, destination)
        height, width = image.shape[:2]
        sources[source["source_id"]] = {
            **source, "path": str(destination.resolve()), "sha256": _sha256_file(destination),
            "width": width, "height": height,
        }
    observations = []
    for query in spec["queries"]:
        source = sources[query["source_id"]]
        image = cv2.imread(source["path"], cv2.IMREAD_COLOR)
        for frame_index, view in enumerate(spec["views"]):
            output, homography = _materialize_view(image, view)
            path = run_dir / "observations" / f"{query['query_id']}-{frame_index:02d}-{view['view_id']}.jpg"
            if not cv2.imwrite(str(path), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(path)
            transform = lambda value: _transform_box(value, source["width"], source["height"], homography)
            observations.append({
                **query, "frame_index": frame_index, "view": view,
                "image_path": str(path.resolve()), "image_sha256": _sha256_file(path),
                "candidate_boxes": {key: transform(value) for key, value in source["candidate_boxes"].items()},
                "carriers": [{**item, "box": transform(item["box"])} for item in source["carriers"]],
                "cues": [{**item, "box": transform(item["box"])} for item in source.get("cues", [])],
            })
    module_path = Path(inspect.getsourcefile(freeze_cohort) or "").resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": _utc_now(), "frozen_before_ocr_or_model_open": True,
        "spec_path": str(spec_path.resolve()), "spec_sha256": _sha256_file(spec_path),
        "module_path": str(module_path), "module_sha256": _sha256_file(module_path),
        "source_count": len(sources), "query_count": len(spec["queries"]),
        "observation_count": len(observations), "sources": list(sources.values()),
        "observations": observations, "models": spec["models"],
        "data_role": spec["data_role"], "claim_boundary": spec["claim_boundary"],
    }
    manifest_path = run_dir / "cohort-manifest.json"
    _atomic_json(manifest_path, manifest)
    seal = {"manifest_sha256": _sha256_file(manifest_path), "ocr_calls_at_freeze": 0, "model_calls_at_freeze": 0}
    _atomic_json(run_dir / "cohort-seal.json", seal)
    return seal


def _verify(cohort_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = cohort_dir / "cohort-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads((cohort_dir / "cohort-seal.json").read_text(encoding="utf-8"))
    if _sha256_file(manifest_path) != seal["manifest_sha256"]:
        raise ValueError("manifest seal mismatch")
    module_path = Path(inspect.getsourcefile(_verify) or "").resolve()
    if _sha256_file(module_path) != manifest["module_sha256"]:
        raise ValueError("freeze/evaluate module changed after freeze")
    for source in manifest["sources"]:
        if _sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"source drift: {source['source_id']}")
    for row in manifest["observations"]:
        if _sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"observation drift: {row['query_id']} {row['frame_index']}")
    return manifest, seal


def _extended_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    metrics = _metrics(rows, arm)
    wrong = lambda row: row[arm]["state"] == "TARGET" and row[arm]["selected_candidate"] != row["truth"]
    metrics["directional_correct"] = sum(row["scenario"] == "directional" and row[arm]["state"] == "TARGET" and row[arm]["selected_candidate"] == row["truth"] for row in rows)
    metrics["directional_observations"] = sum(row["scenario"] == "directional" for row in rows)
    metrics["directory_false_bindings"] = sum(row["scenario"] == "directory_trap" and wrong(row) for row in rows)
    return metrics


def evaluate(
    *, cohort_dir: Path, run_dir: Path, runtime_root: Path,
    v3_a_model_path: Path, v3_c_model_path: Path,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite evaluation: {run_dir}")
    manifest, seal = _verify(cohort_dir)
    expected = manifest["models"]
    if _sha256_file(v3_a_model_path) != expected["v3_a_full_sha256"]:
        raise ValueError("V3-A model hash mismatch")
    if _sha256_file(v3_c_model_path) != expected["v3_c_sha256"]:
        raise ValueError("V3-C model hash mismatch")
    run_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3_a = _load_model(v3_a_model_path, FROZEN_MODEL_SHA256["v3_full"], device)
    v3_c, training = load_checkpoint(v3_c_model_path, device)
    ocr = RapidOcrPolygonAdapter(runtime_root)
    rows = []
    permutation_errors = {"v3_c_no_authority_typing": [], "v3_c_full": []}
    query_ids = list(dict.fromkeys(row["query_id"] for row in manifest["observations"]))
    for query_id in query_ids:
        episode = [row for row in manifest["observations"] if row["query_id"] == query_id]
        beliefs = {name: ReferentBelief(("A", "B", "C")) for name in ("v2_heuristic", "v3_a_full", "v3_c_no_authority_typing", "v3_c_full")}
        target = TargetGraph(tuple(episode[0]["target_tokens"]))
        for observation in episode:
            image = cv2.imread(observation["image_path"], cv2.IMREAD_COLOR)
            raw_ocr = ocr.read(image)
            tokens = _target_tokens_from_ocr(raw_ocr, target)
            blur, laplacian = _measured_blur(image)
            candidates = tuple(Candidate(key, Box(*value)) for key, value in sorted(observation["candidate_boxes"].items()))
            frame = Frame(
                query_id, int(observation["frame_index"]), observation["view"]["view_id"],
                candidates, tokens, blur, float(observation["view"]["perspective"]),
                observation["truth"], observation["expected_state"], observation["note"],
            )
            carriers = tuple(SignCarrier(item["carrier_id"], Box(*item["box"])) for item in observation["carriers"])
            v2_scores = graph_candidate_scores(target, frame)
            v3_a_scores, v3_a_diagnostics = predict_scores(v3_a, target, frame, include_relations=True, device=device)
            v3_c_scores, v3_c_diagnostics = {}, {}
            for arm, typing in (("v3_c_no_authority_typing", False), ("v3_c_full", True)):
                score, diagnostic = score_candidates(v3_c, target, frame, image, carriers, observation["cues"], authority_typing=typing, device=device)
                v3_c_scores[arm], v3_c_diagnostics[arm] = score, diagnostic
                reversed_frame = Frame(frame.episode_id, frame.frame_index, frame.viewpoint, tuple(reversed(frame.candidates)), frame.tokens, frame.blur, frame.perspective, frame.truth, frame.expected_state, frame.note)
                permuted, _ = score_candidates(v3_c, target, reversed_frame, image, carriers, observation["cues"], authority_typing=typing, device=device)
                permutation_errors[arm].append(max(abs(score[key]["score"] - permuted[key]["score"]) for key in score))
            rows.append({
                "query_id": query_id, "source_id": observation["source_id"], "frame_index": frame.frame_index,
                "viewpoint": frame.viewpoint, "target_tokens": list(target.tokens), "truth": frame.truth,
                "expected_state": frame.expected_state, "scenario": observation["scenario"], "note": frame.note,
                "image_path": observation["image_path"], "image_sha256": observation["image_sha256"],
                "ocr_raw": raw_ocr, "laplacian_variance": laplacian, "frame": asdict(frame),
                "v2_heuristic": beliefs["v2_heuristic"].update(frame, v2_scores),
                "v3_a_full": beliefs["v3_a_full"].update(frame, v3_a_scores),
                "v3_c_no_authority_typing": beliefs["v3_c_no_authority_typing"].update(frame, v3_c_scores["v3_c_no_authority_typing"]),
                "v3_c_full": beliefs["v3_c_full"].update(frame, v3_c_scores["v3_c_full"]),
                "diagnostics": {"v3_a": v3_a_diagnostics, "v3_c": v3_c_diagnostics},
            })
    metrics = {arm: _extended_metrics(rows, arm) for arm in ("v2_heuristic", "v3_a_full", "v3_c_no_authority_typing", "v3_c_full")}
    for arm, errors in permutation_errors.items():
        metrics[arm]["candidate_permutation_max_abs_error"] = max(errors, default=0.0)
        metrics[arm]["candidate_permutation_invariant_observations"] = sum(value <= 1e-6 for value in errors)
    raw_path = run_dir / "raw-decisions.json"
    _atomic_json(raw_path, {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "cohort_manifest_sha256": seal["manifest_sha256"], "rows": rows})
    full, v2, ablation = metrics["v3_c_full"], metrics["v2_heuristic"], metrics["v3_c_no_authority_typing"]
    checks = {
        "correct_exceeds_v2": full["correct_terminal_observations"] > v2["correct_terminal_observations"],
        "wrong_lock_not_above_v2": full["wrong_locks"] <= v2["wrong_locks"],
        "directory_false_binding_below_no_typing": full["directory_false_bindings"] < ablation["directory_false_bindings"],
        "directional_correct_present": full["directional_correct"] > 0,
        "permutation_invariant_all": full["candidate_permutation_invariant_observations"] == len(rows),
    }
    report = {
        "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(), "data_role": "FRESH_NATURAL_SOURCE_DISJOINT_DEVELOPMENT",
        "cohort": {"sources": manifest["source_count"], "queries": manifest["query_count"], "observations": manifest["observation_count"], "manifest_sha256": seal["manifest_sha256"]},
        "models": {"v3_a_sha256": _sha256_file(v3_a_model_path), "v3_c_sha256": _sha256_file(v3_c_model_path), "v3_c_training": training},
        "runtime": {"device": str(device), "torch": torch.__version__, "ocr": ocr.receipt()},
        "metrics": metrics,
        "decision": {"checks": checks, "status": "V3_C_NATURAL_EFFECT_SUPPORTED" if all(checks.values()) else "CLOSE_NATURAL_SAGE_R"},
        "interpretation_boundary": [
            "Sources, queries, truth, candidate/carrier/cue proposals, transformations and hashes were frozen before OCR or model calls.",
            "Candidate, carrier and direction-cue proposals are predeclared inputs; proposal detection and cue-orientation recognition are not evaluated.",
            "The cohort is small curated Development evidence, not population generalization, navigation, Android, safety or product evidence.",
            "V2 belief and terminal thresholds remain unchanged. V3-B and V4 do not run.",
        ],
        "raw_decisions_sha256": _sha256_file(raw_path),
    }
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True); freeze.add_argument("--source-dir", type=Path, required=True); freeze.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser = subs.add_parser("evaluate")
    evaluate_parser.add_argument("--cohort-dir", type=Path, required=True); evaluate_parser.add_argument("--run-dir", type=Path, required=True); evaluate_parser.add_argument("--runtime-root", type=Path, required=True); evaluate_parser.add_argument("--v3-a-model", type=Path, required=True); evaluate_parser.add_argument("--v3-c-model", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze_cohort(args.spec.resolve(), args.source_dir.resolve(), args.run_dir.resolve())
    else:
        report = evaluate(cohort_dir=args.cohort_dir.resolve(), run_dir=args.run_dir.resolve(), runtime_root=args.runtime_root.resolve(), v3_a_model_path=args.v3_a_model.resolve(), v3_c_model_path=args.v3_c_model.resolve())
        result = {"metrics": report["metrics"], "decision": report["decision"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
