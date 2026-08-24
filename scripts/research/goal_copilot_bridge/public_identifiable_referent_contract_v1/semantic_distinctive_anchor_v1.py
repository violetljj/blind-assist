"""Controlled semantic-anchor demo over the frozen Active Distinctive V0 sequence.

The source passive arm is copied from the consumed V0 receipt.  This successor
changes only the available evidence: natural OCR, a goal-selected distinctive
sign patch, a printed package code, or an ArUco marker.  Semantic identity never
falls back to appearance and abstains when the expected anchor is not unique.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


EXPERIMENT_ID = "SEMANTIC_DISTINCTIVE_ANCHOR_V1"
SCHEMA_VERSION = "blindassist_semantic_distinctive_anchor_v1"
CLAIM_CEILING = (
    "CONTROLLED_DERIVED_DEVELOPMENT_DEMO_INDEPENDENT_VISIBLE_ANCHORS_"
    "NO_GENERAL_EXACT_INSTANCE_P1_NAVIGATION_SAFETY_OR_DEFAULT_APP_CLAIM"
)
SOURCE_EXPERIMENT_ID = "ACTIVE_DISTINCTIVE_EVIDENCE_ACQUISITION_V0"
STEP_SECONDS = 0.7
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
PATCH_RATIO = 0.75
PATCH_MIN_INLIERS = 8
PATCH_MIN_INLIER_RATIO = 0.50


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    modality: str
    expected: str | int
    distractor: str | int | None
    evidence_origin: str


TARGET_SPECS = {
    "storefront-starbucks-dazaifu": TargetSpec(
        "storefront-starbucks-dazaifu",
        "NATURAL_OCR",
        "COFFEE",
        None,
        "Naturally occurring text in the existing Wikimedia target frames; no overlay.",
    ),
    "storefront-tsuiwah-tko": TargetSpec(
        "storefront-tsuiwah-tko",
        "DISTINCTIVE_SIGN_PATCH",
        "HOUSE_BAKE_SIGN",
        None,
        "Goal-selected HOUSE BAKE sign cropped from the public reference and injected as a visible active-scan observation.",
    ),
    "washington-cereal_box-1": TargetSpec(
        "washington-cereal_box-1",
        "OCR_PRODUCT_CODE",
        "BA101",
        "BA102",
        "Deterministic printed-code intervention on the consumed public product frames.",
    ),
    "washington-keyboard-1": TargetSpec(
        "washington-keyboard-1",
        "ARUCO_MARKER",
        17,
        23,
        "Deterministic DICT_4X4_50 marker canary on the consumed public personal-item frames.",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    _require(image is not None, f"cannot read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(bool(cv2.imwrite(str(path), image)), f"cannot write image: {path}")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(character for character in value if character.isalnum())


def text_anchor_present(texts: Sequence[str], expected: str) -> bool:
    needle = normalize_text(expected)
    return bool(needle) and any(needle in normalize_text(text) for text in texts)


def decide_unique_anchor(evidence: Mapping[str, bool]) -> dict[str, Any]:
    matches = sorted(slot for slot, present in evidence.items() if present)
    if len(matches) == 1:
        return {"decision": "LOCK", "selected_candidate": matches[0], "rank1_candidate": matches[0]}
    return {"decision": "ABSTAIN", "selected_candidate": None, "rank1_candidate": None}


def make_aruco_marker(marker_id: int, side: int) -> np.ndarray:
    _require(0 <= marker_id < 50, "marker ID outside DICT_4X4_50")
    _require(side >= 80, "marker side too small")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def detect_aruco_ids(image: np.ndarray) -> list[int]:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(image)
    return [] if ids is None else sorted(int(value) for value in ids.ravel())


def _paste_placard(image: np.ndarray, placard: np.ndarray, *, x_fraction: float = 0.04, y_fraction: float = 0.05) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    max_width = max(96, round(width * 0.30))
    scale = min(1.0, max_width / placard.shape[1])
    resized = cv2.resize(
        placard,
        (max(2, round(placard.shape[1] * scale)), max(2, round(placard.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    border = max(4, round(min(height, width) * 0.008))
    resized = cv2.copyMakeBorder(resized, border, border, border, border, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    x = min(max(0, round(width * x_fraction)), max(0, width - resized.shape[1]))
    y = min(max(0, round(height * y_fraction)), max(0, height - resized.shape[0]))
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def make_text_placard(text: str, width: int = 420, height: int = 128) -> np.ndarray:
    placard = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(placard, (3, 3), (width - 4, height - 4), (25, 25, 25), 5)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 2.0
    thickness = 4
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (width - text_width) // 2)
    y = min(height - 18, (height + text_height) // 2)
    cv2.putText(placard, text, (x, y), font, scale, (15, 15, 15), thickness, cv2.LINE_AA)
    return placard


def _paste_marker(image: np.ndarray, marker_id: int) -> np.ndarray:
    side = max(96, round(min(image.shape[:2]) * 0.24))
    return _paste_placard(image, make_aruco_marker(marker_id, side))


def _extract_distinctive_patch(reference: np.ndarray) -> np.ndarray:
    height, width = reference.shape[:2]
    # The fixed public reference crop contains the HOUSE BAKE sub-sign.  It is
    # selected for its target-specific semantics, not from outcome observations.
    x0, x1 = round(width * 0.225), round(width * 0.405)
    y0, y1 = round(height * 0.335), round(height * 0.485)
    patch = reference[y0:y1, x0:x1].copy()
    _require(min(patch.shape[:2]) >= 80, "distinctive sign patch unexpectedly small")
    return patch


def patch_evidence(template: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    detector = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.02)
    template_points, template_descriptors = detector.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
    candidate_points, candidate_descriptors = detector.detectAndCompute(cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY), None)
    if template_descriptors is None or candidate_descriptors is None:
        return {"present": False, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(template_descriptors, candidate_descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < PATCH_RATIO * pair[1].distance]
    if len(good) < 4:
        return {"present": False, "good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0}
    source = np.float32([template_points[match.queryIdx].pt for match in good])
    target = np.float32([candidate_points[match.trainIdx].pt for match in good])
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    inliers = 0 if mask is None else int(mask.ravel().sum())
    ratio = inliers / len(good)
    return {
        "present": inliers >= PATCH_MIN_INLIERS and ratio >= PATCH_MIN_INLIER_RATIO,
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": ratio,
    }


class RapidOcrAdapter:
    def __init__(self, runtime_root: Path):
        site_packages = runtime_root / "site-packages"
        _require(site_packages.is_dir(), f"missing OCR runtime: {site_packages}")
        sys.path.insert(0, str(site_packages.resolve()))
        from rapidocr import RapidOCR

        self.engine = RapidOCR(
            params={
                "Global.model_root_dir": str((runtime_root / "models").resolve()),
                "Global.log_level": "error",
                "Global.text_score": 0.50,
            }
        )

    def read(self, image: np.ndarray) -> dict[str, Any]:
        result = self.engine(image)
        texts = list(result.txts or ())
        scores = [float(value) for value in (result.scores or ())]
        return {"texts": texts, "scores": scores}


def _runtime_receipt(runtime_root: Path) -> dict[str, Any]:
    import onnxruntime

    models = sorted((runtime_root / "models").rglob("*.onnx"))
    receipt: dict[str, Any] = {
        "runtime_root": str(runtime_root.resolve()),
        "opencv": cv2.__version__,
        "onnxruntime_imported": onnxruntime.__version__,
        "onnx_models": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in models
        ],
    }
    for distribution in ("rapidocr", "onnxruntime", "omegaconf", "pyclipper", "Shapely"):
        try:
            receipt[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            receipt[distribution] = "not-found"
    return receipt


def _materialize_inputs(
    source_manifest: Mapping[str, Any], run_dir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    targets = []
    templates: dict[str, np.ndarray] = {}
    for source_target in source_manifest["targets"]:
        target_id = source_target["target_id"]
        _require(target_id in TARGET_SPECS, f"unexpected source target: {target_id}")
        spec = TARGET_SPECS[target_id]
        target_root = run_dir / "inputs" / target_id
        source_references = [Path(value) for value in source_target["reference_paths"]]
        reference_images = [_read_image(path) for path in source_references]
        if spec.modality == "DISTINCTIVE_SIGN_PATCH":
            templates[target_id] = _extract_distinctive_patch(reference_images[0])
            _write_image(target_root / "anchor-template.jpg", templates[target_id])
        reference_paths = []
        for index, image in enumerate(reference_images):
            if spec.modality == "OCR_PRODUCT_CODE":
                image = _paste_placard(image, make_text_placard(str(spec.expected)))
            elif spec.modality == "ARUCO_MARKER":
                image = _paste_marker(image, int(spec.expected))
            path = target_root / "reference" / f"ref-{index + 1:02d}.jpg"
            _write_image(path, image)
            reference_paths.append(str(path.resolve()))
        views = []
        for index, source_view in enumerate(source_target["views"]):
            target_image = _read_image(Path(source_view["target_path"]))
            hard_image = _read_image(Path(source_view["hard_path"]))
            if spec.modality == "DISTINCTIVE_SIGN_PATCH":
                target_image = _paste_placard(target_image, templates[target_id])
            elif spec.modality == "OCR_PRODUCT_CODE":
                target_image = _paste_placard(target_image, make_text_placard(str(spec.expected)))
                hard_image = _paste_placard(hard_image, make_text_placard(str(spec.distractor)))
            elif spec.modality == "ARUCO_MARKER":
                target_image = _paste_marker(target_image, int(spec.expected))
                hard_image = _paste_marker(hard_image, int(spec.distractor))
            target_path = target_root / "search" / f"view-{index + 1:02d}-target.jpg"
            hard_path = target_root / "search" / f"view-{index + 1:02d}-hard.jpg"
            _write_image(target_path, target_image)
            _write_image(hard_path, hard_image)
            views.append(
                {
                    "label": source_view["label"],
                    "target_path": str(target_path.resolve()),
                    "hard_path": str(hard_path.resolve()),
                }
            )
        lost_paths = []
        for index, source_path in enumerate(source_target["lost_candidates"]):
            image = _read_image(Path(source_path))
            if spec.modality == "OCR_PRODUCT_CODE" and index == 0:
                image = _paste_placard(image, make_text_placard(str(spec.distractor)))
            elif spec.modality == "ARUCO_MARKER" and index == 0:
                image = _paste_marker(image, int(spec.distractor))
            path = target_root / "search" / f"lost-{index + 1:02d}.jpg"
            _write_image(path, image)
            lost_paths.append(str(path.resolve()))
        targets.append(
            {
                "target_id": target_id,
                "scenario": source_target["scenario"],
                "modality": spec.modality,
                "expected_anchor": spec.expected,
                "distractor_anchor": spec.distractor,
                "evidence_origin": spec.evidence_origin,
                "reference_paths": reference_paths,
                "views": views,
                "lost_candidates": lost_paths,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "CONTROLLED_DERIVED_DEVELOPMENT_DEMO",
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_manifest_sha256": None,
        "selection_disclosure": (
            "The frozen four-target, 16-decision V0 sequence and candidate roles are retained. "
            "Starbucks uses naturally occurring COFFEE text. Tsui Wah receives a public-reference HOUSE BAKE sign patch, "
            "the cereal receives deterministic BA101/BA102 printed codes, and the keyboard receives ArUco 17/23 markers."
        ),
        "targets": targets,
        "claim_ceiling": CLAIM_CEILING,
    }
    return manifest, templates


def _candidate_evidence(
    spec: TargetSpec,
    image: np.ndarray,
    ocr: RapidOcrAdapter,
    template: np.ndarray | None,
) -> dict[str, Any]:
    if spec.modality in {"NATURAL_OCR", "OCR_PRODUCT_CODE"}:
        output = ocr.read(image)
        return {**output, "present": text_anchor_present(output["texts"], str(spec.expected))}
    if spec.modality == "ARUCO_MARKER":
        ids = detect_aruco_ids(image)
        return {"marker_ids": ids, "present": int(spec.expected) in ids}
    _require(template is not None, "distinctive patch template missing")
    return patch_evidence(template, image)


def _arm_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    present = [row for row in rows if row["target_present"]]
    top1 = sum(row[arm]["rank1_candidate"] == row["target_slot"] for row in present)
    wrong_locks = sum(
        row[arm]["decision"] == "LOCK"
        and (not row["target_present"] or row[arm]["selected_candidate"] != row["target_slot"])
        for row in rows
    )
    reacquired = 0
    opportunities = 0
    for target_id in sorted({row["target_id"] for row in rows}):
        sequence = sorted((row for row in rows if row["target_id"] == target_id), key=lambda row: row["step_index"])
        for index, row in enumerate(sequence[:-1]):
            if row["target_present"]:
                continue
            opportunities += 1
            next_present = next((candidate for candidate in sequence[index + 1 :] if candidate["target_present"]), None)
            if (
                next_present is not None
                and next_present[arm]["decision"] == "LOCK"
                and next_present[arm]["selected_candidate"] == next_present["target_slot"]
            ):
                reacquired += 1
    return {
        "target_present_decisions": len(present),
        "target_top1": top1,
        "target_top1_rate": top1 / len(present),
        "wrong_target_locks": wrong_locks,
        "reacquisition": reacquired,
        "reacquisition_opportunities": opportunities,
        "abstentions": sum(row[arm]["decision"] == "ABSTAIN" for row in rows),
    }


def _render_demo(rows: Sequence[Mapping[str, Any]], run_dir: Path) -> None:
    cards = []
    for row in rows:
        if not row["target_present"]:
            continue
        target_slot = row["target_slot"]
        target_path = Path(row["candidate_paths"][target_slot])
        image = _read_image(target_path)
        image = cv2.resize(image, (320, 220), interpolation=cv2.INTER_AREA)
        color = (40, 180, 40) if row["semantic"]["selected_candidate"] == target_slot else (40, 40, 220)
        cv2.rectangle(image, (0, 0), (319, 219), color, 6)
        label = f"{row['modality']}  {row['semantic']['decision']}"
        cv2.rectangle(image, (0, 0), (319, 34), (20, 20, 20), -1)
        cv2.putText(image, label[:42], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cards.append(image)
    rows_of_cards = []
    for start in range(0, len(cards), 4):
        rows_of_cards.append(cv2.hconcat(cards[start : start + 4]))
    board = cv2.vconcat(rows_of_cards)
    _write_image(run_dir / "demo-board.jpg", board)


def run(
    source_run_dir: Path,
    runtime_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    source_manifest_path = source_run_dir / "cohort-manifest.json"
    source_raw_path = source_run_dir / "raw-decisions.json"
    source_report_path = source_run_dir / "final-report.json"
    source_manifest = _read_json(source_manifest_path)
    source_raw = _read_json(source_raw_path)
    source_report = _read_json(source_report_path)
    _require(source_manifest.get("experiment_id") == SOURCE_EXPERIMENT_ID, "source manifest experiment drift")
    _require(source_raw.get("experiment_id") == SOURCE_EXPERIMENT_ID, "source raw experiment drift")
    _require(source_report["metrics"]["passive"]["target_top1"] == 11, "source passive top-1 drift")
    _require(source_report["metrics"]["passive"]["wrong_target_locks"] == 9, "source passive wrong-lock drift")
    _require(source_report["metrics"]["passive"]["reacquisition"] == 3, "source passive reacquisition drift")

    manifest, templates = _materialize_inputs(source_manifest, run_dir)
    manifest["source_manifest_sha256"] = _sha256_file(source_manifest_path)
    _atomic_json(run_dir / "cohort-manifest.json", manifest)
    ocr = RapidOcrAdapter(runtime_root)
    targets_by_id = {target["target_id"]: target for target in manifest["targets"]}
    source_rows_by_key = {(row["target_id"], row["step_index"]): row for row in source_raw["rows"]}
    rows = []
    for target in manifest["targets"]:
        target_id = target["target_id"]
        spec = TARGET_SPECS[target_id]
        steps = []
        for index, view in enumerate(target["views"]):
            if index == 2:
                steps.append({"label": "target-lost", "target_present": False, "paths": target["lost_candidates"]})
            steps.append(
                {
                    "label": view["label"],
                    "target_present": True,
                    "target_path": view["target_path"],
                    "hard_path": view["hard_path"],
                }
            )
        for step_index, step in enumerate(steps):
            source_row = source_rows_by_key[(target_id, step_index)]
            if step["target_present"]:
                candidate_paths = {
                    slot: step["target_path"] if role == "TARGET" else step["hard_path"]
                    for slot, role in source_row["candidate_roles"].items()
                }
            else:
                candidate_paths = {slot: path for slot, path in zip(sorted(source_row["candidate_roles"]), step["paths"], strict=True)}
            evidence = {
                slot: _candidate_evidence(spec, _read_image(Path(path)), ocr, templates.get(target_id))
                for slot, path in candidate_paths.items()
            }
            decision = decide_unique_anchor({slot: bool(value["present"]) for slot, value in evidence.items()})
            rows.append(
                {
                    "target_id": target_id,
                    "scenario": target["scenario"],
                    "modality": spec.modality,
                    "step_index": step_index,
                    "time_seconds": step_index * STEP_SECONDS,
                    "view_label": step["label"],
                    "target_present": source_row["target_present"],
                    "target_slot": source_row["target_slot"],
                    "candidate_roles": source_row["candidate_roles"],
                    "candidate_paths": candidate_paths,
                    "passive": source_row["passive"],
                    "semantic": {**decision, "evidence": evidence},
                }
            )
    raw = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "source_receipts": {
            "cohort_manifest": {"path": str(source_manifest_path.resolve()), "sha256": _sha256_file(source_manifest_path)},
            "raw_decisions": {"path": str(source_raw_path.resolve()), "sha256": _sha256_file(source_raw_path)},
            "final_report": {"path": str(source_report_path.resolve()), "sha256": _sha256_file(source_report_path)},
        },
        "semantic_policy": "LOCK iff exactly one candidate contains the goal-selected anchor; otherwise ABSTAIN; no appearance fallback or tracker identity",
        "runtime_receipt": _runtime_receipt(runtime_root),
        "rows": rows,
        "claim_ceiling": CLAIM_CEILING,
    }
    _atomic_json(run_dir / "raw-decisions.json", raw)
    passive = _arm_metrics(rows, "passive")
    semantic = _arm_metrics(rows, "semantic")
    by_modality = {}
    for modality in sorted({row["modality"] for row in rows}):
        selected = [row for row in rows if row["modality"] == modality]
        by_modality[modality] = {"passive": _arm_metrics(selected, "passive"), "semantic": _arm_metrics(selected, "semantic")}
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "CONTROLLED_DERIVED_DEVELOPMENT_DEMO",
        "metrics": {"passive": passive, "semantic": semantic, "by_modality": by_modality},
        "delta": {
            "target_top1": semantic["target_top1"] - passive["target_top1"],
            "wrong_target_locks": semantic["wrong_target_locks"] - passive["wrong_target_locks"],
            "reacquisition": semantic["reacquisition"] - passive["reacquisition"],
        },
        "interpretation_boundary": [
            "This is a controlled evidence intervention over the same sequence and candidate roles, not a same-pixel matcher comparison.",
            "Starbucks OCR is naturally occurring; the other three modalities are deterministic derived canaries.",
            "A marker or unique printed code carries direct identity, while a logo/sign patch carries only the scoped goal semantics and may repeat elsewhere.",
            "No appearance fallback, open-set threshold fitting, tracker, Android integration, navigation, safety, or default-App change is evaluated.",
        ],
        "terminal": "SEMANTIC_ANCHOR_CONTROLLED_DEMO_MEASURED",
        "claim_ceiling": CLAIM_CEILING,
        "raw_decisions_sha256": _sha256_file(run_dir / "raw-decisions.json"),
    }
    _atomic_json(run_dir / "final-report.json", report)
    _render_demo(rows, run_dir)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(not args.run_dir.exists(), f"refusing to overwrite run: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    try:
        report = run(args.source_run_dir.resolve(), args.runtime_root.resolve(), args.run_dir.resolve())
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        _atomic_json(
            args.run_dir / "failure.json",
            {"experiment_id": EXPERIMENT_ID, "failed_at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
