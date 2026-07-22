from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "blindassist_ustrf_detector_taxonomy_coverage_v1":
        raise ValueError("unexpected detector coverage schema")
    forbidden = {"result", "results", "verdict", "selected_candidate"}
    if forbidden.intersection(config):
        raise ValueError("result leakage in preregistration")
    parent = config["parent"]
    detector = config["detector"]
    receipts = config["implementation_receipts"]
    bound = {
        Path(parent["config_path"]): parent["config_sha256"],
        Path(parent["windows_path"]): parent["windows_sha256"],
        Path(detector["model_path"]): detector["model_sha256"],
        Path(detector["labels_path"]): detector["labels_sha256"],
        Path(receipts["android_image_preprocessor_path"]): receipts["android_image_preprocessor_sha256"],
        Path(receipts["android_decoder_path"]): receipts["android_decoder_sha256"],
    }
    for path, expected in bound.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"hash-bound input mismatch: {path}")
    windows = json.loads(Path(parent["windows_path"]).read_text(encoding="utf-8"))
    counts: dict[str, set[int]] = {}
    for row in windows["windows"]:
        frame_ids = counts.setdefault(row["source_id"], set())
        frame_ids.update(range(int(row["start_frame"]), int(row["end_frame"]) + 1))
    actual_counts = {key: len(value) for key, value in counts.items()}
    if actual_counts != parent["source_frame_counts"] or sum(actual_counts.values()) != parent["frame_count"]:
        raise ValueError("frozen frame inventory mismatch")
    labels = Path(detector["labels_path"]).read_text(encoding="utf-8").splitlines()
    person_index = int(detector["person_class_index"])
    if person_index != 0 or labels[person_index] != detector["person_label"]:
        raise ValueError("COCO person index/label mismatch")
    if detector["input_shape"] != [1, 320, 320, 3] or detector["output_shape"] != [1, 84, 2100]:
        raise ValueError("frozen tensor contract mismatch")
    if detector["confidence_threshold"] != 0.35 or detector["nms_iou_threshold"] != 0.45:
        raise ValueError("threshold/NMS drift")
    candidates = config["gates"]["G5_candidate_comparison"]["candidate_roster"]
    if len(candidates) > config["gates"]["G5_candidate_comparison"]["max_preregistered_candidates"]:
        raise ValueError("candidate roster exceeds preregistered maximum")
    locks = config["stage_locks"]
    if any(locks.values()):
        raise ValueError("downstream authority must remain closed in preregistration")
    return {
        "schema": "blindassist_ustrf_detector_taxonomy_coverage_manifest_validation_v1",
        "config_sha256": sha256(config_path),
        "frame_count": sum(actual_counts.values()),
        "source_frame_counts": actual_counts,
        "candidate_count": len(candidates),
        "manifest_valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.config), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
