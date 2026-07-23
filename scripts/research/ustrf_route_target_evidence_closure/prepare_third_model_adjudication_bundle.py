from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import load_json, sha256_file, validate_prereg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite adjudication bundle: {args.output}")

    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    fusion_config = config["seen_truth_proposal_protocol"]["fusion"]
    adjudicator = fusion_config["third_model_adjudicator"]
    expected_output = repo / adjudicator["planned_bundle_path"]
    if args.output.resolve() != expected_output.resolve():
        raise ValueError("adjudication bundle path differs from preregistration")

    fusion_path = repo / fusion_config["materialized_output"]["path"]
    fusion = load_json(fusion_path)
    frame_by_hash = {frame["image_sha256"]: frame for frame in fusion["frames"]}
    selected = [
        tracklet for tracklet in fusion["tracklets"]
        if tracklet["status"] == "third_model_adjudication_required"
    ]
    image_hashes = sorted({member["image_sha256"] for tracklet in selected for member in tracklet["members"]})
    frames = []
    for image_sha256 in image_hashes:
        frame = frame_by_hash[image_sha256]
        frames.append({
            "source_id": frame["source_id"],
            "blind_window_id": frame["blind_window_id"],
            "frame_id": frame["frame_id"],
            "image_path": frame["image_path"],
            "image_sha256": image_sha256,
        })
    payload = {
        "schema": "blindassist_ustrf_third_model_adjudication_bundle_r1",
        "authority": "disagreement_only_person_proposal_adjudication_never_candidate_or_truth_by_itself",
        "config_sha256": sha256_file(args.config),
        "fusion_sha256": sha256_file(fusion_path),
        "candidate_alerts_exposed": False,
        "baseline_app_detector_outputs_exposed": False,
        "scoring_labels_exposed": False,
        "adjudication_tracklet_count": len(selected),
        "frame_count": len(frames),
        "frames": sorted(frames, key=lambda row: (row["source_id"], row["blind_window_id"], int(row["frame_id"]))),
        "adjudication_tracklets": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "adjudication_tracklet_count": len(selected),
        "frame_count": len(frames),
        "sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
