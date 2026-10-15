"""Replay one disclosed posthoc candidate-pool hypothesis; no new model fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "candidate_pool_probe"
    out.mkdir(exist_ok=True)
    config = {
        "lane": "EXPLORE posthoc mechanism probe after observing candidate filtering",
        "new_predictions_require_no_gt": True,
        "presence_rule": "unchanged max DINO score >= .25",
        "selection_rule": "rank full recorded >=.05 pool by already-computed CLIP cosine, no retraining",
        "parameters_searched": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    source = root / "inference/clip_predictions.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        for candidate in row["candidates"]:
            candidate["ranking_score"] = candidate["clip_cosine"]
        row["candidates"].sort(key=lambda candidate: (
            -candidate["ranking_score"], -candidate["score"], tuple(candidate["bbox_xyxy"])))
        row["variant"] = "CLIP_all_recorded_pool_posthoc"
        row["elapsed_s"] = None
    predictions = out / "all_pool_predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pilot_analysis", Path(__file__).with_name("analyze_pilot.py"))
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    manifest = analysis.load_manifest(root / "data/samples.jsonl")
    observations, _ = analysis.load_predictions([predictions])
    result = analysis.analyze(manifest, observations)
    analysis.write_results(result, out / "analysis")
    baseline, _ = analysis.load_predictions([root / "inference/base_predictions.jsonl"])
    baseline_outputs = {analysis.row_key(row): analysis.top_box(
        baseline[analysis.row_key(row)]["candidates"], .25) for row in manifest}
    trial_outputs = {analysis.row_key(row): analysis.top_box(
        observations[analysis.row_key(row)]["candidates"], .25) for row in manifest}
    summary = {
        "config": config, "metrics": result["baseline_thresholded"],
        "paired_vs_dino": analysis.paired_outputs(manifest, baseline_outputs, trial_outputs, .5),
        "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
