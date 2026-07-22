from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import read_json, sha256, write_json


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    candidate = read_json(args.candidate)
    if candidate.get("candidate_alerts_frozen_before_review") is not True:
        raise ValueError("candidate freeze receipt missing")
    args.output.mkdir(parents=True)
    sources = []
    for source in candidate["sources"]:
        source_dir = args.output / source["source_id"]
        source_dir.mkdir()
        truth_path = source_dir / "route_truth.jsonl"
        prediction_path = source_dir / "causal_route_prediction.jsonl"
        write_jsonl(truth_path, source["route_truth"])
        write_jsonl(prediction_path, source["route_predictions"])
        if any(float(row["predicted_at_s"]) > float(row["timestamp_s"]) + 1e-9 for row in source["route_predictions"]):
            raise ValueError("future-dated causal prediction")
        sources.append({
            "source_id": source["source_id"],
            "route_truth_path": truth_path.relative_to(args.output).as_posix(),
            "route_truth_sha256": sha256(truth_path),
            "causal_route_prediction_path": prediction_path.relative_to(args.output).as_posix(),
            "causal_route_prediction_sha256": sha256(prediction_path),
            "frame_count": len(source["route_truth"]),
            "causal_prediction_policy": candidate["route_prediction_policy"],
        })
    write_json(args.output / "manifest.json", {
        "schema": "blindassist_ustrf_sensor_replay_r3_route_ledger_manifest_v1",
        "candidate_evaluation_sha256": sha256(args.candidate),
        "truth_and_prediction_physically_separate": True,
        "sources": sources,
        "production_authority": False,
    })
    print(json.dumps({"sources": len(sources), "manifest_sha256": sha256(args.output / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
