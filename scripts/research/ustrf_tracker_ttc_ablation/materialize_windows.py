from __future__ import annotations

import argparse
import json
from pathlib import Path


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    config = read_json(args.config)
    if config.get("schema") != "blindassist_ustrf_tracker_ttc_ablation_v1":
        raise ValueError("unexpected tracker ablation config schema")
    before = int(config["window_policy"]["context_before_frames"])
    after = int(config["window_policy"]["context_after_frames"])
    windows: list[dict] = []
    for source_name, source in config["inputs"].items():
        if source_name in {"prereg_sha256"}:
            continue
        consensus_path = Path(
            "artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1"
        ) / ("dynamics_0-review-consensus-v2.json" if source_name == "dynamics_0" else "lt_changes_dynamics_0-review-v1/review-consensus-v2.json")
        consensus = read_json(consensus_path)
        if sha256(consensus_path) != source["event_consensus_sha256"]:
            raise ValueError(f"event consensus hash mismatch: {source_name}")
        source_row = next(row for row in consensus["sources"] if row["source_id"] == source["source_id"])
        frame_count = int(source["frame_count"])
        positives: list[tuple[int, int]] = []
        for event in source_row["events"]:
            start = max(0, int(event["onset_frame"]) - before)
            end = min(frame_count - 1, int(event["end_frame"]) + after)
            positives.append((start, end))
            windows.append({
                "window_id": f"{source_name}-positive-{event['event_id']}",
                "source_id": source["source_id"],
                "window_type": "positive",
                "start_frame": start,
                "end_frame": end,
                "event_id": event["event_id"],
                "critical": bool(event["critical"]),
                "truth_anchors": {
                    "onset_frame": int(event["onset_frame"]),
                    "alertable_frame": int(event["alertable_frame"]),
                    "passed_or_cleared_frame": int(event["passed_or_cleared_frame"]),
                    "end_frame": int(event["end_frame"]),
                },
            })
        occupied = list(positives)
        cursor = 0
        for index, positive in enumerate(positives, start=1):
            length = positive[1] - positive[0] + 1
            while cursor + length <= frame_count:
                candidate = (cursor, cursor + length - 1)
                cursor += 1
                if any(overlaps(candidate, used) for used in occupied):
                    continue
                occupied.append(candidate)
                windows.append({
                    "window_id": f"{source_name}-negative-{index:02d}",
                    "source_id": source["source_id"],
                    "window_type": "negative",
                    "start_frame": candidate[0],
                    "end_frame": candidate[1],
                    "event_id": None,
                    "critical": False,
                    "truth_anchors": None,
                })
                break
            else:
                raise ValueError(f"could not materialize matched negative: {source_name} {index}")
    windows.sort(key=lambda row: (row["source_id"], row["start_frame"], row["window_type"], row["window_id"]))
    payload = {
        "schema": "blindassist_ustrf_tracker_ttc_ablation_windows_v1",
        "authority": "truth_partition_only_not_detector_input",
        "config_sha256": sha256(args.config),
        "windows": windows,
        "truth_visible_to_detector": False,
        "candidate_visible_to_detector": False,
        "event_truth_sha256": {
            key: value["event_consensus_sha256"]
            for key, value in config["inputs"].items()
            if key != "prereg_sha256"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"windows": len(windows), "positive": sum(w["window_type"] == "positive" for w in windows), "negative": sum(w["window_type"] == "negative" for w in windows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
