"""Differential validation on explicitly supplied, consumed UE sensory input.

One hashed detector ledger feeds both engines. The original batch function is
invoked at every causal prefix; the candidate advances once. No evaluator,
reserved scenario recipe, final source, or counterfactual motion is opened.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import ue_fixed_replay as fixed
import ue_dtr_replay as replay
from ue_incremental import source_paths
from ue_replay_cache import cached_replay_inputs


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def first_difference(left, right, path="frame"):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return {"path": path, "left_only": sorted(left.keys()-right.keys()),
                    "right_only": sorted(right.keys()-left.keys())}
        for key in sorted(left):
            result = first_difference(left[key], right[key], f"{path}.{key}")
            if result:
                return result
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            result = first_difference(a, b, f"{path}[{index}]")
            if result:
                return result
    elif left != right:
        return {"path": path, "batch": left, "incremental": right}
    return None


def extract(contract, output, expected_model_hash):
    os.environ["YOLO_CONFIG_DIR"] = str(output / "yolo-config")
    import torch
    from ultralytics import YOLO
    require(torch.cuda.is_available(), "CUDA is required for the detector extraction")
    weights = replay.detector.DEFAULT_MODEL.resolve(strict=True)
    model_hash = fixed.sha(weights)
    require(model_hash.lower() == expected_model_hash.lower(), "Historical detector checkpoint changed")
    model = YOLO(str(weights), task="segment").to("cuda:0")
    names = replay.detector.model_names(model)
    ledger = output / "candidates.jsonl"
    started = time.perf_counter()
    count = 0
    try:
        with ledger.open("x", encoding="utf-8") as stream:
            for episode in contract.episodes:
                for observation in episode.observations:
                    source = {"episode_id": episode.episode_id, "sample_index": observation.sample_index,
                              "time_s": observation.time_s, "world_frame": observation.world_frame,
                              "frame_id": f"{episode.episode_id}/{observation.sample_index:06d}"}
                    value = replay.detector.infer_one(model, names,
                        replay.detector.ImageInput(observation.rgb.path, source, observation.rgb.sha256),
                        model_path=weights, model_sha256=model_hash, device="0", run_kind="FIXED_SENSORY_REPLAY")
                    stream.write(json.dumps(value, allow_nan=False) + "\n")
                    count += 1
                stream.flush()
                print(f"EXTRACT {episode.episode_id}: {count} frames", flush=True)
        torch.cuda.synchronize()
        receipt = {"status": "COMPLETE", "frames": count, "candidate_sha256": fixed.sha(ledger),
                   "dataset_manifest_sha256": contract.manifest_sha256, "model_sha256": model_hash,
                   "device": str(next(model.model.parameters()).device), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0), "elapsed_s": time.perf_counter()-started,
                   "authority": "CONSUMED_SYNTHETIC_DEVELOPMENT_FIXED_INPUT"}
        fixed.write(output / "candidate-receipt.json", receipt)
        return receipt
    finally:
        del model
        torch.cuda.empty_cache()
        print("DETECTOR_RELEASED", flush=True)


def run(dataset, output, historical, *, ledger_root=None, episode_ids=None):
    output = Path(output).resolve()
    require(output.is_relative_to((replay.REPO / "artifacts.local").resolve()), "Output must be under artifacts.local")
    output.mkdir(parents=True, exist_ok=False)
    source_identity = {str(p.relative_to(replay.REPO)): fixed.sha(p) for p in source_paths()}
    source_identity[str(Path(__file__).resolve().relative_to(replay.REPO))] = fixed.sha(Path(__file__))
    fixed.write(output / "source-identity.json", source_identity)
    started = time.perf_counter()
    try:
        # The caller must name the consumed dataset; no directory discovery.
        integrity = fixed.verify_dataset(dataset)
        contract = replay.load_contract(Path(dataset))
        reference_report = fixed.read(Path(historical) / "replay.json")
        require(reference_report["dataset_integrity_sha256"] == fixed.sha(Path(dataset) / "integrity.json"),
                "Historical reference belongs to a different dataset")
        ledger_root = Path(ledger_root) if ledger_root else output
        if ledger_root == output:
            receipt = extract(contract, output, reference_report["model_sha256"])
        else:
            receipt = fixed.read(ledger_root / "candidate-receipt.json")
        require(receipt["status"] == "COMPLETE", "Detector ledger is incomplete")
        require(receipt["candidate_sha256"] == fixed.sha(ledger_root / "candidates.jsonl"), "Detector ledger changed")
        require(receipt["dataset_manifest_sha256"] == contract.manifest_sha256, "Detector/source identity mismatch")
        require(receipt["model_sha256"].lower() == reference_report["model_sha256"].lower(), "Detector model mismatch")
        rows = [json.loads(line) for line in (ledger_root / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        candidates = {(row["source"]["episode_id"], row["source"]["sample_index"]): row for row in rows}
        expected = {(e.episode_id, o.sample_index) for e in contract.episodes for o in e.observations}
        require(len(candidates) == len(rows) == len(expected) and candidates.keys() == expected, "Detector ledger source join incomplete")
        historical_rows = [json.loads(line) for line in (Path(historical)/"frames.jsonl").read_text().splitlines()]
        old = {(r["episode_id"], r["sample_index"]): r for r in historical_rows}
        episodes = [e for e in contract.episodes if episode_ids is None or e.episode_id in episode_ids]
        require(episodes and (episode_ids is None or {e.episode_id for e in episodes} == set(episode_ids)), "Invalid episode selection")
        results, historical_equal, total_frames = [], 0, 0
        with (output / "frame-equivalence.jsonl").open("x", encoding="utf-8") as stream:
            for episode in episodes:
                initialization = time.perf_counter()
                engine = replay.IncrementalX73(episode.episode_id, episode.route_frame, contract.calibration)
                initialization = time.perf_counter()-initialization
                seen, batch_s, incremental_s, oracle_s, full_frames = [], 0., 0., 0., 0
                for observation in episode.observations:
                    key = (episode.episode_id, observation.sample_index)
                    seen.append(candidates[key])
                    tick = time.perf_counter()
                    compact = engine.update(observation, seen[-1])
                    incremental_s += time.perf_counter()-tick
                    comparisons = {}
                    if len(seen) > 1:
                        prefix = replace(episode, observations=episode.observations[:len(seen)])
                        with cached_replay_inputs():
                            tick = time.perf_counter()
                            batch = replay.predict_episode(prefix, seen, contract.calibration)
                            batch_s += time.perf_counter()-tick
                            tick = time.perf_counter()
                            with replay.native_depth_loader():
                                metric = replay.x24.predict_episode(prefix, seen, contract.calibration)["frames"][-1]
                                rigid = replay.x25.predict_episode(prefix, seen, contract.calibration)["frames"][-1]
                            oracle_s += time.perf_counter()-tick
                        for name, a, b in (("x73", batch["frames"][-1], engine.last_frame),
                                           ("metric", metric, engine.last_metric_frame),
                                           ("rigid", rigid, engine.last_rigid_frame),
                                           ("compact", replay.compact_rows(episode.episode_id,batch)[-1], compact)):
                            diff = first_difference(a, b)
                            if diff:
                                fixed.write(output / "mismatch.json", {"key": key, "layer": name, "difference": diff,
                                                                     "batch": a, "incremental": b})
                                raise AssertionError(f"Differential mismatch {key} {name}: {diff}")
                            comparisons[name] = digest(a)
                        full_frames += 1
                    # Original fixed replay omitted WARMUP support fields.
                    reference = old[key]
                    common = ("route_risk", "event") if observation.sample_index == 0 else (
                        "route_risk", "event", "minimum_entry_s", "risk_state", "support_state",
                        "track_dispositions", "route_mode", "confirmed_risk_track_ids")
                    historical_match = all(reference.get(k) == compact.get(k) for k in common)
                    historical_equal += int(historical_match)
                    stream.write(json.dumps({"episode_id": key[0], "sample_index": key[1],
                                             "compared_sha256": comparisons, "historical_match": historical_match,
                                             "route_risk": compact["route_risk"], "event": compact["event"]})+"\n")
                    total_frames += 1
                    if len(seen) % 10 == 0:
                        stream.flush()
                        print(f"COMPARE {key[0]} frame {key[1]}: batch={batch_s:.3f}s incremental={incremental_s:.3f}s", flush=True)
                require(engine.processed_count == len(seen), "Observation update count mismatch")
                require(all(s.processed_count == len(seen) for s in engine.stages), "A stage replayed or skipped a frame")
                require(all(len(s.frames.items) <= 2 for s in engine.stages), "A stage retained a replay prefix")
                result = {"episode_id": episode.episode_id, "frames": len(seen), "full_frame_comparisons": full_frames,
                          "batch_predictor_s": batch_s, "incremental_predictor_s": incremental_s,
                          "incremental_initialization_s": initialization, "auxiliary_oracle_s": oracle_s,
                          "stats": engine.stats}
                results.append(result)
                fixed.write(output / "progress.json", {"status": "RUNNING", "episodes": results})
                stream.flush()
                print(f"EPISODE_PASS {episode.episode_id}: {len(seen)} frames", flush=True)
                for path, fingerprint in source_identity.items():
                    require(fixed.sha(replay.REPO/path) == fingerprint, f"Source changed during validation: {path}")
        batch_s = sum(row["batch_predictor_s"] for row in results)
        incremental_s = sum(row["incremental_predictor_s"]+row["incremental_initialization_s"] for row in results)
        report = {"schema": "ue-incremental-x73-validation-v1", "status": "PASS", "frames": total_frames,
                  "dataset_frames": integrity["frames"], "episodes": results,
                  "full_frame_comparisons": sum(r["full_frame_comparisons"] for r in results),
                  "historical_compact_equal_frames": historical_equal, "historical_compact_different_frames": total_frames-historical_equal,
                  "batch_predictor_s": batch_s, "incremental_predictor_with_initialization_s": incremental_s,
                  "predictor_speed_ratio": batch_s/incremental_s, "elapsed_s": time.perf_counter()-started,
                  "detector_ledger": str(ledger_root/"candidates.jsonl"), "detector_receipt": receipt,
                  "source_identity_sha256": fixed.sha(output/"source-identity.json"),
                  "prediction_backend": "CPU", "backend_reason": "TASK_NOT_GPU_SUITABLE",
                  "evaluator_truth_opened": False, "ue_launched": False,
                  "authority": "CONSUMED_SYNTHETIC_DEVELOPMENT_IMPLEMENTATION_EQUIVALENCE"}
        fixed.write(output / "validation.json", report)
        print(json.dumps({k:v for k,v in report.items() if k not in ("episodes","detector_receipt")}), flush=True)
        return report
    except Exception as error:
        fixed.write(output / "failure.json", {"status": "FAILED", "error": repr(error), "elapsed_s": time.perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path)
    parser.add_argument("--episode", action="append")
    args = parser.parse_args()
    run(args.dataset, args.output, args.historical, ledger_root=args.ledger_root, episode_ids=args.episode)


if __name__ == "__main__":
    main()
