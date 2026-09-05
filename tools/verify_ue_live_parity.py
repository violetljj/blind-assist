"""Compare retained batch-prefix and incremental X73 on completed live sensor input.

Reuses identity-checked saved detector candidates. No detector inference, actor
truth, evaluator outputs, parameter fitting or counterfactual motion is accessed.
The first source frame is checked as deferred WARMUP; every subsequent prefix
compares the complete last X73 frame and the complete compact response.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
UNREAL = REPO / "research/active/dtr-r0/unreal"
sys.path.insert(0, str(UNREAL))
import ue_fixed_replay as fixed
import ue_dtr_replay as replay
from ue_incremental import source_paths
from ue_replay_cache import cached_replay_inputs
from validate_ue_incremental import first_difference, digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cached_candidates(source_run, episode, model_hash):
    root = source_run / "sensor-worker/candidate-cache" / episode.episode_id
    values, hashes = [], {}
    for observation in episode.observations:
        path = root / f"{observation.sample_index:06d}.json"
        value = fixed.read(path)
        source = {"episode_id": observation.episode_id, "sample_index": observation.sample_index,
                  "time_s": observation.time_s, "world_frame": observation.world_frame,
                  "frame_id": f"{observation.episode_id}/{observation.sample_index:06d}"}
        expected = {"model_sha256": model_hash, "rgb_sha256": observation.rgb.sha256, "source": source}
        require(value.get("identity") == expected, f"Candidate cache source/model mismatch: {path.name}")
        candidate = value["candidate"]
        require(candidate.get("model", {}).get("sha256") == model_hash, "Candidate model identity mismatch")
        actual = candidate.get("source", {})
        require(all(actual.get(k) == v for k, v in source.items()), "Candidate source identity mismatch")
        require(actual.get("image_sha256") == observation.rgb.sha256, "Candidate RGB identity mismatch")
        require(isinstance(candidate.get("candidates"), list), "Candidate payload must contain a list")
        values.append(candidate)
        hashes[str(path.relative_to(source_run))] = fixed.sha(path)
    require(len(list(root.glob("*.json"))) == len(values), "Cache has frames outside completed source descriptor")
    return values, hashes


def verify(source_run, output, episode_id):
    started = time.perf_counter()
    source_run, output = Path(source_run).resolve(strict=True), Path(output).resolve()
    require(not (source_run / "owner.lock").exists(), "Live source is still owned; wait for launcher cleanup")
    require(fixed.read(source_run / "run.json").get("status") == "COMPLETE", "Requires a completed live run")
    require(output.is_relative_to((REPO / "artifacts.local").resolve()), "Output must be under canonical artifacts.local")
    require(not output.is_relative_to(source_run), "Parity evidence must not modify the completed source run")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "ue-live-x73-prefix-parity-v1", "status": "RUNNING", "episode_id": episode_id,
               "evaluator_truth_opened": False, "detector_inference_calls": 0,
               "backend": "CPU", "cpu_reason": "TASK_NOT_GPU_SUITABLE",
               "backend_scope": "Unchanged scalar/NumPy tracker and reference state; no learned model inference",
               "frames": 0, "compared_prefixes": 0, "mismatches": 0,
               "claim": "Consumed live synthetic sensor-input implementation parity; no method performance or safety claim"}
    try:
        snapshot_started = time.perf_counter()
        fixed.export_dataset(source_run, output / "sensor-snapshot", hardlink=True)
        contract = replay.load_contract(output / "sensor-snapshot")
        selected = [e for e in contract.episodes if e.episode_id == episode_id]
        require(len(selected) == 1, "Select exactly one completed opaque episode identifier")
        episode = selected[0]
        require(len(episode.observations) >= 2, "Parity requires at least two observed source frames")
        backend_path = source_run / "sensor-worker/backend.json"
        model_hash = fixed.read(backend_path)["model_sha256"]
        candidates, cache_hashes = cached_candidates(source_run, episode, model_hash)
        paths = [*source_paths(), Path(__file__).resolve(), UNREAL / "ue_fixed_replay.py",
                 UNREAL / "validate_ue_incremental.py"]
        fixed.write(output / "input-identity.json", {"candidate_cache_sha256": cache_hashes,
            "worker_backend_sha256": fixed.sha(backend_path), "model_sha256": model_hash,
            "snapshot_integrity_sha256": fixed.sha(output / "sensor-snapshot/integrity.json"),
            "source_sha256": {str(p.relative_to(REPO)): fixed.sha(p) for p in paths}})
        with (output / "candidates.jsonl").open("x", encoding="utf-8") as stream:
            for value in candidates:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
        receipt.update(model_sha256=model_hash, snapshot_elapsed_s=time.perf_counter()-snapshot_started,
            source_manifest_sha256=contract.manifest_sha256,
            source_dt_s=sorted({round(b.time_s-a.time_s, 10) for a,b in zip(episode.observations, episode.observations[1:])}),
            candidate_ledger_sha256=fixed.sha(output / "candidates.jsonl"))
        engine = replay.IncrementalX73(episode_id, episode.route_frame, contract.calibration)
        incremental_s, batch_s = 0., 0.
        with (output / "frame-equivalence.jsonl").open("x", encoding="utf-8") as stream:
            for index, observation in enumerate(episode.observations):
                tick = time.perf_counter()
                compact = engine.update(observation, candidates[index])
                incremental_s += time.perf_counter() - tick
                row = {"sample_index": index, "time_s": observation.time_s}
                if index == 0:
                    require(compact.get("event") == "WARMUP" and engine.last_frame is None and engine.processed_count == 0,
                            "First observation must remain deferred WARMUP")
                    row.update(status="WARMUP_CONTRACT_PASS", batch_prefix_compared=False)
                else:
                    tick = time.perf_counter()
                    prefix = replace(episode, observations=episode.observations[:index+1])
                    with cached_replay_inputs():
                        batch = replay.predict_episode(prefix, candidates[:index+1], contract.calibration)
                    batch_s += time.perf_counter() - tick
                    pairs = (("last_frame", batch["frames"][-1], engine.last_frame),
                             ("compact", replay.compact_rows(episode_id, batch)[-1], compact))
                    differences = {name: difference for name, a, b in pairs
                                   if (difference := first_difference(a, b)) is not None}
                    receipt["compared_prefixes"] += 1
                    receipt["mismatches"] += int(bool(differences))
                    row.update(status="MISMATCH" if differences else "PASS", differences=differences,
                               compared_sha256={name: {"batch": digest(a), "incremental": digest(b)} for name,a,b in pairs})
                receipt["frames"] += 1
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if index % 10 == 0 or index == len(candidates)-1:
                    print(f"PARITY {episode_id}: {index+1}/{len(candidates)} frames; mismatches={receipt['mismatches']}", flush=True)
        require(not (source_run / "owner.lock").exists(), "Source was re-owned during verification")
        fixed.verify_dataset(output / "sensor-snapshot")
        require(all(fixed.sha(source_run / p) == h for p,h in cache_hashes.items()), "Saved candidate cache changed during verification")
        receipt.update(status="PASS" if receipt["mismatches"] == 0 else "FAIL",
                       incremental_elapsed_s=incremental_s, batch_prefix_elapsed_s=batch_s, engine_stats=engine.stats)
    except BaseException as error:
        receipt.update(status="ERROR", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_s"] = time.perf_counter() - started
        fixed.write(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()
    result = verify(args.source_run, args.output, args.episode_id)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
