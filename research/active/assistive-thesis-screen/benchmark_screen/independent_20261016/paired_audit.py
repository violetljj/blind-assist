"""Independent, CPU-only audit of existing ORBIT score dumps.

No images, models, Torch, SciPy, network, or original scorer imports are used.
The original ``orbit_dump.py`` NPZ schema was inspected before implementation:
``names, valid, y, user, full, part@0, head:ad64_s0/s1/s2,
head:poolonly_s0``. A missing field/method stays unknown; it is never scored 0.

Example (OUT must be a new JSON file, outside the input dump):
  python paired_audit.py --dump PATH/test_learned --out PATH/audit.json
  python paired_audit.py --self-test

Two separate estimands are retained throughout:
* legacy_50_tasks: the old scorer's sorted NPZ order, Random(0), 50 draws of
  min(200, n_valid) positions per video; the old flattened CI is reproduced
  solely as a reporting convention, not as a CI for unseen-user performance.
* all_cached_valid_frames: every valid row already present in these dumps.
  This is a description of the inherited frame pool, not a corrected/raw-video
  evaluation. In particular, no absent >1000 frame scores are manufactured.

For each pool, videos have equal weight for the video estimand; users have
equal weight for the user estimand (each user's video means are averaged).
Paired bootstrap resamples users, carrying all their available paired videos.
The head_three_seed_mean method averages three per-seed accuracies, not logits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before importing NumPy; the audit creates no child processes.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_var] = "1"
import numpy as np


BOOTSTRAP_SEED = 20261016
LEGACY_SAMPLING_SEED = 0
TASKS = 50
TARGET_FRAMES = 200
SEED_HEADS = ("head:ad64_s0", "head:ad64_s1", "head:ad64_s2")
POOL_HEAD = "head:poolonly_s0"
HEAD_MEAN = "head_three_seed_mean_accuracy"
RAW_METHODS = ("full", "part@0", *SEED_HEADS, POOL_HEAD)
METHODS = (*RAW_METHODS, HEAD_MEAN)
MODES = ("h0", "ema_half_life_30_frames", "finite_last_30_frames", "cum")
POOLS = ("legacy_50_tasks", "all_cached_valid_frames")
COMPARISONS = (
    ("part_minus_full", "part@0", "full"),
    ("head_seed_mean_minus_part", HEAD_MEAN, "part@0"),
    ("pool52_minus_head_seed_mean", POOL_HEAD, HEAD_MEAN),
    ("pool52_minus_part", POOL_HEAD, "part@0"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def standardize(scores: np.ndarray) -> np.ndarray:
    """Match the old frame-wise population z score, retaining input dtype.

    This function receives scores only. No ground truth or validity flag enters
    normalization, temporal accumulation, or prediction.
    """
    return ((scores - scores.mean(axis=1, keepdims=True)) /
            (scores.std(axis=1, keepdims=True) + 1e-6))


def exponential_mean(scores: np.ndarray, half_life: float) -> np.ndarray:
    """Vectorized float64 EMA, in bounded blocks; no per-frame Python loop."""
    alpha = 0.5 ** (1.0 / half_life)
    state = np.zeros(scores.shape[1], dtype=np.float64)
    out = np.empty(scores.shape, dtype=np.float64)
    # For half-life >=1, 512-row blocks keep powers far above underflow.
    # Current dumps have <=1000 rows: only one or two vectorized blocks.
    for start in range(0, len(scores), 512):
        block = scores[start:start + 512].astype(np.float64, copy=False)
        powers = alpha ** np.arange(len(block), dtype=np.float64)
        uncorrected = ((1.0 - alpha) * np.cumsum(block / powers[:, None], axis=0) * powers[:, None]
                       + state[None, :] * (alpha * powers[:, None]))
        ends = np.arange(start + 1, start + len(block) + 1)
        out[start:start + len(block)] = uncorrected / (1.0 - alpha ** ends)[:, None]
        state = uncorrected[-1]
    return out


def temporal(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "h0":
        return scores
    if mode == "ema_half_life_30_frames":
        return exponential_mean(scores, 30.0)
    if mode == "cum":
        # NumPy's input-dtype cumsum matches the original cached-score scorer.
        return np.cumsum(scores, axis=0) / np.arange(1, len(scores) + 1)[:, None]
    if mode == "finite_last_30_frames":
        # New sensitivity: actual last 30 original rows, including the current
        # row. Early frames use all available history; no future rows enter.
        cs = np.vstack((np.zeros((1, scores.shape[1])),
                        np.cumsum(scores, axis=0, dtype=np.float64)))
        ends = np.arange(1, len(scores) + 1)
        starts = np.maximum(0, ends - 30)
        return (cs[ends] - cs[starts]) / (ends - starts)[:, None]
    raise ValueError(f"Unknown temporal mode: {mode}")


def draw_legacy_positions(n_valid: int, rng: random.Random) -> np.ndarray:
    return np.asarray([
        sorted(rng.sample(range(n_valid), min(TARGET_FRAMES, n_valid)))
        for _ in range(TASKS)
    ], dtype=np.int64)


def score_correctness(correct_valid: np.ndarray, positions: np.ndarray | None) -> dict:
    """Ground-truth comparison has happened; only valid-row correctness enters."""
    result = {
        "all_cached_valid_frames": float(correct_valid.mean()),
        "correct_valid_sum": float(correct_valid.sum()),
        "legacy_50_tasks": None,
        "legacy_task_n": None,
        "legacy_task_sum": None,
        "legacy_task_square_sum": None,
    }
    if positions is not None:
        task_scores = correct_valid[positions].mean(axis=1)
        result.update(
            legacy_50_tasks=float(task_scores.mean()),
            legacy_task_n=len(task_scores),
            legacy_task_sum=float(task_scores.sum()),
            legacy_task_square_sum=float(np.square(task_scores).sum()),
        )
    return result


def _scalar(archive, name: str):
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return value.item()


def read_video(path: Path, rng: random.Random, alignment_known: bool) -> tuple[dict, bool]:
    payload = path.read_bytes()  # One disk read; hash and NPZ decoding share these bytes.
    record = {
        "video": path.stem, "file": path.name, "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(), "status": "unknown", "user": None,
        "n_rows": None, "n_valid_rows": None, "n_objects": None,
        "frame_number_min": None, "frame_number_max": None,
        "missing_keys": [], "issues": [], "method_status": {}, "metrics": {},
    }
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            keys = set(archive.files)
            record["available_keys"] = sorted(keys)
            record["missing_keys"] = sorted(set(("names", "valid", "y", "user")) - keys)

            # Consume exactly the old sampling calls even if another metadata
            # field is missing. If validity itself is unknown, later RNG
            # alignment is unknown rather than silently starting a new sample.
            valid = None
            if "valid" in keys:
                candidate = np.asarray(archive["valid"])
                if candidate.ndim == 1 and candidate.dtype.kind == "b":
                    valid = candidate
                    record["n_rows"] = len(valid)
                    record["n_valid_rows"] = int(valid.sum())
                else:
                    record["issues"].append("valid is not a one-dimensional boolean array")
            if valid is None:
                record["issues"].append("legacy sampling RNG alignment unknown from this file onward")
                return record, False
            positions = draw_legacy_positions(int(valid.sum()), rng) if alignment_known else None
            if not alignment_known:
                record["issues"].append("legacy sample unknown after an earlier unreadable validity field")
            if record["missing_keys"]:
                return record, alignment_known
            if int(valid.sum()) < 50:
                record["issues"].append("dump violates the >=50-valid-frame admission rule; not silently included")
                return record, alignment_known

            names = np.asarray(archive["names"])
            if names.ndim != 1 or len(names) != len(valid):
                raise ValueError("names and valid lengths differ")
            fnums = [int(str(name).rsplit("-", 1)[1].split(".")[0]) for name in names]
            if any(b <= a for a, b in zip(fnums, fnums[1:])):
                raise ValueError("frame names are not in strictly increasing original-frame order")
            record["n_frame_number_gaps"] = sum(b != a + 1 for a, b in zip(fnums, fnums[1:]))
            if record["n_frame_number_gaps"]:
                record["issues"].append("frame-number gaps: finite window counts cached observations, not elapsed frames")
            record["frame_number_min"], record["frame_number_max"] = min(fnums), max(fnums)
            user = _scalar(archive, "user")
            if not isinstance(user, str) or not user:
                raise ValueError("user is not a nonempty string scalar")
            record["user"] = user
            y_raw = _scalar(archive, "y")
            if isinstance(y_raw, bool) or not isinstance(y_raw, (int, np.integer)) or y_raw < 0:
                raise ValueError("y is not a nonnegative integer scalar")
            y = int(y_raw)  # Evaluator-only target, never passed to the predictor.
            record["scorer_only_target_index"] = y
            record["status"] = "metadata_validated"
            correct = {}
            object_counts = set()
            for method in RAW_METHODS:
                if method not in keys:
                    record["method_status"][method] = {"status": "unknown_missing_key", "key": method}
                    continue
                raw = np.asarray(archive[method])
                if (raw.ndim != 2 or raw.shape[0] != len(valid) or raw.shape[1] <= y
                        or raw.dtype.kind != "f" or not np.isfinite(raw).all()):
                    record["method_status"][method] = {
                        "status": "unknown_invalid_scores", "shape": list(raw.shape), "dtype": str(raw.dtype)}
                    continue
                object_counts.add(raw.shape[1])
                z = standardize(raw)
                # All chronological rows, including invalid rows, enter history.
                # Validity and y are used only after making predictions.
                correct[method] = {}
                for mode in MODES:
                    predicted = temporal(z, mode).argmax(axis=1)
                    correct[method][mode] = (predicted == y)[valid].astype(np.float64)
                record["method_status"][method] = {"status": "known", "key": method,
                                                      "shape": list(raw.shape), "dtype": str(raw.dtype)}
            if len(object_counts) == 1:
                record["n_objects"] = object_counts.pop()
            elif len(object_counts) > 1:
                record["issues"].append("inconsistent object counts among score arms")
                record["status"] = "unknown_inconsistent_object_counts"
                return record, alignment_known

            if all(method in correct for method in SEED_HEADS):
                correct[HEAD_MEAN] = {
                    mode: np.mean(np.stack([correct[method][mode] for method in SEED_HEADS]), axis=0)
                    for mode in MODES
                }
                record["method_status"][HEAD_MEAN] = {
                    "status": "known", "definition": "mean of three per-seed accuracies; not a score ensemble"}
            else:
                record["method_status"][HEAD_MEAN] = {
                    "status": "unknown_missing_seed", "missing_seeds": [k for k in SEED_HEADS if k not in correct]}
            record["metrics"] = {
                mode: {method: score_correctness(values[mode], positions)
                       for method, values in correct.items()}
                for mode in MODES
            }
            return record, alignment_known
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        record["status"] = "unknown_unreadable_or_invalid_input"
        record["issues"].append(f"{type(error).__name__}: {error}")
        # An exception before valid was established makes later legacy draws
        # impossible to align. Otherwise the old calls have already been made.
        if record["n_valid_rows"] is None:
            alignment_known = False
        return record, alignment_known


def available_rows(records: list[dict], pool: str, mode: str, method: str) -> list[tuple]:
    result = []
    for record in records:
        stat = record["metrics"].get(mode, {}).get(method)
        if stat is not None and stat.get(pool) is not None:
            result.append((record, stat))
    return result


def aggregate(records: list[dict], pool: str, mode: str, method: str) -> dict:
    rows = available_rows(records, pool, mode, method)
    if not rows:
        return {"status": "unknown_no_available_scores", "n_videos": 0, "n_users": 0,
                "video_mean_accuracy_pct": None, "user_mean_accuracy_pct": None,
                "full_dump_video_mean_accuracy_pct": None, "per_user": {}}
    by_user = {}
    for record, stat in rows:
        by_user.setdefault(record["user"], []).append((record, stat))
    per_user = {
        user: {
            "n_videos": len(values), "n_cached_valid_frames": sum(r["n_valid_rows"] for r, _ in values),
            "video_mean_accuracy_pct": 100.0 * float(np.mean([s[pool] for _, s in values])),
        }
        for user, values in sorted(by_user.items())
    }
    video_mean = 100.0 * float(np.mean([stat[pool] for _, stat in rows]))
    available_names = {r["video"] for r, _ in rows}
    result = {
        "status": "complete" if len(rows) == len(records) else "partial_available_subset",
        "n_videos": len(rows), "n_users": len(by_user),
        "n_cached_valid_frames": sum(r["n_valid_rows"] for r, _ in rows),
        "video_mean_accuracy_pct": video_mean,
        "user_mean_accuracy_pct": float(np.mean([v["video_mean_accuracy_pct"] for v in per_user.values()])),
        "full_dump_video_mean_accuracy_pct": video_mean if len(rows) == len(records) else None,
        "missing_videos": [r["video"] for r in records if r["video"] not in available_names],
        "per_user": per_user,
    }
    if pool == "legacy_50_tasks":
        n = sum(stat["legacy_task_n"] for _, stat in rows)
        mean = sum(stat["legacy_task_sum"] for _, stat in rows) / n
        variance = max(0.0, sum(stat["legacy_task_square_sum"] for _, stat in rows) / n - mean * mean)
        result.update(
            n_task_video_samples=n,
            legacy_flattened_ci95_halfwidth_pct=100.0 * 1.96 * math.sqrt(variance / n),
            legacy_ci_interpretation="old reporting formula; not an unseen-user generalization interval",
        )
    else:
        result["frame_micro_accuracy_pct"] = (
            100.0 * sum(stat["correct_valid_sum"] for _, stat in rows) /
            sum(record["n_valid_rows"] for record, _ in rows)
        )
    return result


def paired_bootstrap(user_differences: dict[str, list[float]], n_bootstrap: int,
                     draw_cache: dict | None = None) -> dict:
    """Resample users with replacement; retain all videos within each draw."""
    users = sorted(user_differences)
    if not users:
        return {"status": "unknown_no_paired_scores", "video_weighted": None, "user_weighted": None}
    sums = np.asarray([sum(user_differences[u]) for u in users], dtype=np.float64)
    counts = np.asarray([len(user_differences[u]) for u in users], dtype=np.int64)
    means = sums / counts
    point_video, point_user = float(sums.sum() / counts.sum()), float(means.mean())
    result = {
        "status": "known" if len(users) >= 2 else "unknown_insufficient_user_clusters_for_ci",
        "n_users": len(users), "n_paired_videos": int(counts.sum()),
        "n_users_positive": int((means > 0).sum()), "n_users_negative": int((means < 0).sum()),
        "n_users_tied": int((means == 0).sum()),
        "per_user": {u: {"n_videos": int(counts[i]), "mean_difference_pp": 100.0 * float(means[i])}
                     for i, u in enumerate(users)},
        "video_weighted": {"difference_pp": 100.0 * point_video, "ci95_percentile_pp": None},
        "user_weighted": {"difference_pp": 100.0 * point_user, "ci95_percentile_pp": None},
    }
    if len(users) >= 2:
        if draw_cache is None:
            draw_cache = {}
        cache_key = (n_bootstrap, len(users))
        if cache_key not in draw_cache:
            draw_cache[cache_key] = np.random.default_rng(BOOTSTRAP_SEED).integers(
                0, len(users), size=(n_bootstrap, len(users)))
        draws = draw_cache[cache_key]
        # Repeated users repeat their whole clusters, not only their means.
        video_boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        user_boot = means[draws].mean(axis=1)
        for name, values in (("video_weighted", video_boot), ("user_weighted", user_boot)):
            result[name]["ci95_percentile_pp"] = (100.0 * np.quantile(values, [.025, .975])).tolist()
    return result


def compare(records: list[dict], pool: str, mode: str, left: str, right: str, n_bootstrap: int,
            draw_cache: dict) -> dict:
    per_user = {}
    paired_videos = []
    for record in records:
        metrics = record["metrics"].get(mode, {})
        left_score, right_score = metrics.get(left, {}).get(pool), metrics.get(right, {}).get(pool)
        if left_score is not None and right_score is not None:
            paired_videos.append(record["video"])
            per_user.setdefault(record["user"], []).append(left_score - right_score)
    result = paired_bootstrap(per_user, n_bootstrap, draw_cache)
    paired_set = set(paired_videos)
    result.update(
        left=left, right=right, unit="percentage_points",
        coverage="complete" if len(paired_videos) == len(records) else "paired_available_subset_only",
        missing_videos=[r["video"] for r in records if r["video"] not in paired_set],
    )
    return result


def optional_source_hashes(script: Path) -> dict:
    result = {"paired_audit.py": {"path": str(script), "sha256": sha256_file(script), "status": "known"}}
    for name in ("orbit_dump.py", "orbit_temporal.py", "evidence_head.py", "train_evidence_head.py"):
        path = script.parent.parent / name
        result[name] = ({"path": str(path), "sha256": sha256_file(path), "status": "known"}
                        if path.is_file() else {"path": str(path), "sha256": None, "status": "unknown_not_copied"})
    return result


def serializable_video(record: dict) -> dict:
    # Raw predictions/frames are not exported. The target is explicitly scorer-only.
    result = {key: value for key, value in record.items() if key != "metrics"}
    result["accuracy_pct"] = {
        pool: {mode: {method: (100.0 * record["metrics"].get(mode, {}).get(method, {})[pool]
                               if record["metrics"].get(mode, {}).get(method, {}).get(pool) is not None else None)
                      for method in METHODS} for mode in MODES}
        for pool in POOLS
    }
    return result


def self_test() -> None:
    """Small hand-computable numerical fixtures; no files are read or written."""
    toy = np.asarray([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    expected = np.asarray([[2.0, 0.0], [2.0 / 3, 4.0 / 3], [2.0 / 7, 12.0 / 7]])
    np.testing.assert_allclose(exponential_mean(toy, 1.0), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(temporal(toy, "cum"), [[2, 0], [1, 1], [2 / 3, 4 / 3]])
    # The production finite window is 30: at row 31 the first observation expires.
    long_toy = np.zeros((31, 2)); long_toy[0, 0] = 30; long_toy[-1, 1] = 30
    finite = temporal(long_toy, "finite_last_30_frames")
    np.testing.assert_allclose(finite[29], [1, 0]); np.testing.assert_allclose(finite[30], [0, 1])
    altered = toy.copy(); altered[-1] = [1000, -1000]
    for mode in MODES:
        np.testing.assert_allclose(temporal(toy, mode)[:2], temporal(altered, mode)[:2])
    np.testing.assert_array_equal(standardize(toy).argmax(1), toy.argmax(1))
    # User A contributes two videos, B one: video and user estimands differ.
    clustered = paired_bootstrap({"A": [1.0, 1.0], "B": [-1.0]}, 5000)
    assert abs(clustered["video_weighted"]["difference_pp"] - 100 / 3) < 1e-12
    assert clustered["user_weighted"]["difference_pp"] == 0.0
    assert clustered["video_weighted"]["ci95_percentile_pp"] == [-100.0, 100.0]
    zero = paired_bootstrap({"A": [0.0, 0.0], "B": [0.0]}, 50)
    assert zero["video_weighted"]["ci95_percentile_pp"] == [0.0, 0.0]
    positions = draw_legacy_positions(60, random.Random(0))
    assert positions.shape == (50, 60)
    assert np.array_equal(positions, np.tile(np.arange(60), (50, 1)))
    np.testing.assert_allclose(np.mean([[1, 0], [0, 0], [1, 1]], axis=0), [2 / 3, 1 / 3])
    print("PASS: hand-computable EMA/cumulative/window fixtures, causality, paired user bootstrap, legacy sampling.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path, help="Existing test_learned NPZ directory (read only)")
    parser.add_argument("--out", type=Path, help="New output JSON path, outside --dump")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Paired user-cluster draws (default: 5000)")
    parser.add_argument("--self-test", action="store_true", help="Run hand-computable fixtures only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.dump is None or args.out is None:
        parser.error("--dump and --out are required unless --self-test is used")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    dump, out = args.dump.resolve(), args.out.resolve()
    if not dump.is_dir():
        parser.error(f"Input dump directory does not exist: {dump}")
    if out == dump or dump in out.parents:
        parser.error("--out must be outside the read-only input dump")
    if out.exists():
        parser.error(f"Refusing to overwrite an existing result: {out}")
    files = sorted(dump.glob("*.npz"))
    if not files:
        parser.error("No NPZ files found; no result created")
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    started = datetime.now(timezone.utc).isoformat()
    meta_path = dump / "meta.json"
    metadata = {"path": str(meta_path), "sha256": None, "status": "unknown_missing", "content": None}
    if meta_path.is_file():
        metadata["sha256"] = sha256_file(meta_path)
        try:
            metadata.update(status="known", content=json.loads(meta_path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError) as error:
            metadata.update(status="unknown_unreadable", error=str(error))
    records = []
    rng, alignment_known = random.Random(LEGACY_SAMPLING_SEED), True
    for i, path in enumerate(files, 1):
        record, alignment_known = read_video(path, rng, alignment_known)
        records.append(record)
        if i % 50 == 0 or i == len(files):
            print(f"Read/scored {i}/{len(files)} cached score files", flush=True)
    aggregates = {
        pool: {mode: {method: aggregate(records, pool, mode, method) for method in METHODS} for mode in MODES}
        for pool in POOLS
    }
    draw_cache = {}
    comparisons = {
        pool: {mode: {name: compare(records, pool, mode, left, right, args.bootstrap, draw_cache)
                      for name, left, right in COMPARISONS} for mode in MODES}
        for pool in POOLS
    }
    manifest = [{k: r[k] for k in ("file", "bytes", "sha256")} for r in records]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    valid_records = [r for r in records if r["status"] == "metadata_validated"]
    methods_complete = all(all(r["method_status"].get(m, {}).get("status") == "known" for m in METHODS)
                           for r in records)
    report = {
        "schema": "orbit_independent_paired_audit_v1", "started_at_utc": started,
        "dump": str(dump), "status": ("complete" if len(valid_records) == len(records) and methods_complete
                                         else "partial_unknown_inputs"),
        "claim_scope": "Exploratory reanalysis of already-consumed cached scores; not new data or independent confirmation",
        "units": {"accuracy": "percent", "differences_and_ci": "percentage_points", "time": "seconds"},
        "input_metadata": metadata, "input_npz_manifest": manifest, "input_npz_manifest_sha256": manifest_hash,
        "source_hashes": optional_source_hashes(Path(__file__).resolve()),
        "counts": {
            "npz_files": len(records), "metadata_validated_videos": len(valid_records),
            "metadata_validated_users": len({r["user"] for r in valid_records}),
            "known_cached_rows": sum(r["n_rows"] for r in records if r["n_rows"] is not None),
            "known_cached_valid_rows": sum(r["n_valid_rows"] for r in records if r["n_valid_rows"] is not None),
            "frame_counts_complete": all(r["n_rows"] is not None for r in records),
            "all_requested_methods_complete": methods_complete,
        },
        "protocol": {
            "legacy_50_tasks": {"tasks_per_video": TASKS, "target_frames_max": TARGET_FRAMES,
                                "rng": "Python random.Random", "seed": LEGACY_SAMPLING_SEED,
                                "ordering": "sorted direct-child *.npz filenames, then 50 calls per video",
                                "positions": "indices within the existing valid-row list, sampled without replacement",
                                "compatible_old_modes": {"h0": "h0", "ema_half_life_30_frames": "h30", "cum": "cum"},
                                "rng_alignment_known_to_end": alignment_known},
            "all_cached_valid_frames": "Exact descriptive means over existing valid rows; inherits dump cap/filter order",
            "history": "All existing chronological rows, including invalid rows, precede scoring-only validity filtering",
            "ema": "Bias-corrected exponential mean; half-life 30 frames, unbounded causal history, not a 1-second hard window",
            "finite_window": "At most 30 original chronological rows including current; assumes no missing rows in cached input",
            "normalization": "Per-frame population z score over all object slots, epsilon=1e-6; input dtype retained",
            "head_seed_mean": "Arithmetic mean of separate seed accuracies (s0/s1/s2); not an ensemble prediction",
            "ground_truth": "Only scalar y comparison and valid scoring mask; never used to normalize, predict or accumulate",
            "selection": "Fixed existing arms; no best-seed, threshold, subset or model selection",
        },
        "bootstrap": {"n": args.bootstrap, "seed": BOOTSTRAP_SEED, "unit": "user cluster",
                      "type": "paired percentile, 2.5% and 97.5% quantiles",
                      "video_estimand": "Recompute sum(video paired differences)/number of videos after whole-user draws",
                      "user_estimand": "Equal mean of each sampled user's mean paired-video difference",
                      "sampling_uncertainty": "Legacy task draws held fixed; bootstrap addresses user clusters, not new model seeds",
                      "multiplicity": "Descriptive intervals, without family-wise or post-selection correction"},
        "aggregates": aggregates, "paired_comparisons": comparisons,
        "per_video": [serializable_video(r) for r in records],
        "limitations": [
            "The same public test users already informed development; these intervals do not undo adaptive test reuse.",
            "No official checkpoint predictions were supplied; no paired inference against the cited 75.38% is performed.",
            "These dumps cap original frame numbers; no scores beyond the inherited cap are recovered or invented.",
            "EMA half-life 30 frames and a finite 30-frame window are different temporal estimands; seconds require frame-rate evidence.",
            "Missing keys are unknown; partial aggregates and pairwise intersections identify their available denominators.",
        ],
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                        "processes": 1, "blas_thread_environment": {k: os.environ[k] for k in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}, "gpu_used": False},
    }
    report["cost"] = {"wall_seconds_before_json_write": time.perf_counter() - wall_start,
                      "cpu_seconds_before_json_write": time.process_time() - cpu_start,
                      "input_npz_bytes": sum(r["bytes"] for r in records),
                      "new_inference_frames": 0, "new_image_reads": 0, "new_download_bytes": 0}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(f"Users/videos: {report['counts']['metadata_validated_users']}/{len(valid_records)}; "
          f"cached valid rows: {report['counts']['known_cached_valid_rows']}; bootstrap: {args.bootstrap}")
    for pool in POOLS:
        print(pool + " (h0; paired user-cluster 95% intervals, pp)")
        for name, _, _ in COMPARISONS:
            comparison = comparisons[pool]["h0"][name]
            for weighting in ("video_weighted", "user_weighted"):
                estimate = comparison[weighting]
                print(f"  {name} / {weighting}: " + ("UNKNOWN" if estimate is None else
                      f"{estimate['difference_pp']:+.4f}, CI {estimate['ci95_percentile_pp']}"))
    print(f"JSON: {out}")
    print(f"Wall/CPU before JSON write: {report['cost']['wall_seconds_before_json_write']:.3f}/"
          f"{report['cost']['cpu_seconds_before_json_write']:.3f} s")


if __name__ == "__main__":
    main()
