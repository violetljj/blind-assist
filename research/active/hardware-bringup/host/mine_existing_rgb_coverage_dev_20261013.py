"""Index-only mining of the 18 consumed October 12 captures; no inference.

All timing comparisons retain their original clock domain.  Source-frame identity
is distinct from image-content identity and from independent scene evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[4]
KEYS = tuple("065950 070046 070523 070601 070630 070657 070849 070942 071014 072853 072939 073020 073138 073245 073554 073635 073740 073821".split())
THIN_KEYS = ("073554", "073635", "073740", "073821")
BOUNDARY = (
    "Consumed Development; weak scene names only, no new target labels. "
    "More RGB observations in the same 18 recordings do not add independent scenes. "
    "Android receipt matching is not exposure synchronization. "
    "Atom device timestamps remain unmapped. Pose must not be interpolated or "
    "copied to new RGB observations without separate validation. "
    "Existing pose/match gates and frozen protocols are unchanged."
)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(row):
    source = row["source_id"]
    boot = row.get("boot_id") or row.get("source_headers", {}).get("x-sequence-id")
    if boot is None:
        raise ValueError("source boot identity is required")
    if row.get("source_frame_id") is not None:
        return source, boot, "source_frame_id", str(row["source_frame_id"])
    if row.get("camera_timestamp_ns") is not None:
        return source, boot, "camera_timestamp_ns", str(row["camera_timestamp_ns"])
    raise ValueError("frame identity is absent")


def stats(values):
    a = sorted(values)
    if not a:
        return {"n": 0}
    def q(p):
        pos = (len(a) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(a) - 1)
        return round(a[lo] + (a[hi] - a[lo]) * (pos - lo), 6)
    return dict(n=len(a), min=round(a[0], 6), p50=q(.5), p90=q(.9), p95=q(.95), max=round(a[-1], 6),
                zero=sum(x == 0 for x in a), negative=sum(x < 0 for x in a))


def intervals(rows, field):
    return [(b[field] - a[field]) / 1e6 for a, b in zip(rows, rows[1:])]


def offsets(rows):
    a = [(r["rgb_t_ns"] - r["t_ns"]) / 1e6 for r in rows]
    return dict(future=sum(x > 0 for x in a), past=sum(x < 0 for x in a), equal=sum(x == 0 for x in a),
                signed_rgb_minus_tof_ms=stats(a), absolute_ms=stats([abs(x) for x in a]))


def portable(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "artifacts.local/work/phone-new-capture-20261012/raw")
    parser.add_argument("--cache-manifest", type=Path, default=ROOT / "artifacts.local/work/fusion-refinement-dev-20261013/cache_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts.local/work/data-potential-dev-20261013/local_mining")
    args = parser.parse_args()
    start = time.perf_counter()
    manifest_bytes = args.cache_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert set(manifest["sessions"]) == set(KEYS), "Only the 18 consumed captures are admitted"
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "run_contract.json", dict(goal="Recover unused temporal RGB coverage without inference or new labels",
         scope=list(KEYS), thin_focus=list(THIN_KEYS), cpu_command_wall_budget_s=240, gpu_s=0,
         inference_calls=0, download_bytes=0, output_budget_bytes=100_000_000,
         adjustable="Index summaries and continuity slices only", boundary=BOUNDARY,
         raw_root=portable(args.raw_root), cache_manifest=portable(args.cache_manifest),
         cache_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
         identity_rule="source_id + boot_id + source_frame_id, fallback camera_timestamp_ns",
         slice_rule="within-session source/boot-contiguous, positive device delta and positive receipt delta, both <=250 ms; >=5 observations and at least one uncached observation; no detector output used"))
    all_obs = []
    cached_ids = set()
    selected_ids = set()
    per_session = []
    slices = []
    sample_hash_checks = []
    aggregates = defaultdict(list)
    ready_all = []
    cache_all = []
    raw_input_hashes = {}
    total_valid_tof = 0
    indexed_cache_paths = set()
    for key in KEYS:
        if time.perf_counter() - start > 200:
            raise TimeoutError("Reserved 40 s for output/verification within 240 s budget")
        session = manifest["sessions"][key]
        folder = args.raw_root / session["session"]
        source_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        rgb = lines(folder / "rgb.jsonl")
        tof = [r for r in lines(folder / "tof.index.jsonl") if r.get("valid_frame")]
        total_valid_tof += len(tof)
        assert len(rgb) == source_manifest["rgb_count"]
        assert len(tof) == source_manifest["tof_count"]
        assert [r["index"] for r in rgb] == list(range(1, len(rgb) + 1))
        raw_input_hashes[key] = hashlib.sha256((folder / "rgb.jsonl").read_bytes()).hexdigest()
        ready = session["frames"]
        cached = []
        selected_local = set()
        cached_local = set()
        for f in ready:
            r = rgb[f["rgb_index"]]
            assert r["received_elapsed_ns"] == f["rgb_t_ns"]
            assert tof[f["i"]]["received_elapsed_ns"] == f["t_ns"]
            assert tof[f["i"]]["sequence"] == f["seq"]
            assert bool(abs(f["rgb_t_ns"] - f["t_ns"]) <= 100_000_000) == f["matched"]
            assert abs(f["rgb_t_ns"] - f["t_ns"]) == min(abs(x["received_elapsed_ns"] - f["t_ns"]) for x in rgb)
            rid = identity(r)
            selected_local.add(rid)
            if "cache" in f:
                cache_path = args.cache_manifest.parent / f["cache"]
                assert cache_path.is_file() and cache_path.stat().st_size > 0
                assert f["matched"] and f["pose_ok"]
                indexed_cache_paths.add(cache_path.resolve())
                cached_local.add(rid)
                cached.append(f)
        selected_ids.update(selected_local)
        cached_ids.update(cached_local)
        ready_all.extend(ready)
        cache_all.extend(cached)
        seen = set()
        unique_rgb = []
        local_obs = []
        for j, r in enumerate(rgb):
            rid = identity(r)
            path = folder / r["file"]
            assert path.is_file() and path.stat().st_size == r["bytes"]
            assert len(r["sha256"]) == 64
            ob = dict(session_key=key, session=session["session"], scene_name_weak=session["label"]["scene"],
                      rgb_index_zero_based=j, raw_index_one_based=r["index"], file=portable(path),
                      source_identity=list(rid), sha256=r["sha256"], bytes=r["bytes"],
                      received_elapsed_ns=r["received_elapsed_ns"], camera_timestamp_ns=r["camera_timestamp_ns"],
                      device_capture_timestamp_us=r.get("device_capture_timestamp_us"),
                      camera_clock_domain=r.get("camera_clock_domain"),
                      selected_by_ready_tof=rid in selected_local, has_existing_depth_cache=rid in cached_local,
                      pose_status="UNASSIGNED_NOT_INTERPOLATED", label_status="UNLABELED_OBSERVATION")
            all_obs.append(ob)
            if rid not in seen:
                seen.add(rid)
                unique_rgb.append(r)
                local_obs.append(ob)
        for j in sorted({0, len(rgb) // 2, len(rgb) - 1}):
            path = folder / rgb[j]["file"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == rgb[j]["sha256"]
            sample_hash_checks.append(dict(session=key, rgb_index_zero_based=j, sha256=actual))
        interval_sets = dict(raw_unique_receipt_ms=intervals(unique_rgb, "received_elapsed_ns"),
                             raw_unique_device_ms=intervals(unique_rgb, "camera_timestamp_ns"),
                             ready_tof_receipt_ms=intervals(ready, "t_ns"),
                             cached_tof_receipt_ms=intervals(cached, "t_ns"),
                             cached_selected_rgb_receipt_ms=intervals(cached, "rgb_t_ns"))
        for name, values in interval_sets.items():
            aggregates[name].extend(values)
        hashes = {r["sha256"] for r in rgb}
        local = dict(session_key=key, session=session["session"], scene_name_weak=session["label"]["scene"],
                     thin_focus=key in THIN_KEYS, raw_rgb_rows=len(rgb), raw_unique_source_frames=len(seen),
                     unique_content_sha256=len(hashes), duplicate_source_rows=len(rgb)-len(seen),
                     distinct_source_frames_sharing_content=len(seen)-len(hashes),
                     raw_tof_valid_frames=len(tof), ready_tof_rows=len(ready), depth_cache_files=len(cached),
                     selected_unique_rgb=len(selected_local), cached_unique_rgb=len(cached_local),
                     uncached_unique_rgb=len(seen-cached_local), ready_without_cache=len(ready)-len(cached),
                     raw_unique_rgb_never_selected=len(seen-selected_local),
                     selected_unique_rgb_without_cache=len(selected_local-cached_local),
                     matched_false_rows=sum(not f["matched"] for f in ready),
                     pose_rejected_rows=sum(not f["pose_ok"] for f in ready),
                     ready_pairing=offsets(ready), cached_pairing=offsets(cached),
                     intervals={name:stats(values) for name,values in interval_sets.items()})
        assert local["cached_unique_rgb"] + local["uncached_unique_rgb"] == len(seen)
        per_session.append(local)
        runs = []
        run = []
        for ob in local_obs:
            if run:
                prev = run[-1]
                dr = ob["received_elapsed_ns"] - prev["received_elapsed_ns"]
                dd = ob["camera_timestamp_ns"] - prev["camera_timestamp_ns"]
                if ob["source_identity"][:2] != prev["source_identity"][:2] or not (0 < dr <= 250_000_000 and 0 < dd <= 250_000_000):
                    runs.append(run)
                    run = []
            run.append(ob)
        if run:
            runs.append(run)
        for n, run in enumerate(runs):
            uncached = sum(not r["has_existing_depth_cache"] for r in run)
            if len(run) >= 5 and uncached:
                slices.append(dict(slice_id=f"{key}-continuous-{n:03d}", session_key=key, scene_name_weak=local["scene_name_weak"],
                    thin_focus=key in THIN_KEYS, start_rgb_index=run[0]["rgb_index_zero_based"], end_rgb_index=run[-1]["rgb_index_zero_based"],
                    start_received_elapsed_ns=run[0]["received_elapsed_ns"], end_received_elapsed_ns=run[-1]["received_elapsed_ns"],
                    receipt_span_s=(run[-1]["received_elapsed_ns"]-run[0]["received_elapsed_ns"])/1e9,
                    unique_rgb_frames=len(run), existing_cached_unique_rgb=len(run)-uncached, uncached_unique_rgb=uncached,
                    purpose="candidate continuity/reacquisition slice, not new labels or independent scene evidence"))
    by_id = defaultdict(list)
    by_sha = defaultdict(set)
    for ob in all_obs:
        rid = tuple(ob["source_identity"])
        by_id[rid].append(ob)
        by_sha[ob["sha256"]].add(rid)
    for rid, observations in by_id.items():
        assert len({o["sha256"] for o in observations}) == 1, f"Conflicting identity {rid}"
    uncached_ids = set(by_id) - cached_ids
    cached_shas = {by_id[rid][0]["sha256"] for rid in cached_ids}
    uncached_manifest = []
    for rid, observations in by_id.items():
        if rid in uncached_ids:
            row = dict(observations[0])
            row["content_sha256_already_cached"] = row["sha256"] in cached_shas
            row["source_frame_occurrences"] = len(observations)
            uncached_manifest.append(row)
    target = args.output / "uncached_unique_rgb.jsonl"
    with target.open("w", encoding="utf-8") as stream:
        for row in uncached_manifest:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    assert len(lines(target)) == len(uncached_ids)
    actual_cache_paths = {p.resolve() for p in (args.cache_manifest.parent / "cache").rglob("*.npz")}
    summary = dict(evidence="consumed Development index audit", boundary=BOUNDARY, sessions=len(KEYS),
        raw_rgb_rows=len(all_obs), unique_source_frames=len(by_id), duplicate_source_rows=len(all_obs)-len(by_id),
        unique_content_sha256=len(by_sha), distinct_source_frames_sharing_content=len(by_id)-len(by_sha),
        raw_tof_valid_frames=total_valid_tof, ready_tof_rows=len(ready_all), declared_depth_cache_rows=len(cache_all),
        actual_referenced_depth_cache_files=len(indexed_cache_paths), actual_cache_directory_files=len(actual_cache_paths),
        unreferenced_cache_files=len(actual_cache_paths-indexed_cache_paths), selected_unique_rgb=len(selected_ids),
        cached_unique_rgb=len(cached_ids), uncached_unique_rgb=len(uncached_ids),
        raw_unique_rgb_never_selected=len(set(by_id)-selected_ids),
        selected_unique_rgb_without_cache=len(selected_ids-cached_ids),
        uncached_new_content_sha256=len(set(by_sha)-cached_shas), uncached_id_with_already_cached_content=sum(r["content_sha256_already_cached"] for r in uncached_manifest),
        ready_without_cache=len(ready_all)-len(cache_all), ready_pairing=offsets(ready_all), cached_pairing=offsets(cache_all),
        intervals={name:stats(v) for name,v in aggregates.items()}, continuous_slices=len(slices),
        thin_focus={k:next(s for s in per_session if s["session_key"]==k) for k in THIN_KEYS},
        gpu_s=0, inference_calls=0, download_bytes=0)
    assert len(cached_ids) + len(uncached_ids) == len(by_id)
    assert len(selected_ids) <= len(ready_all)
    assert len(cache_all) == len(indexed_cache_paths)
    dump(args.output / "per_session.json", per_session)
    dump(args.output / "continuous_tracking_slices.json", dict(boundary=BOUNDARY, slices=slices))
    dump(args.output / "duplicate_content.json", {sha:[list(rid) for rid in sorted(ids)] for sha,ids in by_sha.items() if len(ids)>1})
    with (args.output / "per_session.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["session_key", "scene_name_weak", "raw_rgb_rows", "raw_unique_source_frames", "unique_content_sha256", "ready_tof_rows", "depth_cache_files", "cached_unique_rgb", "uncached_unique_rgb", "raw_receipt_p50_ms", "cache_receipt_p50_ms", "ready_future", "ready_past", "cached_future", "cached_past"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for r in per_session:
            row = {k:r[k] for k in fields if k in r}
            row.update(raw_receipt_p50_ms=r["intervals"]["raw_unique_receipt_ms"].get("p50"), cache_receipt_p50_ms=r["intervals"]["cached_tof_receipt_ms"].get("p50"), ready_future=r["ready_pairing"]["future"], ready_past=r["ready_pairing"]["past"], cached_future=r["cached_pairing"]["future"], cached_past=r["cached_pairing"]["past"])
            writer.writerow(row)
    summary["script_wall_s"] = round(time.perf_counter()-start, 3)
    dump(args.output / "summary.json", summary)
    checks = dict(status="PASS", rgb_index_rows_verified=len(all_obs), ready_rows_aligned_to_raw_rgb_and_tof=len(ready_all),
        all_ready_rows_are_receipt_nearest=True, all_raw_payload_paths_and_sizes_verified=True,
        content_identity_conflicts=0, referenced_cache_paths_present=len(indexed_cache_paths),
        conservation=f"{len(by_id)} = {len(cached_ids)} cached + {len(uncached_ids)} uncached source frames",
        uncached_manifest_rows_verified=len(uncached_manifest), raw_index_sha256=raw_input_hashes,
        sampled_payload_sha256_verified=sample_hash_checks,
        full_payload_hash_recomputation=False, new_inference_or_detection_validation=False)
    dump(args.output / "verification.json", checks)
    output_bytes = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
    assert output_bytes <= 100_000_000
    print(json.dumps({k:v for k,v in summary.items() if k not in ("thin_focus", "boundary", "intervals")}, ensure_ascii=False, indent=2))
    print(json.dumps(dict(output_bytes=output_bytes, output=portable(args.output), verification="PASS")))


if __name__ == "__main__":
    main()
