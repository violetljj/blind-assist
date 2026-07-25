"""Run the bounded USTRF four-arm continuous-score probe.

This probe deliberately reuses the frozen 15 positive / 15 matched-negative
windows.  It compares ranking signal only; it does not choose an alert
threshold, run lifecycle logic, or grant Android, human, or production
authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA = "blindassist_ustrf_four_arm_signal_probe_r1"
WINDOWS_SHA256 = "383c1a8aabfbe3bb2d5ccbf53c67a091d012b06c31a02fd44b93cea6ce1445b5"
FRAME_COUNT = 4_594
GRID_WIDTH = 63
GRID_HEIGHT = 63
ROUTE_HALF_WIDTH_FRACTION = 0.08
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 8.0
FRAME_DENSE_QUANTILE = 0.90
WINDOW_QUANTILES = (0.50, 0.90, 0.95)
MINIMUM_WINDOW_ELIGIBLE_FRACTION = 0.10
MINIMUM_WINDOW_ELIGIBLE_FRAMES = 5
PRIMARY_WINDOW_QUANTILE = "q90"
ARM_A = "A_bbox_matched_route"
ARM_B = "B_dense_matched_route"
ARM_C = "C_dense_uniform_route"
ARM_D = "D_dense_shuffled_route"
ARMS = (ARM_A, ARM_B, ARM_C, ARM_D)


class ProbeError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    bundle_dir: Path
    candidate_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeError(f"cannot read JSON {path}: {error}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def quantile(values: Sequence[float], q: float) -> float:
    require(bool(values), "cannot calculate a quantile from an empty sequence")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def quantile_name(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def two_sided_sign_p_value(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def compare_delta(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "win"
    if delta < -tolerance:
        return "loss"
    return "tie"


def summarize_deltas(deltas: Sequence[float]) -> dict[str, Any]:
    outcomes = [compare_delta(value) for value in deltas]
    wins = outcomes.count("win")
    ties = outcomes.count("tie")
    losses = outcomes.count("loss")
    lower, upper = wilson_interval(wins, len(deltas))
    return {
        "pair_count": len(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(deltas) if deltas else 0.0,
        "win_rate_wilson_95": [lower, upper],
        "median_positive_minus_negative": float(statistics.median(deltas)) if deltas else None,
        "two_sided_sign_p_value_excluding_ties": two_sided_sign_p_value(wins, losses),
    }


def source_specs(repo: Path) -> list[SourceSpec]:
    parent = (
        repo
        / "artifacts.local/evidence/ustrf-sensor-replay-r3/"
        "source-replacement-lilocbench-v1"
    )
    return [
        SourceSpec(
            source_id="lilocbench_dynamics_0_front",
            bundle_dir=parent / "dynamics_0-normalized-v1/lilocbench_dynamics_0_front",
            candidate_path=parent / "dynamics_0-candidate-evaluation-frozen-v1.json",
        ),
        SourceSpec(
            source_id="lilocbench_lt_changes_dynamics_0_front",
            bundle_dir=parent
            / "lt_changes_dynamics_0-normalized-v1/lilocbench_lt_changes_dynamics_0_front",
            candidate_path=parent / "lt_changes_dynamics_0-candidate-v1.json",
        ),
    ]


def read_frames_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProbeError(f"invalid frames JSONL at {path}:{line_number}") from error
        require(isinstance(row, dict), f"frames row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def load_source(
    spec: SourceSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    bundle_path = spec.bundle_dir / "bundle.json"
    frames_path = spec.bundle_dir / "frames.jsonl"
    bundle = load_json(bundle_path)
    frames = read_frames_jsonl(frames_path)
    candidate = load_json(spec.candidate_path)
    require(bundle["source"]["source_id"] == spec.source_id, f"{spec.source_id}: bundle identity drift")
    require(bundle["frame_count"] == len(frames), f"{spec.source_id}: bundle frame count drift")
    require(
        sha256_file(frames_path) == bundle["frames_sha256"],
        f"{spec.source_id}: frames.jsonl differs from bundle receipt",
    )
    require(candidate["candidate_alerts_frozen_before_review"] is True, f"{spec.source_id}: candidate not frozen")
    require(candidate["production_authority"] is False, f"{spec.source_id}: candidate authority drift")
    candidate_sources = [row for row in candidate["sources"] if row["source_id"] == spec.source_id]
    require(len(candidate_sources) == 1, f"{spec.source_id}: candidate source missing or duplicated")
    candidate_source = candidate_sources[0]
    require(
        candidate_source["source_frames_sha256"] == bundle["frames_sha256"],
        f"{spec.source_id}: candidate/frame binding drift",
    )
    routes = candidate_source["route_predictions"]
    require(len(routes) == len(frames), f"{spec.source_id}: route/frame count drift")
    require(
        [row["frame_id"] for row in routes] == [row["frame_id"] for row in frames],
        f"{spec.source_id}: route/frame identity drift",
    )
    hashes = {
        "bundle_sha256": sha256_file(bundle_path),
        "frames_jsonl_sha256": sha256_file(frames_path),
        "candidate_sha256": sha256_file(spec.candidate_path),
    }
    return bundle, frames, routes, hashes


def load_depth_grid(frame: Mapping[str, Any], source_root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    path = source_root / str(frame["depth_path"])
    payload = path.read_bytes()
    actual_sha = sha256_bytes(payload)
    require(actual_sha == frame["depth_sha256"], f"depth hash drift: {path}")
    with Image.open(BytesIO(payload)) as image:
        depth = np.asarray(image, dtype=np.float32)
    if frame["depth_encoding"] == "uint16_png_z_meters":
        depth = depth / float(frame["depth_scale"])
    else:
        raise ProbeError(f"unsupported depth encoding: {frame['depth_encoding']}")
    require(depth.ndim == 2, f"depth is not two-dimensional: {path}")
    height, width = depth.shape
    y_index = np.rint(np.linspace(0, height - 1, GRID_HEIGHT)).astype(np.int64)
    x_index = np.rint(np.linspace(0, width - 1, GRID_WIDTH)).astype(np.int64)
    sampled = depth[np.ix_(y_index, x_index)]
    valid = np.isfinite(sampled) & (sampled > DEPTH_MIN_M) & (sampled < DEPTH_MAX_M)
    risk = np.zeros_like(sampled, dtype=np.float64)
    risk[valid] = np.clip(
        (DEPTH_MAX_M - sampled[valid]) / (DEPTH_MAX_M - DEPTH_MIN_M),
        0.0,
        1.0,
    )
    encoded = np.where(valid, np.rint(risk * 1_000_000), 0xFFFFFFFF).astype("<u4", copy=False)
    field_sha = sha256_bytes(encoded.tobytes(order="C"))
    return risk, valid, field_sha


def route_mask(
    uv: Sequence[float],
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    require(len(uv) == 2, "route uv must have two values")
    u, v = float(uv[0]), float(uv[1])
    require(0.0 <= u < source_width and 0.0 <= v < source_height, "known route uv is out of frame")
    radius = min(source_width, source_height) * ROUTE_HALF_WIDTH_FRACTION
    xs = np.linspace(0.0, source_width - 1.0, GRID_WIDTH)
    ys = np.linspace(0.0, source_height - 1.0, GRID_HEIGHT)
    return (np.abs(ys[:, None] - v) <= radius) & (np.abs(xs[None, :] - u) <= radius)


def dense_route_score(
    risk: np.ndarray,
    valid: np.ndarray,
    uv: Sequence[float],
    *,
    source_width: int,
    source_height: int,
) -> float | None:
    eligible = valid & route_mask(
        uv,
        source_width=source_width,
        source_height=source_height,
    )
    if not np.any(eligible):
        return None
    return float(np.quantile(risk[eligible], FRAME_DENSE_QUANTILE))


def dense_uniform_score(risk: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    return float(np.quantile(risk[valid], FRAME_DENSE_QUANTILE))


def rectangle_intersection_area(
    left_a: float,
    top_a: float,
    right_a: float,
    bottom_a: float,
    left_b: float,
    top_b: float,
    right_b: float,
    bottom_b: float,
) -> float:
    width = max(0.0, min(right_a, right_b) - max(left_a, left_b))
    height = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
    return width * height


def bbox_route_score(
    detections: Sequence[Mapping[str, Any]],
    uv: Sequence[float],
    *,
    source_width: int,
    source_height: int,
) -> float:
    radius = min(source_width, source_height) * ROUTE_HALF_WIDTH_FRACTION
    u, v = float(uv[0]), float(uv[1])
    patch = (
        max(0.0, u - radius),
        max(0.0, v - radius),
        min(float(source_width), u + radius),
        min(float(source_height), v + radius),
    )
    patch_area = max(1e-12, (patch[2] - patch[0]) * (patch[3] - patch[1]))
    best = 0.0
    for detection in detections:
        if detection.get("label") != "person":
            continue
        box = detection.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        overlap = rectangle_intersection_area(
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            *patch,
        )
        score = float(detection["confidence"]) * min(1.0, overlap / patch_area)
        best = max(best, score)
    return best


def build_pairs(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for source_id in sorted({str(row["source_id"]) for row in windows}):
        source_windows = [row for row in windows if row["source_id"] == source_id]
        positives = sorted(
            (row for row in source_windows if row["window_type"] == "positive"),
            key=lambda row: int(row["start_frame"]),
        )
        negatives = sorted(
            (row for row in source_windows if row["window_type"] == "negative"),
            key=lambda row: int(row["start_frame"]),
        )
        require(len(positives) == len(negatives), f"{source_id}: positive/negative count mismatch")
        for ordinal, (positive, negative) in enumerate(zip(positives, negatives, strict=True), start=1):
            positive_count = int(positive["end_frame"]) - int(positive["start_frame"]) + 1
            negative_count = int(negative["end_frame"]) - int(negative["start_frame"]) + 1
            require(positive_count == negative_count, f"{source_id}: matched windows are not equal length")
            pairs.append(
                {
                    "pair_id": f"{source_id}__pair_{ordinal:02d}",
                    "source_id": source_id,
                    "positive_window_id": positive["window_id"],
                    "negative_window_id": negative["window_id"],
                    "frame_count_each": positive_count,
                }
            )
    return pairs


def validate_windows(
    windows: Sequence[Mapping[str, Any]],
    *,
    canonical_keys: set[tuple[str, str]],
    frames_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[tuple[str, str]]:
    require(len(windows) == 30, "frozen window count is not 30")
    require(sum(row["window_type"] == "positive" for row in windows) == 15, "positive window count drift")
    require(sum(row["window_type"] == "negative" for row in windows) == 15, "negative window count drift")
    selected: set[tuple[str, str]] = set()
    for window in windows:
        source_id = str(window["source_id"])
        frames = frames_by_source[source_id]
        start = int(window["start_frame"])
        end = int(window["end_frame"])
        require(0 <= start <= end < len(frames), f"{window['window_id']}: invalid frame interval")
        for index in range(start, end + 1):
            key = (source_id, str(frames[index]["frame_id"]))
            require(key not in selected, f"frozen windows overlap at {key}")
            selected.add(key)
    require(len(selected) == FRAME_COUNT, "frozen window union is not 4594 frames")
    require(selected == canonical_keys, "frozen window identities differ from canonical Android-parity ledger")
    return selected


def make_frame_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def compute(repo: Path) -> tuple[dict[str, Any], bytes]:
    windows_path = repo / "artifacts.local/evidence/ustrf-tracker-ttc-ablation-v1/windows-v2.json"
    canonical_path = (
        repo
        / "artifacts.local/evidence/ustrf-detector-target-attribution-r1/"
        "canonical-host-ledger-v2.json"
    )
    attribution_path = (
        repo
        / "artifacts.local/evidence/ustrf-detector-target-attribution-r1/"
        "target-attribution-result-r1.json"
    )
    require(sha256_file(windows_path) == WINDOWS_SHA256, "frozen windows SHA drift")
    windows_document = load_json(windows_path)
    windows = windows_document["windows"]
    canonical = load_json(canonical_path)
    require(canonical["frame_count"] == FRAME_COUNT, "canonical ledger frame count drift")
    require(len(canonical["frames"]) == FRAME_COUNT, "canonical ledger row count drift")
    require(
        canonical["input_tensor_exact_match_count"] == FRAME_COUNT,
        "canonical Android/host input parity is incomplete",
    )
    require(
        canonical["raw_output_within_frozen_tolerance_count"] == 4_575,
        "canonical raw-output tolerance receipt drift",
    )
    attribution = load_json(attribution_path)
    require(attribution["G1b_canonical_semantic_parity"] == "pass", "G1b semantic parity is not closed")
    require(attribution["hard_gate_passed"] is True, "target attribution hard gate is not closed")
    require(attribution["frame_count"] == FRAME_COUNT, "target attribution frame count drift")
    require(
        attribution["threshold_nms_route_event_changed"] is False,
        "target attribution parent changed detector/route/event axes",
    )
    canonical_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in canonical["frames"]:
        key = (str(row["source_id"]), str(row["frame_id"]))
        require(key not in canonical_by_key, f"duplicate canonical frame {key}")
        canonical_by_key[key] = row

    frames_by_source: dict[str, list[dict[str, Any]]] = {}
    routes_by_source: dict[str, list[dict[str, Any]]] = {}
    roots_by_source: dict[str, Path] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for spec in source_specs(repo):
        bundle, frames, routes, hashes = load_source(spec)
        frames_by_source[spec.source_id] = frames
        routes_by_source[spec.source_id] = routes
        roots_by_source[spec.source_id] = Path(bundle["source_root"])
        source_hashes[spec.source_id] = hashes
    validate_windows(
        windows,
        canonical_keys=set(canonical_by_key),
        frames_by_source=frames_by_source,
    )
    pairs = build_pairs(windows)
    require(len(pairs) == 15, "matched pair count is not 15")

    windows_by_source: dict[str, list[Mapping[str, Any]]] = {}
    donors: dict[str, Mapping[str, Any]] = {}
    for source_id in sorted(frames_by_source):
        ordered = sorted(
            (row for row in windows if row["source_id"] == source_id),
            key=lambda row: str(row["window_id"]),
        )
        windows_by_source[source_id] = ordered
        for index, window in enumerate(ordered):
            donors[str(window["window_id"])] = ordered[(index + 1) % len(ordered)]

    frame_rows: list[dict[str, Any]] = []
    window_results: dict[str, dict[str, Any]] = {}
    selected_depth_inventory: list[dict[str, str]] = []
    for source_id in sorted(frames_by_source):
        frames = frames_by_source[source_id]
        routes = routes_by_source[source_id]
        source_root = roots_by_source[source_id]
        for window in windows_by_source[source_id]:
            window_id = str(window["window_id"])
            donor = donors[window_id]
            start = int(window["start_frame"])
            end = int(window["end_frame"])
            donor_start = int(donor["start_frame"])
            donor_count = int(donor["end_frame"]) - donor_start + 1
            window_count = end - start + 1
            arm_values: dict[str, list[float]] = {arm: [] for arm in ARMS}
            eligible_count = 0
            for offset, frame_index in enumerate(range(start, end + 1)):
                donor_offset = round(offset * (donor_count - 1) / max(1, window_count - 1))
                donor_frame_index = donor_start + donor_offset
                frame = frames[frame_index]
                matched_route = routes[frame_index]
                shuffled_route = routes[donor_frame_index]
                key = (source_id, str(frame["frame_id"]))
                canonical_frame = canonical_by_key[key]
                source_width, source_height = [int(value) for value in canonical_frame["source_size"]]
                risk, valid, field_sha = load_depth_grid(frame, source_root)
                selected_depth_inventory.append(
                    {
                        "source_id": source_id,
                        "frame_id": str(frame["frame_id"]),
                        "depth_path": str(frame["depth_path"]),
                        "depth_sha256": str(frame["depth_sha256"]),
                    }
                )
                row: dict[str, Any] = {
                    "source_id": source_id,
                    "window_id": window_id,
                    "window_type": window["window_type"],
                    "frame_id": str(frame["frame_id"]),
                    "frame_index": frame_index,
                    "matched_route_status": matched_route["status"],
                    "shuffled_route_source_window_id": donor["window_id"],
                    "shuffled_route_source_frame_id": str(frames[donor_frame_index]["frame_id"]),
                    "shuffled_route_status": shuffled_route["status"],
                    "dense_field_sha256": field_sha,
                    "common_eligible": False,
                    "scores": {arm: None for arm in ARMS},
                }
                if matched_route["status"] == "known" and shuffled_route["status"] == "known":
                    matched_uv = matched_route["uv"]
                    shuffled_uv = shuffled_route["uv"]
                    score_b = dense_route_score(
                        risk,
                        valid,
                        matched_uv,
                        source_width=source_width,
                        source_height=source_height,
                    )
                    score_c = dense_uniform_score(risk, valid)
                    score_d = dense_route_score(
                        risk,
                        valid,
                        shuffled_uv,
                        source_width=source_width,
                        source_height=source_height,
                    )
                    if score_b is not None and score_c is not None and score_d is not None:
                        score_a = bbox_route_score(
                            canonical_frame["post_nms_detections_canonical_320"],
                            matched_uv,
                            source_width=source_width,
                            source_height=source_height,
                        )
                        scores = {
                            ARM_A: score_a,
                            ARM_B: score_b,
                            ARM_C: score_c,
                            ARM_D: score_d,
                        }
                        for arm, value in scores.items():
                            arm_values[arm].append(float(value))
                        row["common_eligible"] = True
                        row["matched_route_uv"] = [float(value) for value in matched_uv]
                        row["shuffled_route_uv"] = [float(value) for value in shuffled_uv]
                        row["scores"] = scores
                        eligible_count += 1
                frame_rows.append(row)
            minimum_eligible = max(
                MINIMUM_WINDOW_ELIGIBLE_FRAMES,
                math.ceil(window_count * MINIMUM_WINDOW_ELIGIBLE_FRACTION),
            )
            require(
                eligible_count >= minimum_eligible,
                f"{window_id}: common eligible frames {eligible_count} < {minimum_eligible}",
            )
            summaries: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                require(len(arm_values[arm]) == eligible_count, f"{window_id}: arm eligibility drift")
                summaries[arm] = {
                    quantile_name(q): quantile(arm_values[arm], q) for q in WINDOW_QUANTILES
                }
            window_results[window_id] = {
                "source_id": source_id,
                "window_type": window["window_type"],
                "frame_count": window_count,
                "common_eligible_frame_count": eligible_count,
                "common_eligible_fraction": eligible_count / window_count,
                "shuffled_route_source_window_id": donor["window_id"],
                "scores": summaries,
            }

    pair_rows: list[dict[str, Any]] = []
    for pair in pairs:
        positive = window_results[pair["positive_window_id"]]
        negative = window_results[pair["negative_window_id"]]
        deltas: dict[str, dict[str, float]] = {}
        outcomes: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            deltas[arm] = {}
            outcomes[arm] = {}
            for q_name in (quantile_name(q) for q in WINDOW_QUANTILES):
                delta = positive["scores"][arm][q_name] - negative["scores"][arm][q_name]
                deltas[arm][q_name] = delta
                outcomes[arm][q_name] = compare_delta(delta)
        pair_rows.append({**pair, "deltas": deltas, "outcomes": outcomes})

    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        summaries[arm] = {}
        for q_name in (quantile_name(q) for q in WINDOW_QUANTILES):
            deltas = [row["deltas"][arm][q_name] for row in pair_rows]
            by_source = {}
            for source_id in sorted(frames_by_source):
                source_deltas = [
                    row["deltas"][arm][q_name]
                    for row in pair_rows
                    if row["source_id"] == source_id
                ]
                by_source[source_id] = summarize_deltas(source_deltas)
            summaries[arm][q_name] = {
                **summarize_deltas(deltas),
                "by_source": by_source,
            }

    primary_b = summaries[ARM_B][PRIMARY_WINDOW_QUANTILE]
    primary_competitor_wins = {
        arm: summaries[arm][PRIMARY_WINDOW_QUANTILE]["wins"] for arm in (ARM_A, ARM_C, ARM_D)
    }
    conditions = {
        "B_primary_wins_at_least_12_of_15": primary_b["wins"] >= 12,
        "B_primary_wilson_lower_above_half": primary_b["win_rate_wilson_95"][0] > 0.5,
        "B_primary_wins_exceed_each_comparator_by_at_least_2": all(
            primary_b["wins"] >= wins + 2 for wins in primary_competitor_wins.values()
        ),
        "B_primary_median_delta_positive_in_each_source": all(
            row["median_positive_minus_negative"] is not None
            and row["median_positive_minus_negative"] > 0.0
            for row in primary_b["by_source"].values()
        ),
        "B_noninferior_win_count_at_q50_and_q95": all(
            summaries[ARM_B][q_name]["wins"]
            >= max(summaries[arm][q_name]["wins"] for arm in (ARM_A, ARM_C, ARM_D))
            for q_name in ("q50", "q95")
        ),
    }
    stable_win = all(conditions.values())
    decision = (
        "B_STABLE_WIN_DENSE_LIFECYCLE_NEXT"
        if stable_win
        else "STOP_CURRENT_DENSE_USTRF_EXPRESSION"
    )
    frame_lines = make_frame_lines(frame_rows)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "VALID",
        "authority": {
            "research_signal_probe_only": True,
            "alert_threshold_selected": False,
            "tracker_ttc_lifecycle_used": False,
            "android_runtime_authorized": False,
            "human_safety_authorized": False,
            "production_authorized": False,
        },
        "parent_bindings": {
            "windows_path": str(windows_path.relative_to(repo)).replace("\\", "/"),
            "windows_sha256": sha256_file(windows_path),
            "canonical_ledger_path": str(canonical_path.relative_to(repo)).replace("\\", "/"),
            "canonical_ledger_sha256": sha256_file(canonical_path),
            "target_attribution_path": str(attribution_path.relative_to(repo)).replace("\\", "/"),
            "target_attribution_sha256": sha256_file(attribution_path),
            "G1b_canonical_semantic_parity": attribution["G1b_canonical_semantic_parity"],
            "source_bindings": source_hashes,
            "selected_depth_inventory_sha256": sha256_bytes(
                canonical_json_bytes(selected_depth_inventory)
            ),
            "implementation_sha256": sha256_file(Path(__file__)),
        },
        "method": {
            "arms": list(ARMS),
            "dense_field": {
                "source": "registered_metric_depth_from_frozen_lilocbench_bundle",
                "grid": [GRID_WIDTH, GRID_HEIGHT],
                "risk_transform": "(8.0m-depth)/(8.0m-0.1m), clipped to [0,1]",
                "valid_depth_m": [DEPTH_MIN_M, DEPTH_MAX_M],
                "frame_summary": f"q{int(FRAME_DENSE_QUANTILE * 100)}",
                "same_field_required_for_B_C_D": True,
            },
            "matched_route": "frozen past_pose_prefix_only route_predictions",
            "route_patch_half_width_fraction": ROUTE_HALF_WIDTH_FRACTION,
            "uniform_route": "all valid cells of the same dense field",
            "shuffled_route": "within-source window_id cyclic shift-one, normalized-time frame mapping, no seed or labels",
            "bbox_score": "max person confidence times fraction of matched-route patch covered by bbox",
            "common_eligibility": "matched and shuffled route known plus valid dense values for B/C/D",
            "window_score_quantiles": list(WINDOW_QUANTILES),
            "primary_window_score": PRIMARY_WINDOW_QUANTILE,
            "threshold_search_or_alarm_tuning": False,
        },
        "inventory": {
            "source_count": len(frames_by_source),
            "window_count": len(windows),
            "positive_window_count": 15,
            "negative_window_count": 15,
            "matched_pair_count": len(pairs),
            "selected_frame_count": len(frame_rows),
            "all_windows_pass_common_eligibility_floor": True,
        },
        "windows": [window_results[key] | {"window_id": key} for key in sorted(window_results)],
        "pairs": pair_rows,
        "summary": summaries,
        "decision_gate": {
            "conditions": conditions,
            "all_conditions_passed": stable_win,
            "decision": decision,
            "next_authority_if_passed": "causal_lifecycle_research_only",
            "next_authority_if_failed": "stop_current_dense_expression",
        },
        "frame_scores_sha256": sha256_bytes(frame_lines),
    }
    return report, frame_lines


def write_output(output_dir: Path, report: Mapping[str, Any], frame_lines: bytes) -> None:
    if output_dir.exists():
        require(not any(output_dir.iterdir()), f"refusing to overwrite output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    (output_dir / "frame-scores.jsonl").write_bytes(frame_lines)
    (output_dir / "report.json").write_text(pretty_json_text(report), encoding="utf-8", newline="\n")


def validate_existing(repo: Path, output_dir: Path) -> dict[str, Any]:
    existing_report = load_json(output_dir / "report.json")
    existing_frame_lines = (output_dir / "frame-scores.jsonl").read_bytes()
    report, frame_lines = compute(repo)
    require(existing_frame_lines == frame_lines, "frame score replay differs")
    require(existing_report == report, "report replay differs")
    return {
        "status": "VALID_REPLAY_MATCH",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "frame_scores_sha256": sha256_file(output_dir / "frame-scores.jsonl"),
        "decision": report["decision_gate"]["decision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--validate-existing", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.output_dir is not None:
            report, frame_lines = compute(repo)
            write_output(args.output_dir.resolve(), report, frame_lines)
            result = {
                "status": report["status"],
                "decision": report["decision_gate"]["decision"],
                "report": str((args.output_dir.resolve() / "report.json")),
            }
        else:
            result = validate_existing(repo, args.validate_existing.resolve())
    except (OSError, ProbeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
