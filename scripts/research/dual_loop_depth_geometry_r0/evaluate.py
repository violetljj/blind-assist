"""Join frozen target-depth features to existing REveL radial truth for Discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FEATURE_ROWS = 770
EXPECTED_JOINED_ROWS = 770
EXPECTED_MOTION_ROWS = 488


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            result[indexed[position][0]] = average_rank
        cursor = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0 else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rank(left), rank(right))


def finite_pairs(
    rows: Iterable[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    for row in rows:
        a = row.get(left_key)
        b = row.get(right_key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isfinite(float(a)) and math.isfinite(float(b)):
                left.append(float(a))
                right.append(float(b))
    return left, right


def key(selected_index: int, normalized_area: float) -> tuple[int, str]:
    return selected_index, f"{normalized_area:.12f}"


def evaluate(
    features_path: Path,
    producer_receipt_path: Path,
    radial_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    receipt = read_json(producer_receipt_path)
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("vicon_truth_opened") is not False
        or receipt.get("oracle_roi_opened") is not True
    ):
        raise ValueError(
            "producer receipt is not a completed Vicon-blind/oracle-ROI receipt"
        )
    if receipt.get("output_sha256") != sha256_file(features_path):
        raise ValueError("feature output hash differs from producer receipt")
    features = read_jsonl(features_path)
    radial = read_jsonl(radial_path)
    if len(features) != EXPECTED_FEATURE_ROWS:
        raise ValueError(f"expected {EXPECTED_FEATURE_ROWS} feature rows")

    radial_by_key = {
        key(int(row["selected_index"]), float(row["normalized_area"])): row
        for row in radial
    }
    joined: list[dict[str, Any]] = []
    for feature in features:
        match = radial_by_key.get(
            key(int(feature["selected_index"]), float(feature["normalized_area"]))
        )
        if match is None:
            raise ValueError("depth feature does not join to frozen radial ledger")
        joined.append({**feature, **match})
    if len(joined) != EXPECTED_JOINED_ROWS:
        raise ValueError("joined denominator drift")
    motion = [row for row in joined if row.get("source_motion_available") is True]
    if len(motion) != EXPECTED_MOTION_ROWS:
        raise ValueError("motion-available denominator drift")

    range_metrics: dict[str, Any] = {}
    for feature_key in ("roi_depth_median", "roi_depth_frame_normalized"):
        depth, ranges = finite_pairs(motion, feature_key, "sensor_local_range_m")
        range_metrics[feature_key] = {
            "n": len(depth),
            "spearman_with_range": spearman(depth, ranges),
            "expected_sign_for_larger_means_closer": "negative",
        }

    temporal_rows: list[dict[str, Any]] = []
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in motion:
        by_target[str(row["class_name"])].append(row)
    for target_rows in by_target.values():
        target_rows.sort(key=lambda row: int(row["source_timestamp_ns"]))
        for previous, current in zip(target_rows, target_rows[1:]):
            dt = (
                int(current["source_timestamp_ns"]) - int(previous["source_timestamp_ns"])
            ) / 1_000_000_000.0
            if dt <= 0 or dt > 2.0:
                continue
            for feature_key in ("roi_depth_median", "roi_depth_frame_normalized"):
                rate = (float(current[feature_key]) - float(previous[feature_key])) / dt
                temporal_rows.append(
                    {
                        "feature": feature_key,
                        "target": str(current["class_name"]),
                        "rate": rate,
                        "truth_signed_approach_mps": -float(
                            current["source_radial_range_rate_mps"]
                        ),
                        "truth_state": str(current["source_radial_motion"]),
                    }
                )

    temporal_metrics: dict[str, Any] = {}
    for feature_key in ("roi_depth_median", "roi_depth_frame_normalized"):
        subset = [row for row in temporal_rows if row["feature"] == feature_key]
        rates = [float(row["rate"]) for row in subset]
        truth_rates = [float(row["truth_signed_approach_mps"]) for row in subset]
        direction = Counter()
        for row in subset:
            predicted = (
                "approaching"
                if row["rate"] > 0
                else "receding"
                if row["rate"] < 0
                else "quasi_static"
            )
            direction[f"{row['truth_state']}->{predicted}"] += 1
        correct = sum(
            count
            for transition, count in direction.items()
            if transition.split("->", 1)[0] == transition.split("->", 1)[1]
        )
        wrong_signed = sum(
            count
            for transition, count in direction.items()
            if transition
            in {"approaching->receding", "receding->approaching"}
        )
        temporal_metrics[feature_key] = {
            "n": len(subset),
            "spearman_rate_with_truth_approach_rate": spearman(rates, truth_rates),
            "zero_deadband_direction_correct_fraction": (
                correct / len(subset) if subset else None
            ),
            "zero_deadband_wrong_signed_fraction": (
                wrong_signed / len(subset) if subset else None
            ),
            "direction_matrix": dict(sorted(direction.items())),
        }

    result = {
        "schema": "blindassist.dual_loop_target_depth_discovery_evaluation.v1",
        "status": "COMPLETE",
        "stage": "DISCOVERY",
        "claim_ceiling": "SINGLE_CAPTURE_ORACLE_ROI_MODEL_DEPTH_DIAGNOSTIC_ONLY",
        "model_generated_depth": True,
        "larger_output_means_closer": True,
        "threshold_search_performed": False,
        "feature_rows": len(features),
        "joined_rows": len(joined),
        "motion_available_rows": len(motion),
        "targets": dict(sorted(Counter(str(row["class_name"]) for row in motion).items())),
        "truth_states": dict(
            sorted(Counter(str(row["source_radial_motion"]) for row in motion).items())
        ),
        "range_metrics": range_metrics,
        "temporal_metrics": temporal_metrics,
        "features_sha256": sha256_file(features_path),
        "producer_receipt_sha256": sha256_file(producer_receipt_path),
        "radial_ledger_sha256": sha256_file(radial_path),
        "limitations": [
            "single burned REveL capture",
            "oracle ground-truth ROI",
            "model-generated relative depth, not sensor truth",
            "sparse 512-frame sampling",
            "no alert or fusion outcome",
            "no Android runtime or latency claim",
        ],
    }
    atomic_write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--radial-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.features,
                args.producer_receipt,
                args.radial_ledger,
                args.output,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
