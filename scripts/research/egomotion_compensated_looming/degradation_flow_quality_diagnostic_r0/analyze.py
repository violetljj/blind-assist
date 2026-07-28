from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .extract import (
    GATE_FB_MEDIAN_LIMIT_PX,
    PAIR_COUNT,
    PROTOCOL_ID,
    gate_reasons,
    rank_average,
)


SESSIONS = (13, 14, 15, 17)
THRESHOLD_PER_S = 0.01
REQUIRED_CONSECUTIVE = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def prevalence(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(values[mask]))


def enrichment(
    target: np.ndarray, degraded: np.ndarray, eligible: np.ndarray
) -> dict[str, float | int | None]:
    exposed = eligible & degraded
    unexposed = eligible & ~degraded
    exposed_prevalence = prevalence(target, exposed)
    unexposed_prevalence = prevalence(target, unexposed)
    risk_ratio = (
        safe_ratio(exposed_prevalence, unexposed_prevalence)
        if exposed_prevalence is not None and unexposed_prevalence is not None
        else None
    )
    captured = (
        safe_ratio(
            float(np.count_nonzero(target & exposed)),
            float(np.count_nonzero(target & eligible)),
        )
        if np.any(target & eligible)
        else None
    )
    return {
        "eligible_pair_count": int(np.count_nonzero(eligible)),
        "degraded_pair_count": int(np.count_nonzero(exposed)),
        "target_pair_count": int(np.count_nonzero(target & eligible)),
        "target_prevalence_degraded": exposed_prevalence,
        "target_prevalence_complement": unexposed_prevalence,
        "risk_ratio": risk_ratio,
        "target_fraction_captured": captured,
    }


def bottom_stratum(values: Iterable[float], fraction: float = 0.2) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return rank_average(array) <= fraction


def top_stratum(values: Iterable[float], fraction: float = 0.2) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return rank_average(array) >= 1.0 - fraction


def recompute_gate_accept(
    proxy_rows: list[dict[str, Any]], median_fb_limit: float
) -> np.ndarray:
    return np.asarray(
        [
            not gate_reasons(row, median_fb_limit)
            for row in proxy_rows
        ],
        dtype=bool,
    )


def gated_triggers(
    r3_rows: list[dict[str, Any]], gate_accept: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(r3_rows), dtype=bool)
    streak = 0
    for index, (row, accepted) in enumerate(zip(r3_rows, gate_accept)):
        response = row.get("compensated_expansion_median_per_s")
        above = bool(
            accepted
            and row.get("evaluable") is True
            and response is not None
            and float(response) > THRESHOLD_PER_S
        )
        streak = streak + 1 if above else 0
        result[index] = streak >= REQUIRED_CONSECUTIVE
    return result


def summarize_session(
    session: int,
    proxy_rows: list[dict[str, Any]],
    r3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(proxy_rows) != PAIR_COUNT or len(r3_rows) != PAIR_COUNT:
        raise ValueError(f"PAIR_COUNT_MISMATCH:{session}")
    expected = list(range(PAIR_COUNT))
    if (
        [row.get("pair_index") for row in proxy_rows] != expected
        or [row.get("pair_index") for row in r3_rows] != expected
    ):
        raise ValueError(f"PAIR_IDENTITY_MISMATCH:{session}")
    if any(
        row.get("risk_label_accessed") is not False
        or row.get("response_accessed_during_extraction") is not False
        for row in proxy_rows
    ):
        raise ValueError(f"STAGE_1_FIREWALL_VIOLATION:{session}")

    evaluable = np.asarray(
        [row.get("evaluable") is True for row in r3_rows], dtype=bool
    )
    responses = np.asarray(
        [
            (
                float(row["compensated_expansion_median_per_s"])
                if row.get("compensated_expansion_median_per_s") is not None
                else float("nan")
            )
            for row in r3_rows
        ],
        dtype=np.float64,
    )
    high_response = np.zeros(PAIR_COUNT, dtype=bool)
    high_response[evaluable] = top_stratum(np.abs(responses[evaluable]))
    signed_positive = evaluable & (responses > THRESHOLD_PER_S)

    blur = bottom_stratum(
        row["sharpness_laplacian_variance"] for row in proxy_rows
    )
    low_texture = bottom_stratum(
        row["detected_features_per_valid_megapixel"] for row in proxy_rows
    )
    gait = top_stratum(
        row["gait_oscillation_proxy_score"] for row in proxy_rows
    )
    gate_accept = recompute_gate_accept(
        proxy_rows, GATE_FB_MEDIAN_LIMIT_PX
    )
    gate_rejected = ~gate_accept

    original_triggers = np.asarray(
        [
            row.get("compensated_three_pair_trigger") is True
            for row in r3_rows
        ],
        dtype=bool,
    )
    gated = gated_triggers(r3_rows, gate_accept)
    original_density = float(np.mean(original_triggers))
    gated_density = float(np.mean(gated))
    relative_reduction = (
        (original_density - gated_density) / original_density
        if original_density > 0
        else None
    )

    proxy_masks = {
        "motion_blur_proxy": blur,
        "low_texture_proxy": low_texture,
        "gait_oscillation_proxy": gait,
        "flow_quality_gate_rejected": gate_rejected,
    }
    attribution = {
        name: {
            "high_absolute_response": enrichment(
                high_response, mask, evaluable
            ),
            "signed_positive_response": enrichment(
                signed_positive, mask, evaluable
            ),
        }
        for name, mask in proxy_masks.items()
    }
    gate_proxy_enrichment = {
        name: enrichment(gate_rejected, mask, np.ones(PAIR_COUNT, dtype=bool))
        for name, mask in {
            "motion_blur_proxy": blur,
            "low_texture_proxy": low_texture,
            "gait_oscillation_proxy": gait,
        }.items()
    }

    sensitivities: dict[str, Any] = {}
    for limit in (0.5, 1.0):
        accepted = recompute_gate_accept(proxy_rows, limit)
        sensitivity_triggers = gated_triggers(r3_rows, accepted)
        sensitivities[f"median_fb_limit_{limit:.1f}px"] = {
            "gate_rejected_fraction": float(np.mean(~accepted)),
            "gated_trigger_density_fixed_601": float(
                np.mean(sensitivity_triggers)
            ),
        }

    gate_rr = attribution["flow_quality_gate_rejected"][
        "high_absolute_response"
    ]["risk_ratio"]
    proxy_rrs = [
        value["risk_ratio"]
        for value in gate_proxy_enrichment.values()
        if value["risk_ratio"] is not None
    ]
    session_checks = {
        "bounded_rejection_0p02_to_0p30": bool(
            0.02 <= float(np.mean(gate_rejected)) <= 0.30
        ),
        "gate_high_response_rr_gte_1p5": bool(
            gate_rr is not None and gate_rr >= 1.5
        ),
        "trigger_density_relative_reduction_gte_0p20": bool(
            relative_reduction is not None and relative_reduction >= 0.20
        ),
        "named_proxy_enriched_in_gate_rejections_rr_gte_1p5": bool(
            any(value >= 1.5 for value in proxy_rrs)
        ),
    }
    return {
        "session": session,
        "fixed_pair_denominator": PAIR_COUNT,
        "r3_evaluable_pair_count": int(np.count_nonzero(evaluable)),
        "r3_evaluable_fraction": float(np.mean(evaluable)),
        "proxy_stratum_fractions": {
            name: float(np.mean(mask))
            for name, mask in {
                "motion_blur_proxy": blur,
                "low_texture_proxy": low_texture,
                "gait_oscillation_proxy": gait,
            }.items()
        },
        "attribution": attribution,
        "gate_proxy_enrichment": gate_proxy_enrichment,
        "flow_quality_gate": {
            "rejected_pair_count": int(np.count_nonzero(gate_rejected)),
            "rejected_fraction_fixed_601": float(np.mean(gate_rejected)),
            "retained_r3_evaluable_pair_count": int(
                np.count_nonzero(evaluable & gate_accept)
            ),
            "retained_r3_evaluable_fraction_fixed_601": float(
                np.mean(evaluable & gate_accept)
            ),
            "original_trigger_density_fixed_601": original_density,
            "gated_trigger_density_fixed_601": gated_density,
            "relative_trigger_density_reduction": relative_reduction,
            "session_checks": session_checks,
        },
        "sensitivity": sensitivities,
    }


def analyze(
    proxy_root: Path,
    r3_runs_root: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("OUTPUT_PATH_EXISTS")
    sessions: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for session in SESSIONS:
        proxy_path = proxy_root / f"advio-{session:02d}" / "proxy_ledger.jsonl"
        r3_path = (
            r3_runs_root
            / f"advio-{session:02d}_r3_fixed_601"
            / "pair_ledger.jsonl"
        )
        for path in (proxy_path, r3_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        input_hashes[proxy_path.as_posix()] = sha256_file(proxy_path)
        input_hashes[r3_path.as_posix()] = sha256_file(r3_path)
        sessions.append(
            summarize_session(
                session, read_jsonl(proxy_path), read_jsonl(r3_path)
            )
        )

    checks = [
        row["flow_quality_gate"]["session_checks"] for row in sessions
    ]
    counts = {
        key: sum(bool(row[key]) for row in checks)
        for key in checks[0]
    }
    no_label_or_sealed_access = True
    pass_terminal = bool(
        counts["bounded_rejection_0p02_to_0p30"] >= 3
        and counts["gate_high_response_rr_gte_1p5"] >= 3
        and counts[
            "trigger_density_relative_reduction_gte_0p20"
        ] >= 3
        and counts[
            "named_proxy_enriched_in_gate_rejections_rr_gte_1p5"
        ] >= 3
        and no_label_or_sealed_access
    )
    result = {
        "schema": "rcle.degradation_flow_quality.analysis.v1",
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": input_hashes,
        "independent_descriptive_unit": "capture_session",
        "pair_records_are_longitudinal_not_independent_samples": True,
        "risk_label_accessed": False,
        "sealed_session_accessed": False,
        "threshold_per_s": THRESHOLD_PER_S,
        "required_consecutive_pairs": REQUIRED_CONSECUTIVE,
        "session_results": sessions,
        "decision_counts_of_four_sessions": counts,
        "terminal": (
            "PRIORITIZE_FLOW_QUALITY_DEVELOPMENT_OVER_MORE_ROTATION"
            if pass_terminal
            else "HOLD_FLOW_QUALITY_GATE"
        ),
        "claim_ceiling": (
            "DEVELOPMENT_PRIORITY_ONLY_NO_FALSE_TRIGGER_OR_PERFORMANCE_CLAIM"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--r3-runs-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.proxy_root.resolve(),
        args.r3_runs_root.resolve(),
        args.contract.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "decision_counts": result["decision_counts_of_four_sessions"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
