#!/usr/bin/env python3
"""Run the frozen HFTF R4 deterministic analytic-terrain component."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any


PROTOCOL_SCHEMA = "blindassist_hftf_stage_b_split_source_validation_r4"
PROTOCOL_STATUS = "FROZEN_BEFORE_R4_OUTCOME"
RESULT_SCHEMA = "blindassist_hftf_r4_analytic_terrain_result_r0"
TERRAIN_STOP = "R4_ANALYTIC_TERRAIN_MECHANICS_NOT_SUPPORTED_STOP"
TERRAIN_SUPPORTED = "R4_ANALYTIC_TERRAIN_MECHANICS_SUPPORTED"
RISK = "RISK"
NO_RISK = "NO_RISK"
UNKNOWN = "UNKNOWN"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_sibling(protocol_path: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("Parent result path must be relative")
    protocol_directory = protocol_path.resolve().parent
    resolved = (protocol_directory / relative_path).resolve()
    try:
        resolved.relative_to(protocol_directory)
    except ValueError as error:
        raise ValueError("Parent result path escapes protocol directory") from error
    return resolved


def _validate_protocol(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = protocol_path.resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("R4 protocol schema mismatch")
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("R4 protocol is not frozen before outcome")
    if protocol.get("workflow_profile") != "DEVELOPMENT_STANDARD":
        raise ValueError("R4 workflow profile mismatch")
    parent_path = _resolve_sibling(
        protocol_path, str(protocol["parent_result_path"])
    )
    observed_parent_hash = _sha256(parent_path)
    if observed_parent_hash != protocol.get("parent_result_sha256"):
        raise ValueError("R4 parent result hash mismatch")
    terrain = protocol.get("terrain_source_role")
    if not isinstance(terrain, dict):
        raise ValueError("R4 terrain source role is missing")
    if terrain.get("source") != "deterministic_analytic_metric_height_profiles":
        raise ValueError("R4 analytic terrain source role mismatch")
    if terrain.get("semantic_labels_used_for_truth") is not False:
        raise ValueError("R4 analytic truth must not use semantic labels")
    return protocol, {
        "path": str(parent_path),
        "sha256": observed_parent_hash,
        "expected_terminal": protocol.get("parent_terminal"),
        "ok": True,
    }


def _profile_is_risk(
    heights: list[float],
    maximum_step_rise_m: float,
    maximum_drop_m: float,
) -> bool:
    return any(
        delta > maximum_step_rise_m or delta < -maximum_drop_m
        for delta in (
            right - left for left, right in zip(heights, heights[1:])
        )
    )


def _derive_truth(
    exact_profile: list[float],
    supported_section_count: int,
    minimum_supported_sections: int,
    maximum_step_rise_m: float,
    maximum_drop_m: float,
) -> str:
    if supported_section_count < minimum_supported_sections:
        return UNKNOWN
    return (
        RISK
        if _profile_is_risk(
            exact_profile, maximum_step_rise_m, maximum_drop_m
        )
        else NO_RISK
    )


def _family_profiles(family: dict[str, Any]) -> list[list[float]]:
    raw_profiles = family.get("height_profiles_m")
    if raw_profiles is None:
        raw_profiles = [family.get("heights_m")]
    if (
        not isinstance(raw_profiles, list)
        or not raw_profiles
        or any(not isinstance(profile, list) for profile in raw_profiles)
    ):
        raise ValueError(f"Invalid profile family: {family.get('name')}")
    profiles = [
        [float(height) for height in profile] for profile in raw_profiles
    ]
    if any(len(profile) != 5 for profile in profiles):
        raise ValueError("Every analytic terrain profile must have five sections")
    return profiles


def _occluded_supported_indices(case_index: int, section_count: int) -> list[int]:
    # Rotate a deterministic three-section window. Sorting keeps spatial order.
    return sorted((case_index + offset) % section_count for offset in range(3))


def _sample_observations(
    exact_profile: list[float],
    supported_indices: set[int],
    observation_noise: list[float],
) -> list[list[float] | None]:
    return [
        (
            [height + noise for noise in observation_noise]
            if section_index in supported_indices
            else None
        )
        for section_index, height in enumerate(exact_profile)
    ]


def _expand_cases(terrain: dict[str, Any]) -> list[dict[str, Any]]:
    centers = [float(value) for value in terrain["section_centers_m"]]
    lateral_offsets = [
        float(value) for value in terrain["lateral_offsets_m"]
    ]
    noise = [
        float(value) for value in terrain["observation_height_noise_m"]
    ]
    candidate = terrain["candidate"]
    minimum_points = int(candidate["minimum_ground_points_per_section"])
    minimum_sections = int(
        candidate["minimum_supported_sections_for_known"]
    )
    rise = float(candidate["maximum_step_rise_m"])
    drop = float(candidate["maximum_drop_m"])
    if len(centers) != 5 or len(set(centers)) != 5:
        raise ValueError("R4 requires five unique section centers")
    if len(lateral_offsets) != 3 or len(noise) != 3:
        raise ValueError("R4 requires exactly three observations per section")
    if minimum_points != 3:
        raise ValueError("R4 minimum points per section drifted")
    if noise != [-0.01, 0.0, 0.01]:
        raise ValueError("R4 observation noise contract drifted")

    families = terrain["scenario_families"]
    if not isinstance(families, list) or not families:
        raise ValueError("R4 scenario families are missing")
    named_profiles = {
        str(family["name"]): _family_profiles(family)
        for family in families
        if family.get("name") != "occluded_unknown"
    }
    cases: list[dict[str, Any]] = []
    for family in families:
        family_name = str(family["name"])
        count = int(family["count"])
        expected_truth = str(family["truth"])
        if count <= 0:
            raise ValueError(f"Invalid family count: {family_name}")
        if family_name == "occluded_unknown":
            base_names = [str(name) for name in family["base_profiles"]]
            supported_count = int(family["supported_section_count"])
            if supported_count != 3 or len(base_names) == 0:
                raise ValueError("Occluded family support contract drifted")
            for case_index in range(count):
                base_name = base_names[case_index % len(base_names)]
                if base_name not in named_profiles:
                    raise ValueError(f"Unknown occluded base profile: {base_name}")
                occurrence = case_index // len(base_names)
                profiles = named_profiles[base_name]
                profile_index = occurrence % len(profiles)
                exact_profile = profiles[profile_index]
                supported_indices = _occluded_supported_indices(
                    case_index, len(centers)
                )
                # Truth is derived before any noisy observation is sampled.
                truth = _derive_truth(
                    exact_profile,
                    supported_count,
                    minimum_sections,
                    rise,
                    drop,
                )
                if truth != expected_truth:
                    raise ValueError(
                        f"Protocol truth mismatch for {family_name}_{case_index:02d}"
                    )
                observations = _sample_observations(
                    exact_profile, set(supported_indices), noise
                )
                cases.append(
                    {
                        "case_id": f"{family_name}_{case_index:02d}",
                        "family": family_name,
                        "base_family": base_name,
                        "profile_index": profile_index,
                        "exact_profile_m": list(exact_profile),
                        "supported_section_indices": supported_indices,
                        "truth": truth,
                        "observations_m": observations,
                    }
                )
            continue

        profiles = named_profiles[family_name]
        for case_index in range(count):
            profile_index = case_index % len(profiles)
            exact_profile = profiles[profile_index]
            supported_indices = list(range(len(centers)))
            # Truth is derived before any noisy observation is sampled.
            truth = _derive_truth(
                exact_profile,
                len(supported_indices),
                minimum_sections,
                rise,
                drop,
            )
            if truth != expected_truth:
                raise ValueError(
                    f"Protocol truth mismatch for {family_name}_{case_index:02d}"
                )
            observations = _sample_observations(
                exact_profile, set(supported_indices), noise
            )
            cases.append(
                {
                    "case_id": f"{family_name}_{case_index:02d}",
                    "family": family_name,
                    "base_family": None,
                    "profile_index": profile_index,
                    "exact_profile_m": list(exact_profile),
                    "supported_section_indices": supported_indices,
                    "truth": truth,
                    "observations_m": observations,
                }
            )
    return cases


def _section_medians(
    case: dict[str, Any], minimum_points_per_section: int
) -> list[tuple[int, float]]:
    medians: list[tuple[int, float]] = []
    for index, points in enumerate(case["observations_m"]):
        if points is None or len(points) < minimum_points_per_section:
            continue
        medians.append((index, float(median(float(value) for value in points))))
    return medians


def _candidate_prediction(
    case: dict[str, Any], candidate: dict[str, Any]
) -> str:
    medians = _section_medians(
        case, int(candidate["minimum_ground_points_per_section"])
    )
    if len(medians) < int(candidate["minimum_supported_sections_for_known"]):
        return UNKNOWN
    heights = [height for _, height in medians]
    return (
        RISK
        if _profile_is_risk(
            heights,
            float(candidate["maximum_step_rise_m"]),
            float(candidate["maximum_drop_m"]),
        )
        else NO_RISK
    )


def _semantic_safe_prediction(case: dict[str, Any]) -> str:
    return NO_RISK if _section_medians(case, 1) else UNKNOWN


def _endpoint_delta_prediction(
    case: dict[str, Any], candidate: dict[str, Any]
) -> str:
    medians = _section_medians(case, 1)
    if len(medians) < 2:
        return UNKNOWN
    endpoint_delta = medians[-1][1] - medians[0][1]
    return (
        RISK
        if endpoint_delta > float(candidate["maximum_step_rise_m"])
        or endpoint_delta < -float(candidate["maximum_drop_m"])
        else NO_RISK
    )


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _evaluate_predictions(
    cases: list[dict[str, Any]], predictions: dict[str, str]
) -> dict[str, Any]:
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "abstained_positive": 0,
        "abstained_negative": 0,
    }
    unknown_truth = {RISK: 0, NO_RISK: 0, UNKNOWN: 0}
    family_risk_totals: dict[str, int] = {}
    family_true_positives: dict[str, int] = {}
    known_scenarios = 0
    classified_known_scenarios = 0
    for case in cases:
        truth = str(case["truth"])
        prediction = predictions[case["case_id"]]
        if prediction not in {RISK, NO_RISK, UNKNOWN}:
            raise ValueError(f"Invalid prediction: {prediction}")
        if truth == UNKNOWN:
            unknown_truth[prediction] += 1
            continue
        known_scenarios += 1
        if prediction != UNKNOWN:
            classified_known_scenarios += 1
        if truth == RISK:
            family = str(case["family"])
            family_risk_totals[family] = family_risk_totals.get(family, 0) + 1
            if prediction == RISK:
                confusion["true_positive"] += 1
                family_true_positives[family] = (
                    family_true_positives.get(family, 0) + 1
                )
            elif prediction == NO_RISK:
                confusion["false_negative"] += 1
            else:
                confusion["abstained_positive"] += 1
        elif prediction == RISK:
            confusion["false_positive"] += 1
        elif prediction == NO_RISK:
            confusion["true_negative"] += 1
        else:
            confusion["abstained_negative"] += 1

    tp = confusion["true_positive"]
    fp = confusion["false_positive"]
    tn = confusion["true_negative"]
    fn = confusion["false_negative"] + confusion["abstained_positive"]
    negative_total = (
        confusion["true_negative"]
        + confusion["false_positive"]
        + confusion["abstained_negative"]
    )
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, negative_total)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    unknown_count = sum(unknown_truth.values())
    return {
        "confusion": confusion,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "known_coverage": _safe_divide(
            classified_known_scenarios, known_scenarios
        ),
        "unknown_truth_predictions": unknown_truth,
        "unknown_abstention_rate": _safe_divide(
            unknown_truth[UNKNOWN], unknown_count
        ),
        "unknown_to_safe_violations": unknown_truth[NO_RISK],
        "per_hazard_family_recall": {
            family: _safe_divide(
                family_true_positives.get(family, 0), total
            )
            for family, total in sorted(family_risk_totals.items())
        },
    }


def _gate(
    name: str,
    observed: Any,
    operator: str,
    threshold: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def run(protocol_path: Path) -> dict[str, Any]:
    protocol, parent_validation = _validate_protocol(protocol_path)
    terrain = protocol["terrain_source_role"]
    candidate_contract = terrain["candidate"]
    cases = _expand_cases(terrain)

    candidate_predictions = {
        case["case_id"]: _candidate_prediction(case, candidate_contract)
        for case in cases
    }
    semantic_predictions = {
        case["case_id"]: _semantic_safe_prediction(case) for case in cases
    }
    endpoint_predictions = {
        case["case_id"]: _endpoint_delta_prediction(case, candidate_contract)
        for case in cases
    }
    candidate_metrics = _evaluate_predictions(cases, candidate_predictions)
    baseline_metrics = {
        "semantic_support_is_safe": _evaluate_predictions(
            cases, semantic_predictions
        ),
        "endpoint_elevation_delta": _evaluate_predictions(
            cases, endpoint_predictions
        ),
    }
    best_baseline_name, best_baseline = max(
        baseline_metrics.items(), key=lambda item: (item[1]["f1"], item[0])
    )
    f1_delta = candidate_metrics["f1"] - best_baseline["f1"]

    known_count = sum(case["truth"] != UNKNOWN for case in cases)
    unknown_count = sum(case["truth"] == UNKNOWN for case in cases)
    gates = terrain["gates"]
    family_recalls = candidate_metrics["per_hazard_family_recall"]
    ordered_gates = [
        _gate(
            "exact_scenario_count",
            len(cases),
            "==",
            int(gates["exact_scenario_count"]),
            len(cases) == int(gates["exact_scenario_count"]),
        ),
        _gate(
            "exact_known_scenario_count",
            known_count,
            "==",
            int(gates["exact_known_scenario_count"]),
            known_count == int(gates["exact_known_scenario_count"]),
        ),
        _gate(
            "exact_unknown_scenario_count",
            unknown_count,
            "==",
            int(gates["exact_unknown_scenario_count"]),
            unknown_count == int(gates["exact_unknown_scenario_count"]),
        ),
        _gate(
            "candidate_known_coverage_on_known_scenarios",
            candidate_metrics["known_coverage"],
            ">=",
            float(gates["minimum_candidate_known_coverage_on_known_scenarios"]),
            candidate_metrics["known_coverage"]
            >= float(
                gates["minimum_candidate_known_coverage_on_known_scenarios"]
            ),
        ),
        _gate(
            "unknown_abstention_rate",
            candidate_metrics["unknown_abstention_rate"],
            ">=",
            float(gates["minimum_unknown_abstention_rate"]),
            candidate_metrics["unknown_abstention_rate"]
            >= float(gates["minimum_unknown_abstention_rate"]),
        ),
        *[
            _gate(
                f"candidate_{metric}",
                candidate_metrics[metric],
                ">=",
                float(gates[f"minimum_candidate_{metric}"]),
                candidate_metrics[metric]
                >= float(gates[f"minimum_candidate_{metric}"]),
            )
            for metric in ("precision", "recall", "f1", "specificity")
        ],
        _gate(
            "recall_each_hazard_family",
            family_recalls,
            ">=",
            float(gates["minimum_recall_each_hazard_family"]),
            bool(family_recalls)
            and all(
                value
                >= float(gates["minimum_recall_each_hazard_family"])
                for value in family_recalls.values()
            ),
        ),
        _gate(
            "f1_delta_over_best_baseline",
            f1_delta,
            ">=",
            float(gates["minimum_f1_delta_over_best_baseline"]),
            f1_delta
            >= float(gates["minimum_f1_delta_over_best_baseline"]),
        ),
        _gate(
            "unknown_to_safe_violations",
            candidate_metrics["unknown_to_safe_violations"],
            "<=",
            int(gates["maximum_unknown_to_safe_violations"]),
            candidate_metrics["unknown_to_safe_violations"]
            <= int(gates["maximum_unknown_to_safe_violations"]),
        ),
    ]
    supported = all(item["passed"] for item in ordered_gates)
    terminal = TERRAIN_SUPPORTED if supported else TERRAIN_STOP
    case_results = [
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "base_family": case["base_family"],
            "profile_index": case["profile_index"],
            "exact_profile_m": case["exact_profile_m"],
            "supported_section_indices": case["supported_section_indices"],
            "truth": case["truth"],
            "candidate": candidate_predictions[case["case_id"]],
            "semantic_support_is_safe": semantic_predictions[case["case_id"]],
            "endpoint_elevation_delta": endpoint_predictions[case["case_id"]],
        }
        for case in cases
    ]
    return {
        "schema": RESULT_SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_ceiling": terrain["evidence_ceiling"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "parent_result_validation": parent_validation,
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "truth_derivation": (
            "exact_frozen_profile_and_support_count_before_observation_sampling;"
            "adjacent_rise_strictly_gt_0.18_or_drop_strictly_lt_-0.15"
        ),
        "scenario_count": len(cases),
        "known_scenario_count": known_count,
        "unknown_scenario_count": unknown_count,
        "family_counts": {
            family["name"]: int(family["count"])
            for family in terrain["scenario_families"]
        },
        "cases": case_results,
        "candidate": {
            "name": candidate_contract["name"],
            "metrics": candidate_metrics,
        },
        "baselines": {
            name: {"name": name, "metrics": metrics}
            for name, metrics in baseline_metrics.items()
        },
        "best_baseline": {
            "name": best_baseline_name,
            "f1": best_baseline["f1"],
        },
        "candidate_f1_delta_over_best_baseline": f1_delta,
        "ordered_gates": ordered_gates,
        "all_terrain_gates_passed": supported,
        "joint_terminal_decided": False,
        "joint_terminal_if_obstacle_component_also_passes": protocol[
            "joint_ordered_terminals"
        ][-1],
        "stage_c_protocol_freeze_authorized": False,
        "stage_c_execution_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "production_authorized": False,
        "safety_claim_authorized": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = run(args.protocol.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {"terminal": report["terminal"], "output": str(output)},
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
