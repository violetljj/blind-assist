"""Deterministic fixed-clip analysis units for the D0-A successor.

The successor deliberately keeps Agent output at observation level.  This
module only converts a predeclared, label-independent set of observations into
a fixed analysis-unit summary.  It never infers onset, clearance, continuity,
or a parent-natural-event boundary from labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


PRESENT = "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT"
NO_EVIDENCE = "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE"
NOT_EVALUABLE = "NOT_EVALUABLE"
OBSERVATION_LABELS = (PRESENT, NO_EVIDENCE, NOT_EVALUABLE)

STABLE_PRESENT = "STABLE_PRESENT"
STABLE_NO_EVIDENCE = "STABLE_NO_EVIDENCE"
MIXED_OBSERVATION = "MIXED_OBSERVATION"
UNIT_STATES = (STABLE_PRESENT, STABLE_NO_EVIDENCE, MIXED_OBSERVATION, NOT_EVALUABLE)


class FixedClipUnitError(ValueError):
    """Raised when a fixed unit or review cannot be interpreted fail-closed."""


def _require_string(row: dict[str, Any], key: str, *, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise FixedClipUnitError(f"{where}: {key} must be a non-empty string")
    return value


def _require_int(row: dict[str, Any], key: str, *, where: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixedClipUnitError(f"{where}: {key} must be an integer")
    return value


def derive_unit_state(labels: Iterable[str]) -> str:
    """Apply the frozen observation-to-unit mapping.

    The mapping is intentionally conservative and has no tunable majority
    threshold:

    * any NOT_EVALUABLE makes the unit NOT_EVALUABLE;
    * all PRESENT makes STABLE_PRESENT;
    * all NO_EVIDENCE makes STABLE_NO_EVIDENCE;
    * a PRESENT/NO_EVIDENCE mixture is MIXED_OBSERVATION.
    """

    values = list(labels)
    if not values:
        raise FixedClipUnitError("cannot derive a state from zero observations")
    unknown = sorted(set(values).difference(OBSERVATION_LABELS))
    if unknown:
        raise FixedClipUnitError(f"unknown observation labels: {unknown}")
    if NOT_EVALUABLE in values:
        return NOT_EVALUABLE
    if all(value == PRESENT for value in values):
        return STABLE_PRESENT
    if all(value == NO_EVIDENCE for value in values):
        return STABLE_NO_EVIDENCE
    return MIXED_OBSERVATION


def _review_index(review_observations: list[dict[str, Any]], *, where: str) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for ordinal, row in enumerate(review_observations):
        if not isinstance(row, dict):
            raise FixedClipUnitError(f"{where}[{ordinal}]: expected object")
        unit_id = _require_string(row, "unit_id", where=f"{where}[{ordinal}]")
        slot = _require_int(row, "slot_ordinal", where=f"{where}[{ordinal}]")
        label = row.get("label")
        if label not in OBSERVATION_LABELS:
            raise FixedClipUnitError(f"{where}[{ordinal}]: invalid label {label!r}")
        key = (unit_id, slot)
        if key in indexed:
            raise FixedClipUnitError(f"{where}: duplicate observation key {key}")
        indexed[key] = row
    return indexed


def _manifest_index(observations: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for ordinal, row in enumerate(observations):
        if not isinstance(row, dict):
            raise FixedClipUnitError(f"manifest observations[{ordinal}]: expected object")
        unit_id = _require_string(row, "unit_id", where=f"manifest observations[{ordinal}]")
        slot = _require_int(row, "slot_ordinal", where=f"manifest observations[{ordinal}]")
        if row.get("claim_critical") is not True:
            raise FixedClipUnitError(
                f"manifest observations[{ordinal}]: successor calibration requires claim_critical=true"
            )
        key = (unit_id, slot)
        if key in indexed:
            raise FixedClipUnitError(f"manifest: duplicate observation key {key}")
        indexed[key] = row
    if not indexed:
        raise FixedClipUnitError("manifest contains zero observations")
    return indexed


def derive_fixed_units(
    manifest_observations: list[dict[str, Any]],
    review_observations: list[dict[str, Any]],
    *,
    review_name: str,
) -> list[dict[str, Any]]:
    """Derive fixed-unit summaries from one raw observation review."""

    manifest = _manifest_index(manifest_observations)
    review = _review_index(review_observations, where=review_name)
    if set(review) != set(manifest):
        missing = sorted(set(manifest).difference(review))
        extra = sorted(set(review).difference(manifest))
        raise FixedClipUnitError(f"{review_name}: manifest key mismatch; missing={missing}; extra={extra}")

    grouped: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, expected in manifest.items():
        unit_id, slot = key
        grouped[unit_id].append((slot, expected, review[key]))

    units: list[dict[str, Any]] = []
    for unit_id in sorted(grouped):
        rows = sorted(grouped[unit_id], key=lambda item: item[0])
        slots = [slot for slot, _, _ in rows]
        if slots != list(range(len(slots))):
            raise FixedClipUnitError(f"{review_name}: non-contiguous slots for {unit_id}: {slots}")
        expected = [expected for _, expected, _ in rows]
        observed = [reviewed for _, _, reviewed in rows]
        expected_frames = [_require_int(row, "source_frame_index", where=unit_id) for row in expected]
        if expected_frames != sorted(expected_frames) or len(set(expected_frames)) != len(expected_frames):
            raise FixedClipUnitError(f"{review_name}: fixed unit frames are not strictly increasing: {unit_id}")
        labels = [row["label"] for row in observed]
        units.append(
            {
                "unit_id": unit_id,
                "source_id": _require_string(expected[0], "source_id", where=unit_id),
                "session_id": _require_string(expected[0], "session_id", where=unit_id),
                "clip_id": _require_string(expected[0], "clip_id", where=unit_id),
                "start_frame_index": expected_frames[0],
                "end_frame_index": expected_frames[-1],
                "observation_count": len(expected),
                "observation_labels": labels,
                "unit_state": derive_unit_state(labels),
                "boundary_rule": "DECLARED_FIXED_TIME_WINDOW_AND_SLOT_ORDER",
                "review_name": review_name,
            }
        )
    return units


def compare_observation_reviews(
    manifest: dict[str, Any],
    primary: dict[str, Any],
    isolated: dict[str, Any],
    *,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Return descriptive calibration metrics and the fail-closed decision."""

    expected = manifest.get("observations")
    if not isinstance(expected, list):
        raise FixedClipUnitError("successor manifest observations are missing")
    primary_rows = primary.get("observations")
    isolated_rows = isolated.get("observations")
    if not isinstance(primary_rows, list) or not isinstance(isolated_rows, list):
        raise FixedClipUnitError("both reviews must contain an observations array")

    manifest_index = _manifest_index(expected)
    primary_index = _review_index(primary_rows, where="primary review")
    isolated_index = _review_index(isolated_rows, where="isolated review")
    expected_keys = set(manifest_index)
    if set(primary_index) != expected_keys or set(isolated_index) != expected_keys:
        raise FixedClipUnitError("review observations do not exactly cover the frozen fixed slots")

    comparisons = []
    unresolved = 0
    union_not_evaluable = 0
    agreement_count = 0
    critical_agreement_count = 0
    critical_count = 0
    for key in sorted(expected_keys):
        p = primary_index[key]
        s = isolated_index[key]
        p_label = p["label"]
        s_label = s["label"]
        matches = p_label == s_label
        agreement_count += int(matches)
        critical = manifest_index[key].get("claim_critical") is True
        critical_count += int(critical)
        critical_agreement_count += int(matches and critical)
        unresolved += int(not matches)
        union_not_evaluable += int(NOT_EVALUABLE in (p_label, s_label))
        comparisons.append(
            {
                "unit_id": key[0],
                "slot_ordinal": key[1],
                "primary_label": p_label,
                "isolated_label": s_label,
                "match": matches,
                "claim_critical": critical,
            }
        )

    primary_units = derive_fixed_units(expected, primary_rows, review_name="primary")
    isolated_units = derive_fixed_units(expected, isolated_rows, review_name="isolated")
    primary_by_id = {row["unit_id"]: row for row in primary_units}
    isolated_by_id = {row["unit_id"]: row for row in isolated_units}
    unit_state_matches = sum(
        int(primary_by_id[unit_id]["unit_state"] == isolated_by_id[unit_id]["unit_state"])
        for unit_id in primary_by_id
    )
    unit_count = len(primary_units)
    source_ids = {row["source_id"] for row in expected}
    total = len(expected)
    overall = agreement_count / total if total else None
    critical = critical_agreement_count / critical_count if critical_count else None
    unresolved_fraction = unresolved / total if total else None
    not_evaluable_fraction = union_not_evaluable / total if total else None
    unit_state_match_rate = unit_state_matches / unit_count if unit_count else None

    def threshold_pass(name: str, actual: float | None, *, minimum: bool = True) -> bool:
        threshold = gates.get(name)
        if actual is None or not isinstance(threshold, (int, float)):
            return False
        return actual >= threshold if minimum else actual <= threshold

    gate_results = {
        "minimum_fixed_units": unit_count >= int(gates.get("minimum_fixed_units", 0)),
        "minimum_source_count": len(source_ids) >= int(gates.get("minimum_source_count", 0)),
        "overall_observation_label_agreement": threshold_pass(
            "overall_observation_label_agreement", overall
        ),
        "claim_critical_observation_label_agreement": threshold_pass(
            "claim_critical_observation_label_agreement", critical
        ),
        "unresolved_fraction": threshold_pass("unresolved_fraction", unresolved_fraction, minimum=False),
        "union_not_evaluable_fraction": threshold_pass(
            "union_not_evaluable_fraction", not_evaluable_fraction, minimum=False
        ),
        "fixed_unit_boundary_reproducibility": True,
        "fixed_unit_state_match_rate": threshold_pass("fixed_unit_state_match_rate", unit_state_match_rate),
    }
    all_pass = all(gate_results.values())
    return {
        "schema_version": "blindassist.central_obstruction_d0a_successor_calibration_result.v1",
        "protocol_id": manifest.get("protocol_id"),
        "evidence_instance": manifest.get("evidence_instance"),
        "status": "VALID",
        "analysis_unit": "FIXED_CLIP",
        "natural_event_grouping_used": False,
        "third_agent_adjudication_used": False,
        "observation_count": total,
        "fixed_unit_count": unit_count,
        "source_count": len(source_ids),
        "primary_review_id": primary.get("review_id"),
        "isolated_review_id": isolated.get("review_id"),
        "metrics": {
            "overall_observation_label_agreement": overall,
            "claim_critical_observation_label_agreement": critical,
            "unresolved_fraction": unresolved_fraction,
            "union_not_evaluable_fraction": not_evaluable_fraction,
            "fixed_unit_boundary_reproducibility": 1.0,
            "fixed_unit_state_match_rate": unit_state_match_rate,
            "unresolved_observation_count": unresolved,
        },
        "gate_results": gate_results,
        "comparisons": comparisons,
        "primary_fixed_units": primary_units,
        "isolated_fixed_units": isolated_units,
        "decision": (
            {
                "terminal": "D0_A_SUCCESSOR_FIXED_CLIP_CALIBRATION_PASS",
                "central_obstruction_role": "OBSERVATION_LEVEL_AUXILIARY_FEATURE_CANDIDATE",
                "d0a2_authorized": False,
                "d0a3_authorized": False,
                "d0a4_authorized": False,
                "next_permitted_action": "SEPARATE_REVIEW_REQUIRED_BEFORE_ANY_D0_A2",
            }
            if all_pass
            else {
                "terminal": "CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY",
                "central_obstruction_role": "AUXILIARY_FEATURE_ONLY",
                "d0a2_authorized": False,
                "d0a3_authorized": False,
                "d0a4_authorized": False,
                "next_permitted_action": "STOP_D0_A3_A4_AND_DO_NOT_EXTEND_THIS_ROUTE",
            }
        ),
    }


def validate_review_envelope(review: dict[str, Any], *, expected_protocol_id: str, where: str) -> None:
    if review.get("protocol_id") != expected_protocol_id:
        raise FixedClipUnitError(f"{where}: protocol id mismatch")
    if review.get("candidate_output_visible") is not False:
        raise FixedClipUnitError(f"{where}: candidate output firewall is not closed")
    if review.get("prior_review_visible") is not False:
        raise FixedClipUnitError(f"{where}: prior review was visible")
    if review.get("other_review_visible_before_submission") is not False:
        raise FixedClipUnitError(f"{where}: other review was visible")
    if review.get("source_only_view") is not True:
        raise FixedClipUnitError(f"{where}: source-only view is not declared")
    context = review.get("review_context")
    if context not in {"FRESH_ISOLATED_PRIMARY", "FRESH_ISOLATED_SECOND"}:
        raise FixedClipUnitError(f"{where}: review context is not fresh isolated")
