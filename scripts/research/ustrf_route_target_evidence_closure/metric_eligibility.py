"""Candidate-blind R2-L1 metric eligibility materialization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from contract import ContractError, load_json, sha256_file


SCHEMA = "blindassist_ustrf_route_target_metric_eligibility_r2_l1_v1"
MASK_SCHEMA = "blindassist_ustrf_metric_eligibility_mask_r2_l1_v1"
RECEIPT_SCHEMA = "blindassist_ustrf_metric_denominator_receipt_r2_l1_v1"
METRICS = (
    "event_recall",
    "critical_miss",
    "repeat",
    "clearance",
    "regeneration",
    "false_alerts_per_minute",
    "evidence_age",
    "unknown_or_stale_alert",
)
ACTIVE_ROLES = frozenset({"approaching_route", "route_intersecting"})
ALLOWED_ROUTE_STATES = frozenset({"known", "unknown"})
EXPOSURE_EXCLUSION_REASONS = (
    "source_not_authorized_for_r2_l1_exposure",
    "frame_id_not_consecutive",
    "timestamp_nonpositive",
    "timestamp_gap_exceeds_maximum",
    "causal_route_unknown_at_endpoint",
    "route_relevant_person_truth_incomplete_at_endpoint",
    "active_truth_role_at_endpoint",
)
PARENT_METRIC_ALIASES = {
    "event_recall": "event_recall",
    "critical_miss": "critical_miss",
    "repeat": "repeat_within_observation",
    "clearance": "clearance",
    "regeneration": "event_regeneration_after_clear",
    "false_alerts_per_minute": "false_alerts_per_minute",
    "evidence_age": "evidence_age",
    "unknown_or_stale_alert": "unknown_or_stale_active_alert",
}
FORBIDDEN_CANDIDATE_PATH_FRAGMENTS = (
    "holdout-candidate-selection",
    "holdout-candidate-result",
    "holdout-app-detector-ledger",
    "seen-oracle-attribution",
    "ustrf-tracker-ttc-ablation-v1/result",
    "candidate-profile",
)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    return value


def _get_path(value: Any, dotted: str) -> Any:
    current = value
    for token in dotted.split("."):
        if not isinstance(current, dict) or token not in current:
            raise ContractError(f"missing asserted field {dotted}")
        current = current[token]
    return current


class BoundLoader:
    """Load only the frozen dependency allowlist and record every opened JSON."""

    def __init__(self, config: Mapping[str, Any], repo: Path):
        self.config = config
        self.repo = repo.resolve()
        self.rows: dict[str, Mapping[str, Any]] = {}
        self.paths: dict[str, Path] = {}
        self.read_ids: set[str] = set()
        self.manifest_receipt: list[dict[str, Any]] = []

        rows = config.get("candidate_blind_dependency_manifest")
        if not isinstance(rows, list) or not rows:
            raise ContractError("candidate_blind_dependency_manifest must be non-empty")
        forbidden = tuple(
            str(value).lower()
            for value in _require_object(
                config.get("dependency_policy"), "dependency_policy"
            ).get("forbidden_path_fragments", [])
        )
        for index, raw in enumerate(rows):
            row = _require_object(raw, f"dependency[{index}]")
            dep_id = str(row.get("id", ""))
            if not dep_id or dep_id in self.rows:
                raise ContractError("dependency ids must be unique and non-empty")
            relative = Path(str(row.get("path", "")))
            if relative.is_absolute():
                raise ContractError(f"dependency {dep_id} must be repo-relative")
            normalized = relative.as_posix().lower()
            if any(fragment in normalized for fragment in forbidden):
                raise ContractError(f"dependency {dep_id} points at forbidden output")
            path = (self.repo / relative).resolve()
            try:
                path.relative_to(self.repo)
            except ValueError as exc:
                raise ContractError(f"dependency {dep_id} escapes repository") from exc
            if not path.is_file():
                raise ContractError(f"dependency {dep_id} is missing")
            actual_sha = sha256_file(path)
            if actual_sha != row.get("sha256"):
                raise ContractError(f"dependency {dep_id} hash mismatch")
            self.rows[dep_id] = row
            self.paths[dep_id] = path
            self.manifest_receipt.append(
                {
                    "id": dep_id,
                    "path": relative.as_posix(),
                    "sha256": actual_sha,
                    "bytes": path.stat().st_size,
                }
            )

    def load(self, dep_id: str) -> Any:
        if dep_id not in self.rows:
            raise ContractError(f"attempted non-allowlisted dependency read: {dep_id}")
        value = load_json(self.paths[dep_id])
        assertions = self.rows[dep_id].get("assertions", {})
        if not isinstance(assertions, dict):
            raise ContractError(f"dependency {dep_id} assertions must be an object")
        for dotted, expected in assertions.items():
            actual = _get_path(value, str(dotted))
            if actual != expected:
                raise ContractError(
                    f"dependency {dep_id} assertion failed: {dotted}"
                )
        self.read_ids.add(dep_id)
        return value

    def require_all_read(self) -> None:
        expected = set(self.rows)
        if self.read_ids != expected:
            missing = sorted(expected - self.read_ids)
            extra = sorted(self.read_ids - expected)
            raise ContractError(
                f"dependency read set mismatch; missing={missing}, extra={extra}"
            )


def validate_config(config: Any, *, repo: Path) -> Mapping[str, Any]:
    row = _require_object(config, "metric eligibility config")
    if row.get("schema") != SCHEMA:
        raise ContractError("metric eligibility config schema mismatch")
    if row.get("status") != "candidate_blind_materialization_protocol_frozen":
        raise ContractError("metric eligibility protocol is not frozen")
    if tuple(row.get("metric_roster", [])) != METRICS:
        raise ContractError("metric roster drifted")
    event_universe = _require_object(row.get("event_universe"), "event_universe")
    if event_universe.get("expected_event_count") != 6369:
        raise ContractError("event universe expected count drifted")
    authority = _require_object(row.get("authority"), "authority")
    forbidden_true = (
        "candidate_execution_in_this_task",
        "candidate_winner_allowed",
        "selection_confirmation_android_h2_human_or_production_authority",
    )
    if any(authority.get(field) is not False for field in forbidden_true):
        raise ContractError("metric materialization authority expanded")
    dependency_policy = _require_object(
        row.get("dependency_policy"), "dependency_policy"
    )
    if (
        dependency_policy.get("exact_allowlist_required") is not True
        or dependency_policy.get(
            "directory_scan_or_glob_for_inputs_allowed"
        )
        is not False
        or dependency_policy.get("candidate_output_inputs_read_must_equal")
        != 0
        or dependency_policy.get(
            "candidate_module_import_or_execution_allowed"
        )
        is not False
        or tuple(dependency_policy.get("forbidden_path_fragments", []))
        != FORBIDDEN_CANDIDATE_PATH_FRAGMENTS
    ):
        raise ContractError("candidate-blind dependency policy drifted")
    parent = _require_object(row.get("parent_standard"), "parent_standard")
    parent_path = repo / str(parent.get("path", ""))
    if not parent_path.is_file() or sha256_file(parent_path) != parent.get("sha256"):
        raise ContractError("parent evidence maturity standard drifted")
    if row.get("parent_metric_aliases") != PARENT_METRIC_ALIASES:
        raise ContractError("parent metric alias map drifted")
    if (
        tuple(row.get("negative_exposure_pair_exclusion_reason_priority", []))
        != EXPOSURE_EXCLUSION_REASONS
    ):
        raise ContractError("negative exposure exclusion taxonomy drifted")
    parent_standard = load_json(parent_path)
    levels = {
        str(level.get("id")): level
        for level in parent_standard.get("maturity_levels", [])
    }
    l1_parent = _require_object(
        levels.get("L1_EXPLORATORY_METRIC_PROFILE"), "parent L1 level"
    )
    floors = _require_object(
        l1_parent.get("per_metric_screening_floors"), "parent L1 floors"
    )
    expected_event_floors = {
        "event_recall": int(floors["event_recall_event_count"]),
        "critical_miss": int(floors["critical_event_count"]),
        "repeat": int(floors["repeat_observed_event_count"]),
        "clearance": int(floors["clearance_event_count"]),
        "regeneration": int(floors["event_regeneration_interval_count"]),
    }
    for metric, expected in expected_event_floors.items():
        field = (
            "minimum_actual_candidate_delivered_denominator"
            if metric == "repeat"
            else "minimum_denominator"
        )
        if int(row["l1_readiness"][metric][field]) != expected:
            raise ContractError(f"{metric} L1 floor does not inherit parent V2")
    expected_exposure_ns = int(
        float(floors["negative_exposure_minutes"]) * 60_000_000_000
    )
    if (
        int(
            row["l1_readiness"]["false_alerts_per_minute"][
                "minimum_exposure_ns"
            ]
        )
        != expected_exposure_ns
    ):
        raise ContractError("false-alert L1 floor does not inherit parent V2")
    evaluability = _require_object(
        parent_standard.get("metric_evaluability"), "parent metric evaluability"
    )
    if (
        float(evaluability["clearance"]["assessment_horizon_ms"])
        != float(
            row["candidate_blind_operational_definitions"]["clearance"][
                "assessment_horizon_ms"
            ]
        )
        or float(
            evaluability["event_regeneration_after_clear"][
                "post_clear_observation_horizon_ms"
            ]
        )
        != float(
            row["candidate_blind_operational_definitions"]["regeneration"][
                "assessment_horizon_ms"
            ]
        )
    ):
        raise ContractError("lifecycle horizons do not inherit parent V2")
    return row


def _event_key(
    dataset_group: str, source_id: str, sequence_id: str, event_id: str
) -> str:
    return "::".join((dataset_group, source_id, sequence_id, event_id))


def _metric_entry(
    classification: str,
    *,
    reasons: Iterable[str] = (),
    censor_state: str | None = None,
    support_variant: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if classification not in {"eligible", "ineligible", "not_event_grain"}:
        raise ContractError(f"invalid metric classification {classification}")
    row: dict[str, Any] = {
        "classification": classification,
        "reasons": sorted(set(reasons)),
    }
    if censor_state is not None:
        row["censor_state"] = censor_state
    if support_variant is not None:
        row["support_variant"] = support_variant
    if details:
        row["details"] = dict(details)
    return row


def _contiguous_runs(frames: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted(set(frames))
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _critical_interval(
    *,
    critical: bool,
    role_frames: Iterable[int],
    route_status: Mapping[int, str],
) -> tuple[tuple[int, int] | None, list[str]]:
    if not critical:
        return None, ["not_critical_event"]
    qualifying = [
        run for run in _contiguous_runs(role_frames) if run[1] - run[0] + 1 >= 3
    ]
    if len(qualifying) != 1:
        return None, ["critical_interval_not_frozen_or_ambiguous"]
    start, end = qualifying[0]
    if any(route_status.get(frame) != "known" for frame in range(start, end + 1)):
        return None, ["causal_route_not_known_for_scored_interval"]
    return (start, end), []


def _continuous_post_clear_identity_ms(
    *,
    clear_frame: int,
    clear_timestamp_ns: int,
    identity_frames: Mapping[int, int],
    maximum_gap_ns: int,
) -> float | None:
    """Measure only the gap-free same-person interval immediately after clear."""

    frame_id = clear_frame + 1
    previous_timestamp_ns = clear_timestamp_ns
    last_timestamp_ns: int | None = None
    while frame_id in identity_frames:
        timestamp_ns = int(identity_frames[frame_id])
        gap_ns = timestamp_ns - previous_timestamp_ns
        if gap_ns <= 0 or gap_ns > maximum_gap_ns:
            break
        last_timestamp_ns = timestamp_ns
        previous_timestamp_ns = timestamp_ns
        frame_id += 1
    if last_timestamp_ns is None:
        return None
    return (last_timestamp_ns - clear_timestamp_ns) / 1_000_000.0


def _require_matching_terminal_clear(
    consensus_clear_frame: int, proxy_clear_frame: Any, *, event_key: Any
) -> None:
    if proxy_clear_frame is None or int(proxy_clear_frame) != consensus_clear_frame:
        raise ContractError(
            f"LILocBench terminal clear anchor drifted for {event_key}"
        )


def _base_event_row(
    *,
    dataset_group: str,
    provenance_family: str,
    source_id: str,
    sequence_id: str,
    event_id: str,
    truth_status: str,
    critical: bool,
    onset_frame: int | None,
    alertable_start_frame: int | None,
    clear_frame: int | None,
    end_frame: int | None,
    raw_exclusion_reasons: Iterable[str],
) -> dict[str, Any]:
    return {
        "unit_id": _event_key(dataset_group, source_id, sequence_id, event_id),
        "dataset_group": dataset_group,
        "provenance_family": provenance_family,
        "source_id": source_id,
        "sequence_id": sequence_id,
        "event_id": event_id,
        "truth_status": truth_status,
        "critical": bool(critical),
        "anchors": {
            "onset_frame": onset_frame,
            "alertable_start_frame": alertable_start_frame,
            "alertable_deadline_frame": None,
            "truth_terminal_clear_frame": clear_frame,
            "end_frame": end_frame,
        },
        "raw_exclusion_reasons": sorted(set(raw_exclusion_reasons)),
        "metrics": {},
    }


def _classify_event(
    row: dict[str, Any],
    *,
    identity_onset_to_alertable: bool,
    route_onset_to_alertable: bool,
    identity_through_clear: bool,
    complete_active_episode: bool,
    critical_interval: tuple[int, int] | None,
    critical_reasons: Iterable[str],
    post_clear_followup_ms: float | None,
    post_clear_identity_followup_ms: float | None,
    superseded_nonvisual: bool = False,
    incomplete_all_person_truth: bool = False,
    lifecycle_identity_loss: bool = False,
) -> None:
    metrics = row["metrics"]
    recall_reasons = ["alertable_deadline_not_frozen"]
    if superseded_nonvisual:
        recall_reasons.extend(
            [
                "superseded_nonvisual_lidar_event_proposal",
                "camera_bound_person_identity_missing",
            ]
        )
    elif not identity_onset_to_alertable:
        recall_reasons.append("person_identity_not_bound_onset_to_alertable")
    if not route_onset_to_alertable:
        recall_reasons.append("causal_route_not_known_for_scored_interval")
    metrics["event_recall"] = _metric_entry(
        "ineligible",
        reasons=recall_reasons,
        details={
            "identity_onset_to_alertable": identity_onset_to_alertable,
            "causal_route_onset_to_alertable": route_onset_to_alertable,
        },
    )

    critical_reason_list = list(critical_reasons)
    if superseded_nonvisual and row["critical"]:
        critical_reason_list.extend(
            [
                "superseded_nonvisual_lidar_event_proposal",
                "camera_bound_person_identity_missing",
            ]
        )
    if row["critical"] and critical_interval is not None and not superseded_nonvisual:
        metrics["critical_miss"] = _metric_entry(
            "eligible",
            details={
                "critical_interval_start_frame": critical_interval[0],
                "critical_interval_end_frame": critical_interval[1],
            },
        )
    else:
        metrics["critical_miss"] = _metric_entry(
            "ineligible", reasons=critical_reason_list
        )

    if superseded_nonvisual:
        metrics["repeat"] = _metric_entry(
            "ineligible",
            reasons=[
                "superseded_nonvisual_lidar_event_proposal",
                "camera_bound_person_identity_missing",
            ],
        )
    elif complete_active_episode:
        metrics["repeat"] = _metric_entry(
            "eligible",
            support_variant="complete_active_episode_truth_pool",
            details={
                "scoring_denominator_activation": "first_candidate_delivery_required"
            },
        )
    else:
        repeat_reasons = ["person_identity_not_continuous"]
        if row["anchors"]["truth_terminal_clear_frame"] is None:
            repeat_reasons.append(
                "right_censored_identity_loss"
                if lifecycle_identity_loss
                else "right_censored_administrative"
            )
        metrics["repeat"] = _metric_entry(
            "ineligible",
            reasons=repeat_reasons,
            censor_state=(
                "right_censored_identity_loss"
                if lifecycle_identity_loss
                else "right_censored_administrative"
            ),
        )

    clear_frame = row["anchors"]["truth_terminal_clear_frame"]
    row["observability"] = {
        "same_person_truth_terminal_clear_observed": (
            clear_frame is not None and identity_through_clear
        )
    }
    if clear_frame is None:
        clearance_reasons = [
            "not_evaluable_pre_clear",
            "truth_terminal_clear_missing",
        ]
        if lifecycle_identity_loss:
            clearance_reasons.append("right_censored_identity_loss")
        metrics["clearance"] = _metric_entry(
            "ineligible",
            reasons=clearance_reasons,
            censor_state="not_evaluable_pre_clear",
        )
    elif not identity_through_clear:
        metrics["clearance"] = _metric_entry(
            "ineligible",
            reasons=[
                "not_evaluable_pre_clear",
                "truth_clear_not_covered_by_same_person_identity",
                "right_censored_identity_loss",
            ],
            censor_state="not_evaluable_pre_clear",
        )
    elif post_clear_followup_ms is None or post_clear_followup_ms < 1500.0:
        metrics["clearance"] = _metric_entry(
            "ineligible",
            reasons=[
                "post_clear_followup_shorter_than_horizon",
                "right_censored_administrative",
            ],
            censor_state="right_censored_administrative",
            details={"post_clear_followup_ms": post_clear_followup_ms},
        )
    else:
        metrics["clearance"] = _metric_entry(
            "eligible",
            details={"post_clear_followup_ms": post_clear_followup_ms},
        )

    if clear_frame is None:
        metrics["regeneration"] = _metric_entry(
            "ineligible",
            reasons=["not_evaluable_pre_clear", "truth_terminal_clear_missing"],
            censor_state="not_evaluable_pre_clear",
        )
    elif not identity_through_clear:
        metrics["regeneration"] = _metric_entry(
            "ineligible",
            reasons=[
                "not_evaluable_pre_clear",
                "truth_clear_not_covered_by_same_person_identity",
                "right_censored_identity_loss",
            ],
            censor_state="not_evaluable_pre_clear",
        )
    elif post_clear_followup_ms is None or post_clear_followup_ms < 2000.0:
        metrics["regeneration"] = _metric_entry(
            "ineligible",
            reasons=[
                "post_clear_followup_shorter_than_horizon",
                "right_censored_administrative",
            ],
            censor_state="right_censored_administrative",
            details={"post_clear_followup_ms": post_clear_followup_ms},
        )
    elif (
        post_clear_identity_followup_ms is None
        or post_clear_identity_followup_ms < 2000.0
    ):
        metrics["regeneration"] = _metric_entry(
            "ineligible",
            reasons=["post_clear_identity_not_bound"],
            censor_state="right_censored_identity_loss",
            details={
                "post_clear_followup_ms": post_clear_followup_ms,
                "post_clear_identity_followup_ms": post_clear_identity_followup_ms,
            },
        )
    else:
        metrics["regeneration"] = _metric_entry(
            "eligible",
            details={
                "post_clear_followup_ms": post_clear_followup_ms,
                "post_clear_identity_followup_ms": post_clear_identity_followup_ms,
            },
        )

    if incomplete_all_person_truth:
        row["raw_exclusion_reasons"] = sorted(
            set(row["raw_exclusion_reasons"])
            | {"incomplete_all_person_route_role_truth"}
        )
    metrics["false_alerts_per_minute"] = _metric_entry(
        "not_event_grain",
        reasons=["metric_uses_exposure_ledger_not_event_denominator"],
        details={"ledger_id": "negative_exposure_intervals"},
    )
    metrics["evidence_age"] = _metric_entry(
        "not_event_grain",
        reasons=[
            "metric_uses_frame_ledger_not_event_denominator",
            "candidate_consuming_timestamp_required",
        ],
        details={"ledger_id": "preoutput_frame_masks"},
    )
    metrics["unknown_or_stale_alert"] = _metric_entry(
        "not_event_grain",
        reasons=[
            "metric_uses_frame_ledger_not_event_denominator",
            "candidate_alert_outcome_required",
        ],
        details={"ledger_id": "preoutput_frame_masks"},
    )
    if tuple(metrics) != METRICS:
        raise ContractError("event metric roster construction drifted")


class Materialization:
    def __init__(self, config: Mapping[str, Any], repo: Path):
        self.config = config
        self.repo = repo
        self.loader = BoundLoader(config, repo)
        self.events: list[dict[str, Any]] = []
        self.frame_units: dict[
            tuple[str, str, str, int], tuple[int, str]
        ] = {}
        self.exposure_elements: set[
            tuple[str, str, str, int, int]
        ] = set()
        self.excluded_exposure_ns: Counter[str] = Counter()
        self.exposure_pair_units: dict[str, dict[str, Any]] = {}

    def add_frame(
        self,
        *,
        family: str,
        source_id: str,
        sequence_id: str,
        frame_id: int,
        timestamp_ns: int,
        route_state: str,
    ) -> None:
        key = (family, source_id, sequence_id, frame_id)
        value = (timestamp_ns, route_state)
        existing = self.frame_units.get(key)
        if existing is not None and existing != value:
            raise ContractError(f"conflicting frame truth for {key}")
        self.frame_units[key] = value

    def add_exposure_pair(
        self,
        *,
        family: str,
        source_id: str,
        sequence_id: str,
        start_ns: int,
        end_ns: int,
        authorized: bool,
    ) -> None:
        if end_ns <= start_ns or end_ns - start_ns > 1_000_000_000:
            raise ContractError("invalid negative exposure interval")
        if authorized:
            self.exposure_elements.add(
                (family, source_id, sequence_id, start_ns, end_ns)
            )
        else:
            self.excluded_exposure_ns[source_id] += end_ns - start_ns

    def record_exposure_candidate_pair(
        self,
        *,
        family: str,
        source_id: str,
        sequence_id: str,
        start_frame_id: int,
        end_frame_id: int,
        start_ns: int,
        end_ns: int,
        reasons: Iterable[str],
        authorized: bool,
    ) -> None:
        reason_set = {str(reason) for reason in reasons}
        if not authorized:
            reason_set.add("source_not_authorized_for_r2_l1_exposure")
        unknown = reason_set - set(EXPOSURE_EXCLUSION_REASONS)
        if unknown:
            raise ContractError(f"unknown negative exposure reasons: {sorted(unknown)}")
        ordered_reasons = [
            reason for reason in EXPOSURE_EXCLUSION_REASONS if reason in reason_set
        ]
        delta_ns = end_ns - start_ns
        legal_duration_ns = delta_ns if 0 < delta_ns <= 1_000_000_000 else None
        classification = "eligible" if not ordered_reasons else "ineligible"
        unit_id = (
            f"{family}::{source_id}::{sequence_id}::"
            f"{start_frame_id}-{end_frame_id}"
        )
        row = {
            "unit_id": unit_id,
            "provenance_family": family,
            "source_id": source_id,
            "sequence_id": sequence_id,
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": legal_duration_ns,
            "classification": classification,
            "primary_exclusion_reason": (
                ordered_reasons[0] if ordered_reasons else None
            ),
            "exclusion_reasons": ordered_reasons,
        }
        existing = self.exposure_pair_units.get(unit_id)
        if existing is not None and existing != row:
            raise ContractError(f"conflicting negative exposure pair {unit_id}")
        self.exposure_pair_units[unit_id] = row
        if classification == "eligible":
            if legal_duration_ns is None:
                raise ContractError("eligible negative exposure pair lacks duration")
            self.add_exposure_pair(
                family=family,
                source_id=source_id,
                sequence_id=sequence_id,
                start_ns=start_ns,
                end_ns=end_ns,
                authorized=True,
            )
        elif reason_set == {"source_not_authorized_for_r2_l1_exposure"}:
            if legal_duration_ns is None:
                raise ContractError("unauthorized exposure pair lacks legal duration")
            self.add_exposure_pair(
                family=family,
                source_id=source_id,
                sequence_id=sequence_id,
                start_ns=start_ns,
                end_ns=end_ns,
                authorized=False,
            )

    def process_lilocbench(self) -> None:
        dynamics = self.loader.load("lilocbench_dynamics_event_consensus")
        lt = self.loader.load("lilocbench_lt_event_consensus")
        review = self.loader.load("lilocbench_review_bundle")
        blind = self.loader.load("lilocbench_blind_review_bundle")
        proxy = self.loader.load("lilocbench_person_route_truth")

        consensus_sources: dict[str, Mapping[str, Any]] = {}
        for document in (dynamics, lt):
            for source in document.get("sources", []):
                if source.get("route_event_admitted") is True:
                    consensus_sources[str(source["source_id"])] = source

        review_sources = {str(row["source_id"]): row for row in review["sources"]}
        blind_sources = {str(row["source_id"]): row for row in blind["sources"]}
        window_lookup: dict[tuple[str, str], tuple[Mapping[str, Any], str]] = {}
        blind_frame_to_window: dict[tuple[str, int], str] = {}
        for source_id, source in review_sources.items():
            blind_source = blind_sources.get(source_id)
            if blind_source is None or len(source["windows"]) != len(
                blind_source["windows"]
            ):
                raise ContractError("LILocBench blind/review window inventory drifted")
            for visible, hidden in zip(
                source["windows"], blind_source["windows"], strict=True
            ):
                if len(visible["frames"]) != len(hidden["frames"]):
                    raise ContractError("LILocBench blind window frame count drifted")
                for left, right in zip(
                    visible["frames"], hidden["frames"], strict=True
                ):
                    if (
                        left["frame_id"] != right["frame_id"]
                        or left["image_sha256"] != right["image_sha256"]
                    ):
                        raise ContractError("LILocBench blind frame identity drifted")
                blind_id = str(hidden["blind_window_id"])
                if visible.get("event_id") is not None:
                    window_lookup[(source_id, str(visible["event_id"]))] = (
                        visible,
                        blind_id,
                    )
                for frame in visible["frames"]:
                    frame_id = int(frame["frame_id"])
                    blind_frame_to_window[(source_id, frame_id)] = blind_id
                    self.add_frame(
                        family="lilocbench",
                        source_id=source_id,
                        sequence_id=source_id,
                        frame_id=frame_id,
                        timestamp_ns=int(frame["source_capture_timestamp_ns"]),
                        route_state=str(frame["route_status"]),
                    )

        target_episodes: dict[tuple[str, str], Mapping[str, Any]] = {}
        active_by_window_frame: set[tuple[str, int]] = set()
        for source in proxy["sources"]:
            source_id = str(source["source_id"])
            for episode in source["person_episodes"]:
                if episode.get("is_frozen_target") is True:
                    target_episodes[
                        (source_id, str(episode["legacy_event_id"]))
                    ] = episode
                for frame in episode["frames"]:
                    if frame.get("role") in ACTIVE_ROLES:
                        active_by_window_frame.add(
                            (str(episode["blind_window_id"]), int(frame["frame_id"]))
                        )
        quarantined_windows = {
            str(row["blind_window_id"])
            for row in proxy.get("quarantined_identity_episodes", [])
        }

        for source_id, source in consensus_sources.items():
            for truth in source["events"]:
                event_id = str(truth["event_id"])
                key = (source_id, event_id)
                if key not in window_lookup or key not in target_episodes:
                    raise ContractError(f"missing LILocBench event evidence for {key}")
                window, _blind_id = window_lookup[key]
                target = target_episodes[key]
                target_frames = [int(frame["frame_id"]) for frame in target["frames"]]
                route_status = {
                    int(frame["frame_id"]): str(frame["route_status"])
                    for frame in window["frames"]
                }
                onset = int(truth["onset_frame"])
                alertable = int(truth["alertable_frame"])
                clear = int(truth["passed_or_cleared_frame"])
                proxy_clear = target.get("event_truth", {}).get("clear_frame")
                _require_matching_terminal_clear(
                    clear, proxy_clear, event_key=key
                )
                identity_onset_alertable = (
                    min(target_frames) <= onset
                    and max(target_frames) >= alertable
                    and all(
                        frame in set(target_frames)
                        for frame in range(onset, alertable + 1)
                    )
                )
                route_onset_alertable = all(
                    route_status.get(frame) == "known"
                    for frame in range(onset, alertable + 1)
                )
                identity_through_clear = (
                    min(target_frames) <= alertable
                    and max(target_frames) >= clear
                    and all(
                        frame in set(target_frames)
                        for frame in range(alertable, clear + 1)
                    )
                )
                role_frames = [
                    int(frame["frame_id"])
                    for frame in target["frames"]
                    if frame.get("role") == "route_intersecting"
                ]
                interval, critical_reasons = _critical_interval(
                    critical=bool(truth["critical"]),
                    role_frames=role_frames,
                    route_status=route_status,
                )
                window_frames = {
                    int(frame["frame_id"]): frame for frame in window["frames"]
                }
                clear_row = window_frames.get(clear)
                last_row = window["frames"][-1]
                followup_ms = (
                    (
                        int(last_row["source_capture_timestamp_ns"])
                        - int(clear_row["source_capture_timestamp_ns"])
                    )
                    / 1_000_000.0
                    if clear_row is not None
                    else None
                )
                target_identity_timestamps = {
                    int(frame["frame_id"]): int(
                        frame["source_capture_timestamp_ns"]
                    )
                    for frame in target["frames"]
                    if int(frame["frame_id"]) > clear
                }
                post_identity_ms = (
                    _continuous_post_clear_identity_ms(
                        clear_frame=clear,
                        clear_timestamp_ns=int(
                            clear_row["source_capture_timestamp_ns"]
                        ),
                        identity_frames=target_identity_timestamps,
                        maximum_gap_ns=int(
                            self.config["candidate_blind_operational_definitions"][
                                "regeneration"
                            ]["maximum_adjacent_identity_frame_gap_ns"]
                        ),
                    )
                    if clear_row is not None
                    else None
                )
                row = _base_event_row(
                    dataset_group="lilocbench_seen",
                    provenance_family="lilocbench",
                    source_id=source_id,
                    sequence_id=source_id,
                    event_id=event_id,
                    truth_status="accepted_model_review_consensus",
                    critical=bool(truth["critical"]),
                    onset_frame=onset,
                    alertable_start_frame=alertable,
                    clear_frame=clear,
                    end_frame=int(truth["end_frame"]),
                    raw_exclusion_reasons=[],
                )
                _classify_event(
                    row,
                    identity_onset_to_alertable=identity_onset_alertable,
                    route_onset_to_alertable=route_onset_alertable,
                    identity_through_clear=identity_through_clear,
                    complete_active_episode=identity_through_clear,
                    critical_interval=interval,
                    critical_reasons=critical_reasons,
                    post_clear_followup_ms=followup_ms,
                    post_clear_identity_followup_ms=post_identity_ms,
                    lifecycle_identity_loss=not identity_through_clear,
                )
                self.events.append(row)

        for source_id, source in review_sources.items():
            frames_by_id: dict[int, tuple[Mapping[str, Any], str]] = {}
            for window, hidden in zip(
                source["windows"],
                blind_sources[source_id]["windows"],
                strict=True,
            ):
                blind_id = str(hidden["blind_window_id"])
                for frame in window["frames"]:
                    frame_id = int(frame["frame_id"])
                    value = (frame, blind_id)
                    existing = frames_by_id.get(frame_id)
                    if existing is not None and existing != value:
                        raise ContractError(
                            f"conflicting LILocBench frame window binding "
                            f"{source_id}:{frame_id}"
                        )
                    frames_by_id[frame_id] = value
            ordered_frames = [
                (frame_id, *frames_by_id[frame_id])
                for frame_id in sorted(frames_by_id)
            ]
            for index in range(len(ordered_frames) - 1):
                left_frame, left, left_blind_id = ordered_frames[index]
                right_frame, right, right_blind_id = ordered_frames[index + 1]
                start_ns = int(left["source_capture_timestamp_ns"])
                end_ns = int(right["source_capture_timestamp_ns"])
                reasons: list[str] = []
                if right_frame != left_frame + 1:
                    reasons.append("frame_id_not_consecutive")
                if end_ns <= start_ns:
                    reasons.append("timestamp_nonpositive")
                elif end_ns - start_ns > 1_000_000_000:
                    reasons.append("timestamp_gap_exceeds_maximum")
                if (
                    str(left["route_status"]) != "known"
                    or str(right["route_status"]) != "known"
                ):
                    reasons.append("causal_route_unknown_at_endpoint")
                if (
                    left_blind_id != right_blind_id
                    or left_blind_id in quarantined_windows
                    or right_blind_id in quarantined_windows
                ):
                    reasons.append(
                        "route_relevant_person_truth_incomplete_at_endpoint"
                    )
                if (
                    (left_blind_id, left_frame) in active_by_window_frame
                    or (right_blind_id, right_frame) in active_by_window_frame
                ):
                    reasons.append("active_truth_role_at_endpoint")
                self.record_exposure_candidate_pair(
                    family="lilocbench",
                    source_id=source_id,
                    sequence_id=source_id,
                    start_frame_id=left_frame,
                    end_frame_id=right_frame,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    reasons=reasons,
                    authorized=True,
                )

    @staticmethod
    def _route_index(route: Mapping[str, Any]) -> tuple[
        dict[tuple[str, str, int], Mapping[str, Any]],
        list[tuple[str, str, Mapping[str, Any]]],
    ]:
        index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
        sequences: list[tuple[str, str, Mapping[str, Any]]] = []
        for source in route["sources"]:
            source_id = str(source["source_id"])
            for sequence in source["sequences"]:
                sequence_id = str(sequence["sequence_id"])
                sequences.append((source_id, sequence_id, sequence))
                for frame in sequence["route_predictions"]:
                    key = (source_id, sequence_id, int(frame["frame_id"]))
                    if key in index:
                        raise ContractError(f"duplicate route frame {key}")
                    index[key] = frame
        return index, sequences

    def process_crowdbot(
        self,
        *,
        dataset_group: str,
        truth_id: str,
        fusion_id: str,
        route_id: str,
        authorize_exposure: bool,
        superseded_nonvisual: bool,
    ) -> None:
        truth = self.loader.load(truth_id)
        route = self.loader.load(route_id)
        route_index, route_sequences = self._route_index(route)
        for source_id, sequence_id, sequence in route_sequences:
            for frame in sequence["route_predictions"]:
                state = str(frame["status"])
                self.add_frame(
                    family="crowdbot_qolo",
                    source_id=source_id,
                    sequence_id=sequence_id,
                    frame_id=int(frame["frame_id"]),
                    timestamp_ns=int(frame["source_capture_timestamp_ns"]),
                    route_state=state,
                )
        del route_sequences
        del route

        fusion = self.loader.load(fusion_id)
        proposal_lookup = {
            str(row["event_id"]): row for row in fusion.get("event_proposals", [])
        }
        truth_events: list[tuple[str, Mapping[str, Any]]] = []
        truth_events.extend(("accepted", row) for row in truth["accepted_events"])
        truth_events.extend(
            ("quarantined", row) for row in truth["quarantined_events"]
        )
        if not superseded_nonvisual and set(proposal_lookup) != {
            str(row["event_id"]) for _, row in truth_events
        }:
            raise ContractError("CrowdBot replacement event inventory drifted")

        frames_by_sequence: dict[
            tuple[str, str], list[Mapping[str, Any]]
        ] = defaultdict(list)
        persons_by_sequence_frame: dict[
            tuple[str, str, int], set[str]
        ] = defaultdict(set)
        roles_by_person: dict[
            tuple[str, str, str], dict[int, str]
        ] = defaultdict(dict)
        timestamp_by_sequence_frame: dict[tuple[str, str, int], int] = {}
        for frame in fusion["frames"]:
            source_id = str(frame["source_id"])
            sequence_id = str(frame["sequence_id"])
            frame_id = int(frame["frame_id"])
            route_frame = route_index.get((source_id, sequence_id, frame_id))
            if route_frame is None:
                raise ContractError("CrowdBot fusion frame missing causal route")
            timestamp_ns = int(frame["source_capture_timestamp_ns"])
            if int(route_frame["source_capture_timestamp_ns"]) != timestamp_ns:
                raise ContractError("CrowdBot fusion/route timestamp mismatch")
            frames_by_sequence[(source_id, sequence_id)].append(frame)
            timestamp_by_sequence_frame[(source_id, sequence_id, frame_id)] = (
                timestamp_ns
            )
            for person in frame.get("persons", []):
                person_id = str(person.get("person_id", ""))
                if not person_id:
                    continue
                persons_by_sequence_frame[
                    (source_id, sequence_id, frame_id)
                ].add(person_id)
                roles_by_person[(source_id, sequence_id, person_id)][frame_id] = str(
                    person.get("role")
                )

        for key, frames in frames_by_sequence.items():
            frames.sort(key=lambda row: int(row["frame_id"]))
            previous: Mapping[str, Any] | None = None
            for frame in frames:
                source_id, sequence_id = key
                frame_id = int(frame["frame_id"])
                timestamp_ns = int(frame["source_capture_timestamp_ns"])
                route_state = str(
                    route_index[(source_id, sequence_id, frame_id)]["status"]
                )
                active = any(
                    person.get("role") in ACTIVE_ROLES
                    for person in frame.get("persons", [])
                )
                if previous is not None:
                    previous_id = int(previous["frame_id"])
                    previous_ts = int(previous["source_capture_timestamp_ns"])
                    previous_route_state = str(
                        route_index[(source_id, sequence_id, previous_id)]["status"]
                    )
                    previous_active = any(
                        person.get("role") in ACTIVE_ROLES
                        for person in previous.get("persons", [])
                    )
                    reasons: list[str] = []
                    if frame_id != previous_id + 1:
                        reasons.append("frame_id_not_consecutive")
                    if timestamp_ns <= previous_ts:
                        reasons.append("timestamp_nonpositive")
                    elif timestamp_ns - previous_ts > 1_000_000_000:
                        reasons.append("timestamp_gap_exceeds_maximum")
                    if route_state != "known" or previous_route_state != "known":
                        reasons.append("causal_route_unknown_at_endpoint")
                    if (
                        frame.get("route_relevant_person_truth_complete") is not True
                        or previous.get("route_relevant_person_truth_complete")
                        is not True
                    ):
                        reasons.append(
                            "route_relevant_person_truth_incomplete_at_endpoint"
                        )
                    if active or previous_active:
                        reasons.append("active_truth_role_at_endpoint")
                    self.record_exposure_candidate_pair(
                        family="crowdbot_qolo",
                        source_id=source_id,
                        sequence_id=sequence_id,
                        start_frame_id=previous_id,
                        end_frame_id=frame_id,
                        start_ns=previous_ts,
                        end_ns=timestamp_ns,
                        reasons=reasons,
                        authorized=authorize_exposure,
                    )
                previous = frame

        for truth_status, raw in truth_events:
            source_id = str(raw["source_id"])
            sequence_id = str(raw["sequence_id"])
            event_id = str(raw["event_id"])
            onset = int(raw["onset_frame"])
            alertable = int(raw["alertable_start_frame"])
            clear = (
                int(raw["clear_frame"])
                if raw.get("clear_frame") is not None
                else None
            )
            end_frame = int(raw["window_end_frame"])
            route_status = {
                frame: str(
                    route_index[(source_id, sequence_id, frame)]["status"]
                )
                for frame in range(onset, min(end_frame, onset + 5000) + 1)
                if (source_id, sequence_id, frame) in route_index
            }
            proposal = proposal_lookup.get(event_id)
            if superseded_nonvisual:
                person_id = ""
                identity_onset_alertable = False
                identity_through_clear = False
                complete_active = False
                interval = None
                critical_reasons = (
                    ["not_critical_event"]
                    if not bool(raw["critical"])
                    else ["critical_interval_not_frozen_or_ambiguous"]
                )
                post_identity_ms = None
                lifecycle_identity_loss = True
            else:
                if proposal is None:
                    raise ContractError(f"missing CrowdBot proposal {event_id}")
                person_id = str(proposal["person_id"])
                identity_onset_alertable = all(
                    person_id
                    in persons_by_sequence_frame.get(
                        (source_id, sequence_id, frame), set()
                    )
                    for frame in range(onset, alertable + 1)
                )
                identity_through_clear = (
                    clear is not None
                    and all(
                        person_id
                        in persons_by_sequence_frame.get(
                            (source_id, sequence_id, frame), set()
                        )
                        for frame in range(alertable, clear + 1)
                    )
                )
                complete_active = bool(raw.get("identity_continuous")) and (
                    clear is not None
                )
                roles = roles_by_person.get(
                    (source_id, sequence_id, person_id), {}
                )
                contiguous_observation: list[int] = []
                frame = onset
                while person_id in persons_by_sequence_frame.get(
                    (source_id, sequence_id, frame), set()
                ):
                    contiguous_observation.append(frame)
                    frame += 1
                role_frames = [
                    frame
                    for frame in contiguous_observation
                    if roles.get(frame) == "route_intersecting"
                ]
                interval, critical_reasons = _critical_interval(
                    critical=bool(raw["critical"]),
                    role_frames=role_frames,
                    route_status=route_status,
                )
                lifecycle_identity_loss = not complete_active
                post_identity_ms = None
                if clear is not None:
                    clear_ts = timestamp_by_sequence_frame.get(
                        (source_id, sequence_id, clear)
                    )
                    if clear_ts is not None:
                        identity_timestamps: dict[int, int] = {}
                        frame = clear + 1
                        while person_id in persons_by_sequence_frame.get(
                            (source_id, sequence_id, frame), set()
                        ):
                            identity_timestamps[frame] = timestamp_by_sequence_frame[
                                (source_id, sequence_id, frame)
                            ]
                            frame += 1
                        post_identity_ms = _continuous_post_clear_identity_ms(
                            clear_frame=clear,
                            clear_timestamp_ns=clear_ts,
                            identity_frames=identity_timestamps,
                            maximum_gap_ns=int(
                                self.config[
                                    "candidate_blind_operational_definitions"
                                ]["regeneration"][
                                    "maximum_adjacent_identity_frame_gap_ns"
                                ]
                            ),
                        )

            route_onset_alertable = all(
                route_status.get(frame) == "known"
                for frame in range(onset, alertable + 1)
            )
            followup_ms = None
            if clear is not None:
                clear_ts = timestamp_by_sequence_frame.get(
                    (source_id, sequence_id, clear)
                )
                end_ts = int(raw["window_end_timestamp_ns"])
                if clear_ts is not None:
                    followup_ms = (end_ts - clear_ts) / 1_000_000.0
            raw_reasons = list(raw.get("quarantine_reasons", []))
            row = _base_event_row(
                dataset_group=dataset_group,
                provenance_family="crowdbot_qolo",
                source_id=source_id,
                sequence_id=sequence_id,
                event_id=event_id,
                truth_status=(
                    "superseded_quarantined_nonvisual_lidar_proposal"
                    if superseded_nonvisual
                    else truth_status
                ),
                critical=bool(raw["critical"]),
                onset_frame=onset,
                alertable_start_frame=alertable,
                clear_frame=clear,
                end_frame=end_frame,
                raw_exclusion_reasons=raw_reasons,
            )
            _classify_event(
                row,
                identity_onset_to_alertable=identity_onset_alertable,
                route_onset_to_alertable=route_onset_alertable,
                identity_through_clear=identity_through_clear,
                complete_active_episode=complete_active,
                critical_interval=interval,
                critical_reasons=critical_reasons,
                post_clear_followup_ms=followup_ms,
                post_clear_identity_followup_ms=post_identity_ms,
                superseded_nonvisual=superseded_nonvisual,
                incomplete_all_person_truth=superseded_nonvisual,
                lifecycle_identity_loss=lifecycle_identity_loss,
            )
            self.events.append(row)

    def exposure_pair_rows(self) -> list[dict[str, Any]]:
        return [
            self.exposure_pair_units[key]
            for key in sorted(self.exposure_pair_units)
        ]

    def merged_exposure_intervals(self) -> list[dict[str, Any]]:
        grouped: dict[
            tuple[str, str, str], list[tuple[int, int]]
        ] = defaultdict(list)
        for family, source_id, sequence_id, start_ns, end_ns in sorted(
            self.exposure_elements
        ):
            grouped[(family, source_id, sequence_id)].append((start_ns, end_ns))
        rows: list[dict[str, Any]] = []
        for (family, source_id, sequence_id), intervals in sorted(grouped.items()):
            current_start: int | None = None
            current_end: int | None = None
            for start_ns, end_ns in sorted(set(intervals)):
                if current_start is None:
                    current_start, current_end = start_ns, end_ns
                elif start_ns == current_end:
                    current_end = end_ns
                elif start_ns >= current_end:
                    rows.append(
                        {
                            "unit_id": (
                                f"{family}::{source_id}::{sequence_id}::"
                                f"{current_start}-{current_end}"
                            ),
                            "provenance_family": family,
                            "source_id": source_id,
                            "sequence_id": sequence_id,
                            "start_ns": current_start,
                            "end_ns": current_end,
                            "duration_ns": current_end - current_start,
                        }
                    )
                    current_start, current_end = start_ns, end_ns
                else:
                    raise ContractError("overlapping negative exposure intervals")
            if current_start is not None and current_end is not None:
                rows.append(
                    {
                        "unit_id": (
                            f"{family}::{source_id}::{sequence_id}::"
                            f"{current_start}-{current_end}"
                        ),
                        "provenance_family": family,
                        "source_id": source_id,
                        "sequence_id": sequence_id,
                        "start_ns": current_start,
                        "end_ns": current_end,
                        "duration_ns": current_end - current_start,
                    }
                )
        return rows

    def frame_ledger_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (
            family,
            source_id,
            sequence_id,
            frame_id,
        ), (timestamp_ns, route_state) in sorted(self.frame_units.items()):
            rows.append(
                {
                    "unit_id": (
                        f"{family}::{source_id}::{sequence_id}::{frame_id}"
                    ),
                    "provenance_family": family,
                    "source_id": source_id,
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "source_capture_timestamp_ns": timestamp_ns,
                    "route_validity_state": route_state,
                }
            )
        return rows

    def frame_mask_rows(self) -> list[dict[str, Any]]:
        grouped: dict[
            tuple[str, str, str], list[tuple[int, int, str]]
        ] = defaultdict(list)
        for (
            family,
            source_id,
            sequence_id,
            frame_id,
        ), (timestamp_ns, route_state) in self.frame_units.items():
            grouped[(family, source_id, sequence_id)].append(
                (frame_id, timestamp_ns, route_state)
            )
        rows: list[dict[str, Any]] = []
        for (family, source_id, sequence_id), frames in sorted(grouped.items()):
            frames.sort()
            encoded = json.dumps(
                frames, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
            route_counts = Counter(frame[2] for frame in frames)
            rows.append(
                {
                    "unit_id": f"{family}::{source_id}::{sequence_id}",
                    "provenance_family": family,
                    "source_id": source_id,
                    "sequence_id": sequence_id,
                    "frame_count": len(frames),
                    "first_frame_id": frames[0][0],
                    "last_frame_id": frames[-1][0],
                    "first_timestamp_ns": frames[0][1],
                    "last_timestamp_ns": frames[-1][1],
                    "capture_timestamp_complete": all(
                        isinstance(frame[1], int) for frame in frames
                    ),
                    "route_validity_complete": all(
                        frame[2] in ALLOWED_ROUTE_STATES for frame in frames
                    ),
                    "route_state_counts": dict(sorted(route_counts.items())),
                    "frame_mask_sha256": sha256_bytes(encoded),
                }
            )
        return rows

    def build(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.process_lilocbench()
        self.process_crowdbot(
            dataset_group="crowdbot_initial_superseded_proposals",
            truth_id="crowdbot_initial_truth_windows",
            fusion_id="crowdbot_initial_person_route_truth",
            route_id="crowdbot_initial_causal_route",
            authorize_exposure=False,
            superseded_nonvisual=True,
        )
        self.process_crowdbot(
            dataset_group="crowdbot_replacement",
            truth_id="crowdbot_replacement_truth_windows",
            fusion_id="crowdbot_replacement_person_route_truth",
            route_id="crowdbot_replacement_causal_route",
            authorize_exposure=True,
            superseded_nonvisual=False,
        )
        self.loader.require_all_read()
        self.events.sort(key=lambda row: row["unit_id"])
        expected = int(self.config["event_universe"]["expected_event_count"])
        if len(self.events) != expected:
            raise ContractError(
                f"event universe count mismatch: {len(self.events)} != {expected}"
            )
        unit_ids = [row["unit_id"] for row in self.events]
        if len(set(unit_ids)) != len(unit_ids):
            raise ContractError("event universe contains duplicate unit ids")
        exposure = self.merged_exposure_intervals()
        exposure_pairs = self.exposure_pair_rows()
        frame_ledger = self.frame_ledger_rows()
        frames = self.frame_mask_rows()
        mask = {
            "schema": MASK_SCHEMA,
            "protocol_schema": SCHEMA,
            "authority": "candidate_blind_preoutput_support_only",
            "candidate_outputs_read": [],
            "candidate_outputs_executed": False,
            "event_count": len(self.events),
            "metric_count_per_event": len(METRICS),
            "event_metric_classification_count": len(self.events) * len(METRICS),
            "events": self.events,
            "negative_exposure_pair_audit": exposure_pairs,
            "negative_exposure_intervals": exposure,
            "preoutput_frame_ledger": frame_ledger,
            "preoutput_frame_masks": frames,
        }
        receipt = aggregate_receipt(
            config=self.config,
            events=self.events,
            exposure_pairs=exposure_pairs,
            exposure=exposure,
            frame_ledger=frame_ledger,
            frame_masks=frames,
            manifest=self.loader.manifest_receipt,
            excluded_exposure_ns=self.excluded_exposure_ns,
        )
        return mask, receipt


def _contributions(
    units: Iterable[tuple[str, str, int]], *, total: int
) -> dict[str, Any]:
    by_family: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    for family, source_id, amount in units:
        by_family[family] += amount
        by_source[source_id] += amount

    def rows(counter: Counter[str]) -> list[dict[str, Any]]:
        return [
            {
                "id": key,
                "amount": value,
                "share": (value / total if total else None),
            }
            for key, value in sorted(counter.items())
        ]

    return {
        "by_provenance_family": rows(by_family),
        "by_source": rows(by_source),
    }


def _metric_result_template(
    *,
    support_status: str,
    denominator: int | float | None,
    denominator_unit: str,
    l1_readiness: str,
    contributions: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "support_status": support_status,
        "result_status": "not_tested",
        "numerator": None,
        "denominator": denominator,
        "denominator_unit": denominator_unit,
        "value": None,
        "ci_method": None,
        "ci_lower": None,
        "ci_upper": None,
        "bound_sufficient": False,
        "gate_result": "not_applicable",
        "l1_readiness": l1_readiness,
        "contributions": dict(contributions),
    }
    if extra:
        row.update(extra)
    return row


def aggregate_receipt(
    *,
    config: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    exposure_pairs: list[Mapping[str, Any]],
    exposure: list[Mapping[str, Any]],
    frame_ledger: list[Mapping[str, Any]],
    frame_masks: list[Mapping[str, Any]],
    manifest: list[Mapping[str, Any]],
    excluded_exposure_ns: Counter[str],
) -> dict[str, Any]:
    ready = config["l1_readiness"]["ready_status"]
    conditional = config["l1_readiness"]["conditional_status"]
    l0 = config["l1_readiness"]["not_ready_status"]
    eligible_events: dict[str, list[Mapping[str, Any]]] = {}
    for metric in ("event_recall", "critical_miss", "repeat", "clearance", "regeneration"):
        eligible_events[metric] = [
            row
            for row in events
            if row["metrics"][metric]["classification"] == "eligible"
        ]

    metric_rows: dict[str, Any] = {}
    event_floors = {
        metric: int(config["l1_readiness"][metric]["minimum_denominator"])
        for metric in (
            "event_recall",
            "critical_miss",
            "clearance",
            "regeneration",
        )
    }
    for metric, floor in event_floors.items():
        units = eligible_events[metric]
        count = len(units)
        metric_rows[metric] = _metric_result_template(
            support_status=(
                "not_evaluable"
                if count == 0
                else "evaluable_powered"
                if count >= floor
                else "evaluable_underpowered"
            ),
            denominator=count,
            denominator_unit="events",
            l1_readiness=ready if count >= floor else l0,
            contributions=_contributions(
                (
                    (
                        str(row["provenance_family"]),
                        str(row["source_id"]),
                        1,
                    )
                    for row in units
                ),
                total=count,
            ),
        )

    repeat_pool = eligible_events["repeat"]
    repeat_count = len(repeat_pool)
    metric_rows["repeat"] = _metric_result_template(
        support_status="not_evaluable",
        denominator=None,
        denominator_unit="candidate_delivered_events",
        l1_readiness=(
            conditional
            if repeat_count
            >= int(
                config["l1_readiness"]["repeat"][
                    "minimum_actual_candidate_delivered_denominator"
                ]
            )
            else l0
        ),
        contributions={
            "actual_candidate_denominator": {
                "by_provenance_family": [],
                "by_source": [],
            }
        },
        extra={
            "truth_observation_pool_count": repeat_count,
            "truth_observation_pool_contributions": _contributions(
                (
                    (
                        str(row["provenance_family"]),
                        str(row["source_id"]),
                        1,
                    )
                    for row in repeat_pool
                ),
                total=repeat_count,
            ),
            "minimum_actual_candidate_delivered_denominator": int(
                config["l1_readiness"]["repeat"][
                    "minimum_actual_candidate_delivered_denominator"
                ]
            ),
        },
    )

    exposure_ns = sum(int(row["duration_ns"]) for row in exposure)
    eligible_exposure_pairs = [
        row for row in exposure_pairs if row["classification"] == "eligible"
    ]
    ineligible_exposure_pairs = [
        row for row in exposure_pairs if row["classification"] == "ineligible"
    ]
    eligible_pair_duration_ns = sum(
        int(row["duration_ns"]) for row in eligible_exposure_pairs
    )
    if eligible_pair_duration_ns != exposure_ns:
        raise ContractError("eligible exposure pairs do not sum to merged exposure")
    primary_exposure_exclusions: Counter[str] = Counter(
        str(row["primary_exclusion_reason"])
        for row in ineligible_exposure_pairs
    )
    all_exposure_exclusions: Counter[str] = Counter()
    primary_exclusion_duration_ns: Counter[str] = Counter()
    for pair in ineligible_exposure_pairs:
        all_exposure_exclusions.update(
            str(reason) for reason in pair["exclusion_reasons"]
        )
        if pair["duration_ns"] is not None:
            primary_exclusion_duration_ns[
                str(pair["primary_exclusion_reason"])
            ] += int(pair["duration_ns"])
    minimum_exposure_ns = int(
        config["l1_readiness"]["false_alerts_per_minute"]["minimum_exposure_ns"]
    )
    metric_rows["false_alerts_per_minute"] = _metric_result_template(
        support_status=(
            "not_evaluable"
            if exposure_ns == 0
            else "evaluable_powered"
            if exposure_ns >= minimum_exposure_ns
            else "evaluable_underpowered"
        ),
        denominator=exposure_ns,
        denominator_unit="nanoseconds",
        l1_readiness=ready if exposure_ns >= minimum_exposure_ns else l0,
        contributions=_contributions(
            (
                (
                    str(row["provenance_family"]),
                    str(row["source_id"]),
                    int(row["duration_ns"]),
                )
                for row in exposure
            ),
            total=exposure_ns,
        ),
        extra={"denominator_minutes": exposure_ns / 60_000_000_000.0},
    )

    frame_count = sum(int(row["frame_count"]) for row in frame_masks)
    if frame_count != len(frame_ledger):
        raise ContractError("frame summary count does not match explicit frame ledger")
    capture_complete = frame_count > 0 and all(
        row["capture_timestamp_complete"] is True for row in frame_masks
    )
    route_complete = frame_count > 0 and all(
        row["route_validity_complete"] is True for row in frame_masks
    )
    frame_contributions = _contributions(
        (
            (
                str(row["provenance_family"]),
                str(row["source_id"]),
                int(row["frame_count"]),
            )
            for row in frame_masks
        ),
        total=frame_count,
    )
    metric_rows["evidence_age"] = _metric_result_template(
        support_status=(
            "evaluable_underpowered" if capture_complete else "not_evaluable"
        ),
        denominator=frame_count if capture_complete else 0,
        denominator_unit="preoutput_masked_frames",
        l1_readiness=conditional if capture_complete else l0,
        contributions=frame_contributions if capture_complete else {},
        extra={
            "runtime_completeness_requirement": (
                "candidate_consuming_timestamp_for_every_masked_frame"
            )
        },
    )
    metric_rows["unknown_or_stale_alert"] = _metric_result_template(
        support_status=(
            "evaluable_powered" if route_complete else "not_evaluable"
        ),
        denominator=frame_count if route_complete else 0,
        denominator_unit="preoutput_masked_frames",
        l1_readiness=ready if route_complete else l0,
        contributions=frame_contributions if route_complete else {},
    )

    reason_counts: dict[str, dict[str, int]] = {}
    censor_counts: dict[str, dict[str, int]] = {}
    raw_exclusion_reasons: Counter[str] = Counter()
    for event in events:
        raw_exclusion_reasons.update(
            str(reason) for reason in event.get("raw_exclusion_reasons", [])
        )
    for metric in METRICS:
        reasons: Counter[str] = Counter()
        censors: Counter[str] = Counter()
        for event in events:
            entry = event["metrics"][metric]
            reasons.update(entry.get("reasons", []))
            if entry.get("censor_state") is not None:
                censors[str(entry["censor_state"])] += 1
        reason_counts[metric] = dict(sorted(reasons.items()))
        censor_counts[metric] = dict(sorted(censors.items()))

    recall_eligible = eligible_events["event_recall"]
    recall_clear = [
        row
        for row in recall_eligible
        if row["observability"]["same_person_truth_terminal_clear_observed"] is True
    ]
    terminal_observability = {
        "numerator_same_person_terminal_clear_observed": len(recall_clear),
        "denominator_recall_eligible_events": len(recall_eligible),
        "value": (
            len(recall_clear) / len(recall_eligible)
            if recall_eligible
            else None
        ),
        "support_status": (
            "evaluable_underpowered" if recall_eligible else "not_evaluable"
        ),
    }
    raw_clear = sum(
        row["anchors"]["truth_terminal_clear_frame"] is not None for row in events
    )
    same_person_clear = len(eligible_events["clearance"])
    same_person_terminal_clear_observed = sum(
        row["observability"]["same_person_truth_terminal_clear_observed"] is True
        for row in events
    )

    l1_ready = [
        metric
        for metric in METRICS
        if metric_rows[metric]["l1_readiness"] == ready
    ]
    l1_conditional = [
        metric
        for metric in METRICS
        if metric_rows[metric]["l1_readiness"] == conditional
    ]
    l0_metrics = [
        metric
        for metric in METRICS
        if metric_rows[metric]["l1_readiness"] == l0
    ]

    return {
        "schema": RECEIPT_SCHEMA,
        "protocol_schema": SCHEMA,
        "evidence_authority": "candidate_blind_preoutput_support_only",
        "candidate_outputs_read": [],
        "candidate_outputs_executed": False,
        "dependency_manifest": manifest,
        "event_universe": {
            "total": len(events),
            "by_dataset_group": dict(
                sorted(Counter(str(row["dataset_group"]) for row in events).items())
            ),
            "event_metric_classification_count": len(events) * len(METRICS),
        },
        "metrics": metric_rows,
        "terminal_clear_observability": terminal_observability,
        "terminal_clear_inventory": {
            "raw_clear_marker_count": raw_clear,
            "same_person_terminal_clear_observed_count": (
                same_person_terminal_clear_observed
            ),
            "same_person_clearance_eligible_count": same_person_clear,
        },
        "raw_exclusion_reason_counts": dict(sorted(raw_exclusion_reasons.items())),
        "reason_counts": reason_counts,
        "censor_counts": censor_counts,
        "negative_exposure": {
            "candidate_pair_count": len(exposure_pairs),
            "eligible_pair_count": len(eligible_exposure_pairs),
            "ineligible_pair_count": len(ineligible_exposure_pairs),
            "eligible_interval_count": len(exposure),
            "eligible_duration_ns": exposure_ns,
            "eligible_duration_minutes": exposure_ns / 60_000_000_000.0,
            "candidate_pair_contributions": _contributions(
                (
                    (
                        str(row["provenance_family"]),
                        str(row["source_id"]),
                        1,
                    )
                    for row in exposure_pairs
                ),
                total=len(exposure_pairs),
            ),
            "ineligible_pair_contributions": _contributions(
                (
                    (
                        str(row["provenance_family"]),
                        str(row["source_id"]),
                        1,
                    )
                    for row in ineligible_exposure_pairs
                ),
                total=len(ineligible_exposure_pairs),
            ),
            "primary_exclusion_reason_counts": dict(
                sorted(primary_exposure_exclusions.items())
            ),
            "all_exclusion_reason_counts": dict(
                sorted(all_exposure_exclusions.items())
            ),
            "positive_duration_ns_by_primary_exclusion_reason": dict(
                sorted(primary_exclusion_duration_ns.items())
            ),
            "multi_reason_counts_are_non_additive": True,
            "otherwise_eligible_but_source_unauthorized_duration_ns_by_source": dict(
                sorted(excluded_exposure_ns.items())
            ),
            "source_unauthorized_pairs_never_enter_denominator": True,
        },
        "frame_support": {
            "sequence_mask_count": len(frame_masks),
            "frame_count": frame_count,
            "capture_timestamp_complete": capture_complete,
            "route_validity_complete": route_complete,
            "explicit_frame_ledger_sha256": sha256_bytes(
                json_bytes(frame_ledger)
            ),
        },
        "l1_routing": {
            "l1_exploratory_eligible_metrics": l1_ready,
            "l1_conditional_on_candidate_observation_metrics": l1_conditional,
            "l0_engineering_diagnostic_metrics": l0_metrics,
            "next_task": (
                "open_independent_c1_c3_single_run_exploratory_profile_task"
                if l1_ready
                else "stop_at_l0_and_fill_metric_specific_gap_matrix"
            ),
            "candidate_winner_allowed": False,
        },
        "authority": {
            "maximum": "L1_EXPLORATORY_METRIC_PROFILE",
            "candidate_execution_in_this_task": False,
            "candidate_winner": False,
            "selection": False,
            "confirmation": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "production": False,
        },
    }


def materialize(
    config: Any, *, repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = validate_config(config, repo=repo)
    materialization = Materialization(row, repo)
    mask, receipt = materialization.build()
    config_sha = sha256_file(config_path)
    mask_sha = sha256_bytes(json_bytes(mask))
    receipt = {
        **receipt,
        "protocol_binding": {
            "path": config_path.relative_to(repo).as_posix(),
            "sha256": config_sha,
        },
        "event_mask_binding": {
            "path": str(row["outputs"]["root"])
            + "/"
            + str(row["outputs"]["event_mask"]),
            "sha256": mask_sha,
        },
    }
    return mask, receipt
