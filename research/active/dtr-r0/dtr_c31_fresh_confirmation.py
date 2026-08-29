"""One-shot source-disjoint confirmation of the frozen C31 mechanism.

The prediction phase consumes only frozen truth-blind traces and sealed point
ledgers.  Labels, the cohort roster, evaluator identities, and outcomes are
opened only by the score phase after C30/C31 predictions and dropout ledgers
have been sealed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


DTR = Path(__file__).resolve().parent
REPO = DTR.parents[2]
if str(DTR) not in sys.path:
    sys.path.insert(0, str(DTR))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import dtr_c25_fresh_point_flow_confirmation as c25  # noqa: E402
import dtr_c27_persistent_point_support as c27  # noqa: E402
import dtr_c28_visibility_conditioned_point_memory as c28  # noqa: E402
import dtr_c30_consensus_motion_authority as c30  # noqa: E402
import dtr_c30_raw_residual_authority_trace as c30_trace  # noqa: E402
import dtr_c31_temporal_component_authority as c31  # noqa: E402
from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
)
from dtr_c2_fresh_global_obb_replay import _tracks  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from dtr_r5_dropout_canary import cases_from_tracks  # noqa: E402
from dtr_r7_occupancy_flow_canary import FlowLedger, load_flow_ledger  # noqa: E402
from jrdb_rgb_bridge import require, sha256_file, write_json  # noqa: E402
from jrdb_rgb_bridge import read_bag_pose_and_rgb  # noqa: E402
from skydiscover_c29 import evaluator as base  # noqa: E402


SCHEMA = "blindassist-dtr-c31-source-disjoint-confirmation-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c31-source-disjoint-predictions-v1"
LEDGER_SCHEMA = "blindassist-dtr-c31-source-disjoint-dropout-ledger-v1"
ADMISSION_SCHEMA = "blindassist-dtr-c31-source-disjoint-admission-v1"
FREEZE_SCHEMA = "blindassist-dtr-c31-source-disjoint-freeze-v1"
PREFLIGHT_SCHEMA = "blindassist-dtr-c31-source-disjoint-source-preflight-v1"
STATUS_MET = "DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_MET"
STATUS_NOT_MET = "DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET"
ROSTER_STATUS = "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY"
ARMS = ("M1_PDC_GLOBAL", "C30_CONSENSUS", "C31_SIGNED_TRANSPORT")


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sequences": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "timeline_duration_s": sum(float(row["timeline_duration_s"]) for row in rows),
        "known_non_contact_s": sum(float(row["known_non_contact_s"]) for row in rows),
        "bounded_contact_events": sum(int(row["bounded_contact_events"]) for row in rows),
        "unique_responsible_events": sum(int(row["unique_responsible_events"]) for row in rows),
    }


def admit(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.c1_result.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    require(
        source.get("schema") == "blindassist-dtr-c1-global-obb-cohort-admission-v1",
        "c1_result_schema_drift",
    )
    consumed = set(str(value) for value in source["selected_sequence_names"])
    consumed.update(str(value) for value in source["source"]["excluded_consumed_sequences"])
    consumed_rosters = []
    for path_value in (args.c10_roster, args.c11_roster, args.c25_roster):
        path = path_value.resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        require(value.get("schema") == ROSTER_SCHEMA, f"roster_schema:{path}")
        consumed.update(str(row["sequence"]) for row in value["selected_sequences"])
        consumed_rosters.append({"path": str(path), "sha256": sha256_file(path)})

    candidates = [
        row for row in source["sequence_scan"]
        if str(row["sequence"]) not in consumed
    ]
    excluded_structural: list[dict[str, Any]] = []
    if args.source_preflight is not None:
        preflight_path = args.source_preflight.resolve(strict=True)
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        require(
            preflight.get("schema") == PREFLIGHT_SCHEMA
            and preflight.get("truth_blind") is True,
            "source_preflight_schema",
        )
        by_sequence = {str(row["sequence"]): row for row in preflight["sequences"]}
        require(
            set(by_sequence) == {str(row["sequence"]) for row in candidates},
            "source_preflight_coverage",
        )
        selected = [
            row for row in candidates
            if bool(by_sequence[str(row["sequence"])]["evaluable"])
        ]
        excluded_structural = [
            {
                "sequence": sequence,
                "reason": str(row["reason"]),
            }
            for sequence, row in sorted(by_sequence.items())
            if not bool(row["evaluable"])
        ]
    else:
        selected = candidates
    selected.sort(key=lambda row: str(row["sequence"]))
    totals = _totals(selected)
    require(5 <= len(selected) <= 10, "fresh_sequence_count_outside_5_to_10")
    require(totals["bounded_contact_events"] > 0, "fresh_contact_events_missing")
    require(totals["known_non_contact_s"] >= 60.0, "fresh_non_contact_exposure_too_small")

    frozen_files = [
        Path(__file__).resolve(),
        Path(c31.__file__).resolve(),
        Path(c30.__file__).resolve(),
        Path(c25.__file__).resolve(),
        DTR / "dtr_c28_visibility_conditioned_point_memory.py",
        DTR / "dtr_c30_raw_residual_authority_trace.py",
        DTR / "dtr_c27_persistent_point_support.py",
        DTR / "dtr_m1_raw_point_direct_velocity.py",
        DTR / "dtr_m1_confident_direct_velocity.py",
        DTR / "dtr_r7_occupancy_flow_canary.py",
        DTR / "dtr_c4_detector_independent_global_risk.py",
    ]
    roster = {
        "schema": ROSTER_SCHEMA,
        "status": ROSTER_STATUS,
        "claim_ceiling": "ALGORITHM_FRESH_PUBLIC_REPLAY_NO_PRODUCT_OR_SAFETY_CLAIM",
        "dataset": "JRDB public train split",
        "contract": source["contract"],
        "selection_policy": {
            "algorithm_consumed_sequences_excluded": sorted(consumed),
            "ordering": "all remaining algorithm-unexposed JRDB train sequences, lexicographic",
            "candidate_sequences": len(candidates),
            "selected_all_structurally_evaluable_remaining": True,
            "source_preflight_applied": args.source_preflight is not None,
            "structurally_not_evaluable": excluded_structural,
        },
        "selected_sequences": [
            {
                "sequence": str(row["sequence"]),
                "first_frame": int(row["first_frame"]),
                "last_frame": int(row["last_frame"]),
                "frames": int(row["frames"]),
                "timeline_duration_s": float(row["timeline_duration_s"]),
                "known_non_contact_s": float(row["known_non_contact_s"]),
                "bounded_contact_events": int(row["bounded_contact_events"]),
                "unique_responsible_events": int(row["unique_responsible_events"]),
                "bounded_contact_event_details": list(row["events"]),
            }
            for row in selected
        ],
        "selected_totals": totals,
        "frozen_algorithm": {
            "arms": list(ARMS),
            "representation": (
                "sealed M1-PDC versus frozen C30 local consensus versus frozen "
                "C31 temporal velocity components with signed transport authority"
            ),
            "route_geometry_motion_bounds_and_lifecycle": "UNCHANGED",
            "gate": {
                "m1_pdc_contact_recall_not_lower_than_r7": True,
                "m1_pdc_false_segment_factor_vs_r7": 0.70,
                "m1_pdc_event_f1_higher_than_r7": True,
                "m1_pdc_dropout_recovery_not_lower_than_r7": True,
                "c31_contact_recall_all_events": True,
                "c31_contact_recall_not_lower_than_pdc": True,
                "c31_false_segments_not_higher_than_pdc": True,
                "c31_every_pdc_recalled_event_no_later_than_pdc": True,
                "stable_gain": (
                    "dropout recovery gain >= max(1, ceil(10% of induced trials)) "
                    "OR median first-alert lead gain >= 0.25 s"
                ),
            },
            "files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
                for path in frozen_files
            ],
        },
        "source_authority": {
            "labels_archive_name": Path(source["source"]["labels"]).name,
            "labels_sha256": source["source"]["labels_sha256"],
            "timestamps_archive_name": Path(source["source"]["timestamps"]).name,
            "timestamps_sha256": source["source"]["timestamps_sha256"],
            "c1_result": str(source_path),
            "c1_result_sha256": sha256_file(source_path),
            "consumed_rosters": consumed_rosters,
            "source_preflight": (
                None if args.source_preflight is None else str(args.source_preflight.resolve())
            ),
            "source_preflight_sha256": (
                None if args.source_preflight is None else sha256_file(args.source_preflight.resolve())
            ),
        },
        "forbidden": [
            "changing C31/C30/C25 thresholds, radii, decay, lifecycle, route, or gate after admission",
            "opening selected future OBB labels before all C30/C31 predictions are sealed",
            "adding or replacing a sequence after observing an outcome",
            "rerunning a failed arm with different parameters",
        ],
    }
    write_json(args.roster.resolve(), roster)
    freeze = {
        "schema": FREEZE_SCHEMA,
        "prediction_visible_only": True,
        "sequence_names": [str(row["sequence"]) for row in selected],
        "frozen_algorithm": roster["frozen_algorithm"],
        "roster_sha256": sha256_file(args.roster.resolve()),
        "excluded_from_prediction_view": [
            "bounded CONTACT event details",
            "future OBB labels",
            "evaluator target identity",
            "per-sequence outcome metrics",
        ],
    }
    write_json(args.freeze_manifest.resolve(), freeze)
    result = {
        "schema": ADMISSION_SCHEMA,
        "status": "DTR_C31_SOURCE_DISJOINT_COHORT_ADMITTED_METADATA_ONLY",
        "selected_sequences": [str(row["sequence"]) for row in selected],
        "selected_totals": totals,
        "artifacts": {
            "roster": str(args.roster.resolve()),
            "roster_sha256": sha256_file(args.roster.resolve()),
            "freeze_manifest": str(args.freeze_manifest.resolve()),
            "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        },
    }
    write_json(args.output.resolve(), result)
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve(strict=True)
    acquisition_path = args.acquisition.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "preflight_roster_schema")
    output = []
    with zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for roster_row in roster["selected_sequences"]:
            sequence = str(roster_row["sequence"])
            bag_path, bag_row = c25._acquired_bag(acquisition_path, sequence)
            timestamps = _load_timestamps(timestamps_zip, sequence)
            poses, _rgb, bag_authority = read_bag_pose_and_rgb(bag_path)
            reason = "EVALUABLE"
            evaluable = True
            failed_frame: int | None = None
            try:
                for frame in sorted(timestamps):
                    c25._causal_pose(poses, round(float(timestamps[frame]) * 1e9))
            except RuntimeError as error:
                evaluable = False
                failed_frame = int(frame)
                reason = str(error)
            output.append(
                {
                    "sequence": sequence,
                    "evaluable": evaluable,
                    "reason": reason,
                    "failed_frame": failed_frame,
                    "frames": len(timestamps),
                    "bag_sha256": bag_row["sha256"],
                    "bag_authority": bag_authority,
                }
            )
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "truth_blind": True,
        "check": "every public frame timestamp has a causal current-or-past native pose",
        "sequences": output,
        "evaluable_sequences": sum(bool(row["evaluable"]) for row in output),
        "not_evaluable_sequences": sum(not bool(row["evaluable"]) for row in output),
        "source": {
            "preliminary_roster_sha256": sha256_file(roster_path),
            "acquisition_sha256": sha256_file(acquisition_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
    }
    write_json(args.source_preflight_output.resolve(), result)
    return result


def baseline_score(args: argparse.Namespace) -> dict[str, Any]:
    delegated = argparse.Namespace(
        predictions=args.baseline_predictions,
        roster=args.roster,
        labels=args.labels,
        timestamps=args.timestamps,
        output=args.baseline_result,
    )
    result = c25.score(delegated)
    roster = json.loads(args.roster.resolve(strict=True).read_text(encoding="utf-8"))
    totals = roster["selected_totals"]
    result["question"] = "Fresh M1-PDC/C30/C31 source-disjoint reference scoring input"
    result["claim_limits"] = [
        f"{totals['sequences']} source-disjoint public JRDB sequences with "
        f"{totals['bounded_contact_events']} bounded CONTACT events.",
        "Predictions were sealed before evaluator-only future OBB truth and identity were opened.",
        "This intermediate baseline score does not decide C31 confirmation.",
        "No product, deployment, user-benefit, reliability, or safety claim follows.",
    ]
    write_json(args.baseline_result.resolve(), result)
    return result


def materialize_trace(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = args.freeze_manifest.resolve(strict=True)
    baseline_path = args.baseline_predictions.resolve(strict=True)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    require(freeze.get("schema") == FREEZE_SCHEMA, "trace_freeze_schema")
    require(
        baseline.get("schema") == c25.PREDICTION_SCHEMA
        and baseline.get("truth_blind") is True,
        "trace_baseline_schema",
    )
    names = {str(row["sequence"]) for row in baseline["sequences"]}
    require(names == {str(value) for value in freeze["sequence_names"]}, "trace_sequence_coverage")
    for path in (
        Path(__file__),
        Path(c28.__file__),
        Path(c30_trace.__file__),
    ):
        _verify_frozen(freeze, path)

    c28._SOURCE_BY_SEQUENCE = {
        str(row["sequence"]): row["sources"] for row in baseline["sequences"]
    }
    c28._TRACE_BY_SEQUENCE = {}
    ray_selection = c28._select_ray_backend(
        *c28._representative_ray_workload(baseline),
        args.ray_backend_receipt.resolve(),
    )
    c28._RAY_BACKEND = str(ray_selection["selected_backend"])
    point_selection = c27._select_backend(
        *c27._representative(baseline),
        args.point_lineage_backend_receipt.resolve(),
    )
    point_backend = str(point_selection["selected_backend"])
    status_counts: Counter[str] = Counter()
    for row in baseline["sequences"]:
        sequence = str(row["sequence"])
        sources = row["sources"]["ledgers"]
        pd = c27._load_arrays(
            Path(sources["M1_PD_GLOBAL"]["ledger"]),
            Path(sources["M1_PD_GLOBAL"]["manifest"]),
            {
                "frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m",
                "frame_ego_yaw_rad", "offsets", "forward_m", "left_m",
                "velocity_forward_mps", "velocity_left_mps", "source_point_count",
                "flow_support",
            },
        )
        pdc = c27._load_arrays(
            Path(sources["M1_PDC_GLOBAL"]["ledger"]),
            Path(sources["M1_PDC_GLOBAL"]["manifest"]),
            {
                "frames", "offsets", "forward_m", "left_m",
                "velocity_forward_mps", "velocity_left_mps", "confidence",
            },
        )
        _ledger, _sidecar, manifest = c28._build_memory(
            sequence=sequence,
            pd=pd,
            pdc=pdc,
            backend=point_backend,
        )
        status_counts.update(manifest["status_counts"])

    visibility_trace = {
        "schema": "blindassist-dtr-c29-truth-blind-authority-trace-v1",
        "truth_blind": True,
        "prediction_boundary": (
            "frozen C28 causal lineage/ray features from raw bags and sealed point ledgers; "
            "no labels, roster event details, evaluator identity, or future truth"
        ),
        "sequences": [
            c28._TRACE_BY_SEQUENCE[key] for key in sorted(c28._TRACE_BY_SEQUENCE)
        ],
        "source": {
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "c28_algorithm_sha256": sha256_file(Path(c28.__file__)),
            "ray_backend_receipt_sha256": sha256_file(args.ray_backend_receipt.resolve()),
            "point_lineage_backend_receipt_sha256": sha256_file(args.point_lineage_backend_receipt.resolve()),
        },
    }
    write_json(args.visibility_trace.resolve(), visibility_trace)
    raw_trace = c30_trace.run(
        argparse.Namespace(
            c29_trace=args.visibility_trace,
            c25_predictions=args.baseline_predictions,
            output=args.trace,
        )
    )
    result = {
        "schema": "blindassist-dtr-c31-source-disjoint-trace-receipt-v1",
        "truth_blind": True,
        "sequences": len(raw_trace["sequences"]),
        "frames": sum(len(row["frames"]) for row in raw_trace["sequences"]),
        "status_counts": dict(status_counts),
        "raw_pd_residual_cells": int(raw_trace["diagnostics"]["raw_pd_residual_cells"]),
        "source": {
            "freeze_manifest_sha256": sha256_file(freeze_path),
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "visibility_trace": str(args.visibility_trace.resolve()),
            "visibility_trace_sha256": sha256_file(args.visibility_trace.resolve()),
            "raw_trace": str(args.trace.resolve()),
            "raw_trace_sha256": sha256_file(args.trace.resolve()),
            "c28_algorithm_sha256": sha256_file(Path(c28.__file__)),
            "c30_trace_algorithm_sha256": sha256_file(Path(c30_trace.__file__)),
        },
        "backends": {
            "point_lineage": point_selection,
            "ray_visibility": ray_selection,
        },
    }
    write_json(args.trace_receipt.resolve(), result)
    return result


def _verify_frozen(roster: Mapping[str, Any], path: Path) -> None:
    expected = {
        str(Path(row["path"]).resolve()).lower(): str(row["sha256"])
        for row in roster["frozen_algorithm"]["files"]
    }
    resolved = str(path.resolve()).lower()
    require(resolved in expected, f"frozen_file_missing:{path.name}")
    require(sha256_file(path.resolve()) == expected[resolved], f"frozen_file_drift:{path.name}")


def _save_ledger(path: Path, ledger: FlowLedger, source: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(
            stream,
            frames=ledger.frames,
            offsets=ledger.offsets,
            forward_m=ledger.forward_m,
            left_m=ledger.left_m,
            velocity_forward_mps=ledger.velocity_forward_mps,
            velocity_left_mps=ledger.velocity_left_mps,
            component_id=ledger.component_id,
        )
    os.replace(partial, path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = {
        "schema": LEDGER_SCHEMA,
        "ledger": str(path.resolve()),
        "ledger_sha256": sha256_file(path.resolve()),
        "frames": int(len(ledger.frames)),
        "cells": int(ledger.offsets[-1]),
        "source": dict(source),
    }
    write_json(manifest_path, manifest)
    return {
        "ledger": str(path.resolve()),
        "ledger_sha256": manifest["ledger_sha256"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
    }


def _load_ledger(row: Mapping[str, Any]) -> FlowLedger:
    path = Path(row["ledger"]).resolve(strict=True)
    manifest_path = Path(row["manifest"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == LEDGER_SCHEMA, "confirmation_ledger_schema")
    require(sha256_file(path) == row["ledger_sha256"] == manifest["ledger_sha256"], "confirmation_ledger_hash")
    require(sha256_file(manifest_path) == row["manifest_sha256"], "confirmation_manifest_hash")
    with np.load(path, allow_pickle=False) as data:
        return FlowLedger(
            manifest=manifest,
            frames=data["frames"].copy(),
            offsets=data["offsets"].copy(),
            forward_m=data["forward_m"].copy(),
            left_m=data["left_m"].copy(),
            velocity_forward_mps=data["velocity_forward_mps"].copy(),
            velocity_left_mps=data["velocity_left_mps"].copy(),
            component_id=data["component_id"].copy(),
        )


def predict(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = args.freeze_manifest.resolve(strict=True)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == FREEZE_SCHEMA
        and freeze.get("prediction_visible_only") is True,
        "freeze_manifest_schema",
    )
    _verify_frozen(freeze, Path(c31.__file__))
    _verify_frozen(freeze, Path(c30.__file__))
    _verify_frozen(freeze, Path(__file__))

    trace_path = args.trace.resolve(strict=True)
    baseline_path = args.baseline_predictions.resolve(strict=True)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    require(
        trace.get("schema") == "blindassist-dtr-c30-truth-blind-raw-residual-authority-trace-v1"
        and trace.get("truth_blind") is True,
        "fresh_trace_contract",
    )
    require(baseline.get("schema") == c25.PREDICTION_SCHEMA and baseline.get("truth_blind") is True, "fresh_baseline_contract")
    trace_rows = {str(row["sequence"]): row for row in trace["sequences"]}
    baseline_rows = {str(row["sequence"]): row for row in baseline["sequences"]}
    roster_names = {str(value) for value in freeze["sequence_names"]}
    require(set(trace_rows) == set(baseline_rows) == roster_names, "fresh_prediction_coverage")

    selection = c31._select_matching_backend(args.backend_receipt.resolve())
    c31._MATCH_BACKEND = str(selection["selected_backend"])
    predictions = []
    totals: Counter[str] = Counter()
    for sequence in sorted(roster_names):
        trace_row = trace_rows[sequence]
        baseline_row = baseline_rows[sequence]
        frames = [int(row["frame"]) for row in trace_row["frames"]]
        timestamps = {int(row["frame"]): float(row["frame_time_s"]) for row in trace_row["frames"]}
        pdc_source = baseline_row["sources"]["ledgers"]["M1_PDC_GLOBAL"]
        pdc = c27._load_arrays(
            Path(pdc_source["ledger"]),
            Path(pdc_source["manifest"]),
            {"frames", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
        )
        require([int(value) for value in pdc["frames"]] == frames, f"pdc_frame_drift:{sequence}")
        authority = c31.TemporalComponentAuthority()
        selected_by_arm: dict[str, dict[int, list[Mapping[str, Any]]]] = {
            "C30_CONSENSUS": {},
            "C31_SIGNED_TRANSPORT": {},
        }
        previous_time: float | None = None
        sequence_totals: Counter[str] = Counter()
        for frame_row in trace_row["frames"]:
            frame = int(frame_row["frame"])
            frame_time = float(frame_row["frame_time_s"])
            delta_s = 0.0 if previous_time is None else frame_time - previous_time
            previous_time = frame_time
            rows = [dict(row) for row in frame_row["rows"]]
            indices, synthesized, diagnostics, base_indices = authority.choose(
                c31._public_rows(rows), delta_s=delta_s
            )
            selected_by_arm["C30_CONSENSUS"][frame] = [
                rows[index] for index in base_indices
                if rows[index]["status"] != "OBSERVED_PDC"
            ]
            selected = [
                rows[index] for index in indices
                if rows[index]["status"] != "OBSERVED_PDC"
            ]
            selected.extend(synthesized)
            selected_by_arm["C31_SIGNED_TRANSPORT"][frame] = selected
            sequence_totals.update(diagnostics)
        totals.update(sequence_totals)

        arms: dict[str, Any] = {"M1_PDC_GLOBAL": baseline_row["arms"]["M1_PDC_GLOBAL"]}
        ledgers: dict[str, Any] = {}
        for arm in ("C30_CONSENSUS", "C31_SIGNED_TRANSPORT"):
            extension = base._ledger_from_rows(frames, selected_by_arm[arm], None)
            combined = base._ledger_from_rows(frames, selected_by_arm[arm], pdc)
            arms[arm] = base._candidate_prediction(
                frames, timestamps, extension, baseline_row["arms"]["M1_PDC_GLOBAL"]
            )
            ledgers[arm] = _save_ledger(
                args.ledger_root.resolve() / sequence / f"{arm.lower()}.npz",
                combined,
                {
                    "trace_sha256": sha256_file(trace_path),
                    "baseline_predictions_sha256": sha256_file(baseline_path),
                    "frozen_c31_sha256": sha256_file(Path(c31.__file__)),
                },
            )
        predictions.append(
            {
                "sequence": sequence,
                "arms": arms,
                "dropout_ledgers": ledgers,
                "authority_counts": dict(sequence_totals),
            }
        )

    result = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": (
            "frozen truth-blind C30 trace, sealed M1-PDC ledgers, and frozen C30/C31 code only; "
            "no roster event details, labels, future OBB truth, evaluator identity, or prior score"
        ),
        "sequences": predictions,
        "authority_counts": dict(totals),
        "backend": selection,
        "source": {
            "trace": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "baseline_predictions": str(baseline_path),
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": sha256_file(freeze_path),
            "c31_algorithm_sha256": sha256_file(Path(c31.__file__)),
            "c30_algorithm_sha256": sha256_file(Path(c30.__file__)),
            "backend_receipt": str(args.backend_receipt.resolve()),
            "backend_receipt_sha256": sha256_file(args.backend_receipt.resolve()),
        },
    }
    write_json(args.predictions.resolve(), result)
    return result


def _aggregate_dropout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    misses = sum(int(row["track_only_window_misses"]) for row in rows)
    recovered = sum(
        int(row["m1_ct_recovered_track_only_window_misses"]) for row in rows
    )
    return {
        "trials": sum(int(row["trials"]) for row in rows),
        "track_only_window_misses": misses,
        "dropout_recovery": recovered,
        "dropout_recovery_rate": recovered / misses if misses else None,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve(strict=True)
    freeze_path = args.freeze_manifest.resolve(strict=True)
    predictions_path = args.predictions.resolve(strict=True)
    baseline_path = args.baseline_predictions.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema")
    require(freeze.get("schema") == FREEZE_SCHEMA, "freeze_manifest_schema")
    require(predictions.get("schema") == PREDICTION_SCHEMA and predictions.get("truth_blind") is True, "prediction_schema")
    require(baseline.get("schema") == c25.PREDICTION_SCHEMA and baseline.get("truth_blind") is True, "baseline_prediction_schema")
    require(freeze["roster_sha256"] == sha256_file(roster_path), "freeze_roster_hash")
    require(predictions["source"]["freeze_manifest_sha256"] == sha256_file(freeze_path), "prediction_freeze_hash")
    require(predictions["source"]["baseline_predictions_sha256"] == sha256_file(baseline_path), "prediction_baseline_hash")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash")
    require(roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path), "timestamps_hash")
    _verify_frozen(roster, Path(c31.__file__))
    _verify_frozen(roster, Path(c30.__file__))

    prediction_rows = {str(row["sequence"]): row for row in predictions["sequences"]}
    baseline_rows = {str(row["sequence"]): row for row in baseline["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(set(prediction_rows) == set(baseline_rows) == set(roster_rows), "score_coverage")
    arm_scores: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    dropout_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    per_sequence = []
    pdc_event_nonregression: list[bool] = []

    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in sorted(roster_rows):
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(frames=frames, timestamps=timestamps, boxes_by_frame=boxes)
            prediction_row = prediction_rows[sequence]
            baseline_row = baseline_rows[sequence]
            scores = {}
            for arm in ARMS:
                arm_prediction = prediction_row["arms"][arm]
                arm_score = c27.score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(frames, arm_prediction),
                )
                scores[arm] = arm_score
                arm_scores[arm].append(arm_score)

            pdc_events = {row["event_id"]: row for row in scores["M1_PDC_GLOBAL"]["event_rows"]}
            for event in scores["C31_SIGNED_TRANSPORT"]["event_rows"]:
                pdc_event = pdc_events[event["event_id"]]
                if pdc_event["recalled"]:
                    pdc_event_nonregression.append(
                        bool(event["recalled"])
                        and float(event["first_alert_lead_s"]) + 1e-9 >= float(pdc_event["first_alert_lead_s"])
                    )

            sources = baseline_row["sources"]["ledgers"]
            pd = c27._load_arrays(
                Path(sources["M1_PD_GLOBAL"]["ledger"]),
                Path(sources["M1_PD_GLOBAL"]["manifest"]),
                {"frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m", "frame_ego_yaw_rad"},
            )
            frame_poses = {frame: c27._pose(pd, index) for index, frame in enumerate(frames)}
            cases = cases_from_tracks(
                _tracks(boxes_by_frame=boxes, timestamps=timestamps, frame_poses=frame_poses)
            )
            cases_by_key = {(case.label_id, case.segment_index): case for case in cases}
            r7 = load_flow_ledger(
                Path(sources["R7_P_GLOBAL"]["ledger"]),
                Path(sources["R7_P_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            pd_ledger = c27.load_point_ledger(
                Path(sources["M1_PD_GLOBAL"]["ledger"]),
                Path(sources["M1_PD_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            pdc_ledger = c27.load_confident_ledger(
                Path(sources["M1_PDC_GLOBAL"]["ledger"]),
                Path(sources["M1_PDC_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            arm_ledgers = {
                "M1_PDC_GLOBAL": pdc_ledger,
                "C30_CONSENSUS": _load_ledger(prediction_row["dropout_ledgers"]["C30_CONSENSUS"]),
                "C31_SIGNED_TRANSPORT": _load_ledger(prediction_row["dropout_ledgers"]["C31_SIGNED_TRANSPORT"]),
            }
            sequence_dropout = {}
            for arm, ledger in arm_ledgers.items():
                stress = c27.dropout_stress(
                    roster_sequence=roster_rows[sequence],
                    cases=cases_by_key,
                    r7=r7,
                    m1=pd_ledger,
                    m1_ct=ledger,
                )
                dropout_rows[arm].append(stress)
                sequence_dropout[arm] = stress
            per_sequence.append(
                {"sequence": sequence, "scores": scores, "dropout_stress": sequence_dropout}
            )

    aggregate = {arm: c27.aggregate_scores(rows) for arm, rows in arm_scores.items()}
    dropout = {arm: _aggregate_dropout(rows) for arm, rows in dropout_rows.items()}
    pdc = aggregate["M1_PDC_GLOBAL"]
    candidate = aggregate["C31_SIGNED_TRANSPORT"]
    trials = int(dropout["C31_SIGNED_TRANSPORT"]["track_only_window_misses"])
    recovered_key = "dropout_recovery"
    pdc_recovered = int(dropout["M1_PDC_GLOBAL"][recovered_key])
    c31_recovered = int(dropout["C31_SIGNED_TRANSPORT"][recovered_key])
    required_dropout_gain = max(1, math.ceil(0.10 * trials))
    dropout_gain = c31_recovered - pdc_recovered
    lead_gain = float(candidate["median_first_alert_lead_s"]) - float(pdc["median_first_alert_lead_s"])
    checks = {
        "c31_contact_recall_all_events": int(candidate["bounded_contact_events_recalled"]) == int(candidate["bounded_contact_events"]),
        "c31_contact_recall_not_lower_than_pdc": int(candidate["bounded_contact_events_recalled"]) >= int(pdc["bounded_contact_events_recalled"]),
        "c31_false_segments_not_higher_than_pdc": int(candidate["false_alert_segments"]) <= int(pdc["false_alert_segments"]),
        "c31_every_pdc_recalled_event_no_later_than_pdc": all(pdc_event_nonregression),
        "stable_dropout_or_lead_gain": dropout_gain >= required_dropout_gain or lead_gain >= 0.25,
    }
    passed = all(checks.values())
    result = {
        "schema": SCHEMA,
        "terminal_status": STATUS_MET if passed else STATUS_NOT_MET,
        "truth_blind_predictions": True,
        "metrics": {
            arm: {
                "contact_recall": int(value["bounded_contact_events_recalled"]),
                "contact_events": int(value["bounded_contact_events"]),
                "false_alert_segments": int(value["false_alert_segments"]),
                "event_f1": float(value["bounded_contact_event_f1"]),
                "median_first_alert_lead_s": float(value["median_first_alert_lead_s"]),
                "dropout_recovery": int(dropout[arm][recovered_key]),
                "dropout_trials": int(dropout[arm]["track_only_window_misses"]),
            }
            for arm, value in aggregate.items()
        },
        "gain_vs_pdc": {
            "dropout_recovery": dropout_gain,
            "required_dropout_gain": required_dropout_gain,
            "median_first_alert_lead_s": lead_gain,
            "false_alert_segments": int(candidate["false_alert_segments"]) - int(pdc["false_alert_segments"]),
        },
        "gate": {"passed": passed, "checks": checks},
        "per_sequence": per_sequence,
        "source": {
            "sealed_predictions": str(predictions_path),
            "sealed_predictions_sha256": sha256_file(predictions_path),
            "baseline_predictions": str(baseline_path),
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "c31_algorithm_sha256": sha256_file(Path(c31.__file__)),
        },
        "evidence_boundary": [
            "One-shot source-disjoint replay on all seven remaining algorithm-unexposed JRDB train sequences.",
            "Evaluator identity and future OBB truth were opened only after C30/C31 predictions and dropout ledgers were sealed.",
            "Failure does not authorize parameter tuning or a rescue rerun.",
            "This is algorithm evidence, not Android, product, user-benefit, reliability, or safety evidence.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    root = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("admit", "preflight", "trace", "baseline-score", "predict", "score"))
    parser.add_argument("--c1-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-c1" / "global-obb-cohort-admission" / "result.json")
    parser.add_argument("--c10-roster", type=Path, default=DTR / "dtr_c10_fresh_confirmation_roster.json")
    parser.add_argument("--c11-roster", type=Path, default=DTR / "dtr_c11_fresh_confirmation_roster.json")
    parser.add_argument("--c25-roster", type=Path, default=DTR / "dtr_c25_fresh_confirmation_roster.json")
    parser.add_argument("--roster", type=Path, default=DTR / "dtr_c31_fresh_confirmation_roster.json")
    parser.add_argument("--source-preflight", type=Path)
    parser.add_argument("--source-preflight-output", type=Path, default=root / "source-preflight.json")
    parser.add_argument("--acquisition", type=Path, default=root / "acquisition.json")
    parser.add_argument("--freeze-manifest", type=Path, default=root / "freeze-manifest.json")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--baseline-predictions", type=Path, default=root / "baseline-predictions.json")
    parser.add_argument("--baseline-result", type=Path, default=root / "baseline-result.json")
    parser.add_argument("--trace", type=Path, default=root / "raw-residual-authority-trace.json")
    parser.add_argument("--visibility-trace", type=Path, default=root / "visibility-authority-trace.json")
    parser.add_argument("--trace-receipt", type=Path, default=root / "trace-receipt.json")
    parser.add_argument("--ray-backend-receipt", type=Path, default=root / "ray-backend.json")
    parser.add_argument("--point-lineage-backend-receipt", type=Path, default=root / "point-lineage-backend.json")
    parser.add_argument("--ledger-root", type=Path, default=root / "dropout-ledgers")
    parser.add_argument("--backend-receipt", type=Path, default=root / "component-backend.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.json")
    parser.add_argument("--output", type=Path, default=root / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "admit":
        result = admit(args)
        print(json.dumps({"status": result["status"], "selected": result["selected_sequences"], "totals": result["selected_totals"]}, sort_keys=True))
    elif args.mode == "preflight":
        result = preflight(args)
        print(json.dumps({"evaluable_sequences": result["evaluable_sequences"], "not_evaluable_sequences": result["not_evaluable_sequences"], "sequences": result["sequences"]}, sort_keys=True))
    elif args.mode == "trace":
        result = materialize_trace(args)
        print(json.dumps({"schema": result["schema"], "sequences": result["sequences"], "frames": result["frames"], "raw_pd_residual_cells": result["raw_pd_residual_cells"], "backends": result["backends"]}, sort_keys=True))
    elif args.mode == "baseline-score":
        result = baseline_score(args)
        print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, sort_keys=True))
    elif args.mode == "predict":
        result = predict(args)
        print(json.dumps({"schema": result["schema"], "sequences": len(result["sequences"]), "authority_counts": result["authority_counts"]}, sort_keys=True))
    else:
        result = score(args)
        print(json.dumps({"terminal_status": result["terminal_status"], "metrics": result["metrics"], "gain_vs_pdc": result["gain_vs_pdc"], "gate": result["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
