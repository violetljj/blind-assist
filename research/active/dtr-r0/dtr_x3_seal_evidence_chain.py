"""Validate and seal the amended X3 evidence chain in three explicit stages.

This helper is not part of the frozen X3 algorithm.  It validates completed
artifacts and binds the truth-blind empty-support amendment into the durable
evidence chain before prediction and scoring are accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    require,
    sha256_file,
    write_json,
)


FREEZE_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x3-full-materialization-v1"
LEDGER_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-predictions-v1"
RESULT_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-replay-v1"
AMENDMENT_SCHEMA = "blindassist-dtr-x3-empty-support-amendment-v1"
CHAIN_SCHEMA = "blindassist-dtr-x3-amended-evidence-chain-v1"
AMENDMENT_SEQUENCE = "huang-lane-2019-02-12_0"
AMENDMENT_FRAME = 4
SEQUENCE_COUNT = 6


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"x3_seal_json_object:{path}")
    return value


def _resolved_recorded_path(value: Any, *, label: str) -> Path:
    require(isinstance(value, str) and value, f"x3_seal_path:{label}")
    return Path(value).resolve(strict=True)


def _require_same_path(actual: Path, expected: Path, *, label: str) -> None:
    require(actual == expected, f"x3_seal_path_mismatch:{label}:{actual}:{expected}")


def _write_once_or_equal(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        require(_load_json(path) == value, f"x3_seal_existing_content_differs:{path}")
        return "ALREADY_IDENTICAL"
    write_json(path, value)
    require(_load_json(path) == value, f"x3_seal_write_verify:{path}")
    return "WRITTEN"


def _baseline_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _load_json(path)
    require(value.get("truth_blind") is True, "x3_seal_baseline_truth")
    sequences = value.get("sequences")
    require(isinstance(sequences, list), "x3_seal_baseline_sequences")
    rows = {str(row["sequence"]): row for row in sequences}
    require(len(rows) == SEQUENCE_COUNT == len(sequences), "x3_seal_baseline_sequence_count")
    return value, rows


def _validate_freeze(root: Path, baseline_path: Path) -> tuple[dict[str, Any], str]:
    path = (root / "freeze.json").resolve(strict=True)
    freeze = _load_json(path)
    require(freeze.get("schema") == FREEZE_SCHEMA, "x3_seal_freeze_schema")
    require(freeze.get("truth_blind_materialization") is True, "x3_seal_freeze_truth")
    algorithm_files = freeze.get("algorithm_files")
    require(
        isinstance(algorithm_files, list) and len(algorithm_files) == 4,
        "x3_seal_freeze_algorithm_files",
    )
    for row in algorithm_files:
        algorithm_path = _resolved_recorded_path(row.get("path"), label="freeze_algorithm")
        require(
            row.get("sha256") == sha256_file(algorithm_path),
            f"x3_seal_algorithm_hash:{algorithm_path}",
        )
    inputs = freeze.get("inputs")
    require(isinstance(inputs, dict), "x3_seal_freeze_inputs")
    require(
        inputs.get("baseline_predictions_sha256") == sha256_file(baseline_path),
        "x3_seal_baseline_hash",
    )
    return freeze, sha256_file(path)


def _validate_ledger_arrays(
    ledger_path: Path,
    baseline_ledger_path: Path,
    *,
    sequence: str,
) -> tuple[list[int], int]:
    with np.load(baseline_ledger_path, allow_pickle=False) as baseline:
        expected_frames = [int(value) for value in baseline["frames"]]
        expected_times = np.asarray(baseline["frame_time_s"], dtype=np.float64)
    with np.load(ledger_path, allow_pickle=False) as values:
        required = {
            "frames",
            "frame_time_s",
            "offsets",
            "forward_m",
            "left_m",
            "velocity_forward_mps",
            "velocity_left_mps",
            "component_id",
            "source_point_count",
            "flow_support",
        }
        require(required <= set(values.files), f"x3_seal_ledger_fields:{sequence}")
        frames = [int(value) for value in values["frames"]]
        require(frames == expected_frames, f"x3_seal_ledger_frames:{sequence}")
        frame_times = np.asarray(values["frame_time_s"], dtype=np.float64)
        require(
            frame_times.shape == expected_times.shape
            and np.array_equal(frame_times, expected_times),
            f"x3_seal_ledger_times:{sequence}",
        )
        offsets = np.asarray(values["offsets"], dtype=np.int64)
        require(len(offsets) == len(frames) + 1, f"x3_seal_offsets_length:{sequence}")
        require(offsets[0] == 0 and bool(np.all(np.diff(offsets) >= 0)), f"x3_seal_offsets:{sequence}")
        cells = int(offsets[-1])
        for name in (
            "forward_m",
            "left_m",
            "velocity_forward_mps",
            "velocity_left_mps",
            "component_id",
            "source_point_count",
            "flow_support",
        ):
            array = np.asarray(values[name])
            require(array.ndim == 1 and len(array) == cells, f"x3_seal_array_length:{sequence}:{name}")
            if np.issubdtype(array.dtype, np.number):
                require(bool(np.isfinite(array).all()), f"x3_seal_array_finite:{sequence}:{name}")
        require(bool(np.all(values["source_point_count"] >= 0)), f"x3_seal_source_count:{sequence}")
        require(bool(np.all(values["flow_support"] == 1.0)), f"x3_seal_flow_support:{sequence}")
    return expected_frames, cells


def _validate_amendment(
    root: Path,
    baseline_path: Path,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    freeze_sha256: str,
    amendment_script: Path,
) -> dict[str, Any]:
    sequence_dir = root / "sequences" / AMENDMENT_SEQUENCE
    receipt_path = (sequence_dir / "empty-support-amendment.json").resolve(strict=True)
    receipt = _load_json(receipt_path)
    require(receipt.get("schema") == AMENDMENT_SCHEMA, "x3_seal_amendment_schema")
    require(receipt.get("status") == "EMPTY_SUPPORT_FAIL_CLOSED", "x3_seal_amendment_status")
    require(receipt.get("truth_blind") is True, "x3_seal_amendment_truth")
    require(receipt.get("sequence") == AMENDMENT_SEQUENCE, "x3_seal_amendment_sequence")
    require(receipt.get("empty_frames") == [AMENDMENT_FRAME], "x3_seal_amendment_frames")
    require(receipt.get("empty_frame_count") == 1, "x3_seal_amendment_count")
    source = receipt.get("source")
    require(isinstance(source, dict), "x3_seal_amendment_source")
    require(source.get("freeze_sha256") == freeze_sha256, "x3_seal_amendment_freeze")
    require(
        source.get("baseline_predictions_sha256") == sha256_file(baseline_path),
        "x3_seal_amendment_baseline",
    )
    bag_path = _resolved_recorded_path(
        baseline_rows[AMENDMENT_SEQUENCE]["sources"]["bag"], label="amendment_bag"
    )
    require(source.get("bag_sha256") == sha256_file(bag_path), "x3_seal_amendment_bag")
    amendment_script = amendment_script.resolve(strict=True)
    amendment_script_sha256 = sha256_file(amendment_script)
    require(
        source.get("script_sha256") == amendment_script_sha256,
        "x3_seal_amendment_script",
    )

    checkpoint_path = (
        sequence_dir / "checkpoints" / f"{AMENDMENT_FRAME:06d}.npz"
    ).resolve(strict=True)
    with np.load(checkpoint_path, allow_pickle=False) as values:
        require(int(values["output_frame"][0]) == AMENDMENT_FRAME, "x3_seal_amendment_frame")
        for name in (
            "forward_m",
            "left_m",
            "velocity_forward_mps",
            "velocity_left_mps",
            "source_point_count",
        ):
            require(len(values[name]) == 0, f"x3_seal_amendment_not_empty:{name}")
        optimization_s = float(values["optimization_s"][0])
        delay_s = float(values["delay_s"][0])
        require(optimization_s == 0.0, "x3_seal_amendment_optimization")
        require(math.isfinite(delay_s) and delay_s > 0.0, "x3_seal_amendment_delay")

    return {
        "schema": AMENDMENT_SCHEMA,
        "sequence": AMENDMENT_SEQUENCE,
        "semantics": receipt.get("semantics"),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "script": str(amendment_script),
        "script_sha256": amendment_script_sha256,
        "checkpoints": [
            {
                "output_frame": AMENDMENT_FRAME,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            }
        ],
    }


def _validate_materialization(
    *,
    root: Path,
    baseline_path: Path,
    amendment_script: Path,
    seal_amendment: bool,
) -> dict[str, Any]:
    baseline_path = baseline_path.resolve(strict=True)
    _baseline, baseline_rows = _baseline_rows(baseline_path)
    _freeze, freeze_sha256 = _validate_freeze(root, baseline_path)
    materialization_path = (root / "materialization.json").resolve(strict=True)
    materialization_sha256_before = sha256_file(materialization_path)
    materialization = _load_json(materialization_path)
    require(materialization.get("schema") == MATERIALIZATION_SCHEMA, "x3_seal_materialization_schema")
    require(materialization.get("status") == "COMPLETE", "x3_seal_materialization_status")
    require(materialization.get("truth_blind") is True, "x3_seal_materialization_truth")
    require(materialization.get("sequences") == SEQUENCE_COUNT, "x3_seal_materialization_count")
    require(materialization.get("freeze_sha256") == freeze_sha256, "x3_seal_materialization_freeze")
    _require_same_path(
        _resolved_recorded_path(materialization.get("freeze"), label="materialization_freeze"),
        (root / "freeze.json").resolve(strict=True),
        label="materialization_freeze",
    )

    sequence_rows = materialization.get("sequence_manifests")
    require(isinstance(sequence_rows, list), "x3_seal_sequence_manifests")
    rows_by_sequence = {str(row["sequence"]): row for row in sequence_rows}
    require(
        len(rows_by_sequence) == SEQUENCE_COUNT == len(sequence_rows)
        and set(rows_by_sequence) == set(baseline_rows),
        "x3_seal_sequence_manifest_coverage",
    )
    sealed_sequences = []
    optimized_frames = 0
    for sequence in sorted(baseline_rows):
        sequence_dir = (root / "sequences" / sequence).resolve(strict=True)
        ledger_path = (sequence_dir / "lag-floxel.npz").resolve(strict=True)
        manifest_path = (sequence_dir / "lag-floxel.json").resolve(strict=True)
        sequence_row = rows_by_sequence[sequence]
        _require_same_path(
            _resolved_recorded_path(sequence_row.get("manifest"), label=f"manifest:{sequence}"),
            manifest_path,
            label=f"manifest:{sequence}",
        )
        manifest_sha256 = sha256_file(manifest_path)
        require(
            sequence_row.get("manifest_sha256") == manifest_sha256,
            f"x3_seal_materialization_manifest_hash:{sequence}",
        )
        manifest = _load_json(manifest_path)
        require(manifest.get("schema") == LEDGER_SCHEMA, f"x3_seal_manifest_schema:{sequence}")
        require(manifest.get("truth_blind") is True, f"x3_seal_manifest_truth:{sequence}")
        require(manifest.get("oracle") is False, f"x3_seal_manifest_oracle:{sequence}")
        require(manifest.get("sequence") == sequence, f"x3_seal_manifest_sequence:{sequence}")
        _require_same_path(
            _resolved_recorded_path(manifest.get("ledger"), label=f"ledger:{sequence}"),
            ledger_path,
            label=f"ledger:{sequence}",
        )
        ledger_sha256 = sha256_file(ledger_path)
        require(manifest.get("ledger_sha256") == ledger_sha256, f"x3_seal_ledger_hash:{sequence}")
        source = manifest.get("source")
        require(isinstance(source, dict), f"x3_seal_manifest_source:{sequence}")
        require(source.get("freeze_sha256") == freeze_sha256, f"x3_seal_manifest_freeze:{sequence}")
        baseline_row = baseline_rows[sequence]
        expected_bag = _resolved_recorded_path(baseline_row["sources"]["bag"], label=f"bag:{sequence}")
        expected_baseline_ledger = _resolved_recorded_path(
            baseline_row["sources"]["ledgers"]["M1_PD_GLOBAL"]["ledger"],
            label=f"baseline_ledger:{sequence}",
        )
        _require_same_path(
            _resolved_recorded_path(source.get("bag"), label=f"manifest_bag:{sequence}"),
            expected_bag,
            label=f"manifest_bag:{sequence}",
        )
        _require_same_path(
            _resolved_recorded_path(
                source.get("baseline_ledger"), label=f"manifest_baseline_ledger:{sequence}"
            ),
            expected_baseline_ledger,
            label=f"manifest_baseline_ledger:{sequence}",
        )
        require(source.get("bag_sha256") == sha256_file(expected_bag), f"x3_seal_bag_hash:{sequence}")
        require(
            source.get("baseline_ledger_sha256") == sha256_file(expected_baseline_ledger),
            f"x3_seal_baseline_ledger_hash:{sequence}",
        )
        frames, cells = _validate_ledger_arrays(
            ledger_path, expected_baseline_ledger, sequence=sequence
        )
        expected_optimized = max(0, len(frames) - 4)
        require(manifest.get("frames") == len(frames), f"x3_seal_manifest_frames:{sequence}")
        require(
            manifest.get("diagnostics", {}).get("optimized_frames") == expected_optimized,
            f"x3_seal_manifest_optimized:{sequence}",
        )
        require(
            manifest.get("diagnostics", {}).get("output_cells") == cells,
            f"x3_seal_manifest_cells:{sequence}",
        )
        optimized_frames += expected_optimized
        sealed_sequences.append(
            {
                "sequence": sequence,
                "ledger": str(ledger_path),
                "ledger_sha256": ledger_sha256,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
            }
        )
    require(
        materialization.get("optimized_frames") == optimized_frames,
        "x3_seal_materialization_optimized",
    )

    amendment = _validate_amendment(
        root, baseline_path, baseline_rows, freeze_sha256, amendment_script
    )
    amendments = [amendment]
    if "amendments" in materialization:
        require(materialization["amendments"] == amendments, "x3_seal_materialization_amendments_differ")
        write_status = "ALREADY_IDENTICAL"
    else:
        require(seal_amendment, "x3_seal_materialization_amendment_missing")
        require(not (root / "predictions.json").exists(), "x3_seal_predictions_already_exist")
        require(not (root / "result.json").exists(), "x3_seal_result_already_exists")
        require(not (root / "evidence-chain.json").exists(), "x3_seal_chain_already_exists")
        require(
            sha256_file(materialization_path) == materialization_sha256_before,
            "x3_seal_materialization_changed_during_validation",
        )
        materialization["amendments"] = amendments
        write_json(materialization_path, materialization)
        require(
            _load_json(materialization_path) == materialization,
            "x3_seal_materialization_write_verify",
        )
        write_status = "WRITTEN"

    return {
        "materialization_path": materialization_path,
        "materialization_sha256": sha256_file(materialization_path),
        "freeze_path": (root / "freeze.json").resolve(strict=True),
        "freeze_sha256": freeze_sha256,
        "amendment": amendment,
        "sequences": sealed_sequences,
        "materialization_write": write_status,
    }


def _validate_predictions(root: Path, materialization_evidence: Mapping[str, Any]) -> dict[str, Any]:
    predictions_path = (root / "predictions.json").resolve(strict=True)
    predictions = _load_json(predictions_path)
    require(predictions.get("schema") == PREDICTION_SCHEMA, "x3_seal_predictions_schema")
    require(predictions.get("truth_blind") is True, "x3_seal_predictions_truth")
    source = predictions.get("source")
    require(isinstance(source, dict), "x3_seal_predictions_source")
    require(
        source.get("materialization_sha256") == materialization_evidence["materialization_sha256"],
        "x3_seal_predictions_materialization",
    )
    require(
        source.get("freeze_sha256") == materialization_evidence["freeze_sha256"],
        "x3_seal_predictions_freeze",
    )
    expected = {row["sequence"]: row for row in materialization_evidence["sequences"]}
    prediction_rows = predictions.get("sequences")
    require(isinstance(prediction_rows, list), "x3_seal_prediction_sequences")
    rows = {str(row["sequence"]): row for row in prediction_rows}
    require(
        len(rows) == SEQUENCE_COUNT == len(prediction_rows) and set(rows) == set(expected),
        "x3_seal_prediction_coverage",
    )
    for sequence, expected_row in expected.items():
        candidate = rows[sequence].get("candidate_ledger")
        require(isinstance(candidate, dict), f"x3_seal_prediction_candidate:{sequence}")
        for kind in ("ledger", "manifest"):
            path = _resolved_recorded_path(candidate.get(kind), label=f"prediction_{kind}:{sequence}")
            expected_path = Path(expected_row[kind]).resolve(strict=True)
            _require_same_path(path, expected_path, label=f"prediction_{kind}:{sequence}")
            require(
                candidate.get(f"{kind}_sha256") == sha256_file(path),
                f"x3_seal_prediction_{kind}_hash:{sequence}",
            )
            require(
                candidate.get(f"{kind}_sha256") == expected_row[f"{kind}_sha256"],
                f"x3_seal_prediction_{kind}_materialization:{sequence}",
            )
    return {
        "predictions_path": predictions_path,
        "predictions_sha256": sha256_file(predictions_path),
    }


def seal_materialization(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    evidence = _validate_materialization(
        root=root,
        baseline_path=args.baseline_predictions,
        amendment_script=args.amendment_script,
        seal_amendment=True,
    )
    return {
        "status": "X3_MATERIALIZATION_AMENDMENT_SEALED",
        "materialization": str(evidence["materialization_path"]),
        "materialization_sha256": evidence["materialization_sha256"],
        "amendment_receipt_sha256": evidence["amendment"]["receipt_sha256"],
        "write": evidence["materialization_write"],
    }


def verify_predictions(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    materialization = _validate_materialization(
        root=root,
        baseline_path=args.baseline_predictions,
        amendment_script=args.amendment_script,
        seal_amendment=False,
    )
    predictions = _validate_predictions(root, materialization)
    return {
        "status": "X3_PREDICTIONS_CHAIN_VERIFIED",
        "materialization_sha256": materialization["materialization_sha256"],
        "predictions_sha256": predictions["predictions_sha256"],
        "sequences": len(materialization["sequences"]),
    }


def seal_result(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    materialization = _validate_materialization(
        root=root,
        baseline_path=args.baseline_predictions,
        amendment_script=args.amendment_script,
        seal_amendment=False,
    )
    predictions = _validate_predictions(root, materialization)
    result_path = (root / "result.json").resolve(strict=True)
    result = _load_json(result_path)
    require(result.get("schema") == RESULT_SCHEMA, "x3_seal_result_schema")
    require(
        result.get("status")
        in {"DTR_X3_FULL_LAG_FLOXEL_GATE_MET", "DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET"},
        "x3_seal_result_status",
    )
    result_source = result.get("source")
    require(isinstance(result_source, dict), "x3_seal_result_source")
    _require_same_path(
        _resolved_recorded_path(result_source.get("sealed_predictions"), label="result_predictions"),
        predictions["predictions_path"],
        label="result_predictions",
    )
    require(
        result_source.get("sealed_predictions_sha256") == predictions["predictions_sha256"],
        "x3_seal_result_predictions_hash",
    )
    require(
        result_source.get("freeze_sha256") == materialization["freeze_sha256"],
        "x3_seal_result_freeze",
    )

    chain = {
        "schema": CHAIN_SCHEMA,
        "status": "X3_AMENDED_EVIDENCE_CHAIN_SEALED",
        "truth_blind_predictions": True,
        "freeze": {
            "path": str(materialization["freeze_path"]),
            "sha256": materialization["freeze_sha256"],
        },
        "materialization": {
            "path": str(materialization["materialization_path"]),
            "sha256": materialization["materialization_sha256"],
        },
        "amendments": [materialization["amendment"]],
        "predictions": {
            "path": str(predictions["predictions_path"]),
            "sha256": predictions["predictions_sha256"],
        },
        "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "sequence_artifacts": materialization["sequences"],
        "sealer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "boundary": (
            "The empty-support amendment is a truth-blind fail-closed protocol correction "
            "outside the frozen X3 algorithm and is explicitly hash-bound here."
        ),
    }
    chain_path = root / "evidence-chain.json"
    write_status = _write_once_or_equal(chain_path, chain)
    return {
        "status": chain["status"],
        "evidence_chain": str(chain_path.resolve(strict=True)),
        "evidence_chain_sha256": sha256_file(chain_path),
        "write": write_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("seal-materialization", "verify-predictions", "seal-result"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-c31"
        / "fresh-confirmation"
        / "baseline-predictions.json",
    )
    parser.add_argument(
        "--amendment-script",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_x3_empty_support_amendment.py"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "seal-materialization":
        value = seal_materialization(args)
    elif args.mode == "verify-predictions":
        value = verify_predictions(args)
    else:
        value = seal_result(args)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
