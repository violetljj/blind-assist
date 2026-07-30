from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ"
SOURCE_ID = "commons_iran_shiraz_city_tour_2021_5"
CANDIDATE_COMMIT = "039757b2da41c051373f8ee3189c4b06028f5295"
INPUT_MANIFEST_SHA256 = (
    "af0ab3c735d96737f451a6e64d1784681966345c7849131ad51bd46c9d7e6571"
)
TRUTH_LEDGER_SHA256 = (
    "b2865cbeeb955fab62f02123031fe0f29af0a48a18443cf4581e6572e267a26c"
)
TRUTH_FREEZE_RECEIPT_SHA256 = (
    "7ddd0e4d9cf968a7594c9e960b4f76e3b1c2380e5f4f2b13f6e585bbf84aacf0"
)
RANK2_PROTOCOL_SHA256 = (
    "fe5862afce85c6d4e0f90891d61293f0c482c176b1d70f702c7f1da0e75098d9"
)
SOURCE_ACTIVATION_SHA256 = (
    "5208305d0e2f1f02b3ad340b8a51f165f5d71291621eb25c820107b908b9a2e0"
)
FRAME_COUNT = 4891
FRAME_STEP_NS = 100_000_000
DELAY_TOLERANCE_NS = 250_000_000
MAX_CAPTURE_TIMESTAMP_NS = (FRAME_COUNT - 1) * FRAME_STEP_NS
REPO_ROOT = Path(__file__).resolve().parents[3]
RANK2_PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/research/dual-loop/"
    "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_PROTOCOL_2026-07-31.json"
)
PARENT_PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/research/dual-loop/"
    "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_PROTOCOL_2026-07-31.json"
)
SOURCE_ACTIVATION_PATH = (
    REPO_ROOT
    / "artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/"
    "rank2-shiraz/source_activation_receipt.json"
)
TRUTH_FINALIZER_PATH = (
    REPO_ROOT
    / "scripts/research/dual_loop_unseen_natural_event_r0/"
    "finalize_rank2_truth.py"
)


@dataclass(frozen=True)
class TruthItem:
    item_id: str
    should_alert: bool
    start_ns: int
    alertable_start_ns: int | None
    end_ns: int
    category: str
    region: str
    confidence: float


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_protocol_chain() -> tuple[Path, Path]:
    require_equal(
        sha256_file(RANK2_PROTOCOL_PATH),
        RANK2_PROTOCOL_SHA256,
        "rank-2 protocol hash",
    )
    protocol = read_json(RANK2_PROTOCOL_PATH)
    require_equal(protocol.get("protocol_id"), PROTOCOL_ID, "rank-2 protocol id")
    require_equal(
        sha256_file(PARENT_PROTOCOL_PATH),
        "4585a9ab5c3a1e351b6ff0a3ab5622f3439b560618ba73b449f4a00e86b3cc9f",
        "parent protocol file hash",
    )
    require_equal(
        protocol.get("parent_protocol_sha256"),
        "4585a9ab5c3a1e351b6ff0a3ab5622f3439b560618ba73b449f4a00e86b3cc9f",
        "parent protocol hash",
    )
    require_equal(
        protocol.get("status"),
        "TRUTH_FROZEN_ADEQUATE_BASELINE_ONLY_AUTHORIZED",
        "rank-2 protocol status",
    )
    frozen_truth = protocol.get("frozen_truth")
    if not isinstance(frozen_truth, dict):
        raise ValueError("rank-2 protocol: missing frozen truth")
    require_equal(
        frozen_truth.get("truth_freeze_receipt_sha256"),
        TRUTH_FREEZE_RECEIPT_SHA256,
        "rank-2 protocol truth receipt binding",
    )
    require_equal(
        frozen_truth.get("truth_finalizer_sha256"),
        sha256_file(TRUTH_FINALIZER_PATH),
        "rank-2 protocol truth finalizer binding",
    )
    activation = protocol.get("ordered_activation")
    if not isinstance(activation, dict):
        raise ValueError("rank-2 protocol: missing ordered activation")
    require_equal(
        activation.get("source_activation_receipt_sha256"),
        SOURCE_ACTIVATION_SHA256,
        "rank-2 source activation binding",
    )
    require_equal(
        sha256_file(SOURCE_ACTIVATION_PATH),
        SOURCE_ACTIVATION_SHA256,
        "rank-2 source activation hash",
    )
    source_activation = read_json(SOURCE_ACTIVATION_PATH)
    require_equal(
        source_activation.get("status"),
        "ACTIVATED_AFTER_PRIOR_TERMINAL_DISCLOSURE",
        "source activation status",
    )
    require_equal(
        source_activation.get("evidence_instance"),
        "RANK2_SHIRAZ",
        "source activation instance",
    )
    require_equal(
        source_activation.get("source_id"),
        SOURCE_ID,
        "source activation source",
    )
    return RANK2_PROTOCOL_PATH, SOURCE_ACTIVATION_PATH


def validate_build_identity(path: Path) -> dict[str, Any]:
    value = read_json(path)
    require_equal(
        value.get("schema_version"),
        "blindassist.dual_loop_unseen_rank2_build_identity.v1",
        "build identity schema",
    )
    require_equal(value.get("protocol_id"), PROTOCOL_ID, "build identity protocol")
    require_equal(value.get("status"), "COMPLETE", "build identity status")
    require_equal(
        value.get("candidate_commit"),
        CANDIDATE_COMMIT,
        "build identity candidate commit",
    )
    require_equal(
        value.get("runtime_path_diff_empty"),
        True,
        "build identity runtime diff",
    )
    require_equal(
        value.get("rank2_protocol_sha256"),
        RANK2_PROTOCOL_SHA256,
        "build identity protocol hash",
    )
    for key in ("app_apk_sha256", "test_apk_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"build identity: invalid {key}")
    return value


def validate_truth(
    truth_root: Path,
) -> tuple[list[TruthItem], Path, Path]:
    ledger = truth_root / "truth_ledger.jsonl"
    receipt_path = truth_root / "truth_freeze_receipt.json"
    receipt = read_json(receipt_path)
    require_equal(
        sha256_file(receipt_path),
        TRUTH_FREEZE_RECEIPT_SHA256,
        "truth freeze receipt hash",
    )
    require_equal(receipt.get("protocol_id"), PROTOCOL_ID, "truth protocol")
    require_equal(receipt.get("status"), "TRUTH_FROZEN_ADEQUATE", "truth status")
    require_equal(receipt.get("truth_adequacy"), True, "truth adequacy")
    require_equal(receipt.get("baseline_output_opened"), False, "truth baseline state")
    require_equal(receipt.get("candidate_output_opened"), False, "truth candidate state")
    require_equal(
        receipt.get("truth_ledger_sha256"),
        TRUTH_LEDGER_SHA256,
        "truth ledger receipt hash",
    )
    require_equal(
        receipt.get("implementation_sha256"),
        sha256_file(TRUTH_FINALIZER_PATH),
        "truth finalizer implementation hash",
    )
    require_equal(sha256_file(ledger), TRUTH_LEDGER_SHA256, "truth ledger hash")
    items: list[TruthItem] = []
    for row in read_jsonl(ledger):
        require_equal(row.get("source_id"), SOURCE_ID, "truth source")
        require_equal(
            row.get("outcome_access_state"),
            "BASELINE_AND_CANDIDATE_UNOPENED",
            "truth outcome access",
        )
        item = TruthItem(
            item_id=str(row["item_id"]),
            should_alert=bool(row["should_alert"]),
            start_ns=int(row["start_ns"]),
            alertable_start_ns=(
                None
                if row.get("alertable_start_ns") is None
                else int(row["alertable_start_ns"])
            ),
            end_ns=int(row["end_ns"]),
            category=str(row["category"]),
            region=str(row["region"]),
            confidence=float(row["truth_confidence"]),
        )
        if not math.isfinite(item.confidence) or not 0 <= item.confidence <= 1:
            raise ValueError(f"{item.item_id}: invalid confidence")
        if not item.region.strip():
            raise ValueError(f"{item.item_id}: empty region")
        if (
            item.start_ns < 0
            or item.end_ns > MAX_CAPTURE_TIMESTAMP_NS
            or item.end_ns < item.start_ns
        ):
            raise ValueError(f"{item.item_id}: inverted truth interval")
        if item.should_alert:
            if item.alertable_start_ns is None:
                raise ValueError(f"{item.item_id}: missing alertable start")
            if not item.start_ns <= item.alertable_start_ns <= item.end_ns:
                raise ValueError(f"{item.item_id}: invalid alertable start")
        elif item.alertable_start_ns is not None:
            raise ValueError(f"{item.item_id}: negative item has alertable start")
        items.append(item)
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("duplicate truth item id")
    if sum(item.should_alert for item in items) != 7:
        raise ValueError("unexpected positive truth count")
    if sum(not item.should_alert for item in items) != 6:
        raise ValueError("unexpected negative truth count")
    metric_intervals = sorted(
        (
            (
                item.alertable_start_ns
                if item.should_alert
                else item.start_ns
            ),
            item.end_ns,
            item.item_id,
        )
        for item in items
    )
    for previous, current in zip(metric_intervals, metric_intervals[1:]):
        if previous[1] >= current[0]:
            raise ValueError(
                "truth metric intervals overlap at closed endpoints: "
                f"{previous[2]} and {current[2]}"
            )
    return items, ledger, receipt_path


def validate_input(input_root: Path) -> tuple[Path, Path]:
    manifest = input_root / "manifest.jsonl"
    receipt_path = input_root / "input_receipt.json"
    receipt = read_json(receipt_path)
    require_equal(receipt.get("protocol_id"), PROTOCOL_ID, "input protocol")
    require_equal(receipt.get("status"), "COMPLETE", "input status")
    require_equal(receipt.get("source_id"), SOURCE_ID, "input source")
    require_equal(receipt.get("truth_read"), False, "input truth access")
    require_equal(receipt.get("baseline_output_read"), False, "input baseline access")
    require_equal(receipt.get("candidate_output_read"), False, "input candidate access")
    require_equal(receipt.get("frame_count"), FRAME_COUNT, "input frame count")
    require_equal(receipt.get("frame_step_ns"), FRAME_STEP_NS, "input frame step")
    require_equal(
        receipt.get("manifest_sha256"),
        INPUT_MANIFEST_SHA256,
        "input receipt manifest hash",
    )
    require_equal(sha256_file(manifest), INPUT_MANIFEST_SHA256, "input manifest hash")
    return manifest, receipt_path


def validate_frame_schedule(
    rows: list[dict[str, Any]],
    schema: str,
    authority: str,
) -> None:
    require_equal(len(rows), FRAME_COUNT, "trace row count")
    for index, row in enumerate(rows):
        require_equal(row.get("schema_version"), schema, f"trace schema frame {index}")
        require_equal(row.get("protocol_id"), PROTOCOL_ID, f"trace protocol frame {index}")
        require_equal(row.get("authority"), authority, f"trace authority frame {index}")
        require_equal(row.get("source_id"), SOURCE_ID, f"trace source frame {index}")
        require_equal(row.get("frame_id"), index, f"trace frame id {index}")
        require_equal(
            row.get("source_capture_timestamp_ns"),
            index * FRAME_STEP_NS,
            f"trace timestamp frame {index}",
        )


def validate_baseline(
    baseline_root: Path,
) -> tuple[list[dict[str, Any]], Path, Path]:
    trace = baseline_root / "trace.jsonl"
    receipt_path = baseline_root / "producer_receipt.json"
    receipt = read_json(receipt_path)
    require_equal(receipt.get("protocol_id"), PROTOCOL_ID, "baseline protocol")
    require_equal(receipt.get("status"), "COMPLETE", "baseline status")
    require_equal(
        receipt.get("authority"),
        "FROZEN_UNSEEN_BASELINE_ONLY",
        "baseline authority",
    )
    require_equal(receipt.get("truth_read"), False, "baseline truth access")
    require_equal(
        receipt.get("candidate_output_read"),
        False,
        "baseline candidate access",
    )
    require_equal(receipt.get("frame_count"), FRAME_COUNT, "baseline frame count")
    require_equal(
        receipt.get("input_manifest_sha256"),
        INPUT_MANIFEST_SHA256,
        "baseline input hash",
    )
    require_equal(receipt.get("trace_sha256"), sha256_file(trace), "baseline trace hash")
    for key in ("installed_app_apk_sha256", "installed_test_apk_sha256"):
        digest = receipt.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"baseline receipt: invalid {key}")
    rows = read_jsonl(trace)
    validate_frame_schedule(
        rows,
        "blindassist.dual_loop_unseen_rank2_baseline_trace.v1",
        "FROZEN_UNSEEN_BASELINE_ONLY",
    )
    trigger_count = sum(bool(row.get("feedback_triggered")) for row in rows)
    require_equal(
        receipt.get("feedback_trigger_count"),
        trigger_count,
        "baseline trigger count",
    )
    for index, row in enumerate(rows):
        detections = row.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"baseline frame {index}: missing detections")
        require_equal(
            row.get("detection_count"),
            len(detections),
            f"baseline detection count frame {index}",
        )
        if not isinstance(row.get("detector_metrics"), dict):
            raise ValueError(f"baseline frame {index}: missing detector metrics")
    return rows, trace, receipt_path


def triggers_in_interval(
    rows: Iterable[dict[str, Any]],
    start_ns: int,
    end_ns: int,
    field: str,
) -> list[int]:
    return [
        int(row["source_capture_timestamp_ns"])
        for row in rows
        if start_ns <= int(row["source_capture_timestamp_ns"]) <= end_ns
        and bool(row.get(field))
    ]


def item_observations(
    items: list[TruthItem],
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for item in items:
        start = (
            item.alertable_start_ns
            if item.should_alert
            else item.start_ns
        )
        assert start is not None
        result[item.item_id] = triggers_in_interval(rows, start, item.end_ns, field)
    return result


def atomic_publish_json_directory(output: Path, files: dict[str, dict[str, Any]]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for name, value in files.items():
            (temporary / name).write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        temporary.replace(output)
    except Exception:
        raise


def evaluate_baseline(
    baseline_root: Path,
    truth_root: Path,
    input_root: Path,
    build_identity_path: Path,
    output: Path,
) -> dict[str, Any]:
    rank2_protocol, source_activation = validate_protocol_chain()
    build_identity = validate_build_identity(build_identity_path)
    items, ledger, truth_receipt = validate_truth(truth_root)
    manifest, input_receipt = validate_input(input_root)
    rows, baseline_trace, baseline_receipt = validate_baseline(baseline_root)
    baseline_receipt_value = read_json(baseline_receipt)
    require_equal(
        baseline_receipt_value.get("installed_app_apk_sha256"),
        build_identity["app_apk_sha256"],
        "baseline installed app APK",
    )
    require_equal(
        baseline_receipt_value.get("installed_test_apk_sha256"),
        build_identity["test_apk_sha256"],
        "baseline installed test APK",
    )
    observations = item_observations(items, rows, "feedback_triggered")
    positive_hits = [
        item.item_id
        for item in items
        if item.should_alert and observations[item.item_id]
    ]
    alerted_negatives = [
        item.item_id
        for item in items
        if not item.should_alert and observations[item.item_id]
    ]
    adequate = bool(positive_hits) and bool(alerted_negatives)
    assessment = {
        "schema_version": "blindassist.dual_loop_unseen_rank2_baseline_assessment.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "BASELINE_ADEQUATE" if adequate else "BASELINE_NOT_EVALUABLE",
        "source_id": SOURCE_ID,
        "frame_count": FRAME_COUNT,
        "positive_event_count": sum(item.should_alert for item in items),
        "baseline_hit_positive_ids": positive_hits,
        "baseline_hit_positive_count": len(positive_hits),
        "negative_window_count": sum(not item.should_alert for item in items),
        "baseline_alerted_negative_ids": alerted_negatives,
        "baseline_alerted_negative_count": len(alerted_negatives),
        "minimum_baseline_hit_positive_events": 1,
        "minimum_baseline_alerted_negative_windows": 1,
        "candidate_authorized": adequate,
        "rank2_protocol_sha256": sha256_file(rank2_protocol),
        "source_activation_receipt_sha256": sha256_file(source_activation),
        "baseline_trace_sha256": sha256_file(baseline_trace),
        "baseline_receipt_sha256": sha256_file(baseline_receipt),
        "installed_app_apk_sha256": build_identity["app_apk_sha256"],
        "installed_test_apk_sha256": build_identity["test_apk_sha256"],
        "build_identity_sha256": sha256_file(build_identity_path),
        "truth_ledger_sha256": sha256_file(ledger),
        "truth_freeze_receipt_sha256": sha256_file(truth_receipt),
        "input_manifest_sha256": sha256_file(manifest),
        "input_receipt_sha256": sha256_file(input_receipt),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    files = {"baseline_assessment.json": assessment}
    if adequate:
        authorization = {
            "schema_version": (
                "blindassist.dual_loop_unseen_rank2_candidate_authorization.v1"
            ),
            "protocol_id": PROTOCOL_ID,
            "status": "AUTHORIZED",
            "source_id": SOURCE_ID,
            "baseline_adequacy": True,
            "candidate_output_opened": False,
            "candidate_commit": CANDIDATE_COMMIT,
            "rank2_protocol_sha256": RANK2_PROTOCOL_SHA256,
            "source_activation_receipt_sha256": SOURCE_ACTIVATION_SHA256,
            "input_manifest_sha256": INPUT_MANIFEST_SHA256,
            "truth_ledger_sha256": TRUTH_LEDGER_SHA256,
            "truth_freeze_receipt_sha256": sha256_file(truth_receipt),
            "baseline_trace_sha256": sha256_file(baseline_trace),
            "baseline_receipt_sha256": sha256_file(baseline_receipt),
            "installed_app_apk_sha256": build_identity["app_apk_sha256"],
            "installed_test_apk_sha256": build_identity["test_apk_sha256"],
            "build_identity_sha256": sha256_file(build_identity_path),
            "evaluator_implementation_sha256": sha256_file(Path(__file__)),
            "baseline_assessment_sha256": hashlib.sha256(
                (
                    json.dumps(
                        assessment,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest(),
            "next_allowed_action": "run fixed candidate-only replay",
        }
        files["candidate_authorization.json"] = authorization
    atomic_publish_json_directory(output, files)
    return assessment


def verify_baseline_authorization(
    baseline_root: Path,
    truth_root: Path,
    input_root: Path,
    build_identity_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    validate_protocol_chain()
    build_identity = validate_build_identity(build_identity_path)
    items, _, truth_receipt = validate_truth(truth_root)
    validate_input(input_root)
    baseline_rows, baseline_trace, baseline_receipt = validate_baseline(
        baseline_root
    )
    observations = item_observations(items, baseline_rows, "feedback_triggered")
    positive_ids = sorted(
        item.item_id
        for item in items
        if item.should_alert and observations[item.item_id]
    )
    negative_ids = sorted(
        item.item_id
        for item in items
        if not item.should_alert and observations[item.item_id]
    )
    if not positive_ids or not negative_ids:
        raise ValueError(
            "baseline adequacy is false; candidate must remain unopened"
        )

    assessment_path = authorization_path.with_name("baseline_assessment.json")
    assessment = read_json(assessment_path)
    require_equal(
        assessment.get("status"), "BASELINE_ADEQUATE", "baseline assessment status"
    )
    require_equal(
        assessment.get("candidate_authorized"),
        True,
        "baseline assessment authorization",
    )
    require_equal(
        assessment.get("baseline_hit_positive_ids"),
        positive_ids,
        "baseline assessment positive ids",
    )
    require_equal(
        assessment.get("baseline_hit_positive_count"),
        len(positive_ids),
        "baseline assessment positive count",
    )
    require_equal(
        assessment.get("baseline_alerted_negative_ids"),
        negative_ids,
        "baseline assessment negative ids",
    )
    require_equal(
        assessment.get("baseline_alerted_negative_count"),
        len(negative_ids),
        "baseline assessment negative count",
    )
    require_equal(
        assessment.get("baseline_trace_sha256"),
        sha256_file(baseline_trace),
        "baseline assessment trace hash",
    )
    require_equal(
        assessment.get("baseline_receipt_sha256"),
        sha256_file(baseline_receipt),
        "baseline assessment receipt hash",
    )
    require_equal(
        assessment.get("truth_freeze_receipt_sha256"),
        sha256_file(truth_receipt),
        "baseline assessment truth receipt hash",
    )
    require_equal(
        assessment.get("build_identity_sha256"),
        sha256_file(build_identity_path),
        "baseline assessment build identity hash",
    )
    require_equal(
        assessment.get("implementation_sha256"),
        sha256_file(Path(__file__)),
        "baseline assessment evaluator hash",
    )

    authorization = read_json(authorization_path)
    require_equal(
        authorization.get("schema_version"),
        "blindassist.dual_loop_unseen_rank2_candidate_authorization.v1",
        "authorization schema",
    )
    require_equal(authorization.get("status"), "AUTHORIZED", "authorization status")
    require_equal(
        authorization.get("baseline_adequacy"), True, "authorization adequacy"
    )
    require_equal(
        authorization.get("candidate_output_opened"),
        False,
        "authorization candidate state",
    )
    require_equal(
        authorization.get("baseline_assessment_sha256"),
        sha256_file(assessment_path),
        "authorization assessment hash",
    )
    require_equal(
        authorization.get("baseline_trace_sha256"),
        sha256_file(baseline_trace),
        "authorization baseline trace hash",
    )
    require_equal(
        authorization.get("baseline_receipt_sha256"),
        sha256_file(baseline_receipt),
        "authorization baseline receipt hash",
    )
    require_equal(
        authorization.get("truth_freeze_receipt_sha256"),
        sha256_file(truth_receipt),
        "authorization truth receipt hash",
    )
    require_equal(
        authorization.get("build_identity_sha256"),
        sha256_file(build_identity_path),
        "authorization build identity hash",
    )
    require_equal(
        authorization.get("evaluator_implementation_sha256"),
        sha256_file(Path(__file__)),
        "authorization evaluator hash",
    )
    require_equal(
        authorization.get("installed_app_apk_sha256"),
        build_identity["app_apk_sha256"],
        "authorization app APK hash",
    )
    require_equal(
        authorization.get("installed_test_apk_sha256"),
        build_identity["test_apk_sha256"],
        "authorization test APK hash",
    )
    return {
        "status": "CANDIDATE_AUTHORIZATION_VERIFIED",
        "baseline_hit_positive_ids": positive_ids,
        "baseline_alerted_negative_ids": negative_ids,
        "baseline_trace_sha256": sha256_file(baseline_trace),
        "authorization_sha256": sha256_file(authorization_path),
        "evaluator_implementation_sha256": sha256_file(Path(__file__)),
    }


def validate_candidate(
    candidate_root: Path,
    items: list[TruthItem],
    baseline_rows: list[dict[str, Any]],
    baseline_trace: Path,
    baseline_receipt_path: Path,
    truth_receipt_path: Path,
    build_identity_path: Path,
    authorization_path: Path,
) -> tuple[list[dict[str, Any]], Path, Path]:
    build_identity = validate_build_identity(build_identity_path)
    authorization = read_json(authorization_path)
    require_equal(
        authorization.get("schema_version"),
        "blindassist.dual_loop_unseen_rank2_candidate_authorization.v1",
        "authorization schema",
    )
    require_equal(authorization.get("protocol_id"), PROTOCOL_ID, "authorization protocol")
    require_equal(authorization.get("status"), "AUTHORIZED", "authorization status")
    require_equal(
        authorization.get("baseline_adequacy"), True, "authorization baseline gate"
    )
    require_equal(
        authorization.get("candidate_output_opened"),
        False,
        "authorization candidate state",
    )
    require_equal(
        authorization.get("candidate_commit"), CANDIDATE_COMMIT, "authorization commit"
    )
    require_equal(
        authorization.get("rank2_protocol_sha256"),
        RANK2_PROTOCOL_SHA256,
        "authorization protocol hash",
    )
    require_equal(
        authorization.get("source_activation_receipt_sha256"),
        SOURCE_ACTIVATION_SHA256,
        "authorization source activation hash",
    )
    require_equal(
        authorization.get("input_manifest_sha256"),
        INPUT_MANIFEST_SHA256,
        "authorization input hash",
    )
    require_equal(
        authorization.get("truth_ledger_sha256"),
        TRUTH_LEDGER_SHA256,
        "authorization truth hash",
    )
    require_equal(
        authorization.get("truth_freeze_receipt_sha256"),
        sha256_file(truth_receipt_path),
        "authorization truth receipt hash",
    )
    require_equal(
        authorization.get("baseline_trace_sha256"),
        sha256_file(baseline_trace),
        "authorization baseline trace hash",
    )
    require_equal(
        authorization.get("baseline_receipt_sha256"),
        sha256_file(baseline_receipt_path),
        "authorization baseline receipt hash",
    )
    require_equal(
        authorization.get("build_identity_sha256"),
        sha256_file(build_identity_path),
        "authorization build identity hash",
    )
    require_equal(
        authorization.get("evaluator_implementation_sha256"),
        sha256_file(Path(__file__)),
        "authorization evaluator implementation hash",
    )
    require_equal(
        authorization.get("installed_app_apk_sha256"),
        build_identity["app_apk_sha256"],
        "authorization app APK hash",
    )
    require_equal(
        authorization.get("installed_test_apk_sha256"),
        build_identity["test_apk_sha256"],
        "authorization test APK hash",
    )
    assessment_path = authorization_path.with_name("baseline_assessment.json")
    require_equal(
        authorization.get("baseline_assessment_sha256"),
        sha256_file(assessment_path),
        "authorization baseline assessment hash",
    )
    assessment = read_json(assessment_path)
    require_equal(
        assessment.get("schema_version"),
        "blindassist.dual_loop_unseen_rank2_baseline_assessment.v1",
        "baseline assessment schema",
    )
    require_equal(
        assessment.get("status"), "BASELINE_ADEQUATE", "baseline assessment status"
    )
    require_equal(
        assessment.get("candidate_authorized"),
        True,
        "baseline assessment authorization",
    )
    require_equal(
        assessment.get("baseline_trace_sha256"),
        sha256_file(baseline_trace),
        "baseline assessment trace hash",
    )
    require_equal(
        assessment.get("baseline_receipt_sha256"),
        sha256_file(baseline_receipt_path),
        "baseline assessment receipt hash",
    )
    require_equal(
        assessment.get("truth_freeze_receipt_sha256"),
        sha256_file(truth_receipt_path),
        "baseline assessment truth receipt hash",
    )
    require_equal(
        assessment.get("implementation_sha256"),
        sha256_file(Path(__file__)),
        "baseline assessment evaluator hash",
    )
    baseline_observations = item_observations(
        items, baseline_rows, "feedback_triggered"
    )
    positive_hits = sum(
        item.should_alert and bool(baseline_observations[item.item_id])
        for item in items
    )
    alerted_negatives = sum(
        not item.should_alert and bool(baseline_observations[item.item_id])
        for item in items
    )
    if positive_hits < 1 or alerted_negatives < 1:
        raise ValueError("candidate authorization violates baseline adequacy")
    require_equal(
        assessment.get("baseline_hit_positive_count"),
        positive_hits,
        "baseline assessment positive count",
    )
    require_equal(
        assessment.get("baseline_alerted_negative_count"),
        alerted_negatives,
        "baseline assessment negative count",
    )

    trace = candidate_root / "trace.jsonl"
    receipt_path = candidate_root / "producer_receipt.json"
    receipt = read_json(receipt_path)
    require_equal(receipt.get("protocol_id"), PROTOCOL_ID, "candidate protocol")
    require_equal(receipt.get("status"), "COMPLETE", "candidate status")
    require_equal(
        receipt.get("authority"),
        "FROZEN_UNSEEN_CANDIDATE_ONLY",
        "candidate authority",
    )
    require_equal(receipt.get("truth_read"), False, "candidate truth access")
    require_equal(receipt.get("baseline_trace_read"), True, "candidate baseline input")
    require_equal(receipt.get("frame_count"), FRAME_COUNT, "candidate frame count")
    require_equal(receipt.get("risk_mutation_count"), 0, "candidate risk mutation count")
    require_equal(
        receipt.get("baseline_trace_sha256"),
        sha256_file(baseline_trace),
        "candidate baseline trace hash",
    )
    require_equal(
        receipt.get("authorization_sha256"),
        sha256_file(authorization_path),
        "candidate authorization hash",
    )
    require_equal(
        receipt.get("installed_app_apk_sha256"),
        build_identity["app_apk_sha256"],
        "candidate installed app APK",
    )
    require_equal(
        receipt.get("installed_test_apk_sha256"),
        build_identity["test_apk_sha256"],
        "candidate installed test APK",
    )
    require_equal(receipt.get("trace_sha256"), sha256_file(trace), "candidate trace hash")
    rows = read_jsonl(trace)
    validate_frame_schedule(
        rows,
        "blindassist.dual_loop_unseen_rank2_candidate_trace.v1",
        "FROZEN_UNSEEN_CANDIDATE_ONLY",
    )
    trigger_count = sum(
        bool(row.get("candidate_feedback_triggered")) for row in rows
    )
    require_equal(
        receipt.get("feedback_trigger_count"),
        trigger_count,
        "candidate trigger count",
    )
    behavior_state_diverged = False
    event_mutation_allowed_count = 0
    for index, (baseline, candidate) in enumerate(zip(baseline_rows, rows, strict=True)):
        bindings = (
            ("image_sha256", "image_sha256"),
            ("detector_output_sha256", "detector_output_sha256"),
            ("raw_risk_sha256", "baseline_raw_risk_sha256"),
            ("stable_risk_sha256", "baseline_stable_risk_sha256"),
            ("feedback_triggered", "baseline_feedback_triggered"),
            ("feedback_reason", "baseline_feedback_reason"),
        )
        for baseline_key, candidate_key in bindings:
            require_equal(
                candidate.get(candidate_key),
                baseline.get(baseline_key),
                f"candidate baseline binding frame {index} field {candidate_key}",
            )
        require_equal(
            candidate.get("candidate_raw_risk_sha256"),
            baseline.get("raw_risk_sha256"),
            f"candidate raw risk frame {index}",
        )
        require_equal(
            candidate.get("candidate_stable_risk_sha256"),
            baseline.get("stable_risk_sha256"),
            f"candidate stable risk frame {index}",
        )
        dual_loop = candidate.get("dual_loop")
        if not isinstance(dual_loop, dict):
            raise ValueError(f"candidate frame {index}: missing dual-loop observation")
        if dual_loop.get("event_mutation_allowed") is not False:
            event_mutation_allowed_count += 1
        vetoed = (
            bool(baseline.get("feedback_triggered"))
            and not bool(candidate.get("candidate_feedback_triggered"))
            and candidate.get("candidate_feedback_reason") == "DUAL_LOOP_CONTRADICTED"
        )
        if vetoed:
            behavior_state_diverged = True
        if not behavior_state_diverged:
            require_equal(
                candidate.get("candidate_risk_event"),
                baseline.get("risk_event"),
                f"pre-veto risk event frame {index}",
            )
    require_equal(
        event_mutation_allowed_count,
        0,
        "second-loop event mutation permission rows",
    )
    require_equal(
        receipt.get("event_mutation_allowed_count"),
        0,
        "candidate event mutation permission count",
    )
    return rows, trace, receipt_path


def first_or_none(values: list[int]) -> int | None:
    return min(values) if values else None


def classify_terminal(
    guardrails: dict[str, bool],
    corrected_negative_windows: int,
) -> tuple[str, str]:
    if not all(guardrails.values()):
        return "FIRST_UNSEEN_SOURCE_GUARDRAIL_FAILED", "ACTIVE_R1_REJECTED"
    if corrected_negative_windows == 0:
        return "FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT", "DENSITY_SIGNAL_ONLY"
    return (
        "FIRST_UNSEEN_SOURCE_EVENT_SIGNAL",
        "SECOND_INDEPENDENT_SESSION_REQUIRED",
    )


def evaluate_candidate(
    baseline_root: Path,
    candidate_root: Path,
    truth_root: Path,
    input_root: Path,
    build_identity_path: Path,
    authorization_path: Path,
    output: Path,
) -> dict[str, Any]:
    rank2_protocol, source_activation = validate_protocol_chain()
    build_identity = validate_build_identity(build_identity_path)
    items, ledger, truth_receipt = validate_truth(truth_root)
    manifest, input_receipt = validate_input(input_root)
    baseline_rows, baseline_trace, baseline_receipt = validate_baseline(baseline_root)
    candidate_rows, candidate_trace, candidate_receipt = validate_candidate(
        candidate_root,
        items,
        baseline_rows,
        baseline_trace,
        baseline_receipt,
        truth_receipt,
        build_identity_path,
        authorization_path,
    )
    baseline_observations = item_observations(
        items, baseline_rows, "feedback_triggered"
    )
    candidate_observations = item_observations(
        items, candidate_rows, "candidate_feedback_triggered"
    )

    positive_results: list[dict[str, Any]] = []
    baseline_hits: set[str] = set()
    candidate_hits: set[str] = set()
    retained: set[str] = set()
    timely: set[str] = set()
    for item in (value for value in items if value.should_alert):
        baseline_first = first_or_none(baseline_observations[item.item_id])
        candidate_first = first_or_none(candidate_observations[item.item_id])
        if baseline_first is not None:
            baseline_hits.add(item.item_id)
        if candidate_first is not None:
            candidate_hits.add(item.item_id)
        if baseline_first is not None and candidate_first is not None:
            retained.add(item.item_id)
            delay_ns: int | None = candidate_first - baseline_first
            if delay_ns <= DELAY_TOLERANCE_NS:
                timely.add(item.item_id)
        else:
            delay_ns = None
        positive_results.append(
            {
                "item_id": item.item_id,
                "baseline_hit": baseline_first is not None,
                "candidate_hit": candidate_first is not None,
                "baseline_first_feedback_ns": baseline_first,
                "candidate_first_feedback_ns": candidate_first,
                "delay_ns": delay_ns if candidate_first is not None else "POSITIVE_INFINITY",
                "retained_baseline_hit": item.item_id in retained,
                "within_delay_tolerance": item.item_id in timely,
                "baseline_feedback_rows": len(baseline_observations[item.item_id]),
                "candidate_feedback_rows": len(candidate_observations[item.item_id]),
            }
        )

    negative_results: list[dict[str, Any]] = []
    pairing_counts = {
        "corrected": 0,
        "retained_false": 0,
        "induced_false": 0,
        "both_clear": 0,
    }
    for item in (value for value in items if not value.should_alert):
        baseline_false = bool(baseline_observations[item.item_id])
        candidate_false = bool(candidate_observations[item.item_id])
        if baseline_false and not candidate_false:
            pairing = "corrected"
        elif baseline_false and candidate_false:
            pairing = "retained_false"
        elif not baseline_false and candidate_false:
            pairing = "induced_false"
        else:
            pairing = "both_clear"
        pairing_counts[pairing] += 1
        negative_results.append(
            {
                "item_id": item.item_id,
                "category": item.category,
                "pairing": pairing,
                "baseline_false_alert": baseline_false,
                "candidate_false_alert": candidate_false,
                "baseline_feedback_rows": len(baseline_observations[item.item_id]),
                "candidate_feedback_rows": len(candidate_observations[item.item_id]),
            }
        )

    guardrails = {
        "absolute_positive_recall_non_decreasing": (
            len(candidate_hits) >= len(baseline_hits)
        ),
        "baseline_hit_retention_complete": retained == baseline_hits,
        "timely_baseline_hit_retention_complete": timely == baseline_hits,
        "induced_negative_windows_zero": pairing_counts["induced_false"] == 0,
        "semantic_risk_invariants_preserved": all(
            row["baseline_raw_risk_sha256"] == row["candidate_raw_risk_sha256"]
            and row["baseline_stable_risk_sha256"]
            == row["candidate_stable_risk_sha256"]
            for row in candidate_rows
        ),
        "second_loop_event_mutation_permission_zero": all(
            row["dual_loop"]["event_mutation_allowed"] is False
            for row in candidate_rows
        ),
    }
    guardrails_pass = all(guardrails.values())
    terminal, disposition = classify_terminal(
        guardrails,
        pairing_counts["corrected"],
    )

    result = {
        "schema_version": "blindassist.dual_loop_unseen_rank2_effect_result.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "stage": "DEVELOPMENT",
        "source_id": SOURCE_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "rank2_protocol_sha256": sha256_file(rank2_protocol),
        "source_activation_receipt_sha256": sha256_file(source_activation),
        "delivery_level": "SIMULATED_FEEDBACK_CONTROLLER_ACCEPTED_ON_DEVICE_REPLAY",
        "positive_event_count": sum(item.should_alert for item in items),
        "baseline_positive_hit_ids": sorted(baseline_hits),
        "candidate_positive_hit_ids": sorted(candidate_hits),
        "baseline_absolute_positive_recall": len(baseline_hits) / 7,
        "candidate_absolute_positive_recall": len(candidate_hits) / 7,
        "baseline_hit_retention": len(retained) / len(baseline_hits),
        "timely_baseline_hit_retention": len(timely) / len(baseline_hits),
        "delay_tolerance_ns": DELAY_TOLERANCE_NS,
        "positive_event_results": positive_results,
        "negative_window_count": sum(not item.should_alert for item in items),
        "negative_window_pairing": pairing_counts,
        "negative_window_results": negative_results,
        "guardrails": guardrails,
        "guardrails_pass": guardrails_pass,
        "baseline_total_feedback_rows": sum(
            bool(row.get("feedback_triggered")) for row in baseline_rows
        ),
        "candidate_total_feedback_rows": sum(
            bool(row.get("candidate_feedback_triggered")) for row in candidate_rows
        ),
        "scene_contradict_rows": sum(
            row.get("dual_loop", {}).get("correction_decision")
            == "CONTRADICT_APPROACH"
            for row in candidate_rows
        ),
        "vetoed_feedback_opportunity_rows": sum(
            row.get("candidate_feedback_reason") == "DUAL_LOOP_CONTRADICTED"
            for row in candidate_rows
        ),
        "terminal": terminal,
        "disposition": disposition,
        "claim_ceiling": (
            "SINGLE_NEW_SOURCE_MODEL_REVIEWED_EVENT_LEVEL_DEVELOPMENT_CANARY_ONLY"
        ),
        "next_allowed_action": (
            "disable active R1; retain shadow and mechanism evidence"
            if not guardrails_pass
            else (
                "close R1 with density-only signal; do not add complexity"
                if pairing_counts["corrected"] == 0
                else "run one second independent source without retuning R1"
            )
        ),
        "baseline_trace_sha256": sha256_file(baseline_trace),
        "baseline_receipt_sha256": sha256_file(baseline_receipt),
        "candidate_trace_sha256": sha256_file(candidate_trace),
        "candidate_receipt_sha256": sha256_file(candidate_receipt),
        "candidate_authorization_sha256": sha256_file(authorization_path),
        "build_identity_sha256": sha256_file(build_identity_path),
        "installed_app_apk_sha256": build_identity["app_apk_sha256"],
        "installed_test_apk_sha256": build_identity["test_apk_sha256"],
        "truth_ledger_sha256": sha256_file(ledger),
        "truth_freeze_receipt_sha256": sha256_file(truth_receipt),
        "input_manifest_sha256": sha256_file(manifest),
        "input_receipt_sha256": sha256_file(input_receipt),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    terminal_receipt = {
        "schema_version": "blindassist.dual_loop_unseen_rank2_terminal.v1",
        "protocol_id": PROTOCOL_ID,
        "status": terminal,
        "disposition": disposition,
        "guardrails_pass": guardrails_pass,
        "corrected_negative_windows": pairing_counts["corrected"],
        "induced_negative_windows": pairing_counts["induced_false"],
        "result_sha256": hashlib.sha256(
            (
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            ).encode("utf-8")
        ).hexdigest(),
        "candidate_commit": CANDIDATE_COMMIT,
        "claim_ceiling": result["claim_ceiling"],
        "next_allowed_action": result["next_allowed_action"],
    }
    atomic_publish_json_directory(
        output,
        {
            "effect_result.json": result,
            "terminal.json": terminal_receipt,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--baseline-root", type=Path, required=True)
    baseline.add_argument("--truth-root", type=Path, required=True)
    baseline.add_argument("--input-root", type=Path, required=True)
    baseline.add_argument("--build-identity", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-authorization")
    verify.add_argument("--baseline-root", type=Path, required=True)
    verify.add_argument("--truth-root", type=Path, required=True)
    verify.add_argument("--input-root", type=Path, required=True)
    verify.add_argument("--build-identity", type=Path, required=True)
    verify.add_argument("--authorization", type=Path, required=True)
    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--baseline-root", type=Path, required=True)
    candidate.add_argument("--candidate-root", type=Path, required=True)
    candidate.add_argument("--truth-root", type=Path, required=True)
    candidate.add_argument("--input-root", type=Path, required=True)
    candidate.add_argument("--build-identity", type=Path, required=True)
    candidate.add_argument("--authorization", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baseline":
        value = evaluate_baseline(
            args.baseline_root,
            args.truth_root,
            args.input_root,
            args.build_identity,
            args.output,
        )
    elif args.command == "verify-authorization":
        value = verify_baseline_authorization(
            args.baseline_root,
            args.truth_root,
            args.input_root,
            args.build_identity,
            args.authorization,
        )
    else:
        value = evaluate_candidate(
            args.baseline_root,
            args.candidate_root,
            args.truth_root,
            args.input_root,
            args.build_identity,
            args.authorization,
            args.output,
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
