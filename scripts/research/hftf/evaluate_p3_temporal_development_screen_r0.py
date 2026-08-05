#!/usr/bin/env python3
"""Development-only P3 temporal screen; deliberately not a holdout evaluator.

The runner owns prediction generation.  It verifies every input hash before it
loads either model, evaluates exactly the manifest's three validation parents,
and refuses to overwrite an output directory.  It contains no bootstrap,
p-value, sealed-bundle, or promotion logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dav2_temporal_392_student_p3_r0_1 import (  # noqa: E402
    DecoupledTemporalStateHead,
    STATES,
    TRANSITIONS,
    build_temporal_evidence,
)
from evaluate_dav2_model_variant_gate_r0 import (  # noqa: E402
    depth_metrics,
)
from evaluate_metric3d_clearance_field_a0 import clearance_field  # noqa: E402


SCHEMA = "blindassist_p3_temporal_development_screen_r0_result"
MANIFEST_SCHEMA = "blindassist_p3_temporal_development_complete_manifest_r0"
LEDGER_SCHEMA = "blindassist_p3_temporal_development_screen_r0_prediction_ledger"
SHA_CHARS = frozenset("0123456789ABCDEF")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _sha(value: Any, label: str) -> str:
    value = str(value).upper()
    if len(value) != 64 or set(value) - SHA_CHARS:
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _f1(truth: list[str], predicted: list[str]) -> float | None:
    if not truth:
        return None
    scores: list[float] = []
    for name in TRANSITIONS:
        tp = sum(t == name and p == name for t, p in zip(truth, predicted))
        fp = sum(t != name and p == name for t, p in zip(truth, predicted))
        fn = sum(t == name and p != name for t, p in zip(truth, predicted))
        scores.append((2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def _majority_transition_baseline(truth: list[str]) -> float | None:
    """Exact agreement of the fixed majority-class predictor, not a tuned model."""
    if not truth:
        return None
    majority = max(TRANSITIONS, key=lambda value: (truth.count(value), value))
    return truth.count(majority) / len(truth)


def _state_from_field(field: dict[str, Any], band: int) -> str:
    if field.get("status") != "VALID":
        return "UNKNOWN_GROUND"
    value = field["bands"][band]["clearance_m"]
    if value is None:
        return "UNKNOWN_GROUND"
    return "OCCUPIED" if float(value) <= 1.5 else "CLEAR"


def _field_clearances(field: dict[str, Any]) -> list[float | None]:
    if field.get("status") != "VALID":
        return [None, None, None]
    return [field["bands"][band]["clearance_m"] for band in range(3)]


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    raw: list[float] = []
    aligned: list[float] = []
    clearance_errors: list[float] = []
    false_clear = known = valid_to_unknown = valid_count = external_abstain = 0
    for row in rows:
        depth = row[arm]["depth"]
        raw.append(float(depth["metric_abs_rel_median"]))
        aligned.append(float(depth["scale_aligned_abs_rel_median"]))
        truth = row["truth"]
        predicted = row[arm]
        for band in range(3):
            t_state = truth["state"][band]
            p_state = predicted["state"][band]
            p_clearance = predicted["clearance_m"][band]
            t_clearance = truth["clearance_m"][band]
            abstain = bool(predicted["external_abstain"][band])
            external_abstain += int(abstain)
            final = "UNKNOWN_GROUND" if abstain else p_state
            if t_state != "UNKNOWN_GROUND":
                valid_count += 1
                valid_to_unknown += int(final == "UNKNOWN_GROUND")
                known += 1
                false_clear += int(t_state == "OCCUPIED" and final == "CLEAR")
            if _finite(t_clearance) and _finite(p_clearance):
                clearance_errors.append(abs(float(t_clearance) - float(p_clearance)))
    return {
        "raw_abs_rel_median": _median(raw),
        "scale_aligned_abs_rel_median": _median(aligned),
        "clearance_mae_m": _mean(clearance_errors),
        "external_abstention_rate": _rate(external_abstain, len(rows) * 3),
        "false_clear_rate": _rate(false_clear, known),
        "valid_to_unknown_rate": _rate(valid_to_unknown, valid_count),
    }


def _temporal_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    deltas: list[float] = []
    truth_transition: list[str] = []
    predicted_transition: list[str] = []
    head_delta_errors: list[float] = []
    head_truth: list[str] = []
    head_predicted: list[str] = []
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row["clip_id"])].append(row)
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda value: int(value["frame_index"]))
        for prior, current in zip(clip_rows, clip_rows[1:]):
            for band in range(3):
                t0, t1 = prior["truth"]["clearance_m"][band], current["truth"]["clearance_m"][band]
                p0, p1 = prior[arm]["clearance_m"][band], current[arm]["clearance_m"][band]
                if all(_finite(value) for value in (t0, t1, p0, p1)):
                    deltas.append(abs((float(p1) - float(p0)) - (float(t1) - float(t0))))
                t = f"{prior['truth']['state'][band]}_TO_{current['truth']['state'][band]}"
                p = f"{prior[arm]['state'][band]}_TO_{current[arm]['state'][band]}"
                truth_transition.append(t)
                predicted_transition.append(p)
                if arm == "p3":
                    prediction = current["p3"].get("head_from_previous")
                    if prediction is not None:
                        delta = prediction["clearance_delta_m"][band]
                        if all(_finite(value) for value in (t0, t1, delta)):
                            head_delta_errors.append(abs(float(delta) - (float(t1) - float(t0))))
                        head_truth.append(t)
                        head_predicted.append(prediction["transition"] [band])
    exact = _rate(sum(t == p for t, p in zip(truth_transition, predicted_transition)), len(truth_transition))
    result = {
        "clearance_delta_mae_m": _mean(deltas),
        "transition_exact_agreement": exact,
        "transition_macro_f1": _f1(truth_transition, predicted_transition),
        "transition_majority_exact_baseline": _majority_transition_baseline(truth_transition),
        "transition_pairs": len(truth_transition),
    }
    if arm == "p3":
        result.update({
            "head_clearance_delta_mae_m": _mean(head_delta_errors),
            "head_transition_exact_agreement": _rate(sum(t == p for t, p in zip(head_truth, head_predicted)), len(head_truth)),
            "head_transition_macro_f1": _f1(head_truth, head_predicted),
            "head_transition_pairs": len(head_truth),
        })
    return result


def _parentwise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_id"])].append(row)
    return {parent: {"a2": {**_summary(value, "a2"), **_temporal_summary(value, "a2")}, "p3": {**_summary(value, "p3"), **_temporal_summary(value, "p3")}} for parent, value in sorted(grouped.items())}


def decide(protocol: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any], parentwise: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    rules = protocol["development_screen"]["decision_rules"]
    # The route decision compares equivalent geometry-derived clearance deltas.
    # The learned head is reported separately below and is not substituted for
    # the depth-to-geometry operator.
    delta_a2 = baseline["clearance_delta_mae_m"]
    delta_p3 = candidate["clearance_delta_mae_m"]
    improvement = None if not (_finite(delta_a2) and _finite(delta_p3) and float(delta_a2) > 0.0) else 1.0 - float(delta_p3) / float(delta_a2)
    parent_improved = sum(
        _finite(row["a2"].get("clearance_delta_mae_m")) and _finite(row["p3"].get("clearance_delta_mae_m")) and row["p3"]["clearance_delta_mae_m"] < row["a2"]["clearance_delta_mae_m"]
        for row in parentwise.values()
    )
    quality_ok = all((
        candidate["raw_abs_rel_median"] <= baseline["raw_abs_rel_median"] + float(rules["maximum_raw_abs_rel_increase"]),
        candidate["clearance_mae_m"] <= baseline["clearance_mae_m"] + float(rules["maximum_clearance_mae_increase_m"]),
        candidate["false_clear_rate"] <= baseline["false_clear_rate"] + float(rules["maximum_false_clear_rate_increase"]),
        candidate["valid_to_unknown_rate"] <= float(rules["maximum_valid_to_unknown_rate"]),
        candidate["head_transition_macro_f1"] > 0.0,
        candidate["head_transition_exact_agreement"] > candidate["transition_majority_exact_baseline"],
    ))
    signal = improvement is not None and improvement >= float(rules["minimum_delta_mae_improvement_fraction"]) and parent_improved >= int(rules["minimum_parent_improvements"])
    terminal = "P3_TEMPORAL_DEVELOPMENT_SIGNAL_SUPPORTED" if signal and quality_ok else ("P3_TEMPORAL_DEVELOPMENT_SIGNAL_NOT_SUPPORTED" if improvement is None or improvement <= 0.0 or (parent_improved <= 1 and not quality_ok) else "P3_TEMPORAL_DEVELOPMENT_SIGNAL_MIXED")
    return terminal, {"head_delta_improvement_fraction": improvement, "parents_with_head_delta_improvement": parent_improved, "quality_guardrails_passed": quality_ok}


def evaluate_records(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure aggregation used by tests and by the inference runner."""
    parents = {str(row["parent_id"]) for row in rows}
    if len(parents) != 3:
        raise ValueError("development screen requires exactly three clip-capable validation parents")
    baseline = {**_summary(rows, "a2"), **_temporal_summary(rows, "a2")}
    candidate = {**_summary(rows, "p3"), **_temporal_summary(rows, "p3")}
    parentwise = _parentwise(rows)
    terminal, decision = decide(protocol, baseline, candidate, parentwise)
    return {"schema": SCHEMA, "data_role": "DEVELOPMENT_SIGNAL_ONLY", "claim_ceiling": "NOT_HOLDOUT_NOT_GENERALIZATION_NOT_PRODUCT_NOT_SAFETY_NOT_DEPLOYMENT", "baseline_a2": baseline, "candidate_p3": candidate, "parent_wise": parentwise, "decision": decision, "terminal": terminal, "bootstrap_used": False, "p_values_used": False}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bound(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or sha256_file(path) != _sha(expected, label):
        raise ValueError(f"{label} hash mismatch")


def _resolve_bound(repo_root: Path, binding: dict[str, Any], label: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} binding fields drifted")
    path = (repo_root / str(binding["path"])).resolve()
    _bound(path, binding["sha256"], label)
    return path


def validate_activation_bindings(
    activation: dict[str, Any], activation_sha256: str, protocol_sha256: str
) -> dict[str, Any]:
    required = {
        "schema", "protocol_sha256", "claim_ceiling", "train_manifest",
        "validation_manifest", "class_weights", "disagreement_cache", "runtime_state",
        "terminal",
    }
    if set(activation) != required:
        raise ValueError("activation bindings fields drifted")
    if activation["schema"] != "blindassist_p3_temporal_development_screen_r0_activation_bindings":
        raise ValueError("activation bindings schema drift")
    if activation["protocol_sha256"] != protocol_sha256:
        raise ValueError("activation protocol binding mismatch")
    if activation["claim_ceiling"] != "DEVELOPMENT_SIGNAL_ONLY":
        raise ValueError("activation evidence ceiling drift")
    if activation["terminal"] != "P3_TEMPORAL_DEVELOPMENT_ASSETS_MATERIALIZED_DEVELOPMENT_SIGNAL_ONLY":
        raise ValueError("activation terminal drift")
    state = activation["runtime_state"]
    expected_state = {
        "bonn_sealed_bundle_read": False, "holdout_outcomes_opened": False,
        "p3_model_constructed": False, "optimizer_constructed": False,
        "training_started": False, "a2_loaded_only_for_frozen_disagreement": True,
    }
    if state != expected_state:
        raise ValueError("activation runtime-state boundary drift")
    # `activation_sha256` is intentionally passed in rather than represented in
    # the protocol: this asset cannot exist before materialization.
    _sha(activation_sha256, "activation bindings")
    return activation


def validate_training_result(
    training: dict[str, Any], protocol: dict[str, Any], activation: dict[str, Any],
    activation_sha256: str,
) -> tuple[Path, str]:
    """Bind the post-training checkpoint without retroactively editing protocol.

    The protocol knows only immutable pre-training inputs.  This receipt is the
    sole authority for the selected P3 checkpoint and its validation manifest.
    """
    required = {
        "schema", "data_role", "claim_ceiling", "protocol_sha256",
        "activation_bindings_sha256", "train_manifest_sha256",
        "validation_manifest_sha256", "a2_checkpoint_sha256", "teacher_depth_sha256",
        "seed", "epochs_completed", "best_epoch", "best_validation_composite_total",
        "history", "checkpoint", "training_duration_s", "sealed_holdout_opened", "terminal",
    }
    if set(training) != required:
        raise ValueError("training result fields drifted")
    if training["schema"] != "blindassist_p3_temporal_development_screen_r0_training_result":
        raise ValueError("training result schema drift")
    if training["protocol_sha256"] != sha256_file(protocol["_path"]):
        raise ValueError("training result protocol binding mismatch")
    if training["data_role"] != "DEVELOPMENT_SIGNAL_ONLY" or training["claim_ceiling"] != "DEVELOPMENT_SIGNAL_ONLY":
        raise ValueError("training result evidence ceiling drift")
    if training["activation_bindings_sha256"] != activation_sha256:
        raise ValueError("training result activation bindings mismatch")
    if training["train_manifest_sha256"] != activation["train_manifest"]["sha256"]:
        raise ValueError("training result train manifest mismatch")
    if training["validation_manifest_sha256"] != activation["validation_manifest"]["sha256"]:
        raise ValueError("training result validation manifest mismatch")
    if training["a2_checkpoint_sha256"] != protocol["a2"]["checkpoint"]["sha256"]:
        raise ValueError("training result A2 checkpoint mismatch")
    if training["teacher_depth_sha256"] != protocol["teacher_cache"]["depth"]["sha256"]:
        raise ValueError("training result teacher depth mismatch")
    if int(training["seed"]) != int(protocol["training"]["seed"]):
        raise ValueError("training result seed mismatch")
    if int(training["epochs_completed"]) != 3:
        raise ValueError("training result did not complete the frozen three epochs")
    if not isinstance(training["history"], list) or len(training["history"]) != 3:
        raise ValueError("training result history drift")
    if not 1 <= int(training["best_epoch"]) <= 3 or not _finite(training["best_validation_composite_total"]):
        raise ValueError("training result checkpoint selection drift")
    if not _finite(training["training_duration_s"]) or float(training["training_duration_s"]) <= 0:
        raise ValueError("training result duration drift")
    if training["sealed_holdout_opened"] is not False:
        raise ValueError("training result opened sealed holdout")
    if training["terminal"] != "P3_TEMPORAL_DEVELOPMENT_SCREEN_TRAINING_COMPLETE_EVALUATION_PENDING":
        raise ValueError("training result terminal drift")
    checkpoint = training["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
        raise ValueError("training result checkpoint fields drifted")
    path = Path(str(checkpoint["path"])).resolve()
    _bound(path, checkpoint["sha256"], "P3 checkpoint from training result")
    return path, _sha(checkpoint["sha256"], "P3 checkpoint")


def _load_models(dav2_repo: Path, a2_checkpoint: Path, p3_checkpoint: Path, device: str):
    import torch
    sys.path.insert(0, str(dav2_repo / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2
    def backbone(state_path: Path):
        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
    a2 = backbone(a2_checkpoint)
    packed = torch.load(p3_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(packed, dict) or set(packed) != {"backbone", "temporal_head"}:
        raise ValueError("P3 checkpoint must contain exact backbone and temporal_head state dictionaries")
    p3 = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
    p3.load_state_dict(packed["backbone"], strict=True)
    head = DecoupledTemporalStateHead()
    head.load_state_dict(packed["temporal_head"], strict=True)
    return torch, a2.to(device).eval(), p3.to(device).eval(), head.to(device).eval()


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields drifted: {sorted(set(value) ^ keys)}")


def _read_depth(frame: dict[str, Any]) -> np.ndarray:
    raw = cv2.imread(str(frame["truth_depth_path"]), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"cannot decode truth depth: {frame['truth_depth_path']}")
    return np.asarray(raw, dtype=np.float32) * float(frame["truth_depth_scale_m"])


def _predict_depth(torch: Any, model: Any, bgr: np.ndarray, device: str) -> np.ndarray:
    image, _ = model.image2tensor(bgr, 392)
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
        prediction = model(image.to(device))
        prediction = torch.nn.functional.interpolate(prediction[:, None], size=bgr.shape[:2], mode="bilinear", align_corners=True)[:, 0]
    return prediction[0].float().cpu().numpy()


def _transition_name(index: int) -> str:
    return TRANSITIONS[int(index)]


def materialize_predictions(
    manifest: dict[str, Any], *, torch: Any, a2: Any, p3: Any, head: Any, device: str
) -> list[dict[str, Any]]:
    """Generate the complete ledger from RGB, independent depth and both models.

    The complete manifest is deliberately exact.  The role identity manifest is
    not enough: it has no independent depth/intrinsics mapping and cannot be
    silently guessed here.
    """
    _exact(manifest, {"schema", "protocol_sha256", "evidence_limit", "role", "clips"}, "development complete manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["role"] != "validation":
        raise ValueError("development complete manifest schema/role drift")
    if manifest["evidence_limit"] != "DEVELOPMENT_SIGNAL_ONLY":
        raise ValueError("development manifest evidence ceiling drift")
    clips = manifest["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("validation clips missing")
    parents = {str(clip.get("parent_id")) for clip in clips}
    if len(parents) != 3:
        raise ValueError("validation asset manifest must contain exactly three parents")
    clip_fields = {"clip_id", "parent_id", "video_id", "frames"}
    frame_fields = {"frame_id", "parent_id", "video_id", "timestamp_ns", "rgb_identity", "rgb_sha256", "teacher_depth_ref", "teacher_depth_sha256", "teacher_timestamp_ns", "teacher_valid", "tof_valid", "frozen_a2_mean_abs_log_depth_disagreement", "clearance_m", "geometry_state", "geometry_target_valid", "truth_depth_path", "truth_depth_sha256", "truth_depth_scale_m", "intrinsics_fx_fy_cx_cy"}
    result: list[dict[str, Any]] = []
    for clip in clips:
        _exact(clip, clip_fields, "clip")
        frames = clip["frames"]
        if not isinstance(frames, list) or len(frames) != 4:
            raise ValueError("every validation clip must be exactly four frames")
        timestamps: list[int] = []
        bgrs: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        for frame in frames:
            _exact(frame, frame_fields, "validation frame")
            if frame["parent_id"] != clip["parent_id"] or frame["video_id"] != clip["video_id"]:
                raise ValueError("frame identity drift")
            rgb_path, depth_path = Path(frame["rgb_path"]), Path(frame["truth_depth_path"])
            _bound(rgb_path, frame["rgb_sha256"], "RGB")
            _bound(depth_path, frame["truth_depth_sha256"], "truth depth")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"cannot decode RGB: {rgb_path}")
            timestamps.append(int(frame["timestamp_ns"]))
            bgrs.append(bgr)
            truths.append(_read_depth(frame))
        if not all(0 < right - left <= 500_000_000 for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("validation timestamps violate frozen clip cadence")
        a2_depth = [_predict_depth(torch, a2, image, device) for image in bgrs]
        p3_depth = [_predict_depth(torch, p3, image, device) for image in bgrs]
        sample_timestamps = torch.tensor([timestamps], dtype=torch.int64, device=device)
        teacher_timestamps = torch.tensor([[int(frame["teacher_timestamp_ns"]) for frame in frames]], dtype=torch.int64, device=device)
        teacher_valid = torch.tensor([[bool(frame["teacher_valid"]) for frame in frames]], device=device)
        tof_valid = torch.tensor([[bool(frame["tof_valid"]) for frame in frames]], device=device)
        disagreement = torch.tensor([[float(frame["frozen_a2_mean_abs_log_depth_disagreement"]) for frame in frames]], device=device)
        evidence = build_temporal_evidence(sample_timestamps, teacher_timestamps, teacher_valid, tof_valid, disagreement)
        p3_batch = torch.from_numpy(np.stack(p3_depth)[None]).to(device)
        seconds = (sample_timestamps[:, 1:] - sample_timestamps[:, :-1]).float() / 1_000_000_000.0
        with torch.inference_mode():
            head_output = head(p3_batch, evidence, seconds)
            head_delta = head_output["clearance_delta_m"][0].cpu().numpy()
            head_transition = head_output["geometry_transition_logits"][0].argmax(dim=-1).cpu().numpy()
            head_abstain = (torch.sigmoid(head_output["external_abstention_logits"][0]) >= 0.5).cpu().numpy()
        for index, frame in enumerate(frames):
            intrinsics = np.asarray([[float(frame["intrinsics_fx_fy_cx_cy"][0]), 0.0, float(frame["intrinsics_fx_fy_cx_cy"][2])], [0.0, float(frame["intrinsics_fx_fy_cx_cy"][1]), float(frame["intrinsics_fx_fy_cx_cy"][3])], [0.0, 0.0, 1.0]])
            truth_field = clearance_field(truths[index], intrinsics)
            a2_field = clearance_field(a2_depth[index], intrinsics)
            p3_field = clearance_field(p3_depth[index], intrinsics)
            truth = {"clearance_m": _field_clearances(truth_field), "state": [_state_from_field(truth_field, band) for band in range(3)]}
            a2_arm = {"depth": depth_metrics(a2_depth[index], truths[index]), "clearance_m": _field_clearances(a2_field), "state": [_state_from_field(a2_field, band) for band in range(3)], "external_abstain": [False, False, False]}
            p3_arm = {"depth": depth_metrics(p3_depth[index], truths[index]), "clearance_m": _field_clearances(p3_field), "state": [_state_from_field(p3_field, band) for band in range(3)], "external_abstain": [bool(value) for value in head_abstain[index]]}
            if index:
                p3_arm["head_from_previous"] = {"clearance_delta_m": [float(value) for value in head_delta[index - 1]], "transition": [_transition_name(value) for value in head_transition[index - 1]]}
            result.append({"clip_id": clip["clip_id"], "parent_id": clip["parent_id"], "frame_index": index, "truth": truth, "a2": a2_arm, "p3": p3_arm})
    return result


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-bindings", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("refusing to overwrite development screen output root")
    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path) | {"_path": protocol_path}
    if protocol.get("schema") != "blindassist_p3_temporal_development_screen_r0_protocol":
        raise ValueError("development protocol schema drift")
    if sha256_file(Path(__file__).resolve()) != _sha(protocol["implementation"]["evaluator_sha256"], "evaluator"):
        raise ValueError("evaluator source hash mismatch")
    a2_checkpoint = _resolve_bound(repo_root, protocol["a2"]["checkpoint"], "A2 checkpoint")
    activation_path = args.activation_bindings.resolve()
    activation_sha256 = sha256_file(activation_path)
    activation = validate_activation_bindings(
        _load_json(activation_path), activation_sha256, sha256_file(protocol_path)
    )
    validation_manifest = _resolve_bound(repo_root, activation["validation_manifest"], "validation manifest")
    training_result = _load_json(args.training_result)
    p3_checkpoint, p3_checkpoint_sha256 = validate_training_result(
        training_result, protocol, activation, activation_sha256
    )
    manifest = _load_json(validation_manifest)
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("complete manifest protocol binding mismatch")
    # Complete manifests retain the frozen role-manifest relative RGB identity;
    # bind it to the explicitly supplied read-only ARKit source root here.
    for clip in manifest.get("clips", []):
        for frame in clip.get("frames", []):
            frame["rgb_path"] = str((args.source_root / str(frame["video_id"]) / str(frame["rgb_identity"])).resolve())
    torch, a2, p3, head = _load_models(args.dav2_repo.resolve(), a2_checkpoint, p3_checkpoint, args.device)
    rows = materialize_predictions(manifest, torch=torch, a2=a2, p3=p3, head=head, device=args.device)
    ledger = {"schema": LEDGER_SCHEMA, "protocol_sha256": sha256_file(protocol_path), "activation_bindings_sha256": activation_sha256, "training_result_sha256": sha256_file(args.training_result), "validation_manifest_sha256": sha256_file(validation_manifest), "a2_checkpoint_sha256": sha256_file(a2_checkpoint), "p3_checkpoint_sha256": p3_checkpoint_sha256, "rows": rows}
    result = evaluate_records(protocol, rows) | {"protocol_sha256": ledger["protocol_sha256"], "validation_manifest_sha256": ledger["validation_manifest_sha256"], "a2_checkpoint_sha256": ledger["a2_checkpoint_sha256"], "p3_checkpoint_sha256": ledger["p3_checkpoint_sha256"], "prediction_ledger_sha256": hashlib.sha256((json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest().upper()}
    args.output_root.mkdir(parents=True)
    _atomic_json(args.output_root / "prediction-ledger.json", ledger)
    _atomic_json(args.output_root / "development-screen-result.json", result)
    print(json.dumps({"terminal": result["terminal"], "result": str(args.output_root / "development-screen-result.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
