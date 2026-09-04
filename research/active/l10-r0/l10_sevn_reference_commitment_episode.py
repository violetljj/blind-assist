"""Frozen PAN episodes with a separate reference-conditioned commit stage.

The candidate, observation policy and confidence-based selection are unchanged.
No verifier rejection causes a fallback candidate or another observation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from ultralytics import YOLO

import l10_sevn_progressive_episode as previous
import l10_sevn_fresh_pan_panel as panel
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2
import l10_sevn_ppocrv6_medium_portal_witness as witness
from l10_panolab import require, sha256_file, utc_now
from l10_reference_commitment import CONTRACT, ReferenceCommitment, features


ARMS = ("PASSIVE", "FIXED_SWEEP", "TRIGGERED_ACTIVE", "TRIGGERED_VERIFIED")


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_scalar) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cpu_portal_adapter(work):
    """Execute the exact frozen postprocessor on CPU without changing its source.

    Only model.predict(device=...) and the device synchronization are adapted.
    This isolated function-globals copy does not patch torch or global CUDA.
    """
    original = v2.portal_masks
    namespace = dict(original.__globals__)
    namespace["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
    portable = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    measured = {}

    class CpuModel:
        def __init__(self, model):
            self.model = model

        def predict(self, **kwargs):
            kwargs["device"] = "cpu"
            if not measured:
                import sys
                sys.path.insert(0, str(Path("tools").resolve()))
                import research_backend as backend
                output = []

                def probe():
                    result = self.model.predict(**kwargs)
                    output.append(result)
                    return result

                measured.update(backend.select_backend(backend.Workload.MODEL_INFERENCE,
                    cpu=backend.BackendCandidate("portal-cpu-capacity-fallback", "cpu", probe,
                        lambda result: backend.torch_observation(model=self.model.model)),
                    cpu_reason="ACCELERATOR_UNAVAILABLE", record_path=work / "backend.json"))
                return output[0]
            return self.model.predict(**kwargs)

    v2.portal_masks = lambda model, postprocessor, image, contract: portable(CpuModel(model), postprocessor, image, contract)


def in_view(box, shape):
    h, w = shape[:2]
    return 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h


def rectangle_mask(box, shape):
    mask = np.zeros(shape[:2], np.uint8)
    x1, y1, x2, y2 = box
    mask[max(0, int(np.floor(y1))):min(shape[0], int(np.ceil(y2))),
         max(0, int(np.floor(x1))):min(shape[1], int(np.ceil(x2)))] = 1
    return mask


def select_view(rows):
    emitted = [r for r in rows if r["output"].get("selected_binding") is not None]
    return min(emitted, key=lambda r: (-previous.selected_confidence(r["output"]), r["frame_id"], r["heading"])) if emitted else None


def evaluate(selected, truth, contract):
    if selected is None:
        return "UNKNOWN", None
    box = v1.annotation_to_viewport(truth["target_door_annotation"], selected["render"], 224, 84)
    if not in_view(box, selected["image"].shape):
        return "WRONG_BINDING", None
    portal_id = selected["output"]["selected_binding"]["portal"]["candidate_id"]
    metrics = v2.truth_mask_metrics(selected["masks"][portal_id], box, contract)
    return ("CORRECT_BINDING" if metrics["correct"] else "WRONG_BINDING"), metrics


def metrics(rows):
    result = {}
    passive = [row["arms"]["PASSIVE"] for row in rows]
    for arm in ARMS:
        result[arm] = previous.summarize([row["arms"][arm] for row in rows], passive)
    return result


def run(args):
    started = time.perf_counter()
    protocol = read_json(args.protocol)
    identity = sha256_file(args.protocol)
    work = args.work.resolve()
    require(not args.output.exists(), "Terminal result already exists; frozen panel cannot be rerun")
    work.mkdir(parents=True, exist_ok=True)
    lock = work / "owner.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        for entry in protocol["frozen_files"]:
            require(sha256_file(Path(entry["path"])) == entry["sha256"], f"Changed input: {entry['path']}")
        require(CONTRACT == protocol["verifier"], "Verifier contract changed")
        versions = previous.runtime_versions()
        require(versions == protocol["runtime_versions"], f"Runtime changed: {versions}")
        source, truth = read_json(protocol["source"]), read_json(protocol["truth"])
        panel.validate_pan_public_truth(source, truth, 4)
        old = read_json(protocol["baseline_protocol"])
        pixel = read_json(old["frozen_inputs"]["v2_protocol"]["path"])
        require(args.archive.stat().st_size == pixel["high_resolution_archive"]["bytes"], "Archive size mismatch")
        # A prior full-archive integrity receipt is hash-bound in the protocol.
        # Every actually opened member also receives its CRC and SHA-256 below.
        portal_path = previous.verify_sha256(pixel["models"]["portal_model"]["weights"])
        post_path = previous.verify_sha256(pixel["models"]["portal_model"]["postprocessor"])
        primary_root = Path(pixel["models"]["ocr"]["model_root"])
        for name, digest in pixel["models"]["ocr"]["sha256"].items():
            require(sha256_file(primary_root / name) == digest, "Primary OCR changed")
        medium_det = previous.verify_model_directory(old["models"]["medium_detection"])
        medium_rec = previous.verify_model_directory(old["models"]["medium_recognition"])
        torch.set_num_threads(4)
        cv2.setNumThreads(4)
        cpu_portal_adapter(work)
        model = YOLO(str(portal_path))
        postprocessor = v1.load_module(post_path)
        primary = RapidOCR(params={"Global.model_root_dir": str(primary_root.resolve()),
                                  "Global.log_level": "error", "EngineConfig.onnxruntime.intra_op_num_threads": 4,
                                  "EngineConfig.onnxruntime.inter_op_num_threads": 1})
        medium = PaddleOCR(text_detection_model_dir=str(medium_det), text_recognition_model_dir=str(medium_rec),
                           engine="onnxruntime", device="cpu", use_doc_orientation_classify=False,
                           use_doc_unwarping=False, use_textline_orientation=False)
        witness.PORTAL_CONTRACT = old["portal_medium_observation"]
        rows = []
        with zipfile.ZipFile(args.archive) as archive:
            for number, episode in enumerate(source["episodes"], 1):
                episode_started = time.perf_counter()
                episode_id = episode["episode_id"]
                checkpoint = work / f"{episode_id}.json"
                if checkpoint.exists():
                    saved = read_json(checkpoint)
                    require(saved["protocol_sha256"] in [identity, *protocol.get("checkpoint_compatible_protocol_sha256", [])], "Checkpoint protocol changed")
                    rows.append(saved["result"])
                    continue
                write_json(work / "progress.json", {"stage": "RUNNING", "completed": len(rows), "total": 8,
                                                    "active": episode_id, "pid": os.getpid(), "updated_at": utc_now()})
                members, receipts, cache = {}, {}, {}

                def panorama(frame):
                    if frame not in members:
                        name = f"panos/pano_{frame:06d}.png"
                        info = archive.getinfo(name)
                        payload = archive.read(info)
                        members[frame] = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                        require(members[frame] is not None and list(members[frame].shape) == [1280, 3840, 3], "Invalid panorama")
                        receipts[frame] = {"member": name, "bytes": len(payload), "crc32": f"{info.CRC:08x}",
                                           "sha256": hashlib.sha256(payload).hexdigest()}
                    return members[frame]

                refs = []
                for reference in episode["public_reference_views"]:
                    spec = reference["extraction"]
                    img, receipt = v1.render_viewport(panorama(spec["annotation"]["frame_id"]),
                                                    spec["panorama_angle_degrees"], spec["heading_degrees"], 135)
                    box = v1.annotation_to_viewport(spec["annotation"], receipt, 224, 84)
                    require(in_view(box, img.shape), "Public reference extraction is not fully visible")
                    mask_slice = v2.text_box_slice(box, img.shape[1], img.shape[0])
                    refs.append(img[mask_slice].copy())
                verifier = ReferenceCommitment(refs)

                def observe(observation_id, heading):
                    obs = source["observations"][observation_id]
                    frame = obs["frame_id"]
                    key = (frame, round(heading, 6))
                    if key in cache:
                        return cache[key]
                    img, render = v1.render_viewport(panorama(frame), obs["camera_pose"]["panorama_angle_degrees"], heading, 135)
                    _, output, masks = witness.infer_successor(img, episode["mission"], primary, medium, model, postprocessor, pixel["pixel_contract"])
                    row = {"image": img, "render": render, "frame_id": frame, "heading": heading,
                           "observation_id": observation_id, "output": output, "masks": masks}
                    cache[key] = row
                    return row

                def action_views(action):
                    start = episode["start_observation_id"]
                    edge = episode["transitions"][start][action]
                    require(edge["action_executed"], "Unavailable action")
                    destination = edge["to_observation_id"]
                    return [observe(destination, h) for h in source["observations"][destination]["viewport_headings_degrees"]]

                start_id = episode["start_observation_id"]
                start_obs = source["observations"][start_id]
                first = observe(start_id, start_obs["viewport_headings_degrees"][0])
                # Trigger is fixed before the counterfactual sweep is evaluated.
                action, reason = previous.trigger_action(previous.concise_output(first["output"]), episode, first["image"].shape[1])
                triggered_views = [first] if action == "HOLD" else action_views(action)
                selected = select_view(triggered_views)
                extra = 0 if action == "HOLD" else (3 if action == "SWEEP" else 1)
                verification_started = time.perf_counter()
                verification = None
                if selected is not None:
                    selected["features"] = features(selected["image"])
                    portal_id = selected["output"]["selected_binding"]["portal"]["candidate_id"]
                    verification = verifier.verify(selected["features"], selected["masks"][portal_id])
                verification_seconds = time.perf_counter() - verification_started
                sweep = action_views("SWEEP")
                selections = {"PASSIVE": select_view([first]), "FIXED_SWEEP": select_view(sweep),
                              "TRIGGERED_ACTIVE": selected,
                              "TRIGGERED_VERIFIED": selected if verification and verification["accepted"] else None}
                # Runtime choices are now sealed; evaluator truth cannot change them.
                sealed = {arm: None if chosen is None else {"frame": chosen["frame_id"], "heading": chosen["heading"],
                          "output": previous.concise_output(chosen["output"])} for arm, chosen in selections.items()}
                sealed_sha = hashlib.sha256(json.dumps(sealed, sort_keys=True).encode()).hexdigest()
                episode_truth = truth["episodes"][episode_id]
                arms = {}
                for arm, chosen in selections.items():
                    outcome, overlap = evaluate(chosen, episode_truth, pixel["pixel_contract"]["evaluation"])
                    action_name = "HOLD" if arm == "PASSIVE" else "SWEEP" if arm == "FIXED_SWEEP" else action
                    arms[arm] = {"episode_id": episode_id, "outcome": outcome,
                                 "binding_confidence": previous.selected_confidence(chosen["output"]) if chosen else None,
                                 "action": action_name, "action_executed": int(action_name != "HOLD"),
                                 "extra_observation_count": 0 if arm == "PASSIVE" else 3 if arm == "FIXED_SWEEP" else extra,
                                 "evaluation": overlap}
                # Truth-proposed diagnostics are separate from policy choices/costs.
                controls = []

                def query_features(view):
                    if "features" not in view:
                        view["features"] = features(view["image"])
                    return view["features"]

                annotations = [("TARGET_ORACLE", episode_truth["target_door_annotation"])] + [
                    ("SIBLING_ORACLE", a) for a in episode_truth["different_address_nonoverlapping_sibling_annotations"]]
                for kind, annotation in annotations:
                    found = None
                    for view in cache.values():
                        box = v1.annotation_to_viewport(annotation, view["render"], 224, 84)
                        if in_view(box, view["image"].shape):
                            qf = query_features(view)
                            found = {"kind": kind, "heading": view["heading"],
                                     "verifier": verifier.verify(qf, rectangle_mask(box, view["image"].shape))}
                            break
                    controls.append(found or {"kind": kind, "status": "NOT_EVALUABLE_IN_POLICY_VIEWS"})
                # Current-target region is outside these policy viewports. This is
                # region absence in a known panorama, not general open-world absence.
                absence = []
                for view in cache.values():
                    target_box = v1.annotation_to_viewport(episode_truth["target_door_annotation"], view["render"], 224, 84)
                    width = view["image"].shape[1]
                    if target_box[2] <= 0 or target_box[0] >= width:
                        for portal in view["output"].get("portal_mask_candidates", []):
                            qf = query_features(view)
                            absence.append({"heading": view["heading"], "portal_id": portal["candidate_id"],
                                            "verifier": verifier.verify(qf, view["masks"][portal["candidate_id"]])})
                row = {"episode_id": episode_id, "mission": episode["mission"], "action_reason": reason,
                       "arms": arms, "sealed_runtime": sealed, "sealed_runtime_sha256": sealed_sha,
                       "verification": verification, "verification_seconds": round(verification_seconds, 4),
                       "reference_setup_views": 2, "diagnostic_controls": controls, "target_region_absent_controls": absence,
                       "archive_members": list(receipts.values()), "unique_inferred_views": len(cache),
                       "seconds": round(time.perf_counter() - episode_started, 4)}
                rows.append(row)
                write_json(checkpoint, {"protocol_sha256": identity, "result": row})
                print(json.dumps({"progress": f"{number}/8", "episode": episode_id, "action": action,
                                  "outcomes": {a: r["outcome"] for a, r in arms.items()}, "seconds": row["seconds"]}), flush=True)
        summary = metrics(rows)
        baseline, checked = summary["TRIGGERED_ACTIVE"], summary["TRIGGERED_VERIFIED"]
        retained = sum(r["arms"]["TRIGGERED_ACTIVE"]["outcome"] == "CORRECT_BINDING" and
                       r["arms"]["TRIGGERED_VERIFIED"]["outcome"] == "CORRECT_BINDING" for r in rows)
        retention = retained / baseline["correct_binding"] if baseline["correct_binding"] else None
        removed = baseline["wrong_binding"] - checked["wrong_binding"]
        all_controls = [c for r in rows for c in r["diagnostic_controls"]]
        siblings = [c for c in all_controls if c["kind"] == "SIBLING_ORACLE" and "verifier" in c]
        absent = [c for r in rows for c in r["target_region_absent_controls"]]
        gates = {"wrong_reduction_evaluable": baseline["wrong_binding"] > 0,
                 "at_least_half_wrong_removed": baseline["wrong_binding"] > 0 and removed / baseline["wrong_binding"] >= .5,
                 "correct_retention_at_least_80_percent": retention is not None and retention >= .8,
                 "no_new_wrong_binding": all(r["arms"]["TRIGGERED_VERIFIED"]["outcome"] != "WRONG_BINDING" or
                                              r["arms"]["TRIGGERED_ACTIVE"]["outcome"] == "WRONG_BINDING" for r in rows),
                 "no_extra_online_observation": checked["extra_observation_count"] == baseline["extra_observation_count"],
                 "all_sibling_controls_evaluable": len(siblings) == 8,
                 "no_sibling_false_accept": not any(c["verifier"]["accepted"] for c in siblings)}
        passed = all(gates.values())
        decision = "L10_SEVN_REFERENCE_COMMITMENT_FRESH_DEVELOPMENT_GATE_" + ("MET" if passed else "NOT_MET")
        result = {"schema": "blindassist-l10-sevn-reference-commitment-result-v1", "generated_at": utc_now(),
                  "decision": decision, "protocol_sha256": identity, "metrics": summary, "gates": gates,
                  "correct_retention": retention, "wrong_removed": removed,
                  "sibling_controls": {"evaluable": len(siblings), "false_accepts": sum(c["verifier"]["accepted"] for c in siblings)},
                  "target_region_absent_controls": {"candidate_count": len(absent), "accepted": sum(c["verifier"]["accepted"] for c in absent)},
                  "reference_setup_views": 16, "wall_seconds_this_invocation": time.perf_counter()-started,
                  "actual_model_device": str(next(model.model.parameters()).device), "runtime_versions": versions,
                  "episodes": rows, "claim_boundary": protocol["claim_boundary"]}
        write_json(args.output, result)
        write_json(work / "progress.json", {"stage": "COMPLETE", "completed": 8, "total": 8, "decision": decision, "updated_at": utc_now()})
        print(json.dumps({"decision": decision, "metrics": summary, "gates": gates}, indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
