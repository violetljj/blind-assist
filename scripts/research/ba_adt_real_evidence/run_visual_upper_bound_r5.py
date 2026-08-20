#!/usr/bin/env python3
"""Run one frozen SAM 3 visual-query teacher over RGB without GT access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mine_goal_episodes import sha256


def select_trusted_exemplar(observations: dict) -> dict:
    """Select the strongest detector frame in the first acquired segment, using RGB output only."""
    events = observations["events"]
    acquired = next(row["frame_index"] for row in events if row["event"] == "ACQUIRED")
    lost = next(row["frame_index"] for row in events if row["event"] == "LOST" and row["frame_index"] > acquired)
    candidates = [
        row
        for row in observations["frames"][acquired:lost]
        if row["observation_source"] == "detector" and row["bbox_xyxy"] is not None
    ]
    if not candidates:
        raise ValueError("first acquired segment has no detector exemplar")
    best = max(candidates, key=lambda row: (row["target_confidence"], -row["frame_index"]))
    if best["target_confidence"] < 0.70:
        raise ValueError("trusted exemplar confidence is below frozen 0.70 floor")
    return {
        "frame_index": int(best["frame_index"]),
        "bbox_xyxy": [float(value) for value in best["bbox_xyxy"]],
        "confidence": float(best["target_confidence"]),
        "selection_rule": "max_detector_confidence_in_first_acquired_segment_with_0.70_floor",
    }


def empty_prediction(device):
    return {
        "obj_id_to_mask": {},
        "obj_id_to_score": {},
        "obj_id_to_cls": {},
        "obj_id_to_tracker_score": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stop-after-frame", type=int, help="Mechanical preflight only; formal R5 must omit")
    args = parser.parse_args()

    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    if observations.get("groundtruth_argument_supported") is not False:
        raise ValueError("R5 teacher requires an RGB-only observation lineage")
    exemplar = select_trusted_exemplar(observations)
    target_text = observations["model"]["target_class"]

    import torch
    import ultralytics
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    class DelayedPromptPredictor(SAM3VideoSemanticPredictor):
        def inference(self, im, bboxes=None, labels=None, text=None, *extra, **kwargs):
            frame = self.dataset.frame - 1
            self.inference_state["im"] = im
            if frame < exemplar["frame_index"]:
                return empty_prediction(self.device)
            if "text_ids" not in self.inference_state:
                if frame != exemplar["frame_index"]:
                    raise RuntimeError("teacher prompt was not initialized on the frozen exemplar frame")
                self.add_prompt(
                    frame_idx=frame,
                    text=[target_text],
                    bboxes=[exemplar["bbox_xyxy"]],
                    labels=[1],
                )
            return self._run_single_frame_inference(frame, reverse=False)

    overrides = {
        "conf": args.confidence,
        "task": "segment",
        "mode": "predict",
        "model": str(args.model),
        "quantize": 16,
        "imgsz": args.imgsz,
        "device": args.device,
        "verbose": False,
    }
    predictor = DelayedPromptPredictor(
        overrides=overrides,
        score_threshold_detection=args.confidence,
        new_det_thresh=args.confidence,
    )
    start = time.perf_counter()
    frames = []
    with torch.inference_mode():
        results = predictor(
            source=str(args.video),
            text=[target_text],
            bboxes=[exemplar["bbox_xyxy"]],
            labels=[1],
            stream=True,
        )
        for frame_index, result in enumerate(results):
            boxes = []
            if result.boxes is not None:
                for bbox, confidence, class_id in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                    strict=True,
                ):
                    boxes.append({
                        "bbox_xyxy": [float(value) for value in bbox],
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                    })
            frames.append({"frame_index": frame_index, "candidates": boxes})
            if args.stop_after_frame is not None and frame_index >= args.stop_after_frame:
                break

    output = {
        "schema_version": "ba_adt_visual_upper_bound_r5_teacher_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5",
        "teacher": {
            "name": "SAM 3",
            "implementation": f"ultralytics-{ultralytics.__version__}",
            "model_path_name": args.model.name,
            "model_sha256": sha256(args.model),
            "imgsz": args.imgsz,
            "quantization": "fp16",
            "candidate_confidence_floor": args.confidence,
            "text_prompt": target_text,
            "visual_exemplar": exemplar,
        },
        "inputs": {
            "video_path_name": args.video.name,
            "video_sha256": sha256(args.video),
            "observations_path_name": args.observations.name,
            "observations_sha256": sha256(args.observations),
        },
        "groundtruth_argument_supported": False,
        "future_location_or_visibility_input_supported": False,
        "formal_run": args.stop_after_frame is None,
        "stop_after_frame": args.stop_after_frame,
        "source_frame_size": observations["frame_size"],
        "frame_count": len(frames),
        "elapsed_seconds": time.perf_counter() - start,
        "frames": frames,
        "claim_ceiling": "consumed_development_teacher_capability_only_no_edge_product_or_safety_claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "frame_count": len(frames), "formal_run": output["formal_run"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
