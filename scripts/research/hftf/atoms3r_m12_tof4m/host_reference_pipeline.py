from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class HostReferenceYoloPipeline:
    identity = "HOST_REFERENCE_YOLO11N_RAW_SCORE_RISK_R0_NOT_PRODUCTION"

    def __init__(self, model_path: Path, feedback_mode: str) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except Exception as error:
            raise RuntimeError(
                "ai-edge-litert is required for the host reference pipeline"
            ) from error
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.feedback_mode = feedback_mode

    def infer(self, image: np.ndarray, _metadata: dict[str, Any]) -> np.ndarray:
        shape = [int(value) for value in self.input_detail["shape"]]
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError(f"Unsupported detector input shape: {shape}")
        resized = cv2.resize(
            image, (shape[2], shape[1]), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        dtype = self.input_detail["dtype"]
        if dtype == np.float32:
            input_tensor = rgb.astype(np.float32) / 255.0
        elif dtype == np.uint8:
            input_tensor = rgb.astype(np.uint8)
        elif dtype == np.int8:
            input_tensor = (rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
        else:
            input_tensor = rgb.astype(dtype)
        self.interpreter.set_tensor(
            self.input_detail["index"], np.expand_dims(input_tensor, axis=0)
        )
        self.interpreter.invoke()
        return np.asarray(self.interpreter.get_tensor(self.output_detail["index"]))

    @staticmethod
    def calculate_risk(output: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
        squeezed = (
            np.squeeze(output, axis=0)
            if output.ndim == 3 and output.shape[0] == 1
            else output
        )
        if squeezed.ndim != 2:
            top_score = 0.0
        elif squeezed.shape[0] >= 5 and squeezed.shape[0] < squeezed.shape[1]:
            top_score = float(np.max(squeezed[4:, :]))
        elif squeezed.shape[1] >= 5:
            top_score = float(np.max(squeezed[:, 4:]))
        else:
            top_score = 0.0
        tof_close = bool(metadata["tof_valid"] and metadata["tof_range_mm"] < 1000)
        return {
            "schema": "blindassist_host_reference_risk_r0",
            "top_raw_class_score": top_score,
            "tof_under_1m": tof_close,
            "risk": bool(top_score >= 0.45 and tof_close),
            "authority": "TIMING_ONLY_NOT_PRODUCTION_RISK",
        }

    def emit_feedback(
        self, risk: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        result = {
            "schema": "blindassist_host_reference_feedback_dispatch_r0",
            "would_emit": bool(risk["risk"]),
            "mode": self.feedback_mode,
            "physical_output_emitted": False,
            "authority": "TIMING_ONLY_NO_VOICE_OR_VIBRATION",
            "frame_sequence": metadata["frame_sequence"],
        }
        if self.feedback_mode == "console" and result["would_emit"]:
            print(
                f"TIMING_ONLY risk frame={metadata['frame_sequence']} "
                f"range_mm={metadata['tof_range_mm']}"
            )
        return result


def build_pipeline(
    model_path: Path | None, feedback_mode: str
) -> HostReferenceYoloPipeline:
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(f"Host reference model is missing: {model_path}")
    return HostReferenceYoloPipeline(model_path, feedback_mode)
