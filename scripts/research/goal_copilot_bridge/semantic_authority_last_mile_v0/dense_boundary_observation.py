"""DeepLSD dense-field boundary candidates for SAGE-LM V1-B-R3."""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from .observation import RgbEpisodeInput, RgbEpisodeTruth
from .two_view_observation import (
    ImageLine,
    SourceCameraPose,
    SourcePoseTwoViewBoundaryProvider,
    _image_line_from_points,
)


DISTANCE_THRESHOLD_PX = 2.5
VERTICAL_ORIENTATION_TOLERANCE_DEG = 20.0
FINAL_SUPPORT_MINIMUM_PX = 9.0


def _angular_distance_mod_pi(first: np.ndarray, second: float) -> np.ndarray:
    delta = np.abs(first - second)
    return np.minimum(delta, np.pi - delta)


def _effective_row_support(mask: np.ndarray, line: ImageLine) -> float:
    supported = 0
    for y in range(mask.shape[0]):
        x = int(round(line.x_at(float(y))))
        if x < 0 or x >= mask.shape[1]:
            continue
        if np.any(mask[y, max(0, x - 2) : min(mask.shape[1], x + 3)]):
            supported += 1
    return float(supported)


def fit_dense_vertical_lines(support_mask: np.ndarray) -> tuple[list[ImageLine], dict]:
    """Fuse short dense-field fragments before producing fitted line candidates."""

    raw = cv2.HoughLinesP(
        support_mask.astype(np.uint8) * 255,
        1,
        np.pi / 180.0,
        threshold=5,
        minLineLength=4,
        maxLineGap=6,
    )
    fragments = []
    if raw is not None:
        for x1, y1, x2, y2 in raw[:, 0, :]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dy) < 4.0 or abs(dx) > max(4.0, abs(dy) * math.tan(math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG))):
                continue
            slope = dx / dy
            x_mid = float(x1 + (support_mask.shape[0] * 0.5 - y1) * slope)
            fragments.append(
                {
                    "points": [(float(x1), float(y1)), (float(x2), float(y2))],
                    "slope": slope,
                    "x_mid": x_mid,
                    "length": math.hypot(dx, dy),
                }
            )
    fragments.sort(key=lambda row: row["x_mid"])
    groups: list[list[dict]] = []
    for fragment in fragments:
        best_group = None
        best_distance = math.inf
        for group in groups:
            weight = sum(row["length"] for row in group)
            group_x = sum(row["x_mid"] * row["length"] for row in group) / weight
            group_slope = sum(row["slope"] * row["length"] for row in group) / weight
            distance = abs(fragment["x_mid"] - group_x)
            if distance <= 5.0 and abs(fragment["slope"] - group_slope) <= 0.16 and distance < best_distance:
                best_group = group
                best_distance = distance
        if best_group is None:
            groups.append([fragment])
        else:
            best_group.append(fragment)

    lines = []
    rejected_short = 0
    for group in groups:
        points = [point for fragment in group for point in fragment["points"]]
        fitted = _image_line_from_points(points, 1.0, len(group))
        support = _effective_row_support(support_mask, fitted)
        if support < FINAL_SUPPORT_MINIMUM_PX:
            rejected_short += 1
            continue
        lines.append(ImageLine(fitted.coefficients, support, len(group)))
    lines.sort(key=lambda line: line.x_at((support_mask.shape[0] - 1) * 0.5))
    return lines, {
        "support_pixel_count": int(np.count_nonzero(support_mask)),
        "raw_fragment_count": len(fragments),
        "fused_group_count": len(groups),
        "rejected_short_support_count": rejected_short,
        "fitted_candidate_count": len(lines),
        "fitted_support_lengths_px": [line.support_length_px for line in lines],
    }


class DeepLsdDenseFieldExtractor:
    """Load the official network and expose dense-support fitted candidates."""

    def __init__(self, repository_root: Path, runtime_root: Path, checkpoint: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.runtime_root = runtime_root.resolve()
        self.checkpoint = checkpoint.resolve()
        for path in (self.repository_root, self.runtime_root, self.checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest().upper()
        self.repository_commit = subprocess.run(
            ["git", "-C", str(self.repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        sys.path[:0] = [str(self.runtime_root), str(self.repository_root)]
        if "pytlsd" not in sys.modules:
            stub = types.ModuleType("pytlsd")
            stub.lsd = lambda *args, **kwargs: None
            sys.modules["pytlsd"] = stub
        import torch
        from deeplsd.models.deeplsd_inference import DeepLSD

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_value = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model = DeepLSD({"detect_lines": False})
        self.model.load_state_dict(checkpoint_value["model"])
        self.model = self.model.to(self.device).eval()

    @property
    def identity(self) -> dict:
        return {
            "official_repository": "https://github.com/cvg/DeepLSD",
            "repository_commit": self.repository_commit,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
            "distance_threshold_px": DISTANCE_THRESHOLD_PX,
            "vertical_orientation_tolerance_deg": VERTICAL_ORIENTATION_TOLERANCE_DEG,
            "final_support_minimum_px": FINAL_SUPPORT_MINIMUM_PX,
            "official_discrete_lsd": "NOT_USED",
        }

    def predict_field(self, bgr: np.ndarray, masked_bbox: tuple[int, int, int, int]) -> dict:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = masked_bbox
        gray[max(0, y1 - 3) : min(gray.shape[0], y2 + 4), max(0, x1 - 3) : min(gray.shape[1], x2 + 4)] = 0
        tensor = self.torch.from_numpy(gray.copy()).float().to(self.device)[None, None] / 255.0
        with self.torch.inference_mode():
            output = self.model({"image": tensor})
        distance = output["df"][0].detach().cpu().numpy()
        orientation = output["line_level"][0].detach().cpu().numpy()
        return {"distance": distance, "orientation": orientation, "masked_gray": gray}

    def detect(self, bgr: np.ndarray, masked_bbox: tuple[int, int, int, int]) -> tuple[list[ImageLine], dict]:
        field = self.predict_field(bgr, masked_bbox)
        distance = field["distance"]
        orientation = field["orientation"]
        x1, y1, x2, y2 = masked_bbox
        vertical = _angular_distance_mod_pi(orientation, np.pi / 2) <= math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG)
        support = (distance <= DISTANCE_THRESHOLD_PX) & vertical
        support[max(0, y1 - 3) : min(support.shape[0], y2 + 4), max(0, x1 - 3) : min(support.shape[1], x2 + 4)] = False
        lines, diagnostics = fit_dense_vertical_lines(support)
        diagnostics.update(
            {
                "distance_min_px": float(np.min(distance)),
                "distance_median_px": float(np.median(distance)),
                "distance_threshold_px": DISTANCE_THRESHOLD_PX,
                "vertical_orientation_tolerance_deg": VERTICAL_ORIENTATION_TOLERANCE_DEG,
                "final_support_minimum_px": FINAL_SUPPORT_MINIMUM_PX,
            }
        )
        return lines, diagnostics


class DenseFieldBoundaryProvider(SourcePoseTwoViewBoundaryProvider):
    """Keep B1/B2 association and geometry unchanged, replacing candidates only."""

    def __init__(
        self,
        episode_input: RgbEpisodeInput,
        truth: RgbEpisodeTruth | None,
        pose_a: SourceCameraPose,
        pose_b: SourceCameraPose,
        arm: str,
        extractor: DeepLsdDenseFieldExtractor,
    ) -> None:
        super().__init__(episode_input, truth, pose_a, pose_b, arm)
        self.extractor = extractor
        self.arm_name = f"SAGE_LM_V1B_R3_DENSE_FIELD_{arm.upper()}"

    def _detect(self) -> tuple[list[ImageLine], list[ImageLine], tuple[int, int, int, int]]:
        visible = [row for row in self.input.exact_anchor_observations if row.visible]
        first = next((row for row in visible if row.frame_index == 0), None)
        second = next((row for row in visible if row.frame_index == self.input.active_parallax_frame_index), None)
        if first is None or second is None or first.referent_id != second.referent_id:
            raise ValueError("active source-pose pair requires the same visible exact anchor")
        frame_a = self._load(self.input.rgb_frames[first.frame_index])
        frame_b = self._load(self.input.rgb_frames[second.frame_index])
        lines_a, diagnostics_a = self.extractor.detect(frame_a, first.bbox_xyxy)
        lines_b, diagnostics_b = self.extractor.detect(frame_b, second.bbox_xyxy)
        self.diagnostics.update({"dense_field_a": diagnostics_a, "dense_field_b": diagnostics_b})
        return lines_a, lines_b, first.bbox_xyxy
