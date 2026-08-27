from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

import artvideo_ocr_replay as replay


LOCKED_WEIGHTS = (0.45, 0.40, 0.15)
LOCKED_GATES = (0.38, 0.60, 0.64, 0.025)
REACQUIRE_GATES = (0.58, 0.55, 0.67, 0.04)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize an empty embedding")
    return (vector / norm).astype(np.float32, copy=False)


class SemanticVisualMemory:
    """OCR admits identity; DINO crop evidence only preserves continuity.

    The class deliberately never reads evaluator-only ``truth_ids``. A visual
    match cannot acquire a goal on its own: initial acquisition and every
    reacquisition retain a lexical gate.
    """

    name = "L10_SC0_semantic_visual_memory"

    def __init__(
        self,
        goal: str,
        lexical: Callable[[str, str], float] = replay.lexical,
        spatial: Callable[[tuple[float, float], tuple[float, float]], float] = replay.spatial,
    ):
        self.goal = goal
        self.lexical = lexical
        self.spatial = spatial
        self.center: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.long: np.ndarray | None = None
        self.short: np.ndarray | None = None
        self.lost = False
        self.pending_center: tuple[float, float] | None = None
        self.pending_hits = 0

    @staticmethod
    def _embedding(candidate: Any) -> np.ndarray:
        value = getattr(candidate, "pooled_patch_embedding", None)
        if value is None:
            raise ValueError("candidate is missing pooled_patch_embedding")
        return value

    def _spatial_score(self, candidate: Any) -> float:
        if self.center is None:
            return 0.5
        predicted = (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1],
        )
        return self.spatial(candidate.center, predicted)

    def _enter_lost(self) -> None:
        self.lost = True
        self.pending_center = None
        self.pending_hits = 0

    def _accept(self, candidate: Any, index: int) -> int:
        embedding = self._embedding(candidate)
        self.long = (
            embedding.copy()
            if self.long is None
            else _normalize(0.85 * self.long + 0.15 * embedding)
        )
        self.short = embedding
        previous = self.center
        self.center = candidate.center
        if previous is not None:
            self.velocity = (
                0.65 * self.velocity[0] + 0.35 * (self.center[0] - previous[0]),
                0.65 * self.velocity[1] + 0.35 * (self.center[1] - previous[1]),
            )
        self.lost = False
        self.pending_center = None
        self.pending_hits = 0
        return index

    @staticmethod
    def _runner_up(rows: list[tuple[float, ...]]) -> float:
        ordered = sorted((row[0] for row in rows), reverse=True)
        return ordered[1] if len(ordered) > 1 else 0.0

    def _acquire(self, candidates: list[Any]) -> int | None:
        if not candidates:
            return None
        lexical_score, _ocr, index = max(
            (self.lexical(self.goal, item.text), item.score, index)
            for index, item in enumerate(candidates)
        )
        return self._accept(candidates[index], index) if lexical_score >= 0.58 else None

    def _track(self, candidates: list[Any]) -> int | None:
        if not candidates:
            self._enter_lost()
            return None
        if self.long is None or self.short is None:
            raise RuntimeError("continuity memory is not initialized")
        rows: list[tuple[float, float, float, float, float, int]] = []
        for index, item in enumerate(candidates):
            embedding = self._embedding(item)
            lexical_score = self.lexical(self.goal, item.text)
            visual = max(float(embedding @ self.short), float(embedding @ self.long))
            local = self._spatial_score(item)
            combined = 0.45 * lexical_score + 0.40 * visual + 0.15 * local
            rows.append((combined, lexical_score, visual, local, item.score, index))
        combined, lexical_score, visual, _local, _ocr, index = max(rows)
        if (
            lexical_score < LOCKED_GATES[0]
            or visual < LOCKED_GATES[1]
            or combined < LOCKED_GATES[2]
            or combined - self._runner_up(rows) < LOCKED_GATES[3]
        ):
            self._enter_lost()
            return None
        return self._accept(candidates[index], index)

    def _reacquire(self, candidates: list[Any]) -> int | None:
        if not candidates:
            self.pending_center = None
            self.pending_hits = 0
            return None
        if self.long is None:
            raise RuntimeError("long memory is not initialized")
        rows: list[tuple[float, float, float, float, float, int]] = []
        for index, item in enumerate(candidates):
            lexical_score = self.lexical(self.goal, item.text)
            visual = float(self._embedding(item) @ self.long)
            local = self._spatial_score(item)
            combined = 0.45 * lexical_score + 0.40 * visual + 0.15 * local
            rows.append((combined, lexical_score, visual, local, item.score, index))
        combined, lexical_score, visual, _local, _ocr, index = max(rows)
        if (
            lexical_score < REACQUIRE_GATES[0]
            or visual < REACQUIRE_GATES[1]
            or combined < REACQUIRE_GATES[2]
            or combined - self._runner_up(rows) < REACQUIRE_GATES[3]
        ):
            self.pending_center = None
            self.pending_hits = 0
            return None
        candidate = candidates[index]
        if self.pending_center is not None and self.spatial(candidate.center, self.pending_center) >= 0.55:
            self.pending_hits += 1
        else:
            self.pending_center = candidate.center
            self.pending_hits = 1
        return self._accept(candidate, index) if self.pending_hits >= 2 else None

    def step(self, candidates: list[Any]) -> int | None:
        if self.long is None:
            return self._acquire(candidates)
        return self._reacquire(candidates) if self.lost else self._track(candidates)


def _crop_bounds(box: np.ndarray, width: int, height: int, context: float = 0.25) -> tuple[int, int, int, int]:
    x0, y0 = box[:, 0].min(), box[:, 1].min()
    x1, y1 = box[:, 0].max(), box[:, 1].max()
    dx, dy = max(1.0, x1 - x0) * context, max(1.0, y1 - y0) * context
    return (
        max(0, int(math.floor(x0 - dx))),
        max(0, int(math.floor(y0 - dy))),
        min(width, int(math.ceil(x1 + dx))),
        min(height, int(math.ceil(y1 + dy))),
    )


def build_embedding_cache(
    dataset: Path,
    ocr_cache: dict[str, Any],
    model_dir: Path,
    cache_path: Path,
    index_path: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    crops: list[Image.Image] = []
    items: list[dict[str, Any]] = []
    for frame_key in sorted(ocr_cache["frames"]):
        cached = ocr_cache["frames"][frame_key]
        image = cv2.imread(str(dataset / Path(frame_key)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(dataset / Path(frame_key))
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for detection_index, detection in enumerate(cached["detections"]):
            box = np.asarray(detection["box"], dtype=np.float32)
            x0, y0, x1, y1 = _crop_bounds(box, width, height)
            crops.append(Image.fromarray(rgb[y0:y1, x0:x1]))
            items.append(
                {
                    "embedding_index": len(items),
                    "frame_key": frame_key,
                    "detection_index": detection_index,
                    "ocr_text": detection["text"],
                    "ocr_score": float(detection["score"]),
                    "quadrilateral": box.astype(float).tolist(),
                    "crop_xyxy": [x0, y0, x1, y1],
                }
            )
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = processor(images=crops[start : start + batch_size], return_tensors="pt")
            output = model(pixel_values=batch["pixel_values"].to(device)).last_hidden_state
            pooled = torch.nn.functional.normalize(output[:, 1:].mean(dim=1), dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
    pooled_patch = np.concatenate(vectors, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, pooled_patch=pooled_patch)
    index_path.write_text(
        json.dumps(
            {
                "schema": "l10_sc0_dinov2_crop_embedding_index_v0",
                "context_fraction_per_axis": 0.25,
                "model": str(model_dir),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "crops": len(items),
        "dimensions": int(pooled_patch.shape[1]),
        "device": device,
        "embedding_wall_s": round(time.perf_counter() - started, 4),
    }


def attach_embeddings(
    videos: list[dict[str, Any]],
    annotation_paths: list[Path],
    index_payload: dict[str, Any],
    pooled_patch: np.ndarray,
) -> int:
    by_key: dict[str, dict[int, dict[str, Any]]] = {}
    for item in index_payload["items"]:
        by_key.setdefault(item["frame_key"], {})[int(item["detection_index"])] = item
    attached = 0
    for video, annotation_path in zip(videos, annotation_paths, strict=True):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = {int(row["frame_id"]): str(row["frame_jpg"]) for row in annotation["frame"]}
        for frame_id, candidates in video["candidates"].items():
            key = (Path("Test") / "frame" / annotation_path.stem / names[frame_id]).as_posix()
            indexed = by_key.get(key, {})
            if len(indexed) != len(candidates):
                raise ValueError(f"embedding/candidate count mismatch for {key}")
            for detection_index, candidate in enumerate(candidates):
                item = indexed[detection_index]
                if item["ocr_text"] != candidate.text:
                    raise ValueError(f"OCR/index mismatch for {key} #{detection_index}")
                candidate.pooled_patch_embedding = pooled_patch[int(item["embedding_index"])]
                attached += 1
    return attached


def _summarize_by_video(
    specs: list[tuple[Any, ...]],
    tracker_type: type,
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        rows.setdefault(spec[0]["name"], []).append(replay.run_episode(tracker_type, *spec))
    return {name: replay.summarize(items) for name, items in sorted(rows.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L10-SC0 semantic-admitted DINO continuity replay.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    ocr_cache = json.loads(args.ocr_cache.read_text(encoding="utf-8"))
    build_receipt = None
    if not args.embedding_cache.exists() or not args.embedding_index.exists():
        if args.model is None:
            parser.error("--model is required when embedding cache/index do not exist")
        build_receipt = build_embedding_cache(
            args.dataset, ocr_cache, args.model, args.embedding_cache, args.embedding_index
        )
    index_payload = json.loads(args.embedding_index.read_text(encoding="utf-8"))
    with np.load(args.embedding_cache, allow_pickle=False) as payload:
        pooled_patch = payload["pooled_patch"].astype(np.float32, copy=True)

    annotation_paths = [args.dataset / "Test" / "json" / f"{name}.json" for name in args.videos]
    videos = [replay.load_video(args.dataset, path, ocr_cache) for path in annotation_paths]
    attached = attach_embeddings(videos, annotation_paths, index_payload, pooled_patch)
    specs = []
    for video in videos:
        for target_id, goal, frame_ids in replay.eligible_tracks(video, args.minimum_frames):
            for fraction in (0.30, 0.50, 0.70):
                start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
                specs.append((video, target_id, goal, frame_ids, start, args.gap_length))

    metrics = {}
    per_video = {}
    for tracker_type in (replay.PerFrameText, replay.StickyText, SemanticVisualMemory):
        rows = [replay.run_episode(tracker_type, *spec) for spec in specs]
        metrics[tracker_type.name] = replay.summarize(rows)
        per_video[tracker_type.name] = _summarize_by_video(specs, tracker_type)
    b1, sc0 = metrics[replay.StickyText.name], metrics[SemanticVisualMemory.name]
    relative_gate = {
        "accuracy_above_b1": sc0["target_frame_accuracy"] > b1["target_frame_accuracy"],
        "wrong_below_b1": sc0["wrong_selection_frames"] < b1["wrong_selection_frames"],
        "reacquire_not_below_b1": sc0["gap_reacquire_rate"] >= b1["gap_reacquire_rate"],
    }
    result = {
        "schema": "l10_sc0_artvideo_semantic_visual_replay_v0",
        "status": "DEVELOPMENT_SOURCE_EFFECT" if all(relative_gate.values()) else "SOURCE_GATE_NOT_MET",
        "videos": args.videos,
        "episode_protocol": {
            "eligible_tracks": len(specs) // 3,
            "episodes": len(specs),
            "minimum_track_frames": args.minimum_frames,
            "artificial_gap_frames": args.gap_length,
            "gap_positions": [0.30, 0.50, 0.70],
            "reacquire_window_frames": 5,
        },
        "algorithm": {
            "identity_authority": "OCR lexical admission",
            "continuity_only": "DINOv2 pooled-patch crop embedding plus short/long memory and current-camera motion",
            "appearance_only_acquire": False,
            "appearance_only_reacquire": False,
            "locked_weights": LOCKED_WEIGHTS,
            "locked_gates": LOCKED_GATES,
            "reacquire_gates": REACQUIRE_GATES,
        },
        "embedding_cache": {
            "path": str(args.embedding_cache),
            "rows": int(pooled_patch.shape[0]),
            "dimensions": int(pooled_patch.shape[1]),
            "attached_candidates": attached,
            "build": build_receipt,
        },
        "metrics": metrics,
        "per_video": per_video,
        "relative_b1_gate": {"checks": relative_gate, "passed": all(relative_gate.values())},
        "runtime_s": round(time.perf_counter() - started, 4),
        "claim_ceiling": (
            "ArTVideo detector-plus-recognizer Development replay with evaluator-injected proposal gaps; "
            "not open-world identity, natural-distribution tracking, navigation, product, user-benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": sc0, "gate": result["relative_b1_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
