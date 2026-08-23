"""Independent adapter seams for named-referent engineering evidence."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image

from .schema import (
    CHANNELS,
    CLAIM_CEILING,
    SCHEMA_VERSION,
    CurrentFrame,
    GoalReferencePack,
    available_channel,
    evidence_item,
    not_evaluable_channel,
    provider_identity,
    sha256_file,
    validate_output,
)


class EvidenceAdapter(Protocol):
    channel: str

    def identity(self) -> Mapping[str, Any]: ...

    def collect(self, frame: CurrentFrame, goal: GoalReferencePack) -> Mapping[str, Any]: ...


def normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in folded if character.isalnum())


def text_match(raw_text: str, query: str) -> dict[str, Any]:
    normalized_text = normalize_text(raw_text)
    normalized_query = normalize_text(query)
    ratio = SequenceMatcher(None, normalized_text, normalized_query).ratio() if normalized_text and normalized_query else 0.0
    exact = bool(normalized_text and normalized_text == normalized_query)
    substring = bool(
        normalized_text
        and normalized_query
        and min(len(normalized_text), len(normalized_query)) >= 2
        and (normalized_query in normalized_text or normalized_text in normalized_query)
    )
    return {
        "recognized_text": normalized_text,
        "query": normalized_query,
        "exact": exact,
        "substring": substring,
        "fuzzy_ratio": ratio,
        "match_class": "EXACT" if exact else "SUBSTRING" if substring else "FUZZY" if ratio >= 0.70 else "NONE",
    }


def _safe_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _exception_result(channel: str, identity: Mapping[str, Any], started: float, error: Exception) -> dict[str, Any]:
    return not_evaluable_channel(
        channel=channel,
        identity=identity,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        code=f"{type(error).__name__.upper()}",
        message=str(error),
        retryable=False,
    )


class PaddleOCRV5Adapter:
    channel = "text_evidence"

    def __init__(self, *, device: str, cache_dir: Path):
        self.device = device
        self.cache_dir = cache_dir.resolve()
        self._engine: Any | None = None
        self._initialization_latency_ms: float | None = None

    def identity(self) -> Mapping[str, Any]:
        artifacts: dict[str, str] = {}
        if self.cache_dir.is_dir():
            for path in sorted(self.cache_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".json", ".yml", ".yaml", ".pdiparams", ".pdmodel"}:
                    artifacts[str(path.relative_to(self.cache_dir))] = sha256_file(path)
        return provider_identity(
            provider="PaddleOCR",
            implementation_version="PP-OCRv5-mobile-det-rec-adapter-v0",
            model_repository="PaddlePaddle/PaddleOCR",
            model_revision="PP-OCRv5_mobile_det+PP-OCRv5_mobile_rec",
            artifact_sha256=artifacts,
            runtime={
                "python": platform.python_version(),
                "paddleocr": _safe_version("paddleocr"),
                "paddlepaddle_gpu": _safe_version("paddlepaddle-gpu"),
                "paddlepaddle": _safe_version("paddlepaddle"),
                "device": self.device,
                "initialization_latency_ms": self._initialization_latency_ms,
            },
        )

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(self.cache_dir)
        from paddleocr import PaddleOCR

        started = time.perf_counter()
        self._engine = PaddleOCR(
            device=self.device,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self._initialization_latency_ms = (time.perf_counter() - started) * 1000.0
        return self._engine

    @staticmethod
    def _payload(prediction: Any) -> Mapping[str, Any]:
        value = prediction.json if hasattr(prediction, "json") else prediction
        if callable(value):
            value = value()
        if isinstance(value, Mapping) and isinstance(value.get("res"), Mapping):
            value = value["res"]
        if not isinstance(value, Mapping):
            raise ValueError("PaddleOCR prediction has no mapping payload")
        return value

    def collect(self, frame: CurrentFrame, goal: GoalReferencePack) -> Mapping[str, Any]:
        started = time.perf_counter()
        try:
            engine = self._ensure_engine()
            predictions = list(engine.predict(str(frame.image_path)))
            variants = (goal.name, *goal.aliases)
            items: list[dict[str, Any]] = []
            for page_index, prediction in enumerate(predictions):
                payload = self._payload(prediction)
                texts = payload.get("rec_texts", [])
                scores = payload.get("rec_scores", [])
                polygons = payload.get("rec_polys", payload.get("dt_polys", []))
                boxes = payload.get("rec_boxes", [])
                for index, raw_text in enumerate(texts):
                    raw = str(raw_text)
                    comparisons = [text_match(raw, query) for query in variants]
                    best = max(comparisons, key=lambda item: (item["exact"], item["substring"], item["fuzzy_ratio"]))
                    polygon = polygons[index].tolist() if index < len(polygons) and hasattr(polygons[index], "tolist") else polygons[index] if index < len(polygons) else None
                    bbox = boxes[index].tolist() if index < len(boxes) and hasattr(boxes[index], "tolist") else boxes[index] if index < len(boxes) else None
                    source: dict[str, Any] = {
                        "image_path": str(frame.image_path),
                        "image_sha256": sha256_file(frame.image_path),
                        "page_index": page_index,
                        "polygon": polygon,
                        "bbox_xyxy": bbox,
                        "crop": None,
                    }
                    items.append(evidence_item(
                        item_id=f"ocr-{frame.frame_id}-{page_index:02d}-{index:03d}",
                        source=source,
                        raw_match={
                            "recognized_text": raw,
                            "confidence": float(scores[index]) if index < len(scores) else None,
                            "goal_variants": list(variants),
                        },
                        normalized_match={"best": best, "all_goal_variants": comparisons},
                    ))
            return available_channel(
                channel=self.channel,
                identity=self.identity(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                items=items,
            )
        except Exception as error:  # optional provider must fail closed
            return _exception_result(self.channel, self.identity(), started, error)

    def close(self) -> None:
        self._engine = None
        gc.collect()
        try:
            import paddle

            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass


class DINOv2ReferenceAdapter:
    channel = "visual_reference_evidence"
    MODEL_REPOSITORY = "facebook/dinov2-small"
    MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    MODEL_FILES = {
        "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
        "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    }

    def __init__(self, *, model_dir: Path, device: str):
        self.model_dir = model_dir.resolve()
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None

    def identity(self) -> Mapping[str, Any]:
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(self.model_dir.iterdir())
            if path.is_file() and path.name in {"model.safetensors", "config.json", "preprocessor_config.json"}
        } if self.model_dir.is_dir() else {}
        return provider_identity(
            provider="TransformersAutoModel",
            implementation_version="dinov2-global-cls-cosine-v0",
            model_repository=self.MODEL_REPOSITORY,
            model_revision=self.MODEL_REVISION,
            artifact_sha256=artifacts,
            runtime={
                "python": platform.python_version(),
                "torch": _safe_version("torch"),
                "transformers": _safe_version("transformers"),
                "device": self.device,
            },
        )

    def _ensure_engine(self) -> tuple[Any, Any, Any]:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for DINOv2 but is unavailable")
        for name, expected in self.MODEL_FILES.items():
            path = self.model_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"DINOv2 model artifact missing or changed: {name}")
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_dir, local_files_only=True, use_safetensors=True).to(self.device).eval()
        return torch, self._processor, self._model

    def _embeddings(self, paths: Sequence[Path]) -> Any:
        torch, processor, model = self._ensure_engine()
        images = []
        for path in paths:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vectors = model(**inputs).last_hidden_state[:, 0, :].float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors.detach().cpu()

    def collect(self, frame: CurrentFrame, goal: GoalReferencePack) -> Mapping[str, Any]:
        started = time.perf_counter()
        try:
            references = [("reference_image", path) for path in goal.reference_images]
            if goal.logo:
                references.append(("logo", goal.logo))
            if not references:
                return not_evaluable_channel(
                    channel=self.channel,
                    identity=self.identity(),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    code="REFERENCE_INPUT_ABSENT",
                    message="goal pack has neither reference_images nor logo",
                    retryable=False,
                )
            vectors = self._embeddings([frame.image_path, *(path for _, path in references)])
            similarities = [float(vectors[0].dot(vectors[index])) for index in range(1, len(references) + 1)]
            order = sorted(range(len(similarities)), key=lambda index: (-similarities[index], str(references[index][1])))
            ranks = {index: rank for rank, index in enumerate(order, start=1)}
            items = []
            for index, ((kind, reference), similarity) in enumerate(zip(references, similarities)):
                items.append(evidence_item(
                    item_id=f"visual-{frame.frame_id}-{index:03d}",
                    source={
                        "image_path": str(frame.image_path),
                        "image_sha256": sha256_file(frame.image_path),
                        "crop": None,
                        "reference_path": str(reference),
                        "reference_sha256": sha256_file(reference),
                        "reference_kind": kind,
                    },
                    raw_match={"cosine_similarity": similarity},
                    normalized_match={
                        "embedding_normalization": "L2",
                        "cosine_similarity": similarity,
                        "reference_rank": ranks[index],
                    },
                ))
            return available_channel(
                channel=self.channel,
                identity=self.identity(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                items=items,
            )
        except Exception as error:
            return _exception_result(self.channel, self.identity(), started, error)

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class GroundingDINOProposalAdapter:
    channel = "proposal_evidence"

    def __init__(self, *, model_dir: Path):
        self.model_dir = model_dir.resolve()

    def identity(self) -> Mapping[str, Any]:
        from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino

        weights = self.model_dir / dino.WEIGHTS_FILENAME
        return provider_identity(
            provider="GroundingDINO",
            implementation_version="reuse-existing-proposal-mechanical-smoke-v0",
            model_repository=dino.MODEL_REPOSITORY,
            model_revision=dino.MODEL_REVISION,
            artifact_sha256={dino.WEIGHTS_FILENAME: sha256_file(weights)} if weights.is_file() else {},
            runtime={
                "torch": _safe_version("torch"),
                "transformers": _safe_version("transformers"),
                "prompt": dino.PROMPT,
                "box_threshold": dino.BOX_THRESHOLD,
                "text_threshold": dino.TEXT_THRESHOLD,
                "authority": "VISUAL_PROPOSAL_ONLY",
            },
        )

    def collect(self, frame: CurrentFrame, goal: GoalReferencePack) -> Mapping[str, Any]:
        del goal
        started = time.perf_counter()
        try:
            from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino

            with Image.open(frame.image_path) as opened:
                width, height = opened.size
            image_sha = sha256_file(frame.image_path)
            outputs, runtime = dino.run_inference(self.model_dir, [{
                "id": frame.frame_id,
                "path": str(frame.image_path),
                "width": width,
                "height": height,
                "image_sha256": image_sha,
            }])
            identity = dict(self.identity())
            identity["runtime"] = {**identity["runtime"], **runtime}
            items = []
            for index, proposal in enumerate(outputs[0]["proposals"]):
                box = [float(value) for value in proposal["bbox_xyxy"]]
                items.append(evidence_item(
                    item_id=f"proposal-{frame.frame_id}-{index:03d}",
                    source={
                        "image_path": str(frame.image_path),
                        "image_sha256": image_sha,
                        "bbox_xyxy": box,
                        "crop": None,
                    },
                    raw_match={"label": str(proposal["label"]), "model_score": float(proposal["score"])},
                    normalized_match={
                        "bbox_normalized_xyxy": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
                        "score_semantics": "MODEL_PROPOSAL_RANKING_SCORE_NOT_TRUTH",
                        "provider_rank": index + 1,
                    },
                ))
            return available_channel(
                channel=self.channel,
                identity=identity,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                items=items,
            )
        except Exception as error:
            return _exception_result(self.channel, self.identity(), started, error)


class MapBearingAdapter:
    channel = "bearing_evidence"

    def identity(self) -> Mapping[str, Any]:
        return provider_identity(provider="GoalReferencePack", implementation_version="optional-map-bearing-v0")

    def collect(self, frame: CurrentFrame, goal: GoalReferencePack) -> Mapping[str, Any]:
        started = time.perf_counter()
        if goal.map_bearing_degrees is None:
            return not_evaluable_channel(
                channel=self.channel,
                identity=self.identity(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                code="OPTIONAL_MAP_BEARING_ABSENT",
                message="goal reference pack did not provide map_bearing_degrees",
                retryable=False,
            )
        delta = None
        if frame.heading_degrees is not None:
            delta = (goal.map_bearing_degrees - frame.heading_degrees + 540.0) % 360.0 - 180.0
        item = evidence_item(
            item_id=f"bearing-{frame.frame_id}",
            source={"map_bearing_degrees": goal.map_bearing_degrees},
            raw_match={"map_bearing_degrees": goal.map_bearing_degrees, "current_heading_degrees": frame.heading_degrees},
            normalized_match={"signed_clockwise_delta_degrees": delta, "normalization": "[-180,180)"},
        )
        return available_channel(
            channel=self.channel,
            identity=self.identity(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            items=[item],
        )


class NamedReferentProviderV0:
    """Run each evidence adapter independently and never fuse their outputs."""

    def __init__(self, adapters: Mapping[str, EvidenceAdapter | None]):
        unknown = set(adapters) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown adapter channels: {sorted(unknown)}")
        self.adapters = dict(adapters)

    def run(self, frame: CurrentFrame, goal: GoalReferencePack) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        for channel in CHANNELS:
            adapter = self.adapters.get(channel)
            if adapter is None:
                evidence[channel] = not_evaluable_channel(
                    channel=channel,
                    identity=provider_identity(provider="UNCONFIGURED", implementation_version="v0"),
                    latency_ms=0.0,
                    code="PROVIDER_UNCONFIGURED",
                    message=f"no {channel} adapter was configured",
                    retryable=False,
                )
                continue
            started = time.perf_counter()
            try:
                result = dict(adapter.collect(frame, goal))
                if result.get("channel") != channel:
                    raise ValueError(f"adapter returned {result.get('channel')!r} for {channel}")
                evidence[channel] = result
            except Exception as error:
                evidence[channel] = _exception_result(channel, adapter.identity(), started, error)
        output = {
            "schema_version": SCHEMA_VERSION,
            "claim_ceiling": CLAIM_CEILING,
            "fusion": "FORBIDDEN_NOT_PERFORMED",
            "frame": frame.to_dict(),
            "goal_reference_pack": goal.to_dict(),
            "evidence": evidence,
        }
        validate_output(output)
        return output
