"""Pinned, minimal runtime adapter for the official PDM PerMIR scorer.

The official repository is consumed from an ignored, hash-locked checkout. No
upstream source is vendored here. The adapter preserves the published PerMIR
configuration (Stable Diffusion v1.4, prompt ``A photo``, DIFT t=261/up=1,
ensemble=2, top-k=1, timestep 49, Q/K appearance maps) while making two
mechanical execution changes explicit:

* DIFT and DDIM-attention models run sequentially to fit an 8 GB GPU.
* The native reference mask is resized to the 64x64 comparison grid, matching
  the paper's shared-grid equations rather than the repository's accidental
  32x32 mask/64x64 index mismatch.

There is one fixed score and no layer, prompt, crop, timestep, precision,
aggregation, threshold, or model sweep.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    pdm_hard_error_unary_probe as probe,
)


SCHEMA_VERSION = "blindassist_pdm_official_runtime_adapter_v0"
PDM_REPO_COMMIT = "3ac3a12650dceb7666c07d1b1294495cc24384b5"
PDM_SOURCE_SHA256 = {
    "pdm_permir.py": "afa8c36424e4555458c7a9f84a3bca36e02492bead64abb32e2398e2c230df7d",
    "dift.py": "f62e819b92a4de72dcad8647a7df49d4e557649cc6351cd3cf9e748321f35c59",
    "StableDiffusionPipelineWithDDIMInversion.py": "3021bf488d67ae06ac86389bc3102eb919057a146a388236675e7bb2bd0c09e5",
    "ptp_utils.py": "6ff8fd9d940084ecf48b2785b1c043614fbbedcafc9e8dd76d5a8cba49b8c01b",
    "attention_store.py": "1b2f8d382cf5a6bacfd8aba095281fe2a8286d8a3fae7860ee15942a509fe00f",
}
MODEL_REPO_ID = "CompVis/stable-diffusion-v1-4"
MODEL_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
PROMPT = "A photo"
IMAGE_SIZE = 512
ATTENTION_SIZE = 64
DIFT_TIMESTEP = 261
DIFT_UP_BLOCK_INDEX = 1
DIFT_ENSEMBLE_SIZE = 2
DIFT_TOPK = 1
ATTENTION_TIMESTAMP = 49
ATTENTION_PLACE_STEM = "up_blocks_3_attentions_1_transformer_blocks_0_attn1"
ATTENTION_KEY_FORMAT = f"{ATTENTION_PLACE_STEM}_self"
TORCH_DTYPE = "float16"
REQUIRED_DISTRIBUTIONS = {
    "diffusers": "0.18.2",
    "transformers": "4.31.0",
    "huggingface-hub": "0.16.4",
    "tokenizers": "0.13.3",
    "accelerate": "0.21.0",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise probe.PdmHardErrorProbeError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return probe._sha256_file(path)


def _body_hash(value: Mapping[str, Any]) -> str:
    return probe._body_hash(value)


def _atomic_json(path: Path, value: Any) -> None:
    probe._atomic_json(path, value)


def _load_json(path: Path) -> Any:
    return probe._load_json(path)


def _verify_body_hash(value: Mapping[str, Any], name: str) -> None:
    probe._verify_body_hash(value, name)


def _git_head(repo_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _model_files(model_dir: Path) -> list[Path]:
    required_suffixes = {
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "vae/config.json",
    }
    relative = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    _require(required_suffixes <= relative, f"Stable Diffusion snapshot is incomplete: {required_suffixes - relative}")
    _require(
        any(name.startswith("unet/diffusion_pytorch_model") for name in relative),
        "Stable Diffusion UNet weights missing",
    )
    _require(
        any(name.startswith("vae/diffusion_pytorch_model") for name in relative),
        "Stable Diffusion VAE weights missing",
    )
    _require(
        any(name.startswith("text_encoder/pytorch_model") or name.startswith("text_encoder/model") for name in relative),
        "Stable Diffusion text encoder weights missing",
    )
    return sorted(path for path in model_dir.rglob("*") if path.is_file())


def lock_provider(repo_dir: Path, model_dir: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), f"provider lock already exists: {output_path}")
    _require(_git_head(repo_dir) == PDM_REPO_COMMIT, "official PDM checkout commit drifted")
    source_hashes = {}
    for relative, expected in PDM_SOURCE_SHA256.items():
        actual = _sha256_file(repo_dir / relative)
        _require(actual == expected, f"official PDM source hash drifted: {relative}")
        source_hashes[relative] = actual
    distributions = {
        name: importlib.metadata.version(name)
        for name in REQUIRED_DISTRIBUTIONS
    }
    _require(distributions == REQUIRED_DISTRIBUTIONS, "PDM dependency versions drifted")
    model_files = _model_files(model_dir)
    model_hashes = {
        path.relative_to(model_dir).as_posix(): _sha256_file(path)
        for path in model_files
    }
    lock = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "official_repository": "https://github.com/dvirsamuel/PDM",
        "official_commit": PDM_REPO_COMMIT,
        "official_source_sha256": source_hashes,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256_file(Path(__file__)),
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir.resolve()),
        "model_file_sha256": model_hashes,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "dependency_versions": distributions,
        "runtime_contract": {
            "prompt": PROMPT,
            "image_size": IMAGE_SIZE,
            "attention_size": ATTENTION_SIZE,
            "dift_timestep": DIFT_TIMESTEP,
            "dift_up_block_index": DIFT_UP_BLOCK_INDEX,
            "dift_ensemble_size": DIFT_ENSEMBLE_SIZE,
            "dift_topk": DIFT_TOPK,
            "attention_timestamp": ATTENTION_TIMESTAMP,
            "attention_key_format": ATTENTION_KEY_FORMAT,
            "torch_dtype": TORCH_DTYPE,
            "score": "MEAN_OVER_Q_AND_K_OF_MEAN_REFERENCE_MASKED_TOP1_DIFT_WEIGHTED_COSINE",
            "reference_mask_grid": "NATIVE_VISIBLE_MASK_RESIZED_NEAREST_TO_64X64_SHARED_GRID",
            "candidate_scoring": "INDEPENDENT_UNARY",
            "threshold": None,
            "training": False,
            "sweep": False,
        },
        "terminal": "PDM_PROVIDER_LOCKED_READY",
    }
    lock["body_sha256"] = _body_hash(lock)
    _atomic_json(output_path, lock)
    return lock


def _verify_provider_lock(lock: Mapping[str, Any], repo_dir: Path, model_dir: Path) -> None:
    _verify_body_hash(lock, "PDM provider lock")
    _require(lock["terminal"] == "PDM_PROVIDER_LOCKED_READY", "PDM provider is not ready")
    _require(lock["official_commit"] == _git_head(repo_dir) == PDM_REPO_COMMIT, "PDM commit drifted")
    _require(lock["adapter_sha256"] == _sha256_file(Path(__file__)), "PDM adapter hash drifted")
    for relative, digest in lock["official_source_sha256"].items():
        _require(_sha256_file(repo_dir / relative) == digest, f"PDM source changed: {relative}")
    for relative, digest in lock["model_file_sha256"].items():
        _require(_sha256_file(model_dir / relative) == digest, f"PDM model artifact changed: {relative}")
    _require(lock["runtime_contract"]["torch_dtype"] == TORCH_DTYPE, "PDM precision drifted")


def _install_provider_imports(repo_dir: Path) -> tuple[Any, Any, Any, Any]:
    repo = str(repo_dir.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    dift = importlib.import_module("dift")
    pipeline_module = importlib.import_module("StableDiffusionPipelineWithDDIMInversion")
    ptp_utils = importlib.import_module("ptp_utils")
    attention_store = importlib.import_module("attention_store")
    return dift, pipeline_module, ptp_utils, attention_store


def _crop_image(item: Mapping[str, Any], *, mask: bool = False, size: int = IMAGE_SIZE) -> Image.Image:
    with Image.open(item["mask_path"] if mask else item["image_path"]) as source:
        image = source.convert("L" if mask else "RGB")
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in item["crop_bbox_xyxy_normalized"]]
    bounds = (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    return image.crop(bounds).resize((size, size), resampling)


def _crop_id(item: Mapping[str, Any]) -> str:
    value = {
        "image_sha256": item["image_sha256"],
        "crop_bbox_xyxy_normalized": item["crop_bbox_xyxy_normalized"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _tensor_from_pil(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def _release_cuda(value: Any) -> None:
    del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _SelectiveAttentionStore:
    """Controller-compatible store retaining only the frozen Q/K timestep."""

    def __init__(self) -> None:
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.is_inject = False
        self.step_store: dict[str, torch.Tensor] = {}
        self.attention_store: dict[int, dict[str, torch.Tensor]] = {}
        self.target_key_steps: dict[str, set[int]] = {}

    @property
    def num_uncond_att_layers(self) -> int:
        return 0

    def __call__(self, attention: torch.Tensor, is_cross: bool, place: str) -> torch.Tensor:
        key = f"{place}_{'cross' if is_cross else 'self'}"
        if key in {f"Q_{ATTENTION_KEY_FORMAT}", f"K_{ATTENTION_KEY_FORMAT}"}:
            self.target_key_steps.setdefault(key, set()).add(self.cur_step)
        if (
            self.cur_step == ATTENTION_TIMESTAMP
            and key in {f"Q_{ATTENTION_KEY_FORMAT}", f"K_{ATTENTION_KEY_FORMAT}"}
        ):
            self.step_store[key] = attention.detach().cpu()
        return attention

    def check_next_step(self) -> None:
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def between_steps(self) -> None:
        completed_step = self.cur_step - 1
        if completed_step == ATTENTION_TIMESTAMP:
            self.attention_store[completed_step] = self.step_store
        self.step_store = {}


def _load_dift_pipeline(model_dir: Path, dift: Any, dtype: torch.dtype) -> Any:
    from diffusers import DDIMScheduler

    unet = dift.MyUNet2DConditionModel.from_pretrained(
        str(model_dir), subfolder="unet", torch_dtype=dtype, local_files_only=True
    )
    pipe = dift.OneStepSDPipeline.from_pretrained(
        str(model_dir),
        unet=unet,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.vae.decoder = None
    pipe.scheduler = DDIMScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


@torch.no_grad()
def _extract_dift(pipe: Any, image: Image.Image) -> torch.Tensor:
    tensor = _tensor_from_pil(image).repeat(DIFT_ENSEMBLE_SIZE, 1, 1, 1).to(
        device="cuda", dtype=pipe.unet.dtype
    )
    prompt_embeds = pipe._encode_prompt(
        prompt=PROMPT,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    ).repeat(DIFT_ENSEMBLE_SIZE, 1, 1)
    torch.manual_seed(42)
    output = pipe(
        img_tensor=tensor,
        t=DIFT_TIMESTEP,
        up_ft_indices=[DIFT_UP_BLOCK_INDEX],
        prompt_embeds=prompt_embeds,
    )
    return output["up_ft"][DIFT_UP_BLOCK_INDEX].mean(0, keepdim=True).float().cpu()


def _load_attention_pipeline(model_dir: Path, pipeline_module: Any, dtype: torch.dtype) -> Any:
    from diffusers import DDIMScheduler

    pipeline_class = pipeline_module.StableDiffusionPipelineWithDDIMInversion
    scheduler = DDIMScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    pipe = pipeline_class.from_pretrained(
        str(model_dir),
        safety_checker=None,
        requires_safety_checker=False,
        scheduler=scheduler,
        torch_dtype=dtype,
        local_files_only=True,
    )
    return pipe.to("cuda")


@torch.no_grad()
def _extract_attention(pipe: Any, ptp_utils: Any, image: Image.Image) -> list[torch.Tensor]:
    inverse = pipe.invert(PROMPT, image=image, guidance_scale=1.0, num_inference_steps=50)
    controller = _SelectiveAttentionStore()
    ptp_utils.register_attention_control_efficient(pipe, controller)
    pipe(
        PROMPT,
        latents=inverse.latents,
        guidance_scale=1.0,
        num_inference_steps=50,
        output_type="latent",
    )
    store = controller.attention_store.get(ATTENTION_TIMESTAMP, {})
    keys = [f"Q_{ATTENTION_KEY_FORMAT}", f"K_{ATTENTION_KEY_FORMAT}"]
    observed = {key: sorted(controller.target_key_steps.get(key, set())) for key in keys}
    _require(
        all(key in store for key in keys),
        "PDM frozen attention maps were not captured "
        f"(completed_steps={controller.cur_step}, attention_layers={controller.num_att_layers}, "
        f"target_key_steps={observed})",
    )
    result = [store[key][0].float().cpu() for key in keys]
    _require(all(tuple(value.shape[:1]) == (ATTENTION_SIZE * ATTENTION_SIZE,) for value in result), "PDM attention grid drifted")
    return result


def _reference_mask64(reference: Mapping[str, Any]) -> np.ndarray:
    mask = np.asarray(_crop_image(reference, mask=True, size=ATTENTION_SIZE)) > 0
    _require(bool(mask.any()), "PDM reference mask is empty after crop")
    return mask


def _pdm_score(
    dift_module: Any,
    reference_dift: torch.Tensor,
    candidate_dift: torch.Tensor,
    reference_attention: Sequence[torch.Tensor],
    candidate_attention: Sequence[torch.Tensor],
    reference_mask: np.ndarray,
) -> float:
    ref_points, candidate_points, dift_scores = dift_module.get_correspondences_seg(
        reference_dift.cuda(),
        candidate_dift.cuda(),
        reference_mask,
        img_size=ATTENTION_SIZE,
        topk=DIFT_TOPK,
    )
    per_map = []
    for reference_map, candidate_map in zip(reference_attention, candidate_attention):
        point_scores = []
        reference_map = reference_map.cuda()
        candidate_map = candidate_map.cuda()
        for ref_point, candidate_point, dift_score in zip(ref_points, candidate_points, dift_scores):
            reference_index = np.ravel_multi_index(ref_point, dims=(ATTENTION_SIZE, ATTENTION_SIZE))
            candidate_indices = np.ravel_multi_index(
                candidate_point.T, dims=(ATTENTION_SIZE, ATTENTION_SIZE)
            )
            source = reference_map[torch.tensor(reference_index, device="cuda")].reshape(1, -1)
            target = candidate_map[
                torch.tensor(candidate_indices, device="cuda")
            ].reshape(DIFT_TOPK, -1)
            cosine = (F.normalize(source, dim=-1) @ F.normalize(target, dim=-1).T).mean()
            point_scores.append(float(dift_score) * float(cosine.item()))
        _require(point_scores, "PDM reference mask produced no correspondence scores")
        per_map.append(float(np.mean(point_scores)))
    score = float(np.mean(per_map))
    _require(math.isfinite(score), "PDM unary score is non-finite")
    return score


def _prepare_run_config(
    public_manifest_path: Path,
    provider_lock_path: Path,
    repo_dir: Path,
    model_dir: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    public = _load_json(public_manifest_path)
    lock = _load_json(provider_lock_path)
    probe._verify_body_hash(public, "PDM challenger public manifest")
    probe._assert_public_blind(public)
    _verify_provider_lock(lock, repo_dir, model_dir)
    prepared = probe.prepare_challenger_inputs(public_manifest_path)
    crops: dict[str, Mapping[str, Any]] = {}

    def register(item: Mapping[str, Any]) -> str:
        crop_id = _crop_id(item)
        previous = crops.get(crop_id)
        if previous is not None:
            _require(
                previous["image_sha256"] == item["image_sha256"]
                and previous["crop_bbox_xyxy_normalized"] == item["crop_bbox_xyxy_normalized"],
                "PDM crop ID collision",
            )
        else:
            crops[crop_id] = dict(item)
        return crop_id

    pairs = []
    references: dict[str, Mapping[str, Any]] = {}
    for row in prepared["pairs"]:
        references[row["case_id"]] = row["reference"]
        pairs.append(
            {
                "pair_id": row["pair_id"],
                "case_id": row["case_id"],
                "reference_crop_id": register(row["reference"]),
                "candidate_crop_ids": {
                    slot: register(row["candidates"][slot]) for slot in ("A", "B")
                },
            }
        )
    absences = []
    for row in prepared["absences"]:
        references[row["case_id"]] = row["reference"]
        absences.append(
            {
                "absence_id": row["absence_id"],
                "case_id": row["case_id"],
                "reference_crop_id": register(row["reference"]),
                "candidate_crop_id": register(row["candidate"]),
            }
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": probe.PROTOCOL_ID,
        "stage": "PDM_CHALLENGER",
        "created_at_utc": _utc_now(),
        "adapter_sha256": _sha256_file(Path(__file__)),
        "public_manifest_body_sha256": public["body_sha256"],
        "provider_lock": lock,
        "provider_lock_file_sha256": _sha256_file(provider_lock_path),
        "score_contract": lock["runtime_contract"],
        "crop_count": len(crops),
        "pairs": pairs,
        "absences": absences,
        "claim_ceiling": probe.CLAIM_CEILING,
    }
    probe._assert_public_blind(config)
    config["body_sha256"] = _body_hash(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "run-config.json", config)
    crop_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_config_body_sha256": config["body_sha256"],
        "crops": crops,
    }
    crop_manifest["body_sha256"] = _body_hash(crop_manifest)
    _atomic_json(run_dir / "crop-manifest.json", crop_manifest)
    return config, crops


def _load_or_prepare(
    public_manifest_path: Path,
    provider_lock_path: Path,
    repo_dir: Path,
    model_dir: Path,
    run_dir: Path,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if not run_dir.exists():
        _require(not resume, "cannot resume a PDM run that does not exist")
        return _prepare_run_config(
            public_manifest_path, provider_lock_path, repo_dir, model_dir, run_dir
        )
    _require(resume, f"PDM run directory already exists: {run_dir}")
    _require(not (run_dir / "final-report.json").exists(), "sealed PDM run cannot resume")
    config = _load_json(run_dir / "run-config.json")
    crop_manifest = _load_json(run_dir / "crop-manifest.json")
    _verify_body_hash(config, "PDM run config")
    _verify_body_hash(crop_manifest, "PDM crop manifest")
    _require(crop_manifest["run_config_body_sha256"] == config["body_sha256"], "PDM crop/config binding drifted")
    lock = _load_json(provider_lock_path)
    _verify_provider_lock(lock, repo_dir, model_dir)
    _require(config["provider_lock"]["body_sha256"] == lock["body_sha256"], "PDM provider changed on resume")
    return config, crop_manifest["crops"]


def _checkpoint(run_dir: Path, config: Mapping[str, Any], stage: str, completed: int, total: int) -> None:
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "stage": stage,
        "completed": completed,
        "total": total,
        "next_index": completed,
    }
    checkpoint["body_sha256"] = _body_hash(checkpoint)
    _atomic_json(run_dir / "checkpoint.json", checkpoint)


def execute(
    public_manifest_path: Path,
    provider_lock_path: Path,
    repo_dir: Path,
    model_dir: Path,
    run_dir: Path,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    _require(device == "cuda" and torch.cuda.is_available(), "PDM requires the frozen CUDA runtime")
    config, crops = _load_or_prepare(
        public_manifest_path, provider_lock_path, repo_dir, model_dir, run_dir, resume
    )
    dift_module, pipeline_module, ptp_utils, _ = _install_provider_imports(repo_dir)
    dtype = torch.float16
    crop_items = sorted(crops.items())
    dift_dir = run_dir / "features" / "dift"
    attention_dir = run_dir / "features" / "attention"
    try:
        missing_dift = [(crop_id, item) for crop_id, item in crop_items if not (dift_dir / f"{crop_id}.pt").exists()]
        if missing_dift:
            pipe = _load_dift_pipeline(model_dir, dift_module, dtype)
            for index, (crop_id, item) in enumerate(missing_dift, start=1):
                feature = _extract_dift(pipe, _crop_image(item))
                _atomic_torch_save(feature, dift_dir / f"{crop_id}.pt")
                _checkpoint(run_dir, config, "DIFT", len(crop_items) - len(missing_dift) + index, len(crop_items))
            _release_cuda(pipe)

        missing_attention = [
            (crop_id, item)
            for crop_id, item in crop_items
            if not (attention_dir / f"{crop_id}.pt").exists()
        ]
        if missing_attention:
            pipe = _load_attention_pipeline(model_dir, pipeline_module, dtype)
            for index, (crop_id, item) in enumerate(missing_attention, start=1):
                feature = _extract_attention(pipe, ptp_utils, _crop_image(item))
                _atomic_torch_save(feature, attention_dir / f"{crop_id}.pt")
                _checkpoint(
                    run_dir,
                    config,
                    "PDM_ATTENTION",
                    len(crop_items) - len(missing_attention) + index,
                    len(crop_items),
                )
            _release_cuda(pipe)

        cache = {
            crop_id: {
                "dift": torch.load(dift_dir / f"{crop_id}.pt", map_location="cpu", weights_only=True),
                "attention": torch.load(
                    attention_dir / f"{crop_id}.pt", map_location="cpu", weights_only=True
                ),
            }
            for crop_id, _ in crop_items
        }
        reference_by_case = {}
        for pair in config["pairs"]:
            reference_by_case[pair["case_id"]] = crops[pair["reference_crop_id"]]
        for absence in config["absences"]:
            reference_by_case[absence["case_id"]] = crops[absence["reference_crop_id"]]
        masks = {case_id: _reference_mask64(reference) for case_id, reference in reference_by_case.items()}

        pair_rows = []
        for pair in config["pairs"]:
            reference = cache[pair["reference_crop_id"]]
            scores = {}
            for slot in ("A", "B"):
                candidate = cache[pair["candidate_crop_ids"][slot]]
                scores[slot] = _pdm_score(
                    dift_module,
                    reference["dift"],
                    candidate["dift"],
                    reference["attention"],
                    candidate["attention"],
                    masks[pair["case_id"]],
                )
            pair_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "case_id": pair["case_id"],
                    "candidate_scores": scores,
                    "winner_slot": (
                        "A" if scores["A"] > scores["B"] else "B" if scores["B"] > scores["A"] else "TIE"
                    ),
                    "slot_margin_a_minus_b": scores["A"] - scores["B"],
                }
            )
        absence_rows = []
        for absence in config["absences"]:
            reference = cache[absence["reference_crop_id"]]
            candidate = cache[absence["candidate_crop_id"]]
            absence_rows.append(
                {
                    "absence_id": absence["absence_id"],
                    "case_id": absence["case_id"],
                    "candidate_score": _pdm_score(
                        dift_module,
                        reference["dift"],
                        candidate["dift"],
                        reference["attention"],
                        candidate["attention"],
                        masks[absence["case_id"]],
                    ),
                }
            )
        raw = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": probe.PROTOCOL_ID,
            "stage": "PDM_CHALLENGER",
            "created_at_utc": _utc_now(),
            "run_config_body_sha256": config["body_sha256"],
            "crop_count": len(crops),
            "pairs": pair_rows,
            "absences": absence_rows,
            "claim_ceiling": probe.CLAIM_CEILING,
        }
        raw["body_sha256"] = _body_hash(raw)
        _atomic_json(run_dir / "raw-scores.json", raw)
        _checkpoint(run_dir, config, "RAW_SCORES_COMPLETE", len(crops), len(crops))
        return raw
    except Exception as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": probe.PROTOCOL_ID,
            "created_at_utc": _utc_now(),
            "run_config_body_sha256": config.get("body_sha256"),
            "error_class": type(error).__name__,
            "error": str(error),
            "terminal": "PDM_CHALLENGER_NOT_EVALUABLE_RUNTIME",
            "claim_ceiling": probe.CLAIM_CEILING,
        }
        failure["body_sha256"] = _body_hash(failure)
        _atomic_json(run_dir / "failure-report.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock-provider")
    lock.add_argument("--repo-dir", type=Path, required=True)
    lock.add_argument("--model-dir", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--public-manifest", type=Path, required=True)
    run.add_argument("--provider-lock", type=Path, required=True)
    run.add_argument("--repo-dir", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "lock-provider":
        result = lock_provider(args.repo_dir, args.model_dir, args.output)
    elif args.command == "run":
        result = execute(
            args.public_manifest,
            args.provider_lock,
            args.repo_dir,
            args.model_dir,
            args.run_dir,
            args.device,
            args.resume,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
