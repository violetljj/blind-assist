"""Fixed 96-image Development OCR observation run; never reads field labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import urllib.request


MODELS = {
    "det": ("ch_PP-OCRv5_det_server.onnx", "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad"),
    "rec": ("ch_PP-OCRv5_rec_server.onnx", "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wall-budget-s", type=float, default=1100)
    args = ap.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "predictions.jsonl"
    if result_path.exists():
        raise RuntimeError("Refusing to overwrite an existing observation run")
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    downloads = []
    downloaded_bytes = 0
    for stage, (name, expected) in MODELS.items():
        model = model_dir / name
        url = f"https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/{stage}/{name}"
        t0 = time.perf_counter()
        cached = model.exists() and sha(model) == expected
        if not cached:
            req = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-Development-OCR/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response, model.open("wb") as f:
                length = int(response.headers.get("Content-Length", "0"))
                if downloaded_bytes + length > 300_000_000:
                    raise RuntimeError("Model response exceeds 300 MB download budget")
                while chunk := response.read(1024 * 1024):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > 300_000_000:
                        raise RuntimeError("Model stream exceeds 300 MB download budget")
                    f.write(chunk)
        actual = sha(model)
        if actual != expected:
            raise RuntimeError(f"Model hash mismatch: {name}: {actual}")
        downloads.append({"stage": stage, "path": str(model.resolve()), "url": url,
                          "bytes": model.stat().st_size, "sha256": actual,
                          "already_cached": cached, "wall_s": time.perf_counter() - t0})
        print(json.dumps({"event": "model_ready", **downloads[-1]}), flush=True)
        write_json(args.output / "download_receipts.json", downloads)

    import cv2
    import numpy as np
    import onnxruntime as ort
    import rapidocr
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    from omegaconf import OmegaConf

    cls_source = Path(rapidocr.__file__).parent / "models" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    cls_path = model_dir / cls_source.name
    shutil.copyfile(cls_source, cls_path)
    cv2.setNumThreads(1)
    params = {
        "Global.model_root_dir": str(model_dir.resolve()),
        "Global.text_score": 0.0,  # Preserve low-confidence lines; default-accepted is marked separately.
        "Global.log_level": "warning",
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.SERVER,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str((model_dir / MODELS["det"][0]).resolve()),
        # Current RapidOCR defaults target v6. v5's own official normalization is explicit.
        "Det.mean": [0.485, 0.456, 0.406],
        "Det.std": [0.229, 0.224, 0.225],
        "Cls.model_path": str(cls_path.resolve()),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": str((model_dir / MODELS["rec"][0]).resolve()),
        "EngineConfig.onnxruntime.intra_op_num_threads": args.threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
    }
    t0 = time.perf_counter()
    engine = RapidOCR(params=params)
    initialization_s = time.perf_counter() - t0
    source = args.input / "image_receipts.json"
    inputs = json.loads(source.read_text(encoding="utf-8"))
    if len(inputs) != 96:
        raise RuntimeError(f"Expected existing fixed 96 images; got {len(inputs)}")
    providers = {stage: getattr(engine, "text_" + stage).session.session.get_providers()
                 for stage in ["det", "cls", "rec"]}
    if any(v != ["CPUExecutionProvider"] for v in providers.values()):
        raise RuntimeError(f"Unexpected execution providers: {providers}")
    manifest = {
        "evidence_class": "EXPLORE Development reuse; OCR observations only",
        "input_manifest": str(source.resolve()), "input_manifest_sha256": sha(source),
        "cohort_count": len(inputs), "input_order": "existing image_receipts.json unchanged",
        "source_sha256": sha(Path(__file__)), "source_path": str(Path(__file__).resolve()),
        "versions": {name: importlib.metadata.version(name) for name in ["rapidocr", "onnxruntime", "numpy"]},
        "python": sys.version, "platform": platform.platform(), "pid": os.getpid(),
        "backend": "ONNX Runtime CPU", "providers": providers,
        "cpu_placement_reason": "ACCELERATOR_UNAVAILABLE: primary GPU reserved for concurrent VLM baseline",
        "normalization_source": "https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml",
        "model_route_source": "https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/how_to_use_ppocrv5/",
        "config": OmegaConf.to_container(engine.cfg, resolve=True, enum_to_str=True),
        "models": downloads + [{"stage": "cls", "path": str(cls_path.resolve()), "bytes": cls_path.stat().st_size, "sha256": sha(cls_path), "source": str(cls_source)}],
        "downloaded_bytes": downloaded_bytes, "initialization_s": initialization_s,
        "output_coordinates": "original image pixel coordinates; quadrilateral in RapidOCR returned order",
        "text_score_policy": "retain all nonempty recognition lines (text_score=0); accepted_at_default marks score >= 0.5",
        "inference_labels_access": "no product annotation, question, field target, or evaluator truth is loaded",
        "wall_budget_s": args.wall_budget_s,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"event": "ready", "providers": providers, "init_s": initialization_s}), flush=True)
    rows = []
    with result_path.open("x", encoding="utf-8") as f:
        for index, receipt in enumerate(inputs):
            if time.perf_counter() - started >= args.wall_budget_s:
                break
            image_path = args.input / "images" / receipt["image"]
            t0 = time.perf_counter()
            row = {"index": index, "image": receipt["image"], "source_sha256": receipt["sha256"],
                   "status": "pending", "lines": [], "error": None}
            try:
                actual = sha(image_path)
                if actual != receipt["sha256"]:
                    raise RuntimeError("Source image differs from original image receipt")
                output = engine(str(image_path))
                if output.img is not None:
                    row["height"], row["width"] = map(int, output.img.shape[:2])
                if output.txts is not None:
                    for box, txt, score in zip(output.boxes, output.txts, output.scores):
                        row["lines"].append({"text": str(txt), "confidence": float(score),
                                             "accepted_at_default": float(score) >= 0.5,
                                             "box": np.asarray(box).tolist()})
                row["status"] = "ok" if row["lines"] else "no_text"
                row["text"] = "\n".join(line["text"] for line in row["lines"] if line["accepted_at_default"])
                row["stage_elapsed_s"] = output.elapse_list
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
            row["wall_s"] = time.perf_counter() - t0
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            progress = {"completed": len(rows), "total": len(inputs), "last_image": receipt["image"],
                        "last_status": row["status"], "wall_s": time.perf_counter() - started,
                        "last_inference_s": row["wall_s"]}
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    times = [r["wall_s"] for r in rows]
    terminal = {
        "status": "complete" if len(rows) == len(inputs) else "wall_budget_stop",
        "completed": len(rows), "total": len(inputs), "errors": sum(r["status"] == "error" for r in rows),
        "no_text": sum(r["status"] == "no_text" for r in rows),
        "retained_lines": sum(len(r["lines"]) for r in rows),
        "accepted_lines_default": sum(sum(x["accepted_at_default"] for x in r["lines"]) for r in rows),
        "per_image_median_s": float(np.median(times)), "per_image_p95_s": float(np.percentile(times, 95)),
        "inference_total_s": sum(times), "wall_s": time.perf_counter() - started,
        "predictions_sha256": sha(result_path), "downloaded_bytes": downloaded_bytes,
    }
    write_json(args.output / "terminal.json", terminal)
    print(json.dumps(terminal), flush=True)


if __name__ == "__main__":
    main()
