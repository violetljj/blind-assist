"""Fixed English re-recognition of native-resolution PP-OCRv5 server boxes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
import urllib.request

from run_ocr import sha, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--server-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wall-budget-s", type=float, default=300)
    args = ap.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    outpath = args.output / "predictions.jsonl"
    if outpath.exists():
        raise RuntimeError("Refusing to overwrite prior English observations")
    terminal = json.loads((args.server_output / "terminal.json").read_text(encoding="utf-8"))
    if terminal["status"] != "complete" or terminal["completed"] != 96:
        raise RuntimeError("Main fixed cohort must be complete before decoder comparison")
    model = args.output / "en_PP-OCRv5_rec_mobile.onnx"
    url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx"
    expected = "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8"
    cached = model.exists() and sha(model) == expected
    downloaded = 0
    if not cached:
        with urllib.request.urlopen(url, timeout=60) as response, model.open("wb") as f:
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded + terminal["downloaded_bytes"] > 300_000_000:
                    raise RuntimeError("Combined OCR downloads exceed 300 MB budget")
                f.write(chunk)
    if sha(model) != expected:
        raise RuntimeError("English model checksum mismatch")
    download_receipt_path = args.output / "download_receipt.json"
    if not download_receipt_path.exists():
        write_json(download_receipt_path, {"url": url, "sha256": expected,
                   "bytes": model.stat().st_size, "downloaded_bytes": downloaded,
                   "already_cached": cached})
    download_receipt = json.loads(download_receipt_path.read_text(encoding="utf-8"))
    import cv2
    import numpy as np
    from omegaconf import OmegaConf
    from rapidocr import EngineType, LangRec, ModelType
    from rapidocr.ch_ppocr_cls import TextClassifier
    from rapidocr.ch_ppocr_rec import TextRecInput, TextRecognizer
    from rapidocr.utils.parse_parameters import ParseParams
    from rapidocr.utils.process_img import get_rotate_crop_image
    from rapidocr.utils.typings import LangCls, LangDet, OCRVersion, TaskType
    cv2.setNumThreads(1)
    server_manifest_path = args.server_output / "manifest.json"
    server_manifest = json.loads(server_manifest_path.read_text(encoding="utf-8"))
    confpath = args.output / "inherited_config.yaml"
    inherited = server_manifest["config"]
    # OmegaConf.to_container(enum_to_str=True) stores enum names (ONNXRUNTIME),
    # while RapidOCR's config loader accepts enum values (onnxruntime).
    for section, lang_enum in [("Det", LangDet), ("Cls", LangCls), ("Rec", LangRec)]:
        for key, enum_type in [("engine_type", EngineType), ("model_type", ModelType),
                               ("ocr_version", OCRVersion), ("task_type", TaskType),
                               ("lang_type", lang_enum)]:
            value = inherited[section][key]
            if value in enum_type.__members__:
                inherited[section][key] = enum_type[value].value
    OmegaConf.save(OmegaConf.create(inherited), confpath)
    cfg = ParseParams.load(confpath)
    cfg.Rec.lang_type = LangRec.EN
    cfg.Rec.model_type = ModelType.MOBILE
    cfg.Rec.model_path = str(model.resolve())
    cfg.Rec.engine_cfg = cfg.EngineConfig.onnxruntime
    cfg.Rec.model_root_dir = str(args.output.resolve())
    cfg.Rec.font_path = None
    cfg.Cls.engine_cfg = cfg.EngineConfig.onnxruntime
    cfg.Cls.model_root_dir = str(args.output.resolve())
    recognizer = TextRecognizer(cfg.Rec)
    classifier = TextClassifier(cfg.Cls)
    providers = {"rec": recognizer.session.session.get_providers(), "cls": classifier.session.session.get_providers()}
    if any(x != ["CPUExecutionProvider"] for x in providers.values()):
        raise RuntimeError(f"Unexpected provider: {providers}")
    primary_path = args.server_output / "predictions.jsonl"
    primary = [json.loads(line) for line in primary_path.read_text(encoding="utf-8").splitlines()]
    manifest = {
        "evidence_class": "EXPLORE Development; fixed second decoder, no field truth",
        "source_sha256": sha(Path(__file__)), "helper_sha256": sha(Path(__file__).with_name("run_ocr.py")),
        "model_url": url, "model_sha256": expected, "model_bytes": model.stat().st_size,
        "downloaded_bytes": download_receipt["downloaded_bytes"], "already_cached": cached,
        "parent_predictions_sha256": sha(primary_path), "parent_manifest_sha256": sha(server_manifest_path),
        "providers": providers, "config": OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        "crop_policy": "Warp original-resolution images using server-returned nonempty quadrilaterals; reapply same 0/180 line classifier. Main server rec uses preprocessing-sized crops, so decoder plus native-resolution crop agreement is a compound signal, not an isolated architecture contrast.",
        "scope": "No new detector calls or boxes, no labels or field values loaded",
        "wall_budget_s": args.wall_budget_s,
        "source": "https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html",
    }
    write_json(args.output / "manifest.json", manifest)
    rows = []
    with outpath.open("x", encoding="utf-8") as f:
        for source in primary:
            if time.perf_counter() - started >= args.wall_budget_s:
                break
            t0 = time.perf_counter()
            row = {"index": source["index"], "image": source["image"], "source_sha256": source["source_sha256"],
                   "lines": [], "status": "pending", "error": None}
            try:
                path = args.input / "images" / row["image"]
                if sha(path) != row["source_sha256"]:
                    raise RuntimeError("Input hash mismatch")
                img = cv2.imread(str(path))
                if img is None:
                    raise RuntimeError("Input decode failed")
                row["height"], row["width"] = map(int, img.shape[:2])
                if source["status"] == "error":
                    row["status"] = "upstream_error"
                    row["error"] = source["error"]
                elif source["lines"]:
                    crops = [get_rotate_crop_image(img, np.asarray(line["box"], dtype=np.float32)) for line in source["lines"]]
                    cls = classifier(crops)
                    rec = recognizer(TextRecInput(img=cls.img_list))
                    if len(rec.txts) != len(source["lines"]):
                        raise RuntimeError("Decoder changed the shared crop count")
                    for i, (txt, score, original) in enumerate(zip(rec.txts, rec.scores, source["lines"])):
                        row["lines"].append({"line_index": i, "box": original["box"], "text": str(txt),
                                             "confidence": float(score), "accepted_at_default": bool(txt.strip()) and float(score) >= 0.5})
                    row["status"] = "ok"
                else:
                    row["status"] = "no_text"
                row["text"] = "\n".join(x["text"] for x in row["lines"] if x["accepted_at_default"])
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
            row["wall_s"] = time.perf_counter() - t0
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            progress = {"completed": len(rows), "total": 96, "last_image": row["image"], "last_status": row["status"], "wall_s": time.perf_counter() - started}
            write_json(args.output / "progress.json", progress)
            if len(rows) % 12 == 0 or row["status"] == "error":
                print(json.dumps(progress), flush=True)
    times = [r["wall_s"] for r in rows]
    summary = {"status": "complete" if len(rows) == 96 else "wall_budget_stop", "completed": len(rows), "total": 96,
               "errors": sum(r["status"] in ["error", "upstream_error"] for r in rows), "no_text": sum(r["status"] == "no_text" for r in rows),
               "retained_lines": sum(len(r["lines"]) for r in rows), "accepted_lines_default": sum(sum(l["accepted_at_default"] for l in r["lines"]) for r in rows),
               "wall_s": time.perf_counter() - started, "inference_total_s": sum(times),
               "per_image_median_s": float(np.median(times)), "per_image_p95_s": float(np.percentile(times, 95)),
               "downloaded_bytes": download_receipt["downloaded_bytes"], "predictions_sha256": sha(outpath)}
    write_json(args.output / "terminal.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
