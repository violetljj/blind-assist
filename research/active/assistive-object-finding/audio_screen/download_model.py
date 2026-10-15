#!/usr/bin/env python3
"""Download one public CLAP snapshot with a fixed revision and byte receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / "hf-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    started = time.perf_counter()
    revision = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
    info = HfApi(token=False).model_info("laion/clap-htsat-unfused", revision=revision,
                                      files_metadata=True, token=False)
    if info.sha != revision:
        raise ValueError("Model metadata revision drift; preserve request and inspect")
    files = [{"name": f.rfilename, "bytes": f.size} for f in info.siblings if f.rfilename != ".gitattributes"]
    metadata = {"repo_id": info.id, "revision": info.sha, "gated": info.gated,
                "license": info.card_data.get("license") if info.card_data else None,
                "source": "https://huggingface.co/api/models/laion/clap-htsat-unfused",
                "files": files, "planned_payload_bytes": sum(f["bytes"] for f in files)}
    if metadata["planned_payload_bytes"] > 1_500_000_000:
        raise ValueError("Requested model payload exceeds 1.5GB budget")
    (root / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model_root = root / "models" / "clap-htsat-unfused"
    existing = {f["name"] for f in files if (model_root / f["name"]).exists()}
    snapshot_download(info.id, revision=revision, local_dir=model_root,
                      allow_patterns=[f["name"] for f in files], max_workers=3, token=False)
    receipts = []
    for f in files:
        path = model_root / f["name"]
        if path.stat().st_size != f["bytes"]:
            raise ValueError(f"Size mismatch {path.name}")
        receipts.append({**f, "sha256": digest(path)})
    result = {**metadata, "status": "SUCCESS", "files": receipts,
              "new_file_payload_bytes": sum(f["bytes"] for f in files if f["name"] not in existing),
              "elapsed_s": time.perf_counter() - started, "model_dir": str(model_root)}
    (root / "model_receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "revision", "new_file_payload_bytes", "elapsed_s", "model_dir")}))


if __name__ == "__main__":
    main()
