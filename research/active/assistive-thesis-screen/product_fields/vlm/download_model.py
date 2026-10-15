"""Download pinned public model files anonymously; retain partials and receipts."""
import argparse
import hashlib
import json
import time
import requests
from pathlib import Path

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    info_path = root / "source_model_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        response = requests.get(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true", timeout=60)
        response.raise_for_status()
        info = response.json()
    (root / "source_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    wanted = [row for row in info["siblings"] if row["rfilename"].endswith((".json", ".txt", ".safetensors", ".md"))]
    assert sum(row["size"] for row in wanted) < 6_000_000_000
    receipts = []
    for row in wanted:
        name = row["rfilename"]
        target = root / name
        part = root / (name + ".part")
        # The official card links ModelScope. Hashes must still match pinned HF.
        url = f"https://modelscope.cn/models/{MODEL}/resolve/master/{name}"
        file_started = time.monotonic()
        if not target.exists():
            print(json.dumps({"stage": "download", "file": name, "total": row["size"]}), flush=True)
            offset = part.stat().st_size if part.exists() else 0
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, timeout=(30, 90), stream=True) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("Server did not honor range; partial preserved")
                with part.open("ab" if offset else "wb") as out:
                    last = time.monotonic()
                    for block in response.iter_content(8 * 1024 * 1024):
                        out.write(block)
                        offset += len(block)
                        if time.monotonic() - last > 15:
                            print(json.dumps({"stage": "download", "file": name, "bytes": offset, "total": row["size"]}), flush=True)
                            last = time.monotonic()
            if part.stat().st_size != row["size"]:
                raise RuntimeError("Downloaded size mismatch")
            part.rename(target)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if "lfs" in row:
            assert digest == row["lfs"]["sha256"]
        else:
            raw = target.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            assert git_blob == row["blobId"], f"Pinned source Git blob mismatch: {name}"
        receipts.append({"name": name, "bytes": target.stat().st_size, "sha256": digest, "source": url, "wall_s": time.monotonic() - file_started})
        print(json.dumps(receipts[-1]), flush=True)
    (root / "download_receipt.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "files": receipts, "total_bytes": sum(r["bytes"] for r in receipts), "wall_s": time.monotonic() - started, "credentials": "none; requests anonymous HTTP"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
