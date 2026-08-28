"""Materialize an exact, outcome-blind Wikimedia Commons active-view source."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


USER_AGENT = "BlindAssist-L10-ActiveViews/1.0 (research prototype)"
API = "https://commons.wikimedia.org/w/api.php"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _api(parameters: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({**parameters, "format": "json"})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, local: Path) -> None:
    for attempt, delay in enumerate((0, 2, 6, 15), 1):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, local.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            return
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise


def materialize(protocol_path: Path, output: Path) -> Path:
    protocol = _json(protocol_path)
    filenames = [str(name) for name in protocol["files"]]
    if not filenames or len(set(filenames)) != len(filenames):
        raise ValueError("INVALID_OR_DUPLICATE_SOURCE_FILES")
    manifest_path = output / "source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"SOURCE_ALREADY_MATERIALIZED:{manifest_path}")
    images = output / "images"
    images.mkdir(parents=True, exist_ok=False)
    pages: dict[str, Any] = {}
    for start in range(0, len(filenames), 50):
        payload = _api(
            {
                "action": "query",
                "prop": "imageinfo",
                "titles": "|".join(f"File:{name}" for name in filenames[start : start + 50]),
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": "1280",
            }
        )
        for page in payload["query"]["pages"].values():
            pages[page["title"].removeprefix("File:")] = page
    missing = [name for name in filenames if name not in pages or not pages[name].get("imageinfo")]
    if missing:
        raise ValueError(f"COMMONS_FILES_MISSING:{missing}")
    rows = []
    for index, filename in enumerate(filenames, 1):
        info = pages[filename]["imageinfo"][0]
        if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"UNSUPPORTED_MIME:{filename}:{info.get('mime')}")
        suffix = Path(urllib.parse.urlparse(info["url"]).path).suffix.lower() or ".jpg"
        local = images / f"{index:02d}{suffix}"
        download_url = info.get("thumburl") or info["url"]
        _download(download_url, local)
        rows.append(
            {
                "index": index,
                "commons_file": filename,
                "source_page": "https://commons.wikimedia.org/wiki/File:"
                + urllib.parse.quote(filename.replace(" ", "_")),
                "download_url": download_url,
                "mime": info["mime"],
                "width": int(info["width"]),
                "height": int(info["height"]),
                "bytes": local.stat().st_size,
                "sha256": _sha256(local),
                "local_path": str(local.resolve()),
            }
        )
    inventory_digest = hashlib.sha256(
        "".join(f"{row['index']}\t{row['bytes']}\t{row['sha256']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "l10-named-poi-active-view-source-manifest-v1",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "target_id": protocol["target_id"],
        "selection_authority": protocol["selection_authority"],
        "image_count": len(rows),
        "image_bytes": sum(row["bytes"] for row in rows),
        "inventory_sha256": inventory_digest,
        "frames": rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("named_poi_active_view_source_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.protocol, args.output))


if __name__ == "__main__":
    main()
