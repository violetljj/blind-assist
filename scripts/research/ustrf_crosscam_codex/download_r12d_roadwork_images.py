#!/usr/bin/env python3
"""Range-extract only the preregistered ROADWork image inventory from its remote ZIP."""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from prepare_r12d_training_dataset import select_roadwork_image_ids
from r12d_contract import require, sha256_bytes, sha256_file, validate_matrix, write_json


REMOTE_IMAGES_ZIP = "https://huggingface.co/datasets/anuragxel/roadwork-dataset/resolve/main/images.zip"


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, block_size: int = 8 * 1024 * 1024, maximum_blocks: int = 4):
        self.session = requests.Session()
        response = self.session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.url = response.url
        self.size = int(response.headers["content-length"])
        self.etag = response.headers.get("etag")
        self.block_size = block_size
        self.maximum_blocks = maximum_blocks
        self.position = 0
        self.cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        output = bytearray()
        while size > 0 and self.position < self.size:
            block_index = self.position // self.block_size
            block_offset = self.position % self.block_size
            if block_index not in self.cache:
                start = block_index * self.block_size
                end = min(self.size - 1, start + self.block_size - 1)
                response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                response.raise_for_status()
                self.cache[block_index] = response.content
                while len(self.cache) > self.maximum_blocks:
                    self.cache.popitem(last=False)
            block = self.cache[block_index]
            count = min(size, len(block) - block_offset)
            output.extend(block[block_offset:block_offset + count])
            self.position += count
            size -= count
        return bytes(output)


thread_local = threading.local()


def session() -> requests.Session:
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            response = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
            response.raise_for_status()
            expected = end - start + 1
            require(len(response.content) == expected, f"range length mismatch: {len(response.content)}/{expected}")
            return response.content
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def extract_member(url: str, info: zipfile.ZipInfo, output_root: Path) -> dict[str, Any]:
    destination = output_root / info.filename
    if destination.is_file() and destination.stat().st_size == info.file_size:
        crc = 0
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                crc = zlib.crc32(chunk, crc)
        if crc & 0xFFFFFFFF == info.CRC:
            return {"filename": info.filename, "size": info.file_size, "crc32": f"{info.CRC:08x}", "resumed": True}
    header = fetch_range(url, info.header_offset, info.header_offset + 29)
    values = struct.unpack("<IHHHHHIIIHH", header)
    require(values[0] == 0x04034B50, f"invalid local ZIP header: {info.filename}")
    name_length, extra_length = values[-2], values[-1]
    data_start = info.header_offset + 30 + name_length + extra_length
    compressed = fetch_range(url, data_start, data_start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type}: {info.filename}")
    require(len(payload) == info.file_size, f"uncompressed size mismatch: {info.filename}")
    require(zlib.crc32(payload) & 0xFFFFFFFF == info.CRC, f"CRC mismatch: {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return {"filename": info.filename, "size": info.file_size, "crc32": f"{info.CRC:08x}", "resumed": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve(); matrix_path = args.matrix.resolve(); matrix = validate_matrix(matrix_path, repo)
    annotations = args.annotations.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    selection = matrix["data"]["roadwork"]["selection"]
    names = set()
    split_counts = {}
    for split, filename in (("train", matrix["data"]["roadwork"]["train_annotation"]),
                            ("val", matrix["data"]["roadwork"]["validation_annotation"])):
        document = json.loads((annotations / filename).read_text(encoding="utf-8"))
        images = {row["id"]: row for row in document["images"]}
        selected = select_roadwork_image_ids(document, split, selection)
        split_counts[split] = len(selected)
        names.update(f"images/{images[image_id]['file_name']}" for image_id in selected)
    reader = RangeReader(args.url)
    with zipfile.ZipFile(reader) as archive:
        inventory = {row.filename: row for row in archive.infolist()}
    missing = sorted(names - set(inventory))
    require(not missing, f"selected images absent from remote ZIP: {missing[:5]}")
    selected_info = [inventory[name] for name in sorted(names)]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_member, reader.url, info, output): info.filename for info in selected_info}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"R12D_ROADWORK_DOWNLOAD {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["filename"])
    receipt = {
        "schema": "blindassist_ustrf_r12d_roadwork_range_download_receipt_v1",
        "matrix_sha256": sha256_file(matrix_path), "remote_url": args.url,
        "remote_content_length": reader.size, "remote_etag": reader.etag,
        "selection": selection, "split_counts": split_counts, "selected_unique_count": len(results),
        "selected_uncompressed_bytes": sum(row["size"] for row in results),
        "selected_inventory_sha256": sha256_bytes((f"{row['filename']}\0{row['size']}\0{row['crc32']}\n".encode("utf-8") for row in results)),
        "resumed_file_count": sum(row["resumed"] for row in results),
        "files": results,
    }
    write_json(output / "range_download_receipt.json", receipt)
    print("USTRF_R12D_ROADWORK_DOWNLOAD_OK", len(results), receipt["selected_uncompressed_bytes"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=REMOTE_IMAGES_ZIP)
    parser.add_argument("--workers", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
