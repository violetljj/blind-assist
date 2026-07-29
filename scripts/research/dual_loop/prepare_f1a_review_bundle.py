#!/usr/bin/env python3
"""Build and hash the candidate-blind visual review bundle for dual-loop F-1A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def require_hash(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch: {relative}: {actual} != {expected}")
    return path


def load_frame_ledger(root: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    ledger_path = require_hash(root, item["frames_path"], item["frames_sha256"])
    require_hash(root, item["bundle_path"], item["bundle_sha256"])
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != item["expected_frame_count"]:
        raise ValueError(
            f"{item['input_id']}: frame count {len(rows)} "
            f"!= {item['expected_frame_count']}"
        )
    timestamps = [int(row["source_capture_timestamp_ns"]) for row in rows]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{item['input_id']}: non-increasing timestamp")
    if any(row["source_id"] != item["source_id"] for row in rows):
        raise ValueError(f"{item['input_id']}: source identity drift")
    if any(row["sequence_id"] != item["session_id"] for row in rows):
        raise ValueError(f"{item['input_id']}: session identity drift")
    base = ledger_path.parent
    for row in rows:
        rgb = base / row["rgb_path"]
        if not rgb.is_file():
            raise FileNotFoundError(rgb)
        if sha256_file(rgb) != row["rgb_sha256"]:
            raise ValueError(f"{item['input_id']}: RGB hash mismatch: {rgb}")
    return rows


def sample_ledger(
    root: Path, item: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ledger_path = root / item["frames_path"]
    base = ledger_path.parent
    period_ns = int(float(item["review_sample_period_seconds"]) * 1_000_000_000)
    first_ns = int(rows[0]["source_capture_timestamp_ns"])
    next_ns = first_ns
    samples: list[dict[str, Any]] = []
    for row in rows:
        timestamp_ns = int(row["source_capture_timestamp_ns"])
        if timestamp_ns < next_ns:
            continue
        rgb = base / row["rgb_path"]
        samples.append(
            {
                "input_id": item["input_id"],
                "frame_id": row["frame_id"],
                "timestamp_ns": timestamp_ns,
                "relative_seconds": round((timestamp_ns - first_ns) / 1e9, 6),
                "source_path": str(rgb.resolve()),
                "source_sha256": row["rgb_sha256"],
            }
        )
        next_ns = timestamp_ns + period_ns
    return samples


def probe_video(path: Path, ffprobe: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def sample_video(
    item: dict[str, Any],
    video_path: Path,
    output_dir: Path,
    ffmpeg: Path,
) -> list[dict[str, Any]]:
    sample_dir = output_dir / f"{item['input_id']}_samples"
    sample_dir.mkdir()
    period = float(item["review_sample_period_seconds"])
    output_pattern = sample_dir / "%06d.jpg"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{period}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    paths = sorted(sample_dir.glob("*.jpg"))
    return [
        {
            "input_id": item["input_id"],
            "frame_id": path.stem,
            "timestamp_ns": int(index * period * 1e9),
            "relative_seconds": round(index * period, 6),
            "source_path": str(path.resolve()),
            "source_sha256": sha256_file(path),
        }
        for index, path in enumerate(paths)
    ]


def render_contact_sheets(
    samples: list[dict[str, Any]], output_dir: Path, input_id: str
) -> list[dict[str, Any]]:
    columns = 5
    rows = 4
    tile_width = 320
    tile_height = 240
    label_height = 24
    margin = 4
    page_size = columns * rows
    font = ImageFont.load_default()
    sheets: list[dict[str, Any]] = []
    for page_index in range(math.ceil(len(samples) / page_size)):
        page = samples[page_index * page_size : (page_index + 1) * page_size]
        canvas = Image.new(
            "RGB",
            (
                columns * tile_width + (columns + 1) * margin,
                rows * (tile_height + label_height) + (rows + 1) * margin,
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for index, sample in enumerate(page):
            row = index // columns
            column = index % columns
            x = margin + column * tile_width
            y = margin + row * (tile_height + label_height)
            with Image.open(sample["source_path"]) as image:
                rgb = image.convert("RGB")
                rgb.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                image_x = x + (tile_width - rgb.width) // 2
                image_y = y + (tile_height - rgb.height) // 2
                canvas.paste(rgb, (image_x, image_y))
            label = (
                f"{sample['frame_id']}  t={sample['relative_seconds']:.3f}s"
            )
            draw.rectangle(
                (x, y + tile_height, x + tile_width, y + tile_height + label_height),
                fill="white",
            )
            draw.text(
                (x + 3, y + tile_height + 5), label, fill="black", font=font
            )
        path = output_dir / f"{input_id}_contact_{page_index + 1:03d}.jpg"
        canvas.save(path, format="JPEG", quality=90)
        sheets.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "sample_start": page_index * page_size,
                "sample_end_exclusive": page_index * page_size + len(page),
            }
        )
    return sheets


def build_bundle(
    root: Path,
    spec_path: Path,
    output_dir: Path,
    ffmpeg: Path,
    ffprobe: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"formal output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    require_hash(root, spec["contract"]["path"], spec["contract"]["sha256"])
    require_hash(
        root,
        spec["predecessor_result"]["path"],
        spec["predecessor_result"]["sha256"],
    )
    if spec["candidate_output_visibility"] is not False:
        raise ValueError("candidate output visibility must be false")

    prompt_sha256 = hashlib.sha256(spec["review_prompt"].encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "schema": "blindassist_dual_loop_f1a_review_bundle_v1",
        "protocol_id": spec["protocol_id"],
        "spec_path": str(spec_path.relative_to(root)).replace("\\", "/"),
        "spec_sha256": sha256_file(spec_path),
        "prompt_sha256": prompt_sha256,
        "candidate_output_visibility": False,
        "inputs": [],
    }
    for item in spec["fixed_input_universe"]:
        if item["kind"] == "frame_ledger":
            rows = load_frame_ledger(root, item)
            samples = sample_ledger(root, item, rows)
            evidence = {
                "frame_count": len(rows),
                "first_timestamp_ns": int(rows[0]["source_capture_timestamp_ns"]),
                "last_timestamp_ns": int(rows[-1]["source_capture_timestamp_ns"]),
                "frames_sha256": item["frames_sha256"],
                "bundle_sha256": item["bundle_sha256"],
            }
        elif item["kind"] == "video":
            video_path = require_hash(root, item["video_path"], item["video_sha256"])
            probe = probe_video(video_path, ffprobe)
            duration = float(probe["format"]["duration"])
            if abs(duration - float(item["expected_duration_seconds"])) > 0.05:
                raise ValueError(f"{item['input_id']}: duration drift: {duration}")
            samples = sample_video(item, video_path, output_dir, ffmpeg)
            evidence = {
                "video_sha256": item["video_sha256"],
                "duration_seconds": duration,
                "probe": probe,
            }
        else:
            raise ValueError(f"unsupported input kind: {item['kind']}")
        sheets = render_contact_sheets(samples, output_dir, item["input_id"])
        manifest["inputs"].append(
            {
                "input_id": item["input_id"],
                "role": item["role"],
                "source_id": item["source_id"],
                "session_id": item["session_id"],
                "parent_capture_id": item["parent_capture_id"],
                "outcome_access_state_before_repair": item[
                    "outcome_access_state_before_repair"
                ],
                "candidate_outputs_executed_before_repair": item[
                    "candidate_outputs_executed"
                ],
                "review_sample_period_seconds": item[
                    "review_sample_period_seconds"
                ],
                "sample_count": len(samples),
                "samples": samples,
                "contact_sheets": sheets,
                "input_evidence": evidence,
            }
        )
    manifest["bundle_subject_sha256"] = canonical_json_sha256(
        {
            "protocol_id": manifest["protocol_id"],
            "spec_sha256": manifest["spec_sha256"],
            "prompt_sha256": manifest["prompt_sha256"],
            "candidate_output_visibility": False,
            "inputs": manifest["inputs"],
        }
    )
    write_json(output_dir / "review_bundle_manifest.json", manifest)
    (output_dir / "review_prompt.txt").write_text(
        spec["review_prompt"] + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    spec = args.spec if args.spec.is_absolute() else root / args.spec
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = build_bundle(
        root, spec.resolve(), output.resolve(), args.ffmpeg, args.ffprobe
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "bundle_subject_sha256": manifest["bundle_subject_sha256"],
                "inputs": [
                    {
                        "input_id": item["input_id"],
                        "sample_count": item["sample_count"],
                        "contact_sheet_count": len(item["contact_sheets"]),
                    }
                    for item in manifest["inputs"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
