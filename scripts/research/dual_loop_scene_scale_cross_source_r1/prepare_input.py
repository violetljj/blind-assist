from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_VIDEO_SHA256 = (
    "017f860e002c75d093206772800bd68cb1c19f226b74e4d5a933798916347821"
)
FRAME_RATE_HZ = 10
FRAME_STEP_NS = 100_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(video: Path, output: Path, ffmpeg: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_file(video) != EXPECTED_VIDEO_SHA256:
        raise ValueError("Matoaka source video identity mismatch")
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    frames = temporary / "frames"
    frames.mkdir(parents=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            (
                f"fps={FRAME_RATE_HZ},"
                "scale=640:480:force_original_aspect_ratio=decrease,"
                "pad=640:480:(ow-iw)/2:(oh-ih)/2:black"
            ),
            "-q:v",
            "4",
            "-start_number",
            "0",
            str(frames / "frame_%06d.jpg"),
        ],
        check=True,
    )
    images = sorted(frames.glob("frame_*.jpg"))
    if not images:
        raise ValueError("ffmpeg produced no frames")
    manifest = temporary / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for index, image in enumerate(images):
            if image.name != f"frame_{index:06d}.jpg":
                raise ValueError(f"non-contiguous frame sequence at {image.name}")
            row = {
                "schema_version": "blindassist.replay_rgb_frame.v1",
                "source_id": "wikimedia_commons_matoaka_west_virginia_walk_2019",
                "frame_id": index,
                "source_capture_timestamp_ns": index * FRAME_STEP_NS,
                "image_path": f"frames/{image.name}",
                "image_sha256": sha256_file(image),
                "width": 640,
                "height": 480,
            }
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    receipt = {
        "schema_version": "blindassist.dual_loop_cross_source_input_receipt.v1",
        "protocol_id": "DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA",
        "status": "COMPLETE",
        "truth_read": False,
        "video_sha256": EXPECTED_VIDEO_SHA256,
        "sampling_hz": FRAME_RATE_HZ,
        "frame_step_ns": FRAME_STEP_NS,
        "frame_count": len(images),
        "manifest_sha256": sha256_file(manifest),
        "dimensions": "640x480",
        "transform": "aspect-preserving scale and black letterbox",
    }
    (temporary / "input_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            prepare(args.video, args.output, args.ffmpeg),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
