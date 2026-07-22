#!/usr/bin/env python3
"""Materialize a hash-bound USTRF model-proxy event pilot from contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCENES = (
    "parallel_boundary",
    "step_curb",
    "route_obstacle",
    "lateral_pedestrian_or_ebike",
    "unknown_low_obstacle",
)
ROLES = ("positive", "matched_negative")
PANEL_NAMES = ("first_visible", "approach", "alertable", "cleared")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    return completed.stdout


def version_line(tool: Path) -> str:
    return run([str(tool), "-version"]).splitlines()[0].strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frame_hashes(ffmpeg: Path, video: Path) -> list[dict[str, Any]]:
    output = run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-map", "0:v:0", "-f", "framehash", "-hash", "sha256", "-",
    ])
    frames: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"unexpected framehash row: {line}")
        stream, dts, pts, duration, size, digest = parts
        frames.append({
            "frame_index": len(frames),
            "stream_index": int(stream),
            "dts": int(dts),
            "pts": int(pts),
            "duration": int(duration),
            "decoded_size_bytes": int(size),
            "decoded_frame_sha256": digest.lower(),
        })
    if not frames:
        raise ValueError(f"no decoded frames: {video}")
    return frames


def probe(ffprobe: Path, video: Path) -> dict[str, Any]:
    raw = run([
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,time_base:format=duration",
        "-of", "json", str(video),
    ])
    value = json.loads(raw)
    stream = value["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "time_base": stream["time_base"],
        "duration_ms": round(float(value["format"]["duration"]) * 1000),
    }


def crop_panels(ffmpeg: Path, sheet: Path, panel_dir: Path) -> list[Path]:
    panel_dir.mkdir(parents=True, exist_ok=True)
    crops = ((0, 0), (1, 0), (0, 1), (1, 1))
    result: list[Path] = []
    for index, (column, row) in enumerate(crops):
        output = panel_dir / f"panel_{index + 1}_{PANEL_NAMES[index]}.png"
        run([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(sheet),
            "-vf", f"crop=iw/2:ih/2:{column}*iw/2:{row}*ih/2,scale=1280:720:flags=lanczos",
            "-frames:v", "1", str(output),
        ])
        result.append(output)
    return result


def build_video(ffmpeg: Path, panels: list[Path], video: Path) -> None:
    concat = video.with_suffix(".concat.txt")
    lines: list[str] = []
    for panel in panels:
        lines.extend((f"file '{panel.as_posix()}'", "duration 2.5"))
    lines.append(f"file '{panels[-1].as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-vf", "fps=10,format=yuv420p", "-t", "10",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-movflags", "+faststart", str(video),
        ])
    finally:
        concat.unlink(missing_ok=True)


def materialize(root: Path, config_path: Path, ffmpeg: Path, ffprobe: Path, *, force: bool) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing evidence without --force: {manifest_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("contract_id") != "ustrf_sc_model_proxy_route_event_pilot_v1":
        raise ValueError("unexpected model-proxy pilot config")
    root.mkdir(parents=True, exist_ok=True)
    frozen_config = root / "frozen_config.json"
    shutil.copyfile(config_path, frozen_config)
    contact_root = root / "contact_sheets"
    episodes_root = root / "episodes"
    episodes: list[dict[str, Any]] = []
    for scene in SCENES:
        pair_id = f"model_proxy_session_01__{scene}__pair_01"
        for role in ROLES:
            episode_id = f"{pair_id}__{role}"
            sheet = contact_root / f"{scene}_{role}.png"
            if not sheet.is_file():
                raise FileNotFoundError(sheet)
            episode_root = episodes_root / episode_id
            panels = crop_panels(ffmpeg, sheet, episode_root / "panels")
            video = episode_root / "episode.mp4"
            build_video(ffmpeg, panels, video)
            media = probe(ffprobe, video)
            ledger = {
                "schema": "blindassist_ustrf_sc_model_proxy_frame_ledger_v1",
                "episode_id": episode_id,
                "video_sha256": sha256(video),
                "time_base": media["time_base"],
                "frames": frame_hashes(ffmpeg, video),
            }
            ledger_path = episode_root / "frame_ledger.json"
            write_json(ledger_path, ledger)
            episodes.append({
                "episode_id": episode_id,
                "session_id": "model_proxy_session_01",
                "scene_id": scene,
                "matched_pair_id": pair_id,
                "pair_role": role,
                "expected_should_alert": role == "positive",
                "authority": "synthetic_or_model_proxy_only",
                "contact_sheet_path": sheet.relative_to(root).as_posix(),
                "contact_sheet_sha256": sha256(sheet),
                "panels": [
                    {
                        "panel_index": index + 1,
                        "lifecycle_stage": PANEL_NAMES[index],
                        "path": panel.relative_to(root).as_posix(),
                        "sha256": sha256(panel),
                    }
                    for index, panel in enumerate(panels)
                ],
                "video_path": video.relative_to(root).as_posix(),
                "video_sha256": sha256(video),
                "media": media,
                "frame_ledger_path": ledger_path.relative_to(root).as_posix(),
                "frame_ledger_sha256": sha256(ledger_path),
                "independent_model_reviews": [],
                "model_adjudication": None,
            })
    manifest = {
        "schema": "blindassist_ustrf_sc_model_proxy_route_event_manifest_v1",
        "contract_id": "ustrf_sc_model_proxy_route_event_pilot_v1",
        "status": "awaiting_independent_model_review",
        "benchmark_only": True,
        "human_truth": False,
        "proxy_u0_evaluation_eligible": False,
        "proxy_full_matrix_expansion_eligible": False,
        "training_eligible": False,
        "android_runtime_authorized": False,
        "production_authorized": False,
        "frozen_config_path": frozen_config.relative_to(root).as_posix(),
        "frozen_config_sha256": sha256(frozen_config),
        "toolchain": {
            "ffmpeg_sha256": sha256(ffmpeg),
            "ffmpeg_version": version_line(ffmpeg),
            "ffprobe_sha256": sha256(ffprobe),
            "ffprobe_version": version_line(ffprobe),
        },
        "episodes": episodes,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = materialize(
        args.root.resolve(), args.config.resolve(), args.ffmpeg.resolve(), args.ffprobe.resolve(), force=args.force,
    )
    print(json.dumps({"ok": True, "episode_count": len(manifest["episodes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
