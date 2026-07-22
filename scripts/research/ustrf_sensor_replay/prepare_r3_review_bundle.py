from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from contract import read_json, safe_file, sha256, write_json


PROMPT = """You are one isolated USTRF R3 route/event reviewer. Inspect every sheet for every source in this directory; the sheets cover every consecutive frame in order. Green dots/lines are hash-bound route-truth projections derived only from future mocap camera positions. No candidate alert output is present or available.\n\nFor each source, decide whether the complete sequence contains a credible body/sensor-bound forward route and at least one obstacle event that intersects that route with observable causal lifecycle. Never infer an event merely because a person or object is visible. If admitted, freeze integer original-frame anchors for each event: onset_frame, alertable_frame, passed_or_cleared_frame, end_frame, critical. Anchors must follow onset <= alertable <= passed_or_cleared <= end. If route projection is sparse, non-forward, dominated by camera rotation, or lifecycle is not observable, reject or abstain.\n\nWrite reviewer_<role>_raw.json with schema blindassist_ustrf_sensor_replay_r3_review_v1 and include reviewer_id, reviewer_type=\"ai_model\", a reviewer_role distinct from the other role, provider, model, model_version, review_run_id, workflow_id=\"ustrf_event_review_v1\", prompt_sha256 for this prompt file, input_sha256 for review_inputs.json, independent_review=true, isolated_context=true, other_reviewer_outputs_viewed=false, other_review_visible_before_submission=false, candidate_alerts_viewed=false, candidate_output_visible=false, numeric confidence, abstained, abstain_reasons, verdict, and sources. Each source row must bind source_id and manifest_sha256, set route_valid true/false/\"abstain\", disposition, reason, and events[]. Do not inspect any candidate output or any other reviewer output.\n"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-sheet", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    args.output.mkdir(parents=True)
    candidate = read_json(args.candidate)
    if candidate.get("candidate_alerts_frozen_before_review") is not True:
        raise ValueError("candidate was not frozen before review")
    candidate_by_id = {row["source_id"]: row for row in candidate["sources"]}
    sources = []
    for source_dir in sorted(args.bundle_root.iterdir()):
        if not (source_dir / "bundle.json").is_file():
            continue
        bundle = read_json(source_dir / "bundle.json")
        source_id = bundle["source"]["source_id"]
        root = Path(bundle["source_root"])
        frames = [json.loads(line) for line in (source_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line]
        truth = candidate_by_id[source_id]["route_truth"]
        if len(truth) != len(frames):
            raise ValueError("route truth length mismatch")
        source_output = args.output / source_id
        source_output.mkdir()
        sheets = []
        for sheet_index, start in enumerate(range(0, len(frames), args.frames_per_sheet)):
            selected = list(range(start, min(len(frames), start + args.frames_per_sheet)))
            canvas = Image.new("RGB", (1280, 960), "black")
            draw = ImageDraw.Draw(canvas)
            for slot, frame_index in enumerate(selected):
                image = Image.open(safe_file(root, frames[frame_index]["rgb_path"])).convert("RGB")
                image.thumbnail((128, 96))
                cell = Image.new("RGB", (128, 96), "black")
                cell.paste(image, ((128-image.width)//2, (96-image.height)//2))
                route = truth[frame_index]
                if route["status"] == "known":
                    original_width, original_height = Image.open(safe_file(root, frames[frame_index]["rgb_path"])).size
                    x = int(round(float(route["uv"][0]) * 128 / original_width))
                    y = int(round(float(route["uv"][1]) * 96 / original_height))
                    cell_draw = ImageDraw.Draw(cell)
                    cell_draw.ellipse((x-4, y-4, x+4, y+4), fill="lime", outline="black")
                x0 = (slot % 10) * 128
                y0 = (slot // 10) * 96
                canvas.paste(cell, (x0, y0))
                draw.rectangle((x0, y0, x0+64, y0+13), fill="black")
                draw.text((x0+2, y0+1), str(frame_index), fill="white")
            path = source_output / f"sheet_{sheet_index:03d}_{selected[0]:06d}_{selected[-1]:06d}.jpg"
            canvas.save(path, quality=94)
            sheets.append({"path": path.relative_to(args.output).as_posix(), "sha256": sha256(path), "start_frame": selected[0], "end_frame": selected[-1], "frame_count": len(selected)})
        source_manifest = {
            "source_id": source_id,
            "frame_count": len(frames),
            "source_frames_sha256": sha256(source_dir / "frames.jsonl"),
            "route_truth_sha256": sha256(args.candidate),
            "sheets": sheets,
            "every_consecutive_frame_included_once": sum(row["frame_count"] for row in sheets) == len(frames),
            "candidate_alerts_visible": False,
        }
        source_manifest_path = source_output / "manifest.json"
        write_json(source_manifest_path, source_manifest)
        sources.append({"source_id": source_id, "manifest": source_manifest_path.relative_to(args.output).as_posix(), "manifest_sha256": sha256(source_manifest_path), "frame_count": len(frames), "sheet_count": len(sheets)})
    write_json(args.output / "review_inputs.json", {
        "schema": "blindassist_ustrf_sensor_replay_r3_review_inputs_v1",
        "candidate_alerts_visible": False,
        "complete_consecutive_sequences": True,
        "sources": sources,
        "production_authority": False,
    })
    (args.output / "reviewer_a_prompt.txt").write_text(PROMPT.replace("<role>", "a"), encoding="utf-8")
    (args.output / "reviewer_b_prompt.txt").write_text(PROMPT.replace("<role>", "b"), encoding="utf-8")
    print(json.dumps({"sources": len(sources), "sheets": sum(row["sheet_count"] for row in sources)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
