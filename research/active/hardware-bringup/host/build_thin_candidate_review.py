"""Package already-consumed diagnostic frames for interactive inspection.

This is a selected failure/example view, not an evaluation sample. It never runs
a model or changes candidates, labels, distance states, or frozen thresholds.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import cv2

from atom_camera import AtomCamera
from thin_structure_evaluate_dev_20261013 import matches

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "artifacts.local/work"
CONFIDENCE = WORK / "thin-confidence-dev-20261013"
LABELS = WORK / "thin-structure-dev-20261013/annotations.json"
DEFAULT_OUTPUT = ROOT / "artifacts.local/evidence/useful-quota-20261013/review"
CASES = (
    ("073554", 59, "正前杆 · 异常近距", "weak_target"),
    ("073821", 81, "低头杆 · 锚点外推", "weak_target"),
    ("073635", 39, "背景台阶 · 误入 BODY", "step"),
    ("073635", 22, "侧杆 · 独立 RGB 持续", "weak_target"),
    ("073821", 58, "真杆 · 深度跨度规则损失", "local_loss"),
    ("073740", 41, "横柄 · 远距估计待验证", "weak_target"),
)


def selected_rows(path: Path, wanted: set[tuple[str, int]]) -> dict:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["key"], row["i"]
            if key in wanted:
                if key in result:
                    raise ValueError(f"Duplicate frame identity: {key} in {path}")
                result[key] = row
    if result.keys() != wanted:
        raise ValueError(f"Missing selected frames in {path}: {wanted - result.keys()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ground", type=Path, help="Optional label-blind geometric diagnostic JSONL")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts.local").resolve()):
        raise ValueError("Generated review payload must stay in canonical artifacts.local")
    output.mkdir(parents=True, exist_ok=True)
    wanted = {(key, i) for key, i, _, _ in CASES}
    frames = selected_rows(CONFIDENCE / "frames.jsonl", wanted)
    support = selected_rows(CONFIDENCE / "scale_sensitivity.jsonl", wanted)
    states = selected_rows(CONFIDENCE / "distance_states.jsonl", wanted)
    ground = selected_rows(args.ground, wanted) if args.ground else {}
    labels = {(r["session"], r["i"]): r for r in json.loads(LABELS.read_text(encoding="utf-8"))["samples"]}
    camera = AtomCamera()
    rgb_indexes = {}
    cases = []
    for key, i, title, focus_kind in CASES:
        identity = key, i
        frame, scale, state = frames[identity], support[identity], states[identity]
        session = frame["session"]
        raw_root = WORK / "rgb-real-feasibility-20261012/raw" / session
        if session not in rgb_indexes:
            rgb_indexes[session] = [json.loads(line) for line in (raw_root / "rgb.jsonl").read_text(encoding="utf-8").splitlines()]
        image_path = raw_root / rgb_indexes[session][frame["rgb_index"]]["file"]
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Unreadable source RGB: {image_path}")
        image = camera.undistort_upright(original)
        if image.shape[:2] != (640, 480):
            raise ValueError(f"Unexpected upright coordinates: {image.shape}")
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            raise ValueError("JPEG encoding failed")
        candidates = []
        for c, s, d in zip(frame["candidates"], scale["candidates"], state["candidates"], strict=True):
            coords = [c[name] for name in ("x1", "y1", "x2", "y2")]
            if coords != [d["visual_candidate"][name] for name in ("x1", "y1", "x2", "y2")]:
                raise ValueError(f"Candidate identity mismatch at {identity}")
            if coords != [s[name] for name in ("x1", "y1", "x2", "y2")]:
                raise ValueError(f"Scale-support identity mismatch at {identity}")
            fit = c["confidence"]["frame_fit"]
            candidates.append(dict(segment=coords, stable=c["stable"], hits=c["hits"],
                state=d["distance_state"], z=d["unvalidated_camera_z_estimate_m"],
                disparity=s["scale_support"].get("candidate_disparity_quantiles"),
                anchors=scale.get("anchor_disparity_quantiles"), support=s["scale_support"].get("inside_count"),
                anchor=fit.get("anchor"), fitAccepted=fit["accepted"], groundUpper=None))
        if ground:
            for j, (candidate, raw_ground) in enumerate(zip(candidates, ground[identity]["candidates"], strict=True)):
                ground_coords = [raw_ground["visual_candidate"][name] for name in ("x1", "y1", "x2", "y2")]
                if raw_ground["candidate_index"] != j or candidate["segment"] != ground_coords:
                    raise ValueError(f"Ground diagnostic identity mismatch at {identity}/{j}")
                candidate["groundUpper"] = raw_ground.get("conditional_ground_upper_z_over_height")
        label = labels.get(identity)
        segment = label.get("segment") if label else None
        if focus_kind == "weak_target":
            matched = [j for j, c in enumerate(frame["candidates"]) if segment and matches(segment, c)]
        elif focus_kind == "step":
            matched = [j for j, c in enumerate(frame["candidates"]) if c["stable"] and c["geometry"]["robust_replace"]["query"][1]]
        else:
            matched = [j for j, c in enumerate(frame["candidates"]) if c["stable"] and not c["confidence"]["raw_rules"]["local_consistent"] and any(c["geometry"]["robust_replace"]["query"])]
        if not matched:
            raise ValueError(f"Documented review target not found: {identity}/{focus_kind}")
        focus = max(matched, key=lambda j: (frame["candidates"][j]["stable"], frame["candidates"][j]["hits"]))
        cases.append(dict(id=f"{key}/{i}",title=title,focus=focus,weakSegment=segment,candidates=candidates,
            image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            provenance=f"已消费 Development · {key}/i{i} · " + ("原固定弱标注；不是公制真值" if label else "原报告附加失败样例；不并入22张分母"),
            sourceImage=str(image_path.relative_to(ROOT)),sourceImageSha256=hashlib.sha256(image_path.read_bytes()).hexdigest()))
    pack = dict(evidence="selected consumed examples, not an evaluation", cases=cases)
    payload = json.dumps(pack, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = Path(__file__).with_name("thin_candidate_review.html").read_text(encoding="utf-8")
    if template.count("__THIN_REVIEW_DATA__") != 1:
        raise ValueError("Review template needs one data placeholder")
    fragment = template.replace("__THIN_REVIEW_DATA__", payload)
    if len(fragment.encode("utf-8")) >= 1_000_000:
        raise ValueError("Inline review exceeds 1 MB")
    (output / "thin-candidate-review.html").write_text(fragment, encoding="utf-8")
    (output / "review-pack.json").write_text(json.dumps(pack, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    receipt = dict(cases=len(cases), candidates=sum(len(c["candidates"]) for c in cases), bytes=len(fragment.encode("utf-8")),
        selected_examples_not_evaluation=True, model_runs=0, annotation_sha256=hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        source_files={name:hashlib.sha256((CONFIDENCE/name).read_bytes()).hexdigest() for name in ("frames.jsonl","scale_sensitivity.jsonl","distance_states.jsonl")},
        ground_source_sha256=hashlib.sha256(args.ground.read_bytes()).hexdigest() if args.ground else None,
        output=str(output / "thin-candidate-review.html"))
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
