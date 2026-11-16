"""Adapt one real TUM Freiburg 3 RGB-D clip to the small memory replay."""

import argparse
import io
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from real_multiframe_pilot_20261117 import ARMS, Frame, replay, save


FR3_RGB = (535.4, 539.2, 320.1, 247.6)
STRIDE = 6
INTERVAL_S = .20
MAX_PAIR_S = .04


def select_frames(timestamps):
    start, end = timestamps[0], timestamps[-1]
    targets = np.arange(0., end - start, INTERVAL_S) + start
    indices = nearest(timestamps, targets)
    return np.unique(indices[np.abs(timestamps[indices] - targets) < MAX_PAIR_S])


def nearest(timestamps, query):
    right = np.clip(np.searchsorted(timestamps, query), 1, len(timestamps) - 1)
    left = right - 1
    return np.where(np.abs(timestamps[right] - query) < np.abs(timestamps[left] - query), right, left)


def image_files(members, stream):
    return sorted((m for m in members if m.isfile() and Path(m.name).parent.name == stream
                   and m.name.endswith(".png")), key=lambda m: float(Path(m.name).stem))


def prepare(archive, output, contact):
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        pose_member = next(x for x in members if x.isfile() and Path(x.name).name == "groundtruth.txt")
        with source.extractfile(pose_member) as stream:
            pose = np.loadtxt(io.BytesIO(stream.read()))
        rgb, depth = (image_files(members, name) for name in ("rgb", "depth"))
        rgb_t, depth_t = (np.array([float(Path(m.name).stem) for m in data]) for data in (rgb, depth))
        di = select_frames(depth_t)
        ri = nearest(rgb_t, depth_t[di])
        pi = np.clip(np.searchsorted(pose[:, 0], depth_t[di]), 1, len(pose) - 1)
        good = ((np.abs(rgb_t[ri] - depth_t[di]) < MAX_PAIR_S)
                & (depth_t[di] >= pose[pi - 1, 0]) & (depth_t[di] <= pose[pi, 0])
                & (pose[pi, 0] - pose[pi - 1, 0] < 2 * MAX_PAIR_S))
        di, ri = di[good], ri[good]
        selected_t = depth_t[di]
        rgb_members, depth_members = [rgb[i] for i in ri], [depth[i] for i in di]
        expected_depth = {m.name for m in depth_members}
        decoded = {}
        for member in sorted(rgb_members + depth_members, key=lambda m: m.offset):
            if member.name in decoded:
                continue
            with source.extractfile(member) as stream:
                buffer = np.frombuffer(stream.read(), dtype=np.uint8)
            flags = cv2.IMREAD_UNCHANGED if member.name in expected_depth else cv2.IMREAD_COLOR
            decoded[member.name] = cv2.imdecode(buffer, flags)
    depth_png = np.stack([decoded[m.name] for m in depth_members])
    rgb_images = np.stack([decoded[m.name] for m in rgb_members])
    if depth_png.dtype != np.uint16 or depth_png.shape != rgb_images.shape[:3]:
        raise ValueError("Expected registered Freiburg RGB and 16-bit depth PNGs")
    locations = np.column_stack([np.interp(selected_t, pose[:, 0], pose[:, j]) for j in (1, 2, 3)])
    rotations = Slerp(pose[:, 0], Rotation.from_quat(pose[:, 4:]))(selected_t).as_matrix()
    time = selected_t - depth_t[0]
    with output.open("xb") as stream:
        np.savez(stream, time=time, depth_timestamp=selected_t, rgb_timestamp=rgb_t[ri],
                 depth_png=depth_png, bgr=rgb_images, locations=locations, rotations=rotations)
    contact_sheet(rgb_images, time, contact, range(0, len(time), 3))
    print(f"{len(time)} matched frames in {time[-1]:.2f} s; max RGB gap "
          f"{1000 * np.max(np.abs(selected_t - rgb_t[ri])):.1f} ms", flush=True)


def contact_sheet(images, time, output, indices):
    indices = list(indices)
    width, height, cols = 192, 164, 6
    rows = (len(indices) + cols - 1) // cols
    sheet = Image.new("RGB", (width * cols, height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for box, i in enumerate(indices):
        x, y = (box % cols) * width, (box // cols) * height
        small = cv2.resize(images[i], (width, height - 20), interpolation=cv2.INTER_AREA)
        sheet.paste(Image.fromarray(small[:, :, ::-1]), (x, y))
        draw.text((x + 5, y + height - 18), f"#{i}   {time[i]:.2f}s", fill="black")
    sheet.save(output)


def tum_frames(data, person_masks=None):
    time, depth_png = data["time"], data["depth_png"]
    locations, rotations = data["locations"], data["rotations"]
    h, w = depth_png.shape[1:]
    fx, fy, cx, cy = FR3_RGB
    ys, xs = np.indices((h, w), dtype=np.float32)
    rays = np.stack(((xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)), axis=-1)[::STRIDE, ::STRIDE]
    weights = (STRIDE**2 / (fx * fy)) / np.linalg.norm(rays, axis=-1)**3
    for i, (raw, origin, rotation) in enumerate(zip(depth_png, locations, rotations, strict=True)):
        depth = raw.astype(np.float32) / 5000.
        small = depth[::STRIDE, ::STRIDE]
        valid = (small >= .35) & (small <= 5.46)
        cloud = (rays[valid] * small[valid, None]) @ rotation.T + origin
        # This handheld recording tests the camera-aligned alert volume, not a wearer's path.
        plane = rotation[:, [2, 0, 1]].T * np.array([1, -1, -1])[:, None]

        def clear(points, image=depth, center=origin, orient=rotation):
            camera = (points - center) @ orient
            z = camera[:, 2]
            divisor = np.where(z > .1, z, 1.)
            x = np.rint(fx * camera[:, 0] / divisor + cx).astype(int)
            y = np.rint(fy * camera[:, 1] / divisor + cy).astype(int)
            inside = (z > .1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            observed = image[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
            return inside & (observed > z + .08) & (observed <= 5.46)

        person = person_masks[i, ::STRIDE, ::STRIDE][valid] if person_masks is not None else None
        yield Frame(i, float(time[i]), 0, cloud, weights[valid], origin, plane, clear, person)


def score(predictions, intervals, arms=ARMS):
    summary = []
    for interval in intervals:
        selected = [predictions[i] for i in range(interval["start"], interval["end"] + 1) if i in predictions]
        times = np.array([p["time"] for p in selected])
        alarm = np.array([p["alarm"] for p in selected], dtype=bool).reshape(-1, len(arms), 2).any(axis=2)
        counts = {}
        for i, method in enumerate(arms):
            active = np.flatnonzero(alarm[:, i])
            # A frame after the first inactive one should not be included in the initial tail.
            quiet = np.flatnonzero(~alarm[:, i])
            tail = (float(times[quiet[0]] - times[0]) if len(quiet) else None) if len(times) else None
            counts[method] = dict(alarms=int(len(active)),
                                  first_s=float(times[active[0]]) if len(active) else None,
                                  last_s=float(times[active[-1]]) if len(active) else None,
                                  initial_tail_s=tail)
        summary.append(dict(interval, scored_frames=len(selected), counts=counts))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("stage", choices=("prepare", "score"))
    args = parser.parse_args()
    work = args.work
    frames = work / "tum_frames.npz"
    if args.stage == "prepare":
        prepare(work / "tum_walking_xyz.tgz", frames, work / "tum_contact.jpg")
        return
    intervals = json.loads((work / "tum_reference.json").read_text(encoding="utf-8-sig"))
    with np.load(frames, allow_pickle=False) as data:
        predictions = replay(tum_frames(data), keep_trace=True)
    save(work / "tum_predictions.json", predictions)
    result = score(predictions, intervals)
    save(work / "tum_comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
