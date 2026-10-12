"""Can monocular metric depth on the uncalibrated Atom RGB see the 2026-10-12 hall obstacles? (EXPLORE)

Atom camera intrinsics and camera-to-ToF extrinsics are not calibrated, so the frozen RGB/fusion
readouts (which need public K) cannot be applied. Here UniDepthV2 ViT-L predicts its own camera
and metric points per frame (no K supplied). Points are rotated into the head corridor frame with
the frame's IMU-derived ToF pitch/roll (camera axis assumed parallel to the ToF axis) and counted
inside the same HEAD/BODY query boxes (x +/-0.30 m, y -0.20..0.42 / 0.42..0.90 m below, z 0.3..3 m).
Scene labels are Claude's reading of RGB frames, not truth. Development description only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012/raw'
OUT = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012'
UPSTREAM = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/UniDepth'
WEIGHTS = ROOT / 'artifacts.local/models/lpiccinelli--unidepth-v2-vitl14'
SCENES = {
    '072853': 'H0 floor look-down', '072939': 'H1 hall walk level (toward steps/platform)',
    '073020': 'H2 hall walk looking down', '073138': 'H3 door/wall approach', '073245': 'H4 pillar approach',
    '073554': 'H5 pole ahead (on pouf), level', '073635': 'H6 pole passed on left', '073740': 'H7 held crossbar',
    '073821': 'H8 pole ahead, looking down',
}
BOXES = {'HEAD': (-.20, .42), 'BODY': (.42, .90)}
STRIDE = 2   # every second ToF frame


def lines(path):
    for line in open(path, encoding='utf8'):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def load_model():
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from safetensors.torch import load_file
    from unidepth.models import UniDepthV2
    config = json.loads((WEIGHTS / 'config.json').read_text(encoding='utf-8'))
    config['model']['pixel_encoder']['pretrained'] = None
    model = UniDepthV2(config)
    model.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True)
    return model.float().eval().cuda()


def rot(pitch, roll):
    """corridor_from_sensor for pitch (deg, negative down) and optical-axis roll, as the replay."""
    p, r = math.radians(pitch), math.radians(roll)
    # Matches cnh_cvr_pilot.rotation(-10, 'x'): forward maps to +y (down) when pitch < 0.
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rx @ rz


def main():
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    rotation = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    model = load_model()
    report = {}
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    for d in sorted(RAW.iterdir()):
        key = d.name[17:23]  # UTC HHMMSS
        if key not in SCENES:
            continue
        x = R.load(d, rotation)
        tof_t = [r['received_elapsed_ns'] for r in lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
        rgb = [r for r in lines(d / 'rgb.jsonl')]
        rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
        dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1]
        rows = []
        for i in range(0, len(tof_t), STRIDE):
            j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
            if abs(rgb_t[j] - tof_t[i]) > 1.5e8 or not np.isfinite(x['pitch'][i]):
                continue
            im = Image.open(d / rgb[j]['file']).convert('RGB').rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            a = np.asarray(im)
            tensor = torch.from_numpy(a.transpose(2, 0, 1).copy()).cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                pred = model.infer(tensor)
            pts = pred['points'][0].float().cpu().numpy().reshape(3, -1).T          # camera frame x right, y down, z fwd
            K = pred['intrinsics'][0].float().cpu().numpy()
            c = (rot(x['pitch'][i], x['roll'][i]) @ pts.T).T
            row = dict(tof_index=i, rgb_file=rgb[j]['file'], dt_ms=float((rgb_t[j] - tof_t[i]) / 1e6),
                       pitch=float(x['pitch'][i]), hfov_deg=float(2 * math.degrees(math.atan(a.shape[1] / 2 / K[0, 0]))))
            lower = c[(c[:, 2] > .5) & (c[:, 2] < 6) & (np.abs(c[:, 0]) < 1.5)]
            row['floor_y_m'] = float(np.percentile(lower[:, 1], 95)) if len(lower) > 100 else None
            for name, (lo, hi) in BOXES.items():
                inside = (np.abs(c[:, 0]) <= .30) & (c[:, 1] >= lo) & (c[:, 1] <= hi) & (c[:, 2] >= .30) & (c[:, 2] <= 3.)
                z = np.sort(c[inside, 2])
                row[name] = dict(pixels=int(inside.sum()), z16=float(z[15]) if len(z) > 15 else None)
            centre = dist[i, 2:6, 2:6]
            row['tof_centre_min_m'] = float(np.nanmin(centre) / 1000) if np.isfinite(centre).any() else None
            rows.append(row)
            if i in (0, (len(tof_t) // 2) // STRIDE * STRIDE, (len(tof_t) - 1) // STRIDE * STRIDE):
                depth = pred['depth'][0, 0].float().cpu().numpy()
                vis = (np.clip(depth / 5, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(np.concatenate([a, np.repeat(vis[..., None], 3, -1)], 1)).save(OUT / 'frames' / f'{key}_{i:03d}.jpg', quality=80)
        report[key] = dict(scene=SCENES[key], session=d.name, frames=rows)
        hp = [r['HEAD']['pixels'] for r in rows]; bp = [r['BODY']['pixels'] for r in rows]
        bz = [r['BODY']['z16'] for r in rows if r['BODY']['z16']]; hz = [r['HEAD']['z16'] for r in rows if r['HEAD']['z16']]
        fl = [r['floor_y_m'] for r in rows if r['floor_y_m']]
        print(f"{key} {SCENES[key]:42s} n={len(rows):3d} hfov~{np.median([r['hfov_deg'] for r in rows]):5.1f} floor~{np.median(fl) if fl else float('nan'):4.2f}m "
              f"| HEAD frames>200px {np.mean(np.array(hp) > 200):4.0%} z16 med {np.median(hz) if hz else float('nan'):4.2f} "
              f"| BODY frames>200px {np.mean(np.array(bp) > 200):4.0%} z16 med {np.median(bz) if bz else float('nan'):4.2f}", flush=True)
    (OUT / 'unidepth_query_readout.json').write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
