"""Effect of the measured-pose envelope gate on the 2026-10-12 phone captures (EXPLORE).

Afternoon batch: cached per-frame joint-stack grades (Codex REAL_PHONE_NEW_BATCH, segment-mean
IMU pose) gated with each frame's own IMU pitch/roll. Morning batch: fraction of frames that the
gate would silence (per-frame IMU pose from the raw capture). No model is rerun; cuts unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pose_envelope as PE

ROOT = Path(__file__).resolve().parents[4]
NEW = ROOT / 'artifacts.local/work/phone-new-capture-20261012'
OLD_RAW = ROOT / 'artifacts.local/work/phone-imu-capture-20261012/raw'
OUT = ROOT / 'artifacts.local/work/phone-new-capture-20261012/pose_envelope.json'


def counts(grade, ready):
    g = grade[ready]
    return dict(strong=[int((g[:, q] == 2).sum()) for q in range(2)], light=[int((g[:, q] == 1).sum()) for q in range(2)], ready=int(ready.sum()))


def main():
    report = dict(gate=PE.GATE_ID, afternoon={}, morning={})
    for path in sorted((NEW / 'inputs').glob('capture-*.npz')):
        name = path.stem
        pred = NEW / 'joint_replay' / f'{name}_imu.npz'
        if not pred.exists():
            continue
        with np.load(path) as a, np.load(pred) as b:
            pitch, roll = a['pitch'], a['roll']
            grade, ready = b['S_grade'], b['ready'].astype(bool)
        gated, valid = PE.apply(grade, pitch, roll)
        before, after = counts(grade, ready), counts(gated, ready)
        report['afternoon'][name] = dict(pitch_median=float(np.nanmedian(pitch)), roll_median=float(np.nanmedian(roll)),
                                         gated_ready_frames=int((~valid & ready).sum()), before=before, after=after)
        if before != after:
            print(f"{name[17:23]} pitch {np.nanmedian(pitch):+6.1f} roll {np.nanmedian(roll):+6.1f} gated {(~valid & ready).sum()}/{ready.sum()} "
                  f"strong {before['strong']}->{after['strong']} light {before['light']}->{after['light']}")
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    rotation = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    for d in sorted(OLD_RAW.iterdir()):
        x = R.load(d, rotation)
        valid = PE.in_envelope(x['pitch'], x['roll'])
        frac = float((~valid).mean()) if len(valid) else 0.
        report['morning'][d.name] = dict(frames=int(len(valid)), gated_fraction=frac, pitch_median=float(np.nanmedian(x['pitch'])) if len(valid) else None)
        if frac > 0:
            print(f"morning {d.name[17:23]} pitch {np.nanmedian(x['pitch']):+6.1f} gated {frac:.0%} of {len(valid)}")
    OUT.write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
