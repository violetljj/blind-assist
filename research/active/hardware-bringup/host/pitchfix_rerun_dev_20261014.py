"""Dev rerun of frozen v3/v6 with the camera<-ToF extrinsic pitch removed (EXPLORE, 2026-10-14).

Evidence (hindsight-truth-dev-20261014): corridor vanishing points and a ToF-ranged wall-base test put the Atom camera ~level while
the fitted extrinsic (yaw 10.5, pitch 14.5, roll 1) makes the chain believe it looks ~14.5 deg down. This rerun swaps only the
extrinsic (same yaw/roll, pitch 0); models, thresholds, labels and data are unchanged. Consumed development data only.
Usage: pitchfix_rerun_dev_20261014.py <v3|v6> --raw ... --labels ... --out ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import atom_tof_extrinsic_dev_20261012 as X
import frozen_fusion_eval_v1 as V1
from atom_camera import ROOT

FIXED = ROOT / 'artifacts.local/work/hindsight-truth-dev-20261014/pitchfix/camera_tof_extrinsic_pitch0.json'
if not FIXED.exists():
    FIXED.write_text(json.dumps(dict(camera_from_tof_euler_deg=dict(yaw=10.5, pitch=0.0, roll=1.0), rotation=X.euler(10.5, 0.0, 1.0).tolist(),
                                     note='dev: original fit with pitch component removed'), indent=1), encoding='utf8')
V1.EXTRINSIC = FIXED
V1.FROZEN_FILES = [FIXED if p.name == 'camera_tof_extrinsic.json' else p for p in V1.FROZEN_FILES]
ver = sys.argv.pop(1)
mod = __import__(f'frozen_fusion_eval_{ver}')
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); mod.main(a.raw, a.labels, a.out)
