"""Regenerate the small retained-A Kotlin fixture from the frozen Python oracle.

Run from repository root with the existing research Python runtime; CPU scalar geometry.
Synthetic parity cases do not measure hardware accuracy.
"""
from pathlib import Path
import sys
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))
from tof_corridor_calibration import score_frame, decide

rng = np.random.default_rng(9381)
cases = [('missing', np.zeros((64, 6), np.float32))]
for i, depth in enumerate([.1, .3, .5, 1., 2., 2.8, 3., 3.4, 8.]):
    for box_id, box in enumerate([[.4, .4, .6, .6], [0, 0, .125, .125], [.5, .5, .625, .625]]):
        t = np.zeros((64, 6), np.float32)
        t[17] = [depth / 8, 1, *box]
        cases.append((f'single_{i}_{box_id}', t))
for i in range(40):
    t = np.zeros((64, 6), np.float32)
    for z in range(64):
        y, x = divmod(z, 8)
        t[z] = [rng.uniform(.05, .65), rng.choice([0, 1]), y / 8, x / 8, (y+1)/8, (x+1)/8]
    cases.append((f'mixed_{i}', t))
for index, depth in enumerate(np.linspace(2.9, 3.12, 31)):
    t = np.zeros((64, 6), np.float32)
    t[0] = [depth / 8, 1, .5, .49, .51, .51]
    cases.append((f'depth_boundary_{index}', t))
consumed = ROOT / 'artifacts.local/evidence/ba-local-rescue-fresh-20260923-prepared/observations/tof.npy'
lines = ['# Float32 oracle parity only; no accuracy or fresh-evidence claim.']
if consumed.exists():
    source = np.load(consumed, allow_pickle=False)
    for index in np.linspace(0, len(source)-1, 12, dtype=int):
        cases.append((f'consumed_rescue_frame_{index}', source[index]))
    lines.append('# Consumed observations: ' + str(consumed.relative_to(ROOT)).replace('\\', '/') + ' sha256=' + hashlib.sha256(consumed.read_bytes()).hexdigest())
for name, t in cases:
    boxes = np.rint(t[:, 2:] * [192, 256, 192, 256]).astype(int)
    values = np.where(t[:, 1] == 1, t[:, 0]*8, np.nan)
    scored = score_frame(boxes, values)
    d = decide(scored, .4071309640537889)
    zones = ','.join(str(a['zone']) for a in scored['anchors'] if a['possible'])
    tokens = ';'.join(','.join([str(z), *map(lambda v: repr(float(v)), row)]) for z, row in enumerate(t) if row[1] == 1)
    lines.append('\t'.join(map(str, [name, int(d['alert']), int(d['unknown']), repr(d['score']), d['valid_zones'], d['definite_zones'], d['state'], zones, tokens or '-'])))
out = ROOT / 'core/assist/src/test/resources/hardware_a_parity.tsv'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
print(f'{len(cases)} cases -> {out}')
