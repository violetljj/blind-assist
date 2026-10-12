"""M3-only strong chain (no analytic LOCAL) for the nuisance-DR backbone, sim hold + real manifest.

The raised-M3 cut is matched to the reference default's total strong clear-slot count
(raised OR LOCAL on clean inputs) under each condition. Development description only.
"""
import json, hashlib, os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parent); sys.path.insert(0, '.')
import numpy as np
import cnh_nuisance_dr_m3_eval_dev_20261011 as V

cat = np.load(V.PREV / 'hold_grades_notifications.npz')['category']
sel = lambda a, c: a.transpose(0, 3, 1, 2)[cat == c]
ref = np.load(V.NUIS / 'sim_original.npz')
total = int(sel((V.smooth(ref['m3_raw']) >= V.RA) | (V.smooth(ref['local_raw']) >= V.LT), 'clear').sum())
rows = {}
for name in V.NAMES:
    m = V.smooth(np.load(V.NEW / f'hold_m3_{name}.npy'))
    v = np.sort(sel(m, 'clear').ravel())[::-1]; ra = float(v[total])
    A = m >= ra
    rows[name] = dict(raised_cut=ra, contact_timely=float(sel(A, 'contact')[..., :11].any(-1).mean()),
                      clear_events=float(sel(A, 'clear')[..., :11].any(-1).mean()), pass_events=float(sel(A, 'pass')[..., :11].any(-1).mean()))
    print(f"m3_only|{name:24s} timely {rows[name]['contact_timely']:.3f} clear {rows[name]['clear_events']:.3f} pass {rows[name]['pass_events']:.3f} cut {ra:.2f}")
base = json.loads((V.NEW / 'real_manifest_real_like_nuisance.json').read_text(encoding='utf8'))
base['cuts'] = dict(base['cuts'], raised_tau=rows['real_like_nuisance']['raised_cut'], m3_tau=rows['real_like_nuisance']['raised_cut'],
                    local_tau=1e300, rule='M3-only strong: raised cut matched to reference total strong clear slots under real_like_nuisance; LOCAL disabled; Development only')
(V.NEW / 'real_manifest_m3_only.json').write_text(json.dumps(base, indent=1), encoding='utf8')
(V.NEW / 'm3_only_receipt.json').write_text(json.dumps(dict(reference_total_clear_slots=total, rows=rows,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1), encoding='utf8')
