"""G3 (shared reference SNR check) and G4 (mount/SNR/label pairing) for a scale cohort."""
import json
import os
import sys
import numpy as np


def main(E, n_units=192, mounts=(0, -10)):
    """Two mounts: pairing check as v2/v3. One mount (v4 primary only): same checks on it."""
    problems, frames, g3 = [], 0, None
    for u in range(n_units):
        gp = f'{E}/geometry/unit{u:02d}/unit{u:02d}.json'
        if not os.path.exists(gp):
            continue
        g = json.load(open(gp, encoding='utf-8-sig'))
        arrs = {}
        for m in mounts:
            meta = json.load(open(f'{E}/sensor/unit{u:02d}-mount{m}.json', encoding='utf-8'))
            g3 = meta['G3'] if g3 is None else g3
            if meta['G3'] != g3:
                problems.append((u, m, 'G3 differs'))
            with np.load(f'{E}/sensor/unit{u:02d}-mount{m}-observations.npz') as f:
                arrs[m] = {k: f[k] for k in ('config', 'frame', 'hist', 'world_from_Q', 'rate')}
        a, b = arrs[mounts[0]], arrs[mounts[-1]]
        if len(mounts) > 1 and not (np.array_equal(a['config'], b['config']) and np.array_equal(a['frame'], b['frame'])
                                    and np.allclose(a['world_from_Q'], b['world_from_Q'])):
            problems.append((u, 'mount pairing'))
        if int(a['rate']) != 5 or a['hist'].shape[0] != 3:
            problems.append((u, 'rate/snr'))
        for c in g['configs']:
            sel = a['config'] == c['config']
            if sel.sum() != len(c['labels']):
                problems.append((u, c['config'], 'frames'))
            elif not np.allclose(a['world_from_Q'][sel], np.asarray(c['world_from_Q'])):
                problems.append((u, c['config'], 'pose'))
        if not (np.isfinite(a['hist']).all() and np.isfinite(b['hist']).all()):
            problems.append((u, 'nonfinite'))
        frames += a['hist'].shape[1]
    print(json.dumps(dict(G3=g3['pass_gate'], G3_rows=g3['rows'], G4=not problems, problems=problems[:20],
                          frames_per_mount=frames)))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 192,
         tuple(int(m) for m in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0, -10))
