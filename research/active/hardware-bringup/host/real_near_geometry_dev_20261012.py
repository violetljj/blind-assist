"""CPU-only descriptive native support and pose sensitivity; no fitted classifier."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'artifacts.local/work/cnh-noise-ablation-dev-20261012/diagnosis'
OUT = ROOT / 'artifacts.local/work/cnh-near-residual-dev-20261012/geometry'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'development_only': True, 'coordinate_order': 'model row,column,bin; zero based',
              'note': 'Positive time-mean is diagnostic mass, not target truth. Dorm pitch is assumed.', 'inputs': {}}
    for sid in ('4', '4b', '5', '6', '7', '9'):
        name = 'dorm_' + sid
        data = np.load(SRC / 'inputs' / (name + '.npz'))
        z = data['z']; mean = z.mean(0); mass = np.maximum(mean, 0); mass[..., :3] = 0
        top = np.argsort(mass.ravel())[-12:][::-1]
        peaks = []
        for ix in top:
            r,c,b = np.unravel_index(ix, mass.shape)
            peaks.append({'row': int(r), 'column': int(c), 'bin': int(b), 'mean_z': float(mean[r,c,b]),
                          'positive_frame_fraction': float((z[:,r,c,b] > 0).mean())})
        row = {'frames': len(z), 'top_bin3plus_cells': peaks, 'poses': {}}
        near = mass.copy(); near[..., 7:] = 0
        near_top = np.argsort(near.ravel())[-8:][::-1]
        row['top_bins3to6'] = [dict(zip(('row','column','bin'), map(int,np.unravel_index(ix, near.shape))), mean_z=float(near.ravel()[ix])) for ix in near_top]
        sources = []
        for pitch in (0, 5):
            cell = name + '_pitch' + str(pitch)
            weights = np.load(SRC / (cell + '_geometry.npz'))['source_weight']
            scores = np.load(SRC / 'joint' / (cell + '.npz'))
            ready = scores['ready']; sources.append(weights)
            weighted = mass[None] * weights
            body_top = np.argsort(weighted[1].ravel())[-8:][::-1]
            row['poses'][str(pitch)] = {
                'ready': int(ready.sum()), 'strong': (scores['S_grade'][ready] == 2).sum(0).tolist(),
                'light': (scores['S_grade'][ready] == 1).sum(0).tolist(),
                'm3_median': np.median(scores['m3_score'][ready], axis=0).tolist(),
                'bin3plus_mass': weighted.sum((1,2,3)).tolist(),
                'bin3plus_mass_by_row': weighted.sum((2,3)).tolist(),
                'bin3plus_mass_by_column': weighted.sum((1,3)).tolist(),
                'bins3to6_mass': weighted[..., 3:7].sum((1,2,3)).tolist(),
                'top_body_contribution': [dict(zip(('row','column','bin'), map(int,np.unravel_index(ix,mass.shape))), mass=float(weighted[1].ravel()[ix])) for ix in body_top],
                'top_cell_body_weights': [float(weights[1, p['row'], p['column'], p['bin']]) for p in peaks],
            }
        w0,w5 = sources
        row['body_mass_support_only_pitch0'] = float((mass * w0[1] * (w5[1] == 0)).sum())
        row['body_mass_support_only_pitch5'] = float((mass * w5[1] * (w0[1] == 0)).sum())
        row['body_mass_shared_pitch0'] = float((mass * w0[1] * (w5[1] > 0)).sum())
        row['body_mass_shared_pitch5'] = float((mass * w5[1] * (w0[1] > 0)).sum())
        result['inputs'][name] = row
    (OUT / 'geometry.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    for name,row in result['inputs'].items():
        print(name, json.dumps({'top': row['top_bin3plus_cells'][:3], 'poses': {k: {f:v[f] for f in ('light','m3_median','bin3plus_mass')} for k,v in row['poses'].items()}, 'exclusive0':row['body_mass_support_only_pitch0'], 'exclusive5':row['body_mass_support_only_pitch5']}))


def intervene():
    import torch
    from local_signed_bias_adapter import ContractReplay
    torch.set_num_threads(2)
    dest = OUT / 'interventions'; dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SRC / 'joint/receipt.json').read_text())['manifest']
    engine = ContractReplay(manifest, device='cpu')
    rows = []
    start = time.monotonic()
    try:
        for sid in ('5','7'):
            data = np.load(SRC / 'inputs' / ('dorm_' + sid + '.npz'))
            for pitch in (0,5):
                cell = 'dorm_' + sid + '_pitch' + str(pitch)
                baseline = np.load(SRC / 'joint' / (cell + '.npz'))
                for name,bins in [('drop_bin7plus',list(range(7,16))),('drop_bins3to6',list(range(3,7)))]:
                    target = dest / (cell + '_' + name + '.npz')
                    if target.exists():
                        pred = np.load(target)
                    else:
                        z=data['z'].copy(); z[..., bins]=0
                        engine.engine.configure_geometry(pitch,0.,0.)
                        pred=engine.infer(z,data['seq'],data['ms'])
                        np.savez_compressed(target,**pred)
                    ready=pred['ready']; assert np.array_equal(ready, baseline['ready'])
                    row=dict(cell=cell,intervention=name,ready=int(ready.sum()),
                             strong=(pred['S_grade'][ready]==2).sum(0).tolist(),
                             light=(pred['S_grade'][ready]==1).sum(0).tolist(),
                             baseline_light=(baseline['S_grade'][ready]==1).sum(0).tolist(),
                             m3_median=np.median(pred['m3_score'][ready],axis=0).tolist())
                    rows.append(row); print(json.dumps(row),flush=True)
    finally:
        engine.close()
    receipt=dict(manifest=manifest,seconds=time.monotonic()-start,device='cpu',cuts_frozen=True,
                 note='Global radial shell intervention affects all target and background evidence; does not identify physical source.',records=rows)
    (dest / 'results.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')


if __name__ == '__main__':
    intervene() if '--intervene' in sys.argv else run()
