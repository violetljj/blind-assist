"""Zero-fit paired branch diagnostic on consumed MZ6 observations."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble, load_npz, read, sha, write

EVENTS = ['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR']


def distribution(values):
    return dict(zip(['min', 'median', 'max'], np.quantile(values, [0, .5, 1]).tolist()))


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    base = root/'artifacts.local/work/mz6-short-sequence-20260910'
    src = base/'evaluation-v1'
    capture = base/'capture-v3'
    receipt = read(src/'prepare-receipt.json')
    assert read(capture/'source-admission.json')['status'] == 'PASS'
    for name in ['observations', 'evaluator']:
        assert sha(src/f'{name}.npz') == receipt[f'{name}_sha256']
    assert sha(capture/'model/sensor_manifest.json') == receipt['manifest_sha256']
    checkpoint = root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint) == 'ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    data = load_npz(src/'observations.npz')
    truth = load_npz(src/'evaluator.npz')['truth']
    stored = load_npz(src/'predictions.npz')
    cases = read(capture/'evaluator/spec.json')['cases']
    manifest = read(capture/'model/sensor_manifest.json')['frames']
    assert len(cases) == len(manifest) == len(truth) == 200
    pairs = []
    for i, case in enumerate(cases):
        assert case['clip_id'] == data['clip'][i] == manifest[i]['clip_id']
        if '_empty_target' in case['clip_id']:
            continue
        j = i + 25
        empty = cases[j]
        assert empty['clip_id'] == case['clip_id'].removesuffix('_target') + '_empty_target'
        for key in ['camera', 'floor_z_m', 'frame_in_clip', 'nominal_time_s']:
            assert case[key] == empty[key]
        assert [o for o in case['objects'] if o['name'] != 'target'] == empty['objects']
        pairs.append((i, j))
    torch.set_num_threads(1)
    model = CompactEnsemble.from_checkpoint(checkpoint).eval()
    logits = {}
    for condition, prefix in [('clean', ''), ('stress', 'stress_')]:
        tof = np.concatenate((data[prefix+'ranges'].astype(np.float32).reshape(200,128)/4,
                              data[prefix+'valid'].astype(np.float32).reshape(200,128)), 1)
        with torch.inference_mode():
            rgb, distance = model.branches(torch.from_numpy(data['visual']), torch.from_numpy(tof))
        logits[condition+'/RGB'] = rgb.numpy()
        logits[condition+'/ToF'] = distance.numpy()
        logits[condition+'/CURRENT'] = (rgb.numpy()+distance.numpy())*.5
        np.testing.assert_allclose(logits[condition+'/CURRENT'], stored[condition+'/CURRENT'], atol=1e-4, rtol=1e-5)
        assert np.array_equal(logits[condition+'/CURRENT'] >= 0, stored[condition+'/CURRENT'] >= 0)
    records = []
    summaries = []
    for condition in ['clean', 'stress']:
        for start in [0, 50, 100, 150]:
            ids = np.arange(start, start+25); empty = ids+25
            for q, event in enumerate(EVENTS):
                eligible = truth[ids,q] & ~truth[empty,q]
                if not eligible.any():
                    continue
                selected = ids[eligible]; controls = empty[eligible]
                row = dict(condition=condition, clip=str(data['clip'][start]), event=event,
                           opportunities=int(eligible.sum()), branches={})
                for branch in ['RGB', 'ToF', 'CURRENT']:
                    z = logits[condition+'/'+branch]
                    delta = z[selected] - z[controls]
                    false_bits = ~truth[selected]
                    rival = np.where(false_bits, delta, -np.inf).max(1)
                    row['branches'][branch] = dict(hits=int((z[selected,q]>=0).sum()),
                        target_logit=distribution(z[selected,q]), empty_logit=distribution(z[controls,q]),
                        delta=distribution(delta[:,q]), positive_delta=int((delta[:,q]>0).sum()),
                        selective_positive_delta=int(((delta[:,q]>0)&(delta[:,q]>rival)).sum()))
                miss = logits[condition+'/CURRENT'][selected,q] < 0
                rp = logits[condition+'/RGB'][selected,q] >= 0
                tp = logits[condition+'/ToF'][selected,q] >= 0
                row['miss_categories'] = dict(rgb_positive=int((miss&rp).sum()),
                    tof_positive=int((miss&tp).sum()),both_negative=int((miss&~rp&~tp).sum()))
                summaries.append(row)
        for i,j in pairs:
            records.append(dict(condition=condition, target_index=i,empty_index=j,
                clip=str(data['clip'][i]), frame_in_clip=int(data['index'][i]),
                truth=truth[i].tolist(), empty_truth=truth[j].tolist(),
                branches={b:dict(target=logits[condition+'/'+b][i].tolist(),
                    empty=logits[condition+'/'+b][j].tolist(),
                    delta=(logits[condition+'/'+b][i]-logits[condition+'/'+b][j]).tolist())
                    for b in ['RGB','ToF','CURRENT']}))
    np.savez_compressed(output/'branch-logits.npz', **logits, truth=truth, clip=data['clip'])
    write(output/'paired-rows.json', records)
    write(output/'result.json', summaries)
    write(output/'receipt.json',dict(status='PASS',pairs=len(pairs),training_steps=0,
        current_flag_parity=1600,conditions=2,source='MZ6 consumed Development',
        inputs={str(p):sha(p) for p in [src/'observations.npz',src/'evaluator.npz',src/'predictions.npz',
            capture/'evaluator/spec.json', checkpoint]},
        script_sha256=sha(Path(__file__)), outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    for row in summaries:
        if row['condition']=='clean':
            print(row, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.output)
