"""Seal learned stereo geometry before consumed Development scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import mz101_spatial as m
from run_mz101_spatial import metrics, sha, observation_contract
from run_mz103_depth_frontend import PANELS, event_compare, write

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / 'artifacts.local/work/mz104-foundation-stereo-20260912'
REFERENCE = ROOT / 'artifacts.local/work/mz103-depth-frontend-20260912/native-v1'


def prepare():
    TASK.mkdir(parents=True, exist_ok=True)
    path = TASK / 'rgb-inputs.json'
    if path.exists():
        raise ValueError('Input manifest already exists')
    frames = []
    for panel, (folder, _, _, _) in PANELS.items():
        capture = ROOT / 'artifacts.local/work' / folder / 'capture-v1'
        capture_receipt = json.loads((capture / 'receipt.json').read_text())
        observations = json.loads((REFERENCE / panel / 'observations.json').read_text())
        for obs in observations:
            entry = dict(panel=panel, id=obs['id'])
            for view in ('left', 'right'):
                image = capture / 'frame' / obs['id'] / (view + '.png')
                entry[view] = str(image.resolve())
                entry[view + '_sha256'] = sha(image)
                assert entry[view + '_sha256'] == capture_receipt['hashes'][
                    str(image.relative_to(capture)).replace('\\', '/')]
            frames.append(entry)
    assert len(frames) == 576
    write(path, dict(rig=m.RIG, frames=frames, authority='RGB_AND_CALIBRATION_ONLY'))
    print(path)


def evaluate(frontend, output):
    if output.exists() or output.resolve().parent != TASK.resolve():
        raise ValueError('New task-local output required')
    receipt = json.loads((frontend / 'receipt.json').read_text())
    assert receipt['status'] == 'PASS'
    assert receipt['frames'] == 576
    assert receipt['input_manifest_sha256'] == sha(TASK / 'rgb-inputs.json')
    inputs = json.loads((TASK / 'rgb-inputs.json').read_text())
    assert inputs['rig'] == m.RIG and len(inputs['frames']) == 576
    for panel, (folder, _, _, _) in PANELS.items():
        capture = ROOT / 'artifacts.local/work' / folder / 'capture-v1'
        captured = json.loads((capture / 'receipt.json').read_text())
        entries = [f for f in inputs['frames'] if f['panel'] == panel]
        original_ids = [o['id'] for o in json.loads((REFERENCE/panel/'observations.json').read_text())]
        assert [f['id'] for f in entries] == original_ids
        for f in entries:
            for view in ('left','right'):
                expected = captured['hashes'][f"frame/{f['id']}/{view}.png"]
                assert sha(Path(f[view])) == f[view+'_sha256'] == expected
    for name, digest in receipt['hashes'].items():
        assert sha(frontend / name) == digest, name
    reference_receipt = json.loads((REFERENCE / 'receipt.json').read_text())
    for name, digest in reference_receipt['hashes'].items():
        assert sha(REFERENCE / name) == digest, name
    output.mkdir(parents=True)
    panels = {}
    for panel, (folder, _, _, _) in PANELS.items():
        old = np.load(REFERENCE / panel / 'predictions.npz')
        obs = json.loads((REFERENCE / panel / 'observations.json').read_text())
        support = {k: old[k + '_support'] for k in ('tof', 'sgbm', 'sgbm_union')}
        new_support = []
        depth_hashes = {}
        for o in obs:
            path = frontend / panel / 'depth' / (o['id'] + '.npy')
            rel = str(path.relative_to(frontend)).replace('\\', '/')
            assert rel in {k.replace('\\', '/') for k in receipt['hashes']}, rel
            depth = np.load(path)
            assert depth.shape == (360, 640)
            depth = np.where(np.isfinite(depth) & (depth >= .5) & (depth <= 4), depth, np.nan)
            new_support.append(m.readout(m.depth_points(depth), o['pose'], common_fov=True)[0])
            depth_hashes[rel] = sha(path)
        support['foundation'] = np.array(new_support)
        support['foundation_union'] = support['foundation'] + support['tof']
        ids = [o['episode'] for o in obs]
        final = {k: m.hysteresis(v, ids) for k, v in support.items()}
        raw = {k: v > 0 for k, v in support.items()}
        for k in ('tof', 'sgbm', 'sgbm_union'):
            np.testing.assert_array_equal(final[k], old[k])
        dest = output / panel
        dest.mkdir()
        np.savez_compressed(dest / 'predictions.npz', **final,
                            **{k+'_support': v for k,v in support.items()})
        write(dest / 'prediction-seal.json', dict(prediction_sha256=sha(dest/'predictions.npz'),
              depth_hashes=depth_hashes, baseline_parity=True,
              frontend_receipt_sha256=sha(frontend/'receipt.json')))
        # Read consumed task annotations only after candidate prediction sealing.
        gt = np.load(REFERENCE / panel / 'truth.npy')
        spec_path = ROOT/'artifacts.local/work'/folder/'capture-v1/spec.json'
        reference_seal = json.loads((REFERENCE/panel/'prediction-seal.json').read_text())
        assert sha(spec_path) == reference_seal['spec_sha256']
        spec = json.loads(spec_path.read_text())
        assert observation_contract(spec) == obs
        scores = {stage: {k: metrics(v, gt, obs) for k,v in predictions.items()}
                  for stage, predictions in [('raw', raw), ('final', final)]}
        slices = {family: {stage: {k: metrics(v,gt,obs,[f['family']==family for f in spec['frames']])
                  for k,v in predictions.items()} for stage,predictions in [('raw',raw),('final',final)]}
                  for family in dict.fromkeys(f['family'] for f in spec['frames'])}
        retention = {}
        for family in ('thin_left', 'thin_right', 'small_head', 'occluded_thin'):
            select = np.array([f['family']==family for f in spec['frames']])
            original = raw['sgbm_union'] & gt & select[:,None]
            if original.any():
                retention[family] = float((raw['foundation_union'] & original).sum()/original.sum())
        base, candidate = scores['raw']['sgbm_union'], scores['raw']['foundation_union']
        events = event_compare(scores['final']['foundation_union'], scores['final']['sgbm_union'])
        spatial = dict(fewer_raw_fp=candidate['FP'] < base['FP'],
            no_more_raw_fn=candidate['FN'] <= base['FN'],
            critical_slice_retention=all(v >= .95 for v in retention.values()))
        lifecycle = dict(no_lost_events=not events['lost_events'],
            no_more_false_sessions=scores['final']['foundation_union']['false_sessions'] <= scores['final']['sgbm_union']['false_sessions'],
            paired_delay_within_one_frame=events['max_extra_delay_s'] is None or events['max_extra_delay_s'] <= .25)
        row = dict(scores=scores, slices=slices, critical_retention=retention,
            appearances={appearance: {stage: {k: metrics(v,gt,obs,
                [f['appearance']==appearance for f in spec['frames']])
                for k,v in predictions.items()} for stage,predictions in [('raw',raw),('final',final)]}
                for appearance in ('textured','flat')},
            spatial_gates=spatial, lifecycle_gates=lifecycle, event_comparison=events,
            transitions={stage: dict(lost_tp=int((p['sgbm_union'] & ~p['foundation_union'] & gt).sum()),
                new_tp=int((~p['sgbm_union'] & p['foundation_union'] & gt).sum()),
                removed_fp=int((p['sgbm_union'] & ~p['foundation_union'] & ~gt).sum()),
                added_fp=int((~p['sgbm_union'] & p['foundation_union'] & ~gt).sum()))
                for stage,p in [('raw',raw),('final',final)]},
            fp_sources={k: dict(current=int((p & ~gt & raw[k]).sum()),
                held=int((p & ~gt & ~raw[k]).sum())) for k,p in final.items()},
            per_part={part: {stage: {k: dict(TP=int((p[:,j]&gt[:,j]).sum()),
                FP=int((p[:,j]&~gt[:,j]).sum()), FN=int((~p[:,j]&gt[:,j]).sum()))
                for k,p in predictions.items()} for stage,predictions in [('raw',raw),('final',final)]}
                for j,part in enumerate(m.PARTS)})
        write(dest / 'summary.json', row)
        panels[panel] = row
        print(panel, json.dumps(dict(spatial_gates=spatial,lifecycle_gates=lifecycle)), flush=True)
    pooled = {stage: {k: {x: sum(p['scores'][stage][k][x] for p in panels.values())
              for x in ('TP','FP','FN','missed_events','false_sessions')}
              for k in ('tof','sgbm_union','foundation','foundation_union')}
              for stage in ('raw','final')}
    summary = dict(evidence='CONSUMED_RENDERED_DEVELOPMENT', frames=576,
        pooled_descriptive=pooled, spatial_component_pass=all(all(p['spatial_gates'].values()) for p in panels.values()),
        lifecycle_pass=all(all(p['lifecycle_gates'].values()) for p in panels.values()),
        frontend_receipt_sha256=sha(frontend/'receipt.json'),
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(output/'summary.json', summary)
    write(output/'receipt.json', dict(status='PASS',hashes={str(p.relative_to(output)):sha(p)
          for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--frontend', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        evaluate(args.frontend,args.output)
