"""CPU-only exact replay feasibility of three predeclared natural96000 scenes.

Replay the original electronic renderer using stored physical boxes and the
retained deterministic pose/photon schedules. Compare the retained float16 z1
element by element, using the original float32 z1 algebra on CPU. No new model,
threshold, dual-sensor view or full natural batch is evaluated here.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
OUT = ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005/natural-precheck'
SNAPSHOT = ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/source'
BIAS = SNAPSHOT.parent/'readouts-gpu/primary-mount-10-snr6/bias.npy'
SELECTION = ((96000, 0), (96001, 0), (96002, 0))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Never overwrite existing evidence: {path}'
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def cpu_z1(hist, ambient, bias):
    # Original cnh_learned_features.sequence_features z1 arithmetic is
    # float32 Torch residual / sqrt(clamp_min(float32 variance,1e-9)).
    import torch
    torch.set_num_threads(2)
    h = torch.as_tensor(hist, dtype=torch.float32, device='cpu')
    a = torch.as_tensor(ambient, dtype=torch.float32, device='cpu')
    b = torch.as_tensor(bias, dtype=torch.float32, device='cpu')
    r = h-b
    v = 16*a[..., None]+b.clamp_min(0)
    return (r/v.clamp_min(1e-9).sqrt()).numpy().astype(np.float16)


def run():
    tick = time.monotonic()
    assert not (OUT/'result.json').exists(), 'Precheck result exists; inspect existing results'
    renderer_source = SOURCE/'source/cnh_proposal_attribution_scenes.py'
    source_plan = read(SOURCE/'PLAN.json')
    assert sha(renderer_source) == source_plan['source_sha256'][renderer_source.name]
    manifest_path = SOURCE/'scene_manifest.json'
    paths = [manifest_path, SOURCE/'PLAN.json', renderer_source, BIAS,
             SNAPSHOT/'cnh_route_sensor.py', SNAPSHOT/'cnh_learned_features.py',
             SNAPSHOT/'cnh_track_a_gpu_readout.py', SNAPSHOT/'cnh_track_a_v13_sensor.py']
    paths += [SOURCE/'features/evaluation'/f'unit{u}.npz' for u,c in SELECTION]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    plan=dict(stage='precheck only', created_utc=datetime.now(timezone.utc).isoformat(),
        selected=[dict(unit=u,config=c,mode=u%3) for u,c in SELECTION],
        selection_reason='first evaluation unit in each of mode0/1/2; fixed config0 before replay',
        photon_seed_formula='2026092900+unit*1000+config*10',
        pose_seed_formula='photon_seed+7 (not needed by current-frame z1)',
        source_sha256=hashes, replay_backend='original NumPy renderer; original float32 Torch z1 algebra on CPU',
        gate='all retained float16 z1 values bitwise equal for all three sequences => FEASIBLE_Z1_REPLAY; otherwise NOT_EVALUABLE',
        limitations='raw histograms not retained; exact retained z1 comparison checks photon+normalization reproduction, not downstream z4 or alarm inference')
    if (OUT/'PLAN.json').exists():
        frozen=read(OUT/'PLAN.json')
        assert frozen['selected']==plan['selected'] and frozen['source_sha256']==hashes
    else:
        save(OUT/'PLAN.json',plan)
    # The retained snapshot is a relocated copy whose SOURCE resolver uses
    # __file__. Execute its exact bytes with the original logical source path
    # so dependencies resolve to the declared frozen-v5 directory, without
    # editing old snapshot bytes or importing live renderer code.
    renderer=ModuleType('natural_original_renderer')
    renderer.__file__=str(Path(__file__).with_name('cnh_proposal_attribution_scenes.py'))
    exec(compile(renderer_source.read_text(encoding='utf8'),str(renderer_source),'exec'),renderer.__dict__)
    assert str(SNAPSHOT) == sys.path[0]
    bias = np.load(BIAS)
    manifest = {(int(r['unit']),int(r['config'])):r for r in read(manifest_path) if r['split']=='evaluation'}
    rows=[]
    for unit,config in SELECTION:
        start=time.monotonic()
        row=dict(unit=unit,config=config,mode=unit%3,
            photon_seed=2026092900+unit*1000+config*10,
            pose_noise_seed=2026092900+unit*1000+config*10+7,
            boxes_source=str(manifest_path.relative_to(ROOT)),
            features_source=str((SOURCE/'features/evaluation'/f'unit{unit}.npz').relative_to(ROOT)))
        try:
            ref=renderer.make_scenes(unit)[0]
            scene=dict(poses=ref['poses'],boxes=manifest[(unit,config)]['boxes'])
            obs=renderer.render(scene,row['photon_seed'])
            replay=cpu_z1(obs['hist'],obs['ambient'],bias)
            with np.load(SOURCE/'features/evaluation'/f'unit{unit}.npz') as z:
                keep=z['scene']==config
                assert np.array_equal(z['frame'][keep],np.arange(16))
                original=z['z1'][keep].copy()
            assert replay.shape==original.shape==(16,8,8,16)
            delta=replay.astype(float)-original.astype(float)
            mismatch=replay.view(np.uint16)!=original.view(np.uint16)
            row.update(status='PASS' if not mismatch.any() else 'NOT_EVALUABLE',
                retained_z1_shape=list(original.shape),retained_z1_dtype=str(original.dtype),
                replay_hist_shape=list(obs['hist'].shape),replay_hist_dtype=str(obs['hist'].dtype),
                physical_pose_shape=list(scene['poses'].shape),boxes_count=len(scene['boxes']),
                compared_elements=int(original.size),bitwise_differences=int(mismatch.sum()),
                max_abs_difference=float(np.max(np.abs(delta))),mean_abs_difference=float(np.mean(np.abs(delta))),
                nonzero_numeric_differences=int(np.count_nonzero(delta)),
                bitwise_equal=bool(np.array_equal(replay.view(np.uint16),original.view(np.uint16))),
                first_mismatches=[dict(index=ix.tolist(),original=float(original[tuple(ix)]),replay=float(replay[tuple(ix)]),delta=float(delta[tuple(ix)])) for ix in np.argwhere(mismatch)[:12]])
            np.savez_compressed(OUT/f'unit{unit}_config{config}.npz',hist=obs['hist'],ambient=obs['ambient'],
                poses=scene['poses'],travel=ref['travel'],replay_z1=replay,original_z1=original)
            row['replay_payload_sha256']=sha(OUT/f'unit{unit}_config{config}.npz')
        except Exception as error:
            row.update(status='NOT_EVALUABLE',reason=repr(error))
        row['seconds']=time.monotonic()-start
        rows.append(row)
        print(json.dumps(row),flush=True)
    status='FEASIBLE_Z1_REPLAY' if all(r['status']=='PASS' for r in rows) else 'NOT_EVALUABLE'
    for relative,expected in hashes.items():
        assert sha(ROOT/relative)==expected, f'Source changed during precheck: {relative}'
    result=dict(status=status,rows=rows,sequences=3,seconds=time.monotonic()-tick,
        cpu_only=True,gpu_allocations=0,source_plan_sha256=sha(OUT/'PLAN.json'),
        next_action='No full natural rendering started; parent decides within remaining budget' if status=='FEASIBLE_Z1_REPLAY' else
            'Stop natural branch: exact retained feature replay failed; no replacement or estimation',
        interpretation='Synthetic original single sensor reproducibility only; no new dual natural cost measurement',
        engineering_recovery='Initial isolated import failed because relocated snapshot SOURCE uses __file__ parents; no rendering occurred. Resumed frozen selection with exact snapshot bytes and original logical __file__ to restore frozen-v5 dependency route.')
    save(OUT/'result.json',result)
    # The optional natural branch requires original raw-CNH elementwise
    # reproduction. z1 is lossy float16 normalization and cannot admit it.
    save(OUT/'admission.json',dict(raw_cnh_gate='NOT_EVALUABLE',natural_full_batch='NOT_RUN',
        raw_reason='Original signed/raw CNH histograms were not retained; regenerated raw histograms have no original raw payload for elementwise comparison. Exact float16 z1 reproduction cannot substitute for exact original raw-CNH reproduction.',
        verified_model_input_gate='PASS' if status=='FEASIBLE_Z1_REPLAY' else 'NOT_EVALUABLE',
        verified_model_input='3 sequences x16384 float16 z1 elements; see retained result for actual differences',
        source_result_status=status,source_result_sha256=sha(OUT/'result.json'),
        replay_metadata='Original photon seed schedules and stored physical boxes replayable; deterministic pose schedule replayed.',
        sequences=rows,decision='Stop optional natural branch at raw admission; no full natural generation, substitution or estimation.'))
    print(json.dumps(dict(status=status,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    run()
