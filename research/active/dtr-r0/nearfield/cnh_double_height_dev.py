"""Frozen readout under physical paired height interventions, consumed Development."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-double-height-dev-20261007'
PARENT = WORK / 'cnh-center-readout-ablation-dev-20261007'
TAGS = ('H', 'B', 'HB')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    os.replace(temp, path)


def thresholds(result):
    out = {}
    for row in result['fold_records']:
        if row['key'].startswith('in_domain/calibrated/') and row['key'].split('/')[-1] in ('pair_only', 'original_center'):
            key = (row['fold'], row['key'].split('/')[2], row['key'].split('/')[-1])
            if key in out:
                raise ValueError('duplicate threshold')
            out[key] = row['threshold']
    expected = {(f, c, m) for f in range(3) for c in ('0.025','0.050') for m in ('pair_only','original_center')}
    if set(out) != expected:
        raise ValueError('three folds, two caps, two methods required')
    return out


def decisions(score, theta, deadline):
    score = np.asarray(score)
    if score.shape != (13,) or not np.isfinite(score).all():
        raise ValueError('13 finite causal scores required')
    alarm = score >= theta
    timely = bool(alarm[:deadline+1].any()) if deadline is not None and 0 <= deadline < 13 else False
    post = bool(alarm[2:deadline+1].any()) if deadline is not None and 2 <= deadline < 13 else False
    return dict(timely=timely, timely_excluding_warmup=post,
                false_alarm_intervals=int(alarm[2:12].sum()), warmup_alarm_frames=int(alarm[:2].sum()))


def freeze():
    manifest = read(OUT/'geometry_manifest.json')
    paths = {Path(__file__), OUT/'geometry_manifest.json', PARENT/'PLAN.json', PARENT/'result.json', PARENT/'ledger.npz'}
    paths.update(PARENT/'models'/f'in_domain_fold{f}_pair_only.joblib' for f in range(3))
    paths.update(WORK/f'cnh-margin-labels-20261002/models/M3/model_seed{s}.pt' for s in range(5))
    names = ['cnh_double_height_geometry_dev.py', 'cnh_double_height_render_dev.py', 'cnh_active_scan_dev.py',
             'cnh_heading_uncertainty_dev.py', 'cnh_fused_projection.py', 'cnh_extrinsic_aug_data.py',
             'cnh_location_reference_gpu.py', 'cnh_displacement_ceiling_render.py', 'cnh_temporal_readout_data.py',
             'cnh_temporal_readout_model.py', 'cnh_cvr_pilot.py', 'cnh_cvr_projection.py',
             'cnh_cvr_v2_materialize.py', 'cnh_sequence_observed_geometry.py', 'cnh_tristate_dev.py',
             'cnh_dual_sensor_envelope_natural.py', 'cnh_proposal_attribution_scenes.py',
             'cnh_displacement_ceiling.py', 'cnh_margin_confirm.py', 'cnh_near_range.py',
             'cnh_structure_space.py', 'cnh_corridor_late_fusion.py']
    paths.update(HERE / name for name in names)
    paths.update((WORK/'cnh-track-a-v5-20260928/data/source').glob('*.py'))
    from cnh_double_height_render_dev import source_paths
    paths.update(ROOT / p for p in manifest['hashes'])
    for a in manifest['anchors']:
        paths.update(Path(p) for p in source_paths(a['unit']).values())
    # Normalization is part of the sensor input, not a model fit in this run.
    paths.add(WORK/'cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy')
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    folds = read(PARENT/'PLAN.json')['folds']
    for a in manifest['anchors']:
        f = folds[a['fold']]
        assert a['unit'] in f['evaluation'] and a['unit'] not in f['train'] + f['calibration']
    save(OUT/'PLAN.json', dict(phase='EXPLORE physical structure intervention on consumed Development',
        goal='Test whether frozen nonmonotonic pair readout suppresses real joint-height contact',
        authorization='User active continuous algorithm exploration goal; new physical intervention, no old run extension',
        budget_geometry_cpu_seconds=600, budget_gpu_wall_seconds=1800, budget_analysis_cpu_seconds=600,
        scope='Selected 36 contact +36 clear anchors where geometry support permits, each H/B/HB; missing strata retained; no score-driven replacement',
        heights={'H':[-.1,.26], 'B':[.5,.84], 'HB':[-.1,.84]},
        frozen='Original16 frames/noisy poses/head query errors/photon seeds, 5 M3 seeds and 3 pair models; OOF model and original calibration thresholds, no retraining or threshold fitting',
        comparison='Original max and pair readout at own frozen 2.5% and 5% calibration thresholds; constant legacy M3 threshold appendix',
        truth='Physical all-box surface geometry at original0.9m deadline; one HB physical contact is one event. Clear both queries all13 saved frames. Unknown excluded.',
        controls='Same unit/config, x/z bounds, reflectance, backgrounds, path, error and noise seed; only target y extent changes. HB single continuous AABB.',
        metrics='Timely by last causal output, excluding startup separately; clear FA output2:12 and startup0:2; no three-state metrics without new side queries.',
        mechanism='Report H/B timely to HB missed; strongest witness: HB both smoothed logits >= single-height logits at an old timely frame, pair falls below threshold and HB misses, while max HB timely. Finite counterexamples refute unrestricted inheritance; absence proves no universal retention.',
        decision_check='One anchor is minimum event change; paired physical intervention not fresh participant/simulator confirmation. Separate upstream M3 failure from pair suppression. No acceptance margin retuning.',
        parity=dict(hist_exact=True, ambient_atol=1e-6, z1_atol=.002, raw_atol=.005),
        anchors=len(manifest['anchors']), variants=3*len(manifest['anchors']),
        hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}))
    print('FROZEN', len(manifest['anchors']), 'anchors', flush=True)


def verify(plan):
    for path, expected in plan['hashes'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen input changed: '+path)


def run():
    plan = read(OUT/'PLAN.json'); verify(plan)
    if (OUT/'GPU_START.json').exists():
        raise FileExistsError('Single run only; inspect existing handle/terminal before explicit recovery')
    start = time.time(); save(OUT/'GPU_START.json', {'started_unix':start, 'deadline_unix':start+plan['budget_gpu_wall_seconds'], 'pid':os.getpid()})
    manifest = read(OUT/'geometry_manifest.json'); completed=[]
    from cnh_double_height_render_dev import DoubleHeightRunner, load_source
    def check():
        if time.time()-start >= plan['budget_gpu_wall_seconds']:
            raise TimeoutError('GPU wall budget exhausted; preserve partial outputs')
    try:
        with DoubleHeightRunner(OUT) as runner:
            first = manifest['anchors'][0]
            source = load_source(first['unit'], first['config'])
            parity = runner.anchor_parity(source, **{k:v for k,v in plan['parity'].items() if k != 'hist_exact'})
            save(OUT/'parity.json', parity)
            if not parity['passed']:
                raise ValueError('Unchanged anchor parity failed; do not measure intervention')
            print('PARITY', parity, flush=True)
            for index,a in enumerate(manifest['anchors']):
                check(); tick=time.monotonic(); source=load_source(a['unit'], a['config'])
                target=OUT/'anchors'/a['anchor_id']; target.mkdir(parents=True, exist_ok=True)
                for tag in TAGS:
                    check()
                    if tag == a['source_variant']:
                        data = {k:source[k] for k in ('hist','ambient','z1','nn','qq','raw','smooth')}
                        backend = {'kind':'reused original single-sensor observation and head_c score'}
                    else:
                        data = runner.render_variant(source, a['variants'][tag]['boxes'], tag)
                        backend = data.get('backend', {})
                    for k in ('nn','qq'):
                        np.testing.assert_array_equal(data[k], source[k])
                    dest=target/f'{tag}.npz'
                    if dest.exists(): raise FileExistsError(dest)
                    np.savez_compressed(dest, **{k:data[k] for k in ('hist','ambient','z1','nn','qq','raw','smooth')})
                    save(target/f'{tag}.json', dict(tag=tag, original_reused=tag==a['source_variant'], backend=backend, sha256=sha(dest)))
                    completed.append(a['anchor_id']+'/'+tag)
                save(OUT/'progress.json', dict(stage='render_infer',completed=len(completed),total=plan['variants'],
                    anchor=index+1, anchors=len(manifest['anchors']), elapsed_seconds=time.time()-start, last_activity_unix=time.time()), replace=True)
                print('ANCHOR',index+1,len(manifest['anchors']),a['anchor_id'],round(time.monotonic()-tick,2),'s',flush=True)
        check(); verify(plan)
        save(OUT/'gpu_terminal.json', dict(status='COMPLETE',completed=completed,seconds=time.time()-start,
            cleanup=runner.cleanup_receipt,pid=os.getpid()))
    except BaseException as exc:
        save(OUT/'gpu_terminal.json', dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',
            completed=completed,seconds=time.time()-start,error=repr(exc),traceback=traceback.format_exc(),
            cleanup=runner.cleanup_receipt if 'runner' in locals() else None,pid=os.getpid()))
        raise


def analyze():
    start=time.monotonic(); plan=read(OUT/'PLAN.json'); verify(plan)
    terminal=read(OUT/'gpu_terminal.json')
    if terminal['status']!='COMPLETE':
        raise ValueError('Incomplete intervention; retain missingness, no full analysis')
    sys.path.insert(0,str(WORK/'cnh-direction-information-dev-20261007/runtime'))
    import joblib
    from threadpoolctl import threadpool_limits
    ts=thresholds(read(PARENT/'result.json')); manifest=read(OUT/'geometry_manifest.json')
    models={f:joblib.load(PARENT/'models'/f'in_domain_fold{f}_pair_only.joblib') for f in range(3)}
    rows=[]; scores={}; witnesses=[]
    with threadpool_limits(limits=4):
        for a in manifest['anchors']:
            if time.monotonic()-start>plan['budget_analysis_cpu_seconds']: raise TimeoutError('analysis budget')
            per={}
            for tag in TAGS:
                p=OUT/'anchors'/a['anchor_id']/f'{tag}.npz'
                assert sha(p)==read(p.with_suffix('.json'))['sha256']
                with np.load(p) as z: x=z['smooth']
                assert x.shape==(13,2) and np.isfinite(x).all()
                per[tag]={'x':x,'pair_only':models[a['fold']].predict_proba(x)[:,1],'original_center':x.max(-1)}
                scores[a['anchor_id']+'/'+tag]=per[tag]
                v=a['variants'][tag]; d=v['deadline_index']
                for cap in ('0.025','0.050'):
                    for method in ('pair_only','original_center'):
                        theta=ts[a['fold'],cap,method]; st=decisions(per[tag][method],theta,d)
                        rows.append(dict(anchor=a['anchor_id'],unit=a['unit'],config=a['config'],fold=a['fold'],
                            role=a['role'],family=a['family'],mode=a['mode'],turn=a['turn'],depth_bin=a['depth_bin'],
                            variant=tag,cap=cap,method=method,threshold=theta,deadline=d,**st))
                rows.append(dict(anchor=a['anchor_id'],unit=a['unit'],config=a['config'],fold=a['fold'],
                    role=a['role'],family=a['family'],mode=a['mode'],turn=a['turn'],depth_bin=a['depth_bin'],
                    variant=tag,cap='legacy',method='original_center',threshold=.8557642486787612,deadline=d,
                    **decisions(per[tag]['original_center'],.8557642486787612,d)))
            if a['role']=='contact':
                for cap in ('0.025','0.050'):
                    theta=ts[a['fold'],cap,'pair_only']; ct=ts[a['fold'],cap,'original_center']; d=a['variants']['HB']['deadline_index']
                    hb=decisions(per['HB']['pair_only'],theta,d)
                    for old in ('H','B'):
                        oldst=decisions(per[old]['pair_only'],theta,d)
                        dom=np.all(per['HB']['x']>=per[old]['x'],axis=1)
                        frame=np.flatnonzero(dom & (per[old]['pair_only']>=theta) & (per['HB']['pair_only']<theta) & (np.arange(13)<=d))
                        witnesses.append(dict(anchor=a['anchor_id'],cap=cap,from_variant=old,
                            single_timely=oldst['timely'],hb_timely=hb['timely'],loss=oldst['timely'] and not hb['timely'],
                            hb_center_timely=decisions(per['HB']['original_center'],ct,d)['timely'],
                            dominating_suppression_frames=frame.tolist(),
                            single_logits_deadline=per[old]['x'][d].tolist(),hb_logits_deadline=per['HB']['x'][d].tolist(),
                            single_pair_deadline=float(per[old]['pair_only'][d]),hb_pair_deadline=float(per['HB']['pair_only'][d])))
    groups={'all':lambda r:True}
    groups.update({f'family/{v}':lambda r,v=v:r['family']==v for v in ('none','corner')})
    groups.update({f'mode/{v}':lambda r,v=v:r['mode']==v for v in range(3)})
    groups.update({f'fold/{v}':lambda r,v=v:r['fold']==v for v in range(3)})
    groups.update({f'depth/{v}':lambda r,v=v:r['depth_bin']==v for v in sorted({a['depth_bin'] for a in manifest['anchors'] if a['role']=='contact'})})
    aggregate={}
    for cap,method in [(c,m) for c in ('0.025','0.050') for m in ('pair_only','original_center')]+[('legacy','original_center')]:
        for tag in TAGS:
            for group,keep in groups.items():
                subset=[r for r in rows if r['cap']==cap and r['method']==method and r['variant']==tag and keep(r)]
                e=[r for r in subset if r['role']=='contact']; c=[r for r in subset if r['role']=='clear']
                aggregate[f'{cap}/{method}/{tag}/{group}']=dict(events=len(e),timely=sum(r['timely'] for r in e),
                    timely_excluding_warmup=sum(r['timely_excluding_warmup'] for r in e),controls=len(c),
                    false_alarm_intervals=sum(r['false_alarm_intervals'] for r in c),control_intervals=10*len(c),
                    warmup_alarm_frames=sum(r['warmup_alarm_frames'] for r in c),warmup_control_frames=2*len(c))
    np.savez_compressed(OUT/'score_ledger.npz',**{a+'/'+k:v for a,ss in scores.items() for k,v in ss.items()})
    save(OUT/'result.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,
        anchors=len(manifest['anchors']),independent_units=len({a['unit'] for a in manifest['anchors']}),
        plan_sha256=sha(OUT/'PLAN.json'),metrics=aggregate,rows=rows,paired_height_witnesses=witnesses,
        limitation='Consumed source units with new physical height interventions, same simulator, frozen OOF models; no threshold refit, no tri-state/safety/real-world claims'))
    for key,value in aggregate.items():
        if key.startswith('0.025/') and key.endswith('/all'): print('RESULT',key,value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze'])
    globals()[parser.parse_args().stage]()
