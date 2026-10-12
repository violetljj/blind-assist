"""Independent post-result prior/evidence audit; no training or protected reads."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from audit_sync_fusion_robust_dev import (load, sha, read_npz, assert_close,
    independent_ties, independent_select, cutoff, compact_ci)


def run(root, addon_plan_path, budget_s):
    import joblib
    from threadpoolctl import threadpool_limits
    from sync_rgb_tof_dataset_v1 import peak_readout
    from rgb_body_query_reference_eval import rays, ray_interval

    started = time.monotonic()
    def check():
        if time.monotonic()-started >= budget_s-3:
            raise TimeoutError('Independent addon audit allocation')

    p = load(addon_plan_path)
    original_plan = load(root/'PLAN.json')
    assert sha(root/'evidence_addendum_PLAN.json') == sha(addon_plan_path)
    assert p['budget']['no_budget_expansion'] is True
    assert p['budget']['original_CPU_command_wall_limit_s'] == 1800
    assert p['budget']['audit_cap_s'] == 90
    for name, filename in (('PLAN','PLAN.json'),('features','features.npz'),('oof','oof.npz'),('summary','summary.json')):
        assert sha(root/filename) == p['original'][name+'_sha256']
    terminal = load(root/'evidence_addendum_terminal.json')
    assert terminal['status'] == 'COMPLETE' and terminal['GPU_s'] == 0
    assert terminal['source_sha256'] == sha(Path(__file__).with_name('sync_fusion_robust_evidence_dev.py'))
    d, original = read_npz(root/'features.npz'), read_npz(root/'oof.npz')
    evidence = read_npz(root/'evidence.npz')
    e = load(root/'evidence_summary.json')
    sources = load(root/'evidence_sources.json')
    assert sources['ADDPLAN_sha256'] == e['ADDPLAN_sha256'] == terminal['ADDPLAN_sha256'] == sha(addon_plan_path)
    n = len(d['y']); arms = original_plan['arms']
    counts = evidence['support_counts']
    assert counts.shape == (n,2) and counts.dtype.kind in ('i','u') and (counts>=0).all()
    np.testing.assert_array_equal(counts.mean(1), d['X'][:,2])
    classes = dict(raw_bothK=(counts>=16).all(1).astype(int)+2*(d['rgb_score']>=0).astype(int),
                   raw_mean_margin=(d['tof_score']>=0).astype(int)+2*(d['rgb_score']>=0).astype(int),
                   cal_decisions=original['pred'][:,5].astype(int)+2*original['pred'][:,4].astype(int))
    for name, value in classes.items():
        np.testing.assert_array_equal(evidence[name], value)
        assert ((value>=0)&(value<=3)).all()
    assert np.all(~((counts>=16).all(1)) | (d['tof_score']>=0))
    assert e['bothK_missing_but_mean_supported'] == sum(~(counts>=16).all(1)&(d['tof_score']>=0))
    assert e['raw_neither_queries'] == sum(classes['raw_bothK']==0)
    assert e['original_results_unchanged'] is True

    source_lookup = {(r['frame_id'],r['arm'],r['K']): r for r in sources['paths']}
    assert len(source_lookup) == len(sources['paths']) == 10056
    sampled_frames, sampled_queries = 0, 0
    for arm in arms:
        frames = [r for r in sources['sample_frames'] if r['arm'] == arm]
        assert frames
        for j in np.unique(np.linspace(0,len(frames)-1,min(6,len(frames)),dtype=int)):
            check(); frame = frames[j]; ids = np.asarray(frame['indices'],int)
            assert len(ids)==27
            np.testing.assert_array_equal(d['arm'][ids], np.full(27,arm))
            np.testing.assert_array_equal(d['frame_id'][ids], np.full(27,frame['frame_id']))
            assert d['query_id'][ids].tolist() == [q['name'] for q in frame['queries']]
            K = np.asarray(frame['K']); shape = tuple(frame['shape']); rx,ry = rays(K,shape)
            for repeat in (0,1):
                row = source_lookup[(frame['frame_id'],arm,repeat)]
                assert sha(row['path']) == row['sha256']
                z = read_npz(row['path'])
                depth = peak_readout(z['hist'],z['background'],z['coverage'],z['grid_K'],shape,z['grid_pose'],K,shape,z['rgb_pose'],frame['seal'])[0] if np.isfinite(z['grid_pose']).all() else np.full(shape,np.inf)
                wanted = []
                for q in frame['queries']:
                    entry,exit_,domain = ray_interval(rx,ry,q)
                    inside = domain & np.isfinite(depth) & (depth>0) & (depth>=entry) & (depth<=exit_)
                    wanted.append(int(np.count_nonzero(inside)))
                np.testing.assert_array_equal(counts[ids,repeat], wanted)
                sampled_queries += len(ids)
            sampled_frames += 1
    print('addon audit evidence PASS', sampled_frames, sampled_queries, flush=True)

    prior = root/'prior'; prior_oof = read_npz(prior/'oof.npz')
    assert prior_oof['scores'].shape == prior_oof['pred'].shape == (n,)
    assert np.isfinite(prior_oof['scores']).all()
    np.testing.assert_array_equal(prior_oof['fold'],original['fold'])
    assert p['prior']['indices'] == [16,17,18,19]
    assert p['prior']['features'] == d['feature_names'][16:20].tolist()
    X = d['X'][:,16:20]
    entries = load(prior/'models.json')
    assert len(entries)==15
    cuts = {(r['fold'],r['band']): r for r in load(prior/'cuts.json')}
    assert len(cuts)==15
    tie_count = 0
    with threadpool_limits(limits=2):
        for fold in original_plan['data']['folds']:
            check(); f=fold['fold']; cal=np.flatnonzero(np.isin(d['visit_id'],fold['cal'])); ev=np.flatnonzero(original['fold']==f)
            training=read_npz(root/f'fold{f}'/'B_training.npz')
            train=training['indices']; weights=training['weights']
            wanted_train=np.flatnonzero(np.isin(d['visit_id'],fold['train'])&(d['y']>=0))
            np.testing.assert_array_equal(train,wanted_train)
            expected_weights=np.zeros(len(train))
            for arm in arms:
                mask=d['arm'][train]==arm
                expected_weights[mask]=len(train)/(6*mask.sum())
            np.testing.assert_allclose(weights,expected_weights,rtol=1e-13)
            cal_saved=read_npz(prior/f'cal{f}.npz')
            np.testing.assert_array_equal(cal_saved['indices'],cal)
            model_entries=[r for r in entries if r['fold']==f]
            assert sorted(r['seed'] for r in model_entries)==[955,956,957]
            cal_predictions, eval_predictions = [], []
            for row in model_entries:
                assert sha(row['path'])==row['sha256']
                model=joblib.load(row['path']); m,scaler=model['estimator'],model['scaler']
                assert model['features']==p['prior']['features']
                parameters=m.get_params()
                assert parameters['C']==1 and parameters['solver']=='lbfgs' and parameters['max_iter']==1000
                assert parameters['class_weight'] is None and parameters['random_state']==row['seed']
                assert parameters['penalty']=='l2' or (parameters['penalty']=='deprecated' and parameters['l1_ratio']==0)
                mean=np.average(X[train],axis=0,weights=weights)
                var=np.average((X[train]-mean)**2,axis=0,weights=weights)
                np.testing.assert_allclose(scaler.mean_,mean,atol=1e-10)
                np.testing.assert_allclose(scaler.var_,var,atol=1e-10)
                assert np.isfinite(scaler.scale_).all() and (scaler.scale_>0).all()
                cal_predictions.append(m.predict_proba(scaler.transform(X[cal]))[:,1])
                eval_predictions.append(m.predict_proba(scaler.transform(X[ev]))[:,1])
            np.testing.assert_allclose(cal_saved['scores'],np.mean(cal_predictions,axis=0),atol=1e-12)
            np.testing.assert_allclose(prior_oof['scores'][ev],np.mean(eval_predictions,axis=0),atol=1e-12)
            for b,target in enumerate((.02,.05,.10)):
                take=d['band'][cal]==b
                expected_ties=independent_ties(cal_saved['scores'][take],d['y'][cal][take])
                actual_ties=load(prior/f'ties{f}_{b}.json')
                assert len(expected_ties)==len(actual_ties)
                for want,actual in zip(expected_ties,actual_ties):
                    assert want==(cutoff(actual),actual['W'],actual['F'],actual['U'])
                expected,status=independent_select(cal_saved['scores'][take],d['y'][cal][take],target)
                actual=cuts[(f,b)]
                assert actual['status']==status
                assert expected==(cutoff(actual),actual['W'],actual['F'],actual['U'])
                selected=ev[d['band'][ev]==b]
                np.testing.assert_array_equal(prior_oof['pred'][selected],(prior_oof['scores'][selected]>=expected[0])&(status=='ADOPTED'))
                tie_count+=len(expected_ties)
            print('addon audit prior fold PASS',f,flush=True)

    prior_metrics={(r['fold'],r['arm'],r['band']):r for r in e['prior_metrics']}
    split={(r['fold'],r['arm'],r['band'],r['model'],r['variant'],r['category']):r for r in e['evidence_split']}
    assert len(prior_metrics)==108 and len(split)==5184
    category_names=('neither','tof_only','rgb_only','both')
    for f in (-1,0,1,2,3,4):
        visits=original_plan['data']['visits'] if f==-1 else original_plan['data']['folds'][f]['eval']
        draw=np.random.default_rng(20261011).integers(0,len(visits),size=(2000,len(visits)))
        scope=np.ones(n,bool) if f==-1 else original['fold']==f
        for arm in arms:
            for b in range(3):
                check(); take=np.flatnonzero(scope&(d['arm']==arm)&(d['band']==b))
                y,v=d['y'][take],d['visit_id'][take]; pp=prior_oof['pred'][take]
                den=np.array([[sum((v==visit)&(y==k)) for k in (1,0)] for visit in visits]); bn=np.array([[sum((v==visit)&(y==k)&pp) for k in (1,0)] for visit in visits])
                totals=den.sum(0); counts_total=bn.sum(0); bd,bc=den[draw].sum(1),bn[draw].sum(1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    expected=dict(model='prior',POS=int(totals[0]),W=int(counts_total[0]),FREE=int(totals[1]),F=int(counts_total[1]),W_rate_ci=compact_ci(100*bc[:,0]/bd[:,0]),F_rate_ci=compact_ci(100*bc[:,1]/bd[:,1]))
                assert_close(prior_metrics[(f,arm,b)],expected)
                original_pred=original['pred'][take]
                rgb,tof=original_pred[:,4],original_pred[:,5]
                for variant,all_classes in classes.items():
                    cl=all_classes[take]
                    for mi,model in enumerate(('A','B','C','D')):
                        mp=original_pred[:,mi]
                        for category,name in enumerate(category_names):
                            pos=(y==1)&(cl==category); free=(y==0)&(cl==category)
                            expected=dict(POS=int(sum(pos)),W=int(sum(pos&mp)),FREE=int(sum(free)),F=int(sum(free&mp)),rescue_rgb=int(sum(pos&mp&~rgb)),loss_rgb=int(sum(pos&~mp&rgb)),rescue_tof=int(sum(pos&mp&~tof)),loss_tof=int(sum(pos&~mp&tof)))
                            assert_close(split[(f,arm,b,model,variant,name)],expected)
        print('addon audit summary PASS',f,flush=True)

    with (root/'per_query_evidence.csv').open(newline='',encoding='utf8') as stream:
        rows=0
        for i,row in enumerate(csv.DictReader(stream)):
            assert int(row['index'])==i
            for key in ('arm','visit_id','frame_id','query_id'):
                assert row[key]==d[key][i]
            assert int(row['fold'])==original['fold'][i]
            assert int(row['band'])==d['band'][i] and int(row['y'])==d['y'][i]
            assert int(row['tof_K0_support'])==counts[i,0] and int(row['tof_K1_support'])==counts[i,1]
            assert (row['RGB_raw_minimum']=='True')==(d['rgb_score'][i]>=0)
            for variant,key in (('raw_bothK','raw_bothK_category'),('raw_mean_margin','raw_mean_margin_category'),('cal_decisions','cal_category')):
                assert row[key]==category_names[classes[variant][i]]
            assert float(row['prior_score'])==prior_oof['scores'][i]
            assert (row['prior_pred']=='True')==prior_oof['pred'][i]
            rows+=1
        assert rows==n
    for name,filename in (('PLAN','PLAN.json'),('features','features.npz'),('oof','oof.npz'),('summary','summary.json')):
        assert sha(root/filename)==p['original'][name+'_sha256']
    result=dict(status='PASS',ADDPLAN_sha256=sha(addon_plan_path),rows=n,sampled_frame_arms=sampled_frames,sampled_K_queries=sampled_queries,
                support_count_row_means_checked=n,prior_models=15,prior_cal_ties=tie_count,prior_thresholds=15,prior_metrics=108,
                evidence_split_cells=5184,bootstrap_replicates=2000,original_results_unchanged=True,
                original_features_sha256=sha(root/'features.npz'),original_oof_sha256=sha(root/'oof.npz'),original_summary_sha256=sha(root/'summary.json'),
                evidence_sha256=sha(root/'evidence.npz'),evidence_summary_sha256=sha(root/'evidence_summary.json'),per_query_evidence_sha256=sha(root/'per_query_evidence.csv'),
                prior_oof_sha256=sha(prior/'oof.npz'),audit_source_sha256=sha(__file__),CPU_command_wall_s=time.monotonic()-started,GPU_s=0,protected_reads=0,fits_performed=0,
                boundary='Post-result consumed-Development addendum; neither raw minimum evidence is absent local support, not proof of pure-prior causality')
    (root/'independent_addon_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--budget-s',type=float,default=90)
    args=ap.parse_args()
    run(args.root.resolve(),args.plan.resolve(),args.budget_s)
