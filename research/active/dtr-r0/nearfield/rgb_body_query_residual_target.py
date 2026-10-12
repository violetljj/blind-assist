"""Train/cal-only bounded log-Z targets and a fixed local compatibility proxy.

Public finite-DP sampling is frozen before reference pixels are read. No model,
optimizer, new inference, downloads, evaluation cohort, or threshold selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import shutil
import time
EXEC_STARTED = time.perf_counter()

import numpy as np
from scipy.spatial import cKDTree
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_3rscan import color_coordinates, sample_prediction
from rgb_body_query_reference_eval import rays

PUBLIC = {'environment','scan','split','frame','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape'}
COHORTS = ('original_train56','original_cal16','additional_3rscan240','additional_arkit48')
BANDS = ('below_0.3m','0.3-0.8m','0.8-1.5m','1.5-3m','at_least_3m')


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def bands(z):
    return np.searchsorted(np.array([.3,.8,1.5,3.]),z,side='right').astype(np.int8)


def public_mapping(row):
    # Registered ARKit color/depth share the exact native pixel grid. Retain
    # its declared identity map, rather than introduce inverse-K roundoff.
    if (row['color_shape']==row['depth_shape'] and
            np.array_equal(np.asarray(row['color_K']),np.asarray(row['depth_K']))):
        my,mx = np.indices(row['depth_shape'],dtype=np.float32)
        return mx,my
    return color_coordinates(np.asarray(row['depth_K']),np.asarray(row['color_K']),row['depth_shape'])


def quantile_summary(value):
    value = np.asarray(value,np.float64)
    if not value.size:
        return dict(points=0,status='NOT_EVALUABLE')
    return dict(points=int(value.size),mean=float(value.mean()),median=float(np.median(value)),
        q01_q05_q25_q75_q95_q99=np.quantile(value,[.01,.05,.25,.75,.95,.99]).tolist(),
        minimum=float(value.min()),maximum=float(value.max()),negative=int((value<0).sum()),
        zero=int((value==0).sum()),positive=int((value>0).sum()))


def target_summary(residual, bounds):
    residual = np.asarray(residual,np.float64)
    out = quantile_summary(residual)
    if not residual.size:
        return out
    absolute = np.abs(residual)
    out.update(absolute=quantile_summary(absolute),zero_residual_mae=float(absolute.mean()),
        zero_residual_median_absolute=float(np.median(absolute)),zero_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        bounds=[dict(rule=name,delta=float(delta),reachable=int((absolute<=delta).sum()),
            reachable_fraction=float((absolute<=delta).mean()),
            irreducible_bounded_oracle_mae=float(np.maximum(absolute-delta,0).mean())) for name,delta in bounds])
    return out


def error_stats(error):
    return dict(mae=float(np.mean(np.abs(error))),median_absolute=float(np.median(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),signed_mean=float(error.mean()),
        signed_median=float(np.median(error)),absolute_p95=float(np.quantile(np.abs(error),.95)))


def proxy_summary(residual, median, distances, q25, q75, bounds):
    n = residual.size
    if not n:
        return dict(points=0,status='NOT_EVALUABLE')
    baseline = -residual
    out = dict(points=int(n),target_residual=quantile_summary(residual),
        neighbor_median=quantile_summary(median),neighbor_iqr=quantile_summary(q75-q25),
        nearest_input_distance=quantile_summary(distances[:,0]),
        kth_input_distance=quantile_summary(distances[:,-1]),baseline_zero=error_stats(baseline),
        unbounded_local_median=error_stats(median-residual),
        direction_same=int((np.sign(median)==np.sign(residual)).sum()),
        direction_opposed=int((median*residual<0).sum()),
        target_inside_neighbor_iqr=int(((residual>=q25)&(residual<=q75)).sum()),
        target_outside_neighbor_iqr=int(((residual<q25)|(residual>q75)).sum()),bounds=[])
    for name,delta in bounds:
        error = np.clip(median,-delta,delta)-residual
        out['bounds'].append(dict(rule=name,delta=float(delta),**error_stats(error),
            reachable_target=int((np.abs(residual)<=delta).sum()),
            improved_absolute_error=int((np.abs(error)<np.abs(baseline)).sum()),
            worsened_absolute_error=int((np.abs(error)>np.abs(baseline)).sum()),
            equal_absolute_error=int((np.abs(error)==np.abs(baseline)).sum())))
    out['candidate_rule'] = bounds[3][0]
    out['candidate'] = out['bounds'][3]
    return out


def run(repo, output, budget_s):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'terminal.json').exists():
        raise FileExistsError('Preserve completed/failure allocation; no implicit restart')
    with (output/'writer.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),script=str(Path(__file__).resolve())),f)
    receipt = dict(status='STARTING',pid=os.getpid(),budget_cpu_allocation_wall_s=budget_s,
        completed_public_frames=0,completed_target_frames=0,completed_neighbor_points=0,
        gpu_s=0,optimizer_calls=0,training_calls=0,new_inference_calls=0,downloads_bytes=0,
        backend='scipy-cKDTree CPU',placement_reason='TASK_NOT_GPU_SUITABLE',workers=4,
        evaluation_cohort_reads=False,train_cal_reference_target_reads=False,failures=[])
    tree = None
    try:
        def check(stage):
            elapsed = time.perf_counter()-EXEC_STARTED
            if elapsed >= budget_s-3:
                raise TimeoutError(f'Cumulative CPU allocation at {stage}')
            receipt.update(stage=stage,cpu_allocation_wall_s=elapsed)
        source_hash = sha(__file__)
        shutil.copyfile(__file__,output/'runner_executed.py')
        parent_plan = output.parent/'plan.json'
        plan = load(parent_plan)
        assert plan['budgets']['diagnostic_wall_s'] == 700
        receipt.update(source_sha256=source_hash,parent_plan_sha256=sha(parent_plan))
        work = repo/'artifacts.local/work'
        original = work/'rgb-body-query-interval-distribution-dev-20261010/train-cal-sensor'
        extra = work/'rgb-body-query-negative-reference-dev-20261010'
        extra_dp = work/'rgb-body-query-negative-score-dev-20261010/additional-inference'
        arkit = work/'rgb-body-query-arkit-cal-dev-20261010/reference'
        arkit_dp = work/'rgb-body-query-arkit-cal-dev-20261010/additional-inference'
        training_info_path = work/'rgb-body-query-interval-distribution-dev-20261010/candidate/training_inputs.json'
        fit_path = work/'rgb-body-query-calibrated-transfer-dev-20261009/geometry/fit.json'
        specs = [('original_train56',original,original/'depthpro','train',56),
                 ('original_cal16',original,original/'depthpro','cal',16),
                 ('additional_3rscan240',extra,extra_dp,'additional_cal',240),
                 ('additional_arkit48',arkit,arkit_dp,'additional_cal',48)]
        # Phase A only public observation/prediction manifests and cached DP.
        native_folder = output/'public-native'; native_folder.mkdir()
        roster, source_inputs = [], []
        for cohort, obs_folder, pred_folder, split, count in specs:
            check('public_selection')
            op,pp = obs_folder/'observations.json',pred_folder/'predictions.json'
            obs,pm = load(op),load(pp)
            assert pm['status']=='COMPLETE'
            rr = [r for r in obs['rows'] if r['split']==split]
            assert len(rr)==count and all(set(r)==PUBLIC for r in rr)
            lookup = {(p['scan'],p['frame']):p for p in pm['rows']}
            source_inputs.extend(dict(path=str(p),sha256=sha(p),role='public') for p in (op,pp))
            for r in rr:
                check('public_native_cache')
                p = lookup[r['scan'],r['frame']]
                assert p['rgb_sha256']==r['rgb_sha256'] and sha(p['path'])==p['sha256']
                with np.load(p['path']) as f: full = f['depth']
                assert list(full.shape)==r['color_shape']
                mx,my = public_mapping(r)
                dp = sample_prediction(full,mx,my)
                eligible = np.flatnonzero((np.isfinite(dp)&(dp>0)).ravel())
                assert eligible.size>=512
                selected = eligible[np.rint(np.linspace(0,eligible.size-1,512)).astype(np.int64)]
                assert np.unique(selected).size==512
                cache = native_folder/f'{len(roster):03d}_{r["scan"]}_{r["frame"]:06d}.npz'
                np.savez(cache,depth=dp,sample_flat_index=selected.astype(np.int32))
                roster.append(dict(r,cohort=cohort,role='train' if cohort=='original_train56' else 'cal',
                    frame_index=len(roster),prediction_path=p['path'],prediction_sha256=p['sha256'],
                    public_native_path=str(cache),public_native_sha256=sha(cache),
                    public_finite_positive_dp_pixels=int(eligible.size),public_sample_points=512))
                receipt['completed_public_frames'] = len(roster)
                if len(roster)%32==0:
                    write(output/'progress.json',receipt)
                    print(f'PUBLIC {len(roster)}/360 wall={time.perf_counter()-EXEC_STARTED:.2f}s',flush=True)
        assert len(roster)==len({(r['scan'],r['frame']) for r in roster})==360
        assert len({r['environment'] for r in roster if r['role']=='train'})==7
        train_ids = {(r['scan'],r['frame']) for r in roster if r['role']=='train'}
        assert train_ids.isdisjoint({(r['scan'],r['frame']) for r in roster if r['role']=='cal'})
        assert {r['environment'] for r in roster if r['role']=='train'}.isdisjoint(
            {r['environment'] for r in roster if r['role']=='cal'})
        write(output/'public_roster.json',dict(status='FROZEN_BEFORE_REFERENCE_PIXELS',rows=roster,
            input_manifests=source_inputs,selection='512 rounded linspace indices over native public finite positive DP flat indices per frame; no GT filtering or replacement',
            public_selection_wall_s=time.perf_counter()-EXEC_STARTED,
            reference_pixel_reads=False,evaluation_cohort_reads=False))
        receipt['public_roster_sha256'] = sha(output/'public_roster.json')
        print(f'PUBLIC_FROZEN 360 frames wall={time.perf_counter()-EXEC_STARTED:.2f}s',flush=True)

        # Phase B target references remain train/cal; never evaluation cohorts.
        receipt['train_cal_reference_target_reads'] = True
        reference_lookup = {}
        for cohort,folder,_,split,count in specs:
            mp = folder/'dataset_manifest.json'
            manifest = load(mp)
            selected = [r for r in manifest['rows'] if r['split']==split]
            assert len(selected)==count
            reference_lookup.update({(cohort,r['scan'],r['frame']):r for r in selected})
            source_inputs.append(dict(path=str(mp),sha256=sha(mp),role='train_cal_target_registry'))
        info,fit = load(training_info_path),load(fit_path)
        affine = info['affine']; a,b = affine['a'],affine['b']
        assert fit['models']['log_affine']['a']==a and fit['models']['log_affine']['b']==b
        assert info['points']==1591791
        old_train = {(r['scan'],r['frame']):r for r in info['frames']}
        assert set(old_train)==train_ids
        source_inputs.extend(dict(path=str(p),sha256=sha(p),role='frozen_original_training_identity')
                             for p in (training_info_path,fit_path))
        train_environments = sorted({r['environment'] for r in roster if r['role']=='train'})
        environments = sorted({r['environment'] for r in roster})
        train_parts, all_parts, sample_parts, target_rows = [],[],[],[]
        frame_target_parts = []
        for r in roster:
            check('target_arrays')
            ref = reference_lookup[r['cohort'],r['scan'],r['frame']]
            assert ref['rgb_sha256']==r['rgb_sha256']
            assert sha(ref['reference_path'])==ref['reference_sha256']
            with np.load(r['public_native_path']) as f:
                dp = f['depth']; selected = f['sample_flat_index']
            with np.load(ref['reference_path']) as f:
                z = f['depth']; observed = f['observed']
                assert np.array_equal(f['depth_K'],np.asarray(r['depth_K']))
                assert np.array_equal(f['color_K'],np.asarray(r['color_K']))
                mx,my = public_mapping(r)
                assert np.array_equal(f['map_x'],mx) and np.array_equal(f['map_y'],my)
            assert z.shape==observed.shape==dp.shape and list(z.shape)==r['depth_shape']
            valid = observed & np.isfinite(z)&(z>0)&np.isfinite(dp)&(dp>0)
            idx = np.flatnonzero(valid.ravel()).astype(np.int32)
            rx,ry = rays(r['depth_K'],r['depth_shape'])
            raw = np.stack([np.log(dp.ravel()[idx].astype(np.float64)),rx.ravel()[idx],ry.ravel()[idx]],axis=1)
            residual = np.log(z.ravel()[idx].astype(np.float64))-(a*raw[:,0]+b)
            gt = z.ravel()[idx]
            assert np.isfinite(residual).all() and np.isfinite(raw).all()
            if r['role']=='train':
                old = old_train[r['scan'],r['frame']]
                assert old['paired_pixels']==idx.size
                assert old['reference_path']==ref['reference_path'] and old['reference_sha256']==ref['reference_sha256']
                assert old['prediction_path']==r['prediction_path'] and old['prediction_sha256']==r['prediction_sha256']
                train_parts.append(dict(features_raw=raw,residual=residual,gt_z=gt,flat_index=idx,
                    frame_index=np.full(idx.size,r['frame_index'],np.int16),
                    environment_index=np.full(idx.size,train_environments.index(r['environment']),np.int8)))
            all_parts.append(dict(residual=residual,gt_z=gt,dp_z=dp.ravel()[idx],flat_index=idx,
                frame_index=np.full(idx.size,r['frame_index'],np.int16),gt_band=bands(gt)))
            frame_target_parts.append((residual,gt))
            sample_raw = np.stack([np.log(dp.ravel()[selected].astype(np.float64)),rx.ravel()[selected],ry.ravel()[selected]],axis=1)
            sample_valid = valid.ravel()[selected]
            sample_gt = np.full(512,np.nan,np.float64); sample_gt[sample_valid] = z.ravel()[selected[sample_valid]]
            sample_target = np.full(512,np.nan,np.float64)
            sample_target[sample_valid] = np.log(sample_gt[sample_valid])-(a*sample_raw[sample_valid,0]+b)
            sample_parts.append(dict(features_raw=sample_raw,flat_index=selected,
                frame_index=np.full(512,r['frame_index'],np.int16),target_eligible=sample_valid,
                gt_z=sample_gt,residual=sample_target))
            target_rows.append(dict(r,reference_path=ref['reference_path'],reference_sha256=ref['reference_sha256'],
                eligible_target_points=int(idx.size),sample_target_points=int(sample_valid.sum()),
                sampled_missing_reference_points=int((~sample_valid).sum()),
                reference_native_pixels=int(z.size),reference_observed_pixels=int(observed.sum())))
            receipt['completed_target_frames'] = len(target_rows)
            if len(target_rows)%32==0:
                write(output/'progress.json',receipt)
                print(f'TARGET {len(target_rows)}/360 wall={time.perf_counter()-EXEC_STARTED:.2f}s',flush=True)
        bank = {key:np.concatenate([p[key] for p in train_parts]) for key in train_parts[0]}
        assert bank['residual'].size==1591791
        mean,std = bank['features_raw'].mean(axis=0),bank['features_raw'].std(axis=0)
        assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
        bank['features'] = (bank['features_raw']-mean)/std
        all_targets = {key:np.concatenate([p[key] for p in all_parts]) for key in all_parts[0]}
        local = {key:np.concatenate([p[key] for p in sample_parts]) for key in sample_parts[0]}
        local['features'] = (local['features_raw']-mean)/std
        local['gt_band'] = np.full(local['residual'].size,-1,np.int8)
        local['gt_band'][local['target_eligible']] = bands(local['gt_z'][local['target_eligible']])
        cal_mask = all_targets['frame_index']>=56
        train_abs = np.abs(bank['residual']); cal_abs = np.abs(all_targets['residual'][cal_mask])
        tq95,tq99 = np.quantile(train_abs,[.95,.99]); cq99 = np.quantile(cal_abs,.99)
        bounds = [('fixed_delta_0.2',.2),('train_abs_q95',float(tq95)),('train_abs_q99',float(tq99)),
                  ('train_cal_q99_envelope',float(max(tq99,cq99)))]
        write(output/'bounds.json',dict(affine=affine,bound_rules=[dict(rule=n,delta=d) for n,d in bounds],
            candidate_rule=bounds[3][0],candidate_delta=bounds[3][1],train_points=int(train_abs.size),
            cal_points=int(cal_abs.size),train_q95=float(tq95),train_q99=float(tq99),cal_q99=float(cq99),
            normalization=dict(features=['logDP','ray_x','ray_y'],mean=mean.tolist(),std=std.tolist(),
                role='all original eligible training inputs only; fixed across LOEO, no GT/band/sourceID features'),
            band_partition='below0.3; [0.3,0.8); [0.8,1.5); [1.5,3); >=3 metres',bands=list(BANDS)))
        write(output/'target_roster.json',dict(rows=target_rows,input_manifests=source_inputs,
            public_roster_sha256=receipt['public_roster_sha256'],affine=affine,
            eligibility='observed & finite positive reference optical-Z & finite positive native public DP; same old train mask/maps',
            train_environments=train_environments,all_environments=environments,
            train_cal_reference_target_reads=True,evaluation_cohort_reads=False))
        check('save_compact_arrays')
        np.savez_compressed(output/'train_bank.npz',**bank)
        np.savez_compressed(output/'all_targets.npz',**all_targets)
        # Banks and targets are cached once; neighbor queries contain public inputs only.
        n = local['residual'].size
        local['neighbor_bank_index'] = np.full((n,16),-1,np.int32)
        local['neighbor_distance'] = np.full((n,16),np.nan,np.float64)
        local['neighbor_median'] = np.full(n,np.nan,np.float64)
        local['neighbor_q25'] = np.full(n,np.nan,np.float64)
        local['neighbor_q75'] = np.full(n,np.nan,np.float64)
        frame_env = np.array([train_environments.index(r['environment']) if r['role']=='train' else -1 for r in roster],np.int8)
        local['environment_index'] = frame_env[local['frame_index']]
        queries = [(f'train_LOEO_{env}',local['environment_index']==e,bank['environment_index']!=e)
                   for e,env in enumerate(train_environments)]
        queries.append(('pooled304cal',local['frame_index']>=56,np.ones(bank['residual'].size,bool)))
        query_receipts = []
        for label,selected,allowed in queries:
            check('build_neighbor_bank')
            group_started = time.perf_counter()
            bank_index = np.flatnonzero(allowed).astype(np.int32)
            tree = cKDTree(bank['features'][bank_index],copy_data=False,balanced_tree=True)
            qidx = np.flatnonzero(selected)
            for chunk in np.array_split(qidx,max(1,int(np.ceil(qidx.size/8192)))):
                check('query_fixed_16_neighbors')
                distance,neighbor = tree.query(local['features'][chunk],k=16,workers=4)
                neighbor = bank_index[neighbor]
                if label.startswith('train_LOEO'):
                    assert np.all(bank['environment_index'][neighbor]!=local['environment_index'][chunk,None])
                values = bank['residual'][neighbor]
                q25,median,q75 = np.quantile(values,[.25,.5,.75],axis=1)
                local['neighbor_bank_index'][chunk]=neighbor
                local['neighbor_distance'][chunk]=distance
                local['neighbor_median'][chunk]=median
                local['neighbor_q25'][chunk]=q25
                local['neighbor_q75'][chunk]=q75
                receipt['completed_neighbor_points'] += chunk.size
                write(output/'progress.json',receipt)
            query_receipts.append(dict(group=label,query_points=int(qidx.size),bank_points=int(bank_index.size),
                same_environment_excluded=label.startswith('train_LOEO'),wall_s=time.perf_counter()-group_started))
            tree = None; gc.collect()
            print(f'NEIGHBORS {label} points={qidx.size} wall={time.perf_counter()-EXEC_STARTED:.2f}s',flush=True)
        assert receipt['completed_neighbor_points']==n==360*512
        local['neighbor_iqr'] = local['neighbor_q75']-local['neighbor_q25']
        local['proxy_clipped'] = np.column_stack([np.clip(local['neighbor_median'],-delta,delta) for _,delta in bounds])
        local['baseline_zero_error'] = -local['residual']
        local['unbounded_local_error'] = local['neighbor_median']-local['residual']
        local['clipped_local_error'] = local['proxy_clipped']-local['residual'][:,None]
        np.savez_compressed(output/'local_proxy.npz',**local)
        write(output/'neighbor_bank_receipt.json',dict(queries=query_receipts,k=16,aggregation='median',
            train_bank_sha256=sha(output/'train_bank.npz'),normalization=load(output/'bounds.json')['normalization'],
            local_proxy_sha256=sha(output/'local_proxy.npz'),bank_points=1591791,
            public_sample_points=n,reference_eligible_sample_points=int(local['target_eligible'].sum()),
            no_query_reference_or_band_features=True,cal_not_in_bank=True,train_LOEO_excludes_same_environment=True))
        check('summaries')
        # Full native targets: pixel pooling and independent frame/environment units.
        full_groups = dict(all=target_summary(all_targets['residual'],bounds),cohort={},environment={},
            band={},cohort_band={},environment_band={},frame=[])
        local_groups = dict(all={},cohort={},environment={},band={},cohort_band={},environment_band={},frame=[])
        scopes = [('all','all',list(range(360)))]
        for field in ('cohort','environment'):
            groups = defaultdict(list)
            for i,r in enumerate(roster): groups[r[field]].append(i)
            scopes.extend((field,key,indices) for key,indices in groups.items())
        for kind,label,indices in scopes:
            check('group_summaries')
            rr = np.concatenate([frame_target_parts[i][0] for i in indices])
            zz = np.concatenate([frame_target_parts[i][1] for i in indices])
            if kind!='all': full_groups[kind][label]=dict(target_summary(rr,bounds),frames=len(indices))
            sidx = np.concatenate([np.arange(i*512,(i+1)*512) for i in indices])
            eligible_idx = sidx[local['target_eligible'][sidx]]
            def local_stats(index):
                return proxy_summary(local['residual'][index],local['neighbor_median'][index],
                    local['neighbor_distance'][index],local['neighbor_q25'][index],local['neighbor_q75'][index],bounds)
            summary = dict(local_stats(eligible_idx),frames=len(indices),public_points=int(sidx.size),
                missing_reference_points=int(sidx.size-eligible_idx.size))
            if kind=='all': local_groups['all']=summary
            else: local_groups[kind][label]=summary
            gb = bands(zz)
            for band_index,band_name in enumerate(BANDS):
                key = band_name if kind=='all' else f'{label}/{band_name}'
                category = 'band' if kind=='all' else f'{kind}_band'
                full_groups[category][key]=target_summary(rr[gb==band_index],bounds)
                lidx = eligible_idx[local['gt_band'][eligible_idx]==band_index]
                local_groups[category][key]=local_stats(lidx)
        for i,r in enumerate(roster):
            check('frame_summaries')
            residual,gt = frame_target_parts[i]
            meta = {key:r[key] for key in ('frame_index','cohort','environment','scan','frame','role')}
            full_groups['frame'].append(dict(meta,**target_summary(residual,bounds)))
            si = np.arange(i*512,(i+1)*512); valid_idx = si[local['target_eligible'][si]]
            local_groups['frame'].append(dict(meta,public_points=512,missing_reference_points=int(512-valid_idx.size),
                **proxy_summary(local['residual'][valid_idx],local['neighbor_median'][valid_idx],local['neighbor_distance'][valid_idx],
                    local['neighbor_q25'][valid_idx],local['neighbor_q75'][valid_idx],bounds)))
        macro = {}
        for label,indices in [('all',list(range(360)))]+[(c,[i for i,r in enumerate(roster) if r['cohort']==c]) for c in COHORTS]:
            frame_rows = [local_groups['frame'][i] for i in indices if local_groups['frame'][i]['points']]
            macro[label] = dict(frame_units=len(frame_rows),
                frame_zero_mae=quantile_summary([r['baseline_zero']['mae'] for r in frame_rows]),
                frame_candidate_mae=quantile_summary([r['candidate']['mae'] for r in frame_rows]),
                frame_candidate_improved=int(sum(r['candidate']['mae']<r['baseline_zero']['mae'] for r in frame_rows)),
                frame_candidate_worsened=int(sum(r['candidate']['mae']>r['baseline_zero']['mae'] for r in frame_rows)))
        env_macro = {}
        for cohort in COHORTS:
            envs = sorted({r['environment'] for r in roster if r['cohort']==cohort})
            rr = [local_groups['environment'][e] for e in envs if local_groups['environment'][e]['points']]
            env_macro[cohort]=dict(environment_units=len(rr),
                environment_zero_mae=quantile_summary([r['baseline_zero']['mae'] for r in rr]),
                environment_candidate_mae=quantile_summary([r['candidate']['mae'] for r in rr]),
                environment_candidate_improved=int(sum(r['candidate']['mae']<r['baseline_zero']['mae'] for r in rr)),
                environment_candidate_worsened=int(sum(r['candidate']['mae']>r['baseline_zero']['mae'] for r in rr)))
        summary = dict(status='COMPLETE',bounds=load(output/'bounds.json'),full_targets=full_groups,
            local_proxy=local_groups,frame_macro=macro,environment_macro=env_macro,
            units='full_targets: native eligible first-return points; local_proxy: GT-eligible subset of fixed public512 per frame; frame/env macro unweighted units separately',
            interpretation='Local train-only median is a descriptive compatibility proxy, not a learned model or query benefit; cal descriptions do not establish transfer confirmation',
            features='only train-normalized logDP/ray_x/ray_y; source/GT/band/query/labels excluded',
            target_dtypes={k:str(v.dtype) for k,v in all_targets.items()},bank_dtypes={k:str(v.dtype) for k,v in bank.items()},
            local_dtypes={k:str(v.dtype) for k,v in local.items()},bound_order=[n for n,_ in bounds],
            public_sample_points=n,reference_eligible_sample_points=int(local['target_eligible'].sum()),
            train_LOEO_points=int(local['target_eligible'][:56*512].sum()),cal_points=int(local['target_eligible'][56*512:].sum()),
            no_evaluation_access=True,no_training_or_inference=True)
        write(output/'summary.json',summary)
        receipt.update(status='COMPLETE',train_bank_points=1591791,
            full_target_points=int(all_targets['residual'].size),full_cal_points=int(cal_abs.size),
            public_sample_points=n,eligible_sample_points=int(local['target_eligible'].sum()),
            candidate_bound=float(bounds[3][1]),summary_sha256=sha(output/'summary.json'))
    except Exception as error:
        receipt.update(status='FAILED_PARTIAL',error=repr(error))
        receipt['failures'].append(dict(stage=receipt.get('stage'),error=repr(error),wall_s=time.perf_counter()-EXEC_STARTED))
    finally:
        tree = None; gc.collect()
        receipt.update(cpu_allocation_wall_s=time.perf_counter()-EXEC_STARTED,
            source_delivered_sha256=sha(__file__),resource_release='CPU tree handles released; process exits; durable arrays retained')
        receipt['budget_postcheck'] = receipt['cpu_allocation_wall_s']<=budget_s
        write(output/'terminal.json',receipt)
        write(output/'progress.json',receipt)
        (output/'writer.json').unlink()
        print(json.dumps(receipt),flush=True)
    return receipt['status']=='COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=650.)
    args=p.parse_args()
    raise SystemExit(0 if run(args.repo.resolve(),args.output.resolve(),args.budget_s) else 1)
