"""Frozen native-depth geometry diagnosis; evaluator-only consumed Development.

All 136 existing evaluation and 48 new ARKit calibration frames are mandatory.
Depth statistics union native pixels within each frame's >=16-positive near
queries. Interval statistics deliberately count correlated query-ray pairs.
No model, RGB, refit, new cutoff, label filling or data acquisition is used.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_negative_frozen_score import ARMS, EVAL, paths
from rgb_body_query_fixed_grid import queries
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_query_calibration_probe import FREE, nth_score, write
from rgb_body_query_scene_diagnostic import write_csv

ALL_ARMS = ('depthpro_raw',) + ARMS
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
NAMES = ('min', 'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'max')


def quantiles(values):
    return dict(zip(NAMES, np.quantile(values, QUANTILES).tolist())) if len(values) else None


def depth_summary(gt, predicted):
    count = len(gt)
    return dict(available_native_point_units=count,
        signed_z_error_m=quantiles(predicted-gt), log_z_ratio=quantiles(np.log(predicted/gt)),
        z_ratio=quantiles(predicted/gt), gt_z_m=quantiles(gt), predicted_z_m=quantiles(predicted),
        pred_below_gt=int((predicted < gt).sum()), pred_equal_gt=int((predicted == gt).sum()),
        pred_above_gt=int((predicted > gt).sum()),
        pred_below_near_z_band=int((predicted < .3-1e-12).sum()),
        pred_inside_near_z_band=int(((predicted >= .3-1e-12) & (predicted <= .8+1e-12)).sum()),
        pred_above_near_z_band=int((predicted > .8+1e-12).sum()),
        pred_above_gt_fraction=float((predicted > gt).mean()) if count else None,
        pred_below_gt_fraction=float((predicted < gt).mean()) if count else None,
        pred_inside_near_z_band_fraction=float(((predicted >= .3-1e-12) & (predicted <= .8+1e-12)).mean()) if count else None)


def pair_counts(pred, gt, entry, exit, mask, prefix):
    p, z, lo, hi = [v[mask] for v in (pred, gt, entry, exit)]
    before = p < lo-1e-12
    after = p > hi+1e-12
    inside = ~(before | after)
    return {prefix+'_'+k: v for k, v in dict(query_ray_pair_units=len(p),
        pred_before_interval=int(before.sum()), pred_inside_interval=int(inside.sum()),
        pred_after_interval=int(after.sum()), pred_below_gt=int((p < z).sum()),
        pred_equal_gt=int((p == z).sum()), pred_above_gt=int((p > z).sum()),
        gt_before_interval=int((z < lo-1e-12).sum()),
        gt_inside_interval=int(((z >= lo-1e-12) & (z <= hi+1e-12)).sum()),
        gt_after_interval=int((z > hi+1e-12).sum())).items()}


def run(repo, output, budget_s):
    started = time.perf_counter()
    work, root, candidate, folders = paths(repo)
    runroot = work/'rgb-body-query-near-rank-geometry-dev-20261010'
    if not output.resolve().is_relative_to((runroot/'geometry').resolve()):
        raise ValueError('Geometry payload must remain inside this run geometry/')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve existing geometry attempt and receipts')
    receipt = dict(status='INCOMPLETE', started_utc=utc(), budget_cpu_wall_s=budget_s,
        gpu_s=0, download_bytes=0, training_calls=0, inference_calls=0,
        resources='No processes/workers/GPU/services retained; durable payload owned by this run')
    checks = Counter(); inputs = {}; frame_rows = []; pair_rows = []; values = defaultdict(list)
    replay = []; registered = []; manifest_records = []
    def check():
        if time.perf_counter()-started >= budget_s:
            raise TimeoutError('Geometry CPU wall budget reached; preserve partial attempt')
    def register(path, expected=None):
        check(); p = Path(path); key = str(p)
        if key not in inputs:
            actual = sha(p)
            if expected is not None and actual != expected:
                raise ValueError(f'Input identity differs: {p}')
            inputs[key] = dict(path=key, sha256=actual)
        elif expected is not None and inputs[key]['sha256'] != expected:
            raise ValueError('One payload registered with two different hashes')
        return p
    try:
        parent_plan = load(register(runroot/'plan.json'))
        assert parent_plan['run'] == 'RGB_BODY_QUERY_NEAR_RANK_GEOMETRY_DEV_20261010'
        assert budget_s <= parent_plan['budgets']['geometry_wall_s']
        info = load(register(candidate/'training_inputs.json'))
        fixed = queries()
        cached = work/'rgb-body-query-negative-score-dev-20261010/frozen-score/cached'
        calroot = work/'rgb-body-query-arkit-cal-dev-20261010'
        for cohort in EVAL + ('cal_arkit',):
            check()
            mpath = folders[cohort]/'dataset_manifest.json' if cohort in EVAL else calroot/'reference/dataset_manifest.json'
            ppath = candidate/'predictions'/cohort/'predictions.json' if cohort in EVAL else calroot/'head/predictions.json'
            spath = cached/f'{cohort}_scores.json' if cohort in EVAL else calroot/'score/absolute/cal_arkit_scores.json'
            manifest, pm, scores = [load(register(p)) for p in (mpath, ppath, spath)]
            assert manifest['queries'] == fixed
            refs = {(r['scan'], r['frame']): r for r in manifest['rows']}
            scorelookup = {a: {(r['scan'], r['frame'], r['query']): r for r in scores['arms'][a]} for a in ARMS}
            assert len(refs) == len(pm['rows'])
            if cohort == 'cal_arkit': assert len(pm['rows']) == 48
            for index, row in enumerate(pm['rows']):
                ref = refs[row['scan'], row['frame']]
                register(ref['reference_path'], ref['reference_sha256'])
                register(row['sampled_depth_path'], row['sampled_depth_sha256'])
                dr = row['distributions']['depth_ray_gaussian']
                register(dr['path'], dr['sha256'])
                registered.append((cohort, row, ref, scorelookup, index == 0))
            manifest_records.append(dict(cohort=cohort, frames=len(pm['rows']),
                environment_labels=sorted({r['environment'] for r in pm['rows']})))
        assert len(registered) == 184
        identities = [(r['scan'], r['frame']) for _, r, _, _, _ in registered]
        assert len(set(identities)) == 184
        write(output/'inputs.json', dict(status='FROZEN_BEFORE_PIXEL_ANALYSIS', records=list(inputs.values()),
            frames=manifest_records, frame_identities=[dict(cohort=c, scan=r['scan'], frame=r['frame']) for c,r,_,_,_ in registered]))
        write(output/'plan.json', dict(created_utc=utc(), parent_plan_sha256=sha(runroot/'plan.json'),
            source_sha256=sha(__file__), budget_cpu_wall_s=budget_s, arms=ALL_ARMS,
            frames=184, queries=fixed, threshold_selection=False, RGB_reads=False,
            point_units='Union native pixel indices separately within each frame, only labels1 from near queries whose strict state is POSITIVE>=16; repeated frames/surfaces remain correlated',
            pair_units='Query-ray pairs in POSITIVE near query known-positive rays and strict FREE near queries; duplicates across queries intentionally retained',
            GT_role='Evaluator first-return depth reference only; not physical-depth accuracy proof'))
        dest = output/'native-point-values'; dest.mkdir()
        for completed, (cohort, row, ref, scorelookup, representative) in enumerate(registered, 1):
            check()
            with np.load(ref['reference_path'], allow_pickle=False) as f:
                labels, gt, observed, k = [f[key].copy() for key in ('labels','depth','observed','depth_K')]
                assert np.array_equal(k, ref['depth_K']) and np.array_equal(f['color_K'], ref['color_K'])
            with np.load(row['sampled_depth_path'], allow_pickle=False) as f: dp = f['depth'].copy()
            with np.load(row['distributions']['depth_ray_gaussian']['path'], allow_pickle=False) as f:
                mu, sig, drvalid = [f[key].copy() for key in ('mu','sigma','valid')]
            assert list(gt.shape) == ref['depth_shape'] == row['depth_shape']
            assert dp.shape == gt.shape == mu.shape == sig.shape == observed.shape
            assert labels.shape == (27, *gt.shape)
            assert np.array_equal(k, row['depth_K']) and np.isfinite(k).all() and abs(np.linalg.det(k)) > 0
            valid = np.isfinite(dp) & (dp > 0)
            assert np.array_equal(valid, drvalid)
            assert np.isfinite(mu).all() and np.isfinite(sig).all() and (sig > 0).all()
            checks['native_geometry_mask_finite_head_frames'] += 1
            loga = np.zeros(dp.shape, np.float64)
            loga[valid] = info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b']
            depths = dict(depthpro_raw=dp.astype(np.float64), affine=np.exp(loga), old_depth_ray=np.exp(mu))
            for d in (.05, .1, .2): depths[f'a0_delta_{d:g}'] = np.exp(loga+np.clip(mu-loga, -d, d))
            assert all(np.isfinite(z[valid]).all() and (z[valid] > 0).all() for z in depths.values())
            rx, ry = rays(k, gt.shape); near_info = []; union = np.zeros(gt.shape, bool)
            for qi, q in enumerate(fixed):
                if q['low'][2] != .3: continue
                entry, exit, domain = ray_interval(rx, ry, q); lab = labels[qi]
                assert np.array_equal(lab != 255, domain)
                cnt = {key: int((lab == v).sum()) for key,v in [('positive_pixels',1),('free_ray_pixels',0),('unknown_pixels',2)]}
                state = 'POSITIVE' if cnt['positive_pixels'] >= 16 else FREE if cnt['positive_pixels']==0 and cnt['unknown_pixels']==0 and cnt['free_ray_pixels']>=16 else 'UNKNOWN'
                saved = ref['queries'][qi]
                assert state == saved['state'] and int(domain.sum()) == saved['domain_pixels']
                assert all(cnt[key] == saved[key] for key in cnt)
                known = (lab == 1) | (lab == 0)
                assert (observed[known] & np.isfinite(gt[known]) & (gt[known] > 0)).all()
                assert ((gt[lab==1] >= entry[lab==1]-1e-12) & (gt[lab==1] <= exit[lab==1]+1e-12)).all()
                assert (gt[lab==0] > exit[lab==0]+1e-12).all()
                if 'observed_reachable_rays' in saved: assert int((domain & observed).sum()) == saved['observed_reachable_rays']
                checks['near_native_domain_state_GT_checks'] += 1
                if state == 'POSITIVE': union |= lab == 1
                near_info.append((q, entry, exit, domain, lab, state))
                for arm, pred in depths.items():
                    item = dict(cohort=cohort, environment=ref['environment'], scan=row['scan'], frame=row['frame'], query=q['name'], arm=arm, reference_state=state,
                        near_positive_query_pixel_units=cnt['positive_pixels'] if state=='POSITIVE' else 0,
                        strict_free_query_pixel_units=cnt['free_ray_pixels'] if state==FREE else 0)
                    item.update(pair_counts(pred,gt,entry,exit,(lab==1)&valid if state=='POSITIVE' else np.zeros(gt.shape,bool),'positive'))
                    item.update(pair_counts(pred,gt,entry,exit,(lab==0)&valid if state==FREE else np.zeros(gt.shape,bool),'strict_free'))
                    item['strict_free_false_entry_pair_units'] = item['strict_free_pred_inside_interval']
                    pair_rows.append(item)
                    if representative and arm in ARMS:
                        margin = np.full(gt.shape, -np.inf); mask = domain & valid
                        margin[mask] = np.minimum(pred[mask]-entry[mask],exit[mask]-pred[mask])
                        qs, ws = nth_score(margin), nth_score(margin[lab==1])
                        cachedrow = scorelookup[arm][row['scan'],row['frame'],q['name']]
                        cq, cw = [float(cachedrow[key]) for key in ('query_score','known_positive_score')]
                        assert qs == cq and ws == cw
                        replay.append(dict(cohort=cohort,scan=row['scan'],frame=row['frame'],query=q['name'],arm=arm,query_score=qs,known_positive_score=ws,max_abs=0.))
                        checks['representative_exact_near_score_pairs'] += 1
            available = union & valid
            assert (np.isfinite(gt[union]) & (gt[union] > 0) & observed[union]).all()
            pvpath = dest/f'{row["scan"]}_{row["frame"]:06d}.npz'
            gtvals = gt[available].astype(np.float64)
            payload = dict(native_flat_indices=np.flatnonzero(available), gt_unique=gtvals)
            for arm, pred in depths.items(): payload[arm] = pred[available].astype(np.float64)
            np.savez_compressed(pvpath, **payload)
            pvhash = sha(pvpath)
            for arm in ALL_ARMS:
                p = payload[arm]; values[cohort,arm].append((gtvals,p))
                frame_rows.append(dict(cohort=cohort,environment=ref['environment'],scan=row['scan'],frame=row['frame'],arm=arm,
                    positive_gt_union_pixels=int(union.sum()), available_native_point_units=len(gtvals),
                    unavailable_prediction_native_pixels=int((union & ~valid).sum()),
                    values_path=str(pvpath),values_sha256=pvhash,
                    pred_below_gt=int((p<gtvals).sum()),pred_equal_gt=int((p==gtvals).sum()),pred_above_gt=int((p>gtvals).sum())))
            write(output/'progress.json', dict(completed_frames=completed,expected_frames=184,elapsed_cpu_wall_s=time.perf_counter()-started))
        summary_rows = []; results = {}
        for (cohort, arm), parts in values.items():
            check(); gtvals=np.concatenate([p[0] for p in parts]); pred=np.concatenate([p[1] for p in parts])
            rr=[r for r in pair_rows if r['cohort']==cohort and r['arm']==arm]
            frows=[r for r in frame_rows if r['cohort']==cohort and r['arm']==arm]
            pairs={key:sum(r[key] for r in rr) for key in rr[0] if key.startswith(('positive_','strict_free_'))}
            sm=dict(frames=len(parts),environment_labels=len({r['environment'] for r in frows}),
                positive_gt_union_pixels=sum(r['positive_gt_union_pixels'] for r in frows),
                unavailable_prediction_native_pixels=sum(r['unavailable_prediction_native_pixels'] for r in frows),
                **depth_summary(gtvals,pred),query_ray_pair_counts=pairs)
            results.setdefault(cohort,{})[arm]=sm
            flat={key:v for key,v in sm.items() if not isinstance(v,dict) and key not in ('signed_z_error_m','log_z_ratio','z_ratio','gt_z_m','predicted_z_m')}
            for measure in ('signed_z_error_m','log_z_ratio','z_ratio','gt_z_m','predicted_z_m'):
                for name in NAMES:flat[measure+'_'+name] = sm[measure][name] if sm[measure] else None
            flat.update(pairs);summary_rows.append(dict(cohort=cohort,arm=arm,**flat))
        checks['mandatory_frame_identities'] = len(registered)
        checks['native_point_values_files'] = len(registered)
        write_csv(output/'frame_counts.csv',frame_rows);write_csv(output/'query_pair_counts.csv',pair_rows)
        write_csv(output/'summary.csv',summary_rows)
        write(output/'representative_score_replay.json',dict(status='PASS',records=replay))
        write(output/'checks.json',dict(status='PASS',checks=dict(checks)))
        write(output/'results.json',dict(status='COMPLETE',cohorts=results,arms=ALL_ARMS,
            native_points='Union unique native pixel positions within each frame; count repeated across differentframes, not unique physical surfaces',
            query_ray_pairs='Native positive rays in POSITIVE>=16 near queries or rays in strict FREE near queries; duplicates across boxes deliberately retained',
            z_band_inside='Optical Z .3-.8 only; true perquery 3D ray interval inside reported separately',
            limits='Native sensor first-return references evaluator-only, not guaranteed physical GT accuracy. Consumed Development; no newworkingpoint, inference, refit or training'))
        receipt['status']='COMPLETE'
    except Exception as error:
        receipt['failure']=repr(error)
        write(output/'failure.json',dict(error=repr(error),completed_frames=len(frame_rows)//len(ALL_ARMS),elapsed_cpu_wall_s=time.perf_counter()-started))
        raise
    finally:
        shutil.copyfile(__file__,output/'executed_near_geometry.py')
        receipt.update(elapsed_cpu_wall_s=time.perf_counter()-started,source_sha256=sha(__file__),checks=dict(checks))
        write(output/'terminal.json',receipt)
    print('COMPLETE',receipt['elapsed_cpu_wall_s'],dict(checks),flush=True)


def repair_summary(repo, output, previous, budget_s):
    """Recover only failed CSV publication from preserved completed pixel values."""
    started=time.perf_counter()
    expected=repo/'artifacts.local/work/rgb-body-query-near-rank-geometry-dev-20261010/geometry'
    assert output.resolve().is_relative_to(expected.resolve()) and previous.resolve()==expected.resolve()
    prior=load(previous/'terminal.json')
    assert prior['status']=='INCOMPLETE' and 'dict contains fields not in fieldnames' in prior['failure']
    assert prior['checks']['native_geometry_mask_finite_head_frames']==184
    assert prior['checks']['near_native_domain_state_GT_checks']==1656
    assert prior['checks']['representative_exact_near_score_pairs']==270
    assert prior['elapsed_cpu_wall_s']+budget_s <= 420.
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():raise FileExistsError('Preserve prior repair')
    receipt=dict(status='INCOMPLETE',budget_cpu_wall_s=budget_s,prior_failed_elapsed_cpu_wall_s=prior['elapsed_cpu_wall_s'],gpu_s=0,training_calls=0,inference_calls=0,download_bytes=0)
    write(output/'plan.json',dict(status='PUBLICATION_ONLY_REPAIR',prior=str(previous),prior_terminal_sha256=sha(previous/'terminal.json'),source_sha256=sha(__file__),budget_cpu_wall_s=budget_s,scope='Reuse all184 frame native point values and querypair counts; no newsource, pixels, model or thresholds'))
    try:
        frame_rows=list(csv.DictReader((previous/'frame_counts.csv').open(encoding='utf-8-sig')))
        pair_rows=list(csv.DictReader((previous/'query_pair_counts.csv').open(encoding='utf-8-sig')))
        assert len(frame_rows)==184*len(ALL_ARMS) and len(pair_rows)==184*9*len(ALL_ARMS)
        for rr in (frame_rows,pair_rows):
            for r in rr:
                for key in r:
                    if key not in ('cohort','environment','scan','arm','query','reference_state','values_path','values_sha256'):r[key]=int(r[key])
        values=defaultdict(list); opened={}; quantile_fields=('signed_z_error_m','log_z_ratio','z_ratio','gt_z_m','predicted_z_m')
        for r in frame_rows:
            if time.perf_counter()-started >= budget_s:raise TimeoutError('Summary repair budget reached')
            p=Path(r['values_path'])
            if p not in opened:
                assert sha(p)==r['values_sha256']
                with np.load(p,allow_pickle=False) as f:opened[p]={key:f[key].copy() for key in f.files}
            saved=opened[p];gt=saved['gt_unique'];pred=saved[r['arm']]
            assert len(gt)==len(pred)==r['available_native_point_units']
            assert int((pred<gt).sum())==r['pred_below_gt'] and int((pred>gt).sum())==r['pred_above_gt']
            values[r['cohort'],r['arm']].append((gt,pred))
        assert len(opened)==184
        results={};table=[]
        for (cohort,arm),parts in values.items():
            gt=np.concatenate([v[0] for v in parts]);pred=np.concatenate([v[1] for v in parts])
            rr=[r for r in pair_rows if r['cohort']==cohort and r['arm']==arm];ff=[r for r in frame_rows if r['cohort']==cohort and r['arm']==arm]
            pairs={key:sum(r[key] for r in rr) for key in rr[0] if key.startswith(('positive_','strict_free_'))}
            sm=dict(frames=len(parts),environment_labels=len({r['environment'] for r in ff}),positive_gt_union_pixels=sum(r['positive_gt_union_pixels'] for r in ff),unavailable_prediction_native_pixels=sum(r['unavailable_prediction_native_pixels'] for r in ff),**depth_summary(gt,pred),query_ray_pair_counts=pairs)
            results.setdefault(cohort,{})[arm]=sm
            flat={key:v for key,v in sm.items() if not isinstance(v,dict) and key not in quantile_fields}
            for measure in quantile_fields:
                for name in NAMES:flat[measure+'_'+name]=sm[measure][name] if sm[measure] else None
            flat.update(pairs);table.append(dict(cohort=cohort,arm=arm,**flat))
        shutil.copyfile(previous/'inputs.json',output/'inputs.json')
        write_csv(output/'frame_counts.csv',frame_rows);write_csv(output/'query_pair_counts.csv',pair_rows);write_csv(output/'summary.csv',table)
        write(output/'representative_score_replay.json',dict(status='PASS',passed_exact_pairs=270,replayed_in_prior_attempt=True,prior_terminal_sha256=sha(previous/'terminal.json'),selection='First frame each6cohorts x9nearquery x5cachedarms; exact score andknownwitness assertion in preserved execution source'))
        checks=dict(prior['checks'],publication_repair_verified_point_values=184)
        write(output/'checks.json',dict(status='PASS',checks=checks))
        write(output/'results.json',dict(status='COMPLETE',cohorts=results,arms=ALL_ARMS,publication_repair_from=str(previous),native_points='Union unique nativepixel positions within eachframe; repeated across frames, not uniquephysical surfaces',query_ray_pairs='Nativepositive rays in POSITIVE>=16 nearqueries or strictFREE nearqueries; duplicates acrossboxes retained',z_band_inside='OpticalZ .3-.8 only; perquery3D rayinterval inside separate',limits='Nativefirstreturn reference evaluator-only, not guaranteed physical GTaccuracy. Consumed Development; no newworkingpoint/inference/refit/training'))
        receipt.update(status='COMPLETE',checks=checks,resources='No retained processes; all original raw inputs and failedreceipt preserved')
    except Exception as error:
        receipt['failure']=repr(error);write(output/'failure.json',dict(error=repr(error)));raise
    finally:
        shutil.copyfile(__file__,output/'executed_near_geometry.py')
        receipt.update(elapsed_cpu_wall_s=time.perf_counter()-started,source_sha256=sha(__file__))
        receipt['cumulative_cpu_wall_s']=prior['elapsed_cpu_wall_s']+receipt['elapsed_cpu_wall_s']
        write(output/'terminal.json',receipt)
    print('COMPLETE publication repair',receipt['cumulative_cpu_wall_s'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-s',type=float,default=420.)
    parser.add_argument('--repair-from',type=Path)
    args=parser.parse_args()
    if args.repair_from:repair_summary(args.repo.resolve(),args.output.resolve(),args.repair_from.resolve(),args.budget_s)
    else:run(args.repo.resolve(),args.output.resolve(),args.budget_s)
