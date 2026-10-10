"""Posthoc frozen width-normalized interval margin, consumed Development.

The absolute-margin result remains primary. Public ray/query interval width
removes its distance-band-dependent maximum; models and masks stay frozen.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_negative_frozen_score import ARMS, EVAL, paths, numeric, summaries, compare
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_query_calibration_probe import FREE, COHORTS, nth_score, select_cutoff, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_bounded_residual_proxy import evaluate, pair_summary
from rgb_body_query_scene_diagnostic import write_csv


def scores(cohort, manifest, pm, info, check, inputs, checks, widths):
    refs = {(r['scan'], r['frame']): r for r in manifest['rows']}
    data = {a: [] for a in ARMS}
    for row in pm['rows']:
        check(); ref = refs[row['scan'], row['frame']]
        item = row['distributions']['depth_ray_gaussian']
        for path, expected in [(ref['reference_path'], ref['reference_sha256']),
                               (row['sampled_depth_path'], row['sampled_depth_sha256']), (item['path'], item['sha256'])]:
            assert sha(path) == expected; inputs.append(dict(path=str(path), sha256=expected))
        with np.load(ref['reference_path']) as f: labels = f['labels']
        with np.load(row['sampled_depth_path']) as f: dp = f['depth']
        with np.load(item['path']) as f: mu, drvalid = f['mu'], f['valid']
        valid = np.isfinite(dp) & (dp > 0)
        assert np.array_equal(valid, drvalid) and np.isfinite(mu).all()
        checks['unchanged_prediction_masks'] += 1
        loga = np.zeros(dp.shape, np.float64)
        loga[valid] = info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b']
        depths = {'affine': np.exp(loga), 'old_depth_ray': np.exp(mu)}
        for d in (.05,.1,.2): depths[f'a0_delta_{d:g}'] = np.exp(loga+np.clip(mu-loga,-d,d))
        rx, ry = rays(row['depth_K'], row['depth_shape'])
        for j, q in enumerate(manifest['queries']):
            check(); entry, exit, domain = ray_interval(rx, ry, q)
            width = exit-entry; mask = valid & domain & (width > 1e-12)
            oldmask = valid & domain
            widths.append(dict(cohort=cohort, scan=row['scan'], frame=row['frame'], query=q['name'],
                domain_pixels=int(domain.sum()), zero_width_domain=int((domain & (width==0)).sum()),
                small_or_zero_width_domain=int((domain & (width<=1e-12)).sum()),
                old_valid_domain=int(oldmask.sum()), normalized_valid_domain=int(mask.sum()),
                available_changed=(int(oldmask.sum())>=16)!=(int(mask.sum())>=16)))
            assert not (domain & (width < 0)).any()
            lab = labels[j]; pos, free, unk = [int((lab==n).sum()) for n in (1,0,2)]
            state = 'POSITIVE' if pos>=16 else (FREE if pos==0 and unk==0 and free>=16 else 'UNKNOWN')
            assert state == ref['queries'][j]['state']; checks['strict_query_states'] += 1
            for a, z in depths.items():
                margin = np.minimum(z[mask]-entry[mask], exit[mask]-z[mask])
                normalized = 2*margin/width[mask]
                assert np.isfinite(normalized).all() and (normalized <= 1+2e-14).all()
                assert np.array_equal(normalized>=0, margin>=0)
                assert np.array_equal(np.sign(normalized), np.sign(margin))
                checks['score_max_bound_zero_sign_equivalence_queries'] += 1
                checks['zero_sign_equivalence_pixels'] += len(margin)
                s = np.full(dp.shape,-np.inf); s[mask] = normalized
                data[a].append(dict(cohort=cohort, environment=ref['environment'], scan=row['scan'], frame=row['frame'],
                    query=q['name'], distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m', reference_state=state,
                    valid_ray_count=len(margin), absolute_valid_ray_count=int(oldmask.sum()),
                    query_score=nth_score(s), known_positive_score=nth_score(s[lab==1])))
    return data


def run(repo, out, budget, prior_attempt=None):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'amendment_plan.json').exists(): raise FileExistsError('Preserve prior normalized probe')
    used = 0.
    if prior_attempt:
        prior = load(prior_attempt/'terminal.json'); assert prior['status']=='FAILED_PARTIAL'
        used = prior['probe_wall_s']+prior.get('prior_probe_wall_s',0.)
        budget -= used; assert budget>0
    # Freeze before normalized scores or their outcomes are computed/read.
    write(out/'amendment_plan.json', dict(frozen_utc=utc(), source_sha256=sha(__file__),
        reason='Absolute newcal affine cutoff .43300925036 exceeds near half-width .25 and mid .35: structural band rejection',
        posthoc=True, primary_absolute_margin_preserved=True, arms=ARMS,
        definition='s=2*min(z-entry,exit-z)/(exit-entry); domain & frozen valid & width>1e-12, otherwise -Infinity',
        tangent_policy='Do not change evaluator state or prediction mask; report removed degenerate domain rays and query availability',
        calibration='Original cal16 + additional240; 32 strict sampled FREE only; 16th-largest query score; unique ties retained; closest attainable support<=5%',
        freeze_before_evaluation=True, fitted_in_cal_POS_descriptive_only=True, delta_selection=False,
        cohorts=EVAL, near_band='0.3-0.8m explicitly reported',
        proof='min(z-entry,exit-z)<=width/2 hence s<=1; on nondegenerate valid domain s>=0 iff margin>=0',
        budget_cpu_wall_s=budget, prior_probe_wall_s=used, prior_run_stage_wall_s=45.546,
        gpu_allocation_s=0, new_model_inference=0, training=0, downloads=0,
        stop='One normalized readout probe only; retain negative outcomes, no scale/threshold variants after result',
        decision_check='Paired query witness rescue/loss and FREE removed/added. New cal support has one-query resolution=1/32. Near POS denominators retained, no best delta',
        limits='Posthoc consumed Development; related source frames; strict sampled FREE is not whole-volume safety'))
    shutil.copyfile(__file__,out/'executed_normalized_margin_probe.py')
    start=time.perf_counter(); receipt=dict(status='STARTING',prior_probe_wall_s=used)
    def check():
        if time.perf_counter()-start>=budget: raise TimeoutError('Normalized CPU budget reached')
    try:
        work, root, candidate, cohorts = paths(repo); frozen = out.parent
        info = load(candidate/'training_inputs.json')
        inputs=[dict(path=str(candidate/'training_inputs.json'),sha256=sha(candidate/'training_inputs.json'))]
        checks=Counter(); widths=[]; data={}
        datasets=[('cal_original',root/'train-cal-sensor/dataset_manifest.json',candidate/'predictions/cal/predictions.json')]
        heads=[p for p in frozen.glob('head*') if (p/'terminal.json').exists() and load(p/'terminal.json')['status']=='COMPLETE']
        assert len(heads)==1
        datasets.append(('cal_additional',work/'rgb-body-query-negative-reference-dev-20261010/dataset_manifest.json',heads[0]/'predictions.json'))
        datasets.extend((name,folder/'dataset_manifest.json',candidate/f'predictions/{name}/predictions.json') for name,folder in cohorts.items())
        for name,mpath,ppath in datasets:
            check(); inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [mpath,ppath])
            data[name]=scores(name,load(mpath),load(ppath),info,check,inputs,checks,widths)
            write(out/f'{name}_scores.json',dict(arms=data[name])); print('NORMALIZED_SCORED',name,flush=True)
        cal={a:data['cal_original'][a]+data['cal_additional'][a] for a in ARMS}
        assert all(len(rr)==6912 for rr in cal.values())
        negative=[r for r in cal['affine'] if r['reference_state']==FREE]
        assert len(negative)==32 and len({r['environment'] for r in negative})==12
        cutoffs={a:select_cutoff([r['query_score'] for r in cal[a] if r['reference_state']==FREE]) for a in ARMS}
        write(out/'thresholds.json',dict(status='FROZEN_BEFORE_EVALUATION',frozen_utc=utc(),arms=cutoffs,
            rule='32 legal cal FREE only; >=cutoff ties retained; cal POS/eval/UNKNOWN not used',posthoc=True))
        inputs.append(dict(path=str(out/'thresholds.json'),sha256=sha(out/'thresholds.json')))
        results, table, pairs, pair_results = {}, [], [], {}
        for name in ('cal_fit_description',)+EVAL:
            check(); rr=cal if name=='cal_fit_description' else data[name]
            ev={a:evaluate(r,cutoffs[a]['cutoff']) for a,r in rr.items()}
            absolute_path=frozen/f'final/{name}_evaluation.json'
            absolute=load(absolute_path); inputs.append(dict(path=str(absolute_path),sha256=sha(absolute_path)))
            write(out/f'{name}_evaluation.json',ev)
            results[name]={a:summaries(r) for a,r in ev.items()}
            for a, r in ev.items():
                for group,sm in results[name][a].items(): table.append(dict(cohort=name,arm=a,group=group,cutoff=cutoffs[a]['cutoff'],**sm))
                if a!='affine':
                    pp=compare(r,ev['affine'],name,a,'normalized_affine','normalized_vs_normalized_affine')
                    pairs+=pp; pair_results[f'{name}/{a}/vs_normalized_affine']=pair_summary(pp)
                pp=compare(r,absolute[a],name,a,f'absolute_{a}','normalized_vs_absolute_same_arm_newcal')
                pairs+=pp; pair_results[f'{name}/{a}/vs_absolute_same_arm']=pair_summary(pp)
        loeo=[]
        for environment in sorted({r['environment'] for r in negative}):
            check()
            for a in ARMS:
                free=[r['query_score'] for r in cal[a] if r['reference_state']==FREE and r['environment']!=environment]
                choice=select_cutoff(free)
                held=[r for r in cal[a] if r['reference_state']==FREE and r['environment']==environment]
                heldev=evaluate(held,choice['cutoff'])
                for n in COHORTS:
                    sm=summaries(evaluate(data[n][a],choice['cutoff']))['all']
                    loeo.append(dict(held_negative_environment=environment,arm=a,cohort=n,cutoff=choice['cutoff'],
                        cal_negative_queries=len(free),held_free_total=len(held),held_free_support=sum(r['predicted_positive'] for r in heldev),**sm))
        write(out/'results.json',dict(status='COMPLETE',posthoc=True,thresholds=cutoffs,cohorts=results,pairs=pair_results,
            width_checks=dict(query_rows=len(widths),zero_width_domain=sum(r['zero_width_domain'] for r in widths),
                small_or_zero_width_domain=sum(r['small_or_zero_width_domain'] for r in widths),
                query_availability_changed=sum(r['available_changed'] for r in widths)),
            limits='Consumed Development posthoc readout; no safety/volume proof; cal positives fitted-in description'))
        write_csv(out/'summary.csv',table);write_csv(out/'query_pairs.csv',pairs)
        write_csv(out/'width_diagnostics.csv',widths);write_csv(out/'loeo.csv',loeo)
        write(out/'inputs.json',inputs);write(out/'checks.json',dict(checks))
        receipt.update(status='COMPLETE',checks=dict(checks),frames=392,cal_queries=6912,eval_queries=3672)
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(probe_wall_s=time.perf_counter()-start,prior_run_stage_wall_s=45.546,completed_utc=utc(),
            source_sha256=sha(__file__),gpu=0,training=0,downloads=0,new_inference=0)
        write(out/'terminal.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=150.);p.add_argument('--prior-attempt',type=Path);a=p.parse_args()
    run(a.repo.resolve(),a.output.resolve(),a.budget_s,a.prior_attempt)
