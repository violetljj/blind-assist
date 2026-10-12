"""Two raw backbones, frozen absolute16th readout, fresh Development captures.

Primary copies the existing pooled304 cuts byte-for-byte. Supplemental old6
FREE-only cuts seal before fresh evaluation. Fresh public validity is the full
registered grid; one model's invalid depth never filters the other's queries.
"""
from __future__ import annotations
import time
START=time.perf_counter()
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import shutil
import numpy as np
from rgb_body_query_interval_distribution import load,utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_negative_frozen_score import paths
from rgb_body_query_query_calibration_probe import FREE,nth_score,select_cutoff,write
from rgb_body_query_reference_eval import rays,ray_interval
from rgb_body_query_residual_readout import summary,evaluate,paired
from rgb_body_query_scene_diagnostic import write_csv

ARMS=('dav_raw','uni_raw')
BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m')
OLD=('arkit16','arkit_40777060','arkit_40777065','new_arkit_41069021','new_arkit_41069042','new_arkit_41069048')
COUNT_FIELDS=('positive_total','positive_known_witness','positive_support','free_total','free_support','unknown_total','unknown_support')


def key(row): return row['scan'],row['frame']


class Inputs:
    def __init__(self,budget): self.budget=budget;self.rows=[];self.hashes={};self.checks=Counter()
    def check(self):
        if time.perf_counter()-START>=self.budget: raise TimeoutError('Confirm evaluator command-wall allocation reached')
    def register(self,path,expected=None):
        self.check();path=Path(path);name=str(path.resolve())
        if name not in self.hashes:
            self.hashes[name]=sha(path);self.rows.append(dict(path=name,sha256=self.hashes[name]))
        if expected is not None: assert self.hashes[name]==expected,f'Changed source: {path}'
        return path
    def json(self,path): return load(self.register(path))


def prediction_index(path,inputs):
    source=inputs.json(path);assert source['status']=='COMPLETE'
    rows={key(r):r for r in source['rows']};assert len(rows)==len(source['rows'])
    return rows


def native(ref,index,inputs):
    row=index[key(ref)]
    for field in ('rgb_sha256','depth_K','depth_shape'): assert row[field]==ref[field]
    path=inputs.register(row['path'],row['sha256'])
    if path.suffix=='.npy': z=np.load(path,allow_pickle=False)
    else:
        with np.load(path,allow_pickle=False) as f: z=f['depth']
    assert list(z.shape)==ref['depth_shape']
    return np.asarray(z,np.float64)


def old_data(repo,prior,backbones,inputs):
    _,_,_,folders=paths(repo);manifests={};sources={};indices={}
    for title,root in [('original',backbones),('prior_fresh',prior)]:
        indices[title]={a:prediction_index(root/m/'predictions.json',inputs) for a,m in [('dav_raw','dav2'),('uni_raw','unidepth')]}
    bounded=repo/'artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010'
    for cohort in OLD:
        fresh=cohort.startswith('new_arkit_')
        mpath=prior/'new-eval-sensor'/cohort/'dataset_manifest.json' if fresh else folders[cohort]/'dataset_manifest.json'
        manifests[cohort]=inputs.json(mpath)
        source=dict(indices['prior_fresh' if fresh else 'original'])
        if fresh: source['old_public_DP']=prediction_index(prior/'depthpro/predictions.json',inputs)
        else:
            previous=prediction_index(bounded/f'predictions/{cohort}/predictions.json',inputs)
            source['old_public_DP']={k:dict(r,path=r['sampled_depth_path'],sha256=r['sampled_depth_sha256']) for k,r in previous.items()}
        sources[cohort]=source
    queries=manifests[OLD[0]]['queries']
    assert len(queries)==27 and all(m['queries']==queries for m in manifests.values())
    return manifests,sources


def score(cohort,manifest,sources,inputs,free_only=False,fresh=False):
    data={a:[] for a in ARMS}
    for ref in manifest['rows']:
        inputs.check()
        selected=[j for j,r in enumerate(ref['queries']) if r['state']==FREE] if free_only else list(range(27))
        if not selected: continue
        zz={a:native(ref,sources[a],inputs) for a in ARMS}
        if fresh:
            assert ref['depth_shape']==[192,256]
            public_valid=np.ones(ref['depth_shape'],dtype=bool)
        else:
            dp=native(ref,sources['old_public_DP'],inputs);public_valid=np.isfinite(dp)&(dp>0)
        rx,ry=rays(ref['depth_K'],ref['depth_shape'])
        assert np.isfinite(rx).all() and np.isfinite(ry).all()
        labels=None
        if not free_only:
            inputs.register(ref['reference_path'],ref['reference_sha256'])
            with np.load(ref['reference_path']) as f: labels=f['labels']
            assert labels.shape==(27,*public_valid.shape)
        for j in selected:
            q=manifest['queries'][j];state=ref['queries'][j]['state']
            if free_only: assert state==FREE
            else:
                pos,free,unk=[int((labels[j]==n).sum()) for n in (1,0,2)]
                expected='POSITIVE' if pos>=16 else FREE if pos==0 and unk==0 and free>=16 else 'UNKNOWN'
                assert state==expected;inputs.checks['unchanged_reference_queries']+=1
            entry,exit,domain=ray_interval(rx,ry,q)
            for arm,z in zz.items():
                mask=public_valid&domain&np.isfinite(z)&(z>0)
                margin=np.full(z.shape,-np.inf)
                margin[mask]=np.minimum(z[mask]-entry[mask],exit[mask]-z[mask])
                data[arm].append(dict(cohort=cohort,environment=ref['environment'],scan=ref['scan'],frame=ref['frame'],query=q['name'],
                    distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m',reference_state=state,valid_ray_count=int(mask.sum()),
                    public_domain_rays=int((public_valid&domain).sum()),model_invalid_public_rays=int((public_valid&domain&~(np.isfinite(z)&(z>0))).sum()),
                    public_mask='REGISTERED_GRID_ONES' if fresh else 'ORIGINAL_FROZEN_DP',
                    query_score=nth_score(margin),known_positive_score=-np.inf if free_only else nth_score(margin[labels[j]==1]),
                    calibration_FREE_only=free_only))
            inputs.checks['old_FREE_calibration_queries' if free_only else 'eval_queries']+=1
        inputs.checks['frames_with_old_FREE_calibration' if free_only else 'eval_frames']+=1
    return data


def expanded(rows,pairs=False):
    result=summary(rows,pairs);groups=defaultdict(list)
    for r in rows: groups[r['scan']].append(r)
    for scan,rr in groups.items():
        for group,value in summary(rr,pairs).items():
            if group=='all': result['capture/'+scan]=value
            elif group.startswith('band/'):result['capture_band/'+scan+'/'+group[5:]]=value
    return result


def decisions(data,cuts): return {a:evaluate(rr,cuts[a],False) for a,rr in data.items()}


def save(out,data,cuts,name,folds=None):
    out.mkdir(parents=True,exist_ok=True);cohorts={};pairs=[];pair_summaries={};table=[]
    for c,arms in data.items():
        write(out/f'{c}_evaluation.json',arms)
        cohorts[c]={a:expanded(rr) for a,rr in arms.items()}
        for a,groups in cohorts[c].items():
            for g,s in groups.items(): table.append(dict(cohort=c,arm=a,group=g,**s))
        pp=paired(arms['uni_raw'],arms['dav_raw'],c,'uni_raw','dav_raw',name)
        pairs+=pp;pair_summaries[c]=expanded(pp,True)
    result=dict(status='COMPLETE',protocol=name,thresholds=cuts,cohorts=cohorts,pairs=pair_summaries,
        fold_assignments=folds,data_role='Development; public first-return sampledqueries; no safety or unfiltered generalization proof')
    write(out/'results.json',result);write_csv(out/'summary.csv',table);write_csv(out/'query_pairs.csv',pairs)
    return result


def aggregate(result,cohorts,bands):
    totals={}
    for arm in ARMS:
        groups=[result['cohorts'][c][arm]['band/'+b] for c in cohorts for b in bands]
        totals[arm]={f:sum(g[f] for g in groups) for f in COUNT_FIELDS}
    pp=[result['pairs'][c]['band/'+b] for c in cohorts for b in bands]
    pair_counts={f:sum(r[f] for r in pp) for f in ('positive_rescue','positive_loss','free_added','free_removed')}
    assert totals['uni_raw']['positive_known_witness']-totals['dav_raw']['positive_known_witness']==pair_counts['positive_rescue']-pair_counts['positive_loss']
    assert totals['uni_raw']['free_support']-totals['dav_raw']['free_support']==pair_counts['free_added']-pair_counts['free_removed']
    return dict(cohorts=cohorts,bands=bands,arms=totals,pairs=pair_counts)


def primary_summary(primary,order):
    rows=[];gains=[]
    for c in order:
        dav=primary['cohorts'][c]['dav_raw']['band/'+BANDS[0]]
        uni=primary['cohorts'][c]['uni_raw']['band/'+BANDS[0]]
        assert dav['positive_total']==uni['positive_total'] and dav['free_total']==uni['free_total']
        gain=uni['positive_known_witness']-dav['positive_known_witness']
        if dav['positive_total']>0 and gain>0: gains.append(c)
        rows.append(dict(cohort=c,DAV=dav,Uni=uni,witness_gain=gain,FREE_net_increment=uni['free_support']-dav['free_support'],
            strictly_better_near=bool(dav['positive_total']>0 and gain>0),pairs=primary['pairs'][c]['band/'+BANDS[0]]))
    near=aggregate(primary,order,[BANDS[0]]);midfar=aggregate(primary,order,list(BANDS[1:]))
    nfree=near['arms']['dav_raw']['free_total'];nfdelta=near['arms']['uni_raw']['free_support']-near['arms']['dav_raw']['free_support']
    base=midfar['arms']['dav_raw']['positive_known_witness'];candidate=midfar['arms']['uni_raw']['positive_known_witness']
    loss=base-candidate;mfree=midfar['arms']['dav_raw']['free_total']
    mfdelta=midfar['arms']['uni_raw']['free_support']-midfar['arms']['dav_raw']['free_support']
    complete=len(order)==6
    criteria=dict(scope='FRESH6_ONLY_primary_original_pooled304',fresh_captures_COMPLETE=complete,
        improved_near_captures=gains,improved_near_count=len(gains),required_near_improved_captures=4,
        near_FREE_denominator=nfree,near_FREE_net_increment=nfdelta,near_allowed_FREE_increment=nfree//100,
        near_witness_pass=len(gains)>=4,near_FREE_pass=nfdelta<=nfree//100,
        midfar_DAV_witness=base,midfar_Uni_witness=candidate,midfar_net_witness_loss=loss,
        midfar_loss_ratio=loss/base if base else None,midfar_max_allowed_loss_count=(3*base)//100,
        midfar_witness_pass=(100*loss<=3*base) if base else loss<=0,
        midfar_FREE_denominator=mfree,midfar_FREE_net_increment=mfdelta,midfar_allowed_FREE_increment=(2*mfree)//100,
        midfar_FREE_pass=mfdelta<=(2*mfree)//100)
    criteria['near_pass']=criteria['near_witness_pass'] and criteria['near_FREE_pass']
    criteria['midfar_pass']=criteria['midfar_witness_pass'] and criteria['midfar_FREE_pass']
    criteria['substitution_signal']=bool(complete and criteria['near_pass'] and criteria['midfar_pass'])
    return dict(status='COMPLETE' if complete else 'PARTIAL_NOT_EVALUABLE',criteria=criteria,near_capture_rows=rows,
        near_aggregate=near,midfar_aggregate=midfar,band_aggregates={b:aggregate(primary,order,[b]) for b in BANDS},
        recommendation='UniDepthV2 rawR0替换DAVraw；RGB职责≥0.3m，近带辅助信号，仅Development' if criteria['substitution_signal'] else '保留DAVrawR0≥0.8m；近带局部收益与代价完整展示，不调整阈值')


def calibrate(args,inputs,out):
    source=args.prior/'calibration/thresholds.json';prior=inputs.json(source)
    copied=out/'primary_original_thresholds.json';shutil.copyfile(source,copied)
    assert sha(copied)==sha(source)
    primary={a:prior['pooled'][a+'/R0'] for a in ARMS}
    plan=inputs.json(args.runroot/'PLAN.json')['primary_calibration']
    assert primary['dav_raw']['cutoff']==plan['DAV_raw_R0_cutoff_m']==.24403834342956543
    assert primary['uni_raw']['cutoff']==plan['Uni_raw_R0_cutoff_m']==.09616100788116455
    manifests,sources=old_data(args.repo,args.prior,args.backbones,inputs)
    data={c:score(c,manifests[c],sources[c],inputs,True,False) for c in OLD}
    write(out/'old6_FREE_scores.json',data)
    folds={}
    for fold in range(3):
        held=[c for i,c in enumerate(OLD) if i%3==fold];train=[c for c in OLD if c not in held]
        cuts={a:select_cutoff([r['query_score'] for c in train for r in data[c][a]]) for a in ARMS}
        folds[str(fold)]=dict(held_old=held,calibration_old=train,arms=cuts)
    write(out/'thresholds.json',dict(status='ALL_CUTS_FROZEN_BEFORE_FRESH_EVALUATION',frozen_utc=utc(),
        primary=primary,supplemental=folds,old_capture_order=OLD,old_assignment='ordinal modulo3',fresh_assignment='accepted ordinal modulo3',
        primary_source_path=str(source.resolve()),primary_source_sha256=sha(source),primary_copy_sha256=sha(copied),
        PLAN_sha256=sha(args.runroot/'PLAN.json'),old_scores_free_only=True,fresh_scores_computed=False,
        fresh_public_validity='np.ones((192,256),bool); public geometry domain only; individual model validity never intersects other model',
        fresh_public_grid_mask_sha256=hashlib.sha256(np.ones((192,256),bool).tobytes()).hexdigest()))


def final(args,inputs,out):
    assert inputs.json(args.calibration/'terminal.json')['status']=='COMPLETE'
    frozen=inputs.json(args.calibration/'thresholds.json')
    assert frozen['status']=='ALL_CUTS_FROZEN_BEFORE_FRESH_EVALUATION' and frozen['PLAN_sha256']==sha(args.runroot/'PLAN.json')
    assert sha(args.calibration/'primary_original_thresholds.json')==frozen['primary_source_sha256']==sha(args.prior/'calibration/thresholds.json')
    manifests,sources=old_data(args.repo,args.prior,args.backbones,inputs)
    oldscores={c:score(c,manifests[c],sources[c],inputs,False,False) for c in OLD}
    for c,data in oldscores.items():write(out/f'{c}_scores.json',dict(arms=data))
    oldfold={c:i%3 for i,c in enumerate(OLD)}
    held={c:decisions(oldscores[c],frozen['supplemental'][str(oldfold[c])]['arms']) for c in OLD}
    save(out/'supplemental-held-old',held,frozen['supplemental'],'old6_threefold_held_diagnostics',oldfold)
    roster=inputs.json(args.new_roster)
    order=roster.get('accepted_cohorts_order',list(dict.fromkeys(r['cohort'] for r in roster['rows'])))
    assert len(order)==len(set(order)) and set(order)=={r['cohort'] for r in roster['rows']}
    mapping=inputs.json(args.new_predictions)
    newsources={a:prediction_index(mapping[m],inputs) for a,m in [('dav_raw','dav2'),('uni_raw','unidepth')]}
    oldids={key(r) for m in manifests.values() for r in m['rows']};freshids=set();freshdata={}
    for c in order:
        manifest=inputs.json(args.runroot/'new-eval-sensor'/c/'dataset_manifest.json')
        assert manifest['queries']==manifests[OLD[0]]['queries']
        public={key(r):r for r in roster['rows'] if r['cohort']==c}
        assert len(public)==len(manifest['rows'])==16
        ids={key(r) for r in manifest['rows']};assert ids==set(public) and ids.isdisjoint(oldids|freshids);freshids.update(ids)
        for ref in manifest['rows']:
            for k in ('rgb_sha256','depth_K','depth_shape'):assert ref[k]==public[key(ref)][k]
        freshdata[c]=score(c,manifest,newsources,inputs,False,True)
        write(out/f'{c}_scores.json',dict(arms=freshdata[c]))
    primary={c:decisions(freshdata[c],frozen['primary']) for c in order}
    presult=save(out/'primary',primary,frozen['primary'],'fresh6_original_pooled304')
    assignments={c:i%3 for i,c in enumerate(order)}
    supplement={c:decisions(freshdata[c],frozen['supplemental'][str(assignments[c])]['arms']) for c in order}
    sresult=save(out/'supplemental-fresh',supplement,frozen['supplemental'],'fresh6_assigned_old6_four_cal',assignments)
    result=primary_summary(presult,order)
    result['fresh_capture_order']=order;result['fresh_fold_assignment']=assignments
    result['supplemental_fresh_aggregates']={b:aggregate(sresult,order,[b]) for b in BANDS}
    write(out/'summary.json',result)


def run(args):
    for name in ('repo','runroot','prior','backbones','output'):setattr(args,name,getattr(args,name).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'execution_plan.json').exists():raise FileExistsError('Preserve prior stages and failures')
    inputs=Inputs(args.budget_s);plan=inputs.json(args.runroot/'PLAN.json')
    assert plan['run']=='RGB_BACKBONE_CONFIRM_DEV_20261010'
    write(out/'execution_plan.json',dict(stage=args.stage,created_utc=utc(),source_sha256=sha(__file__),PLAN_sha256=sha(args.runroot/'PLAN.json'),
        budget_CPU_command_wall_s=args.budget_s,ARMS=ARMS,readout='absolute16th unchanged',training=0,neural_inference=0,
        primary='Copy frozen304 cuts; no pooledrefit',supplement='old6ordinal%3,FREE-onlyother4; sealbeforefreshscores',freshpublic='REGISTERED_GRID_ONES'))
    shutil.copyfile(__file__,out/'executed_evaluate.py');receipt=dict(status='STARTING',stage=args.stage)
    try:
        if args.stage=='calibrate':calibrate(args,inputs,out)
        else:final(args,inputs,out)
        receipt['status']='COMPLETE'
    except Exception as exc:receipt.update(status='FAILED_PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(command_wall_s=time.perf_counter()-START,completed_utc=utc(),checks=dict(inputs.checks),source_sha256=sha(__file__),GPU=0,training=0,downloads=0)
        write(out/'inputs.json',inputs.rows);write(out/'terminal.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','runroot','prior','backbones','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=('calibrate','final'),required=True);p.add_argument('--budget-s',type=float,required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--new-roster',type=Path);p.add_argument('--new-predictions',type=Path)
    run(p.parse_args())
