"""Five frozen RGB backbones x four readouts, consumed Development.

Calibration computes strictFREE scores only, including old LOCO calibration
material. All thresholds seal before any POS/UNKNOWN evaluation scores. Existing
train affine fits remain fixed; this program performs no neural inference.
"""
from __future__ import annotations
import time
START = time.perf_counter()
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import numpy as np
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_bounded_evaluate import manifest_rows
from rgb_body_query_negative_frozen_score import paths, EVAL, numeric
from rgb_body_query_query_calibration_probe import FREE, COHORTS, nth_score, select_cutoff, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_calibrated_geometry import correct
from rgb_body_query_residual_readout import summary, evaluate, paired
from rgb_body_query_scene_diagnostic import write_csv

ARMS = ('dav_raw', 'uni_raw', 'uni_affine', 'metric_raw', 'dp_affine')
READOUTS = ('R0', 'R1', 'R2', 'R3')
BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')


def identity(r): return r['scan'], r['frame']
def variant(a, r): return a+'/'+r


class Inputs:
    def __init__(self, budget):
        self.budget = budget; self.rows = []; self.hashes = {}; self.checks = Counter()
    def check(self):
        if time.perf_counter()-START >= self.budget: raise TimeoutError('Readout evaluator command-wall allocation exhausted')
    def register(self, path, expected=None):
        self.check(); path = Path(path); k = str(path.resolve())
        if k not in self.hashes:
            self.hashes[k] = sha(path); self.rows.append(dict(path=k, sha256=self.hashes[k]))
        if expected is not None: assert self.hashes[k] == expected, f'Changed frozen input: {path}'
        return path
    def json(self, path): return load(self.register(path))


def pred_index(path, inputs):
    data = inputs.json(path)
    assert data['status'] == 'COMPLETE', 'No silently selected partial inference subset'
    rows = {identity(r): r for r in data['rows']}
    assert len(rows) == len(data['rows'])
    return rows


def old_sources(repo, oldrun, inputs):
    model = {m:pred_index(oldrun/m/'predictions.json', inputs) for m in ('dav2','unidepth','metric3d')}
    uni = inputs.json(oldrun/'unidepth/evaluation/calibration/fit.json')['affine']
    _, _, candidate, _ = paths(repo)
    dpfit = inputs.json(candidate/'training_inputs.json')['affine']
    dp = {}
    root = repo/'artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010'
    for c in ('cal',)+EVAL:
        source = pred_index(root/f'predictions/{c}/predictions.json', inputs)
        for k,r in source.items():
            assert k not in dp
            dp[k] = dict(r, path=r['sampled_depth_path'], sha256=r['sampled_depth_sha256'])
    model['depthpro'] = dp
    return model, dict(uni_affine=uni, dp_affine=dpfit)


def native(ref, index, inputs):
    item = index[identity(ref)]
    for field in ('rgb_sha256','depth_K','depth_shape'): assert item[field] == ref[field]
    path = inputs.register(item['path'], item['sha256'])
    if path.suffix == '.npy': z = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as f: z = f['depth']
    assert list(z.shape) == ref['depth_shape']
    return np.asarray(z, np.float64)


def depths(ref, source, fits, inputs):
    raw = {m:native(ref, source[m], inputs) for m in ('dav2','unidepth','metric3d','depthpro')}
    return dict(dav_raw=raw['dav2'], uni_raw=raw['unidepth'], uni_affine=correct(raw['unidepth'],fits['uni_affine']),
                metric_raw=raw['metric3d'], dp_affine=correct(raw['depthpro'],fits['dp_affine'])), np.isfinite(raw['depthpro']) & (raw['depthpro']>0)


def score(cohort, manifest, source, fits, inputs, free_only):
    data = {variant(a,r):[] for a in ARMS for r in READOUTS}
    for ref in manifest['rows']:
        inputs.check()
        selected = [j for j,r in enumerate(ref['queries']) if r['state']==FREE] if free_only else list(range(27))
        if not selected: continue
        zz, public_valid = depths(ref,source,fits,inputs)
        labels = None
        if not free_only:
            inputs.register(ref['reference_path'],ref['reference_sha256'])
            with np.load(ref['reference_path']) as f: labels=f['labels']
            assert labels.shape == (27,*public_valid.shape)
        rx,ry = rays(ref['depth_K'],ref['depth_shape'])
        logs = {a:np.log(np.where(np.isfinite(z)&(z>0),z,1.)) for a,z in zz.items()}
        valid = {a:public_valid & np.isfinite(z)&(z>0) for a,z in zz.items()}
        for j in selected:
            query = manifest['queries'][j]; state=ref['queries'][j]['state']
            if free_only: assert state==FREE
            else:
                lab=labels[j]; pos,free,unk=[int((lab==n).sum()) for n in (1,0,2)]
                expected='POSITIVE' if pos>=16 else FREE if pos==0 and unk==0 and free>=16 else 'UNKNOWN'
                assert expected==state; inputs.checks['unchanged_eval_reference_states']+=1
            entry,exit,domain=ray_interval(rx,ry,query)
            halfwidth=(query['high'][2]-query['low'][2])/2
            assert halfwidth>0 and (entry[domain]>0).all() and (exit[domain]>0).all()
            band=f'{query["low"][2]:g}-{query["high"][2]:g}m'
            logentry=np.zeros(entry.shape); logexit=np.zeros(exit.shape)
            logentry[domain]=np.log(entry[domain]); logexit[domain]=np.log(exit[domain])
            for arm,z in zz.items():
                mask=valid[arm]&domain
                absolute=np.full(z.shape,-np.inf)
                absolute[mask]=np.minimum(z[mask]-entry[mask],exit[mask]-z[mask])
                logmargin=np.full(z.shape,-np.inf)
                logmargin[mask]=np.minimum(logs[arm][mask]-logentry[mask],logexit[mask]-logs[arm][mask])
                q0=nth_score(absolute); q2=nth_score(logmargin)
                w0=-np.inf if free_only else nth_score(absolute[labels[j]==1])
                w2=-np.inf if free_only else nth_score(logmargin[labels[j]==1])
                # Division is by one positive public constant per query, so it
                # preserves order exactly; the original16th statistic is retained.
                scores=dict(R0=(q0,w0),R1=(q0/halfwidth,w0/halfwidth),R2=(q2,w2),R3=(q0,w0))
                for readout,(qs,ws) in scores.items():
                    data[variant(arm,readout)].append(dict(cohort=cohort,environment=ref['environment'],scan=ref['scan'],frame=ref['frame'],
                        arm=arm,readout=readout,query=query['name'],distance_band=band,reference_state=state,
                        valid_ray_count=int(mask.sum()),public_valid_domain_rays=int((public_valid&domain).sum()),
                        missing_candidate_public_rays=int((public_valid&domain&~valid[arm]).sum()),
                        query_Z_halfwidth_m=halfwidth,query_score=qs,known_positive_score=ws,
                        calibration_FREE_only=free_only))
            inputs.checks['FREE_calibration_queries' if free_only else 'eval_queries']+=1
        inputs.checks['frames_with_FREE_calibration' if free_only else 'eval_frames']+=1
    return data


def choices(data):
    out={}
    for key,rr in data.items():
        free=[r for r in rr if r['reference_state']==FREE]
        if key.endswith('/R3'):
            out[key]={b:select_cutoff([r['query_score'] for r in free if r['distance_band']==b]) for b in BANDS}
        else: out[key]=select_cutoff([r['query_score'] for r in free])
    return out


def evaluation(data,cuts):
    return {key:evaluate(rr,cuts[key],key.endswith('/R3')) for key,rr in data.items()}


def expanded(rows,pairs=False):
    out=summary(rows,pairs); groups=defaultdict(list)
    for r in rows: groups[r['scan']].append(r)
    for scan,rr in groups.items():
        for key,value in summary(rr,pairs).items():
            if key=='all': out['capture/'+scan]=value
            elif key.startswith('band/'): out['capture_band/'+scan+'/'+key.removeprefix('band/')]=value
    return out


def save_protocol(out,ev,cuts,mode):
    out.mkdir(parents=True,exist_ok=True); ss={}; pp=[]; pair_summaries={}; table=[]
    for cohort,variants in ev.items():
        write(out/f'{cohort}_evaluation.json',variants)
        ss[cohort]={k:expanded(rr) for k,rr in variants.items()}
        for k,groups in ss[cohort].items():
            for g,s in groups.items(): table.append(dict(cohort=cohort,variant=k,group=g,**s))
        for k,rr in variants.items():
            own=variant(k.split('/')[0],'R0')
            for base,title in [(variant('dav_raw','R0'),'vs_DAVraw_R0'),(own,'vs_own_R0')]:
                pairs=paired(rr,variants[base],cohort,k,base,title)
                pp.extend(pairs); pair_summaries[cohort+'/'+k+'/'+title]=expanded(pairs,True)
    write(out/'results.json',dict(status='COMPLETE',protocol=mode,thresholds=cuts,cohorts=ss,pairs=pair_summaries,
        limits='Development; native first-return sampledqueries; UNKNOWN not negative; no safety or fresh confirmatory proof'))
    write_csv(out/'summary.csv',table); write_csv(out/'query_pairs.csv',pp)
    return ss,pair_summaries


def final_summary(ev,cohorts,ss,pair_ss,out,newcohorts):
    arkit=list(COHORTS)+newcohorts
    near=[]; aggregated=[]; strong={}; near_positive=[]; midfar_source=[]; all_eval_band=[]
    countfields=('positive_total','positive_known_witness','positive_support','free_total','free_support','unknown_total','unknown_support')
    for key in ev[COHORTS[0]]:
        arm,readout=key.split('/'); own=variant(arm,'R0'); qualifying=[]; midbase=0; midnow=0
        for c in arkit:
            cur=ss[c][key]['band/'+BANDS[0]]; base=ss[c][own]['band/'+BANDS[0]]
            tolerance=int(np.floor(.01*cur['free_total']))
            evaluable=cur['positive_total']>0 and cur['positive_known_witness'] is not None and base['positive_known_witness'] is not None
            gain=cur['positive_known_witness']-base['positive_known_witness'] if evaluable else None
            free_change=cur['free_support']-base['free_support'] if cur['free_support'] is not None and base['free_support'] is not None else None
            qualifies=bool(evaluable and gain>0 and free_change is not None and free_change<=tolerance)
            if qualifies: qualifying.append(c)
            near.append(dict(cohort=c,new_capture=c in newcohorts,variant=key,**cur,
                baseline_variant=own,baseline_known_witness=base['positive_known_witness'],baseline_free_support=base['free_support'],
                near_POS_evaluable=evaluable,positive_witness_gain=gain,FREE_change=free_change,
                allowed_FREE_increment=tolerance,qualifies=qualifies,
                pairs_vs_own=pair_ss[c+'/'+key+'/vs_own_R0']['band/'+BANDS[0]],
                pairs_vs_DAV=pair_ss[c+'/'+key+'/vs_DAVraw_R0']['band/'+BANDS[0]]))
            for b in BANDS[1:]:
                now=ss[c][key]['band/'+b]['positive_known_witness']; old=ss[c][own]['band/'+b]['positive_known_witness']
                if now is None or old is None: midnow=midbase=None; break
                if midnow is not None: midnow+=now; midbase+=old
        for band in BANDS:
            groups=[ss[c][key]['band/'+band] for c in arkit]
            def total(field):
                values=[g[field] for g in groups]; return sum(values) if all(v is not None for v in values) else None
            aggregated.append(dict(variant=key,band=band,captures=arkit,**{f:total(f) for f in countfields},
                POS_positive_captures=[c for c in arkit if ss[c][key]['band/'+band]['positive_total']>0]))
            allgroups=[ss[c][key]['band/'+band] for c in cohorts]
            allbase=[ss[c][own]['band/'+band] for c in cohorts]
            allcounts={}
            for field in countfields:
                values=[g[field] for g in allgroups]
                allcounts[field]=sum(values) if all(v is not None for v in values) else None
            bvalues=[g['positive_known_witness'] for g in allbase]
            own_w=sum(bvalues) if all(v is not None for v in bvalues) else None
            all_eval_band.append(dict(variant=key,band=band,cohorts=cohorts,**allcounts,own_R0_positive_known_witness=own_w,
                net_witness_change=allcounts['positive_known_witness']-own_w if allcounts['positive_known_witness'] is not None and own_w is not None else None))
        # Count-space criterion avoids a rounded fraction changing the decision.
        loss=None if midnow is None else midbase-midnow
        retention=None if loss is None else (20*loss<=midbase if midbase else loss<=0)
        strong[key]=dict(protocol='pooled304_on_ALL_ARKIT',qualifying_captures=qualifying,
            qualifying_count=len(qualifying),new_qualifying_captures=[c for c in qualifying if c in newcohorts],
            own_R0_midfar_witness=midbase,candidate_midfar_witness=midnow,net_witness_loss=loss,
            max_allowed_net_loss_count=None if midbase is None else int(np.floor(.05*midbase)),
            net_loss_ratio=None if not midbase or loss is None else loss/midbase,
            exact_5percent_count_check=retention,
            signal=bool(len(qualifying)>=3 and any(c in newcohorts for c in qualifying) and retention),
            missing_required_new_capture_coverage=len(newcohorts)<3)
        def sums(rows,fields):
            result={}
            for field in fields:
                values=[r[field] for r in rows]
                result[field]=sum(values) if all(v is not None for v in values) else None
            return result
        sourcegroups=[('old',list(COHORTS)),('new',newcohorts),('all',arkit),
                      ('old_all_eval',list(EVAL)),('3RScan',[c for c in EVAL if c not in COHORTS]),('all_eval',cohorts)]
        for origin,selected in sourcegroups:
            positive_captures=[c for c in selected if ss[c][key]['band/'+BANDS[0]]['positive_total']>0]
            nr=[ss[c][key]['band/'+BANDS[0]] for c in positive_captures]
            br=[ss[c][own]['band/'+BANDS[0]] for c in positive_captures]
            nq=sums(nr,countfields); nb=sums(br,countfields)
            nearpairs=sums([pair_ss[c+'/'+key+'/vs_own_R0']['band/'+BANDS[0]] for c in positive_captures],
                ('positive_rescue','positive_loss','free_added','free_removed'))
            if origin in ('old','new','all'):
                near_positive.append(dict(origin=origin,variant=key,captures=positive_captures,
                    excluded_zero_POS_captures=[c for c in selected if c not in positive_captures],**nq,**nearpairs,
                    own_R0_positive_known_witness=nb['positive_known_witness'],own_R0_free_support=nb['free_support']))
            mr=[ss[c][key]['band/'+b] for c in selected for b in BANDS[1:]]
            mb=[ss[c][own]['band/'+b] for c in selected for b in BANDS[1:]]
            mq=sums(mr,countfields); baseline=sums(mb,countfields)
            mp=sums([pair_ss[c+'/'+key+'/vs_own_R0']['band/'+b] for c in selected for b in BANDS[1:]],
                ('positive_rescue','positive_loss','free_added','free_removed'))
            midfar_source.append(dict(origin=origin,variant=key,captures=selected,**mq,**mp,
                own_R0_positive_known_witness=baseline['positive_known_witness'],own_R0_free_support=baseline['free_support'],
                net_witness_change=mq['positive_known_witness']-baseline['positive_known_witness'] if mq['positive_known_witness'] is not None and baseline['positive_known_witness'] is not None else None,
                net_FREE_change=mq['free_support']-baseline['free_support'] if mq['free_support'] is not None and baseline['free_support'] is not None else None))
    write(out/'summary.json',dict(status='COMPLETE',primary_protocol='pooled304',old_ARKIT=list(COHORTS),new_ARKIT=newcohorts,
        near_rows=near,aggregated_ARKIT=aggregated,strong_signal=strong,near_POS_positive_aggregates=near_positive,
        midfar_by_source=midfar_source,all_eval_bands=all_eval_band,
        reporting='Near signal onlycapturePOS>0; zeroPOS remains explicit in ledgers; all_eval midfar includes184frames(136old+48new) when new3capturecomplete',
        midfar_rule='Integer comparison20*(base-candidate)<=base; if base0 require no loss; net loss not gross pairedloss',
        new_capture_coverage_COMPLETE=len(newcohorts)==3))
    write_csv(out/'near_capture_table.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in near])
    write_csv(out/'aggregated_ARKIT_table.csv',aggregated)


def calibrate(args,inputs,fits,sources,out):
    manifest,mpaths=manifest_rows(args.repo)
    for path in mpaths: inputs.register(path)
    pooled=score('calibration_pooled304',manifest,sources,fits,inputs,True)
    assert all(len(rr)==588 for rr in pooled.values())
    write(out/'pooled_FREE_scores.json',dict(arms=pooled))
    _,_,_,folders=paths(args.repo); old={}
    for cohort in COHORTS:
        manifest=inputs.json(folders[cohort]/'dataset_manifest.json')
        old[cohort]=score(cohort,manifest,sources,fits,inputs,True)
    write(out/'old_LOCO_FREE_scores.json',old)
    lococuts={held:choices({k:[r for c in COHORTS if c!=held for r in old[c][k]] for k in pooled}) for held in COHORTS}
    write(out/'thresholds.json',dict(status='ALL_CUTS_FROZEN_BEFORE_EVAL_POS_UNKNOWN',frozen_utc=utc(),
        plan_sha256=sha(args.runroot/'PLAN.json'),pooled=choices(pooled),old_loco=lococuts,fits=fits,
        pooled_cal_frames=304,pooled_strict_FREE=588,old_LOCO_FREE_calibration_only=True,eval_positive_unknown_scores_computed=False,
        source_prediction_manifests={m:sha(args.oldrun/m/'predictions.json') for m in ('dav2','unidepth','metric3d')}))


def execute_final(args,inputs,fits,sources,out):
    cal=args.calibration; assert cal is not None
    assert inputs.json(cal/'terminal.json')['status']=='COMPLETE'
    frozen=inputs.json(cal/'thresholds.json')
    assert frozen['status']=='ALL_CUTS_FROZEN_BEFORE_EVAL_POS_UNKNOWN'
    assert frozen['plan_sha256']==sha(args.runroot/'PLAN.json') and frozen['fits']==fits
    assert frozen['source_prediction_manifests']=={m:sha(args.oldrun/m/'predictions.json') for m in ('dav2','unidepth','metric3d')}
    _,_,_,folders=paths(args.repo); data={}; cohortlist=list(EVAL)
    for c in EVAL:
        manifest=inputs.json(folders[c]/'dataset_manifest.json')
        data[c]=score(c,manifest,sources,fits,inputs,False)
        write(out/f'{c}_scores.json',dict(arms=data[c]))
    newcohorts=[]
    if args.new_roster is not None:
        roster=inputs.json(args.new_roster); names=sorted({r['cohort'] for r in roster['rows']})
        assert all(n.startswith('new_arkit_') for n in names)
        assert args.new_predictions is not None
        mapping=inputs.json(args.new_predictions)
        newsources={m:pred_index(mapping[m],inputs) for m in ('dav2','unidepth','metric3d','depthpro')}
        oldids={identity(r) for r in sources['dav2'].values()}
        for c in names:
            manifest=inputs.json(args.runroot/'new-eval-sensor'/c/'dataset_manifest.json')
            current={identity(r) for r in manifest['rows']}
            public={identity(r):r for r in roster['rows'] if r['cohort']==c}
            assert current==set(public) and current.isdisjoint(oldids)
            for ref in manifest['rows']:
                for k in ('rgb_sha256','depth_K','depth_shape'): assert ref[k]==public[identity(ref)][k]
            assert manifest['queries']==inputs.json(folders[COHORTS[0]]/'dataset_manifest.json')['queries']
            data[c]=score(c,manifest,newsources,fits,inputs,False); write(out/f'{c}_scores.json',dict(arms=data[c]))
            newcohorts.append(c); cohortlist.append(c)
    pooled={c:evaluation(data[c],frozen['pooled']) for c in cohortlist}
    ss,ps=save_protocol(out/'pooled',pooled,frozen['pooled'],'pooled304_all_old_new')
    loco={c:evaluation(data[c],frozen['old_loco'][c]) for c in COHORTS}
    save_protocol(out/'old-loco',loco,frozen['old_loco'],'old_three_LOCO_secondary')
    final_summary(pooled,cohortlist,ss,ps,out,newcohorts)


def run(args):
    args.repo=args.repo.resolve(); args.runroot=args.runroot.resolve(); args.oldrun=args.oldrun.resolve()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'execution_plan.json').exists(): raise FileExistsError('Preserve stages and account failed attempt command-wall')
    inputs=Inputs(args.budget_s)
    plan=inputs.json(args.runroot/'PLAN.json')
    assert plan['run']=='RGB_NEAR_READOUT_SCALE_DEV_20261010'
    write(out/'execution_plan.json',dict(stage=args.stage,frozen_utc=utc(),parent_plan_sha256=sha(args.runroot/'PLAN.json'),
        source_sha256=sha(__file__),budget_CPU_command_wall_s=args.budget_s,ARMS=ARMS,READOUTS=READOUTS,
        R1='Absolute per-ray margin divided by publicquery Z halfwidth=(highZ-lowZ)/2; positiveconstant order-preserving',
        R2='Per-ray min(logZ-logentry,logexit-logZ), then16th statistic',calibration='FREE-only pooled304 and original3LOCO, ALL cuts seal before POS/UNKNOWN eval',
        neural_inference=0,training=0,downloads=0))
    shutil.copyfile(__file__,out/'executed_evaluate.py'); receipt=dict(status='STARTING',stage=args.stage)
    try:
        source,fits=old_sources(args.repo,args.oldrun,inputs)
        if args.stage=='calibrate': calibrate(args,inputs,fits,source,out)
        else: execute_final(args,inputs,fits,source,out)
        receipt['status']='COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL',error=repr(exc)); raise
    finally:
        receipt.update(command_wall_s=time.perf_counter()-START,completed_utc=utc(),checks=dict(inputs.checks),
            source_sha256=sha(__file__),training=0,GPU=0,download_bytes=0)
        write(out/'inputs.json',inputs.rows); write(out/'terminal.json',receipt); print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--runroot',type=Path,required=True)
    p.add_argument('--oldrun',type=Path,required=True); p.add_argument('--stage',choices=('calibrate','final'),required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--budget-s',type=float,required=True)
    p.add_argument('--calibration',type=Path); p.add_argument('--new-roster',type=Path); p.add_argument('--new-predictions',type=Path)
    run(p.parse_args())
