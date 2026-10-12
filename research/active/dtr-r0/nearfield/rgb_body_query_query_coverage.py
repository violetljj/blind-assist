"""Fixed legacy-query witness and reference-coverage diagnostic; no fitting."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_reference_eval import rays, ray_interval

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))

def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def csvwrite(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def groups(rows, keys, fields):
    buckets=defaultdict(list)
    for r in rows: buckets[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rs in sorted(buckets.items()):
        v=dict(zip(keys,key));v['query_records']=len(rs)
        v.update({f:sum(r[f] for r in rs) for f in fields})
        if 'known' in v:
            for num,den in [('known','domain'),('positive','domain'),('free','domain'),('unknown','domain'),('observed_valid','domain')]:
                v[num+'_fraction']=v[num]/v[den] if v[den] else None
        result.append(v)
    return result

def witness_pair(candidate, baseline):
    a,b=candidate['tp']>=16,baseline['tp']>=16
    return dict(candidate_witness=int(a),baseline_witness=int(b),rescue=int(a and not b),loss=int(b and not a),
                both_hit=int(a and b),neither_hit=int(not a and not b))

def focused():
    # Full predicted support from UNKNOWN never becomes a known positive witness.
    a=dict(tp=15,predicted_support=999);b=dict(tp=16,predicted_support=16)
    p=witness_pair(a,b);assert p['loss']==1 and p['candidate_witness']==0
    rows=[dict(q='x',positive=2),dict(q='y',positive=3),dict(q='x',positive=5)]
    assert {r['q']:r['positive'] for r in groups(rows,['q'],['positive'])}=={'x':7,'y':3}
    labels=np.array([[0,1,255],[0,2,255]],np.uint8)
    reuse=np.isin(labels,[0,1]).sum(axis=0)
    assert np.array_equal(reuse,[2,1,0]) and sum(reuse)==3
    return dict(status='PASS',unknown_full_support_not_witness=True,group_sum=True,physical_ray_reuse=True)

def run(repo,out,budget):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Preserve prior diagnostic')
    def check():
        if time.perf_counter()-start>=budget:raise TimeoutError('CPU budget reached')
    base=repo/'artifacts.local/work';previous=base/'rgb-body-query-calibrated-transfer-dev-20261009/geometry'
    sources={
        'original_validation':base/'rgb-body-query-cross-session-dev-20261009/sensor',
        'new_3rscan':base/'rgb-body-query-transfer-dev-20261009/sensor',
        'arkit':base/'rgb-body-query-transfer-dev-20261009/camera-sensor',
        'native_vga':base/'rgb-body-query-input-baseline-dev-20261009/paired-sensor/native_vga',
        'derived_256':base/'rgb-body-query-input-baseline-dev-20261009/paired-sensor/derived_256'}
    readout=base/'rgb-body-query-metric-diagnostic-dev-20261009/readout'
    frozen_files=[previous/'fit.json',previous/'calibration.json',previous/'plan.json',readout/'receipt.json',readout/'evaluation.json',readout/'observation_normalization.npz',readout/'depth_only_final.pt',readout/'geometry_final.pt']
    frozen={str(p):sha(p) for p in frozen_files}
    plan=dict(status='FROZEN_BEFORE_NEW_COMPUTATION',created_at_utc=datetime.now(timezone.utc).isoformat(),baseline='5844eacd',cpu_budget_s=budget,gpu_s=0,download_bytes=0,
        compute_status='TASK_NOT_GPU_SUITABLE',cohorts=list(sources),arms='All 8 saved arms; unchanged scores/cutoffs/labels',
        witness='TP >=16, only reference-positive queries; full predicted support >=16 reported separately',
        pairs='depth_only vs every other saved arm, all positive-query frame records',
        group_dimensions=['all','environment','distance_band','query','environment_distance_band','environment_query'],
        coverage='All fixed 15 queries; reachable public ray domain, positive/free/unknown/ignored, valid observed first return and before-entry occlusion; no relabeling',
        correlation='Per-frame physical grid rays multiplicity for domain, labeled domain and known positive/free across fixed queries; frames also correlated',
        decision='Locate head witness coverage and missing negative reference; does not establish layout or model causality',
        source_sha256=sha(__file__),frozen_sha256=frozen)
    write(out/'plan.json',plan);write(out/'focused_check.json',focused())
    cover=[];arm_records=[];pair_records=[];reuse_rows=[];counts=Counter(); identities={}
    for cohort,sensor in sources.items():
        check();mf=sensor/'dataset_manifest.json';ep=previous/f'{cohort}_evaluation.json'
        manifest=load(mf);evaluation=load(ep);identities[str(mf)]=sha(mf);identities[str(ep)]=sha(ep)
        queries=manifest['queries'];assert len(queries)==15
        byarm={a:{(r['scan'],r['frame'],r['query']):r for r in v['records']} for a,v in evaluation['arms'].items()}
        assert len(byarm)==8
        frames=[r for r in manifest['rows'] if r['split']=='validation'] if cohort=='original_validation' else manifest['rows']
        for r in frames:
            check();assert sha(r['reference_path'])==r['reference_sha256']
            identities[r['reference_path']]=r['reference_sha256']
            with np.load(r['reference_path']) as ref:
                labels=ref['labels'];depth=ref['depth'];observed=ref['observed']
            rx,ry=rays(r['depth_K'],r['depth_shape']);valid=observed & np.isfinite(depth) & (depth>0)
            dmult=np.zeros(depth.shape,np.uint8);lmult=dmult.copy();kmult=dmult.copy()
            for j,q in enumerate(queries):
                entry,exit,domain=ray_interval(rx,ry,q);lab=labels[j];known=np.isin(lab,[0,1]);labeled=lab!=255
                # VGA shared-boundary exclusions remain IGNORE even when public rays intersect.
                assert not (labeled & ~domain).any()
                positive=int((lab==1).sum());free=int((lab==0).sum());unknown=int((lab==2).sum());ignored=int((domain & (lab==255)).sum())
                assert positive+free+unknown+ignored==int(domain.sum())
                key=r['scan'],r['frame'],q['name'];saved=byarm['depth_only'][key]
                assert positive==saved['tp']+saved['fn'] and free==saved['fp']+saved['tn']
                state=saved['reference_state'];band=saved['distance_band'];location=q['name'].split('_')[0]
                cr=dict(cohort=cohort,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],location=location,distance_band=band,reference_state=state,
                    domain=int(domain.sum()),positive=positive,free=free,unknown=unknown,ignored=ignored,known=positive+free,
                    observed_valid=int((domain & valid).sum()),before_entry=int((domain & valid & (depth<entry-1e-12)).sum()),
                    unavailable=int((domain & ~valid).sum()),positive_query=int(state=='POSITIVE'),sampled_free_query=int(state=='FREE_ON_SAMPLED_RAYS'),unknown_query=int(state=='UNKNOWN'))
                cover.append(cr);dmult+=domain;lmult+=labeled;kmult+=known
                for arm,lookup in byarm.items():
                    a=lookup[key];assert a['tp']+a['fn']==positive and a['fp']+a['tn']==free
                    arm_records.append(dict(**{k:cr[k] for k in ('cohort','environment','scan','frame','query','location','distance_band')},arm=arm,reference_state=state,
                        positive_query=int(state=='POSITIVE'),known_witness_hit=int(state=='POSITIVE' and a['tp']>=16),
                        predicted_support=a['predicted_support'],full_support_any_query=int(a['predicted_support']>=16),
                        unknown_query=int(state=='UNKNOWN'),unknown_full_support=int(state=='UNKNOWN' and a['predicted_support']>=16),
                        full_support_hit=int(state=='POSITIVE' and a['predicted_support']>=16),
                        unsupported_full_hit=int(state=='POSITIVE' and a['predicted_support']>=16 and a['tp']<16),
                        sampled_free_query=int(state=='FREE_ON_SAMPLED_RAYS'),sampled_free_false_support=int(state=='FREE_ON_SAMPLED_RAYS' and a['predicted_support']>=16),
                        tp=a['tp'],fn=a['fn'],fp=a['fp'],tn=a['tn']))
                    counts['arm_record_checks']+=1
                if state=='POSITIVE':
                    head=byarm['depth_only'][key]
                    for baseline,lookup in byarm.items():
                        if baseline=='depth_only':continue
                        pair_records.append(dict(**{k:cr[k] for k in ('cohort','environment','scan','frame','query','location','distance_band')},candidate='depth_only',baseline=baseline,
                            **witness_pair(head,lookup[key])))
            for kind,mult in [('geometric_domain',dmult),('labeled_domain',lmult),('known_positive_free',kmult)]:
                h=np.bincount(mult.ravel(),minlength=16)
                assert sum(int(h[i])*i for i in range(16))==int(mult.sum())
                for m,n in enumerate(h):reuse_rows.append(dict(cohort=cohort,environment=r['environment'],scan=r['scan'],frame=r['frame'],kind=kind,multiplicity=m,physical_grid_rays=int(n),query_ray_units=int(n)*m))
                counts['ray_reuse_checks']+=1
        for arm,v in evaluation['arms'].items():
            rr=[a for a in arm_records if a['cohort']==cohort and a['arm']==arm];s=v['summary']['all']
            for f in ('tp','fn','fp','tn'):assert sum(a[f] for a in rr)==s[f]
            assert sum(a['known_witness_hit'] for a in rr)==s['query_positive_known_witness_hits']
            assert sum(a['full_support_hit'] for a in rr)==s['query_positive_hits']
            assert sum(a['sampled_free_false_support'] for a in rr)==s['sampled_free_false_support']
            counts['full_summary_checks']+=1
    csvwrite(out/'frame_query_reference_coverage.csv',cover);csvwrite(out/'frame_query_all_arms.csv',arm_records);csvwrite(out/'frame_query_head_witness_pairs.csv',pair_records);csvwrite(out/'physical_ray_reuse_histograms.csv',reuse_rows)
    dims={'all':[],'environment':['environment'],'band':['distance_band'],'query':['query'],'environment_band':['environment','distance_band'],'environment_query':['environment','query']}
    coverage_fields=['domain','positive','free','unknown','ignored','known','observed_valid','before_entry','unavailable','positive_query','sampled_free_query','unknown_query']
    arm_fields=['positive_query','known_witness_hit','predicted_support','full_support_any_query','unknown_query','unknown_full_support','full_support_hit','unsupported_full_hit','sampled_free_query','sampled_free_false_support','tp','fn','fp','tn']
    pair_fields=['candidate_witness','baseline_witness','rescue','loss','both_hit','neither_hit']
    result={}
    for label,extra in dims.items():
        c=groups(cover,['cohort']+extra,coverage_fields);a=groups(arm_records,['cohort','arm']+extra,arm_fields);p=groups(pair_records,['cohort','baseline']+extra,pair_fields)
        for row in p:assert row['rescue']-row['loss']==row['candidate_witness']-row['baseline_witness']
        csvwrite(out/f'coverage_{label}.csv',c);csvwrite(out/f'arms_{label}.csv',a);csvwrite(out/f'witness_pairs_{label}.csv',p)
        if label in ('all','band','query'):result[label]=dict(coverage=c,arms=a,pairs=p)
    hist=groups(reuse_rows,['cohort','kind','multiplicity'],['physical_grid_rays','query_ray_units']);csvwrite(out/'physical_ray_reuse_cohort_histograms.csv',hist)
    result['ray_reuse']=hist
    assert all(sha(p)==h for p,h in frozen.items());check()
    write(out/'source_identities.json',identities);shutil.copyfile(__file__,out/'executed_query_coverage.py')
    result.update(status='COMPLETE',checks=dict(counts),wall_s=time.perf_counter()-start,cpu_budget_s=budget,gpu_s=0,download_bytes=0,frozen_sha256=frozen)
    write(out/'evaluation.json',result)
    lines=['# 固定15查询的见证与参考覆盖诊断','','全部既有5cohort/8臂，不训练、不改阈值；TP≥16为已知正见证，full support另列。相关query-ray不当独立N。','','| cohort | 正query | 采样free | UNKNOWN | known/domain | 未知/domain |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result['all']['coverage']:lines.append(f"| {r['cohort']} | {r['positive_query']} | {r['sampled_free_query']} | {r['unknown_query']} | {r['known_fraction']:.4f} | {r['unknown_fraction']:.4f} |")
    lines+=['','| cohort | 对照 | head见证 | 对照见证 | 救回/损失 |','| --- | --- | ---: | ---: | ---: |']
    for r in result['all']['pairs']:lines.append(f"| {r['cohort']} | {r['baseline']} | {r['candidate_witness']} | {r['baseline_witness']} | {r['rescue']}/{r['loss']} |")
    lines+=['','全量逐环境、距离带、查询位置、环境×带/查询CSV并列；UNKNOWN包含前方遮挡和不可用深度，并不表示空闲。即使全采样free也不能证明盒内每处空闲。该分组诊断只能定位条件差异，不能独立认定布局先验或读出机制原因。',f"\n执行 {result['wall_s']:.3f}s / CPU研究上限 {budget:g}s；GPU0、下载0；无后台worker/service，子进程结束后释放。"]
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'completion_receipt.json',dict(status='COMPLETE',measured_research_s=result['wall_s'],conservative_cpu_booking_s=30,cpu_budget_s=budget,gpu_s=0,download_bytes=0,
        source_sha256=sha(__file__),result_sha256=sha(out/'evaluation.json'),frozen_sha256=frozen,resource_release='CPU subprocess exits; no workers/services/downloads'))
    print(json.dumps(dict(status='COMPLETE',checks=dict(counts),wall_s=result['wall_s']),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget-s',type=float,default=300)
    a=p.parse_args();run(a.repo,a.output,a.budget_s)
