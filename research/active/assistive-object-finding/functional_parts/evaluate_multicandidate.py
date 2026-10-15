#!/usr/bin/env python3
"""Frozen multi-candidate Development probe; GT is only used after prediction."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import evaluate as old
from run_baseline import ACTIONS
cv2.setNumThreads(2)


def quality(row):
    return row["GT_relative_peak_proxy"] if row["dataset"] == "AGD20K" else float(row["native_union_hit"])


def compact_summary(rows):
    values = [quality(r) for r in rows if quality(r) is not None]
    result = {"images": len(rows), "evaluable": len(values),
              "quality_mean": float(np.mean(values)) if values else None,
              "risk_mean": float(np.mean([1-v for v in values])) if values else None}
    if rows and rows[0]["dataset"] == "IIT-AFF":
        result.update(hits=sum(r["native_union_hit"] for r in rows),
                      native_empty_images=sum(r["native_empty"] for r in rows),
                      native_nonempty_images=sum(not r["native_empty"] for r in rows),
                      grasp5_hits=sum(r["native_grasp5_hit"] for r in rows),
                      wrap9_hits=sum(r["native_wrap9_hit"] for r in rows))
    return result


def paired(ref, trial):
    before = {r["sample_id"]: r for r in ref}
    after = {r["sample_id"]: r for r in trial}
    if set(before) != set(after):
        raise ValueError("Unpaired comparison")
    delta = [quality(after[s])-quality(before[s]) for s in sorted(before)
             if quality(after[s]) is not None and quality(before[s]) is not None]
    return {"images": len(before), "evaluable": len(delta),
            "mean_quality_delta": float(np.mean(delta)) if delta else None,
            "improved": sum(v>1e-12 for v in delta), "worsened": sum(v < -1e-12 for v in delta),
            "tied": sum(abs(v)<=1e-12 for v in delta)}


def ranked_coverage(rows, confidence):
    # Rank only observed confidence. GT does not choose retained indices or K.
    def key(r):
        value = r["confidence"].get(confidence)
        valid = value is not None and math.isfinite(value)
        return (not valid, -value if valid else 0, r["sample_id"])
    ranked = sorted(rows, key=key)
    curve=[]
    for k in range(1, len(rows)+1):
        result = compact_summary(ranked[:k])
        curve.append({"retained": k, "denominator": len(rows), "coverage": k/len(rows), **result})
    risks = [r["risk_mean"] for r in curve if r["risk_mean"] is not None]
    indices = sorted(set(max(1, math.ceil(c*len(rows))) for c in (.5,.7,.8,.9,1.)))
    return {"interpretation": "Evaluation confidence-rank diagnostic, not a deployable threshold; sample_id ties; nonfinite ranked last",
            "AURC_discrete_mean": float(np.mean(risks)) if risks else None,
            "matched_coverage": [curve[k-1] for k in indices], "curve": curve,
            "ranked_sample_ids": [r["sample_id"] for r in ranked]}


METHODS = ("B0", "B1", "B2", "B3", "C0", "C1", "C2", "C3", "C4")


def predict_methods(observation):
    """Only pixels, model maps and public action enter this function."""
    w,h=int(observation["width"]),int(observation["height"])
    views=observation["views"]
    base_raw=old.load_map(views["base"]["heads"]["ree"])
    B=old.official_minmax(base_raw,(w,h))
    base,degenerate=old.probability(base_raw,(w,h))
    probabilities={"base":base}
    for name,view in views.items():
        if name=="base":continue
        if not view["map_inverse_restored"]:raise ValueError("Unrestored view")
        raw=old.load_map(view["heads"]["ree"])
        probabilities[name]=old.projected_crop(raw,view["bbox_xyxy"],(w,h))[0]
    details={}
    for family in ("candidates","fixed"):
        entries=[]
        for spec in observation[family]:
            view=views[spec["name"]];box=view["bbox_xyxy"]
            x0,y0,x1,y1=box
            L=old.official_minmax(old.load_map(view["heads"]["ree"]),(x1-x0,y1-y0))
            localq=old.argmax_point(L);q=[localq[0]+x0,localq[1]+y0]
            if family=="candidates":c=spec["seed_xy"]
            else:
                localc=old.argmax_point(B[y0:y1,x0:x1]);c=[localc[0]+x0,localc[1]+y0]
            logits=np.load(view["all_action_logits_path"],allow_pickle=False)
            if logits.shape!=(36,14,14) or not np.isfinite(logits).all():raise ValueError("Invalid action logits")
            gap=logits.astype(np.float64).mean(axis=(1,2));exp=np.exp(gap-gap.max())
            semantic=float(exp[ACTIONS.index(observation["action"])]/exp.sum())
            bq=old.bilinear_value(B,q);lc=old.bilinear_value(L,[c[0]-x0,c[1]-y0])
            spatial=math.sqrt(max(0.,bq*lc));combined=max(0.,bq*lc*semantic)**(1./3.)
            entries.append({"view":spec["name"],"anchor_xy":c,"point_xy":q,"bbox_xyxy":box,
                            "base_at_refined":bq,"local_at_anchor":lc,"spatial":spatial,
                            "semantic":semantic,"combined":combined,"local_degenerate":bool(L.max()<=0)})
        if len(entries)!=3:raise ValueError("Three distinct candidates/fixed crops required")
        details[family]=entries
    maps={"B0":base,
          "B1":old.normalize_mass(sum(probabilities[k] for k in ("base","flip","center","flip_center"))*.25),
          "B2":old.normalize_mass((base+sum(probabilities[s["name"]] for s in observation["fixed"] ))*.25),
          "C0":old.normalize_mass((base+sum(probabilities[s["name"]] for s in observation["candidates"] ))*.25)}
    points={k:old.argmax_point(p) for k,p in maps.items()}
    confidence={k:old.spatial_confidence(p) for k,p in maps.items()}
    selected={}
    for method,family,score in (("B3","fixed","combined"),("C1","candidates","combined"),
                              ("C2","candidates","spatial"),("C3","candidates","semantic"),("C4","candidates","combined")):
        entries=details[family]
        index=0 if method=="C1" else max(range(len(entries)),key=lambda i:entries[i][score])
        selected[method]=index;points[method]=entries[index]["point_xy"];confidence[method]=entries[index][score]
    flipconf=None if degenerate or probabilities["flip"].sum()<=0 else 1-math.dist(points["B0"],old.argmax_point(probabilities["flip"]))/math.hypot(w,h)
    timing={}
    groups={"B0":["base"],"B1":["base","flip","center","flip_center"],
            "B2":["base"]+[s["name"] for s in observation["fixed"]]}
    groups["B3"]=groups["B2"]
    for name in ("C0","C1","C2","C3","C4"):groups[name]=["base"]+[s["name"] for s in observation["candidates"]]
    for name,names in groups.items():timing[name]=sum(views[v]["elapsed"]["actual_view_s"] for v in names)
    return maps,points,confidence,selected,details,flipconf,timing,degenerate


def score(args):
    started=time.monotonic();cpu=time.process_time()
    root=args.root;oldroot=args.old_root;out=root/"analysis"
    out.mkdir(parents=True,exist_ok=True)
    if (out/"sealed_gates.json").exists():raise FileExistsError("Preserve sealed run")
    plan=Path(__file__).with_name("MULTICANDIDATE_PLAN.md")
    prediction_path=args.predictions or root/"inference/predictions.jsonl"
    manifest=oldroot/"data/samples.jsonl"
    rows=old.read_jsonl(manifest);observations=old.read_jsonl(prediction_path)
    byid={r["sample_id"]:r for r in observations}
    if len(rows)!=280 or len(observations)!=280 or len(byid)!=280 or {r['sample_id'] for r in rows}!=set(byid):raise ValueError("Expected280 matching unique samples")
    if len({r['image_sha256'] for r in rows})!=280:raise ValueError("Duplicate source image")
    for row in rows:
        row['phase']=old.split_name(row);row['dataset_key']='AGD20K' if old.is_agd(row) else 'IIT-AFF'
    expected={('AGD20K','calibration'):80,('AGD20K','evaluation'):160,('IIT-AFF','calibration'):16,('IIT-AFF','evaluation'):24}
    if dict(Counter((r['dataset_key'],r['phase']) for r in rows))!=expected:raise ValueError("Split counts changed")
    oldgates=json.loads((oldroot/"analysis/sealed_gates.json").read_text(encoding="utf-8"))
    cutoffs=oldgates['calibration_GT_strata_medians_evaluator_only']
    inputs={"plan_sha256":old.sha256(plan),"prediction_sha256":old.sha256(prediction_path),
            "manifest_sha256":old.sha256(manifest),"script_sha256":old.sha256(__file__),
            "old_evaluator_sha256":old.sha256(Path(old.__file__)),"old_gates_sha256":old.sha256(oldroot/"analysis/sealed_gates.json")}
    (out/"evaluate_multicandidate_at_scoring.py").write_bytes(Path(__file__).read_bytes())
    records=[];candidate_records=[]
    def process(row):
        if time.process_time()-cpu>600:raise TimeoutError("CPU budget600s")
        obs=byid[row['sample_id']]
        if obs['protocol_sha256']!=inputs['plan_sha256'] or obs['primary_head']!='ree':raise ValueError("Protocol/head mismatch")
        for key in ('image_sha256','action','width','height'):
            if obs[key]!=row[key]:raise ValueError("Identity mismatch")
        maps,points,confidence,selected,details,flipconf,timing,degenerate=predict_methods(obs)
        # The sole GT open follows all observation-only decisions.
        gt=cv2.imread(row['gt_path'],cv2.IMREAD_GRAYSCALE)
        if gt is None or gt.shape!=(row['height'],row['width']) or old.sha256(row['gt_path'])!=row['gt_sha256']:raise ValueError("GT integrity failure")
        agd=row['dataset_key']=='AGD20K';tree=boundary=None
        region=np.isin(gt,[5,9]);empty=None if agd else not bool(region.any())
        if not agd and (row.get('target_label_ids')!=[5,9] or any(int(v) not in range(10) for v in np.unique(gt))):
            raise ValueError('IIT native label contract changed')
        if agd:
            yy,xx=np.where(gt==gt.max());tree=cKDTree(np.column_stack((xx+.5,yy+.5)))
        elif region.any():
            inside=distance_transform_edt(np.pad(region,1))[1:-1,1:-1]
            boundary=np.where(region,inside,-distance_transform_edt(~region))
        geometry=old.gt_geometry(gt.astype(float) if agd else region.astype(float))
        meta={"sample_id":row['sample_id'],"image_sha256":row['image_sha256'],"dataset":row['dataset_key'],"phase":row['phase'],
              "action":row['action'],"object":row['object'],"native_empty":empty,"geometry":geometry,"base_degenerate":degenerate}
        def metric(point):return old.agd_point_metrics(gt,point,tree) if agd else old.iit_point_metrics(gt,point,boundary)
        for method in METHODS:
            result={**meta,"method":method,"point_xy":points[method],"confidence":{"primary":confidence[method]},
                    "selected_index":selected.get(method),"model_calls":1 if method=='B0' else 4,
                    "view_inference_s":timing[method],**metric(points[method])}
            if method=='B0':result['confidence']['flip_consistency']=flipconf
            if agd and method in maps:
                metricmap=old.load_map(obs['views']['base']['heads']['ree']) if method=='B0' else maps[method]
                result.update(old.native_agd_from_raw(metricmap,gt))
            records.append(result)
        for family,entries in details.items():
            for i,entry in enumerate(entries):
                candidate_records.append({**meta,"family":family,"index":i,**entry,
                    "refined_metrics":metric(entry['point_xy']),"seed_metrics":metric(entry['anchor_xy'])})
    for row in sorted((r for r in rows if r['phase']=='calibration'),key=lambda r:r['sample_id']):process(row)
    settings={}
    for dataset in ('AGD20K','IIT-AFF'):
        settings[dataset]={}
        for method in METHODS:
            subset=[r for r in records if r['dataset']==dataset and r['method']==method]
            settings[dataset][method]=old.calibrate_coverage([r['confidence']['primary'] for r in subset],len(subset))
        subset=[r for r in records if r['dataset']==dataset and r['method']=='B0']
        settings[dataset]['B0_flip_consistency']=old.calibrate_coverage([r['confidence']['flip_consistency'] for r in subset],len(subset))
    old.write_json(out/'sealed_gates.json',{'settings':settings,'inputs':inputs,'evaluation_scored_before_seal':False,
        'evidence':'Reused Development; not fresh confirmation','size_cutoffs_reused':cutoffs})
    for row in sorted((r for r in rows if r['phase']=='evaluation'),key=lambda r:r['sample_id']):process(row)
    previous={r['sample_id']:r for r in old.read_jsonl(oldroot/'analysis/per_image_method_metrics.jsonl') if r['method']=='M1_base_argmax'}
    for r in records:
        if r['method']!='B0':continue
        prior=previous[r['sample_id']]
        key='GT_relative_peak_proxy' if r['dataset']=='AGD20K' else 'native_union_hit'
        if r['point_xy']!=prior['point_xy_pixel_centers'] or r[key]!=prior[key]:
            raise ValueError('B0 inherited point/quality differs from prior pilot')
    old.write_jsonl(out/'per_image_method_metrics.jsonl',records)
    old.write_jsonl(out/'per_candidate_metrics.jsonl',candidate_records)
    summary={'status':'COMPLETE','evidence':'Reused Development','inputs':inputs,'B0_reuse_exact_check_images':280,
             'datasets':{},'paired':{},'gates':{},'ranking':{},'oracle':{},'size_strata':{},'bootstrap':{}}
    gate_rows=[]
    for dataset in ('AGD20K','IIT-AFF'):
        evaluation=[r for r in records if r['dataset']==dataset and r['phase']=='evaluation']
        subsets={m:[r for r in evaluation if r['method']==m] for m in METHODS}
        summary['datasets'][dataset]={phase:{m:compact_summary([r for r in records if r['dataset']==dataset and r['phase']==phase and r['method']==m]) for m in METHODS} for phase in ('calibration','evaluation')}
        for method,subset in subsets.items():
            times=[r['view_inference_s'] for r in subset]
            summary['datasets'][dataset]['evaluation'][method]['timing_s']={'mean':float(np.mean(times)),'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'calls':1 if method=='B0' else 4}
            if dataset=='AGD20K':
                keys=['GT_relative_peak_proxy','local_GT_soft_mass_proxy','GT_peak_distance_diag']
                if method in ('B0','B1','B2','C0'):keys+=['KLD','SIM','NSS']
                summary['datasets'][dataset]['evaluation'][method]['metrics']={key:old.finite_mean([r.get(key) for r in subset],len(subset)) for key in keys}
        summary['paired'][dataset]={trial+'_vs_'+ref:paired(subsets[ref],subsets[trial]) for ref in METHODS for trial in METHODS if ref!=trial}
        summary['ranking'][dataset]={};summary['gates'][dataset]={}
        for method in METHODS+('B0_flip_consistency',):
            subset=subsets['B0' if method=='B0_flip_consistency' else method]
            conf='flip_consistency' if method=='B0_flip_consistency' else 'primary'
            threshold=settings[dataset][method]['threshold']
            accepted=[]
            for row in subset:
                value=row['confidence'][conf]
                accept=value is not None and threshold is not None and math.isfinite(value) and value>=threshold
                gate_rows.append({'sample_id':row['sample_id'],'dataset':dataset,'method':method,'accepted':accept,'confidence':value,'threshold':threshold})
                if accept:accepted.append(row)
            summary['gates'][dataset][method]={'calibration':settings[dataset][method],'coverage_n':len(accepted),'coverage_d':len(subset),
                'retained':compact_summary(accepted),'quality_sum_retained':sum(quality(r) for r in accepted if quality(r) is not None)}
            summary['ranking'][dataset][method]=ranked_coverage(subset,conf)
        evalc=[r for r in candidate_records if r['dataset']==dataset and r['phase']=='evaluation']
        metric_key='GT_relative_peak_proxy' if dataset=='AGD20K' else 'native_union_hit'
        summary['oracle'][dataset]={}
        for family in ('candidates','fixed'):
            group={sid:[r for r in evalc if r['sample_id']==sid and r['family']==family] for sid in sorted({r['sample_id'] for r in evalc})}
            for kind in ('refined_metrics','seed_metrics'):
                oracle={sid:max(float(r[kind][metric_key]) for r in entries) for sid,entries in group.items()}
                summary['oracle'][dataset][family+'_'+kind]={'images':len(oracle),'quality_mean':float(np.mean(list(oracle.values()))),
                    'gap_to_methods':{method:float(np.mean([oracle[r['sample_id']]-quality(r) for r in subsets[method]])) for method in METHODS},
                    'per_image':oracle}
        summary['size_strata'][dataset]={}
        cutoff=cutoffs[dataset]['effective_area_fraction']
        for label in ('small','large','NOT_EVALUABLE'):
            def stratum(r):
                val=r['geometry']['effective_area_fraction']
                return 'NOT_EVALUABLE' if val is None else 'small' if val<=cutoff else 'large'
            summary['size_strata'][dataset][label]={m:compact_summary([r for r in subsets[m] if stratum(r)==label]) for m in METHODS}
        summary['bootstrap'][dataset]={}
        for ref in ('B1','B2','B3'):
            a={r['sample_id']:quality(r) for r in subsets[ref]};b={r['sample_id']:quality(r) for r in subsets['C4']}
            delta=np.asarray([b[s]-a[s] for s in sorted(a)],dtype=float)
            rng=np.random.default_rng(20261015);draw=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
            summary['bootstrap'][dataset]['C4_vs_'+ref]={'mean_delta':float(delta.mean()),'percentile95_low':float(np.percentile(draw,2.5)),
                'percentile95_high':float(np.percentile(draw,97.5)),'draws':2000,'seed':20261015,'unit':'same image','interpretation':'Descriptive reused-Development uncertainty'}
    old.write_jsonl(out/'per_image_gate_decisions.jsonl',gate_rows)
    summary['cpu_seconds']=time.process_time()-cpu;summary['wall_seconds']=time.monotonic()-started
    old.write_json(out/'summary.json',summary)
    old.write_json(out/'receipt.json',{'status':'COMPLETE','inputs':inputs,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-started,
        'new_download_bytes':0,'gpu_seconds':0,'outputs':{p.name:old.sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'status':'COMPLETE','cpu_seconds':summary['cpu_seconds'],'datasets':summary['datasets'],'bootstrap':summary['bootstrap']}))


def self_test():
    assert paired([{'sample_id':'a','dataset':'AGD20K','GT_relative_peak_proxy':.2}],
                  [{'sample_id':'a','dataset':'AGD20K','GT_relative_peak_proxy':.8}])['improved']==1
    rows=[{'sample_id':str(i),'dataset':'AGD20K','GT_relative_peak_proxy':q,'confidence':{'primary':c}} for i,(q,c) in enumerate(((0.,.1),(1.,.9),(.5,.5)))]
    ranked=ranked_coverage(rows,'primary')
    assert ranked['ranked_sample_ids']==['1','2','0']
    assert ranked['curve'][0]['quality_mean']==1
    assert old.calibrate_coverage([.1,.2,.3,.4,.5],5)['threshold']==.2
    old.self_test()
    print(json.dumps({'status':'PASS','checks':['paired deltas','confidence-only ranking and ties','calibration coverage','inherited geometry/native metrics']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--old-root',type=Path)
    parser.add_argument('--predictions',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not args.root or not args.old_root:parser.error('--root and --old-root required')
        score(args)
