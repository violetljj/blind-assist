"""Frozen AGD-trained selectors on IIT categorical masks; seal before GT parse."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import io
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import binomtest
from PIL import Image, ImageDraw
import score_visual_fresh as visual

old = visual.old
dump, lines, load, file_record = visual.dump, visual.lines, visual.load, visual.file_record
MAIN = 'candidates_dino_point_mse'
CONTROLS = ['candidates_dino_gap_mse','candidates_scores_mse','candidates_dino_point_ridge',
            'candidates_dino_point_hgb','fixed_dino_point_hgb','C0','B0']
TRAIN_SHA = '62565df59aa967590cfc9d99c1b58bb12061809a6da2339b2e188eab8a353e43'


def verify_records(records):
    for item in records.values():
        if old.sha256(item['path']) != item['sha256']:
            raise ValueError('Sealed file changed: '+item['path'])


def prepare(args):
    start, cpu = time.monotonic(), time.process_time()
    out = args.root/'evaluation'; out.mkdir(parents=True,exist_ok=True)
    if (out/'sealed_decisions.json').exists(): raise FileExistsError('Preserve seal')
    audit_path=args.root/'data/independent_data_audit.json';data_audit=load(audit_path)
    data_summary=load(args.root/'data/data_summary.json')
    if data_audit['status']!='PASS' or data_audit['GT_content_parsed'] is not False:raise ValueError('Data audit not ready')
    if old.sha256(args.manifest)!=data_summary['samples_sha256']:raise ValueError('Data manifest SHA')
    source = load(args.source_root/'evaluation/sealed_decisions.json')
    if source['calibration_images'] != 79: raise ValueError('Expected AGD79 calibration')
    # Check original gate, calibration-decision and training lineage, not fresh scores.
    verify_records({k: source['files'][k] for k in ('train_seal','gates','cal_decisions','calibration_predictions')})
    train_path = Path(source['files']['train_seal']['path'])
    if old.sha256(train_path) != TRAIN_SHA: raise ValueError('Frozen training SHA')
    training = load(train_path)
    for name, sha in training['files'].items():
        if old.sha256(train_path.parent/name) != sha: raise ValueError('Training file changed')
    gates_path = Path(source['files']['gates']['path']); gates = load(gates_path)
    if len(gates) != 68: raise ValueError('Expected 68 inherited gates')
    learn_gates = load(train_path.parent/'sealed_gates.json')['settings']
    for family, settings in learn_gates.items():
        for method, gate in settings.items():
            if gates[family+'_'+method] != gate: raise ValueError('Training gate lineage')
    pred_seal_path = args.fresh_predictions.parent/'prediction_seal.json'
    pred_seal = load(pred_seal_path)
    if pred_seal['status'] != 'SUCCESS' or pred_seal['fresh_GT_read'] is not False: raise ValueError('Prediction seal')
    for key, path in [('prediction_sha256',args.fresh_predictions),('observation_predictions_sha256',args.fresh_inference),('training_seal_sha256',train_path)]:
        if pred_seal[key] != old.sha256(path): raise ValueError('Prediction lineage '+key)
    observations = old.read_jsonl(args.fresh_inference); lookup = {r['sample_id']:r for r in observations}
    samples = old.read_jsonl(args.manifest); manifest = {r['sample_id']:r for r in samples}
    if not 0 < len(lookup) <= 150 or len(lookup) != len(observations) or set(lookup)!=set(manifest): raise ValueError('Image identity')
    learned = old.read_jsonl(args.fresh_predictions)
    if len(learned)!=2*len(lookup) or {(r['sample_id'],r['family']) for r in learned}!={(s,f) for s in lookup for f in ('candidates','fixed')}: raise ValueError('Family coverage')
    for row in learned:
        obs=lookup[row['sample_id']]; meta=manifest[row['sample_id']]
        if row['action'] != 'hold' or obs['action'] != 'hold' or meta['action'] != 'hold': raise ValueError('Action changed')
        if row['image_sha256'] != obs['image_sha256'] or row['image_sha256'] != meta['image_sha256']: raise ValueError('RGB SHA')
        visual.check_points(obs,row)
    decisions=visual.learners(learned)
    for obs in observations:
        refs=visual.references(obs); visual.independent_reference_check(obs,refs); decisions.extend(refs)
    for decision in decisions:
        point=np.asarray(decision['point_xy']);obs=lookup[decision['sample_id']]
        if not np.array_equal(point-np.floor(point),np.array([.5,.5])): raise ValueError('Non-half-pixel output; report before changing protocol')
        if tuple(np.floor(point).astype(int)) != old.point_pixel(point,(obs['height'],obs['width'])): raise ValueError('Categorical coordinate conventions differ')
    if Counter(r['method'] for r in decisions)!=Counter({m:len(lookup) for m in gates}): raise ValueError('Method coverage')
    shutil.copyfile(gates_path,out/'sealed_gates.json')
    lines(out/'fresh_decisions.jsonl',decisions)
    files={k:file_record(p) for k,p in dict(training_seal=train_path,source_seal=args.source_root/'evaluation/sealed_decisions.json',
        inherited_gates=gates_path,gates=out/'sealed_gates.json',fresh_predictions=args.fresh_predictions,
        prediction_seal=pred_seal_path,fresh_inference=args.fresh_inference,manifest=args.manifest,data_audit=audit_path,
        fresh_decisions=out/'fresh_decisions.jsonl',script=Path(__file__),visual_dependency=Path(visual.__file__),
        metric_dependency=Path(old.__file__),reference_dependency=Path(visual.multi.__file__),
        plan=Path(__file__).with_name('CROSS_SOURCE_PLAN.md')).items()}
    dump(out/'sealed_decisions.json',dict(status='SEALED_BEFORE_FRESH_GT_OPEN',files=files,
        fresh_images=len(lookup),methods=sorted(gates),calibration_images=79,GT_parsed=False,
        cpu_s=time.process_time()-cpu,wall_s=time.monotonic()-start))
    print(json.dumps(dict(stage='prepared',images=len(lookup),methods=len(gates))))


def read_native(meta):
    if old.sha256(meta['gt_path']) != meta['gt_sha256']: raise ValueError('Native file SHA mismatch')
    raw=Path(meta['gt_path']).read_text(encoding='utf-8-sig')
    return parse_native(raw,(meta['height'],meta['width']))


def parse_native(raw,shape):
    arr=np.loadtxt(io.StringIO(raw),dtype=np.float64,ndmin=2)
    if arr.shape!=shape: raise ValueError('Native dimensions')
    if not np.isfinite(arr).all() or not np.equal(arr,np.floor(arr)).all() or arr.min()<0 or arr.max()>9:
        raise ValueError('Native labels must be integer 0..9')
    return arr.astype(np.uint8)


def measure(gt,point,dist):
    x,y=map(float,point);h,w=gt.shape
    if not (math.isfinite(x) and math.isfinite(y) and 0<=x<w and 0<=y<h): raise ValueError('Point outside image')
    ix,iy=math.floor(x),math.floor(y);label=int(gt[iy,ix])
    return dict(hit=label in (5,9),native_label=label,pixel_xy=[ix,iy],
        mask_distance_diag=float(dist[iy,ix]/math.hypot(w,h)) if dist is not None else None)


def aggregate(rows,total=None):
    valid=[r for r in rows if r['status']=='EVALUABLE'];hits=sum(r['hit'] for r in valid)
    nonempty=[r for r in valid if not r['native_empty']]; distances=[r['mask_distance_diag'] for r in nonempty]
    return dict(images=len(rows),evaluable=len(valid),not_evaluable=len(rows)-len(valid),hits=hits,
        hit_fraction=hits/len(rows) if rows and len(valid)==len(rows) else None,
        evaluable_hit_fraction=hits/len(valid) if valid else None,nonempty=len(nonempty),
        nonempty_hit_fraction=hits/len(nonempty) if nonempty else None,
        empty=len(valid)-len(nonempty),grasp5_hits=sum(r['native_label']==5 for r in valid),
        wrap9_hits=sum(r['native_label']==9 for r in valid),
        wrong_native_labels=dict(Counter(str(r['native_label']) for r in valid if not r['hit'])),
        mask_distance_diag_mean=float(np.mean(distances)) if distances else None,
        selected_type_counts=dict(Counter('refined' if r['selected_index']%2 else 'seed' for r in rows if r['selected_index'] is not None)))


def compare(before,after):
    a={r['sample_id']:r for r in before};b={r['sample_id']:r for r in after}
    if set(a)!=set(b):raise ValueError('Unpaired images')
    ids=[s for s in sorted(a) if a[s]['status']==b[s]['status']=='EVALUABLE']
    delta=np.array([int(b[s]['hit'])-int(a[s]['hit']) for s in ids]);n=len(delta)
    if not n:return dict(n=0,total=len(a),not_evaluable=len(a))
    boot=delta[np.random.default_rng(20261015).integers(0,n,size=(2000,n))].mean(1)
    saved=int((delta>0).sum());harmed=int((delta<0).sum())
    return dict(n=n,total=len(a),not_evaluable=len(a)-n,mean_delta=float(delta.mean()),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist(),saved=saved,harmed=harmed,tied=int((delta==0).sum()),
        discordant_exact_p=float(binomtest(saved,saved+harmed,.5).pvalue) if saved+harmed else 1.)


def cases(out,manifest,grouped,candidates,masks):
    main={r['sample_id']:r for r in grouped[MAIN] if r['status']=='EVALUABLE'}
    gap={r['sample_id']:r for r in grouped['candidates_dino_gap_mse']}
    pools=[('point gain',[s for s in sorted(main) if main[s]['hit'] and not gap[s]['hit']]),
           ('point loss',[s for s in sorted(main) if gap[s]['hit'] and not main[s]['hit']]),
           ('oracle gap',[s for s in sorted(main) if candidates[s,'candidates']['oracle_hit'] and not main[s]['hit']])]
    chosen=[]
    for label,pool in pools:
        for sid in [s for s in pool if s not in {r[0] for r in chosen}][:2]:chosen.append((sid,label))
    if not chosen:return
    panels=[];records=[]
    for sid,label in chosen:
        meta=manifest[sid];im=Image.open(meta['image_path']).convert('RGB');rgb=np.array(im);gt=masks[sid]
        overlay=rgb.copy()
        for lab,col in [(5,[40,230,80]),(9,[220,50,220])]:overlay[gt==lab]=(.4*rgb[gt==lab]+.6*np.array(col)).astype(np.uint8)
        scale=min(560/im.width,285/im.height);size=(round(im.width*scale),round(im.height*scale))
        line=Image.new('RGB',(1160,340),'white')
        for index,panel in enumerate([im,Image.fromarray(overlay)]):
            ox=index*580;line.paste(panel.resize(size),(ox+10,35));draw=ImageDraw.Draw(line)
            draw.text((ox+10,8),f'{label}: {sid[:42]}',fill='black')
            for method,color in [(MAIN,'red'),('candidates_dino_gap_mse','blue')]:
                item=main[sid] if method==MAIN else gap[sid];x,y=item['point_xy'];x=ox+10+x*scale;y=35+y*scale
                draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3)
            draw.text((ox+10,320),f'red point={int(main[sid]["hit"])}; blue GAP={int(gap[sid]["hit"])}; green5 magenta9',fill='black')
        panels.append(line);records.append(dict(sample_id=sid,selection=label,point=main[sid],gap=gap[sid],candidates=candidates[sid,'candidates']))
    montage=Image.new('RGB',(1160,len(panels)*340),'white')
    for i,panel in enumerate(panels):montage.paste(panel,(0,i*340))
    montage.save(out/'casebook.jpg',quality=92);dump(out/'cases.json',records)


def score(args):
    start,cpu=time.monotonic(),time.process_time();out=args.root/'evaluation';seal=load(out/'sealed_decisions.json')
    if (out/'summary.json').exists():raise FileExistsError('Preserve completed result')
    if seal['status']!='SEALED_BEFORE_FRESH_GT_OPEN':raise ValueError('Unsealed')
    verify_records(seal['files'])
    gates=load(out/'sealed_gates.json');decisions=old.read_jsonl(out/'fresh_decisions.jsonl')
    manifest={r['sample_id']:r for r in old.read_jsonl(seal['files']['manifest']['path'])}
    byid=defaultdict(list);learned=defaultdict(list)
    for r in decisions:byid[r['sample_id']].append(r)
    for r in old.read_jsonl(seal['files']['fresh_predictions']['path']):learned[r['sample_id']].append(r)
    scored=[];candidate_records=[];invalid=[];masks={};point_checks=0;distance_checks=0
    for sid in sorted(manifest):
        if time.process_time()-cpu+seal['cpu_s']>600:raise TimeoutError('Scoring CPU budget')
        meta=manifest[sid]
        try:gt=read_native(meta)
        except Exception as exc:
            native_path=Path(meta['gt_path'])
            invalid.append(dict(sample_id=sid,status='NOT_EVALUABLE',reason=str(exc),source=file_record(native_path) if native_path.is_file() else dict(path=str(native_path),sha256=None)))
            for d in byid[sid]:scored.append(dict(**d,status='NOT_EVALUABLE',reason=str(exc),hit=None,native_empty=None,native_label=None,mask_distance_diag=None,accepted=None))
            continue
        masks[sid]=gt;target=np.isin(gt,[5,9]);empty=not bool(target.any());dist=distance_transform_edt(~target) if not empty else None
        yy,xx=np.where(target);tree=cKDTree(np.column_stack((xx,yy))) if not empty else None
        for d in byid[sid]:
            metrics=measure(gt,d['point_xy'],dist);px,py=metrics['pixel_xy']
            # Independent flattened categorical lookup, no imported sampling helper.
            independent=int(gt.ravel()[py*gt.shape[1]+px]);assert metrics['native_label']==independent and metrics['hit']==(independent==5 or independent==9);point_checks+=1
            if tree is not None:
                assert abs(metrics['mask_distance_diag']-float(tree.query([px,py])[0]/math.hypot(*gt.shape)))<1e-12;distance_checks+=1
            c=d['confidence'];threshold=gates[d['method']]['threshold']
            scored.append(dict(**d,**metrics,status='EVALUABLE',native_empty=empty,
                accepted=c is not None and math.isfinite(c) and threshold is not None and c>=threshold,
                object_class=meta.get('object',meta.get('selection_object','unknown')),source_image_id=meta.get('source_image_id')))
        for row in learned[sid]:
            metrics=[measure(gt,p,dist) for p in row['points_xy']]
            candidate_records.append(dict(sample_id=sid,family=row['family'],points_xy=row['points_xy'],
                hits=[m['hit'] for m in metrics],native_labels=[m['native_label'] for m in metrics],oracle_hit=any(m['hit'] for m in metrics),native_empty=empty))
    grouped=defaultdict(list)
    for r in scored:grouped[r['method']].append(r)
    summaries={m:aggregate(rows) for m,rows in grouped.items()};refusal={};classes={}
    for name,rows in grouped.items():
        valid=[r for r in rows if r['status']=='EVALUABLE'];kept=[r for r in valid if r['accepted']]
        empty=[r for r in valid if r['native_empty']];wrong=sum(not r['hit'] for r in kept)
        ranked=sorted(valid,key=lambda r:(r['confidence'] is None,-r['confidence'] if r['confidence'] is not None else 0,r['sample_id']))
        curve=[dict(retained=k,coverage=k/len(rows),**aggregate(ranked[:k])) for k in range(1,len(ranked)+1)]
        refusal[name]=dict(threshold=gates[name]['threshold'],retained=len(kept),coverage=len(kept)/len(rows),accepted_correct=len(kept)-wrong,
            accepted_wrong=wrong,accepted_error_fraction=wrong/len(kept) if kept else None,accepted_wrong_fraction_all=wrong/len(rows),
            native_empty=len(empty),native_empty_accepted=sum(r['accepted'] for r in empty),not_evaluable=len(rows)-len(valid),
            AURC=float(np.mean([1-r['evaluable_hit_fraction'] for r in curve])) if curve else None,
            matched_coverage=[curve[math.ceil(f*len(valid))-1] for f in (.5,.7,.8,.9,1.)] if curve else [],curve=curve)
        classes[name]={c:aggregate([r for r in valid if r['object_class']==c]) for c in sorted({r['object_class'] for r in valid})}
    paired={name:compare(grouped[name],grouped[MAIN]) for name in CONTROLS}
    seed={str(s):{ref:compare(grouped[f'candidates_{ref}_mse_seed{s}'],grouped[f'candidates_dino_point_mse_seed{s}']) for ref in ('dino_gap','scores')} for s in (20261015,20261016,20261017)}
    candidates={(r['sample_id'],r['family']):r for r in candidate_records};oracles={}
    for family in ('candidates','fixed'):
        ids={r['sample_id'] for r in candidate_records if r['family']==family and r['oracle_hit']}
        oracles[family]=dict(images=len(manifest),evaluable=len(manifest)-len(invalid),oracle_hits=len(ids),
            selected={name:dict(hits=sum(r['hit'] for r in rows if r['sample_id'] in ids),oracle_positive=len(ids),missed=sum(not r['hit'] for r in rows if r['sample_id'] in ids)) for name,rows in grouped.items() if name.startswith(family+'_')})
    lines(out/'per_image_metrics.jsonl',scored);lines(out/'per_candidate_quality.jsonl',candidate_records);dump(out/'invalid_inputs.json',invalid)
    cases(out,manifest,grouped,candidates,masks)
    result=dict(status='COMPLETE' if not invalid else 'COMPLETE_WITH_NOT_EVALUABLE',images=len(manifest),methods=summaries,
        paired_dino_point_minus_reference=paired,seed_comparisons=seed,refusal=refusal,oracles=oracles,class_diagnostics=classes,
        invalid=invalid,independent_checks=dict(categorical_point_checks=point_checks,edt_vs_kdtree_distance_checks=distance_checks),
        seal_sha256=old.sha256(out/'sealed_decisions.json'),cpu_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    dump(out/'summary.json',result);print(json.dumps(dict(status=result['status'],images=result['images'],paired=paired,cpu_s=result['cpu_s'])))


def selftest():
    assert np.array_equal(parse_native('0 5 9\n1 2 3',(2,3)),np.array([[0,5,9],[1,2,3]]))
    for bad in ('0 5.1 9\n1 2 3','0 10 9\n1 2 3','0 nan 9\n1 2 3','0 5\n1 2'):
        try:parse_native(bad,(2,3))
        except ValueError:pass
        else:raise AssertionError('Invalid native label accepted')
    gt=np.array([[0,5,9],[1,2,3]],dtype=np.uint8);dist=distance_transform_edt(~np.isin(gt,[5,9]))
    assert measure(gt,[1.0,.5],dist)['hit'] and measure(gt,[2.9,.5],dist)['native_label']==9
    assert measure(gt,[.5,.5],dist)['mask_distance_diag']==1/math.sqrt(13)
    assert measure(np.zeros((2,3),np.uint8),[.5,.5],None)['mask_distance_diag'] is None
    a=[dict(sample_id=str(i),hit=h,status='EVALUABLE') for i,h in enumerate([0,1,0])]
    b=[dict(sample_id=str(i),hit=h,status='EVALUABLE') for i,h in enumerate([1,0,0])]
    c=compare(a,b);assert c['saved']==c['harmed']==1 and c['mean_delta']==0 and c['discordant_exact_p']==1
    print('SELFTEST_PASS: categorical floor, empty null, EDT distance, paired bootstrap and exact discordance')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--source-root',type=Path)
    for name in ('fresh-predictions','fresh-inference','manifest'):p.add_argument('--'+name,type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--score',action='store_true');p.add_argument('--selftest',action='store_true')
    args=p.parse_args()
    if args.selftest:selftest()
    elif args.prepare:prepare(args)
    elif args.score:score(args)
    else:p.error('Choose --prepare, --score or --selftest')


if __name__=='__main__':main()
