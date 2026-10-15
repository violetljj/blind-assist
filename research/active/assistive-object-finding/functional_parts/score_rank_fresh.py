"""Seal observation-only decisions, then score fresh native heatmaps once."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import cv2
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import evaluate as old
import evaluate_multicandidate as multi


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_record(path):
    return dict(path=str(Path(path).resolve()), sha256=old.sha256(path))


def references(obs):
    maps, points, confidence, *_ = multi.predict_methods(obs)
    w, h = obs['width'], obs['height']
    flip = old.probability(old.load_map(obs['views']['flip']['heads']['ree']), (w,h))[0]
    two = old.normalize_mass(.5*maps['B0']+.5*flip)
    points['Bflip2'] = old.argmax_point(two)
    confidence['Bflip2'] = old.spatial_confidence(two)
    return [dict(sample_id=obs['sample_id'], method=k, point_xy=points[k], confidence=confidence[k],
                 family=None, selected_index=None) for k in ('B0','Bflip2','B1','B2','B3','C0','C1','C4')]


def independent_reference_check(obs, decisions):
    """Rebuild two baseline points without imported probability/argmax helpers."""
    maps=[]
    for name in ('base','flip'):
        raw=np.load(obs['views'][name]['heads']['ree'],allow_pickle=False).astype(np.float32)
        resized=cv2.resize(raw,(obs['width'],obs['height']),interpolation=cv2.INTER_LINEAR)
        value=((resized-resized.min())/(resized.max()-resized.min()+1e-10)).astype(np.float64)
        maps.append(value/value.sum() if value.sum()>0 else value)
    result={r['method']:r['point_xy'] for r in decisions}
    for name,value in [('B0',maps[0]),('Bflip2',.5*maps[0]+.5*maps[1])]:
        y,x=np.unravel_index(np.argmax(value),value.shape)
        if not np.allclose(result[name],[x+.5,y+.5],atol=0,rtol=0):raise ValueError('Independent reference point mismatch')


def learners(rows):
    output=[]
    for row in rows:
        points=np.asarray(row['points_xy'], dtype=float)
        if points.shape!=(6,2) or not np.isfinite(points).all():
            raise ValueError('Six finite candidate points required')
        vectors=dict(row['scores'])
        for kind, seed_vectors in row.get('seed_scores',{}).items():
            for index, vector in enumerate(seed_vectors): vectors[f'{kind}_seed{20261015+index}']=vector
        for kind, vector in vectors.items():
            vector=np.asarray(vector,dtype=float)
            if vector.shape!=(6,) or not np.isfinite(vector).all():raise ValueError('Invalid learner scores')
            index=int(np.argmax(vector))
            output.append(dict(sample_id=row['sample_id'],method=f"{row['family']}_{kind}",
                point_xy=points[index].tolist(),confidence=float(vector[index]),
                family=row['family'],selected_index=index))
    return output


def prepare(args):
    started=time.monotonic();cpu_started=time.process_time()
    root=args.root; out=root/'evaluation'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'sealed_decisions.json').exists():raise FileExistsError('Preserve existing seal')
    if not args.train_seal.is_file():raise FileNotFoundError('Training seal required before decisions seal')
    training_seal=load(args.train_seal)
    if training_seal['status']!='SUCCESS' or training_seal['fresh_GT_read'] is not False:raise ValueError('Training seal state')
    for name,expected in training_seal['files'].items():
        if old.sha256(args.train_seal.parent/name)!=expected:raise ValueError('Training payload changed')
    prediction_seal_path=args.fresh_predictions.parent/'prediction_seal.json'
    prediction_seal=load(prediction_seal_path)
    if prediction_seal['status']!='SUCCESS' or prediction_seal['fresh_GT_read'] is not False:raise ValueError('Prediction seal state')
    for key,path in [('prediction_sha256',args.fresh_predictions),('training_seal_sha256',args.train_seal),('observation_predictions_sha256',args.fresh_inference)]:
        if prediction_seal[key]!=old.sha256(path):raise ValueError('Prediction seal mismatch')
    cal_ids={r['sample_id'] for r in old.read_jsonl(root/'data/calibration_observations.jsonl')}
    inherited=old.read_jsonl(args.old_predictions)
    cal_obs=[r for r in inherited if r['sample_id'] in cal_ids]
    if len(cal_obs)!=len(cal_ids):raise ValueError('Calibration observation identity mismatch')
    fresh_obs=old.read_jsonl(args.fresh_inference)
    cal_learn=old.read_jsonl(args.cal_predictions);fresh_learn=old.read_jsonl(args.fresh_predictions)
    fresh_ids={r['sample_id'] for r in fresh_obs}
    if fresh_ids & cal_ids:raise ValueError('Fresh/calibration overlap')
    for rows,ids in ((cal_learn,cal_ids),(fresh_learn,fresh_ids)):
        if len(rows)!=2*len(ids) or {(r['sample_id'],r['family']) for r in rows}!={(s,f) for s in ids for f in ('candidates','fixed')}:
            raise ValueError('Expected two unique family rows per image')
    cal=learners(cal_learn);fresh=learners(fresh_learn)
    for obs in cal_obs:
        predicted=references(obs);independent_reference_check(obs,predicted);cal.extend(predicted)
    for obs in fresh_obs:
        if time.process_time()-cpu_started>600:raise TimeoutError('Evaluation stage CPU budget')
        predicted=references(obs);independent_reference_check(obs,predicted);fresh.extend(predicted)
    grouped=defaultdict(list)
    for row in cal:grouped[row['method']].append(row['confidence'])
    gates={method:old.calibrate_coverage(scores,len(cal_ids)) for method,scores in grouped.items()}
    training_gates=load(args.train_seal.parent/'sealed_gates.json')['settings']
    for family,settings in training_gates.items():
        for method,expected in settings.items():
            if gates[f'{family}_{method}']!=expected:raise ValueError('Independent gate recalculation mismatch')
    lines(out/'calibration_decisions.jsonl',cal);lines(out/'fresh_decisions.jsonl',fresh)
    dump(out/'sealed_gates.json',gates)
    files={k:file_record(p) for k,p in dict(train_seal=args.train_seal,
        calibration_predictions=args.cal_predictions,fresh_predictions=args.fresh_predictions,
        fresh_inference=args.fresh_inference,old_predictions=args.old_predictions,
        gates=out/'sealed_gates.json',fresh_decisions=out/'fresh_decisions.jsonl',
        cal_decisions=out/'calibration_decisions.jsonl',script=Path(__file__),
        prediction_seal=prediction_seal_path,metric_dependency=Path(old.__file__),reference_dependency=Path(multi.__file__),
        plan=Path(__file__).with_name('LEARNED_RANK_PLAN.md')).items()}
    dump(out/'sealed_decisions.json',dict(status='SEALED_BEFORE_FRESH_GT_OPEN',files=files,
        calibration_images=len(cal_ids),fresh_images=len(fresh_ids),methods=sorted(gates),
        independent_base_and_flip_point_checks=2*(len(cal_ids)+len(fresh_ids)),
        wall_s=time.monotonic()-started,cpu_s=time.process_time()-cpu_started))
    print(json.dumps(dict(stage='prepared',fresh_images=len(fresh_ids),methods=len(gates),wall_s=time.monotonic()-started)))


def aggregate(rows):
    q=np.asarray([r['quality'] for r in rows],dtype=float)
    return dict(images=len(rows),quality_mean=float(q.mean()) if len(q) else None,
        quality_sum=float(q.sum()),hit_proxy_ge_05=int((q>=.5).sum()),
        local_GT_mass_mean=float(np.mean([r['local_mass'] for r in rows])) if rows else None,
        peak_distance_diag_mean=float(np.mean([r['peak_distance'] for r in rows])) if rows else None,
        selected_type_counts=dict(Counter('refined' if r['selected_index']%2 else 'seed'
            for r in rows if r['selected_index'] is not None)))


def compare(before,after):
    before={r['sample_id']:r for r in before};after={r['sample_id']:r for r in after}
    if set(before)!=set(after):raise ValueError('Unpaired')
    keys=sorted(before);delta=np.array([after[k]['quality']-before[k]['quality'] for k in keys])
    rng=np.random.default_rng(20261015)
    boot=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
    return dict(n=len(delta),mean_delta=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        improved=int((delta>1e-12).sum()),worsened=int((delta< -1e-12).sum()),tied=int((np.abs(delta)<=1e-12).sum()),
        proxy_hit_saved=sum(before[k]['quality']<.5<=after[k]['quality'] for k in keys),
        proxy_hit_harmed=sum(after[k]['quality']<.5<=before[k]['quality'] for k in keys))


def independent_point(gt,point):
    h,w=gt.shape;x=float(np.clip(point[0]-.5,0,w-1));y=float(np.clip(point[1]-.5,0,h-1))
    xa,ya=int(x),int(y);xb,yb=min(xa+1,w-1),min(ya+1,h-1);dx,dy=x-xa,y-ya
    return float(((1-dy)*((1-dx)*gt[ya,xa]+dx*gt[ya,xb])+dy*((1-dx)*gt[yb,xa]+dx*gt[yb,xb]))/gt.max())


def make_cases(out, manifest, grouped, candidates):
    rank={r['sample_id']:r for r in grouped['candidates_rank']}
    mse={r['sample_id']:r for r in grouped['candidates_mse']}
    gains=sorted(rank,key=lambda s:rank[s]['quality']-mse[s]['quality'],reverse=True)
    losses=list(reversed(gains));gaps=sorted(rank,key=lambda s:max(candidates[(s,'candidates')]['qualities'])-rank[s]['quality'],reverse=True)
    selected=[]
    for label,pool in (('rank gain',gains),('rank loss',losses),('oracle gap',gaps)):
        for sid in [s for s in pool if s not in {a[0] for a in selected}][:2]:selected.append((sid,label))
    rows=[];case_records=[]
    for sid,label in selected:
        meta=manifest[sid];im=Image.open(meta['image_path']).convert('RGB');gt=cv2.imread(meta['gt_path'],0)
        w,h=im.size;scale=min(560/w,300/h);size=(round(w*scale),round(h*scale))
        rgb=np.asarray(im);color=cv2.cvtColor(cv2.applyColorMap(gt,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        overlay=Image.fromarray((rgb*.5+color*.5).astype(np.uint8))
        panels=[]
        for panel in (im,overlay):
            canvas=Image.new('RGB',(580,350),'white');canvas.paste(panel.resize(size),(10,40));draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f'{label}: {meta["action"]}  {sid[:36]}',fill='black')
            for name,c in (('candidates_mse','blue'),('candidates_rank','red')):
                item=next(r for r in grouped[name] if r['sample_id']==sid);x,y=item['point_xy'];x=10+x*scale;y=40+y*scale
                draw.ellipse((x-5,y-5,x+5,y+5),outline=c,width=3)
            draw.text((10,330),f'blue=MSE {mse[sid]["quality"]:.3f}; red=rank {rank[sid]["quality"]:.3f}',fill='black')
            panels.append(canvas)
        line=Image.new('RGB',(1160,350),'white');line.paste(panels[0]);line.paste(panels[1],(580,0));rows.append(line)
        case_records.append(dict(sample_id=sid,selection=label,mse=mse[sid],rank=rank[sid],candidates=candidates[(sid,'candidates')]))
    montage=Image.new('RGB',(1160,350*len(rows)),'white')
    for i,line in enumerate(rows):montage.paste(line,(0,350*i))
    montage.save(out/'casebook.jpg',quality=92);dump(out/'cases.json',case_records)


def score(args):
    started=time.monotonic();cpu_started=time.process_time();root=args.root;out=root/'evaluation';seal=load(out/'sealed_decisions.json')
    if (out/'summary.json').exists():raise FileExistsError('Fresh scoring already completed')
    if seal['status']!='SEALED_BEFORE_FRESH_GT_OPEN':raise ValueError('Unsealed')
    for item in seal['files'].values():
        if old.sha256(item['path'])!=item['sha256']:raise ValueError('Sealed file changed: '+item['path'])
    # All predictions and calibration thresholds verified before opening any fresh GT.
    decisions=old.read_jsonl(out/'fresh_decisions.jsonl');gates=load(out/'sealed_gates.json')
    samples=old.read_jsonl(args.manifest);manifest={r['sample_id']:r for r in samples}
    if len(manifest)!=seal['fresh_images'] or set(manifest)!={r['sample_id'] for r in decisions}:raise ValueError('Manifest mismatch')
    byid=defaultdict(list)
    for row in decisions:byid[row['sample_id']].append(row)
    raw_learn=old.read_jsonl(seal['files']['fresh_predictions']['path']);point_rows=defaultdict(list)
    for row in raw_learn:
        meta=manifest[row['sample_id']]
        if row['image_sha256']!=meta['image_sha256'] or row['action']!=meta['action']:raise ValueError('Fresh identity mismatch')
        point_rows[row['sample_id']].append(row)
    scored=[];candidate_records=[];independent_errors=[]
    for sid in sorted(manifest):
        if time.process_time()-cpu_started+seal['cpu_s']>600:raise TimeoutError('Evaluation total CPU budget')
        meta=manifest[sid];gt=cv2.imread(meta['gt_path'],cv2.IMREAD_GRAYSCALE)
        if gt is None or old.sha256(meta['gt_path'])!=meta['gt_sha256'] or gt.shape!=(meta['height'],meta['width']):raise ValueError('GT integrity')
        if gt.max()<=0:raise ValueError('Undefined native heatmap; preserve failure')
        yy,xx=np.where(gt==gt.max());tree=cKDTree(np.column_stack((xx+.5,yy+.5)))
        for decision in byid[sid]:
            metrics=old.agd_point_metrics(gt.astype(float),decision['point_xy'],tree)
            q=metrics['GT_relative_peak_proxy'];independent_errors.append(abs(q-independent_point(gt,decision['point_xy'])))
            threshold=gates[decision['method']]['threshold'];c=decision['confidence']
            scored.append(dict(**decision,quality=q,local_mass=metrics['local_GT_soft_mass_proxy'],
                peak_distance=metrics['GT_peak_distance_diag'],accepted=c is not None and math.isfinite(c) and threshold is not None and c>=threshold))
        for row in point_rows[sid]:
            qs=[independent_point(gt,p) for p in row['points_xy']]
            candidate_records.append(dict(sample_id=sid,family=row['family'],points_xy=row['points_xy'],qualities=qs,oracle_quality=max(qs)))
    grouped=defaultdict(list)
    for row in scored:grouped[row['method']].append(row)
    summary={name:aggregate(rows) for name,rows in grouped.items()};refusal={}
    for name,rows in grouped.items():
        kept=[r for r in rows if r['accepted']]
        ranked=sorted(rows,key=lambda r:(r['confidence'] is None,-r['confidence'] if r['confidence'] is not None else 0,r['sample_id']))
        curve=[dict(retained=k,coverage=k/len(rows),**aggregate(ranked[:k])) for k in range(1,len(rows)+1)]
        refusal[name]=dict(threshold=gates[name]['threshold'],calibration=gates[name],actual_coverage=len(kept)/len(rows),
            retained=aggregate(kept),AURC=float(np.mean([1-r['quality_mean'] for r in curve])),
            matched_coverage=[curve[math.ceil(f*len(rows))-1] for f in (.5,.7,.8,.9,1.)],curve=curve)
    comparison_names=['candidates_mse','candidates_hgb','candidates_ridge','fixed_rank','fixed_hgb','B1','Bflip2','C0']
    paired={name:compare(grouped[name],grouped['candidates_rank']) for name in comparison_names}
    cand={(r['sample_id'],r['family']):r for r in candidate_records};oracles={}
    for family in ('candidates','fixed'):
        oq=np.array([r['oracle_quality'] for r in candidate_records if r['family']==family])
        oracles[family]=dict(images=len(oq),quality_mean=float(oq.mean()),hit_proxy_ge_05=int((oq>=.5).sum()),
            gap={name:float(oq.mean()-s['quality_mean']) for name,s in summary.items() if name.startswith(family+'_')})
    lines(out/'per_image_metrics.jsonl',scored);lines(out/'per_candidate_quality.jsonl',candidate_records)
    make_cases(out,manifest,grouped,cand)
    independent=dict(point_samples=len(independent_errors),max_bilinear_quality_error=max(independent_errors),
        method_count=len(grouped),unique_images=len(manifest),seal_verified=True)
    if independent['max_bilinear_quality_error']>1e-10:raise ValueError('Independent quality mismatch')
    result=dict(status='COMPLETE',methods=summary,paired_rank_minus_reference=paired,refusal=refusal,
        oracles=oracles,independent_checks=independent,wall_s=time.monotonic()-started,cpu_s=time.process_time()-cpu_started,
        seal_sha256=old.sha256(out/'sealed_decisions.json'),manifest=file_record(args.manifest))
    dump(out/'summary.json',result)
    print(json.dumps(dict(status='COMPLETE',images=len(manifest),methods=len(grouped),wall_s=result['wall_s'],paired=paired)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--score',action='store_true')
    for name in ('old-predictions','fresh-inference','cal-predictions','fresh-predictions','train-seal','manifest'):p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.score:score(args)
    else:p.error('Choose prepare or score')


if __name__=='__main__':main()
