"""Validate captured quartets without models; preserve exclusions and background changes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from audit_city_target_uncertainty import read,sha
from city_crossregion import CONDITIONS


def verify(capture,labels,output):
    capture,labels,output=map(lambda p:Path(p).resolve(),(capture,labels,output))
    if output.exists() or not output.is_relative_to((Path(__file__).resolve().parents[1]/'artifacts.local').resolve()):
        raise ValueError('Fresh canonical output required')
    spec=read(capture/'source/spec.json');receipt=read(capture/'receipt.json');validation=read(labels/'label-validation.json')
    if receipt['status']!='PASS' or validation['status']!='PASS' or not receipt['source_unchanged']:raise ValueError('Completed inputs required')
    for path,key in [(capture/'source/spec.json','spec_sha256'),(capture/'receipt.json','receipt_sha256'),
                     (capture/'evaluator/target-raycheck.json','raycheck_sha256'),(labels/'near.npy','near_sha256')]:
        if sha(path)!=validation[key]:raise ValueError('Evidence identity mismatch '+key)
    near=np.load(labels/'near.npy');groups=defaultdict(list)
    poses={r['sample_index']:r for r in receipt['capture_poses']}
    ready={r['index']:r['status'] for r in receipt['view_readiness']}
    target_rows={(t,r['sample_index']):r for t,rs in validation['targets'].items() for r in rs}
    frame_labels={r['sample_index']:r for r in validation['rows']}
    payload=read(capture/'payload-hashes.json')
    for i,case in enumerate(spec['cases']):groups[case['pair_id']].append((i,case))
    output.mkdir(parents=True);results=[];metric=[]
    for gid,members in groups.items():
        reasons=[];contexts=[];images={};excluded_geometry=[];covered=np.zeros((360,640),bool)
        if sorted(c['condition'] for _,c in members)!=sorted(CONDITIONS):reasons.append('INCOMPLETE_QUARTET')
        for i,c in members:
            context={k:v for k,v in c.items() if k not in ('objects','name','condition','variant','expected_near','active_target_ids','absent_target_ids')}
            context['common_objects']=[o for o in c['objects'] if not o.get('target_part')]
            context['actual_pose']=poses[i]
            context['actual_pose']={k:v for k,v in poses[i].items() if k!='sample_index'}
            context['map_sha256']=receipt['map_sha256_after'];context['lights']=receipt['base_light_intensities']
            contexts.append(hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest())
            if poses[i]['rgb']!=poses[i]['depth'] or ready[i]!='READY':reasons.append('POSE_OR_READINESS')
            if near[i].tolist()!=c['expected_near']:reasons.append('NATIVE_TRUTH_DISAGREES')
            if frame_labels[i]['floor_z_m'] is None:reasons.append('MISSING_FLOOR')
            for tid in c['active_target_ids']:
                evidence=target_rows[(tid,i)]
                if evidence['status']!='EVALUABLE':excluded_geometry.append(dict(sample_index=i,target_id=tid))
                p=capture/f'evaluator/isolated/{tid}/{i:04d}.npy'
                if sha(p)!=evidence['isolated_sha256']:raise ValueError('Isolated payload mismatch')
                depth=np.load(p);covered|=(depth>0)&(depth<100)
            rgb_path=capture/f'model/sample/{i:04d}.png'
            if sha(rgb_path)!=payload[f'model/sample/{i:04d}.png']:raise ValueError('RGB payload mismatch')
            with Image.open(rgb_path) as im:images[c['condition']]=np.asarray(im.convert('RGB')).copy()
        if len(set(contexts))!=1:reasons.append('CONTEXT_CHANGED')
        if excluded_geometry:reasons.append('TARGET_GEOMETRY_UNKNOWN')
        deltas=[]
        for a,b in [('NONE','HEAD_ONLY'),('BODY_ONLY','BOTH')]:
            if a in images and b in images:
                diff=np.abs(images[a].astype(float)-images[b].astype(float)).mean(axis=2)
                d=diff[~covered]
                deltas.append(dict(pair=[a,b],background_pixels=len(d),mae_255=float(d.mean()) if len(d) else None,
                                   fraction_above_10=float((d>10).mean()) if len(d) else None))
        eligible=not reasons
        results.append(dict(pair_id=gid,eligible=eligible,reasons=sorted(set(reasons)),geometry_exclusions=excluded_geometry,
                            context_hash=contexts[0] if len(set(contexts))==1 else None,background_change=deltas))
        sheet=Image.new('RGB',(1280,204),'#181818');draw=ImageDraw.Draw(sheet)
        for j,c in enumerate(CONDITIONS):
            if c in images:sheet.paste(Image.fromarray(images[c]).resize((320,180)),(j*320,24))
            draw.text((j*320+4,5),c,fill='white')
        sheet.save(output/(gid+'.png'))
        for (i,c),context_hash in zip(members,contexts):
            metric.append(dict(sample_index=i,region_id=c['region_id'],split=c['split'],pair_id=gid,condition=c['condition'],
                               eligible=eligible,truth=near[i].tolist(),context_hash=context_hash))
    report=dict(status='PASS' if all(r['eligible'] for r in results) else 'REVIEW',frames=len(spec['cases']),quartets=len(results),
                eligible_quartets=sum(r['eligible'] for r in results),groups=results,
                source_sha256=dict(spec=sha(capture/'source/spec.json'),receipt=sha(capture/'receipt.json'),validation=sha(labels/'label-validation.json')),
                model_execution=False,scope='Source checks only; template bits are verified against native truth. Pixel differences include legitimate shadows/reflections.')
    for name,value in [('result.json',report),('metric-inputs.json',metric)]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','frames','quartets','eligible_quartets')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','labels','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();verify(a.capture,a.labels,a.output)
