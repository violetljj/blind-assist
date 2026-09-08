"""Explain preserved target UNKNOWN labels from hashed capture evidence; never relabel."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explain_ray(ray, source_component, instance_index):
    if ray['status'] != 'UNKNOWN':
        return ray['status']
    if 'collision_axial_m' not in ray:
        return 'NO_COLLISION_HIT'
    same = ray.get('component_path') == source_component and (
        instance_index is None or ray.get('instance_index') == instance_index)
    delta = ray['collision_axial_m'] - ray['isolated_axial_m']
    if same:
        return 'TARGET_COLLISION_DEPTH_DISAGREEMENT'
    if -.03 <= delta < 0:
        return 'OTHER_COMPONENT_NEARER_WITHIN_RENDER_TOLERANCE'
    if delta < -.03:
        return 'INCONSISTENT_OCCLUSION_STATUS'
    return 'OTHER_COMPONENT_AT_OR_BEHIND_TARGET'


def audit(capture, labels, output):
    capture, labels, output = map(lambda p: Path(p).resolve(), (capture, labels, output))
    if output.exists() or not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Fresh canonical artifact output required')
    spec_path=capture/'source/spec.json'
    ray_path=capture/'evaluator/target-raycheck.json'
    validation_path=labels/'label-validation.json'
    validation=read(validation_path)
    for path,key in [(spec_path,'spec_sha256'),(ray_path,'raycheck_sha256'),(capture/'receipt.json','receipt_sha256')]:
        if sha(path)!=validation[key]:
            raise ValueError('Evidence identity mismatch: '+key)
    if validation['status']!='PASS':
        raise ValueError('Completed label verification required')
    spec=read(spec_path); rays=read(ray_path)
    if rays['schema']!='city-native-target-raycheck-v1':
        raise ValueError('Unexpected ray schema')
    frames={f['sample_index']:f for f in read(capture/'model/dataset.json')['frames']}
    evidence={(tid,r['sample_index']):r for tid,rows in validation['targets'].items() for r in rows}
    result=[];seen=set();counts=Counter();statuses=Counter()
    output.mkdir(parents=True)
    for group in rays['rows']:
        tid,i=group['target_id'],group['sample_index']; key=(tid,i)
        if key in seen:
            raise ValueError('Duplicate target/frame ray evidence')
        seen.add(key); label=evidence[key];statuses[label['status']]+=1
        if label['status']!='UNKNOWN':
            continue
        details=[]
        for ray in group['rows']:
            if ray['status']=='UNKNOWN':
                reason=explain_ray(ray,group['source_component'],group.get('original_instance_index'))
                counts[reason]+=1;details.append(dict(ray,reason=reason))
        case=spec['cases'][i]
        render_ok=(label.get('native_agree_pixels',0)>=3 and label.get('unexplained_nearer_clone_pixels',1)==0)
        item=dict(target_id=tid,sample_index=i,distance_m=case.get('distance_m'),
            azimuth_deg=case.get('approach_azimuth_deg'),status='UNKNOWN',
            render_gate_passes=render_ok,matched_rays=group['matched_rays'],
            uncertainty_rays=details,source_component=group['source_component'],
            render_evidence=label,overlay=f'{i:04d}-{tid}.png')
        rgb_path=capture/'model'/frames[i]['rgb_path']
        item['rgb_sha256']=sha(rgb_path)
        with Image.open(rgb_path) as image:
            canvas=image.convert('RGB');draw=ImageDraw.Draw(canvas)
            sx,sy=canvas.width/640,canvas.height/360
            for ray in group['rows']:
                x,y=ray['pixel'];x,y=x*sx,y*sy
                color={'MATCH':'lime','OCCLUDED':'cyan','UNKNOWN':'red'}[ray['status']]
                draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
            draw.rectangle((0,0,canvas.width,23),fill='black')
            draw.text((4,5),f'{i} {tid} | green=match cyan=occluded red=unknown',fill='white')
            canvas.save(output/item['overlay'])
        result.append(item)
    report=dict(schema='city-target-uncertainty-audit-v1',status='PASS_DIAGNOSIS_ONLY',
        source_hashes={str(p.relative_to(ROOT.resolve())) if p.is_relative_to(ROOT.resolve()) else str(p):sha(p)
            for p in [spec_path,ray_path,validation_path,Path(__file__)]},
        active_target_status_counts=dict(statuses),unknown_frames=len(result),
        unknown_ray_count=sum(counts.values()),unknown_ray_reasons=dict(counts),rows=result,
        label_changes=0,scope='Saved evidence diagnosis, not new identity evidence or model scoring')
    (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_hashes')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.capture,a.labels,a.output)
