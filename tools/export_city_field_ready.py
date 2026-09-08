"""Export synchronized, filtered City data with per-frame reasons and preserved raw evidence."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
FRAME_FAILURES={'CHANGED_RGB','CHANGED_DEPTH','CALIBRATION','POSE','CAPTURE_POSE_ANOMALY',
                'FLOOR_ANOMALY','LABEL_VALUES','LABEL_ANOMALY','UNKNOWN_ERASURE','FRAME_OUTSIDE_REGION'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def reject_reasons(frame):
    reasons=[]
    for kind in ('rgb','depth'):
        a=frame.get(kind,{})
        if not a.get('valid') or a.get('status')!='PRESENT':reasons.append(kind.upper()+'_INVALID')
    if frame.get('source_load_status')!='READY':reasons.append('LOAD_NOT_READY')
    if frame.get('synchronization',{}).get('status')!='PAIRED':reasons.append('UNPAIRED')
    pose=frame.get('pose_evidence',{})
    if pose.get('status')!='MEASURED' or pose.get('command_mismatch'):reasons.append('POSE_UNVERIFIED')
    if frame.get('floor_probe',{}).get('status')!='CONSISTENT':reasons.append('FLOOR_REVIEW')
    near=frame.get('labels',{}).get('near',[])
    if len(near)!=2 or any(v not in (0,1) for v in near):reasons.append('NEAR_LABEL_UNKNOWN')
    if any(t.get('label_status') not in ('EVALUABLE','NOT_PRESENT') for t in frame.get('targets',[])):
        reasons.append('ACTIVE_TARGET_LABEL_UNKNOWN')
    return reasons


def export_collection(collection,output):
    collection,output=Path(collection).resolve(),Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Fresh output under artifacts.local required')
    plan=read(collection/'plan.json');validation=read(collection/'collection-validation.json')
    routes={r['route_id']:r for r in plan['routes']}
    # Structural/cross-split contradictions cannot be repaired by dropping arbitrary frames.
    if any(i['status']=='FAIL' and i['code'] not in FRAME_FAILURES for i in validation['issues']):
        raise ValueError('Collection has structural/cross-split contract FAIL')
    invalid_frames=defaultdict(list)
    for issue in validation['issues']:
        if issue['status'] in ('FAIL','INCOMPLETE'):invalid_frames[issue['where']].append(issue['code'])
    bundles=[(p,read(p)) for p in sorted(collection.glob('*/*-bundle.json'))]
    if not bundles:raise ValueError('No captured route bundles')
    output.mkdir(parents=True);(output/'samples').mkdir()
    accepted=[];rejected=[];pairs=defaultdict(list);counts=Counter();source_hashes={}
    for path,bundle in bundles:
        source_hashes[str(path.relative_to(collection))]=sha(path)
        source=bundle['source'];capture=Path(bundle['capture']);label_dir=path.parent/'labels'
        if source['status']!='PASS' or source['source_unchanged'] is not True:
            raise ValueError('Incomplete or changed capture')
        check=read(label_dir/'label-validation.json')
        if check['status']!='PASS' or sha(label_dir/'native-route-labels.json')!=bundle['label_manifest_sha256']:
            raise ValueError('Label manifest identity mismatch')
        for rel,key in [('source/spec.json','spec_sha256'),('receipt.json','receipt_sha256')]:
            if sha(capture/rel)!=source[key] or source[key]!=check[key]:raise ValueError('Capture identity mismatch')
        capture_spec=read(capture/'source/spec.json');capture_receipt=read(capture/'receipt.json')
        placed=defaultdict(list)
        for obj in capture_receipt.get('controlled_objects',[]):placed[obj['case']].append(obj)
        for rel,key in [('support.npy','support_sha256'),('near.npy','near_sha256')]:
            if sha(label_dir/rel)!=check[key]:raise ValueError('Label payload identity mismatch')
        support=np.load(label_dir/'support.npy',mmap_mode='r');near=np.load(label_dir/'near.npy',mmap_mode='r')
        target_arrays={}
        for tid,expected in check['target_mask_sha256'].items():
            p=label_dir/'targets'/(tid+'.npy')
            if sha(p)!=expected:raise ValueError('Target mask identity mismatch')
            target_arrays[tid]=np.load(p,mmap_mode='r')
        for frame in bundle['frames']:
            i=frame['sample_index'];uid=f"{bundle['route_id']}-{i:04d}"
            reasons=reject_reasons(frame)
            reasons.extend(invalid_frames.get(f"{bundle['bundle_id']}/{i}",[]))
            for kind in ('rgb','depth'):
                p=Path(frame[kind]['path'])
                if not p.is_file() or sha(p)!=frame[kind].get('sha256'):reasons.append(kind.upper()+'_HASH_OR_MISSING')
            if reasons:
                rejected.append(dict(sample_id=uid,sample_index=i,route_id=bundle['route_id'],reasons=sorted(set(reasons))))
                counts.update(set(reasons));continue
            try:
                with Image.open(frame['rgb']['path']) as image:
                    image.load();rgb_shape=(image.height,image.width)
                    if max(hi-lo for lo,hi in image.convert('RGB').getextrema())<=1:
                        raise ValueError('Uniform/blank RGB')
                depth=np.load(frame['depth']['path'],allow_pickle=False)
                if rgb_shape!=(frame['calibration']['height'],frame['calibration']['width']):
                    raise ValueError('Calibration/image dimensions mismatch')
                if depth.shape!=rgb_shape or depth.dtype!=np.dtype('<f4') or not np.isfinite(depth).all():
                    raise ValueError('RGB/depth shape or finite-depth mismatch')
                if not ((depth>0)&(depth<100)).any():raise ValueError('No valid depth')
                if not np.array_equal(near[i],frame['labels']['near']):raise ValueError('Near-label index mismatch')
            except (ValueError,OSError) as e:
                rejected.append(dict(sample_id=uid,sample_index=i,route_id=bundle['route_id'],reasons=['PAYLOAD_INVALID'],detail=str(e)))
                counts['PAYLOAD_INVALID']+=1;continue
            folder=output/'samples'/uid;folder.mkdir()
            shutil.copyfile(frame['rgb']['path'],folder/'rgb.png')
            shutil.copyfile(frame['depth']['path'],folder/'depth.npy')
            arrays={'near':near[i],'support':support[i]}
            for target in frame['targets']:
                if target['label_status']=='EVALUABLE':arrays['target__'+target['target_id']]=target_arrays[target['target_id']][i]
            np.savez_compressed(folder/'labels.npz',**arrays)
            cal=frame['calibration'];fx=cal['width']/(2*math.tan(math.radians(cal['horizontal_fov_degrees']/2)))
            record=dict(sample_id=uid,source_sample_index=i,route_id=bundle['route_id'],region_id=bundle['region_id'],
                split=bundle['split'],scene_type=bundle['scene_type'],variant=frame['variant'],pair_id=frame.get('pair_id'),
                condition_description=routes[bundle['route_id']].get('condition_description',{}).get(frame['variant']),
                route_distance_m=capture_spec['cases'][i].get('route_distance_m'),
                rgb=f'samples/{uid}/rgb.png',depth=f'samples/{uid}/depth.npy',labels=f'samples/{uid}/labels.npz',
                camera=frame['camera'],calibration=dict(cal,intrinsic_matrix=[[fx,0,(cal['width']-1)/2],[0,fx,(cal['height']-1)/2],[0,0,1]]),
                floor_probe=frame['floor_probe'],targets=frame['targets'],synchronization=frame['synchronization'],
                placed_objects=placed[capture_spec['cases'][i]['name']],
                sha256={name:sha(folder/name) for name in ('rgb.png','depth.npy','labels.npz')})
            write(folder/'metadata.json',record);accepted.append(record)
            if record['pair_id']:pairs[record['pair_id']].append(record)
    with (output/'frames.jsonl').open('w',encoding='utf-8') as f:
        for r in accepted:f.write(json.dumps(r)+'\n')
    write(output/'excluded.json',rejected)
    write(output/'pairs.json',[dict(pair_id=pid,clear=[r['sample_id'] for r in rs if r['variant']=='clear'],
        obstacles=[r['sample_id'] for r in rs if r['variant']!='clear'],
        usable=any(r['variant']=='clear' for r in rs) and any(r['variant']!='clear' for r in rs)) for pid,rs in pairs.items()])
    shutil.copyfile(collection/'plan.json',output/'plan.json')
    summary=dict(status='READY_FILTERED' if accepted else 'EMPTY_AFTER_FILTER',accepted=len(accepted),excluded=len(rejected),
        excluded_reasons=dict(counts),by_route=dict(Counter(r['route_id'] for r in accepted)),
        by_scene=dict(Counter(r['scene_type'] for r in accepted)),by_split=dict(Counter(r['split'] for r in accepted)),
        by_variant=dict(Counter(r['variant'] for r in accepted)),
        missing_hazard_variants=sorted({'thin_pole','body_protrusion','head_bar','suspended_sign'}-{r['variant'] for r in accepted}),
        source_bundles_sha256=source_hashes,source_collection=str(collection),
        exporter_sha256=sha(Path(__file__)),plan_sha256=sha(collection/'plan.json'),
        semantics='Static RGB-D samples; -1 label pixels stay UNKNOWN. Excluded raw frames are preserved. Clear means configured hazard absent, not globally obstacle-free.',
        visual_acceptance='Use the admitted route plan and its scene previews; numerical readiness alone does not certify distant buildings.')
    write(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export_collection(a.collection,a.output)))
