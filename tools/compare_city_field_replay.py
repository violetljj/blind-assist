"""Compare two immutable region acquisitions without treating RGB as bit-exact."""
import argparse
import json
from pathlib import Path


def compare(first,second,output):
    import numpy as np
    from PIL import Image
    from city_field_contract import sha, pose_delta
    first,second=Path(first).resolve(strict=True),Path(second).resolve(strict=True)
    rows=[]; errors=[]
    for collection in (first,second):
        receipt=json.loads((collection/'receipt.json').read_text())
        if receipt.get('completed') is not True or receipt.get('status') in ('FAIL','INCOMPLETE'):
            errors.append('Collection integrity not complete '+str(collection))
    for path in sorted(second.glob('*/*-bundle.json')):
        replay=json.loads(path.read_text()); rid=replay['route_id']
        baseline_path=first/replay['region_id']/path.name
        if not baseline_path.is_file(): errors.append('Missing baseline '+rid);continue
        baseline=json.loads(baseline_path.read_text())
        if baseline['source']['spec_sha256']!=replay['source']['spec_sha256']:
            errors.append('Capture spec differs '+rid)
        if len(baseline['frames'])!=len(replay['frames']): errors.append('Frame count differs '+rid)
        for a,b in zip(baseline['frames'],replay['frames']):
            pair=(rid,b['sample_index'])
            delta=pose_delta(a['camera'],b['camera'])
            ids_equal=a['sample_index']==b['sample_index'] and a['variant']==b['variant']
            with Image.open(a['rgb']['path']) as im: ar=np.asarray(im,dtype=np.float32)
            with Image.open(b['rgb']['path']) as im: br=np.asarray(im,dtype=np.float32)
            ad=np.load(a['depth']['path']); bd=np.load(b['depth']['path'])
            target_a={t['target_id']:t['label_status'] for t in a['targets']}
            target_b={t['target_id']:t['label_status'] for t in b['targets']}
            stable=ids_equal and delta['translation_m']<=.001 and delta['rotation_degrees']<=.01 and a['labels'].get('near')==b['labels'].get('near') and target_a==target_b
            if not stable: errors.append('Pose/index/label differs '+str(pair))
            rows.append(dict(route_id=rid,sample_index=b['sample_index'],stable=stable,pose_delta=delta,
                rgb_identical=a['rgb']['sha256']==b['rgb']['sha256'],rgb_mean_absolute_difference=float(np.abs(ar-br).mean()),
                depth_identical=a['depth']['sha256']==b['depth']['sha256'],depth_mean_absolute_difference_m=float(np.abs(ad-bd).mean()),
                near_identical=a['labels'].get('near')==b['labels'].get('near'),target_status_identical=target_a==target_b))
    if not rows: errors.append('No replay frames')
    result=dict(status='REVIEW' if errors else 'PASS',frame_count=len(rows),errors=errors,
        plan_identical=sha(first/'plan.json')==sha(second/'plan.json'),rows=rows,
        authority='Same configuration/pose/near and target status replay; RGB and depth differences measured, not assumed deterministic')
    if not result['plan_identical']: result['status']='REVIEW';errors.append('Plan differs')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('first','second','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();compare(args.first,args.second,args.output)
