"""Choose measured settling cost only after paired RGB/native agreement."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

def main(root):
    root=root.resolve(); capture=root/'capture'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    assert read(capture/'verification.json')['status']=='PASS'
    spec=read(capture/'evaluator/spec.json'); data=read(capture/'model/dataset.json')
    assert spec['preflight']
    objects=np.load(capture/'evaluator/object_support.npz'); support=np.load(capture/'evaluator/support.npz')
    labels=read(capture/'evaluator/labels.json')['targets']; samples={s['sample_id']:s for s in data['samples']}
    images={}
    for sid,s in samples.items():
        f=data['frames'][s['frame_indices'][-1]]
        images[sid]=np.asarray(Image.open(capture/'model'/f['rgb_path']).convert('RGB')).astype(float)
    checks=[]
    for sid in samples:
        if '_settle8_' not in sid: continue
        candidate=sid.replace('_settle8_','_settle2_');diff=np.abs(images[sid]-images[candidate])
        ious=[];local={}
        for a,b in zip(support[sid]>0,support[candidate]>0):
            union=(a|b).sum();ious.append(float((a&b).sum()/union) if union else 1.)
        for name in ('bar','box'):
            mask=objects[sid+'__'+name]>0
            if not mask.any():continue
            ys,xs=np.where(mask); y0=max(0,int(ys.min()*20)-8); y1=min(360,int((ys.max()+1)*20)+8)
            x0=max(0,int(xs.min()*20)-8);x1=min(640,int((xs.max()+1)*20)+8)
            local[name]=float(diff[y0:y1,x0:x1].mean())
        good=labels[sid]==labels[candidate] and min(ious)>=.98 and diff.mean()<=2 and all(v<=5 for v in local.values())
        checks.append(dict(reference=sid,candidate=candidate,labels_equal=labels[sid]==labels[candidate],support_IoU=ious,global_RGB_MAE=float(diff.mean()),object_crop_RGB_MAE=local,quality_pass=bool(good)))
    profiles=read(capture/'receipt.json')['profiles'];timing={}
    for setting in (8,2):
        rows=[p for p in profiles if p['settling_frames']==setting and p['sample_index']>0]
        timing[str(setting)]={k:float(np.median([r[k] for r in rows])) for k in rows[0] if k.endswith('_s')}
    quality=all(c['quality_pass'] for c in checks);faster=timing['2']['total_s']<timing['8']['total_s']
    selected=2 if quality and faster else 8
    result=dict(status='COMPLETE',selected_settling=selected,quality_pass=quality,faster=faster,comparisons=checks,timing_median_s=timing,
                expected_main_frames=144,estimated_main_steady_s=144*timing[str(selected)]['total_s'],
                startup_overhead_s=read(capture/'receipt.json')['wall_elapsed_s']-sum(p['total_s'] for p in profiles),
                scope='Two source groups only; excludes first32-warmup frame; quality cannot certify all later renders; callbacks/readback timings are wall attribution, not kernel profiles')
    (root/'selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
