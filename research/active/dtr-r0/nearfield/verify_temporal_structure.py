"""NF-G6 evaluator-only acquisition verification, geometric masks and contact sheet."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def rays(pose):
    p,y=np.radians([pose['pitch'],pose['yaw']])
    forward=np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)])
    right=np.array([-np.sin(y),np.cos(y),0.])
    up=np.array([-np.sin(p)*np.cos(y),-np.sin(p)*np.sin(y),np.cos(p)])
    v,u=np.mgrid[:360,:640]
    f=320/np.tan(np.radians(50))
    return forward+((u-319.5)/f)[...,None]*right-((v-179.5)/f)[...,None]*up


def intersect(origin,directions,obj):
    center=np.array(obj['center_m'])
    half=np.array(obj['size_m'])/2
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(center-half-origin)/directions
        b=(center+half-origin)/directions
    near=np.max(np.minimum(a,b),axis=2)
    far=np.min(np.maximum(a,b),axis=2)
    return np.where((far>=near)&(near>0),near,np.inf)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    args=ap.parse_args()
    root=args.capture.resolve()
    assert not (root/'verification.json').exists()
    started=time.perf_counter()
    receipt=json.loads((root/'receipt.json').read_text())
    release=json.loads((root/'process-release.json').read_text())
    manifest=json.loads((root/'model/sensor_manifest.json').read_text())
    spec=json.loads((root/'evaluator/spec.json').read_text())
    assert receipt['status']=='PASS' and receipt['source_unchanged'] and release['released']
    frames=manifest['frames']
    assert len(frames)==36 and [f['sample_index'] for f in frames]==list(range(36))
    assert not list((root/'model').rglob('*.npy'))
    masks=root/'evaluator/projected_masks'
    masks.mkdir()
    rows=[]
    tiles=[]
    for i,(frame,case) in enumerate(zip(frames,spec['cases'])):
        assert frame['clip_id']==case['clip_id'] and frame['frame_in_clip']==case['frame_in_clip']
        assert frame['camera_transform']==case['camera']
        assert 'time_s' not in frame and 'depth_path' not in frame
        image=cv2.imread(str(root/'model'/frame['rgb_path']))
        native=np.load(root/f'evaluator/native/{i:04d}.npy')
        assert image.shape==(360,640,3) and native.shape==(360,640)
        pose=case['camera']
        origin=np.array([pose[k] for k in ('x','y','z')])
        direction=rays(pose)
        obj_rows=[]
        for j,obj in enumerate(case['objects']):
            expected=intersect(origin,direction,obj)
            projected=np.isfinite(expected)
            visible=projected&(native>0)&(np.abs(native-expected)<.03)
            cv2.imwrite(str(masks/f'{i:04d}_{j:02d}.png'),visible.astype(np.uint8)*255)
            err=np.abs(native[visible]-expected[visible])
            obj_rows.append(dict(name=obj['name'],projected_pixels=int(projected.sum()),native_visible_pixels=int(visible.sum()),native_visible_fraction=float(visible.sum()/max(1,projected.sum())),forward_depth_error_p95_m=float(np.quantile(err,.95)) if len(err) else None))
            if obj['name'] in ('bar','flush_wall_mark'):
                assert visible.sum()>0, f'No projected visible target: {i} {obj}'
        floor=(native>0)&(np.abs(origin[2]+native*direction[:,:,2]-.12)<.02)
        assert floor[320:350,300:340].sum()>500, f'Optical ground mismatch frame{i}'
        tile=np.zeros((207,320,3),np.uint8)
        tile[:180]=cv2.resize(image,(320,180))
        label=f"{frame['clip_id']} f{frame['frame_in_clip']}"
        cv2.putText(tile,label,(5,198),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        tiles.append(tile)
        rows.append(dict(sample_index=i,clip_id=frame['clip_id'],rgb_sha256=sha(root/'model'/frame['rgb_path']),native_sha256=sha(root/f'evaluator/native/{i:04d}.npy'),native_valid_fraction=float((native>0).mean()),floor_consistent_pixels=int(floor.sum()),objects=obj_rows))
    sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,36,3)],axis=0)
    assert cv2.imwrite(str(root/'contact-sheet.jpg'),sheet)
    result=dict(status='PASS',frames=36,clips=12,source_unchanged=True,processes_released=True,model_excludes_native_truth=True,calibration='UE ray-AABB forward-depth projection; cx319.5 cy179.5; floorZ0.12',rows=rows,wall_elapsed_s=time.perf_counter()-started,code_sha256=sha(Path(__file__)),scope='Acquisition and projected-reference consistency only; posed static geometric windows, not actual sensor timing or deployment pose quality.')
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__': main()
