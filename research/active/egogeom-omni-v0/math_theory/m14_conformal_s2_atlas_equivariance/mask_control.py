"""Fixed-raster Oxford padding ablation and real-ray shear/pose bridges."""
import json
import time

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from atlas import level_basis,pin_rays,project_pinhole,pull,mercator_geometry,unit
from eval_empirical import (HERE,OUT,OX,ARIA,ROUTE,MEAN,MODEL,digest,infer,session,
                            oxford_inputs,aria_inputs)


def geometry_report():
    stats={}
    for rows in (oxford_inputs(),aria_inputs()):
        for r in rows:
            k=r['k']; basis=level_basis(r['down'])
            cam=pin_rays(288,384,k)
            uv,ok=r['project'](cam)
            _,usable=pull(np.zeros(r['image'].shape[:2],np.uint8),uv,ok)
            # The native pinhole horizontal and vertical scales can differ.
            # Gram of the exact backprojected unit rays w.r.t. pixel coordinates.
            ex=np.array([1/k[0],0,0]); ey=np.array([0,1/k[1],0])
            a=ex-(cam@ex)[...,None]*cam
            b=ey-(cam@ey)[...,None]*cam
            aa=(a*a).sum(axis=-1); ab=(a*b).sum(axis=-1); bb=(b*b).sum(axis=-1)
            disc=np.hypot(aa-bb,2*ab)
            ratio=np.sqrt((aa+bb+disc)/(aa+bb-disc))
            one=dict(pinhole_shear_max=float(np.max(ratio[usable])),pinhole_shear_median=float(np.median(ratio[usable])),
                     mercator_shear_ideal=1.)
            if r['domain']=='Aria':
                stem=r['key']; tags=stem.split('_'); start=int(tags[-1][1:])*900
                trajectory=json.loads((ARIA/r['source']/'x/depth/pinhole_camera_parameters.json').read_text())
                params=trajectory[start]
                q=params['T_world_camera']['QuaternionXYZW']
                with np.load(ROUTE/'zeroshot_eval/aria_g2_20clips'/(stem+'.npz')) as packed:
                    err=float(np.abs(packed['poses_R'][0]-Rotation.from_quat(q).as_matrix()).max())
                    ke=float(np.abs(packed['K_orig']-params['camera']['Parameters']).max())
                    medium=np.median([f['camera']['Parameters'] for f in trajectory[start:start+900:6]],axis=0)
                    bridge=float(np.abs(packed['K_orig']-medium).max())
                    trans=float(np.abs(packed['poses_t'][0]-params['T_world_camera']['Translation']).max())
                if err>1.e-6 or trans>1.e-5 or bridge>1.e-4:
                    raise ValueError('Packed Aria frame did not match reference source')
                one.update(pose_max_abs=err,translation_max_abs_m=trans,clip_median_K_max_abs_px=bridge,first_frame_vs_packed_K_max_abs_px=ke)
            stats[r['domain']+'/'+r['key']]=one
    return stats


def padding():
    sess=session()
    rows=[]
    for row in oxford_inputs():
        basis=level_basis(row['down'])
        # Entire archived 480x640 virtual field rescaled to 384x288, explicit
        # unequal fx/fy for all three arms rather than crop away its margin.
        k=(224.,126.,192.,144.)
        cam=pin_rays(288,384,k)
        p,ok=row['project'](cam)
        image,valid=pull(row['image'],p,ok)
        rays=cam@basis.T
        gtxy,good=row['refproject'](cam)
        gt,gm=pull(cv2.imread(str(row['ref_path']),cv2.IMREAD_UNCHANGED).astype(np.float32)*row['ref_scale'],
                   gtxy,good,nearest=True)
        gm &= valid&(gt>=.3)&(gt<=12.)
        if not gm.any():
            raise ValueError('No evaluable depth')
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        variants={'black':image,'mean':image.copy(),'nearest_valid':image[tuple(nearest)]}
        variants['mean'][~valid]=np.rint(255*MEAN[::-1]).astype(np.uint8)
        preds={}
        for name,frame in variants.items():
            preds[name],_=infer(sess,frame,rays,valid)
        errors={k:float(np.abs(v[gm]-gt[gm]).sum(dtype=float)) for k,v in preds.items()}
        drift={k:float((np.abs(v[gm]-preds['black'][gm])/np.maximum(preds['black'][gm],.001)).sum(dtype=float))
               for k,v in preds.items() if k!='black'}
        rows.append(dict(key=row['key'],source=row['source'],reference_pixels=int(gm.sum()),invalid=int((~valid).sum()),
                         error_m_sum=errors,relative_change_sum=drift))
        print(row['source'],row['key'],flush=True)
    n=sum(r['reference_pixels'] for r in rows)
    return dict(rows=rows,reference_pixels=n,coverage=1-sum(r['invalid'] for r in rows)/(len(rows)*288*384),
                MAE_m={k:sum(r['error_m_sum'][k] for r in rows)/n for k in ('black','mean','nearest_valid')},
                relative_change={k:sum(r['relative_change_sum'][k] for r in rows)/n for k in ('mean','nearest_valid')})


if __name__=='__main__':
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True,parents=True)
    r=dict(geometry=geometry_report(),padding=padding(),model_sha256=digest(MODEL),
           code_sha256={p.name:digest(p) for p in (HERE/'mask_control.py',HERE/'atlas.py',HERE/'eval_empirical.py')})
    (OUT/'padding_control.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    print({k:v for k,v in r['padding'].items() if k!='rows'})
