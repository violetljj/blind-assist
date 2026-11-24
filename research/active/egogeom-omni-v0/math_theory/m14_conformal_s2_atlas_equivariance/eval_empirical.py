"""Zero-shot v4.1 CPU ONNX. Calibrated Oxford raw fisheye and rectified Aria."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from atlas import (level_basis, pin_rays, project_pinhole, project_kb, mercator_geometry,
                   mercator_pixels, yaw, pull)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]/'..'
ROOT=ROOT.resolve()
WORK=ROOT/'artifacts.local/work'
ROUTE=WORK/'egogeom-omni-v0'
OUT=ROUTE/'math_theory'/HERE.name
MODEL=ROUTE/'egogeom_omni_v41_best_pure_rgb_fp32.onnx'
OX=WORK/'paired-low-sites-20261118/oxford'
CONTROL=WORK/'shallow-alerts-20261118/controls/oxford_valid_camera.json'
ARIA=ROOT/'artifacts.local/downloads/aria-gen2-pilot-20261117'
MEAN=np.array([.485,.456,.406],np.float32)
STD=np.array([.229,.224,.225],np.float32)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def oxford_inputs():
    # Read constants from the frozen production of the historical virtual camera.
    sys.path.insert(0,str(ROOT/'research/active/hardware-bringup/host'))
    from spires_insert_dev_20261117 import K0,D0,R_CV,CX,CY
    control=json.loads(CONTROL.read_text())
    manifest=OX/'locked/manifest.json'
    rows={(r['seq'],r['i']):r for r in json.loads(manifest.read_text())['frames']}
    def project(r):
        return project_kb(r@R_CV.T,K0,D0,np.deg2rad(80))
    for a in control['rows']:
        row=rows[(a['seq'],a['i'])]
        p=OX/'original'/a['seq']/a['source']
        dep=OX/row['depth']
        k_native=(280.,280.,CX,CY)
        def refproject(ray):
            return project_pinhole(ray,k_native)
        yield dict(domain='Oxford',source=a['seq'],key=str(a['i']),image=cv2.imread(str(p)),
                   project=project,refproject=refproject, ref_path=dep,ref_scale=row['ref_scale'],
                   k=(224.,224.,192.,144.),down=row['down'],files=[p,dep],
                   reference='sparse TLS geometry, virtual-camera axial depth')


def aria_inputs():
    for p in sorted((ROUTE/'zeroshot_eval/aria_g2_20clips').glob('*.npz')):
        with np.load(p) as a:
            offsets=a['rgb_offsets']
            image=cv2.imdecode(a['rgb_blob'][offsets[0]:offsets[1]],cv2.IMREAD_COLOR)
            orig=a['orig_hw']
            k0=a['K_orig'].astype(float)
            down=-a['poses_R'][0][2]
        keys=p.stem.split('_')
        walk='_'.join(keys[1:3]); idx=int(keys[-1][1:])*150*6
        params=ARIA/walk/'x/depth/pinhole_camera_parameters.json'
        raw=json.loads(params.read_text())[idx]
        refp=params.parent/'depth'/('depth_%08d.png'%raw['index'])
        if not refp.is_file():
            raise FileNotFoundError(refp)
        kh=image.shape[0]/orig[0]; kw=image.shape[1]/orig[1]
        def project(ray):
            xy,ok=project_pinhole(ray,k0)
            return (xy+.5)*np.array([kw,kh])-.5,ok
        def refproject(ray):
            return project_pinhole(ray,raw['camera']['Parameters'])
        # eval_cross_glasses_zeroshot.decode_isotropic_frame, 512x512 input.
        k=k0*.75
        k[3]-=48
        yield dict(domain='Aria',source=walk,key=p.stem,image=image,project=project,
                   refproject=refproject,ref_path=refp,ref_scale=.001,
                   k=k,down=down,files=[p,refp,params],reference='FoundationStereo axial depth, rectified input')


def zoom_report(arrays):
    ok=np.isfinite(arrays)
    return dict(n=int(ok.sum()),p50=float(np.median(arrays[ok])),p95=float(np.quantile(arrays[ok],.95)))


def session():
    op=ort.SessionOptions()
    op.intra_op_num_threads=op.inter_op_num_threads=1
    return ort.InferenceSession(str(MODEL),op,providers=['CPUExecutionProvider'])


def infer(sess,bgr,rays,mask):
    rgb=bgr[:,:,::-1].astype(np.float32)/255.
    rgb=(rgb-MEAN)/STD
    v=rays.astype(np.float32)
    chord=np.max(np.linalg.norm(np.diff(rays,axis=1),axis=-1))
    scale=np.float32(np.clip(350*np.sqrt(chord**2+1.e-12),.2,4.))
    start=time.perf_counter()
    z=sess.run(['depth_m'],dict(rgb=np.ascontiguousarray(rgb.transpose(2,0,1)[None]),
                                grav_rays=np.ascontiguousarray(v.transpose(2,0,1)[None]),
                                host_focal=np.full((1,1,1,1),scale,dtype=np.float32)))[0].squeeze()
    return z,(time.perf_counter()-start)*1000


def read_frame(row,sess):
    h,w=288,384
    basis=level_basis(row['down'])
    k=row['k']
    native_cam=pin_rays(h,w,k)
    pin=native_cam@basis.T
    merc,geometry=mercator_geometry(h,w,k,basis)
    timings=[]
    maps={}
    for chart,r0 in [('pin',pin),('merc',merc)]:
        for steps in (0,12):
            ang=steps*geometry[0]
            r=yaw(r0,ang)
            c=r@basis
            p,valid=row['project'](c)
            img,in_view=pull(row['image'],p,valid)
            # Rotate the virtual camera about gravity. Its head-level coordinate
            # axes yaw with the wearer, so perceptual rays remain r0.
            z,ms=infer(sess,img,r0,in_view)
            timings.append(ms)
            # Interpret frozen depth_m as optical-axis Z in the virtual camera.
            denom=(r0@basis)[...,2]
            rng=z/np.maximum(denom,.0001)
            if chart=='pin':
                coord,ok=project_pinhole(yaw(pin,-ang)@basis,k)
                coord-=.5
            else:
                coord=mercator_pixels(yaw(pin,-ang),h,w,geometry)
                ok=np.ones((h,w),bool)
            val,inside=pull(rng,coord,ok)
            native_support,_=pull(in_view.astype(np.float32),coord,ok)
            joint=inside&(native_support>=.999)&np.isfinite(val)&(val>0)
            maps[chart,steps]=(val,joint,in_view,img,coord)
    ori,m_native=maps['pin',0][:2]
    warped,m_merc=maps['merc',0][:2]
    reb,e=pull(maps['merc',0][3],maps['merc',0][4],m_native&m_merc)
    back,ms=infer(sess,reb,pin,e)
    timings.append(ms)
    back=back/native_cam[...,2]
    gtpos,gtok=row['refproject'](native_cam)
    ref=cv2.imread(str(row['ref_path']),cv2.IMREAD_UNCHANGED)
    if ref is None or ref.ndim!=2:
        raise ValueError(row['ref_path'])
    gt,gm=pull(ref.astype(np.float32)*row['ref_scale'],gtpos,gtok,nearest=True)
    gm &= m_native&(gt>=.3)&(gt<=12.)
    common=gm&m_merc
    shared=m_native&m_merc
    p1,mp1=maps['pin',12][:2]
    m1,mm1=maps['merc',12][:2]
    common_yaw=shared&mp1&mm1
    record={'domain':row['domain'],'source':row['source'],'key':row['key'],
            'files_sha256':{p.name:digest(p) for p in row['files']},'reference':row['reference'],
            'raster_pixels':h*w,'native_valid':int(maps['pin',0][2].sum()),
            'merc_valid':int(maps['merc',0][2].sum()),'common_pixels':int(shared.sum()),
            'yaw_common_pixels':int(common_yaw.sum()),'yaw_degrees':float(np.rad2deg(12*geometry[0])),
            'reference_pixels':int(gm.sum()),'reference_common_pixels':int(common.sum()),'inference_ms':timings}
    def collect(key,arr,mask):
        record[key+'_sum']=float(np.abs(arr[mask]).sum(dtype=np.float64))
    collect('chart_change_relative',(warped-ori)/np.maximum(ori,.001),shared)
    collect('rgb_roundtrip_relative',(back-ori)/np.maximum(ori,.001),shared)
    collect('pin_yaw_relative',(p1-ori)/np.maximum(ori,.001),common_yaw)
    collect('merc_yaw_relative',(m1-warped)/np.maximum(warped,.001),common_yaw)
    for name,a,good in [('pin',ori,m_native),('merc',warped,m_merc),('roundtrip',back,shared)]:
        # Back to the reference virtual camera's axial Z before metric scoring.
        pred=a*native_cam[...,2]
        collect(name+'_common_z_error_m',pred-gt,common)
        collect(name+'_fixed_z_error_m',np.where(good,pred,6.)-gt,gm)
    return record


def full_oxford_mask():
    sys.path.insert(0,str(ROOT/'research/active/hardware-bringup/host'))
    from spires_insert_dev_20261117 import K0,D0,R_CV,CX,CY,H,W
    from shallow_valid_view_20261118 import _oxford_view
    frozen=_oxford_view()
    y,x=np.mgrid[:H,:W]
    rays=np.stack([(x-CX)/280.,(y-CY)/280.,np.ones_like(x)],axis=-1)@R_CV.T
    xy,valid=project_kb(rays,K0,D0,np.deg2rad(80))
    mask=valid&(xy[...,0]>=0)&(xy[...,0]<=1439)&(xy[...,1]>=0)&(xy[...,1]<=1079)
    assert np.array_equal(mask,frozen)
    return dict(valid=int(mask.sum()),invalid=int((~mask).sum()),sha256=hashlib.sha256(mask.tobytes()).hexdigest())


def aggregate(rows):
    def rate(k,n):
        return sum(r[k] for r in rows)/sum(r[n] for r in rows)
    out={'frames':len(rows),'raster_pixels':sum(r['raster_pixels'] for r in rows)}
    for k,n in [('native_valid','raster_pixels'),('merc_valid','raster_pixels'),('common_pixels','raster_pixels')]:
        out[k+'_fraction']=rate(k,n)
    for k,n in [('chart_change_relative','common_pixels'),('rgb_roundtrip_relative','common_pixels'),
                ('pin_yaw_relative','yaw_common_pixels'),('merc_yaw_relative','yaw_common_pixels')]:
        out[k]=rate(k+'_sum',n)
    for k in ('pin','merc','roundtrip'):
        for arm,n in [('common','reference_common_pixels'),('fixed','reference_pixels')]:
            out[k+'_'+arm+'_MAE_m']=rate(k+'_'+arm+'_z_error_m_sum',n)
    out['reference_pixels']=sum(r['reference_pixels'] for r in rows)
    out['reference_common_pixels']=sum(r['reference_common_pixels'] for r in rows)
    out['yaw_common_pixels']=sum(r['yaw_common_pixels'] for r in rows)
    out['inference_ms']=zoom_report(np.array([x for r in rows for x in r['inference_ms']]))
    return out


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True,parents=True)
    sess=session()
    records=[]
    for data in (oxford_inputs(),aria_inputs()):
        for row in data:
            records.append(read_frame(row,sess))
            print(row['source'],row['key'],flush=True)
    agg={d:aggregate([r for r in records if r['domain']==d]) for d in ('Aria','Oxford')}
    result=dict(records=records,aggregate=agg,source_groups={g:aggregate([r for r in records if r['source']==g])
            for g in sorted({r['source'] for r in records})},code_sha256={p.name:digest(p) for p in (Path(__file__),HERE/'atlas.py')},
            model_sha256=digest(MODEL),oxford_historical_mask=full_oxford_mask(),alert_fa_km=None,clear_repeated_km=None)
    (OUT/'v5_empirical_report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(agg))


if __name__=='__main__':
    main()
