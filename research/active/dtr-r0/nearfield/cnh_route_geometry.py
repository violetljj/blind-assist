"""Small native-render geometry canary; exported render triangles are authority."""
from __future__ import annotations
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np


def triangles(root, objects, attributes=False):
    all_tri, all_id, all_normal, all_colour=[],[],[],[]
    for obj in objects:
        path=root/obj['mesh']['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=obj['mesh']['sha256']:
            raise ValueError('Mesh hash differs from captured descriptor')
        mesh=json.loads(path.read_text())
        x,y,z,w=obj['actual_rotation_quaternion']
        rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        for section in mesh['sections']:
            vertex=np.asarray(section['vertices_m'])*obj['actual_scale']
            vertex=vertex@rot.T+obj['actual_translation_m']
            tri=vertex[np.asarray(section['triangles']).reshape(-1,3)]
            all_tri.append(tri);all_id.extend([obj['id']]*len(tri))
            normals=(np.asarray(section['normals'])/obj['actual_scale'])@rot.T
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
            all_normal.append(normals[np.asarray(section['triangles']).reshape(-1,3)])
            all_colour.extend([obj['albedo']]*len(tri))
    result=(np.concatenate(all_tri),np.asarray(all_id,dtype=np.uint16))
    return (*result,np.concatenate(all_normal),np.asarray(all_colour)) if attributes else result


def trace(origin, directions, tri, ids, vertex_normals=None):
    """Two-sided Moller-Trumbore, unnormalized rays retain camera-axial units."""
    result=np.full(len(directions),np.inf); result_id=np.zeros(len(directions),dtype=np.uint16)
    result_normal=np.full((len(directions),3),np.nan)
    result_index=np.full(len(directions),-1,dtype=int)
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];offset=origin-tri[:,0]
    q=np.cross(offset,e1);num=np.einsum('ij,ij->i',e2,q)
    for start in range(0,len(directions),128):
        direction=directions[start:start+128]
        p=np.cross(direction[:,None,:],e2)
        det=np.einsum('rtk,tk->rt',p,e1)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-10)
        u=np.einsum('tk,rtk->rt',offset,p)*inv
        v=np.einsum('rk,tk->rt',direction,q)*inv
        t=num[None,:]*inv
        valid=(np.abs(det)>1e-10)&(u>=0)&(v>=0)&(u+v<=1)&(t>0)
        t=np.where(valid,t,np.inf); winner=t.argmin(axis=1)
        nearest=t[np.arange(len(direction)),winner]
        result[start:start+len(direction)]=nearest
        result_id[start:start+len(direction)]=np.where(np.isfinite(nearest),ids[winner],0)
        if vertex_normals is not None:
            arange=np.arange(len(direction))
            wu,wv=u[arange,winner],v[arange,winner]
            normal=((1-wu-wv)[:,None]*vertex_normals[winner,0]+wu[:,None]*vertex_normals[winner,1]
                    +wv[:,None]*vertex_normals[winner,2])
            normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
            result_normal[start:start+len(direction)]=normal
            result_index[start:start+len(direction)]=np.where(np.isfinite(nearest),winner,-1)
    return (result,result_id,result_normal,result_index) if vertex_normals is not None else (result,result_id)


def inspect(root, pilot_sample=False):
    import OpenEXR
    from PIL import Image
    started=time.monotonic()
    manifest=json.loads((root/'raw-manifest.json').read_text())
    rows=[]
    selected=manifest['frames']
    if pilot_sample:
        selected=[f for f in selected if f['clip_kind']=='centre' and f['frame_in_clip']==20]
    for frame in selected:
        folder=root/frame['folder'];camera=json.loads((folder/'camera.json').read_text())
        objects=json.loads((folder/'instances.json').read_text())['instances']
        tri,ids,normal,colour=triangles(root,objects,attributes=True)
        yy,xx=np.meshgrid(np.arange(4,360,11),np.arange(4,640,11),indexing='ij')
        with Image.open(folder/'instance_left.png') as image:
            instance=np.asarray(image).astype(np.uint16)
        pixel_pairs=set(zip(yy.ravel().tolist(),xx.ravel().tolist()))
        for identifier in sorted(set(ids.tolist())):
            ys,xs=np.where(instance==identifier)
            if len(ys):
                sample=np.linspace(0,len(ys)-1,min(32,len(ys)),dtype=int)
                pixel_pairs.update(zip(ys[sample].tolist(),xs[sample].tolist()))
        pixels=np.asarray(sorted(pixel_pairs));yy,xx=pixels[:,0],pixels[:,1]
        k=np.asarray(camera['K']);t=np.asarray(camera['T_world_camera'])
        rays=np.stack([(xx.ravel()-k[0,2])/k[0,0],(yy.ravel()-k[1,2])/k[1,1],np.ones(xx.size)],axis=1)@t[:3,:3].T
        sides={}
        for side in ('left','right'):
            origin=t[:3,3]+(camera['baseline_m']*t[:3,0] if side=='right' else 0)
            expected,identity,expected_normal,index=trace(origin,rays,tri,ids,normal)
            with OpenEXR.File(str(folder/('depth_'+side+'.exr')),separate_channels=True) as file:
                depth=file.channels()['Z'].pixels[yy,xx].ravel().copy()
            valid=np.isfinite(expected)&np.isfinite(depth)
            error=np.abs(expected[valid]-depth[valid])
            p95=float(np.quantile(error,.95)) if len(error) else None
            missing=float(np.mean(np.isfinite(expected)!=np.isfinite(depth)))
            passed=len(error)>100 and p95<.02 and missing<=.01
            result=dict(status='PASS' if passed else 'FAIL',compared_rays=len(error),
                depth_abs_error_p95_m=p95,missing_disagreement=missing,
                mean_abs_error_m=float(error.mean()) if len(error) else None)
            if side=='left':
                matched=valid&(np.abs(expected-depth)<=.02)
                agreement=float(np.mean(identity[matched]==instance[yy[matched],xx[matched]])) if matched.any() else 0.
                observed_normal=np.load(folder/'normal_left.npy').astype(np.float32)[yy[matched],xx[matched]]
                dots=np.sum(observed_normal*expected_normal[matched],axis=-1)
                observed_albedo=np.load(folder/'albedo_left.npy').astype(np.float32)[yy[matched],xx[matched]]
                colour_error=np.max(np.abs(observed_albedo-colour[index[matched]]),axis=-1)
                normal_dot_p05=float(np.quantile(dots,.05)) if len(dots) else None
                albedo_error_p95=float(np.quantile(colour_error,.95)) if len(colour_error) else None
                result.update(instance_agreement=agreement,normal_dot_p05=normal_dot_p05,
                              albedo_max_channel_error_p95=albedo_error_p95)
                passed=passed and agreement>.98 and normal_dot_p05 is not None and normal_dot_p05>.95 and albedo_error_p95<.015
                result['status']='PASS' if passed else 'FAIL'
            sides[side]=result
        rows.append(dict(id=frame['id'],status='PASS' if all(s['status']=='PASS' for s in sides.values()) else 'FAIL',sides=sides))
    report=dict(status='PASS_GEOMETRIC_CANARY' if rows and all(r['status']=='PASS' for r in rows) else 'FAIL',
        scope='both-eye native depth, left instance/normal/albedo vs exported render mesh at sparse pixel centres; not sensor physical validation',frames=rows,
        selection='predeclared centre frame 20 per layout' if pilot_sample else 'all manifest frames',
        backend='TASK_NOT_GPU_SUITABLE_BOUNDED_CPU_GEOMETRIC_AUDIT',wall_s=time.monotonic()-started,
        directory_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    (root/'geometric-canary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-sample',action='store_true')
    a=p.parse_args();r=inspect(a.output,a.pilot_sample);print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['status']=='PASS_GEOMETRIC_CANARY' else 1)
