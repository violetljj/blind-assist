"""Consumed source diagnostics: retain every frozen ray and gate; never admit a source."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from cnh_route_source_geometry_audit import audit, digest, statistics, zone_coverage


def edge_samples(depth, yy, xx, radius=1, jump_m=.075):
    """Native axial jump diagnostic, independent of mesh residual; no exclusion."""
    center=depth[yy,xx]; valid=np.isfinite(center)&(center>0)
    edge=np.zeros(len(yy),dtype=bool)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            near=depth[np.clip(yy+dy,0,depth.shape[0]-1),np.clip(xx+dx,0,depth.shape[1]-1)]
            nv=np.isfinite(near)&(near>0)
            edge|=(valid!=nv)|(valid&nv&(np.abs(center-near)>jump_m))
    return edge


def group_metrics(mask, residual, native, mesh, weights):
    common=mask&native&mesh
    values=residual[common]
    denominator=float(weights[mask].sum())
    return dict(rays=int(mask.sum()),common=int(common.sum()),
        radial_absolute=statistics(np.abs(values)),
        signed_median_m=float(np.median(values)) if len(values) else None,
        over_75mm=int((np.abs(values)>.075).sum()),
        missing=int((mask&native&~mesh).sum()),extra=int((mask&mesh&~native).sum()),
        missing_fraction=float(weights[mask&native&~mesh].sum()/denominator) if denominator else None,
        extra_fraction=float(weights[mask&mesh&~native].sum()/denominator) if denominator else None)


class Diagnostic:
    def __init__(self,output):
        self.output=output;self.frames=[];self.arrays=[]

    def __call__(self,**d):
        from PIL import Image,ImageDraw,ImageEnhance
        row=d['row'];folder=d['folder'];yy=d['yy'];xx=d['xx']
        observed=d['observed'];predicted=d['predicted'];norm=np.linalg.norm(d['optical'],axis=-1)
        _,native,mesh=zone_coverage(observed,predicted,norm,d['weights'],d['zones'])
        residual=(predicted.astype(float)-observed)*norm
        background=d['sampled_ids']==0;common=native&mesh&background
        depth=np.load(folder/'depth_left.transport.npy',allow_pickle=False)
        edge1=edge_samples(depth,yy,xx);edge3=edge_samples(depth,yy,xx,3)
        angle=np.rad2deg(np.arccos(1/norm))
        groups={}
        for label,selection in [('all',background),('edge_1px',background&edge1),
                ('interior_1px',background&~edge1),('edge_3px',background&edge3),('interior_3px',background&~edge3)]:
            groups[label]=group_metrics(selection,residual,native,mesh,d['weights'])
        for low,high in ((0,5),(5,10),(10,15),(15,20),(20,25),(25,35)):
            groups[f'angle_{low}_{high}']=group_metrics(background&(angle>=low)&(angle<high),residual,native,mesh,d['weights'])
        zones=[]
        for size in (8,4):
            zone=d['zones'] if size==8 else d['zones']//8//2*4+d['zones']%8//2
            for identifier in range(size*size):
                selected=zone==identifier
                zones.append(dict(size=size,zone=identifier,
                    background=group_metrics(selected&background,residual,native,mesh,d['weights']),
                    all_pixels=group_metrics(selected,residual,native,mesh,d['weights'])))
        # Compare alternative depth interpretations on the SAME common set.
        depth_hypotheses={
            'both_axial':statistics(np.abs(predicted[common]-observed[common])*norm[common]),
            'native_radial_mesh_axial':statistics(np.abs(predicted[common]*norm[common]-observed[common])),
            'native_axial_mesh_radial':statistics(np.abs(predicted[common]-observed[common]*norm[common]))}
        attribution=[]
        for surface in sorted(set(d['predicted_surface'][mesh].tolist())):
            selected=d['predicted_surface']==surface
            meta=d['surfaces'][surface] if surface>=0 else {}
            attribution.append(dict(surface_index=surface,mesh=meta.get('mesh',{}).get('asset_path'),
                actor=meta.get('actor_path'),component=meta.get('component_path'),instance=meta.get('instance_index'),
                nanite=meta.get('mesh',{}).get('nanite'),
                metrics=group_metrics(selected&background,residual,native,mesh,d['weights'])))
        worst=[]
        for index in np.flatnonzero(common)[np.argsort(np.abs(residual[common]))[-20:][::-1]]:
            surface=int(d['predicted_surface'][index]);meta=d['surfaces'][surface] if surface>=0 else {}
            worst.append(dict(x=int(xx[index]),y=int(yy[index]),radial_signed_m=float(residual[index]),
                native_axial_m=float(observed[index]),mesh_axial_m=float(predicted[index]),
                angle_deg=float(angle[index]),edge_1px=bool(edge1[index]),edge_3px=bool(edge3[index]),
                surface_index=surface,mesh=meta.get('mesh',{}).get('asset_path')))
        # Shift diagnostics sample recorded depth only, no change to gate or rays.
        shift=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                shifted=depth[np.clip(yy+dy,0,359),np.clip(xx+dx,0,639)]
                selected=common&~edge3&np.isfinite(shifted)&(shifted>0)
                shift.append(dict(dx=dx,dy=dy,radial_absolute=statistics(np.abs(predicted[selected]-shifted[selected])*norm[selected])))
        np.savez_compressed(self.output/(row['id']+'.npz'),yy=yy,xx=xx,observed=observed,predicted=predicted,
            norm=norm,weights=d['weights'],zones=d['zones'],sampled_ids=d['sampled_ids'],
            surface=d['predicted_surface'],edge1=edge1,edge3=edge3,angle=angle)
        (self.output/(row['id']+'-surfaces.json')).write_text(json.dumps(d['surfaces'],indent=2),encoding='utf-8')
        image=Image.open(folder/'left.png').convert('RGB')
        image=ImageEnhance.Brightness(image).enhance(2.0) # Preview only; raw PNG stays intact.
        draw=ImageDraw.Draw(image)
        for mask,color in ((native&~mesh,(255,50,70)),(mesh&~native,(0,230,255)),(common&(np.abs(residual)>.075),(255,220,0))):
            for x,y in zip(xx[mask],yy[mask]):draw.rectangle((int(x)-1,int(y)-1,int(x)+1,int(y)+1),fill=color)
        k=np.asarray(d['camera']['K']);edge=np.tan(np.deg2rad(22.5))
        for v in np.linspace(-edge,edge,9):
            x=int(v*k[0,0]+k[0,2]);y=int(v*k[1,1]+k[1,2])
            draw.line((x,int(k[1,2]-edge*k[1,1]),x,int(k[1,2]+edge*k[1,1])),fill=(120,120,120))
            draw.line((int(k[0,2]-edge*k[0,0]),y,int(k[0,2]+edge*k[0,0]),y),fill=(120,120,120))
        draw.rectangle((0,0,640,32),fill=(0,0,0));draw.text((5,3),row['id']+' '+row['layout_id']+' '+row['clip_id'],fill='white')
        draw.text((5,17),'Red: missing | Cyan: extra | Yellow: >75mm | RGB preview brightness x2',fill='white')
        image.save(self.output/(row['id']+'-overlay.png'))
        self.frames.append(dict(**row,groups=groups,zones=zones,depth_hypotheses=depth_hypotheses,
            pixel_shift_diagnostic=shift,predicted_surface_attribution=attribution,worst_background_rays=worst))
        self.arrays.append(dict(layout_id=row['layout_id'],background=background,native=native,mesh=mesh,
            residual=residual,weights=d['weights'],edge1=edge1,edge3=edge3,angle=angle,observed=observed,predicted=predicted,norm=norm))

    def summary(self):
        result=[]
        for layout in dict.fromkeys(a['layout_id'] for a in self.arrays):
            selected=[a for a in self.arrays if a['layout_id']==layout]
            a={key:np.concatenate([x[key] for x in selected]) for key in selected[0] if key!='layout_id'}
            groups={}
            for label,mask in [('all',a['background']),('edge_1px',a['background']&a['edge1']),('interior_1px',a['background']&~a['edge1']),('edge_3px',a['background']&a['edge3']),('interior_3px',a['background']&~a['edge3'])]:
                groups[label]=group_metrics(mask,a['residual'],a['native'],a['mesh'],a['weights'])
            angle_rows=[]
            for low,high in ((0,5),(5,10),(10,15),(15,20),(20,25),(25,35)):
                angle_rows.append(dict(low=low,high=high,**group_metrics(a['background']&(a['angle']>=low)&(a['angle']<high),a['residual'],a['native'],a['mesh'],a['weights'])))
            common=a['background']&a['native']&a['mesh']
            result.append(dict(layout_id=layout,groups=groups,angle_groups=angle_rows,
                depth_hypotheses={
                    'both_axial':statistics(np.abs(a['residual'][common])),
                    'native_radial_mesh_axial':statistics(np.abs(a['predicted'][common]*a['norm'][common]-a['observed'][common])),
                    'native_axial_mesh_radial':statistics(np.abs(a['predicted'][common]-a['observed'][common]*a['norm'][common]))}))
        return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene-depth-materials',action='store_true');args=parser.parse_args()
    root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve();output=args.output.resolve()
    if not output.is_relative_to(root) or output==root or (output.exists() and any(output.iterdir())):raise ValueError('Fresh empty artifact subdirectory required')
    output.mkdir(parents=True,exist_ok=True)
    observer=Diagnostic(output);gates=audit(args.capture,diagnostic=observer,scene_depth_materials=args.scene_depth_materials)
    result=dict(status='DIAGNOSTIC_ONLY',benchmark_eligible=False,
        edge_definition='Native depth 8-neighborhood axial jump >75mm or validity boundary; radii 1px and 3px. Diagnostic only, no ray excluded from frozen gates.',
        attribution_authority='Exported mesh ray winner only; not native rendered object identity. Missing-ray native ownership UNKNOWN.',
        layouts=observer.summary(),frames=observer.frames,script_sha256=digest(__file__))
    result['depth_parity_gates' if args.scene_depth_materials else 'original_gates']=gates
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],layouts=result['layouts']),allow_nan=False))


if __name__=='__main__':main()
