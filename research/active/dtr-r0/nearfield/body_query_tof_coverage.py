"""CUDA geometric footprint audit, explicitly not a VL53L1X return simulator."""
import argparse
import math
from pathlib import Path
import time
import numpy as np
import torch
from body_query_collection_labels import read,write,sha


def footprints(device):
    yy,xx=torch.meshgrid(torch.arange(360,device=device),torch.arange(640,device=device),indexing='ij')
    f=320/math.tan(math.radians(50));rx=(xx-319.5)/f;ry=(yy-179.5)/f
    norm=(1+rx.square()+ry.square()).sqrt();weight=norm.pow(-3)
    result={}
    for diagonal in (27,15):
        limit=math.tan(math.radians(diagonal/2))/math.sqrt(2)
        mask=(rx.abs()<=limit)&(ry.abs()<=limit)
        result[diagonal]=(mask,weight[mask],norm[mask])
    return result


def describe(values):
    x=np.array(values,dtype=np.float64)
    return dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(1);assert torch.cuda.is_available()
    start=time.perf_counter();footprint=footprints('cuda');records=[];bindings={}
    jobs=[('original_fixture','body-query-background-only-20260909','admission-visible-head-v2','adjustable_cross_member'),('simplified_fixture','body-query-fresh-size-20260909','admission-v1','head_bar')]
    for cohort,task,adname,target in jobs:
        run=a.artifacts/'work'/task/'full-capture-v1';ad=run/adname;receipt=read(ad/'admission.json')
        assert sha(ad/'frame_metadata.json')==receipt['frame_metadata_sha256']
        bindings[cohort]=dict(admission_sha256=sha(ad/'admission.json'),rows_sha256=sha(ad/'frame_metadata.json'))
        rows=read(ad/'frame_metadata.json')
        for begin in range(0,len(rows),16):
            native=[];isolated=[]
            for row in rows[begin:begin+16]:
                root=run/row['region_id']/'capture';i=row['capture_index']
                n=root/f'evaluator/native/{i:04d}.npy';t=root/f'evaluator/isolated/{target}/{i:04d}.npy'
                assert sha(n)==row['native_sha256'] and sha(t)==row['isolated_sha256']
                native.append(np.load(n,allow_pickle=False));isolated.append(np.load(t,allow_pickle=False))
            n=torch.from_numpy(np.stack(native)).cuda();t=torch.from_numpy(np.stack(isolated)).cuda()
            for diag,(mask,w,norm) in footprint.items():
                scene=n[:,mask];target_depth=t[:,mask];valid=torch.isfinite(scene)&(scene>0)&(scene<100)
                target_valid=torch.isfinite(target_depth)&(target_depth>0)&(target_depth<100)
                visible=valid&target_valid&((scene-target_depth).abs()<=.03)
                geometric_range=scene*norm;inrange=valid&(geometric_range<=4.)
                total=w.sum()
                fraction=lambda m:(m*w).sum(1)/total
                target_fraction=fraction(visible);near_target=fraction(visible&inrange);missing=fraction(~valid)
                non_target=fraction(valid&~visible);beyond=fraction(valid&~inrange)
                for j,row in enumerate(rows[begin:begin+16]):
                    def distribution(m):
                        v=geometric_range[j,m]
                        return None if not len(v) else dict(min=float(v.min()),median=float(v.median()),max=float(v.max()))
                    coverage=float(target_fraction[j]);unknown=float(missing[j]);nt=float(non_target[j])
                    records.append(dict(cohort=cohort,region=row['region_id'],pair_id=row['pair_id'],arm=row['arm'],endpoint=row['endpoint'],admitted=row['admitted'],capture_index=row['capture_index'],diagonal_fov_deg=diag,
                          footprint_pixels=int(mask.sum()),visible_target_fraction=coverage,visible_target_within4m_fraction=float(near_target[j]),unknown_fraction=unknown,non_target_fraction=nt,beyond4m_fraction=float(beyond[j]),
                          target_pixels=int(visible[j].sum()),mixed=bool(coverage>0 and nt>0),
                          dominated_geometric_screen=bool(near_target[j]>=.95 and missing[j]<=.01),
                          target_geometric_range_m=distribution(visible[j]),background_geometric_range_m=distribution(valid[j]&~visible[j]),
                          sensor_return_m=None,sensor_validity='UNMODELED'))
    assert len(records)==1200
    summary={}
    for cohort in ('original_fixture','simplified_fixture'):
        for diag in (27,15):
            for scope in ('all','admitted'):
                sub=[r for r in records if r['cohort']==cohort and r['diagonal_fov_deg']==diag and (scope=='all' or r['admitted'])]
                summary[f'{cohort}/{diag}/{scope}']=dict(frames=len(sub),mixed=sum(r['mixed'] for r in sub),dominated=sum(r['dominated_geometric_screen'] for r in sub),target_absent=sum(r['target_pixels']==0 for r in sub),coverage=describe([r['visible_target_fraction'] for r in sub]),unknown=describe([r['unknown_fraction'] for r in sub]))
    write(a.output/'rows.json',records)
    write(a.output/'result.json',dict(status='PASS',scope='IDEAL_GEOMETRIC_FOOTPRINT_ONLY_NO_SENSOR_RETURN',summary=summary,bindings=bindings,rows_sha256=sha(a.output/'rows.json'),source_sha256=sha(__file__),device=torch.cuda.get_device_name(0),seconds=time.perf_counter()-start,frames=600,footprints_per_frame=2,model_inference_frames=0,training_steps=0))
    print(summary)


if __name__=='__main__':main()
