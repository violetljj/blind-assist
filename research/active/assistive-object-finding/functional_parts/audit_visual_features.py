"""Independent raw dense-to-feature check; never opens evaluator truth."""
import argparse
import json
import math
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def rebuild(dense,point,box,mode,matrix):
    if mode=='gap':
        v=np.sum(dense,axis=(1,2))/196.
    else:
        x=max(0.,min(13.,(point[0]-box[0])*14/(box[2]-box[0])-.5))
        y=max(0.,min(13.,(point[1]-box[1])*14/(box[3]-box[1])-.5))
        ix,iy=math.floor(x),math.floor(y);jx,jy=min(13,ix+1),min(13,iy+1)
        ax,ay=x-ix,y-iy
        top=(1-ax)*dense[:,iy,ix]+ax*dense[:,iy,jx]
        bottom=(1-ax)*dense[:,jy,ix]+ax*dense[:,jy,jx]
        v=(1-ay)*top+ay*bottom
    norm=max(math.sqrt(sum(float(a)*float(a) for a in v)),1e-12)
    return (v/norm)@matrix


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--development-inference',type=Path,required=True)
    p.add_argument('--fresh-inference',type=Path,required=True)
    args=p.parse_args();start=time.perf_counter();cpu=time.process_time();root=args.root
    matrix=np.load(root/'training/projection.npy',allow_pickle=False)
    expected=np.random.default_rng(20261015).standard_normal((384,64))/8
    if not np.array_equal(matrix,expected):raise ValueError('Frozen Gaussian projection mismatch')
    development={r['sample_id']:r for r in rows(args.development_inference)}
    fresh={r['sample_id']:r for r in rows(args.fresh_inference)}
    checks=[]
    for split in ('fit','cal','fresh'):
        for family in ('candidates','fixed'):
            for source in ('dino','head'):
                for mode in ('point','gap'):
                    variant=f'{source}_{mode}'
                    if split=='fresh':
                        path=root/'learned_predictions'/f'{family}_{variant}_features.npz';obs=fresh
                    else:
                        path=root/'training'/f'{family}_{variant}_{split}_features_targets.npz';obs=development
                    data=np.load(path,allow_pickle=False)
                    # Deterministic endpoints inspect two different images without metric selection.
                    for i in (0,len(data['sample_ids'])-1):
                        sid=str(data['sample_ids'][i]);row=obs[sid]
                        for j,point in enumerate(data['points'][i]):
                            view=row['views'][row[family][j//2]['name']]
                            dense=np.load(view[source+'_dense_path'],allow_pickle=False).astype(np.float64)
                            if dense.shape!=(384,14,14):raise ValueError('Dense shape')
                            actual=rebuild(dense,point,view['bbox_xyxy'],mode,matrix)
                            error=float(np.max(np.abs(actual-data['X'][i,j,129:])))
                            checks.append(dict(split=split,family=family,variant=variant,sample_id=sid,point_index=j,max_abs_error=error))
                            if error>1e-12:raise ValueError(f'Visual vector mismatch: {checks[-1]}')
                            if time.process_time()-cpu>300:raise TimeoutError('Audit CPU budget')
    report=dict(status='PASS',fresh_GT_read=False,projection_exact=True,checks=len(checks),
        max_abs_error=max(r['max_abs_error'] for r in checks),details=checks,
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start)
    output=root/'evaluation';output.mkdir(exist_ok=True)
    (output/'independent_visual_feature_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'}))


if __name__=='__main__':main()
