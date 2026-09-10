"""Read-only angular/native boundary distances for saved availability tails."""
import argparse
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, sha, write
from mz9_contributors import _cached_geometry


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    work=root/'artifacts.local/work';tail=work/'mz23-availability-20260910/tail-v1'
    run=work/'mz23-availability-20260910/run-v1';cache=work/'mz15-shared-support-20260910/cache-v1'
    inputs={}
    def bind(path, expected=None):
        digest=sha(path);assert expected is None or digest==expected,str(path);inputs[str(path)]=digest
    tr=read(tail/'receipt.json');assert tr['status']=='PASS'
    for name,digest in tr['outputs'].items():bind(tail/name,digest)
    errors=read(tail/'result.json')['errors'];assert len(errors)==8
    rr=read(run/'receipt.json');bind(run/'availability.pt',rr['outputs']['availability.pt'])
    cr=read(cache/'receipt.json')
    for name in ['selected.json','evaluator.npz']:bind(cache/name,cr['files'][name])
    rows=read(cache/'selected.json')
    with np.load(cache/'evaluator.npz') as a:known=a['known']
    grid=torch.load(run/'availability.pt',map_location='cpu',weights_only=True)['grid'].numpy()
    # align_corners=False within the original224ROI at x208,y68.
    pixels=grid*112+np.array([319.5,179.5])
    feature_pixels=((grid+1)*28-1)/2
    zs,cs=np.meshgrid(np.arange(64),np.arange(49),indexing='ij')
    angular=np.stack([(zs//8)*7+cs//7,(zs%8)*7+cs%7],-1)
    keep,rays,factor,zone,cell=_cached_geometry('cpu')
    keep,factor,zone,cell=[v.numpy() for v in [keep,factor,zone,cell]]
    mapped=np.bincount(zone*49+cell,minlength=3136)
    assert (mapped>0).all()
    native_xy=np.stack([keep%640,keep//640],-1)
    details=[]
    for error in errors:
        row=[r for r in rows if r['dataset']==error['cohort']][error['frame']]
        assert row['role']=='DEV_ONLY'
        k=known[row['cache_index']];z,c=error['zone'],error['cell']
        assert not k[z,c] and not error['known']
        p=pixels[z,c];g=angular[z,c]
        kk=np.argwhere(k);delta=angular[k]-g
        euclidean=np.linalg.norm(delta,axis=-1);manhattan=np.abs(delta).sum(-1);chebyshev=np.abs(delta).max(-1)
        best=kk[np.argmin(euclidean)];bz,bc=best
        distances=np.linalg.norm(pixels[k]-p,axis=-1);bp=kk[np.argmin(distances)]
        bind(row['native'],row['native_sha'])
        depth=np.load(row['native']).reshape(-1)[keep].astype(float)
        good=np.isfinite(depth)&(depth>0)&(depth<100)&(depth*factor<=4)
        native_known=np.bincount(zone*49+cell,weights=good.astype(int),minlength=3136)>0
        np.testing.assert_array_equal(native_known.reshape(64,49),k)
        actual_distances=np.linalg.norm(native_xy[good]-p,axis=-1)
        nearest_index=np.flatnonzero(good)[np.argmin(actual_distances)]
        # Pixel-center distance between any raster pixel in the unknown cell and
        # any valid native pixel: this describes the cell footprint, not winner XYZ.
        cellpixels=native_xy[(zone==z)&(cell==c)]
        gap=float(np.sqrt(((cellpixels[:,None,:]-native_xy[good][None,:,:])**2).sum(-1)).min())
        details.append(dict(error,site=row['site'],family=row['family'],group=row['group'],source_index=row['index'],
            winner_angular_rc=g.tolist(),winner_native_center_xy=p.tolist(),winner_feature_center_xy=feature_pixels[z,c].tolist(),
            nearest_known_cell=dict(zone=int(bz),cell=int(bc),angular_rc=angular[bz,bc].tolist(),
                native_center_xy=pixels[bz,bc].tolist(),angular_delta_rc=(angular[bz,bc]-g).tolist(),
                euclidean_cells=float(euclidean.min()),manhattan_cells=int(manhattan.min()),chebyshev_cells=int(chebyshev.min()),
                native_center_gap_px=float(distances.min()),native_nearest_cell_zone=int(bp[0]),native_nearest_cell=int(bp[1]),
                feature_center_gap=float(np.linalg.norm(feature_pixels[bz,bc]-feature_pixels[z,c]))),
            nearest_valid_native_pixel=dict(xy=native_xy[nearest_index].tolist(),center_gap_px=float(actual_distances.min()),
                axial_m=float(depth[nearest_index]),radial_m=float(depth[nearest_index]*factor[nearest_index])),
            winner_cell_pixel_count=len(cellpixels),cell_footprint_to_valid_pixel_gap_px=gap))
    x=np.unique(feature_pixels[...,0]);y=np.unique(feature_pixels[...,1])
    result=dict(details=details,geometry=dict(native_pixels_in_crop=len(keep),angular_cells=3136,
        pixels_per_cell_min=int(mapped.min()),pixels_per_cell_max=int(mapped.max()),empty_cells=int((mapped==0).sum()),
        angular_grid=[56,56],learned_feature_grid=[28,28],distinct_sample_x=len(x),distinct_sample_y=len(y),
        adjacent_feature_step_min=float(np.diff(x).min()),adjacent_feature_step_max=float(np.diff(x).max()),
        feature_sample_min=float(feature_pixels.min()),feature_sample_max=float(feature_pixels.max())),
        inputs=inputs,code_sha256=sha(Path(__file__)),
        scope='Consumed Development diagnostic only. No model inference, fitting, threshold change, mask dilation or operating-point proposal.',
        interpretation='Tail winner maximizes availability among MZ20 cutoff-passing candidates; it need not be MZ20 raw-score winner. Known means native<=4m support, not obstacle/CLEAR.',
        limit='Angular/native proximity and shared feature sampling are descriptive associations. They do not establish resolution as the causal failure or predict that increasing resolution will fix it.')
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',inputs=inputs,code_sha256=sha(Path(__file__)),outputs={'result.json':sha(output/'result.json')},backend='CPU saved geometry and native-coordinate reductions',inference_frames=0,training_steps=0))
    for d in details:print(d['cohort'],d['frame'],d['query'],d['survives'],d['nearest_known_cell'],d['nearest_valid_native_pixel'],d['cell_footprint_to_valid_pixel_gap_px'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.output)
