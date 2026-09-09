"""Engineering verification of approximation error in the frozen jitter study."""
import argparse
import math
from pathlib import Path
import numpy as np
from body_query_collection_labels import read,write,sha
from tof_exact_cone import directional_extrema
from tof_jitter_geometry import rotation
from contact_retina_spec import BODY_BOXES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();receipt=read(a.root/'jitter-v1/receipt.json')
    source=a.root/'jitter-v1/observations.npz';assert sha(source)==receipt['observations_sha256']
    side=read(a.root/'packets-v1/evaluator-sidecar.json')
    parent=read(a.root/'packets-v1/receipt.json');assert sha(a.root/'packets-v1/evaluator-sidecar.json')==parent['evaluator_sidecar_sha256']
    assert sha(a.root/'packets-v1/receipt.json')==receipt['parent_receipt_sha256']
    with np.load(source,allow_pickle=False) as z:values={n:z[n] for n in z.files}
    exact=np.zeros_like(values['support']);r=values['ranges'];valid=values['valid']
    slope=math.tan(math.radians(7.5))/math.sqrt(2)
    for arm in range(2):
        for t,pose in enumerate(receipt['poses']):
            extrema=np.array([directional_extrema(row,slope) for row in rotation(pose['pitch_deg'] if arm else 0,pose['yaw_deg'] if arm else 0)])
            lo=np.maximum(0,r[arm,t]-.1);hi=r[arm,t]+.1
            products=np.stack([lo[...,None]*extrema[:,0],lo[...,None]*extrema[:,1],hi[...,None]*extrema[:,0],hi[...,None]*extrema[:,1]])
            lower=products.min(0)-1e-12;upper=products.max(0)+1e-12
            lower[...,2]+=1.7;upper[...,2]+=1.7
            for k,(bl,bh) in enumerate(BODY_BOXES):
                for half,start in enumerate((bh[0],bh[0]+1.5)):
                    exact[arm,t,:,:,2*k+half]=valid[arm,t]&(lower[...,0]>=start)&(upper[...,0]<=start+1.5)&(lower[...,1]>=bl[1])&(upper[...,1]<=bh[1])&(lower[...,2]>=bl[2])&(upper[...,2]<=bh[2])
    admitted=np.array([s['admitted'] for s in side],bool);assert admitted.sum()==530
    old=values['support'];lost=old&~exact;recovered=exact&~old
    out=a.root/'exact-bounds-audit-v1';assert not out.exists();out.mkdir()
    np.savez_compressed(out/'support.npz',exact=exact)
    result=dict(status='PASS',observation_sha256=sha(source),operator_sha256=sha(__file__),geometry_sha256=sha(Path(__file__).with_name('tof_exact_cone.py')),
        backend='CPU TASK_NOT_GPU_SUITABLE vectorized tiny coordinate bounds; no depth processing',
        states=90000,admitted_endpoints=530,old_positive_removed=int(lost[:,:,:,admitted].sum()),
        added_support_by_arm_and_law=recovered[:,:,:,admitted].sum(axis=(1,3,4)).tolist(),
        identical_support_all_frames=bool(np.array_equal(exact,old)),
        scope='Engineering approximation audit on frozen observation packets; not independent scientific confirmation or real gait evidence')
    write(out/'result.json',result);print(result)


if __name__=='__main__':main()
