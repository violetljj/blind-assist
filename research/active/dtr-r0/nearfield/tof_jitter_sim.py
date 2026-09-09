"""Paired fixed-origin rotational sensitivity; no walking or sensor claims."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from body_query_collection_labels import read, write, sha
from tof_simulated_packets import disturbances, measure, LAWS
from tof_jitter_geometry import trajectory, footprint, support


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    receipt=read(a.source/'receipt.json')
    assert sha(a.source/'evaluator-sidecar.json')==receipt['evaluator_sidecar_sha256']
    rows=read(a.source/'evaluator-sidecar.json');assert len(rows)==600
    assert torch.cuda.is_available();torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter();poses=trajectory()
    footprints=[footprint('cuda',0,0)]+[footprint('cuda',v['pitch_deg'],v['yaw_deg']) for v in poses]
    noise,dropout=disturbances(25*600,seed=29)
    noise=noise.reshape(25,600);dropout=dropout.reshape(25,600)
    ranges=np.full((2,25,3,600),np.nan);valid=np.zeros_like(ranges,dtype=bool)
    flags=np.zeros((*ranges.shape,4),dtype=bool)
    for begin in range(0,600,8):
        batch=rows[begin:begin+8];depth=[]
        for row in batch:
            path=Path(row['native_path']);assert sha(path)==row['native_sha256']
            depth.append(np.load(path,allow_pickle=False))
        native=torch.from_numpy(np.stack(depth)).cuda()
        static=measure(native,footprints[0])
        for step,pose in enumerate(poses):
            moved=measure(native,footprints[step+1])
            for arm,values in enumerate((static,moved)):
                for li,law in enumerate(LAWS):
                    distance,known=values[law]
                    d=distance.cpu().numpy()+noise[step,begin:begin+len(batch)]
                    v=known.cpu().numpy()&~dropout[step,begin:begin+len(batch)]
                    ranges[arm,step,li,begin:begin+len(batch)]=np.where(v,d,np.nan)
                    valid[arm,step,li,begin:begin+len(batch)]=v
                    for j,(value,ok) in enumerate(zip(d,v)):
                        ob=support(float(value),bool(ok),.1,pose['pitch_deg'] if arm else 0,pose['yaw_deg'] if arm else 0)
                        flags[arm,step,li,begin+j]=[name in ob['supported'] for name in ('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')]
    torch.cuda.synchronize();a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'observations.npz',ranges=ranges,valid=valid,support=flags)
    write(a.output/'receipt.json',dict(status='PASS',frames=600,steps=25,arms=['stationary','rotational_jitter'],laws=list(LAWS),poses=poses,
      observations_sha256=sha(a.output/'observations.npz'),parent_receipt_sha256=sha(a.source/'receipt.json'),
      operator_sha256={name:sha(Path(__file__).with_name(name)) for name in ('tof_jitter_sim.py','tof_jitter_geometry.py','tof_simulated_packets.py','body_query_tof_coverage.py','body_query_collection_labels.py','contact_retina_spec.py')},
      device=torch.cuda.get_device_name(),backend='CUDA geometry plus CPU scalar bounds',seconds=time.perf_counter()-start,
      uncertainty_m=.1,noise_seed=29,scope='Fixed-origin rotation, known pose, frozen RGB; no actual walking or hardware evidence'))
    print('PASS',time.perf_counter()-start)


if __name__=='__main__':main()
