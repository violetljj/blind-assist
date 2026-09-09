"""Evaluator-only scoring of paired synthetic rotational observations."""
import argparse
from pathlib import Path
import numpy as np
from body_query_collection_labels import read,write,sha
from tof_jitter_geometry import support


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run=a.root/'jitter-v1';receipt=read(run/'receipt.json')
    assert sha(run/'observations.npz')==receipt['observations_sha256']
    assert sha(a.root/'packets-v1/receipt.json')==receipt['parent_receipt_sha256']
    side=read(a.root/'packets-v1/evaluator-sidecar.json')
    with np.load(run/'observations.npz',allow_pickle=False) as archive:
        z={name:archive[name] for name in archive.files}
    assert z['ranges'].shape==(2,25,3,600)
    rgb=[];truth=[];metadata=[];alerts=[]
    for task,adname in [('body-query-background-only-20260909','admission-visible-head-v2'),('body-query-fresh-size-20260909','admission-v1')]:
        root=Path('artifacts.local/work')/task;ad=root/'returned-full-capture-v1'/adname
        source=read(root/'inference-v1/result.json')
        assert sha(root/'inference-v1/predictions.npz')==source['predictions_sha256']
        assert sha(ad/'admission.json')==source['admission_sha256']
        admission=read(ad/'admission.json');assert sha(ad/'evaluator_truth.npz')==admission['truth_sha256']
        assert sha(ad/'frame_metadata.json')==admission['frame_metadata_sha256']
        saved=np.load(root/'inference-v1/predictions.npz',allow_pickle=False)
        assert np.array_equal(saved['BASE_alerts'],saved['retained_alerts'])
        rgb.extend((saved['JOINT_ranges']>=.5).reshape(-1,4))
        truth.extend((np.load(ad/'evaluator_truth.npz',allow_pickle=False)['counts'].reshape(-1,2,2,3).sum(-1)>=3).reshape(-1,4))
        metadata.extend(read(ad/'frame_metadata.json'));alerts.extend(saved['BASE_alerts'])
    rgb=np.array(rgb);truth=np.array(truth)
    assert len(metadata)==len(side)==len(rgb)==600
    for m,s in zip(metadata,side):assert (m['region_id'],m['capture_index'],m['admitted'])==(s['region_id'],s['capture_index'],s['admitted'])
    admitted=np.array([m['admitted'] for m in metadata],bool);near=np.array([m['endpoint']=='near' for m in metadata])&admitted
    assert admitted.sum()==530 and near.sum()==265
    # Recompute geometry exclusively from scalar range, validity and known pose.
    for arm in range(2):
        for t,pose in enumerate(receipt['poses']):
            for li in range(3):
                for i in range(600):
                    ob=support(z['ranges'][arm,t,li,i],z['valid'][arm,t,li,i],.1,pose['pitch_deg'] if arm else 0,pose['yaw_deg'] if arm else 0)
                    assert list(z['support'][arm,t,li,i])==[n in ob['supported'] for n in ('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')]
    fused=z['support']|rgb[None,None,None,:,:]
    assert np.all(fused|~rgb[None,None,None,:,:])
    count=lambda n,d:dict(numerator=int(n),denominator=int(d))
    result={}
    for li,law in enumerate(receipt['laws']):
        result[law]={}
        for arm,name in enumerate(receipt['arms']):
            f=fused[arm,:,li];s=z['support'][arm,:,li];v=z['valid'][arm,:,li]
            added=s&~rgb[None,:,:]
            result[law][name]=dict(valid=count(v[:,admitted].sum(),25*530),
                correct_near_support=count(s[:,near,2].sum(),25*265),any_time_near_support=count(s[:,near,2].any(0).sum(),265),
                fused_near_hit=count(f[:,near,2].sum(),25*265),
                added_false_events=int((added[:,admitted]&~truth[admitted]).sum()),
                fused_state_flips=count((f[1:,admitted]!=f[:-1,admitted]).any(-1).sum(),24*530),
                support_state_flips=count((s[1:,admitted]!=s[:-1,admitted]).any(-1).sum(),24*530))
    out=run/'evaluation.json';assert not out.exists()
    write(out,dict(status='PASS',results=result,receipt_sha256=sha(run/'receipt.json'),operator_sha256=sha(__file__),
        original_alert_parity=600,backend='CPU TASK_NOT_GPU_SUITABLE scalar scoring',
        scope='Consumed static-source rotation sensitivity; known exact pose; conservative bounds may abstain; no real walking or original alert jitter'))
    print(result)


if __name__=='__main__':main()
