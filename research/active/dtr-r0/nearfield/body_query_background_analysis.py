"""Four-metric old/new identity-matched audit from saved probabilities; no inference."""
import argparse
from pathlib import Path
import numpy as np
from body_query_collection_labels import read, write, sha


def ranges(counts):
    cells=counts.astype(np.float64).reshape(-1,2,2,3,4)
    coefficients=np.zeros(cells.shape[:3]+(3,));coefficients[...,0]=1
    for j in range(3):
        nxt=np.zeros_like(coefficients)
        for total in range(3):
            for k in range(total+1):nxt[...,total]+=coefficients[...,total-k]*cells[...,j,k]
        coefficients=nxt
    return 1-coefficients.sum(-1)


def score(prob,truth,rows,ids,parity):
    predicted=ranges(prob)>=.5
    true=truth.reshape(-1,2,2,3).sum(-1)>=3
    exact=(predicted[:,1]==true[:,1]).all(1)
    near=[i for i in ids if rows[i]['endpoint']=='near']
    pairs={rows[i]['pair_id'] for i in ids}
    positives=truth[ids,6:9]>0
    hits=(1-prob[ids,6:9,0])>=.5
    return dict(near_hit=dict(numerator=int((hits&positives).sum()),denominator=int(positives.sum())),
                near_to_far_confusion=dict(numerator=int(predicted[near,1,1].sum()),denominator=len(near)),
                pair_correct=dict(numerator=sum(all(exact[i] for i in ids if rows[i]['pair_id']==p) for p in pairs),denominator=len(pairs)),
                original_alert_parity=parity)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task','historical','distance','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    ad=a.task/'returned-full-capture-v1/admission-visible-head-v2';run=a.task/'inference-v1'
    receipt=read(run/'result.json');assert sha(run/'predictions.npz')==receipt['predictions_sha256']
    assert sha(ad/'admission.json')==receipt['admission_sha256']
    old_receipt=read(a.historical/'receipt.json');assert sha(a.historical/'predictions.npz')==old_receipt['predictions_sha256']
    native_receipt=read(a.distance/'receipt.json')
    for file,key in [('frame_metadata.json','frame_metadata_sha256'),('evaluator_truth.npz','evaluator_truth_sha256')]:
        assert sha(a.distance/file)==native_receipt[key]
    rows=read(ad/'frame_metadata.json');oldrows=read(a.distance/'frame_metadata.json')
    current=np.load(run/'predictions.npz',allow_pickle=False);old=np.load(a.historical/'predictions.npz',allow_pickle=False)
    truth=np.load(ad/'evaluator_truth.npz',allow_pickle=False)['counts']
    oldtruth=np.load(a.distance/'evaluator_truth.npz',allow_pickle=False)['native_counts'].reshape(-1,12)
    assert len(rows)==120
    for row in rows:
        orig=oldrows[row['original_index']]
        assert (orig['pair_id'],orig['endpoint'])==(row['original_pair_id'],row['endpoint'])
    results={};max_errors={}
    parity=(current['BASE_alerts']==current['retained_alerts']).all(1)
    for scope in ('all',*sorted({r['region_id'] for r in rows})):
        ids=[i for i,r in enumerate(rows) if r['admitted'] and (scope=='all' or r['region_id']==scope)]
        oldids=[rows[i]['original_index'] for i in ids];results[scope]={}
        for method in ('BASE','JOINT'):
            newscore=score(current[method+'_counts'],truth,rows,ids,dict(numerator=int(parity[ids].sum()),denominator=len(ids)))
            oldscore=score(old[method+'_counts'],oldtruth,oldrows,oldids,dict(status='RETAINED_BASE_ALERT_INTERFACE_BY_CONSTRUCTION_NO_OLD_FORWARD_RERUN'))
            results[scope][method]=dict(original_background=oldscore,new_background=newscore)
            if scope=='all':
                assert newscore==receipt['metrics']['background'][method]
                max_errors[method]=float(abs(ranges(current[method+'_counts'])-current[method+'_ranges']).max())
                assert np.array_equal(ranges(current[method+'_counts'])>=.5,current[method+'_ranges']>=.5)
    paired=results['all']['JOINT'];oldpair=paired['original_background']['pair_correct'];newpair=paired['new_background']['pair_correct']
    oldrate=oldpair['numerator']/oldpair['denominator'];newrate=newpair['numerator']/newpair['denominator']
    decision=('BACKGROUND_RETENTION_SUPPORTED_IN_CONTROLLED_COHORT' if newrate>=.8 and parity.all()
              else 'SUBSTANTIAL_BACKGROUND_SENSITIVITY' if oldrate>=.8 and oldrate-newrate>=.2
              else 'BACKGROUND_ATTRIBUTION_INCONCLUSIVE')
    write(a.output,dict(status='PASS',decision=decision,metrics=results,all_frame_alert_parity=dict(numerator=int(parity.sum()),denominator=len(rows)),
          max_probability_reconstruction_error=max_errors,historical_prediction_sha256=sha(a.historical/'predictions.npz'),
          new_prediction_sha256=sha(run/'predictions.npz'),source_sha256=sha(__file__),promotion=False,
          backend='CPU scalar/small-array receipt arithmetic; no model inference'))
    print(decision);print(results['all'])


if __name__=='__main__':main()
