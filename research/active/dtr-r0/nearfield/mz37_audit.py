"""Independent saved-scalar MZ37 audit; never invokes its runner or calibrator."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

OLD_COHORTS = ['DEV','clean','clean_wrong','stress','stress_wrong',
                'relation10000','relation10000_wrong','distance5000','distance5000_wrong']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def metric(score,truth,known):
    pred=score>=0;complete=known.all(1)
    return dict(attempted_frames=len(score),known=known.sum(0).tolist(),unknown=(~known).sum(0).tolist(),
        tp=(pred&truth&known).sum(0).tolist(),fp=(pred&~truth&known).sum(0).tolist(),
        fn=(~pred&truth&known).sum(0).tolist(),tn=(~pred&~truth&known).sum(0).tolist(),
        complete_frames=int(complete.sum()),exact_frames=int((complete&((pred==truth).all(1))).sum()))


def restoration_inputs(baseline,oldscore,rgb,tof,support,negative_confidence,known):
    assert negative_confidence.dtype==np.float32
    positive=(np.float32(1)-negative_confidence).astype(np.float32)
    eligible=(baseline<0)&(oldscore<0)&support&((rgb>=0)!=(tof>=0))&known
    branch=np.where(rgb>=0,rgb,tof)
    assert np.isfinite(positive[known]).all() and ((positive[known]>=0)&(positive[known]<=1)).all()
    assert (branch[eligible]>=0).all()
    return eligible,positive,branch


def calibrate_independent(confidence,eligible,truth):
    thresholds=[];proof=[];disabled=float(np.nextafter(np.float64(1),np.float64(np.inf)))
    for q in range(4):
        values=confidence[:,q][eligible[:,q]].astype(np.float64)
        candidates=sorted(set([.5,disabled]+[float(v) for v in values if v>=.5]))
        rows=[]
        for threshold in candidates:
            chosen=eligible[:,q]&(confidence[:,q].astype(np.float64)>=threshold)
            rows.append(dict(threshold=threshold,tp=int((chosen&truth[:,q]).sum()),fp=int((chosen&~truth[:,q]).sum())))
        feasible=[r for r in rows if r['fp']==0]
        best=max(feasible,key=lambda r:(r['tp'],r['threshold']))
        if best['tp']==0:best=next(r for r in rows if r['threshold']==disabled)
        thresholds.append(best['threshold']);proof.append(dict(query=q,eligible=int(eligible[:,q].sum()),
            candidates=len(candidates),optimum=best,disabled=best['threshold']>1,all_candidates=rows))
    return np.array(thresholds,np.float64),proof


def assert_confidence(negative,rgb,logits,known):
    probability=np.exp(-np.logaddexp(0.,-logits.astype(np.float64)))
    independent=np.where(rgb<0,probability,1-probability)
    np.testing.assert_allclose(negative[known],independent[known],atol=1e-7,rtol=1e-6)
    return float(np.max(np.abs(negative[known]-independent[known]))) if known.any() else 0.


def source_cohorts(root):
    work=root/'artifacts.local/work';old=work/'mz35-responsibility-convergence-20260910/run-v1'
    fresh=work/'mz36-new-source-20260910';paths=[old/'receipt.json',old/'predictions.npz',old/'audit.json',
        fresh/'inference-v1/receipt.json',fresh/'inference-v1/predictions.npz',fresh/'score-v1/receipt.json',fresh/'score-v1/scored.npz']
    assert read(old/'audit.json')['status']=='PASS'
    for folder,filename in [(old,'predictions.npz'),(fresh/'inference-v1','predictions.npz'),(fresh/'score-v1','scored.npz')]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS'
        assert sha(folder/filename)==receipt['outputs'][filename]
    p=load(old/'predictions.npz');cohorts={}
    for name in OLD_COHORTS:
        a={k:p[name+'/'+k] for k in ['baseline','truth','rgb','tof','support']}
        a.update(oldscore=p[name+'/candidate'],negative_confidence=p[name+'/confidence'],selector_logits=p[name+'/selector_logits'],
                 known=np.ones_like(a['truth'],bool),ids=p[name+'/global_ids'])
        cohorts[name]=a
    raw=load(fresh/'inference-v1/predictions.npz');scored=load(fresh/'score-v1/scored.npz')
    allids=scored['frame_ids'];lookup={str(v):i for i,v in enumerate(allids)}
    assert len(lookup)==len(allids)==400
    indices=np.array([lookup[str(v)] for v in raw['frame_ids']]);assert len(indices)==len(np.unique(indices))==380
    np.testing.assert_array_equal(indices,np.flatnonzero(scored['known'].all(1)))
    assert int(scored['known'].sum())==1520 and not scored['known'][~scored['known'].all(1)].any()
    for key in ['MZ5','MZ35']:np.testing.assert_array_equal(raw[key],scored[key][indices])
    a=dict(baseline=scored['MZ5'],oldscore=scored['MZ35'],truth=scored['truth'],known=scored['known'],ids=allids)
    for out,key in [('rgb','rgb'),('tof','tof'),('negative_confidence','MZ35/confidence'),('selector_logits','MZ35/selector_logits')]:
        value=np.full((400,4),np.nan,dtype=raw[key].dtype);value[indices]=raw[key];a[out]=value
    a['support']=np.zeros((400,4),bool);a['support'][indices]=raw['original_support'];cohorts['MZ36']=a
    return cohorts,{str(p):sha(p) for p in paths}


def check_metric(report,expected):
    # The frozen runner uses the MZ36 scalar schema. This independent function
    # deliberately does not import that evaluator or the MZ37 runner.
    assert report==expected,(report,expected)


def main(root,run):
    assert not (run/'audit.json').exists()
    receipt=read(run/'receipt.json');assert receipt['status']=='PASS' and receipt['training_steps']==0
    for name,digest in receipt['outputs'].items():assert sha(run/name)==digest,name
    for path,digest in receipt['inputs'].items():assert sha(path)==digest,path
    for name,digest in receipt.get('code_sha256',{}).items():assert sha(Path(__file__).with_name(name))==digest,name
    inputs_hashes=dict(receipt['inputs']);cohorts,source_hashes=source_cohorts(root)
    result=read(run/'result.json');saved=load(run/'predictions.npz');cut=np.load(run/'cutoff.npy')
    calibration=read(run/'calibration.json')
    freeze=read(run/'calibration-freeze.json')
    assert freeze['status']=='FROZEN_BEFORE_NONCALIBRATION_REPLAY' and freeze['source']=='oldDEV1000 only'
    assert freeze['training_steps']==freeze['model_inference_frames']==0
    assert freeze['cutoff_sha256']==sha(run/'cutoff.npy')
    for path,digest in freeze['inputs'].items():
        assert 'mz36-new-source' not in path and inputs_hashes[path]==digest
    confidence_error={}
    for name,a in cohorts.items():
        confidence_error[name]=assert_confidence(a['negative_confidence'],a['rgb'],a['selector_logits'],a['known'])
        a['eligible'],a['positive_confidence'],a['positive_branch']=restoration_inputs(a['baseline'],a['oldscore'],a['rgb'],a['tof'],a['support'],a['negative_confidence'],a['known'])
    dev=cohorts['DEV'];expected_cut,proof=calibrate_independent(dev['positive_confidence'],dev['eligible'],dev['truth'])
    np.testing.assert_array_equal(cut,expected_cut)
    assert cut.dtype==np.float64
    np.testing.assert_array_equal(freeze['cutoff'],cut)
    np.testing.assert_array_equal(result['cutoff'],cut)
    expected_calibration=[]
    for q,p in enumerate(proof):
        expected_calibration.append(dict(threshold=p['optimum']['threshold'],
            eligible_tp=int((dev['eligible'][:,q]&dev['truth'][:,q]).sum()),
            eligible_fp=int((dev['eligible'][:,q]&~dev['truth'][:,q]).sum()),
            recovered_tp=p['optimum']['tp'],added_fp=p['optimum']['fp'],candidates=p['candidates'],disabled=p['disabled']))
    assert calibration==expected_calibration
    assert result['calibration']==expected_calibration and result['training_steps']==result['model_inference_frames']==0
    reports={};task_bits=0;known_bits=0
    assert set(result['metrics'])==set(cohorts)
    for name,a in cohorts.items():
        added=a['eligible']&(a['positive_confidence'].astype(np.float64)>=cut)
        candidate=np.where(added,a['positive_branch'],a['oldscore'])
        arrays={k:a[k] for k in ['oldscore','truth','known','eligible','positive_confidence','positive_branch','baseline','support','rgb','tof']}
        arrays.update(candidate=candidate,added=added,negative_confidence=a['negative_confidence'],frame_ids=a['ids'])
        for field,value in arrays.items():np.testing.assert_array_equal(saved[name+'/'+field],value,err_msg=name+'/'+field)
        assert not (added&~a['known']).any() and not ((candidate<0)&(a['oldscore']>=0)).any()
        np.testing.assert_array_equal(candidate[~added],a['oldscore'][~added])
        before=metric(a['oldscore'],a['truth'],a['known']);after=metric(candidate,a['truth'],a['known'])
        changes=dict(added_tp=(added&a['truth']&a['known']).sum(0).tolist(),added_fp=(added&~a['truth']&a['known']).sum(0).tolist(),
            lost_tp=((candidate<0)&(a['oldscore']>=0)&a['truth']&a['known']).sum(0).tolist())
        r=result['metrics'][name];check_metric(r['before'],before);check_metric(r['after'],after)
        for key,value in changes.items():assert r[key]==value,(name,key)
        assert r['prior_positives_preserved']==bool(not ((a['oldscore']>=0)&(candidate<0)&a['known']).any())
        assert r['eligible_tp']==(a['eligible']&a['truth']&a['known']).sum(0).tolist()
        assert r['eligible_fp']==(a['eligible']&~a['truth']&a['known']).sum(0).tolist()
        reports[name]=dict(before=before,after=after,**changes,positive_values_retained=bool(np.array_equal(candidate[a['oldscore']>=0],a['oldscore'][a['oldscore']>=0])))
        task_bits+=candidate.size;known_bits+=int(a['known'].sum())
    assert task_bits==32800 and known_bits==32720
    normal=['DEV','clean','stress','relation10000','distance5000','MZ36']
    additions_preserved=True
    for name,a in cohorts.items():
        take=(a['oldscore']>=0)&(a['baseline']<0)&a['known']
        additions_preserved &= bool(np.array_equal(saved[name+'/candidate'][take],a['oldscore'][take]))
    gates=dict(restored_noncalibration_tp=sum(sum(reports[n]['added_tp']) for n in normal if n!='DEV')>0,
        no_added_fp=all(sum(reports[n]['added_fp'])==0 for n in normal),
        all_mz35_positives_retained=all(r['positive_values_retained'] for r in reports.values()),
        prior_additions_preserved=additions_preserved)
    gates['useful_effect']=all(gates.values());assert result['gates']==gates
    controls={n:dict(added_tp=reports[n+'_wrong']['added_tp'],added_fp=reports[n+'_wrong']['added_fp']) for n in normal if n not in ['DEV','MZ36']}
    assert result['wrong_controls']==controls
    for path,digest in {**inputs_hashes,**source_hashes}.items():assert sha(path)==digest,path
    audit=dict(status='PASS',code_sha256=sha(__file__),receipt_sha256=sha(run/'receipt.json'),inputs=inputs_hashes,
        source_inputs=source_hashes,task_bits=task_bits,known_bits=known_bits,mz36_unknown_bits=80,
        gates=gates,noncalibration_restored_tp=sum(sum(reports[n]['added_tp']) for n in normal if n!='DEV'),
        normal_added_fp=sum(sum(reports[n]['added_fp']) for n in normal),
        calibration_optimum=proof,calibration_file_sha256=sha(run/'calibration.json'),confidence_reconstruction_max_error=confidence_error,
        metrics=reports,training_steps=0,neural_inference_frames=0,
        limits=['Calibration uses only oldDEV; MZ36 remains consumed Development.',
            'Independent NumPy scalar replay; no runner calibration/compose functions are imported.',
            'Frozen upstream neural outputs are inherited through bound receipts; no backbone/selector rerun.'])
    write(run/'audit.json',audit);print('PASS',task_bits,'attempted bits,',known_bits,'known bits')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    try:main(a.root,a.run)
    except Exception:
        if not (a.run/'audit.json').exists():write(a.run/'audit.json',dict(status='FAIL',code_sha256=sha(__file__),error=traceback.format_exc()))
        raise
