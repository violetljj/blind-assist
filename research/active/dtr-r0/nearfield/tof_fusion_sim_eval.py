"""Frozen RGB, naive ToF threshold and positive-only body association contrast."""
import argparse
from pathlib import Path
import numpy as np
from body_query_collection_labels import read,write,sha
from tof_body_support import support,add_positive_support

FIELDS={'sample_id','range_m','valid','diagonal_fov_deg','range_uncertainty_m','delta_ms','source'}


def predict(rgb,packet):
    if set(packet)!=FIELDS or packet['source']!='SIMULATED_HYPOTHESIS':raise ValueError('Unexpected model-visible fields')
    fresh=packet['valid'] and 0<=packet['delta_ms']<=100
    naive=[False]*4
    if fresh:naive[2 if packet['range_m']<=1.63 else 3]=True
    observation=support(packet['range_m'],fresh,packet['diagonal_fov_deg'],packet['range_uncertainty_m'])
    fused=add_positive_support(rgb,observation)
    assert all(not old or new for old,new in zip(rgb,fused))
    return dict(RGB=list(rgb),TOF_THRESHOLD=naive,FUSION=fused)


def metric(pred,true,rows,ids):
    near=[i for i in ids if rows[i]['endpoint']=='near'];far=[i for i in ids if rows[i]['endpoint']=='far']
    exact=(pred[:,2:]==true[:,2:]).all(1);pairs={rows[i]['pair_id'] for i in ids}
    count=lambda n,d:dict(numerator=int(n),denominator=int(d))
    return dict(HEAD_near_hit=count(pred[near,2].sum(),len(near)),wrong_far_on_near=count(pred[near,3].sum(),len(near)),
      strict_pair_correct=count(sum(all(exact[i] for i in ids if rows[i]['pair_id']==p) for p in pairs),len(pairs)),
      false_HEAD_near_on_far=count(pred[far,2].sum(),len(far)),BODY_false_event_frames=count(pred[ids,:2].any(1).sum(),len(ids)),
      exact_BODY_HEAD_range_frames=count((pred[ids]==true[ids]).all(1).sum(),len(ids)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    receipt=read(a.packets/'receipt.json');assert receipt['status']=='PASS'
    sidepath=a.packets/'evaluator-sidecar.json';assert sha(sidepath)==receipt['evaluator_sidecar_sha256'];side=read(sidepath)
    work=Path('artifacts.local/work');sources={};ids_by_sid={}
    for cohort,task,adname in [('original_fixture','body-query-background-only-20260909','admission-visible-head-v2'),('simplified_fixture','body-query-fresh-size-20260909','admission-v1')]:
        root=work/task;ad=root/'returned-full-capture-v1'/adname;ar=read(ad/'admission.json');reference=read(root/'inference-v1/result.json')
        assert sha(ad/'admission.json')==receipt['cohort_bindings'][cohort]['admission_sha256']==reference['admission_sha256']
        assert sha(ad/'frame_metadata.json')==ar['frame_metadata_sha256']
        assert sha(ad/'evaluator_truth.npz')==ar['truth_sha256']
        assert sha(root/'inference-v1/predictions.npz')==reference['predictions_sha256']
        rows=read(ad/'frame_metadata.json');saved=np.load(root/'inference-v1/predictions.npz',allow_pickle=False)
        assert np.array_equal(saved['BASE_alerts'],saved['retained_alerts'])
        # Saved evaluator truth is kept out of predict().
        truth=np.load(ad/'evaluator_truth.npz',allow_pickle=False)['counts'].reshape(-1,2,2,3).sum(-1)>=3
        sources[cohort]=dict(rows=rows,rgb=(saved['JOINT_ranges']>=.5).reshape(-1,4),truth=truth.reshape(-1,4),alerts=saved['BASE_alerts'])
    for record in side:
        cohort=record['cohort'];i=record['sample_index'];row=sources[cohort]['rows'][i]
        assert (row['region_id'],row['capture_index'],row['admitted'])==(record['region_id'],record['capture_index'],record['admitted'])
        ids_by_sid[record['sample_id']]=(cohort,i)
    results={};candidates=[]
    for filename,digest in receipt['packet_files'].items():
        path=a.packets/filename;assert sha(path)==digest
        import json
        packets=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()];assert len(packets)==600
        assert [p['sample_id'] for p in packets]==[r['sample_id'] for r in side]
        output={c:{m:np.zeros_like(s['rgb']) for m in ('RGB','TOF_THRESHOLD','FUSION')} for c,s in sources.items()}
        valid={c:np.zeros(len(s['rows']),dtype=bool) for c,s in sources.items()}
        for packet in packets:
            cohort,i=ids_by_sid[packet['sample_id']]
            values=predict(sources[cohort]['rgb'][i],packet)
            for method,value in values.items():output[cohort][method][i]=value
            valid[cohort][i]=packet['valid']
        file_result={};arrays={}
        for cohort,s in sources.items():
            for method,values in output[cohort].items():arrays[cohort+'_'+method]=values
            arrays[cohort+'_retained_alerts']=s['alerts']
            for arm in sorted({r['arm'] for r in s['rows']}):
                ids=[i for i,r in enumerate(s['rows']) if r['admitted'] and r['arm']==arm]
                metrics={m:metric(v,s['truth'],s['rows'],ids) for m,v in output[cohort].items()}
                added=output[cohort]['FUSION']&~output[cohort]['RGB']
                bad=int((added[ids]&~s['truth'][ids]).sum());good=int((added[ids]&s['truth'][ids]).sum())
                key=cohort+'/'+arm
                file_result[key]=dict(metrics=metrics,valid_observations=dict(numerator=int(valid[cohort][ids].sum()),denominator=len(ids)),added_true_events=good,added_false_events=bad,original_alert_parity=dict(numerator=len(ids),denominator=len(ids)))
                if filename.startswith('fov15-'):
                    b=metrics['RGB']['HEAD_near_hit'];f=metrics['FUSION']['HEAD_near_hit'];candidates.append((f['numerator']-b['numerator'])/b['denominator']>=.1 and bad==0)
        out=a.output/(path.stem+'.npz');np.savez_compressed(out,**arrays)
        results[path.stem]=dict(cohorts=file_result,predictions_sha256=sha(out))
    assert len(candidates)==9
    write(a.output/'result.json',dict(status='PASS',decision='IDEAL_FUSION_NEAR_SUPPORT_CANDIDATE' if all(candidates) else 'IDEAL_FUSION_PARTIAL_OR_UNSUPPORTED',results=results,
      packet_receipt_sha256=sha(a.packets/'receipt.json'),source_sha256=sha(__file__),training_steps=0,model_inference_frames=0,all_original_alert_parity=600,
      unavailable_metrics=['HEAD false alarm rate: all source endpoints HEAD-positive','Trigger-distance jitter: no timed approach trajectories'],
      scope='Hypothetical return laws only; source and RGB already consumed; no actual VL53L1X result or promotion'))
    print(read(a.output/'result.json')['decision'])
    for name,value in results.items():
        print(name,{c:{m:d['strict_pair_correct'] for m,d in v['metrics'].items()} for c,v in value['cohorts'].items()})


if __name__=='__main__':main()
