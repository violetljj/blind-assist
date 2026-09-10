"""Independent scalar replay of MZ36 denominators, paired changes and frozen use."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit(task):
    output=task/'audit.json'
    if output.exists():raise FileExistsError(output)
    a=read(task/'admission-v1/result.json');s=read(task/'score-v1/result.json')
    assert a['status']==s['status']=='PASS'
    for stage in ('admission-v1','inference-v1','score-v1'):
        receipt=read(task/stage/'receipt.json');assert receipt['status']=='PASS'
        for p,h in receipt['inputs'].items():assert sha(p)==h,p
        for name,h in receipt['outputs'].items():assert sha(task/stage/name)==h,name
    frozen=read(task/'frozen-models.json')['frozen']
    inference=read(task/'inference-v1/receipt.json')
    actual={str(Path(p).resolve()):h for p,h in inference['frozen'].items()}
    assert all(actual[str(Path(p).resolve())]==h and sha(p)==h for p,h in frozen.items())
    with np.load(task/'score-v1/scored.npz',allow_pickle=False) as z:
        ids=z['frame_ids'];truth=z['truth'];known=z['known'];methods={n:z[n] for n in ('MZ5','MZ28','MZ30','MZ35')}
    assert len(ids)==400 and len(set(ids.tolist()))==400
    assert np.array_equal(known,np.repeat(known[:,:1],4,axis=1))
    assert known.all(1).sum()==a['admitted_frames']==s['admitted_frames']
    assert np.array_equal((~known).sum(0),s['unknown_by_query'])
    method_counts={};compared_bits=0
    for name,logits in methods.items():
        expected={k:[0]*4 for k in ('tp','fp','fn','tn','known','unknown')};exact=0
        for i in range(400):
            correct=[]
            for q in range(4):
                if not known[i,q]:expected['unknown'][q]+=1;continue
                assert np.isfinite(logits[i,q]);expected['known'][q]+=1;compared_bits+=1
                pred=bool(logits[i,q]>=0);target=bool(truth[i,q]);correct.append(pred==target)
                expected['tp' if pred and target else 'fp' if pred else 'fn' if target else 'tn'][q]+=1
            exact+=len(correct)==4 and all(correct)
        for key,value in expected.items():assert value==s['methods'][name][key],(name,key)
        assert exact==s['methods'][name]['exact_frames']
        method_counts[name]=dict(fp=sum(expected['fp']),fn=sum(expected['fn']),exact=exact)
    pairs={}
    for name in ('MZ28','MZ30','MZ35'):
        for base in ('MZ5','MZ28'):
            if name==base:continue
            counter={k:[0]*4 for k in ('tp_gained','tp_lost','fp_added','fp_removed')}
            for i,q in zip(*np.nonzero(known)):
                p=methods[name][i,q]>=0;b=methods[base][i,q]>=0
                if p==b:continue
                key=('tp_gained' if p else 'tp_lost') if truth[i,q] else ('fp_added' if p else 'fp_removed')
                counter[key][q]+=1
            assert counter==s['paired'][name][base]
            pairs[name+'/'+base]=counter
    selected={r['region_id']:r['selected_site_ids'] for r in read(task/'empty-review-v1.json')['regions']}
    positions=read(task/'empty-specs-v1/site-candidates.json')['regions']
    for region in positions:
        sites=[p for p in region['candidate_sites'] if p['site_id'] in selected[region['region_id']]]
        assert len(sites)==10
        for i,site in enumerate(sites):
            xy=np.array(site['camera_xy_m'])
            assert np.linalg.norm(np.array(region['prior_xy'])-xy,axis=1).min()>=6.-1e-7
            for other in sites[:i]:assert np.linalg.norm(np.array(other['camera_xy_m'])-xy)>=6.-1e-7
    assert len(a['groups'])==80 and sum(g['attempted_frames'] for g in a['groups'])==400
    for g in a['groups']:
        assert all(bool(known[i].all())==g['accepted'] for i in g['frame_indices'])
    result=dict(status='PASS',attempted_frames=400,scored_task_bits=compared_bits,
        checks=['All input/output hashes','Pre-capture frozen model bindings','Scalar four-method confusion and exact-frame replay',
                'All paired query changes','UNKNOWN and complete-group denominators','Selected XY separation from historical proposals'],
        methods=method_counts,paired=pairs,code_sha256=sha(__file__),training_steps=0,model_inference_frames=0,
        limits='Does not independently establish physical sensor fidelity or natural-scene labels')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);audit(p.parse_args().task.resolve())
