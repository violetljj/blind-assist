"""Count frozen TRAIN responsibility coverage before changing supervision."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from mz5_ensemble_readout import read, write, sha, load_npz

ORDER=['BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR']

def main(root,output):
    output.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    work=root/'artifacts.local/work';run=work/'mz30-branch-responsibility-20260910/run-v1';inputs={}
    def bind(path,expected=None):
        digest=sha(path);assert expected is None or digest==expected,str(path);inputs[str(path)]=digest
    receipt=read(run/'receipt.json');assert receipt['status']=='PASS';bind(run/'receipt.json')
    bind(run/'branches.npz',receipt['outputs']['branches.npz'])
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    for folder,name in [(oldpath,'observations.npz'),(newpath,'selected.json')]:
        bind(folder/'receipt.json');bind(folder/name,read(folder/'receipt.json')['files'][name])
    sourcepath=work/'body-query-5000-20260909/dataset-v1/index.json'
    sourcereceipt=work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json'
    bind(sourcereceipt);bind(sourcepath,read(sourcereceipt)['source_index_sha256'])
    protocol=Path(__file__).with_name('MZ31_RESPONSIBILITY_COVERAGE_PROTOCOL_20260910.md');bind(protocol)
    data=load_npz(run/'branches.npz');train=data['train_ids'];assert len(train)==7562 and np.array_equal(train,np.unique(data['batches']))
    old=load_npz(oldpath/'observations.npz');newrows=read(newpath/'selected.json');source=read(sourcepath)['frames']
    lookup=np.full(14200,-1,int);lookup[data['global_ids']]=np.arange(len(data['global_ids']));ti=lookup[train];assert (ti>=0).all()
    for gid in train:
        if gid<3500:assert old['role'][gid]=='TRAIN_ONLY'
        elif gid>=3700:assert gid<11200
    rgb=data['normal/rgb'][ti];tof=data['normal/tof'][ti];truth=data['normal/truth'][ti]
    base=data['normal/baseline'][ti]>=0;support=data['normal/support'][ti]
    disagreement=(rgb>=0)!=(tof>=0);original=disagreement&base&~support
    np.testing.assert_array_equal(original,data['normal/eligible'][ti])
    target=(rgb>=0)==truth;tof_correct=(tof>=0)==truth
    assert np.all(target[disagreement]!=tof_correct[disagreement])
    assert int(original.sum())==168 and int((original&target).sum())==120
    def identity(gid):
        if gid<3500:
            row=source[int(old['old_index'][gid])]
            return dict(site=row['site_id'],group=row['group_id'],family=row['family'],source='old5000')
        if gid>=3700:
            row=newrows[gid-3700];assert row['cache_index']==gid-3700
            return dict(site=row['site'],group=row['group'],family=row['family'],source=row['dataset'])
        return dict(site=None,group=None,family=None,source='MZ6_sequence')
    write(output/'start.json',dict(status='STARTED',inputs=inputs,code_sha256=sha(Path(__file__)),training_steps=0,new_inference_frames=0,train_frames=len(train)))
    rows=[]
    for i,q in np.argwhere(disagreement):
        i=int(i);q=int(q);gid=int(train[i]);positive=bool(base[i,q]);supported=bool(support[i,q])
        rows.append(dict(global_id=gid,query=q,query_name=ORDER[q],**identity(gid),rgb=float(rgb[i,q]),tof=float(tof[i,q]),truth=bool(truth[i,q]),
            target_rgb_correct=bool(target[i,q]),baseline_positive=positive,geometry_supported=supported,original_eligible=bool(original[i,q]),
            partition=('positive' if positive else 'negative')+('_supported' if supported else '_unsupported')))
    assert len({(r['global_id'],r['query']) for r in rows})==len(rows)
    assert all(r['global_id'] in set(train.tolist()) for r in rows)
    def summarize(rr):
        result={}
        for q in range(4):
            result[ORDER[q]]={}
            for flag,label in [(True,'RGB_correct'),(False,'ToF_correct')]:
                sub=[r for r in rr if r['query']==q and r['target_rgb_correct']==flag]
                sites={r['site'] for r in sub if r['site'] is not None};groups={r['group'] for r in sub if r['group'] is not None}
                result[ORDER[q]][label]=dict(examples=len(sub),explicit_sites=len(sites),explicit_groups=len(groups),unknown_site_examples=sum(r['site'] is None for r in sub),
                    source_counts=dict(Counter(r['source'] for r in sub)),family_counts=dict(Counter(r['family'] or 'UNKNOWN' for r in sub)),
                    site_counts=dict(sorted(Counter(r['site'] or 'UNKNOWN' for r in sub).items())))
        return result
    all_summary=summarize(rows);original_summary=summarize([r for r in rows if r['original_eligible']])
    partitions={p:summarize([r for r in rows if r['partition']==p]) for p in ['positive_supported','positive_unsupported','negative_supported','negative_unsupported']}
    gates={}
    for q in ORDER:
        for label in ['RGB_correct','ToF_correct']:
            full=all_summary[q][label];assert sum(partitions[p][q][label]['examples'] for p in partitions)==full['examples']
            assert original_summary[q][label]==partitions['positive_unsupported'][q][label]
            gates[q+'/'+label]=full['examples']>=20 and full['explicit_sites']>=5
    result=dict(status='PASS',training_steps=0,new_inference_frames=0,train_frames=len(train),original_examples=int(original.sum()),all_disagreement_examples=len(rows),
        original=original_summary,all_disagreements=all_summary,partitions=partitions,coverage_gates=gates,coverage_pass=all(gates.values()),rows=rows,
        limits=['Original and expanded branch predictions are in-sample.','Expanded coverage does not establish that support-conditioned responsibility transfers.','No model fit, task-effect or fresh-source result is produced.'])
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',backend='NumPy CPU saved TRAIN scalar and identity reductions',training_steps=0,new_inference_frames=0,seconds=time.perf_counter()-tick,
        inputs=inputs,code_sha256=sha(Path(__file__)),outputs={name:sha(output/name) for name in ['start.json','result.json']}))
    print('PASS',len(rows),'disagreements',gates,flush=True)
    for q in ORDER:print(q,[(k,v['examples'],v['explicit_sites']) for k,v in all_summary[q].items()],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
