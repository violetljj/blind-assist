"""Resume only the frozen Street description after source-transfer failure."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from cnh_r1_readout_data import load_street,load_frame
from cnh_r1_readout_run import arms,synthesize,evaluate,sha,write,PROTOCOL

def run(output,worker_capture,part='all',worker_scores=None,worker_receipt=None):
    original=json.loads((output/'result.json').read_text())
    if original['status']!='ALLEY_COMPLETE':raise ValueError('Resume only completed alley, unfinished Street')
    for name,value in original['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=value:raise ValueError('Frozen source changed: '+name)
    dest=output/('street-main-result.json' if part=='main' else 'street-result.json')
    if dest.exists():raise FileExistsError('No repeated Street scoring')
    started=time.monotonic()
    prior=json.loads((output/'failure.json').read_text())['wall_s']
    deadline=started+max(0,1800-prior)
    street=load_street(require_depth=False,worker_capture=worker_capture)
    if part=='main':
        for key in ('rows','frames'):street[key]=street[key][:960]
        for key in ('labels','train','dev','histogram','clip_ids','steps'):street[key]=street[key][:960]
    if worker_scores is None:
        raw,_,_,_=synthesize(street,load_frame,deadline)
        bias=np.load(output/'train-bias.npy',allow_pickle=False)
        scores,full,_=arms(raw,street,bias)
        parity_frames=len(raw)
    else:
        if part!='all':raise ValueError('Merge must cover all Street')
        receipt=json.loads(Path(worker_receipt).read_text())
        if receipt['parity_frames']!=960 or receipt['scores_sha256']!=sha(worker_scores) or receipt['bias_sha256']!=sha(output/'train-bias.npy'):
            raise ValueError('Worker parity or payload identity differs')
        if any(receipt['source_hashes'].get(n)!=v for n,v in original['source_hashes'].items()):
            raise ValueError('Worker frozen code identity differs')
        main_receipt=json.loads((output/'street-main-result.json').read_text())
        if main_receipt['parity_frames']!=960 or main_receipt['scores_sha256']!=sha(output/'street-main-scores.npz'):
            raise ValueError('Main parity or payload differs')
        scores={};full={}
        with np.load(output/'street-main-scores.npz',allow_pickle=False) as first,np.load(worker_scores,allow_pickle=False) as second:
            keys=np.concatenate([first['frame_key'],second['frame_key']]).astype(str)
            y=np.concatenate([first['labels'],second['labels']])
            if list(keys)!=[r['frame_key'] for r in street['rows']] or not np.array_equal(y,street['labels']):
                raise ValueError('Street merged identity/labels differ')
            for name in original['results']:
                scores[name]=np.concatenate([first[name],second[name]])
                if name.startswith('R3_'):
                    k=int(name.split('_K')[1].split('_')[0]);full[name]=street['steps']>=k-1
        parity_frames=1920
    labels=street['labels'];dev=np.ones(len(labels),bool)
    results={name:evaluate(labels,s,~dev,dev,threshold=original['results'][name]['threshold'],full=full.get(name))
             for name,s in scores.items()}
    scorepath=output/('street-main-scores.npz' if part=='main' else 'street-scores.npz')
    np.savez_compressed(scorepath,labels=labels,frame_key=[r['frame_key'] for r in street['rows']],**scores)
    report=dict(status='COMPLETE_STREET_CONTINUATION',frames=len(labels),results=results,
                identity=street['identity'],parity_frames=parity_frames,wall_s=time.monotonic()-started,
                source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                alley_result_sha256=sha(output/'result.json'),bias_sha256=sha(output/'train-bias.npy'),
                scores_sha256=sha(scorepath),worker_scores_sha256=sha(worker_scores) if worker_scores else None,
                reason='Existing source restoration; no alley re-run, refit or selection; original failure preserved')
    write(dest,report)
    if part=='all':
        write(output/'completion.json',dict(status='COMPLETE',alley='result.json',street='street-result.json',
              decision=original['decision'],alley_plus_current_wall_s=prior+report['wall_s']))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--worker-capture',type=Path)
    p.add_argument('--part',choices=('all','main'),default='all');p.add_argument('--worker-scores',type=Path);p.add_argument('--worker-receipt',type=Path)
    a=p.parse_args();result=run(a.output,a.worker_capture,a.part,a.worker_scores,a.worker_receipt);print(json.dumps(dict(status=result['status'],frames=result['frames'])))
