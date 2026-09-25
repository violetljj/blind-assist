"""Resume only the frozen Street description after source-transfer failure."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from cnh_r1_readout_data import load_street,load_frame
from cnh_r1_readout_run import arms,synthesize,evaluate,sha,write,PROTOCOL

def run(output,worker_capture):
    original=json.loads((output/'result.json').read_text())
    if original['status']!='ALLEY_COMPLETE':raise ValueError('Resume only completed alley, unfinished Street')
    for name,value in original['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=value:raise ValueError('Frozen source changed: '+name)
    dest=output/'street-result.json'
    if dest.exists():raise FileExistsError('No repeated Street scoring')
    started=time.monotonic()
    prior=json.loads((output/'failure.json').read_text())['wall_s']
    deadline=started+max(0,1800-prior)
    street=load_street(require_depth=True,worker_capture=worker_capture)
    raw,_,_,_=synthesize(street,load_frame,deadline)
    bias=np.load(output/'train-bias.npy',allow_pickle=False)
    scores,full,_=arms(raw,street,bias)
    labels=street['labels'];dev=np.ones(len(labels),bool)
    results={name:evaluate(labels,s,~dev,dev,threshold=original['results'][name]['threshold'],full=full.get(name))
             for name,s in scores.items()}
    np.savez_compressed(output/'street-scores.npz',labels=labels,frame_key=[r['frame_key'] for r in street['rows']],**scores)
    report=dict(status='COMPLETE_STREET_CONTINUATION',frames=len(labels),results=results,
                identity=street['identity'],parity_frames=len(raw),wall_s=time.monotonic()-started,
                source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                alley_result_sha256=sha(output/'result.json'),bias_sha256=sha(output/'train-bias.npy'),
                scores_sha256=sha(output/'street-scores.npz'),
                reason='Existing source restoration; no alley re-run, refit or selection; original failure preserved')
    write(dest,report)
    write(output/'completion.json',dict(status='COMPLETE',alley='result.json',street='street-result.json',
          decision=original['decision'],scientific_wall_s=prior+report['wall_s']))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--worker-capture',required=True,type=Path)
    a=p.parse_args();result=run(a.output,a.worker_capture);print(json.dumps(dict(status=result['status'],frames=result['frames'])))
