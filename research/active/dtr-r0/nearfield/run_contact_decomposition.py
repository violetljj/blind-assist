"""Frozen selected-model diagnostic. No fit or threshold selection."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from contact_boundary_data import queries,camera_boxes
from contact_decomposition import decompose,crossing,focus
from exact_contact_labels import metric_targets
from metric_contact_model import MetricContactModel
from query_occupancy_data import read,write,sha,new_stage_directory

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
ART=REPO/'artifacts.local/evidence'
BASE=ART/'ba-contact-boundary-20260923'
MODELS=dict(binary=ART/'ba-metric-contact-20260923',exact=ART/'ba-exact-contact-20260923')
EXTRACT=ART/'ba-contact-decomposition-20260923'
DEST=ART/'ba-contact-decomposition-20260923-v3'
PREP=ART/'ba-query-occupancy-20260922-prepared'
CAPTURE=ART/'ba-query-occupancy-20260922-capture'


def entries():
    rows=[(BASE/n,'evaluator') for n in ('features.npy','normalization.npz','cohort.json','training-targets.npz','evaluation-targets.npz','feature-receipt.json')]
    rows += [(PREP/'observations/identities.json','observation'),(CAPTURE/'evaluator/geometry.json','evaluator')]
    for root in MODELS.values():
        rows += [(root/n,'evaluator') for n in ('metric.pt','metric-selection.json','prediction-seal.json','result.json','metric-train-predictions.npz','metric-evaluation-predictions.npz','continuous-train.npy','continuous-evaluation.npy')]
    rows += [(EXTRACT/n,'evaluator') for n in ('source-seal.json','component-seal.json','queries.npy','backend-binary.json','backend-exact.json','binary-train-components.npz','binary-evaluation-components.npz','exact-train-components.npz','exact-evaluation-components.npz')]
    return rows


def spec():
    s=dict(schema='blindassist-asset-run-v1',id='contact-decomposition-20260923-v3',route='ue-query-occupancy',
        question='Are frozen contact errors dominated by probability blocking, inaccurate internal positions or width inconsistency?',
        evaluator='research/active/dtr-r0/nearfield/run_contact_decomposition.py',
        evidence_boundary='Consumed read-only diagnostic; selected models and cutoffs unchanged; no causal or promotion claim',
        reuse=dict(mode='diagnostic',query='Frozen binary and exact-supervision contact checkpoints features geometry probability position width'),
        inputs=[dict(alias='input'+str(i),path=str(p),role=r,purpose='frozen-contact-component-diagnostic') for i,(p,r) in enumerate(entries())],
        outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],result_output='result',
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    p=DEST.with_name(DEST.name+'-run.json');write(p,s);print(p)


def record_backend(model,features,query,name):
    from tools.research_backend import BackendCandidate,Workload,select_backend,torch_observation
    x=torch.tensor(features[:32]);q=torch.tensor(query)
    def probe():
        with torch.inference_mode():return model.components(x,q)[0]
    cpu=BackendCandidate('cpu','cpu',probe,lambda output:torch_observation(model=model),lambda:None)
    return select_backend(Workload.MODEL_INFERENCE,cpu=cpu,gpu=None,cpu_reason='FROZEN_PROTOCOL_CPU_ONLY',record_path=DEST/('backend-'+name+'.json'),warmups=1,repeats=3)


def run():
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    new_stage_directory(DEST);started=time.perf_counter();torch.set_num_threads(4)
    inputs={str(p):sha(p) for p,_ in entries()}
    names=['run_contact_decomposition.py','contact_decomposition.py','CONTACT_DECOMPOSITION_PROTOCOL_20260923.md',
           'metric_contact_model.py','contact_boundary_model.py','contact_boundary_data.py','exact_contact_labels.py']
    code={n:sha(HERE/n) for n in names};snap=DEST/'source-snapshot';snap.mkdir()
    for n in names:(snap/n).write_bytes((HERE/n).read_bytes())
    write(DEST/'source-seal.json',dict(inputs=inputs,code=code))
    assert sha(BASE/'features.npy')==read(BASE/'feature-receipt.json')['features_sha256']
    norm=np.load(BASE/'normalization.npz');features=((np.load(BASE/'features.npy')-norm['mean'])/norm['std']).astype(np.float32)
    indices={k:np.array(v,np.int64) for k,v in read(BASE/'cohort.json')['indices'].items() if k in ('train','evaluation')}
    query=queries()['width_curve'];widths=query[:51,0];anchor=int(np.flatnonzero(np.isclose(widths,.6,atol=1e-7,rtol=0))[0])
    np.save(DEST/'queries.npy',query);files={};selections={};backends={}
    for name,root in MODELS.items():
        seal=read(root/'prediction-seal.json');selection=read(root/'metric-selection.json');selections[name]=selection
        assert sha(root/'metric.pt')==seal['checkpoint_sha256']
        assert sha(root/'metric-selection.json')==seal['selection_sha256']
        assert read(root/'result.json')['actual_device']=='cpu','Frozen backend contract changed'
        # Resume the completed extraction byte-for-byte after fixing only the
        # right-censor equivalence assertion; no second checkpoint extraction.
        extract_seal=read(EXTRACT/'component-seal.json')
        assert extract_seal['checkpoint_hashes'][name]==sha(root/'metric.pt')
        assert extract_seal['cutoffs'][name]==selection['threshold']
        assert sha(EXTRACT/'queries.npy')==extract_seal['queries_sha256']
        np.testing.assert_array_equal(np.load(EXTRACT/'queries.npy'),query)
        backends[name]=read(EXTRACT/('backend-'+name+'.json'))
        for split in indices:
            filename=name+'-'+split+'-components.npz';assert sha(EXTRACT/filename)==extract_seal['files'][filename]
            path=DEST/filename;path.write_bytes((EXTRACT/filename).read_bytes());files[filename]=sha(path)
    write(DEST/'component-seal.json',dict(files=files,queries_sha256=sha(DEST/'queries.npy'),geometry_targets_joined=False,
        checkpoint_hashes={n:sha(p/'metric.pt') for n,p in MODELS.items()},cutoffs={n:s['threshold'] for n,s in selections.items()}))
    # Truth is joined only after all component arrays have been saved and sealed.
    geometry=read(CAPTURE/'evaluator/geometry.json');ids=read(PREP/'observations/identities.json');truths={};parity={};reports={}
    for split,ix in indices.items():
        rows=[]
        for i in ix:
            assert geometry[int(i)]['id']==ids[int(i)]['id'] and ids[int(i)]['split']==split
            rows.append(metric_targets(camera_boxes(geometry[int(i)]),query).reshape(2,51))
        truth=np.asarray(rows);truths[split]=truth;np.save(DEST/(split+'-truth.npy'),truth)
        target=np.load(BASE/('training-targets.npz' if split=='train' else 'evaluation-targets.npz'))['horizon_boundary']
        target=np.where(target<=3.,target,np.inf)
        np.testing.assert_allclose(truth[:,:,anchor],target,rtol=0,atol=1e-9)
    for name,root in MODELS.items():
        reports[name]={};parity[name]={};cut=selections[name]['threshold'];seal=read(root/'prediction-seal.json')
        for split,ix in indices.items():
            values=dict(np.load(DEST/(name+'-'+split+'-components.npz')));q,mu,scale=[values[k] for k in ('q','mu','scale')]
            oldfile='metric-'+split+'-predictions.npz';assert sha(root/oldfile)==seal['files'][oldfile]
            oldq=np.load(root/oldfile)['width_curve'].reshape(q.shape)
            np.testing.assert_allclose(q,oldq,atol=2e-7,rtol=2e-6)
            actual=crossing(q[:,:,anchor],mu[:,:,anchor],scale[:,:,anchor],cut)
            cname='continuous-'+split+'.npy';assert sha(root/cname)==seal['files'][cname];saved=np.load(root/cname)
            np.testing.assert_array_equal(np.isinf(actual),np.isinf(saved));finite=np.isfinite(saved)
            # Saved float32 Q=2 inversion and float64 inversion from Q=102
            # components differ microscopically. Check metric outcomes exactly
            # as well as a 1mm numerical bound, not a scientific cutoff change.
            maximum=float(np.abs(actual[finite]-saved[finite]).max()) if finite.any() else 0.
            assert maximum<=.001
            z=truths[split][:,:,anchor];observed=np.isfinite(z)&(z>.300001)&finite
            np.testing.assert_array_equal(np.abs(actual[observed]-z[observed])<=.050001,np.abs(saved[observed]-z[observed])<=.050001)
            parity[name][split]=dict(max_q_error=float(np.abs(q-oldq).max()),max_crossing_error_m=maximum,
                crossing_numerical_bound_m=.001,finite_masks_and_5cm_decisions_identical=True)
            report=decompose(q,mu,scale,truths[split],widths,cut)
            report['layers']={label:focus(q[:,layer,anchor],mu[:,layer,anchor],scale[:,layer,anchor],truths[split][:,layer,anchor],cut) for layer,label in enumerate(('BODY','HEAD'))}
            report['unknown_sensor_frames']=sum(bool(ids[int(i)]['baseline']['unknown']) for i in ix)
            report['images']=len(ix);report['layouts']=len({ids[int(i)]['base_group_id'] for i in ix});reports[name][split]=report
    for p,h in inputs.items():assert sha(p)==h,p
    for n,h in code.items():assert sha(HERE/n)==h,n
    for n,h in files.items():assert sha(DEST/n)==h,n
    write(DEST/'result.json',dict(status='PASS',reports=reports,parity=parity,cutoffs={n:s['threshold'] for n,s in selections.items()},backend=backends,
        elapsed_s=time.perf_counter()-started,inputs_unchanged=True,trained=False,evidence_boundary='Frozen selected checkpoints; consumed geometry-partitioned diagnostic; no causal separation or deployable q-unity readout'))
    print('COMPLETE',DEST,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['spec','run']);a=p.parse_args()
    spec() if a.command=='spec' else run()
