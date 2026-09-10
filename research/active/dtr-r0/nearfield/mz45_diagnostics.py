"""Independent native-label recount and saved-branch failure attribution."""
import argparse
from pathlib import Path
import time
import numpy as np
from data_lightweight import CompactSource
from mz5_ensemble_readout import read,write,sha,load_npz


def run(root,task):
    started=time.perf_counter(); bind={}
    def checked(path,expected=None):
        digest=sha(path);assert expected is None or digest==expected,str(path)
        bind[str(path)]=digest;return path
    prepared=task/'prepared-v1';inference=task/'inference-v1'
    prep=read(checked(prepared/'receipt.json'));inf=read(checked(inference/'receipt.json'))
    for path,digest in prep['inputs'].items():checked(path,digest)
    arrays=load_npz(checked(inference/'predictions.npz',inf['outputs']['predictions.npz']))
    labels=load_npz(checked(prepared/'evaluator.npz',prep['outputs']['evaluator.npz']))
    rows=read(checked(prepared/'groups.json',prep['outputs']['groups.json']))['records']
    np.testing.assert_array_equal(labels['frame_ids'],arrays['frame_ids'])
    yy,xx=np.indices((360,640));f=320/np.tan(np.deg2rad(50));right=(xx-319.5)/f;up=-(yy-179.5)/f
    counts=[];verified=0
    for block,eid in [('near','mz42-rich-objects-20260910'),('far','mz44-rich-far-objects-20260911')]:
        source=root/'artifacts.local/work'/eid
        verification=read(checked(source/'return-verification.json'))
        package=checked(source/'returned-v1/capture.source.zip',verification['archive_sha256'])
        record=read(checked(source/'returned-v1/dataset-v1'/('result.json' if block=='near' else 'range-result.json')))
        with CompactSource(package) as compact:
            assert compact.read_json('source-integrity.json')['unchanged']
            assert compact.read_json('render-resource-health.json')['ready_data_eligible']
            for row in record['records']:
                assert row['frame_id']==labels['frame_ids'][verified]
                depth=compact.load_array(row['native']).astype(float)
                y=depth*right;z=1.7+depth*up
                valid=np.isfinite(depth)&(depth>0)&(depth<100)&(depth*np.sqrt(1+right*right+up*up)<=4)
                current=[]
                for front,width,zlow,zhigh in ((.18,.28,.65,1.4),(.13,.18,1.4,1.85)):
                    for half in (0,1):
                        inside=valid&(depth>=front+1.5*half)&((depth<front+1.5) if half==0 else (depth<=front+3))&(y>=-width)&(y<=width)&(z>=zlow)&(z<=zhigh)
                        current.append(int(inside.sum()))
                assert current==row['event_counts'],row['frame_id']
                np.testing.assert_array_equal(np.array(current)>=3,labels['truth'][verified])
                assert labels['known'][verified].all() and valid.any()
                counts.append(current);verified+=1
    assert verified==40
    new=np.array([r['family']!='oblique_rod' for r in rows]);positive=labels['truth']&labels['known']&new[:,None]
    negative=~labels['truth']&labels['known']&new[:,None]
    assert positive.sum()==32
    diagnostics={}
    for condition in ('IDEAL','MERGE_CLOSE','DROP_CLOSE'):
        a=lambda name:arrays[condition+'/'+name]
        rgb=a('rgb');tof=a('MZ43_TOF');ens=a('MZ43_ENSEMBLE')
        supported=a('original_support')
        restricted=a('restricted_support')
        # Arithmetic suppression is observable here; physical sensing cause is not.
        suppressed=positive&(tof>=0)&(ens<0)
        union=(rgb>=0)|(tof>=0)
        diagnostics[condition]=dict(new_form_positive_bits=int(positive.sum()),
            rgb_true_bits=int(((rgb>=0)&positive).sum()),
            mixed_tof_true_bits=int(((tof>=0)&positive).sum()),
            mixed_tof_false_bits=int(((tof>=0)&negative).sum()),
            mixed_ensemble_true_bits=int(((ens>=0)&positive).sum()),
            positive_tof_bits_suppressed_by_equal_mean=int(suppressed.sum()),
            suppressed_rgb_logit_range=[float(rgb[suppressed].min()),float(rgb[suppressed].max())] if suppressed.any() else None,
            suppressed_tof_logit_range=[float(tof[suppressed].min()),float(tof[suppressed].max())] if suppressed.any() else None,
            geometric_candidate_support_on_true_bits=int((supported&positive).sum()),
            bank_restricted_candidate_support_on_true_bits=int((restricted&positive).sum()),
            mz28_true_bits=int(((a('MZ28')>=0)&positive).sum()),
            mz37_true_bits=int(((a('MZ37')>=0)&positive).sum()),
            simple_union_diagnostic=dict(tp=int((union&positive).sum()),fp=int((union&negative).sum()),
                interpretation='Saved positive-branch opportunity/cost only; no calibrated or adopted fusion'))
    result=dict(status='PASS',source_sha256=sha(__file__),inputs=bind,independent_native_event_counts=160,
        source_native_counts=counts,branch_diagnostics=diagnostics,training_steps=0,
        backend='CPU independent scalar/native geometry; TASK_NOT_GPU_SUITABLE',seconds=time.perf_counter()-started,
        interpretation='Native events remain valid. Global RGB/mean decisions fail on new forms; partial ToF evidence is suppressed. Candidate support is angular geometry, not verified target-return identity or clearance.')
    path=task/'diagnostics.json';assert not path.exists();write(path,result)
    print('DIAGNOSTICS',diagnostics,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',type=Path,required=True)
    a=parser.parse_args();run(a.root.resolve(),a.task.resolve())
