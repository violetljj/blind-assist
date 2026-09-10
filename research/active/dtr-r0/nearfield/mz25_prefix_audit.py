"""Audit MZ25's failed1200 prefix and zero-optimizer gradient evidence only."""
import argparse
import difflib
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha


def main(root,output):
    assert not output.exists(), 'Do not overwrite an existing failed-prefix audit'
    work=root/'artifacts.local/work';prior=work/'mz24-decisive-availability-20260910/run-v1'
    experiment=work/'mz25-availability-convergence-20260910';run=experiment/'run-v1';gradient=experiment/'gradient-v1'
    inputs={}
    def bind(path,expected=None):
        path=Path(path);key=str(path)
        if key not in inputs:inputs[key]=sha(path)
        assert expected is None or inputs[key]==expected,key
        return inputs[key]
    s24,s25=read(prior/'start.json'),read(run/'start.json')
    for path in [prior/'start.json',run/'start.json',run/'progress.json',run/'prefix-parity.json']:bind(path)
    common=sorted(set(s24['inputs'])&set(s25['inputs']))
    assert len(common)==23
    for path in common:
        assert s24['inputs'][path]==s25['inputs'][path];bind(path,s24['inputs'][path])
    for path,digest in s25['inputs'].items():bind(path,digest)
    shared_modules=sorted(set(s24['code_sha256'])&set(s25['code_sha256']))
    assert shared_modules==['mz23_availability.py','mz24_decisive_loss.py']
    for name in shared_modules:
        assert s24['code_sha256'][name]==s25['code_sha256'][name]
        bind(Path(__file__).with_name(name),s25['code_sha256'][name])
    for name,start in [('mz24_train.py',s24),('mz25_train.py',s25)]:
        bind(Path(__file__).with_name(name),start['code_sha256'][name])
    bind(Path(__file__).with_name('MZ25_CONVERGENCE_PROTOCOL_20260910.md'),s25['protocol_sha256'])
    for name in ['initial.pt','cutoff.npy']:
        assert bind(run/name)==bind(prior/name)
    bind(run/'batches.npy');bind(prior/'batches.npy')
    base=np.load(prior/'batches.npy');batches=np.load(run/'batches.npy')
    assert base.shape==(1200,16) and batches.shape==(4800,16)
    cycles=[np.array_equal(batches[i*1200:(i+1)*1200],base) for i in range(4)];assert all(cycles)
    assert len(np.unique(batches))==len(np.unique(base))==7562
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=True)
    original_initial=torch.load(prior/'initial.pt',map_location='cpu',weights_only=True)
    assert set(initial)==set(original_initial)
    for name in initial:torch.testing.assert_close(initial[name],original_initial[name],rtol=0,atol=0)
    bind(run/'availability-step1200.pt');bind(prior/'availability.pt')
    prefix=torch.load(run/'availability-step1200.pt',map_location='cpu',weights_only=True)
    reference=torch.load(prior/'availability.pt',map_location='cpu',weights_only=True)
    report=read(run/'prefix-parity.json')
    assert report['status']=='FAIL' and report['steps']==1200 and report['atol']==report['rtol']==1e-4
    assert report['reference_sha256']==bind(prior/'availability.pt')
    assert set(prefix)==set(reference)
    errors={};matches={}
    for name,value in prefix.items():
        errors[name]=float((value-reference[name]).abs().max())
        matches[name]=bool(torch.allclose(value,reference[name],atol=1e-4,rtol=1e-4))
    assert errors==report['per_tensor_error'] and matches==report['per_tensor_match']
    assert max(errors.values())==report['max_abs_error'] and not all(matches.values())
    assert matches['grid'] and matches['relative']
    assert all(not matches[name] for name in matches if name not in ['grid','relative'])
    absent=['availability.pt','predictions.npz','loss_samples.npz','result.json']
    assert all(not (run/name).exists() for name in absent)
    gr=read(gradient/'receipt.json');gv=read(gradient/'result.json')
    bind(gradient/'receipt.json')
    assert gr['status']=='PASS' and gr['training_steps']==0
    assert gv['training_steps']==0 and gv['optimizer_created'] is False
    for name,digest in gr['outputs'].items():bind(gradient/name,digest)
    for path,digest in gv['inputs'].items():bind(path,digest)
    bind(Path(__file__).with_name('mz25_gradient_repeat.py'),gv['code_sha256'])
    assert len(gv['losses'])==4 and len(set(gv['losses']))==1
    with np.load(gradient/'repeats.npz') as a:
        fields=[name.split('/',1)[1] for name in a.files if name.startswith('0/')]
        recomputed={}
        for repeat in [1,2,3]:
            row={}
            for field in fields:
                left,right=a['0/'+field],a[f'{repeat}/'+field]
                row[field]=dict(max_abs=float(np.abs(left-right).max()),changed=int((left!=right).sum()))
            recomputed[str(repeat)]=row
    assert recomputed==gv['differences']
    assert all(row['availability']==dict(max_abs=0.,changed=0) for row in recomputed.values())
    assert any(row['gradient/local.0.weight']['changed']>0 for row in recomputed.values())
    assert 'grid_sampler_2d_backward_cuda' in gv['deterministic_error']
    file24=Path(__file__).with_name('mz24_train.py');file25=Path(__file__).with_name('mz25_train.py')
    module_diff=''.join(difflib.unified_diff(file24.read_text().splitlines(True),file25.read_text().splitlines(True),fromfile='mz24_train.py',tofile='mz25_train.py'))
    audit=dict(status='PASS',experiment_status='STOPPED_PREFIX_GATE_FAILED',optimizer_steps_completed=1200,target_steps=4800,
        final4800_candidate_exists=False,final_prediction_metrics_exist=False,shared_input_count=len(common),shared_inputs=common,
        shared_module_hashes={name:s25['code_sha256'][name] for name in shared_modules},
        initial_state_and_cutoff='byte-identical and state-tensor exact',batches=dict(shape=list(batches.shape),cycles_match=cycles,unique_train=7562),
        prefix=dict(status='FAIL',atol=1e-4,rtol=1e-4,max_abs_error=max(errors.values()),per_tensor_error=errors,per_tensor_match=matches),
        gradient_evidence=dict(status='PASS_ZERO_OPTIMIZER_DIAGNOSTIC',losses=gv['losses'],differences=recomputed,
            strict_deterministic_error=gv['deterministic_error'],optimizer_created=False,training_steps=0),
        module_diff=module_diff,
        static_review='Beforestep1200 scientific update code is unchanged; intended differences are budget/cycles, control binding, logging cadence200to400 and extra snapshot/check. No input or initialization discrepancy found.',
        limits=['The failure preserves its original1e-4gate; this audit does not waive or pass the prefix.',
                'Repeated backward differences and strict grid_sampler failure establish a nondeterministic local backward path; they do not prove that path uniquely accounts for all0.3144long-trajectory drift.',
                'Failedrun loss_samples were only scheduled for final write and are absent; step1200weights/progress/prefix report remain evidence. No4800result exists.',
                'Historical Torch/CUDA/cuDNNruntime flags and versions were not fully recorded.'],
        inputs=inputs,code_sha256=sha(Path(__file__)))
    write(output,audit)
    print('PASS audit of FAILED prefix',len(common),'shared inputs','maxweightdifference',max(errors.values()),'trainingsteps1200only')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.output)
