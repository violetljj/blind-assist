"""Forward/adjoint and strict deterministic-backward checks; zero optimizer steps."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read,write,sha
from mz23_availability import AngularAvailability
from mz25_fixed_sampler import FixedBilinearSampler,StableAngularAvailability


def main(root,output):
    output.mkdir(parents=True,exist_ok=False)
    path=root/'artifacts.local/work/mz24-decisive-availability-20260910/run-v1/initial.pt'
    assert sha(path)==read(path.parent/'receipt.json')['outputs']['initial.pt']
    state=torch.load(path,map_location='cpu',weights_only=True);grid=state['grid']
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(250)
    # Actual angular grid includes boundary/outside-grid interpolation. Random
    # features and upstream gradients test the linear operator, not task quality.
    x=torch.randn(2,16,28,28,device='cuda',requires_grad=True);up=torch.randn(2,16,64,49,device='cuda')
    reference=F.grid_sample(x,grid.cuda()[None].expand(2,-1,-1,-1),align_corners=False)
    reference_grad=torch.autograd.grad((reference*up).sum(),x)[0]
    sampler=FixedBilinearSampler(grid).cuda();new=sampler(x);new_grad=torch.autograd.grad((new*up).sum(),x)[0]
    forward_error=float((reference-new).abs().max());backward_error=float((reference_grad-new_grad).abs().max())
    torch.testing.assert_close(new,reference,atol=1e-5,rtol=1e-5);torch.testing.assert_close(new_grad,reference_grad,atol=1e-5,rtol=1e-5)
    # Model-level parity and repeated exact gradients under strict determinism.
    old=AngularAvailability(grid);old.load_state_dict(state);old.cuda()
    stable=StableAngularAvailability(grid);stable.load_state_dict(state);stable.cuda()
    features=torch.randn(2,64,28,28,device='cuda');ranges=torch.rand(2,64,2,device='cuda')*4
    valid=torch.ones_like(ranges,dtype=torch.bool);ranges[0,0,0]=float('nan');valid[0,0,0]=False
    outputs={};gradients=[];target=torch.rand(2,64,49,device='cuda')>.5
    try:
        torch.use_deterministic_algorithms(True)
        for wrong in [False,True]:
            with torch.no_grad():a=old(features,ranges,valid,wrong);b=stable(features,ranges,valid,wrong)
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5);outputs[str(wrong)]=float((a-b).abs().max())
        for attempt in range(4):
            stable.zero_grad(set_to_none=True);a=stable(features,ranges,valid)
            F.binary_cross_entropy_with_logits(a,target.float()).backward()
            gradients.append({name:p.grad.detach().cpu().numpy().copy() for name,p in stable.named_parameters()})
        for later in gradients[1:]:
            for name,value in later.items():np.testing.assert_array_equal(value,gradients[0][name])
        for name,value in stable.state_dict().items():torch.testing.assert_close(value.cpu(),state[name],rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(False)
    result=dict(status='PASS',training_steps=0,forward_max_abs=forward_error,adjoint_max_abs=backward_error,
        model_forward_max_abs=outputs,strict_cuda_backward='PASS',repeated_gradient_arrays='bit-identical across4backwards',
        gradient_elements=sum(v.size for v in gradients[0].values()),parameters=sum(p.numel() for p in stable.parameters()),
        sampling_matrix_bytes=sampler.matrix.numel()*sampler.matrix.element_size(),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        device=torch.cuda.get_device_name(),input_sha256=sha(path),
        limits='Generated-feature operator/gradient check only; no learned candidate, training trajectory, task performance or device timing claim.')
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,outputs={'result.json':sha(output/'result.json')},
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz25_fixed_sampler.py')]}))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
