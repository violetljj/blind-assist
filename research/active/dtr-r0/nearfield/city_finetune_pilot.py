"""One fixed City fit, then paired City/consumed-Willow evaluation. No tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from city_data import CityRGBDataset, CitySupervisedDataset, pixel_support_bce, _array
from decoupled_model import DecoupledModel
from city_pilot_metrics import evaluate

CONFIG=dict(seed=17,steps=200,batch_size=32,learning_rate=1e-5,weight_decay=1e-4,
    optimizer='AdamW',sampling='Uniform TRAIN replacement',batchnorm='Frozen running stats; affine parameters trainable',
    support_weight=.25,checkpoint_selection='FINAL_STEP_ONLY',near_threshold=.5,support_threshold=.5)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


@torch.inference_mode()
def predict(model,dataset):
    model.eval(); near=[]; support=[]
    for batch in DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0):
        n,s=model(batch['rgb'].cuda())
        near.append(n.sigmoid().cpu().numpy()); support.append(s.sigmoid().cpu().numpy())
    return np.concatenate(near),np.concatenate(support)


class RegressionRGB:
    def __init__(self,rgb): self.rgb=rgb
    def __len__(self): return len(self.rgb)
    def __getitem__(self,i):
        return dict(rgb=torch.from_numpy(np.array(self.rgb[i],copy=True)).permute(2,0,1).float()/255.)


def run(a):
    started=time.perf_counter()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    root=Path(__file__).resolve().parents[4]
    artifacts=(root/'artifacts.local').resolve()
    out=a.output.resolve()
    if not out.is_relative_to(artifacts) or out==artifacts or out.exists(): raise ValueError('Fresh artifact output required')
    if sha(a.checkpoint)!=a.checkpoint_sha: raise ValueError('Original checkpoint mismatch')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    out.mkdir(parents=True)
    protocol=Path(__file__).with_name('CITY_FINETUNE_PILOT_20260908.md')
    write(out/'protocol.json',dict(config=CONFIG,protocol_sha256=sha(protocol),
        checkpoint_sha256=a.checkpoint_sha,cache_manifest_sha256=sha(a.cache/'manifest.json'),
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('city_data.py'),
                Path(__file__).with_name('city_pilot_metrics.py'),Path(__file__).with_name('decoupled_model.py'),
                Path(__file__).with_name('representation_model.py'))}))
    fit_steps=0
    try:
        # Only TRAIN RGB and supervision are opened before the final checkpoint.
        train=CitySupervisedDataset(a.cache,'train')
        model=DecoupledModel(a.pretrained).cuda()
        model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
        rng=np.random.default_rng(17); schedule=rng.integers(0,len(train),size=(200,32))
        np.save(out/'training_indices.npy',schedule,allow_pickle=False)
        history=[]; fit_start=time.perf_counter()
        model.train()
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm): layer.eval()
        for indices in schedule:
            samples=[train[int(i)] for i in indices]
            rgb=torch.stack([s['rgb'] for s in samples]).cuda()
            near=torch.stack([s['near'] for s in samples]).cuda()
            support=torch.stack([s['support'] for s in samples]).cuda()
            n,s=model(rgb)
            loss=F.binary_cross_entropy_with_logits(n,near)+.25*pixel_support_bce(s,support)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                raise RuntimeError('Nonfinite gradient')
            optimizer.step(); fit_steps+=1
            if fit_steps%25==0:
                row=dict(step=fit_steps,loss=float(loss.detach()),fit_elapsed_s=time.perf_counter()-fit_start)
                history.append(row); write(out/'progress.json',row); print(json.dumps(row),flush=True)
        torch.cuda.synchronize()
        checkpoint=out/'city_seed17_step200.pt'; torch.save(model.state_dict(),checkpoint)
        fit=dict(steps=fit_steps,fit_seconds=time.perf_counter()-fit_start,history=history,
                 checkpoint_sha256=sha(checkpoint),schedule_sha256=sha(out/'training_indices.npy'))
        write(out/'fit-complete.json',fit)
        del optimizer,train,rgb,near,support,n,s,loss,samples

        # TEST/regression labels are opened only after the immutable fit is done.
        city=CityRGBDataset(a.cache,'test')
        labels=read(a.cache/'evaluator/test.json')
        if labels['sample_indices']!=city.ids: raise ValueError('Test label ordering mismatch')
        cy=np.array(_array(a.cache.resolve(),labels['near']),copy=True)
        cs=np.array(_array(a.cache.resolve(),labels['support']),copy=True)
        split=read(a.split)
        if sha(a.split)!=read(a.cache/'manifest.json')['source_split_sha256']: raise ValueError('Split hash changed')
        groups={r['sample_index']:r['group_id'] for r in split['frames']}
        rm=read(a.regression/'manifest.json')
        for filename,digest in rm['files'].items():
            if sha(a.regression/filename)!=digest: raise ValueError('Regression payload mismatch')
        rr=np.load(a.regression/'rgb.npy',mmap_mode='r',allow_pickle=False)
        ry=np.load(a.regression/'near.npy',allow_pickle=False)
        rs=np.load(a.regression/'support.npy',allow_pickle=False)
        if rr.shape!=(96,144,256,3) or rr.dtype!=np.uint8 or ry.shape!=(96,2) or rs.shape!=(96,2,18,32):
            raise ValueError('Expected declared Willow96 regression data')
        result={}
        for arm,weights in (('baseline',a.checkpoint),('city_finetuned',checkpoint)):
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
            result[arm]={}
            for name,dataset,y,truth,group in (('city',city,cy,cs,[groups[i] for i in city.ids]),
                                             ('willow_regression',RegressionRGB(rr),ry,rs,None)):
                probs,maps=predict(model,dataset)
                np.savez_compressed(out/f'{arm}-{name}-predictions.npz',near=probs,support=maps)
                result[arm][name]=evaluate(probs,maps,y,truth,group)
        b=result['baseline']; f=result['city_finetuned']
        city_gain=f['city']['near_balanced_error']<b['city']['near_balanced_error']
        recall_ok=all(f['city']['heads'][h]['near']['recall']>=b['city']['heads'][h]['near']['recall'] for h in ('BODY','HEAD'))
        regression_ok=all(f['willow_regression']['heads'][h]['near'][k]<=b['willow_regression']['heads'][h]['near'][k]
                          for h in ('BODY','HEAD') for k in ('FP','FN'))
        write(out/'result.json',dict(status='PASS',results=result,
            decision=dict(city_balanced_error_improved=city_gain,city_recall_not_lower=recall_ok,
                willow_no_fp_fn_increase=regression_ok,joint_criterion=city_gain and recall_ok and regression_ok),
            scope='One-seed controlled Development; no promotion, no threshold selection',
            regression_manifest_sha256=sha(a.regression/'manifest.json')))
        if sha(a.checkpoint)!=a.checkpoint_sha or sha(checkpoint)!=fit['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during evaluation')
        write(out/'receipt.json',dict(status='PASS',fit=fit,actual_backend='CUDA',device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,wall_s=time.perf_counter()-started,result_sha256=sha(out/'result.json')))
    except BaseException as exc:
        write(out/'receipt.json',dict(status='FAIL',error=str(exc),completed_steps=fit_steps,wall_s=time.perf_counter()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','checkpoint','pretrained','regression','split','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha',required=True)
    run(p.parse_args())
