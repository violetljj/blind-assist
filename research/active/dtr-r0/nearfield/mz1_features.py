"""Cache frozen visual and clean64-zone observations; targets remain separate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from body_query_collection_labels import read,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['visual','tof','combine'],required=True)
    a=p.parse_args();work=Path('artifacts.local/work');dataset=work/'body-query-5000-20260909/dataset-v1'
    manifest=read(dataset/'manifest.json');assert manifest['status']=='PASS' and sha(dataset/'index.json')==manifest['index_sha256']
    rows=read(dataset/'index.json')['frames'];assert len(rows)==5000 and all(r['status']=='PASS' for r in rows)
    a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.set_num_threads(1)
    if a.phase!='combine':assert torch.cuda.is_available();torch.use_deterministic_algorithms(True)
    if a.phase=='visual':
        from body_query_context_evidence import ContextEvidence
        from body_query_fresh_size_eval import FROZEN
        assert not (a.output/'visual.npz').exists()
        base=work/'body-query-10000-b-20260909/run-v1';decoder=work/'body-query-context-decoder-20260909/run-v1'
        for name,digest in FROZEN.items():assert sha((base if name=='NEW-step2000.pt' else decoder)/name)==digest
        model=ContextEvidence(base,decoder,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
        captured=[];hook=model.decoder.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        features=[];alerts=[];joint=[]
        try:
            with torch.inference_mode():
                for begin in range(0,5000,16):
                    images=[]
                    for row in rows[begin:begin+16]:
                        path=Path(row['rgb_file']);assert sha(path)==row['rgb_sha256']
                        with Image.open(path) as im:images.append(np.array(im.convert('RGB').resize((256,144),Image.Resampling.BOX)))
                    x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
                    out=model(x);assert len(captured)==1
                    feature=torch.cat((captured.pop().flatten(1),out['range_probabilities'].flatten(1)),1)
                    features.append(feature.cpu().numpy());alerts.append(out['alerts'].cpu().numpy());joint.append((out['range_probabilities']>=.5).flatten(1).cpu().numpy())
        finally:hook.remove()
        np.savez_compressed(a.output/'visual.npz',visual=np.concatenate(features),original_alerts=np.concatenate(alerts),joint=np.concatenate(joint))
        write(a.output/'visual-receipt.json',dict(status='PASS',sha256=sha(a.output/'visual.npz'),source_index_sha256=sha(dataset/'index.json'),frozen_hashes=FROZEN,
          seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),source_sha256=sha(__file__)))
    elif a.phase=='tof':
        from multizone64_observation import observe
        assert not (a.output/'tof.npz').exists();features=[]
        for begin in range(0,5000,16):
            frames=[]
            for row in rows[begin:begin+16]:
                path=Path(row['native_file']);assert sha(path)==row['native_sha256'];frames.append(np.load(path,allow_pickle=False))
            packet=observe(torch.from_numpy(np.stack(frames)).cuda(),zones_per_axis=8,readout='multi_surface')
            values=torch.nan_to_num(packet['range_m'],nan=0.).float()/4
            features.append(torch.cat((values.flatten(1),packet['valid'].float().flatten(1)),1).cpu().numpy())
        np.savez_compressed(a.output/'tof.npz',tof=np.concatenate(features))
        write(a.output/'tof-receipt.json',dict(status='PASS',sha256=sha(a.output/'tof.npz'),source_index_sha256=sha(dataset/'index.json'),
          seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),observation_source_sha256=sha(Path(__file__).with_name('multizone64_observation.py')),source_sha256=sha(__file__)))
    else:
        assert not (a.output/'features.npz').exists();vr=read(a.output/'visual-receipt.json');tr=read(a.output/'tof-receipt.json')
        assert vr['source_index_sha256']==tr['source_index_sha256']==sha(dataset/'index.json')
        assert sha(a.output/'visual.npz')==vr['sha256'] and sha(a.output/'tof.npz')==tr['sha256']
        with np.load(a.output/'visual.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        with np.load(a.output/'tof.npz',allow_pickle=False) as z:arrays['tof']=z['tof']
        arrays['truth']=np.array([np.array(r['counts']).reshape(2,2,3).sum(-1).reshape(4)>=3 for r in rows])
        arrays['role']=np.array([r['source_role'] for r in rows])
        evalmask=arrays['role']=='EVAL_ONLY'
        mz0=work/'mz0-clean-20260910/run-v1';rr=read(mz0/'rgb-receipt.json');dr=read(mz0/'depth-receipt.json')
        assert sha(mz0/'rgb.npz')==rr['rgb_sha256'] and sha(mz0/'depth.npz')==dr['depth_sha256']
        with np.load(mz0/'rgb.npz',allow_pickle=False) as z:
            assert np.array_equal(arrays['original_alerts'][evalmask],z['alerts']) and np.array_equal(arrays['joint'][evalmask],z['events'])
        with np.load(mz0/'depth.npz',allow_pickle=False) as z:
            expected=np.concatenate((np.nan_to_num(z['z8_multi_surface_range'],nan=0.).astype(np.float32).reshape(1500,128)/4,z['z8_multi_surface_valid'].reshape(1500,128).astype(np.float32)),1)
            assert np.array_equal(arrays['tof'][evalmask],expected)
        assert arrays['visual'].shape==(5000,772) and arrays['tof'].shape==(5000,256)
        assert np.isfinite(arrays['visual']).all() and np.isfinite(arrays['tof']).all()
        np.savez_compressed(a.output/'features.npz',**arrays)
        write(a.output/'features-receipt.json',dict(status='PASS',feature_sha256=sha(a.output/'features.npz'),source_index_sha256=sha(dataset/'index.json'),
            visual_receipt_sha256=sha(a.output/'visual-receipt.json'),tof_receipt_sha256=sha(a.output/'tof-receipt.json'),eval_MZ0_packet_and_alert_parity=1500,training_steps=0))
    print(a.phase,'PASS',time.perf_counter()-start)


if __name__=='__main__':main()
