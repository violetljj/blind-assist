"""EXPLORE RGB/query point pilot on measured first-return sensor rays.

Full RGB and public K/query define every prediction location. Reference depth
and labels enter training loss and scoring only. UNKNOWN is never negative.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import confusion, ratios, rays, ray_interval


def identity(row):
    return row['scan'], row['frame']


def cutoff_for_fpr(free_scores, target):
    """Lowest empirical cutoff with FP <= floor(target*N), including ties."""
    scores = np.asarray(free_scores, np.float32)
    if scores.size == 0 or not 0 <= target <= 1:
        raise ValueError('Need nonempty free-ray calibration and valid FPR')
    allowed = int(np.floor(target*scores.size))
    if allowed >= scores.size:
        return 0.
    boundary = np.partition(scores, scores.size-allowed-1)[scores.size-allowed-1]
    return float(np.nextafter(boundary, np.float32(np.inf)))


def summarize(records):
    counts = {k: sum(r[k] for r in records) for k in ('tp', 'fn', 'fp', 'tn')}
    pos = [r for r in records if r['reference_state'] == 'POSITIVE']
    free = [r for r in records if r['reference_state'] == 'FREE_ON_SAMPLED_RAYS']
    return dict(**counts, **ratios(counts), query_positive_hits=sum(r['predicted_positive'] for r in pos),
                query_positive_total=len(pos), sampled_free_false_support=sum(r['predicted_positive'] for r in free),
                sampled_free_total=len(free), unknown_query_total=sum(r['reference_state'] == 'UNKNOWN' for r in records),
                frames=len({identity(r) for r in records}))


def run(repo, sensor, output, budget_s=240., steps=200, batch_size=8192):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'receipt.json').exists():
        raise FileExistsError('Preserve run evidence; no implicit restart')
    receipt = dict(status='STARTING', seed=7, fixed_steps=steps, batch_size=batch_size,
                   gpu_allocation_budget_s=budget_s, download_bytes=0, checkpoint_selection='final fixed step; no validation selection',
                   input_contract='full native RGB + public color/depth K + query bounds; no reference crop/GT localization',
                   label_contract='only 0 FREE_RAY and 1 sensor occupied in training/scoring;2/255 excluded',
                   evidence='consumed official-train Development, environment-separated train/cal/validation; no whole-box clearance',
                   observation_sha256=sha(sensor/'observations.json'), dataset_manifest_sha256=sha(sensor/'dataset_manifest.json'),
                   script_sha256=sha(Path(__file__)), stages={})
    torch = backbone = features = projected = heads = None
    def check(reserve=0.):
        if time.perf_counter()-start >= budget_s-reserve:
            raise TimeoutError('Pilot allocation budget reached')
    try:
        import torch
        import torch.nn.functional as F
        from torchvision.models import mobilenet_v3_small
        torch.set_num_threads(4)
        torch.manual_seed(7); np.random.seed(7)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for bounded pilot')
        torch.cuda.reset_peak_memory_stats()
        receipt.update(framework=torch.__version__, device=torch.cuda.get_device_name(0))
        obs = json.loads((sensor/'observations.json').read_text('utf-8-sig'))
        dataset = json.loads((sensor/'dataset_manifest.json').read_text('utf-8-sig'))
        if obs['dataset_manifest_sha256'] != receipt['dataset_manifest_sha256']:
            raise ValueError('Observation/reference manifest identity differs')
        observed, sources = obs['rows'], dataset['rows']
        if [identity(r) for r in observed] != [identity(r) for r in sources]:
            raise ValueError('Observation/reference row order differs')
        q = obs['queries']; nq = len(q)
        if q != dataset['queries']:
            raise ValueError('Public query identity differs')
        groups = {split: sorted({r['environment'] for r in observed if r['split'] == split})
                  for split in ('train', 'cal', 'validation')}
        if any(set(groups[a]) & set(groups[b]) for a,b in [('train','cal'),('train','validation'),('cal','validation')]):
            raise ValueError('Environment leakage')
        if {k:len(v) for k,v in groups.items()} != dict(train=7, cal=2, validation=3):
            raise ValueError('Expected frozen 7/2/3 environment split')
        receipt['groups'] = groups
        checkpoint = repo/'artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth'
        receipt.update(checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                       backbone='torchvision MobileNetV3Small ImageNet cached weights, frozen final 576 channels',
                       feature_projection='fixed seed7 Gaussian 576->32 /sqrt(576); no supervised projection fitting')
        backbone = mobilenet_v3_small(weights=None)
        backbone.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        backbone = backbone.features.to('cuda').half().eval().requires_grad_(False)
        gen = torch.Generator(device='cuda').manual_seed(7)
        projection = torch.randn((32,576,1,1), generator=gen, device='cuda', dtype=torch.float32)/np.sqrt(576)
        projection = projection.half()
        norm_mean = torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
        norm_std = torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
        feature_rows, ray_rows, labels, reachability = [], [], [], []
        feature_times = []
        stage = time.perf_counter()
        for i,r in enumerate(observed):
            check(20.)
            src = sources[i]
            if r['rgb_sha256'] != src['rgb_sha256'] or sha(r['rgb_path']) != r['rgb_sha256']:
                raise ValueError('RGB source identity changed')
            image = np.array(Image.open(r['rgb_path']).convert('RGB'), copy=True)
            if list(image.shape[:2]) != r['color_shape']:
                raise ValueError('RGB shape changed')
            h,w = r['depth_shape']; ch,cw = r['color_shape']
            rx,ry = rays(np.asarray(r['depth_K']), (h,w))
            homogeneous = np.stack([rx,ry,np.ones_like(rx)],-1) @ np.asarray(r['color_K']).T
            u,v = homogeneous[...,0]/homogeneous[...,2], homogeneous[...,1]/homogeneous[...,2]
            grid = torch.as_tensor(np.stack([2*(u+.5)/cw-1,2*(v+.5)/ch-1],-1),dtype=torch.float32,device='cuda')[None]
            x = torch.from_numpy(image).to('cuda',dtype=torch.float32).permute(2,0,1)[None]/255.
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.inference_mode():
                feat = backbone(((x-norm_mean)/norm_std).half())
                projected = F.conv2d(feat, projection)
                sampled = F.grid_sample(projected.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                feature_rows.append(sampled[0].permute(1,2,0).reshape(-1,32).half())
            torch.cuda.synchronize(); feature_times.append(time.perf_counter()-t)
            ray_rows.append(torch.as_tensor(np.stack([rx,ry],-1).reshape(-1,2),dtype=torch.float32,device='cuda'))
            reachability.append(np.stack([ray_interval(rx,ry,query)[2] for query in q]))
            # Reference only becomes available after RGB/public-K feature extraction.
            if sha(src['reference_path']) != src['reference_sha256']:
                raise ValueError('Reference identity changed')
            with np.load(src['reference_path'],allow_pickle=False) as ref:
                labels.append(np.array(ref['labels'],copy=True))
            if labels[-1].shape != (nq,h,w):
                raise ValueError('Reference grid/query shape differs')
            del x, feat, sampled, grid
            if (i+1)%24 == 0:
                print(f'features {i+1}/{len(observed)} elapsed={time.perf_counter()-start:.2f}s',flush=True)
        features = torch.stack(feature_rows); del feature_rows
        ray_tensor = torch.stack(ray_rows); del ray_rows
        label_array = np.stack(labels); del labels
        nr,npix,_ = features.shape
        features_path = output/'features.pt'
        torch.save(dict(features=features.cpu(), observations_sha256=receipt['observation_sha256'],
                        checkpoint_sha256=receipt['checkpoint_sha256'], projection=projection.cpu()),features_path)
        receipt['feature_sha256'] = sha(features_path)
        receipt['stages']['feature'] = dict(wall_s=time.perf_counter()-stage, model_and_projection_s=sum(feature_times),
                                           calls=nr, native_shapes=sorted({tuple(r['color_shape']) for r in observed}),
                                           channels=32, cache_path=str(features_path), inference_mode=True)
        del backbone, projected; backbone = projected = None
        torch.cuda.empty_cache()
        bounds = torch.as_tensor([query['low']+query['high'] for query in q],device='cuda',dtype=torch.float32)
        bounds /= torch.tensor([.9,.55,6.,.9,.55,6.],device='cuda')
        train_rows = np.array([i for i,r in enumerate(observed) if r['split']=='train'])
        train_known = np.flatnonzero((label_array[train_rows] < 2).reshape(-1))
        # Map local train row addresses to global row addresses, independent of labels' values.
        train_local_frame = train_known//(nq*npix)
        train_global = train_rows[train_local_frame]*(nq*npix)+train_known%(nq*npix)
        known = torch.as_tensor(train_global,device='cuda',dtype=torch.long)
        target = torch.as_tensor(label_array.reshape(-1),device='cuda',dtype=torch.uint8)
        make_head = lambda: torch.nn.Sequential(torch.nn.Linear(40,64),torch.nn.ReLU(),torch.nn.Linear(64,64),
                                               torch.nn.ReLU(),torch.nn.Linear(64,1)).to('cuda')
        torch.manual_seed(7); rgb_head = make_head(); geometry_head = make_head()
        geometry_head.load_state_dict(rgb_head.state_dict())
        heads = dict(rgb=rgb_head, geometry=geometry_head)
        initial_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in rgb_head.state_dict().values())).hexdigest()
        receipt.update(train_known_points=len(train_known), training_initial_state_sha256=initial_sha,
                       loss='unweighted BCE sampled uniformly from all known train row/query/pixels; same seed sampling for both arms',
                       head='32 RGB features + 2 optical ray directions + 6 normalized query bounds ->64 ReLU->64 ReLU->1')
        def point_input(fidx, qidx, pidx, arm, source=None):
            fi = fidx if source is None else source
            rgb = features[fi,pidx].float() if arm=='rgb' else torch.zeros((fidx.numel(),32),device='cuda')
            return torch.cat([rgb,ray_tensor[fidx,pidx],bounds[qidx]],1)
        train_stage = time.perf_counter(); step_log = {}
        for arm,head in heads.items():
            check(25.)
            head.train(); optimizer = torch.optim.Adam(head.parameters(),lr=.002)
            sampling = torch.Generator(device='cuda').manual_seed(7)
            losses = []
            for step in range(steps):
                check(25.)
                indices = known[torch.randint(known.numel(),(batch_size,),generator=sampling,device='cuda')]
                fidx = indices//(nq*npix); qidx = indices//npix%nq; pidx = indices%npix
                logits = head(point_input(fidx,qidx,pidx,arm))[:,0]
                loss = F.binary_cross_entropy_with_logits(logits,target[indices].float())
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                if step%50==0 or step==steps-1:
                    losses.append(dict(step=step+1,bce=float(loss.detach())))
            torch.cuda.synchronize()
            head.eval(); path=output/f'{arm}_final.pt'
            torch.save(dict(state_dict=head.state_dict(),seed=7,steps=steps,initial_sha256=initial_sha),path)
            step_log[arm] = dict(steps=steps,loss_samples=losses,checkpoint=str(path),sha256=sha(path))
            del optimizer
        receipt['stages']['train'] = dict(wall_s=time.perf_counter()-train_stage,arms=step_log)
        del known, target
        dp = json.loads((sensor/'evaluation.json').read_text('utf-8-sig'))
        desired_fpr = dp['summary']['cal']['fpr']
        receipt.update(depthpro_evaluation_sha256=sha(sensor/'evaluation.json'),calibration_target_pixel_fpr=desired_fpr)
        val_rows = [i for i,r in enumerate(observed) if r['split']=='validation']
        shuffle_map = {}
        val_groups = groups['validation']
        for fi in val_rows:
            env = observed[fi]['environment']; next_env = val_groups[(val_groups.index(env)+1)%len(val_groups)]
            peers = [i for i in val_rows if observed[i]['environment']==env]
            other = [i for i in val_rows if observed[i]['environment']==next_env]
            shuffle_map[fi] = other[peers.index(fi)%len(other)]
        receipt['rgb_shuffle'] = dict(method='fixed next validation environment by sorted ID, same ordinal frame; never same environment',
                                     row_mapping={str(k):v for k,v in shuffle_map.items()},
                                     projection='source full feature map mapped by source public K at its same sensor-ray ordinal; target ray/bounds retained')
        all_q = torch.arange(nq,device='cuda').repeat_interleave(npix)
        all_p = torch.arange(npix,device='cuda').repeat(nq)
        def predict_rows(indices,head,arm,shuffled=False):
            out=[]
            for fi in indices:
                check(3.)
                chunks=[]
                for begin in range(0,nq*npix,131072):
                    qi=all_q[begin:begin+131072]; pi=all_p[begin:begin+131072]
                    fs=torch.full_like(pi,fi)
                    source=torch.full_like(pi,shuffle_map[fi]) if shuffled else None
                    with torch.inference_mode():
                        scores=head(point_input(fs,qi,pi,arm,source)).sigmoid()[:,0]
                    chunks.append(scores.cpu().numpy())
                out.append((fi,np.concatenate(chunks).reshape(label_array.shape[1:])))
            return out
        def score(predictions,threshold):
            records=[]
            for fi,scores in predictions:
                r=sources[fi]
                for qi,query in enumerate(q):
                    pred=(scores[qi]>=threshold)&reachability[fi][qi]
                    counts=confusion(label_array[fi,qi],pred)
                    records.append(dict(environment=r['environment'],scan=r['scan'],frame=r['frame'],split=r['split'],
                                        query=query['name'],reference_state=r['queries'][qi]['state'],
                                        predicted_support=int(pred.sum()),predicted_positive=int(pred.sum())>=16,**counts))
            return dict(summary=summarize(records),records=records)
        evaluate_stage=time.perf_counter(); arms={}
        cal_rows=[i for i,r in enumerate(observed) if r['split']=='cal']
        for arm,head in heads.items():
            cal=predict_rows(cal_rows,head,arm)
            free_scores=np.concatenate([p[label_array[fi]==0] for fi,p in cal])
            cutoff=cutoff_for_fpr(free_scores,desired_fpr)
            validation=predict_rows(val_rows,head,arm)
            saved=[]
            for fi,p in cal+validation:
                path=output/f'{arm}_{observed[fi]["scan"]}_{observed[fi]["frame"]:06d}.npy'
                np.save(path,p,allow_pickle=False)
                saved.append(dict(row=fi,path=str(path),sha256=sha(path)))
            arms[arm]=dict(cutoff=cutoff,cal=score(cal,cutoff),validation=score(validation,cutoff),predictions=saved)
            print(f'{arm} cal={arms[arm]["cal"]["summary"]} validation={arms[arm]["validation"]["summary"]}',flush=True)
            del cal,validation,free_scores
        shuffled=predict_rows(val_rows,rgb_head,'rgb',True)
        arms['rgb_shuffled']=dict(cutoff=arms['rgb']['cutoff'],validation=score(shuffled,arms['rgb']['cutoff']),predictions=[])
        for fi,p in shuffled:
            path=output/f'rgb_shuffled_{observed[fi]["scan"]}_{observed[fi]["frame"]:06d}.npy'
            np.save(path,p,allow_pickle=False)
            arms['rgb_shuffled']['predictions'].append(dict(row=fi,source_row=shuffle_map[fi],path=str(path),sha256=sha(path)))
        receipt['stages']['evaluate']=dict(wall_s=time.perf_counter()-evaluate_stage)
        result=dict(status='COMPLETE',arms=arms,depthpro=dp['summary'],
                    observation_sha256=receipt['observation_sha256'],dataset_manifest_sha256=receipt['dataset_manifest_sha256'],
                    calibration='each trained arm matches/bounds Depth Pro cal pixel FPR; validation held untouched for threshold/checkpoint',
                    limits=['Sensor first-return ray supervision, no complete query negatives or clearance proof',
                            'ImageNet pretrained/frozen coarse spatial features and fixed random channel projection; no high-resolution mechanism test',
                            'Query support is predicted on all public reachable rays; pixel confusion uses only known sensor rays',
                            'RGB shuffle changes scene content and calibration map; sensitivity diagnostic, not causal deployment proof',
                            '12 consumed official-train environments, not independent confirmation/body-frame/walking-event/mobile evaluation'])
        write(output/'evaluation.json',result)
        receipt['status']='COMPLETE'
    except TimeoutError as error:
        receipt.update(status='PARTIAL_BUDGET',error=repr(error))
    except Exception as error:
        receipt.update(status='FAILED',error=repr(error)); raise
    finally:
        if torch is not None:
            receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            backbone=features=projected=heads=None
            # Release all remaining local CUDA references when run returns, plus cache.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
        receipt.update(allocation_wall_s=time.perf_counter()-start,resource_release='process termination releases remaining local CUDA tensors',
                       cpu_auxiliary_s='included in allocation wall; no separate CPU-only phase')
        write(output/'receipt.json',receipt)
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--sensor',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--budget-s',type=float,default=240.)
    args=parser.parse_args()
    run(args.repo.resolve(),args.sensor.resolve(),args.output.resolve(),args.budget_s)
