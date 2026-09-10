"""Two fixed raster fits with owner-scoped transient features and streamed evaluation."""
import argparse
import gc
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz16_detail_cache import CROP
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import Bindings, CONDITIONS, setup
from mz47_local_enrichment import bound_outputs, packet
from mz51_training_coverage import negative_loss
from mz54_full_rgb_model import RasterQuery, loss_for
from mz54_full_rgb_source import prepare, RGBStore

ARMS = ('CROP_RASTER', 'FULL_RASTER')
MODES = ARMS + ('FULL_CROP_ONLY',)
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
OLD_COHORTS = COHORTS[:3]


def parameters_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_encoder(root, bind):
    work = root / 'artifacts.local/work'
    baseline = work / 'body-query-10000-b-20260909/run-v1'
    context = work / 'body-query-context-decoder-20260909/run-v1'
    reference = read(bind(work / 'mz36-new-source-20260910/inference-v1/receipt.json'))
    for name, digest in reference['code_sha256'].items():
        if name != 'mz36_frozen_inference.py': bind(Path(__file__).with_name(name), digest)
    for name, digest in FROZEN.items():
        bind((baseline if name == 'NEW-step2000.pt' else context) / name, digest)
    whole = ContextEvidence(baseline, context, work / 'body-query-v1-20260908/model-inputs/pretrained')
    base = whole.base.cuda().eval().requires_grad_(False)
    del whole
    norm = load_npz(bind.checkpoint(work / 'mz9-source-supervision-20260910/run-v1', 'normalization.npz'))
    mean, std = (torch.from_numpy(norm[k]).cuda() for k in ('mean', 'std'))
    return base, mean, std


class Features:
    def __init__(self, bundle, base, store, scratch, out):
        self.bundle, self.base, self.store, self.out = bundle, base, store, out
        schedule = bundle['schedule']
        self.new_ids, self.old_ids = np.unique(schedule['shared']), np.unique(schedule['OLD_NEG'])
        assert (len(self.new_ids), len(self.old_ids)) == (1250, 3533)
        self.new_lookup = np.full(2560, -1, np.int32)
        self.old_lookup = np.full(len(bundle['old_truth']), -1, np.int32)
        self.new_lookup[self.new_ids] = np.arange(len(self.new_ids))
        self.old_lookup[self.old_ids] = np.arange(len(self.old_ids)) + len(self.new_ids)
        self.path = scratch / 'training-full.npy'
        self.full = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.float32,
                                              shape=(4783, 64, 45, 80))
        self.crop = np.empty((1250, 64, 28, 28), np.float32)
        self.stats = dict(full_extracted_frames=0, crop_extracted_frames=0,
                          full_cache_eval_hits=0, crop_cache_eval_hits=0)
        np.savez_compressed(out / 'feature-cache-index.npz', new_ids=self.new_ids,
                            old_ids=self.old_ids, new_lookup=self.new_lookup, old_lookup=self.old_lookup)

    def extract(self, images, crop=False):
        if not images: raise ValueError('Empty encoder request')
        rows = [np.asarray(im.crop(CROP) if crop else im, dtype=np.uint8) for im in images]
        rgb = torch.from_numpy(np.stack(rows)).permute(0, 3, 1, 2).cuda().float() / 255
        with torch.inference_mode(): result = fixed_batch_dense(self.base, rgb).cpu().numpy()
        shape = (len(images), 64, 28, 28) if crop else (len(images), 64, 45, 80)
        assert result.shape == shape and np.isfinite(result).all()
        self.stats['crop_extracted_frames' if crop else 'full_extracted_frames'] += len(images)
        return result

    def build(self):
        tick = time.perf_counter()
        # Engineering parity on TRAIN rows only; fixed batch shape/arithmetic.
        ids = self.old_ids[:16]
        images = [self.store.load(self.bundle['old_rgb_refs'][int(i)]) for i in ids]
        actual = self.extract(images, crop=True)
        expected = self.bundle['old_maps'][self.bundle['old_lookup'][ids]]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
        replay = dict(frames=16, values=int(actual.size), max_abs=float(np.abs(actual-expected).max()),
                      atol=2e-5, rtol=1e-6, cohort='Original TRAIN, no fit or calibration selection')
        refs = [self.bundle['cohorts']['mz48']['rgb_refs'][int(i)] for i in self.new_ids]
        refs += [self.bundle['old_rgb_refs'][int(i)] for i in self.old_ids]
        for start in range(0, len(refs), 16):
            images = [self.store.load(r) for r in refs[start:start+16]]
            self.full[start:start+len(images)] = self.extract(images)
            nnew = min(len(images), max(0, len(self.new_ids)-start))
            if nnew: self.crop[start:start+nnew] = self.extract(images[:nnew], crop=True)
            if start % 512 == 0 or start+len(images) == len(refs):
                row = dict(stage='training-features', frames=start+len(images), total=len(refs),
                           seconds=time.perf_counter()-tick)
                write(self.out/'progress.json', row); print('FEATURES', row, flush=True)
        self.full.flush()
        write(self.out/'feature-parity.json', dict(status='PASS', crop_replay=replay,
            full_shape=list(self.full.shape), full_bytes=int(self.full.nbytes), crop_ram_bytes=int(self.crop.nbytes),
            fixed_encoder_batch=16, original_global_inferences=0))
        return time.perf_counter()-tick

    def training(self, arm, ids, old=False):
        ids = np.asarray(ids)
        if arm == 'FULL_RASTER':
            lookup = self.old_lookup if old else self.new_lookup
            assert (lookup[ids] >= 0).all()
            return np.array(self.full[lookup[ids]])
        if old:
            return np.array(self.bundle['old_maps'][self.bundle['old_lookup'][ids]])
        assert (self.new_lookup[ids] >= 0).all()
        return self.crop[self.new_lookup[ids]]

    def evaluation(self, cohort, ids):
        full = np.empty((len(ids), 64, 45, 80), np.float32)
        crop = np.empty((len(ids), 64, 28, 28), np.float32)
        images, full_missing, crop_missing = {}, [], []
        rows = self.bundle['cohorts'][cohort]
        for j, i in enumerate(ids):
            cache = -1
            if cohort in OLD_COHORTS:
                gid = int(self.bundle['predictions'][cohort+'/frame_ids'][i])
                cache = self.old_lookup[gid]
                crop[j] = self.bundle['old_maps'][self.bundle['old_lookup'][gid]]
                self.stats['crop_cache_eval_hits'] += 1
            elif cohort == 'mz48':
                cache = self.new_lookup[i]
                if cache >= 0:
                    crop[j] = self.crop[cache]; self.stats['crop_cache_eval_hits'] += 1
                else: crop_missing.append(j)
            else: crop_missing.append(j)
            if cache >= 0:
                full[j] = self.full[cache]; self.stats['full_cache_eval_hits'] += 1
            else: full_missing.append(j)
            if j in full_missing or j in crop_missing:
                images[j] = self.store.load(rows['rgb_refs'][int(i)])
        for missing, target, is_crop in ((full_missing, full, False), (crop_missing, crop, True)):
            if missing: target[missing] = self.extract([images[j] for j in missing], crop=is_crop)
        return dict(CROP_RASTER=crop, FULL_RASTER=full)

    def close(self):
        if getattr(self, 'full', None) is not None:
            self.full.flush(); self.full._mmap.close(); self.full = None
        self.crop = None


def outputs(model, dense, ranges, valid, mean, std):
    result = model.inspect((torch.from_numpy(dense).cuda()-mean)/std,
        torch.from_numpy(ranges).cuda(), torch.from_numpy(valid).cuda())
    answer={}
    for key,mask in (('OPEN',model.candidate_mask),('CROP',model.crop_mask)):
        field = result['field'].masked_fill(~mask[None, :, :, None], -torch.inf)
        pooled = result[key]
        answer[key]=dict(raw=pooled['raw'].cpu().numpy(), support=pooled['support'].cpu().numpy(),
                         winner=field.flatten(1,2).argmax(1).cpu().numpy().astype(np.int16))
    return answer


def run(root, task):
    started = time.perf_counter(); out=task/'run-v1'; out.mkdir(parents=True, exist_ok=False)
    scratch=task/'scratch-v1'; scratch.mkdir(exist_ok=False)
    bundle=features=store=None
    try:
        bind=Bindings(); setup()
        for name in ('MZ54_FULL_RGB_20260911.md','mz54_full_rgb.py','mz54_full_rgb_model.py',
                     'mz54_full_rgb_source.py','mz51_training_coverage.py','mz16_detail_cache.py',
                     'mz36_frozen_inference.py','mz15_train.py','mz40_packets.py'):
            bind(Path(__file__).with_name(name))
        work=root/'artifacts.local/work'
        bundle=prepare(root,bind); p=bundle['predictions']; groups=bundle['groups']; schedule=bundle['schedule']
        # Bind sealed scalar/composition reports too: scores remain frozen comparators.
        for eid in ('mz51-training-coverage-20260911','mz53-dual-readout-union-20260911'):
            receipt=read(bind(work/eid/'score-v1/receipt.json'))
            bind(work/eid/'score-v1/result.json',receipt['outputs']['result.json'])
        np.savez_compressed(out/'schedule.npz',**schedule)
        write(out/'groups.json',dict(records=bundle['records'],groups={k:np.asarray(v).tolist() for k,v in groups.items()}))
        initial_path=bound_outputs(bind,work/'mz50-echo-independent-local-20260911/run-v1',['spatial.pt'])[0]
        initial=torch.load(initial_path,map_location='cpu',weights_only=True)
        models={arm:RasterQuery(initial,arm).cuda() for arm in ARMS}
        identities={arm:parameters_sha(m) for arm,m in models.items()}
        assert len(set(identities.values()))==1
        assert all(sum(v.numel() for v in m.parameters())==10708 for m in models.values())
        torch.save({k:v.detach().cpu() for k,v in models[ARMS[0]].named_parameters()},out/'initial-weights.pt')
        base,mean,std=make_encoder(root,bind)
        store=RGBStore(); features=Features(bundle,base,store,scratch,out)
        write(out/'start.json',dict(status='STARTED',steps_per_arm=600,total_steps=1200,
            arms=ARMS,initial_parameter_sha256=identities,initial_checkpoint_sha256=sha(initial_path),
            backend='CUDA',device=torch.cuda.get_device_name(),inputs=bind.inputs))
        feature_seconds=features.build(); fits={}
        for arm,model in models.items():
            torch.manual_seed(151);model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            torch.cuda.synchronize();tick=time.perf_counter();history=[]
            for step,ids in enumerate(schedule['shared']):
                condition=CONDITIONS[step%3]
                def forward(ii,old=False):
                    rr,vv=(bundle['old_ranges'],bundle['old_valid']) if old else (bundle['cohorts']['mz48']['ranges'],bundle['cohorts']['mz48']['valid'])
                    obs=packet(rr[ii],vv[ii],condition);dense=features.training(arm,ii,old)
                    return model.inspect((torch.from_numpy(dense).cuda()-mean)/std,
                        torch.from_numpy(obs['ranges']).cuda(),torch.from_numpy(obs['valid']).cuda())
                result=forward(ids)
                native,parts=loss_for(result,torch.from_numpy(bundle['cell_truth'][ids]).cuda(),
                    torch.from_numpy(bundle['cell_known'][ids]).cuda(),torch.from_numpy(p['mz48/truth'][ids]).cuda(),
                    torch.from_numpy(p['mz48/known'][ids]).cuda())
                extra=forward(schedule['OLD_NEG'][step],old=True)
                negative=negative_loss(extra,torch.from_numpy(schedule['query'][step]).cuda())
                loss=native+.25*negative;assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step()
                if step%100==0 or step==599:
                    row=dict(step=step+1,condition=condition,loss=float(loss.detach()),
                        native=float(native.detach()),negative=float(negative.detach()),
                        **{k:float(v.detach()) for k,v in parts.items()})
                    history.append(row);write(out/'progress.json',dict(stage='fit',arm=arm,**row));print(arm,row,flush=True)
            torch.cuda.synchronize();model.eval()
            fits[arm]=dict(steps=600,seconds=time.perf_counter()-tick,history=history)
            torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/(arm+'.pt'))
            del optimizer,result,extra,loss,native,negative
        full_model=models['FULL_RASTER']
        for key in ('crop_mask','sensor_coverage','rays'):
            p['mz54/'+key]=getattr(full_model,key).cpu().numpy()
        prediction_parts={};tick=time.perf_counter()
        with torch.inference_mode():
            for cohort in COHORTS:
                data=bundle['cohorts'][cohort]; n=len(data['ranges'])
                for begin in range(0,n,16):
                    ids=np.arange(begin,min(begin+16,n));dense=features.evaluation(cohort,ids)
                    for condition in CONDITIONS:
                        obs=packet(data['ranges'][ids],data['valid'][ids],condition)
                        for arm in ARMS:
                            both=outputs(models[arm],dense[arm],obs['ranges'],obs['valid'],mean,std)
                            modes={arm:both['OPEN']}
                            if arm=='FULL_RASTER':modes['FULL_CROP_ONLY']=both['CROP']
                            for mode,vals in modes.items():
                                prefix=cohort+'/'+condition+'/MZ54/'+mode+'/'
                                for key,value in vals.items():prediction_parts.setdefault(prefix+key,[]).append(value)
                    if begin%512==0 or begin+len(ids)==n:
                        row=dict(stage='streamed-evaluation',cohort=cohort,frames=begin+len(ids),total=n)
                        write(out/'progress.json',row);print('EVAL',row,flush=True)
        inference_seconds=time.perf_counter()-tick
        for key,values in prediction_parts.items():p[key]=np.concatenate(values)
        del prediction_parts
        flat=bundle['cell_truth'].reshape(2560,3600,4)
        for condition in CONDITIONS:
            for mode in MODES:
                key='mz48/'+condition+'/MZ54/'+mode+'/'
                p[key+'winning_native']=np.take_along_axis(flat,p[key+'winner'][:,None],1)[:,0]&p[key+'support']
        cutoffs={}
        truth=np.concatenate([p['DEV/truth'],p['mz48/truth'][groups['calibration']]])
        for arm in ARMS:
            def collect(key):return np.concatenate([p['DEV/DROP_CLOSE/'+key],p['mz48/DROP_CLOSE/'+key][groups['calibration']]])
            key='MZ54/'+arm+'/'
            cutoffs[arm]=cutoff_zero_added(collect(key+'raw'),collect(key+'support'),collect('MZ37'),truth)
            np.save(out/(arm+'-cutoff.npy'),cutoffs[arm]);fits[arm]['cutoff']=cutoffs[arm].tolist()
        for cohort in COHORTS:
            for condition in CONDITIONS:
                prefix=cohort+'/'+condition+'/'
                for mode in MODES:
                    key=prefix+'MZ54/'+mode+'/'
                    cut=cutoffs['FULL_RASTER' if mode=='FULL_CROP_ONLY' else mode]
                    margin=p[key+'raw'].astype(float)-cut
                    accepted=(p[prefix+'MZ37']<0)&p[key+'support']&(margin>=0)
                    p[key+'candidate']=np.where(accepted,margin,p[prefix+'MZ37'])
        assert features.stats['full_extracted_frames']==10517
        assert features.stats['crop_extracted_frames']==3000  # includes16 TRAIN replay rows
        np.savez_compressed(out/'predictions.npz',**p)
        bind.check()
        receipt=dict(status='PASS',inputs=bind.inputs,steps_per_arm=600,total_steps=1200,fits=fits,
            exact_initial_weights=True,initial_parameter_sha256=identities,trainable_parameters=10708,
            seconds=time.perf_counter()-started,training_feature_seconds=feature_seconds,
            streamed_inference_seconds=inference_seconds,backend='CUDA; CPU SHA/PNG/compact metadata',
            device=torch.cuda.get_device_name(),feature_counts=features.stats,
            rgb_loads=store.loads,rgb_bytes_read=store.bytes_read,
            transient_full_cache_bytes=int(features.full.nbytes),new_crop_ram_bytes=int(features.crop.nbytes),
            old_crop_mmap_bytes=int(bundle['old_maps'].nbytes),new_baseline_inferences=0,
            new_cutoffs=2,threshold_searches=0,native_depth_reads=0,permanent_dense_cache=False,
            crop_candidate_cells=756,full_candidate_cells=3600,original_sensor_field_degrees=45,
            source_unknown_preserved=True,inherited_single_fit_nondeterminism=True,
            outputs={f.name:sha(f) for f in out.iterdir() if f.is_file()})
        write(out/'receipt.json',receipt);print('PASS',dict(seconds=receipt['seconds'],fits=fits),flush=True)
    except BaseException:
        write(task/'failure.json',dict(status='FAIL',error=traceback.format_exc()))
        raise
    finally:
        if features is not None:features.close()
        if store is not None:store.close()
        if bundle is not None and hasattr(bundle.get('old_maps'),'_mmap'):bundle['old_maps']._mmap.close()
        gc.collect()
        write(task/'handle-release.json',dict(status='PASS',mmap_handles_closed=True,rgb_store_closed=True,
            scratch_retained_for_owned_postverification_cleanup=str(scratch),process_exit_required=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.task.resolve())
