"""Frozen models on external observation packets and unchanged RGB.

Native depth and evaluator labels are not read by this runner. Packet generation
owns source depth; this adapter binds packets and preserves the RGB/model path.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from body_query_range import range_from_counts
from representation_model import pretrained_manifest
from multizone64_observation import EVENT_ORDER
from mz37_restore import compose
from mz5_ensemble_readout import CompactEnsemble, read, write, sha, load_npz
from mz16_detail_cache import dense_features, CROP
from mz15_shared_support import LocalSupportReadout
from mz23_availability import restrict_candidates
from mz28_packet_availability import PacketAvailability
from mz30_select import BranchSelector, negative_branch_confidence

CAMERA = dict(width=640, height=360, hfov_deg=100, eye_height_m=1.7, pitch_deg=0, roll_deg=0)
BATCH_SIZE = 16


def fixed_batch_dense(model, rgb, legacy=False):
    """Match the frozen cache's backbone batch shape for short final batches.

    Evaluation-mode convolutions are independent of other images, but CUDA
    arithmetic can change with batch shape. Repeat the last image, then discard
    padding. No new source images or labels are used.
    """
    n = len(rgb)
    assert 0 < n <= BATCH_SIZE
    if n < BATCH_SIZE:
        rgb = torch.cat([rgb,rgb[-1:].expand(BATCH_SIZE-n,-1,-1,-1)])
    return dense_features(model,rgb,legacy=legacy)[:n]


def run(root, manifest_path, packet_path, output):
    root = root.resolve(); artifact_root = (root/'artifacts.local').resolve()
    output = output.absolute()
    if output.exists() or not output.resolve().is_relative_to(artifact_root):
        raise ValueError('Output must be a new path under the resolved artifacts.local tree')
    manifest_path = manifest_path.absolute(); manifest = read(manifest_path)
    if manifest['camera'] != CAMERA:
        raise ValueError('Frozen 640x360 HFOV100, eye1.7m, level-camera calibration required')
    frames = manifest['frames']
    if not frames or any(not isinstance(r['frame_id'], str) or not r['frame_id'] for r in frames):
        raise ValueError('Nonempty string frame IDs required')
    if len({r['frame_id'] for r in frames}) != len(frames):
        raise ValueError('Frame IDs must be unique')
    for row in frames:
        for key in ['rgb_path']:
            if not Path(row[key]).is_absolute() or not Path(row[key]).is_file():
                raise ValueError('An absolute existing raw file is required: '+str(row[key]))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); inputs = {}; frozen = {}; models = []
    source_names = [Path(__file__).name, 'mz5_ensemble_readout.py', 'mz16_detail_cache.py',
        'mz15_shared_support.py', 'mz8_attribution.py', 'mz23_availability.py',
        'mz28_packet_availability.py', 'mz30_select.py', 'multizone64_observation.py',
        'body_query_context_evidence.py', 'body_query_context_decoder.py', 'body_query_model.py',
        'body_query_range.py', 'body_query_fresh_size_eval.py', 'representation_model.py',
        'decoupled_model.py', 'whisker_model.py', 'contact_retina_spec.py', 'mz37_restore.py']
    code = {name: sha(Path(__file__).with_name(name)) for name in source_names}
    timings = dict(io_preprocess_cpu=0., upload_cuda=0., packet_cuda=0., full_rgb_cuda=0.,
                   crop_rgb_cuda=0., mz5_context_branches_cuda=0., mz28_bank_cuda=0.,
                   mz20_restricted_cuda=0., selector_cuda=0., download_cuda=0.)

    def timed(name, action):
        torch.cuda.synchronize(); tick = time.perf_counter(); value = action()
        torch.cuda.synchronize(); timings[name] += time.perf_counter()-tick
        return value

    def bind(path, expected=None, is_frozen=True):
        path = Path(path); digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Frozen input hash mismatch: '+str(path))
        (frozen if is_frozen else inputs)[str(path)] = digest
        return path

    def checkpoint(folder, name, receipt_name='receipt.json'):
        receipt = read(bind(folder/receipt_name))
        assert receipt['status'] == 'PASS'
        return bind(folder/name, receipt['outputs'][name])

    try:
        bind(manifest_path, is_frozen=False)
        for row in frames:
            bind(row['rgb_path'], is_frozen=False)
        packet_path = bind(packet_path, is_frozen=False)
        packets = load_npz(packet_path)
        np.testing.assert_array_equal(packets['frame_ids'], [r['frame_id'] for r in frames])
        assert packets['ranges'].shape == packets['valid'].shape == (len(frames),64,2)
        assert packets['ranges'].dtype == np.float32 and packets['valid'].dtype == np.bool_
        assert np.isfinite(packets['ranges']).all()
        assert (packets['ranges'][~packets['valid']] == 0).all()
        assert (packets['ranges'][packets['valid']] > 0).all()
        assert (packets['ranges'][packets['valid']] <= 4).all()
        bind(Path(__file__).with_name('MZ40_L8CX_CONSTRAINED_PROTOCOL_20260910.md'), is_frozen=False)
        work = root/'artifacts.local/work'
        base = work/'body-query-10000-b-20260909/run-v1'
        context_dir = work/'body-query-context-decoder-20260909/run-v1'
        pretrained = work/'body-query-v1-20260908/model-inputs/pretrained'
        for name, digest in FROZEN.items():
            bind((base if name == 'NEW-step2000.pt' else context_dir)/name, digest)
        for name in ['fits.json', 'receipt.json']:
            bind(context_dir/name)
        pm = pretrained_manifest(pretrained)
        for name in ['manifest.json', 'provenance.json', 'data_datasets.py']:
            if (pretrained/name).exists(): bind(pretrained/name)
        for key in ['source', 'checkpoint']:
            bind(pretrained/pm[key], pm[key+'_sha256'])
        compact_dir = work/'mz5-fixed-ensemble-20260910/compact-v1'
        compact_receipt = read(bind(compact_dir/'export-receipt.json'))
        assert compact_receipt['status'] == 'PASS'
        compact_path = bind(compact_dir/'compact.pt', compact_receipt['compact_sha256'])
        rank_dir = work/'mz20-rank-objective-20260910/run-v1'
        rank_path = checkpoint(rank_dir, 'BODY_RANK.pt')
        rank_cut = np.load(checkpoint(rank_dir, 'BODY_RANK-cutoff.npy'))
        bank_dir = work/'mz28-packet-availability-20260910/run-v1'
        bank_data = load_npz(checkpoint(bank_dir, 'bank.npz'))
        np.testing.assert_array_equal(rank_cut, np.load(checkpoint(bank_dir, 'cutoff.npy')))
        norm_dir = work/'mz9-source-supervision-20260910/run-v1'
        detail_norm = load_npz(checkpoint(norm_dir, 'normalization.npz'))
        selector_dirs = dict(MZ30=work/'mz30-branch-responsibility-20260910/run-v1',
                             MZ35=work/'mz35-responsibility-convergence-20260910/run-v1')
        selector_norm = load_npz(checkpoint(selector_dirs['MZ30'], 'normalization.npz'))
        selector_paths = {name: checkpoint(folder, 'selector.pt') for name, folder in selector_dirs.items()}
        cuts = {name: np.load(checkpoint(folder, 'cutoff.npy')) for name, folder in selector_dirs.items()}
        restore_dir = work/'mz37-positive-restoration-20260910/run-v1'
        restore_cut = np.load(checkpoint(restore_dir, 'cutoff.npy'))
        assert bank_data['ids'].shape == (7362,) and bank_data['packet'].shape == (7362,64,4)
        assert bank_data['known'].shape == (7362,64,49)
        assert torch.cuda.is_available()
        torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False
        # Match MZ15/MZ16 frozen RGB extraction settings. Enabling strict CUDA
        # determinism here can choose different convolution kernels/rounding.
        torch.backends.cudnn.deterministic = False; torch.use_deterministic_algorithms(False)
        context = ContextEvidence(base, context_dir, pretrained).cuda().eval(); models.append(context)
        compact = CompactEnsemble.from_checkpoint(compact_path).cuda().eval(); models.append(compact)
        rank = LocalSupportReadout(shared=False)
        rank.load_state_dict(torch.load(rank_path, map_location='cpu', weights_only=True))
        rank = rank.cuda().eval().requires_grad_(False); models.append(rank)
        bank = PacketAvailability(*(torch.from_numpy(bank_data[k]) for k in ['ids','packet','known'])).cuda().eval(); models.append(bank)
        selectors = {}
        for name, path in selector_paths.items():
            model = BranchSelector(); model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            selectors[name] = model.cuda().eval().requires_grad_(False); models.append(selectors[name])
        detail_mean, detail_std = (torch.from_numpy(detail_norm[k]).cuda() for k in ['mean','std'])
        selector_mean, selector_std = (torch.from_numpy(selector_norm[k]).cuda() for k in ['mean','std'])
        write(output/'start.json', dict(status='STARTED', frames=len(frames), training_steps=0,
            inputs=inputs, frozen=frozen, code_sha256=code, camera=CAMERA, event_order=EVENT_ORDER,
            batch_size=BATCH_SIZE, backbone_batch_size=BATCH_SIZE, short_batch_padding='Repeat final image; discard padded outputs',
            crop_xyxy=CROP, cutoff=dict(MZ20=rank_cut.tolist(), **{k:v.tolist() for k,v in cuts.items()}),
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                deterministic=False, cudnn_deterministic=False, matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
                cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG']),
            interpretation='External constrained packet; no current depth/event/source/query/known labels read. Frozen learned models and ideal-trained bank.'))
        pieces = []
        with torch.inference_mode():
            for begin in range(0,len(frames),BATCH_SIZE):
                rows = frames[begin:begin+BATCH_SIZE]; low = []; high = []; tick = time.perf_counter()
                for row in rows:
                    with Image.open(row['rgb_path']) as image:
                        assert image.size == (640,360); image = image.convert('RGB')
                        low.append(np.array(image.resize((256,144),Image.Resampling.BOX)))
                        high.append(np.array(image.crop(CROP)))
                low = np.stack(low); high = np.stack(high)
                timings['io_preprocess_cpu'] += time.perf_counter()-tick
                def upload():
                    return (torch.from_numpy(low).permute(0,3,1,2).cuda().float()/255,
                            torch.from_numpy(high).permute(0,3,1,2).cuda().float()/255,
                            torch.from_numpy(packets['ranges'][begin:begin+len(rows)]).cuda(),
                            torch.from_numpy(packets['valid'][begin:begin+len(rows)]).cuda())
                low_gpu, high_gpu, ranges, valid = timed('upload_cuda',upload)
                full_dense = timed('full_rgb_cuda',lambda: fixed_batch_dense(context.base,low_gpu,legacy=True))
                high_dense = timed('crop_rgb_cuda',lambda: fixed_batch_dense(context.base,high_gpu))
                assert tuple(full_dense.shape[1:]) == (64,18,32) and tuple(high_dense.shape[1:]) == (64,28,28)
                def branches():
                    m = context.base
                    sampled = (full_dense.flatten(2)@m.query_projection.T).transpose(1,2).reshape(len(rows),12,27,64)
                    qmask = m.query_valid[None,:,:,None]
                    pooled = (sampled*qmask).sum(2)/qmask.sum(2).clamp_min(1)
                    normalized = (pooled-context.feature_mean)/context.feature_std
                    visual = torch.cat([normalized.flatten(1),range_from_counts(context.decoder(normalized)).sigmoid().flatten(1)],1)
                    tof = torch.cat([ranges.flatten(1)/4,valid.flatten(1).float()],1)
                    rh = compact.rgb[:2](visual); th = compact.tof[:2](tof)
                    rgb = compact.rgb[2](rh); tof_logits = compact.tof[2](th)
                    return visual, rgb, tof_logits, torch.cat([rh,th,rgb,tof_logits],1), (rgb+tof_logits)*.5
                visual,rgb,tof_logits,features,baseline = timed('mz5_context_branches_cuda',branches)
                availability = timed('mz28_bank_cuda',lambda: bank(ranges,valid))
                def local():
                    original = rank.inspect((high_dense-detail_mean)/detail_std,ranges,valid)
                    return original, restrict_candidates(original,availability['availability'])
                original,restricted = timed('mz20_restricted_cuda',local)
                def select():
                    normalized = (features-selector_mean)/selector_std
                    return {name: (model(normalized)) for name,model in selectors.items()}
                selector_logits = timed('selector_cuda',select)
                def download():
                    out = dict(ranges=ranges,valid=valid,visual=visual,features=features,rgb=rgb,tof=tof_logits,
                        MZ5=baseline,original_support=original['support'],original_raw=original['logits'],
                        restricted_support=restricted['support'],restricted_raw=restricted['logits'],
                        votes=availability['votes'],neighbor_ids=availability['neighbor_ids'],neighbor_distances=availability['neighbor_distances'])
                    for name,z in selector_logits.items():
                        out[name+'/selector_logits']=z
                        out[name+'/confidence']=negative_branch_confidence(rgb,z)
                    return {key:value.cpu().numpy() for key,value in out.items()}
                a = timed('download_cuda',download)
                margin = np.where(a['restricted_support'],a['restricted_raw'].astype(float)-rank_cut,-1e6)
                accepted = (a['MZ5']<0)&a['restricted_support']&(margin>=0)
                a['MZ28'] = np.where(accepted,margin,a['MZ5']); a['MZ28/accepted']=accepted; a['MZ28/margin']=margin
                eligible = (a['MZ5']>=0)&~a['original_support']&((a['rgb']>=0)!=(a['tof']>=0))
                a['selector_eligible']=eligible
                negative = np.where(a['rgb']<0,a['rgb'],a['tof'])
                for name in selectors:
                    remove = eligible&(a[name+'/confidence'].astype(float)>=cuts[name])
                    a[name]=np.where(remove,negative,a['MZ28']); a[name+'/remove']=remove
                    assert not ((a[name]>=0)&(a['MZ28']<0)).any()
                    np.testing.assert_array_equal(a[name][~eligible],a['MZ28'][~eligible])
                restoration = compose(dict(rgb=a['rgb'],tof=a['tof'],baseline=a['MZ5'],
                    oldscore=a['MZ35'],support=a['original_support'],known=np.ones_like(a['MZ35'],bool),
                    negative_confidence=a['MZ35/confidence']), restore_cut)
                a['MZ37'] = restoration['candidate']; a['MZ37/added'] = restoration['added']
                pieces.append(a)
                write(output/'progress.json',dict(frames=begin+len(rows),total=len(frames),training_steps=0))
                print('INFERENCE',begin+len(rows),'/',len(frames),flush=True)
        arrays = {key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        arrays['frame_ids'] = np.array([r['frame_id'] for r in frames])
        np.savez_compressed(output/'predictions.npz',**arrays)
        for path,expected in {**inputs,**frozen}.items():
            assert sha(path)==expected,'Input changed: '+path
        write(output/'result.json',dict(frames=len(frames),training_steps=0,event_order=EVENT_ORDER,
            timing_seconds=timings,total_seconds=time.perf_counter()-started,
            positive_bits={name:(arrays[name]>=0).sum(0).tolist() for name in ['MZ5','MZ28','MZ30','MZ35','MZ37']},
            evaluator_labels_read=False,bank_rebuilt=False,thresholds_calibrated=False,
            limits='No event accuracy is computed. External packets are declared sensitivity proxies, not calibrated device emulation; absent evidence is not CLEAR.'))
        write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen raw-RGB and external-packet inference',
            inputs=inputs,frozen=frozen,code_sha256=code,timing_seconds=timings,
            outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        print('PASS',output,flush=True)
    except Exception:
        write(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),training_steps=0,
            inputs=inputs,frozen=frozen,code_sha256=code))
        raise
    finally:
        for model in models: model.cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.manifest,args.packet,args.output)
