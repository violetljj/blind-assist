"""RGB-only original NVlabs FoundationStereo; no evaluator imports or readout."""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import types
from dataclasses import asdict


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def disparity_to_depth(disp, rig):
    import numpy as np
    fx = rig['width']/(2*math.tan(math.radians(rig['hfov_deg']/2)))
    positive = np.isfinite(disp) & (disp > 0)
    raw = np.full(disp.shape, np.nan, np.float32)
    raw[positive] = fx*rig['baseline_m']/disp[positive]
    visible = np.arange(rig['width'])[None, :] - disp >= 0
    keep = positive & visible & (raw >= .5) & (raw <= 4.)
    return raw, np.where(keep, raw, np.nan).astype(np.float32)


def load_model(source, checkpoint, cfg, torch, np):
    # Execute the exact two upstream definitions, avoiding unrelated open3d GUI imports.
    tree = ast.parse((source/'Utils.py').read_text(encoding='utf-8'))
    names = {'freeze_model', 'get_resize_keep_aspect_ratio'}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {n.name for n in definitions} == names
    helper = types.ModuleType('Utils'); helper.__dict__.update(np=np, torch=torch)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source/'Utils.py'), 'exec'), helper.__dict__)
    sys.modules['Utils'] = helper
    sys.path.insert(0, str(source))
    import timm
    factory, hubload = timm.create_model, torch.hub.load
    def create_model(name, *args, **kwargs):
        assert name == 'edgenext_small'
        kwargs['pretrained'] = False
        return factory(name, *args, **kwargs)
    def load_hub(repo, model, *args, **kwargs):
        assert repo == 'facebookresearch/dinov2' and model == 'dinov2_vitl14'
        assert kwargs.get('pretrained') is False
        kwargs['source'] = 'local'
        return hubload(str(source/'dinov2'), model, *args, **kwargs)
    timm.create_model, torch.hub.load = create_model, load_hub
    try:
        from core.foundation_stereo import FoundationStereo
        model = FoundationStereo(cfg)
    finally:
        timm.create_model, torch.hub.load = factory, hubload
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    info = dict(global_step=ckpt.get('global_step'), epoch=ckpt.get('epoch'))
    state = ckpt['model']
    model.load_state_dict(state, strict=True)
    info['parameters'] = sum(p.numel() for p in model.parameters())
    del state, ckpt
    gc.collect()
    model.eval().cuda()
    return model, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--canary', action='store_true')
    args = p.parse_args()
    assert json.loads(Path(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL']).read_text(encoding='utf-8-sig'))['state'] == 'running'
    plan = json.loads(args.config.read_text()); out = args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo))
    from tools.research_backend import torch_observation
    assert out.resolve().is_relative_to((repo/'artifacts.local').resolve())
    cache = Path(plan['cache'])
    for key, suffix in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('TMP', 'tmp'), ('TEMP', 'tmp')]:
        dest = cache/suffix; dest.mkdir(parents=True, exist_ok=True); os.environ[key] = str(dest)
    os.environ['XFORMERS_DISABLED'] = '1'
    source, checkpoint = Path(plan['source']), Path(plan['checkpoint'])
    assert sha(checkpoint) == plan['checkpoint_sha256']
    assert sha(checkpoint.parent/'cfg.yaml') == plan['cfg_sha256']
    for name, digest in plan['source_hashes'].items(): assert sha(source/name) == digest, name
    manifest = json.loads(args.inputs.read_text()); rig = manifest['rig']; rows = manifest['frames']
    assert len(rows) == 576 and rig['width'] == 640 and rig['height'] == 360
    assert rig['hfov_deg'] == 70 and rig['baseline_m'] == .1
    assert manifest['authority'] == 'RGB_AND_CALIBRATION_ONLY'
    required = {'panel', 'id', 'left', 'right', 'left_sha256', 'right_sha256'}
    for row in rows:
        assert set(row) == required
        for key in ('panel', 'id'): assert row[key] and not any(v in row[key] for v in ('/', '\\', '..'))
        for view in ('left', 'right'): assert sha(row[view]) == row[view+'_sha256']
    if args.canary:
        wanted = {(r['panel'], r['id']) for r in plan['canary']}
        rows = [r for r in rows if (r['panel'], r['id']) in wanted]
        assert len(rows) == len(wanted) == 3
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    cfg = OmegaConf.load(checkpoint.parent/'cfg.yaml')
    cfg.vit_size = 'vitl'; cfg.valid_iters = 32; cfg.low_memory = plan['low_memory']
    assert cfg.max_disp == 416 and cfg.mixed_precision is True
    started = time.perf_counter(); hashes = {}; timings = []; checks = []
    receipt = dict(status='RUNNING', frames=0, input_manifest_sha256=sha(args.inputs),
                   config_sha256=sha(args.config), producer_sha256=sha(__file__))
    try:
        model, checkpoint_info = load_model(source, checkpoint, cfg, torch, np)
        from core.utils.utils import InputPadder
        load_s = time.perf_counter()-started
        torch.cuda.reset_peak_memory_stats()
        for i, row in enumerate(rows):
            pair_start = time.perf_counter()
            images = [np.array(Image.open(row[view]).convert('RGB')) for view in ('left', 'right')]
            assert all(im.shape == (360, 640, 3) for im in images)
            x = [torch.as_tensor(im, device='cuda').float()[None].permute(0, 3, 1, 2) for im in images]
            padder = InputPadder(x[0].shape, divis_by=32, force_square=False)
            left, right = padder.pad(*x)
            assert tuple(left.shape) == (1, 3, 384, 640)
            assert torch.equal(padder.unpad(left), x[0]) and torch.equal(padder.unpad(right), x[1])
            torch.cuda.synchronize(); begin = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                output = model.forward(left, right, iters=32, test_mode=True)
            assert output.is_cuda and next(model.parameters()).is_cuda
            observed_backend = asdict(torch_observation(model=model, output=output))
            torch.cuda.synchronize(); inference_s = time.perf_counter()-begin
            disp = padder.unpad(output.float()).cpu().numpy().reshape(360, 640)
            if not (np.isfinite(disp) & (disp > 0)).any(): raise RuntimeError('No positive finite disparity')
            raw, depth = disparity_to_depth(disp, rig)
            for folder, values in [('raw_disparity', disp), ('raw_depth', raw), ('depth', depth)]:
                dest = out/row['panel']/folder/(row['id']+'.npy'); dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(dest, values, allow_pickle=False); hashes[dest.relative_to(out).as_posix()] = sha(dest)
            times = dict(panel=row['panel'], id=row['id'], inference_s=inference_s, pair_s=time.perf_counter()-pair_start)
            timings.append(times)
            checks.append(dict(panel=row['panel'], id=row['id'], positive_disparity=int((np.isfinite(disp)&(disp>0)).sum()),
                valid_depth=int(np.isfinite(depth).sum()), disparity_quantiles=np.quantile(disp[np.isfinite(disp)], [.05,.5,.95]).tolist()))
            receipt['frames'] = i+1
            write(out/'progress.json', dict(frames=i+1, total=len(rows), **times))
            if i % 24 == 0: print('FOUNDATION', i+1, '/', len(rows), round(inference_s,3), flush=True)
            del output, left, right, x
        write(out/'manifest.json', dict(model='NVlabs original FoundationStereo23-51-11 ViT-Large',
            checkpoint_info=checkpoint_info, configuration=plan, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), iterations=32, max_disp=416,
            observed_backend=observed_backend, placement='GPU_FIRST',
            preprocessing='RGB0..255; official internal normalization; original640x360 pad384x640 then unpad; no resize',
            compatibility='exact upstream Utils helpers; bundled local DINO; redundant EdgeNext pretrain disabled before strict full checkpoint load; official xFormers-disabled attention',
            output='raw disparity and axial depth preserved; common historical filtered depth has only right-visibility and0.5..4m masks',
            load_s=load_s, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            timings=timings, numeric_checks=checks))
        hashes['manifest.json'] = sha(out/'manifest.json')
        receipt.update(status='PASS', hashes=hashes, elapsed_s=time.perf_counter()-started,
                       inference_mean_s=float(np.mean([r['inference_s'] for r in timings])),
                       pair_mean_s=float(np.mean([r['pair_s'] for r in timings])))
        write(out/'receipt.json', receipt)
    except BaseException as exc:
        receipt.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), hashes=hashes,
                       elapsed_s=time.perf_counter()-started)
        write(out/'receipt.json', receipt)
        raise


if __name__ == '__main__': main()
