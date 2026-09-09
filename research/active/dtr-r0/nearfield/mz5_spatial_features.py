"""Recache the frozen RGB query samples before averaging; no native-depth reads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from mz1_tiny_fusion import sha, write
from mz3_error_attribution import read, load

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'


def run(output):
    output = Path(output).resolve()
    allowed = (WORK / 'mz5-spatial-fusion-20260910').resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError('Use a fresh child of the task artifact directory')
    source = WORK / 'mz1-tiny-fusion-20260910/cache-v1'
    dataset = WORK / 'body-query-5000-20260909/dataset-v1'
    base = WORK / 'body-query-10000-b-20260909/run-v1'
    decoder = WORK / 'body-query-context-decoder-20260909/run-v1'
    manifest = read(dataset / 'manifest.json')
    fr = read(source / 'features-receipt.json')
    assert manifest['status'] == fr['status'] == 'PASS'
    assert sha(dataset / 'index.json') == manifest['index_sha256'] == fr['source_index_sha256']
    assert sha(source / 'features.npz') == fr['feature_sha256']
    rows = read(dataset / 'index.json')['frames']
    data = load(source / 'features.npz')
    assert len(rows) == 5000 and all(r['status'] == 'PASS' for r in rows)
    np.testing.assert_array_equal(data['role'], [r['source_role'] for r in rows])
    for name, digest in FROZEN.items():
        assert sha((base if name == 'NEW-step2000.pt' else decoder) / name) == digest
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # MZ1's frozen visual cache used the original cuDNN TF32-enabled default.
    # Match that extractor arithmetic; the new linear heads disable TF32 separately.
    torch.backends.cudnn.allow_tf32 = True
    output.mkdir(parents=True)
    write(output / 'start-receipt.json', dict(status='STARTED', utc=datetime.now(timezone.utc).isoformat(),
        source_index_sha256=fr['source_index_sha256'], feature_sha256=fr['feature_sha256'],
        source_sha256=sha(__file__), frozen_hashes=FROZEN, backend='CUDA'))
    shutil.copyfile(__file__, output / 'source.py')
    started = time.perf_counter()
    handle = None
    try:
        model = ContextEvidence(base, decoder, WORK / 'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
        captured = []
        handle = model.base.query_point.register_forward_pre_hook(
            lambda m, args: captured.append(args[0][..., :64].detach()))
        mask = model.base.query_valid[None, :, :, None]
        points = np.lib.format.open_memmap(output / 'points.npy', mode='w+', dtype=np.float32,
                                          shape=(5000, 12, 27, 64))
        maxerr = 0.
        for begin in range(0, 5000, 16):
            images = []
            for row in rows[begin:begin + 16]:
                path = Path(row['rgb_file'])
                assert sha(path) == row['rgb_sha256']
                with Image.open(path) as im:
                    images.append(np.array(im.convert('RGB').resize((256, 144), Image.Resampling.BOX)))
            x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).cuda().float() / 255.
            with torch.inference_mode():
                out = model(x)
                assert len(captured) == 1
                raw = captured.pop()
                normalized = (raw - model.feature_mean[:, :, None]) / model.feature_std[:, :, None]
                averaged = ((normalized * mask).sum(2) / mask.sum(2).clamp_min(1)).flatten(1)
                expected = torch.as_tensor(data['visual'][begin:begin+len(images), :768], device='cuda')
                maxerr = max(maxerr, float((averaged - expected).abs().max()))
                assert maxerr < 1e-4
                np.testing.assert_array_equal(out['alerts'].cpu(), data['original_alerts'][begin:begin+len(images)])
                np.testing.assert_array_equal((out['range_probabilities'] >= .5).flatten(1).cpu(),
                                               data['joint'][begin:begin+len(images)])
                np.testing.assert_allclose(out['range_probabilities'].flatten(1).cpu(),
                    data['visual'][begin:begin+len(images), 768:], rtol=0, atol=2e-6)
                points[begin:begin+len(images)] = normalized.cpu().numpy()
            if begin % 800 == 0:
                print(f'cached {begin+len(images)}/5000', flush=True)
        points.flush()
        del points
        np.savez_compressed(output / 'geometry.npz', valid=model.base.query_valid.cpu().numpy(),
            xyz=model.base.query_xyz.cpu().numpy(), grid=model.base.query_grid.cpu().numpy(),
            projection=model.base.query_projection.cpu().numpy())
        torch.cuda.synchronize()
        write(output / 'receipt.json', dict(status='PASS', frames=5000,
            points_sha256=sha(output / 'points.npy'), geometry_sha256=sha(output / 'geometry.npz'),
            source_index_sha256=fr['source_index_sha256'], original_features_sha256=fr['feature_sha256'],
            source_sha256=sha(__file__), frozen_hashes=FROZEN,
            averaged_feature_max_error=maxerr, original_alert_parity=5000, joint_flag_parity=5000,
            normalization='unchanged original JOINT mean/std, applied per point',
            backend='CUDA', device=torch.cuda.get_device_name(), cudnn_allow_tf32=True,
            seconds=time.perf_counter()-started,
            native_depth_read=False, training_steps=0))
        print('CACHE PASS', maxerr, time.perf_counter()-started, flush=True)
    except Exception as error:
        write(output / 'failure.json', dict(status='ENGINEERING_FAILED', type=type(error).__name__, message=str(error)))
        raise
    finally:
        if handle is not None:
            handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
