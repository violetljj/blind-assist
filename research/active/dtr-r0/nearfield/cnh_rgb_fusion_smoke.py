"""One-frame Development RGB/CNH input and untrained forward interface check.

Only public observations are read. The geometry-derived target file is not an
input, and arbitrary model outputs are deliberately omitted from the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cnh_rgb_frustum import zone_rgb_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(bundle, output):
    from PIL import Image
    import torch
    from cnh_rgb_fusion import FrustumFusion

    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError('Fresh smoke receipt required')
    capture = json.loads((bundle / 'raw-manifest.json').read_text(encoding='utf-8-sig'))
    transport = json.loads((bundle / 'format-receipt.json').read_text(encoding='utf-8-sig'))
    materialized = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8-sig'))
    camera = json.loads((bundle / 'camera.json').read_text(encoding='utf-8-sig'))
    raw_row, transport_row, row = (source['frames'][0] for source in (capture, transport, materialized))
    identity = lambda r: (r['id'], r['layout_id'], r['clip_id'], r['pose_index'])
    if identity(raw_row) != identity(transport_row) or identity(row) != identity(raw_row):
        raise ValueError('Capture/materialized first-frame identity differs')
    if any(r['data_role'] != 'Development' for r in (raw_row, transport_row, row)):
        raise ValueError('Development frame only')
    if transport.get('status') != 'PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        raise ValueError('Finalized native RGB transport required')
    if sha(bundle / 'camera.json') != row['camera_sha256'] or sha(bundle / 'camera.json') != transport_row['hashes']['camera.json']:
        raise ValueError('Camera hash differs')
    if sha(bundle / 'left.png') != transport_row['hashes']['left.png']:
        raise ValueError('Native RGB hash differs')
    with np.load(bundle / 'observations.npz', allow_pickle=False) as obs:
        if str(obs['frame_key'][0]) != row['frame_key']:
            raise ValueError('CNH frame identity differs')
        histogram = obs['histogram'][0]
        ambient = obs['ambient'][0]
        distance = obs['distance_m'][0]
        valid = obs['valid'][0]
    if histogram.shape != (8, 8, 16) or ambient.shape != (8, 8) or distance.shape != (8, 8) or valid.shape != (8, 8):
        raise ValueError('H3 observation schema mismatch')
    if not np.isfinite(histogram).all() or not np.isfinite(ambient).all():
        raise ValueError('Nonfinite CNH observation')
    if camera['width'] != 640 or camera['height'] != 360:
        raise ValueError('Unexpected fixed RGB resolution')
    with Image.open(bundle / 'left.png') as image:
        rgb = np.asarray(image.convert('RGB').resize((320, 180), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    k = np.asarray(camera['K'], dtype=float).copy()
    k[0, 0] *= .5
    k[1, 1] *= .5
    k[0, 2] = (k[0, 2] + .5) * .5 - .5
    k[1, 2] = (k[1, 2] + .5) * .5 - .5
    tof_from_camera = np.linalg.inv(np.asarray(camera['T_camera_tof'], dtype=float))
    support = zone_rgb_support(k, tof_from_camera, (320, 180), (45, 80))
    if not (support.sum((1, 2)) > 0).all():
        raise ValueError('A ToF zone has no RGB support')
    torch.manual_seed(20260924)
    model = FrustumFusion(64, 16).eval()
    with torch.no_grad():
        result = model(
            torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None],
            torch.from_numpy(histogram.reshape(1, 64, 16).copy()),
            torch.from_numpy(ambient.reshape(1, 64).copy()),
            torch.from_numpy(distance.reshape(1, 64).copy()),
            torch.from_numpy(valid.reshape(1, 64).copy()),
            torch.zeros((1, 64)),
            torch.from_numpy(support),
            mode='cnh',
        )
    if result['occupancy_logits'].shape != (1, 6) or not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Untrained fused forward is invalid')
    receipt = dict(status='PASS_REAL_DEVELOPMENT_INPUT_INTERFACE_ONLY', frame_key=row['frame_key'],
        layout_id=row['layout_id'], source_role='Development',
        rgb_sha256=sha(bundle / 'left.png'), camera_sha256=sha(bundle / 'camera.json'),
        observations_sha256=sha(bundle / 'observations.npz'),
        frustum_code_sha256=sha(Path(__file__).with_name('cnh_rgb_frustum.py')),
        model_code_sha256=sha(Path(__file__).with_name('cnh_rgb_fusion.py')),
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        support_nonzero_patches=int((support > 0).sum()),
        support_total_patches=int(support.size), query_count=6,
        input_modalities=['native_left_RGB', 'H3_synthetic_CNH', 'ambient', 'same_response_scalar_status'],
        result_scope='Untrained one-frame input/forward smoke; no task performance, calibration or hardware claim',
        evaluator_truth_read=False)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.bundle, args.output), allow_nan=False))
