"""Supplementary fixed four-frame A/A rendering-noise inference; no fitting."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from head_paired_evaluate import CHECKPOINTS, read, sha, write


def run(a):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    diagnostic = read(a.diagnostic/'result.json')
    receipt = read(a.diagnostic/'receipt.json')
    assert diagnostic['status'] == receipt['status'] == 'PASS'
    assert sha(a.diagnostic/'result.json') == receipt['result_sha256']
    for name, digest in diagnostic['source_sha256'].items():
        assert sha(a.source/name) == digest
    sys.path.insert(0, str(a.source))
    import torch
    from decoupled_model import DecoupledModel
    assert torch.cuda.is_available() and torch.__version__ == '2.9.1+cu128'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    spec = read(a.capture/'source/spec.json')
    cases = spec['cases']
    assert len(cases) == 4 and all('-A-repeat' in c['name'] for c in cases)
    frames = [a.capture/f'model/sample/{i:04}.png' for i in range(4)]
    rgb = np.stack([np.asarray(Image.open(p).convert('RGB').resize((256, 144), Image.Resampling.BOX)) for p in frames])
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255).repeat(8, 1, 1, 1).cuda()
    a.output.mkdir(parents=True, exist_ok=False)
    work = a.artifact_root/'work'
    model = DecoupledModel(work/'city-finetune-pilot-20260908/prepared-v1/payload/pretrained').cuda().eval()
    result = dict(status='PASS', training_steps=0, unique_frames=4, batch=32,
        batch_note='Four fixed frames repeated eight times to preserve nominal batch32; only first four scored.',
        spec_sha256=sha(a.capture/'source/spec.json'), input_sha256={str(p): sha(p) for p in frames},
        runner_sha256=sha(__file__), diagnostic_result_sha256=sha(a.diagnostic/'result.json'), models={})
    with torch.inference_mode():
        for arm, (relative, digest) in CHECKPOINTS.items():
            checkpoint = work/relative
            assert sha(checkpoint) == digest
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            n, m = model(x)
            p, s, logits = n.sigmoid().cpu().numpy()[:4], m.sigmoid().cpu().numpy()[:4], n.cpu().numpy()[:4]
            np.savez_compressed(a.output/(arm+'-predictions.npz'), near=p, support=s, near_logits=logits)
            cutoff = np.array(diagnostic['identities'][arm]['DEV_thresholds'])
            rows = []
            for i, j in ((0, 1), (2, 3)):
                rows.append(dict(names=[cases[i]['name'], cases[j]['name']],
                    near_logit_delta=(logits[j]-logits[i]).tolist(), near_probability_delta=(p[j]-p[i]).tolist(),
                    alert_flips=((p[i] >= cutoff) != (p[j] >= cutoff)).tolist(),
                    support_probability_MAE=np.abs(s[j]-s[i]).mean(axis=(1, 2)).tolist(),
                    support_probability_max_delta=np.abs(s[j]-s[i]).max(axis=(1, 2)).tolist()))
            result['models'][arm] = dict(checkpoint_sha256=digest, pairs=rows)
    write(a.output/'result.json', result)
    write(a.output/'receipt.json', dict(status='PASS', result_sha256=sha(a.output/'result.json'),
        optimizer_steps=0, torch_version=torch.__version__, device=torch.cuda.get_device_name()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'source', 'diagnostic', 'artifact-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
