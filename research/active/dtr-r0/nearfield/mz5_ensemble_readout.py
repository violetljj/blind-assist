"""Compact frozen MZ5 readout; remove only always-zero input weight columns.

forward returns four averaged logits. Apply >=0 for event flags; retain original
B alerts independently. Inputs use the unchanged MZ1 cached feature contract.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class CompactEnsemble(nn.Module):
    """772->128->4 RGB and 256->128->4 ToF; fixed equal-logit mean."""
    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.Linear(772, 128), nn.ReLU(), nn.Linear(128, 4))
        self.tof = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))
        self.eval().requires_grad_(False)

    def branches(self, visual, tof):
        if visual.ndim != 2 or visual.shape[1] != 772 or tof.shape != (len(visual), 256):
            raise ValueError('Expected visual[B,772] and tof[B,256] from frozen MZ1 features')
        return self.rgb(visual), self.tof(tof)

    def forward(self, visual, tof):
        rgb_logits, tof_logits = self.branches(visual, tof)
        return 0.5 * (rgb_logits + tof_logits)

    @classmethod
    def from_original(cls, rgb_weights, tof_weights):
        model = cls()
        for branch, weights, columns in ((model.rgb, rgb_weights, slice(0, 772)),
                                         (model.tof, tof_weights, slice(772, 1028))):
            assert weights['layers.0.weight'].shape == (128, 1028)
            branch.load_state_dict({
                '0.weight': weights['layers.0.weight'][:, columns].clone(),
                '0.bias': weights['layers.0.bias'].clone(),
                '2.weight': weights['layers.2.weight'].clone(),
                '2.bias': weights['layers.2.bias'].clone()})
        return model

    @classmethod
    def from_checkpoint(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        return model


class DenseReference(nn.Module):
    """Unmodified dense weight layout and original input-zeroing operation."""
    def __init__(self, weights, arm):
        super().__init__()
        self.arm = arm
        self.layers = nn.Sequential(nn.Linear(1028, 128), nn.ReLU(), nn.Linear(128, 4))
        self.load_state_dict(weights)
        self.eval().requires_grad_(False)

    def forward(self, visual, tof):
        if self.arm == 'RGB_ONLY':
            tof = torch.zeros_like(tof)
        elif self.arm == 'TOF_ONLY':
            visual = torch.zeros_like(visual)
        return self.layers(torch.cat((visual, tof), 1))


class DenseEnsemble(nn.Module):
    def __init__(self, rgb, tof):
        super().__init__()
        self.rgb, self.tof = rgb, tof

    def forward(self, visual, tof):
        return 0.5 * (self.rgb(visual, tof) + self.tof(visual, tof))


def inputs(root):
    work = root/'artifacts.local/work'
    base = work/'mz1-tiny-fusion-20260910'
    return dict(features=base/'cache-v1/features.npz', feature_receipt=base/'cache-v1/features-receipt.json',
        predictions=base/'run-v1/predictions.npz', original_receipt=base/'run-v1/receipt.json',
        RGB_ONLY=base/'run-v1/RGB_ONLY.pt', TOF_ONLY=base/'run-v1/TOF_ONLY.pt', FUSION=base/'run-v1/FUSION.pt',
        historical_mz5=work/'mz5-fixed-ensemble-20260910/run-v1/predictions.npz',
        historical_receipt=work/'mz5-fixed-ensemble-20260910/run-v1/receipt.json',
        source=Path(__file__).resolve(), backend_source=root/'tools/research_backend.py')


def prepare(root, output):
    if output.exists():
        raise FileExistsError('Fresh compact export directory required')
    paths = inputs(root); hashes = {key: sha(path) for key, path in paths.items()}
    receipt, fr = read(paths['original_receipt']), read(paths['feature_receipt'])
    assert hashes['features'] == receipt['feature_sha256'] == fr['feature_sha256']
    assert hashes['predictions'] == receipt['predictions_sha256']
    for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION'):
        assert hashes[arm] == receipt['arms'][arm]['checkpoint_sha256']
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(status='FROZEN', method='Remove only known zero-input columns; no parameter fitting',
        input_sha256=hashes, input_paths={key: str(path) for key, path in paths.items()},
        check_rows=5000, check_devices=['CPU', 'CUDA'], expected_event_threshold=0,
        logit_tolerance=dict(atol=1e-4, rtol=1e-5), signs_must_match=True,
        benchmark=dict(batch_sizes=[1, 128], warmups=10, repeats=50, cpu_threads=1,
            models=['compact_ensemble', 'original_dense_ensemble', 'original_fusion'],
            input_resident=True, includes_device_sync=True, scope='readout-only; excludes RGB extraction, ToF observations and transfers'),
        training_steps=0, weights_changed='Only removal of mathematically inactive first-layer columns'))
    weights = {arm: torch.load(paths[arm], map_location='cpu', weights_only=True) for arm in ('RGB_ONLY', 'TOF_ONLY')}
    model = CompactEnsemble.from_original(weights['RGB_ONLY'], weights['TOF_ONLY'])
    counts = {name: sum(p.numel() for p in branch.parameters()) for name, branch in (('RGB', model.rgb), ('ToF', model.tof))}
    assert counts == dict(RGB=99460, ToF=33412)
    assert sum(counts.values()) == 132872
    torch.save(model.state_dict(), output/'compact.pt')
    restored = CompactEnsemble.from_checkpoint(output/'compact.pt')
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    write(output/'export-receipt.json', dict(status='PASS', backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE',
        compact_sha256=sha(output/'compact.pt'), protocol_sha256=sha(output/'protocol.json'),
        parameters=counts, total_parameters=sum(counts.values()), MACs_per_row=772*128+128*4+256*128+128*4,
        original_stored_parameters=264456, original_dense_MACs_per_row=264192,
        original_single_fusion_parameters=132228, original_single_fusion_MACs_per_row=132096,
        exported_values_identical=True, model_forward_passes=0, GPU_started=False))
    print('EXPORT PASS', counts, 'total', sum(counts.values()))


def validate_and_benchmark(root, output):
    if (output/'engineering-receipt.json').exists() or (output/'failure.json').exists():
        raise FileExistsError('Preserve prior completed/failed verification')
    started = time.perf_counter()
    protocol = read(output/'protocol.json'); paths = {key: Path(path) for key, path in protocol['input_paths'].items()}
    assert {key: sha(path) for key, path in paths.items()} == protocol['input_sha256']
    assert sha(output/'compact.pt') == read(output/'export-receipt.json')['compact_sha256']
    try:
        sys.path.insert(0, str(root))
        from tools.research_backend import BackendCandidate, Workload, select_backend, torch_observation
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA benchmark required; do not silently substitute CPU')
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        data, old, historical = (load_npz(paths[key]) for key in ('features', 'predictions', 'historical_mz5'))
        assert len(data['role']) == 5000
        np.testing.assert_array_equal(data['role'], old['role'])
        np.testing.assert_array_equal(data['truth'], old['truth'])
        np.testing.assert_array_equal(data['original_alerts'], old['original_alerts'])
        expected = dict(RGB=old['RGB_ONLY_logits'], ToF=old['TOF_ONLY_logits'],
            ENSEMBLE=0.5*(old['RGB_ONLY_logits']+old['TOF_ONLY_logits']))
        ev = data['role'] == 'EVAL_ONLY'
        np.testing.assert_array_equal(expected['ENSEMBLE'][ev], historical['ENSEMBLE_logits'])
        compact = CompactEnsemble.from_checkpoint(output/'compact.pt')
        dense = {arm: DenseReference(torch.load(paths[arm], map_location='cpu', weights_only=True), arm)
                 for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION')}
        models = dict(compact_ensemble=compact, original_dense_ensemble=DenseEnsemble(dense['RGB_ONLY'], dense['TOF_ONLY']), original_fusion=dense['FUSION'])
        placed = {device: {name: copy.deepcopy(model).to(device).eval() for name, model in models.items()} for device in ('cpu', 'cuda')}
        resident = {device: (torch.from_numpy(data['visual']).to(device), torch.from_numpy(data['tof']).to(device)) for device in ('cpu', 'cuda')}
        audit, arrays = {}, dict(original_alerts=data['original_alerts'], role=data['role'])
        with torch.inference_mode():
            for device in ('cpu', 'cuda'):
                vis, tof = resident[device]; model = placed[device]['compact_ensemble']
                pieces = dict(RGB=[], ToF=[], ENSEMBLE=[], DENSE_RGB=[], DENSE_ToF=[], DENSE_ENSEMBLE=[])
                for start in range(0, 5000, 256):
                    v, t = vis[start:start+256], tof[start:start+256]
                    r, d = model.branches(v, t)
                    er = placed[device]['original_dense_ensemble'].rgb(v, t)
                    ed = placed[device]['original_dense_ensemble'].tof(v, t)
                    for name, value in dict(RGB=r, ToF=d, ENSEMBLE=0.5*(r+d), DENSE_RGB=er, DENSE_ToF=ed, DENSE_ENSEMBLE=0.5*(er+ed)).items():
                        pieces[name].append(value.cpu().numpy())
                actual = {name: np.concatenate(values) for name, values in pieces.items()}
                device_audit = {}
                for name in ('RGB', 'ToF', 'ENSEMBLE'):
                    p, e = actual[name], expected[name]
                    diff = np.abs(p-e)
                    record = dict(max_absolute_error=float(diff.max()),
                        sign_mismatch_bits=int(((p >= 0) != (e >= 0)).sum()),
                        sign_mismatch_rows=int(((p >= 0) != (e >= 0)).any(1).sum()),
                        min_original_absolute_threshold_margin=float(np.abs(e).min()),
                        min_compact_absolute_threshold_margin=float(np.abs(p).min()),
                        max_error_vs_same_device_dense=float(np.abs(p-actual['DENSE_'+name]).max()),
                        dense_max_error_vs_saved=float(np.abs(actual['DENSE_'+name]-e).max()))
                    device_audit[name] = record
                    np.testing.assert_allclose(p, e, atol=1e-4, rtol=1e-5)
                    if record['sign_mismatch_bits']:
                        raise ValueError(f'Unexpected {device} {name} threshold mismatch: {record}')
                    arrays[device+'_'+name+'_logits'] = p
                audit[device] = device_audit
            np.testing.assert_array_equal(arrays['cpu_ENSEMBLE_logits'][ev] >= 0, historical['ENSEMBLE_flags'])
            np.testing.assert_array_equal(arrays['cuda_ENSEMBLE_logits'][ev] >= 0, historical['ENSEMBLE_flags'])
            np.savez_compressed(output/'parity.npz', **arrays)
            write(output/'parity-receipt.json', dict(status='PASS', rows=5000, by_device=audit,
                original_alert_parity=5000, mz5_eval_sign_parity=1500,
                exported_weights_sha256=sha(output/'compact.pt'), training_steps=0))
            benchmarks = {}
            for name in models:
                for batch in (1, 128):
                    candidates = {}
                    for device in ('cpu', 'cuda'):
                        model = placed[device][name]; v, t = resident[device]
                        candidates[device] = BackendCandidate(name=f'{name}_{device}_b{batch}', expected_device_type=device,
                            run_probe=lambda m=model, x=v[:batch], y=t[:batch]: m(x, y),
                            observe=lambda result, m=model: torch_observation(model=m, output=result),
                            synchronize=torch.cuda.synchronize if device == 'cuda' else lambda: None)
                    key = f'{name}_b{batch}'
                    benchmarks[key] = select_backend(Workload.MODEL_INFERENCE, cpu=candidates['cpu'], gpu=candidates['cuda'],
                        warmups=10, repeats=50, record_path=output/(key+'-backend.json'),
                        capabilities={'torch':torch.__version__, 'cpu_threads':1, 'device':torch.cuda.get_device_name(0),
                            'scope':'warm resident readout forward + device synchronization; no feature extraction or input transfer'})
            write(output/'benchmark-summary.json', {key: dict(selection=value['selection_reason'],
                median_ms={entry['actual_device_type']:entry['median_seconds']*1000 for entry in value['benchmarks']},
                p95_ms={entry['actual_device_type']:float(np.percentile(entry['samples_seconds'],95))*1000 for entry in value['benchmarks']})
                for key, value in benchmarks.items()})
        assert {key: sha(path) for key, path in paths.items()} == protocol['input_sha256']
        write(output/'engineering-receipt.json', dict(status='PASS', scope='Algebra-preserving engineering export; completed scientific result unchanged',
            rows=5000, training_steps=0, inputs_unchanged=True, protocol_sha256=sha(output/'protocol.json'),
            outputs_sha256={path.name:sha(path) for path in output.iterdir()}, seconds=time.perf_counter()-started,
            readout_only_latency=True, input_transfer_timed=False, batch_sizes=[1,128], CPU_threads=1,
            parameters=132872, MACs_per_row=132608, original_alert_parity=5000))
        print('ENGINEERING PASS', json.dumps(audit), read(output/'benchmark-summary.json'))
    except Exception as exc:
        write(output/'failure.json', dict(status='FAILED', error=repr(exc), seconds=time.perf_counter()-started))
        raise
    finally:
        # Process-local tensors die on return/exit; no other worker is touched.
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'validate-benchmark'), required=True)
    args = parser.parse_args(); root, output = args.root.resolve(), args.output.resolve()
    allowed = (root/'artifacts.local/work/mz5-fixed-ensemble-20260910').resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Output must stay inside canonical MZ5 artifact tree')
    (prepare if args.phase == 'prepare' else validate_and_benchmark)(root, output)
