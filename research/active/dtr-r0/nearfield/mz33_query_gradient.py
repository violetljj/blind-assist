"""Zero-update CUDA gradients of four responsibility queries at frozen MZ32 endpoints."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from mz30_select import BranchSelector

QUERIES = ['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fingerprint(state):
    h = hashlib.sha256()
    for key, value in state.items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def partitions(model):
    layout = []; offset = 0; first = []; onehot = []; readout = []
    for key, value in model.named_parameters():
        indices = np.arange(offset, offset + value.numel()).reshape(tuple(value.shape))
        layout.append(dict(name=key, shape=list(value.shape), offset=offset, size=value.numel()))
        if key == 'net.0.weight':
            first.extend(indices[:, :264].ravel()); onehot.extend(indices[:, 264:].ravel())
        elif key == 'net.0.bias':
            first.extend(indices.ravel())
        else:
            readout.extend(indices.ravel())
        offset += value.numel()
    groups = dict(full=np.arange(offset), shared=np.array(first + readout),
                  shared_first=np.array(first), shared_readout=np.array(readout),
                  onehot_columns=np.array(onehot))
    assert offset == 8641 and len(groups['shared']) == 8513 and len(onehot) == 128
    return layout, groups


def pairwise(gradients, groups):
    result = {}
    for name, indices in groups.items():
        g = gradients[:, indices].astype(np.float64)
        norms = np.linalg.norm(g, axis=1)
        pairs = []
        for a, b in itertools.combinations(range(4), 2):
            dot = float(g[a] @ g[b]); den = float(norms[a] * norms[b])
            pairs.append(dict(queries=[QUERIES[a], QUERIES[b]], dot=dot,
                              cosine=dot / den if den > 0 else None,
                              nonzero_norms=den > 0))
        result[name] = dict(parameters=len(indices), norms=norms.tolist(), pairs=pairs)
    return result


def main(root, run):
    run.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter(); model = None; inputs = {}; outputs = {}
    code = {n: sha(Path(__file__).with_name(n)) for n in ['mz33_query_gradient.py', 'mz30_select.py']}
    try:
        work = root / 'artifacts.local/work'
        source = work / 'mz32-expanded-responsibility-20260910/run-v1'
        previous = work / 'mz30-branch-responsibility-20260910/run-v1'
        receipt = read(source / 'receipt.json'); assert receipt['status'] == 'PASS'

        def bind(path, expected=None):
            h = sha(path)
            assert expected is None or h == expected, str(path)
            inputs[str(path)] = h

        bind(source / 'receipt.json')
        for name in ['start.json', 'initial.pt', 'selector.pt', 'supervision.npz']:
            bind(source / name, receipt['outputs'][name])
        start32 = read(source / 'start.json')
        for name in ['branches.npz', 'normalization.npz', 'initial.pt']:
            path = previous / name; bind(path, start32['inputs'][str(path)])
        bind(Path(__file__).with_name('MZ33_QUERY_GRADIENT_PROTOCOL_20260910.md'))
        assert code['mz30_select.py'] == start32['code_sha256']['mz30_select.py']
        with np.load(previous / 'branches.npz') as b, np.load(previous / 'normalization.npz') as n, np.load(source / 'supervision.npz') as s:
            ids = b['global_ids']; train = b['train_ids']; batches = b['batches']
            assert len(train) == 7562 and np.array_equal(train, np.unique(batches))
            lookup = {int(v): i for i, v in enumerate(ids)}
            ti = np.array([lookup[int(v)] for v in train])
            # Only exact TRAIN rows enter any tensor or label computation.
            features = b['normal/features'][ti]
            rgb = b['normal/rgb'][ti]; tof = b['normal/tof'][ti]
            target = b['normal/target'][ti]; truth = b['normal/truth'][ti]
            mask = (rgb >= 0) != (tof >= 0)
            np.testing.assert_array_equal(target, (rgb >= 0) == truth)
            np.testing.assert_array_equal(s['global_ids'], train)
            np.testing.assert_array_equal(s['mask'], mask)
            np.testing.assert_array_equal(s['target'], target)
            np.testing.assert_array_equal(s['batches'], batches)
            mean = n['mean']; std = n['std']
            np.testing.assert_array_equal(mean, features.mean(0, keepdims=True))
            np.testing.assert_array_equal(std, features.std(0, keepdims=True).clip(.1))
            features = (features - mean) / std
        counts = mask.sum(0); positives = (mask & target).sum(0)
        assert counts.tolist() == [198, 150, 304, 513] and positives.tolist() == [152, 69, 65, 192]
        class_counts = [int(positives.sum()), int(mask.sum() - positives.sum())]
        assert class_counts == [478, 687]
        weights = [int(mask.sum()) / (2 * n) for n in class_counts]
        assert weights == start32['class_weights_rgb_tof']
        torch.set_num_threads(1); assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model = BranchSelector().cuda().eval()
        layout, groups = partitions(model)
        xx = torch.from_numpy(features).cuda(); yy = torch.from_numpy(target).cuda()
        mm = torch.from_numpy(mask).cuda()
        ww = torch.where(yy, weights[0], weights[1])
        write(run / 'start.json', dict(status='STARTED', training_steps=0, endpoints=['initial', 'final'],
            train_frames=7562, query_counts=counts.tolist(), class_counts_rgb_tof=class_counts,
            class_weights_rgb_tof=weights, inputs=inputs, code_sha256=code,
            tolerance=dict(atol=1e-5, rtol=1e-4),
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                         deterministic=torch.are_deterministic_algorithms_enabled(), matmul_tf32=False,
                         cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])))
        results = {}; arrays = dict(train_ids=train, supervision=mask, target=target)
        for endpoint, filename in [('initial', 'initial.pt'), ('final', 'selector.pt')]:
            state = torch.load(source / filename, map_location='cpu', weights_only=True)
            if endpoint == 'initial':
                original = torch.load(previous / 'initial.pt', map_location='cpu', weights_only=True)
                assert all(torch.equal(state[k], original[k]) for k in state)
            model.load_state_dict(state)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            before_hash = fingerprint(before)
            # One forward graph; four complete query means, then an independently
            # reduced global objective check. No optimizer is instantiated.
            z = model(xx)
            weighted_bce = F.binary_cross_entropy_with_logits(z, yy.float(), reduction='none') * ww
            query_losses = [(weighted_bce[:, q] * mm[:, q]).sum() / int(counts[q]) for q in range(4)]
            gradients = []
            parameters = tuple(model.parameters())
            for loss in query_losses:
                gradient = torch.autograd.grad(loss, parameters, retain_graph=True)
                gradients.append(torch.cat([g.reshape(-1) for g in gradient]).detach().cpu().numpy())
            global_loss = (weighted_bce * mm).sum() / int(counts.sum())
            global_gradient = torch.autograd.grad(global_loss, parameters)
            gg = torch.cat([g.reshape(-1) for g in global_gradient]).detach().cpu().numpy()
            gradients = np.stack(gradients)
            combined = (gradients.astype(np.float64) * (counts / counts.sum())[:, None]).sum(0)
            np.testing.assert_allclose(combined, gg, atol=1e-5, rtol=1e-4)
            logits = z.detach().cpu().numpy(); errors = ((logits >= 0) != target) & mask
            torch.cuda.synchronize()
            after = model.state_dict(); unchanged = all(torch.equal(v, after[k].detach().cpu()) for k, v in before.items())
            assert unchanged and fingerprint(after) == before_hash
            assert all(p.grad is None for p in model.parameters())
            details = pairwise(gradients, groups)
            # Each query's one-hot gradient has support in its own column only.
            assert all(pair['dot'] == 0 for pair in details['onehot_columns']['pairs'])
            rows = [dict(query=q, examples=int(counts[i]), rgb_correct=int(positives[i]),
                         tof_correct=int(counts[i] - positives[i]), weighted_bce=float(query_losses[i].detach()),
                         responsibility_errors=int(errors[:, i].sum()),
                         rgb_correct_misclassified=int((errors[:, i] & target[:, i]).sum()),
                         tof_correct_misclassified=int((errors[:, i] & ~target[:, i]).sum())) for i, q in enumerate(QUERIES)]
            results[endpoint] = dict(queries=rows, global_weighted_bce=float(global_loss.detach()),
                parameter_groups=details, state_before_sha256=before_hash, state_after_sha256=fingerprint(after),
                state_tensors_bit_identical=unchanged, gradient_identity=dict(max_abs_error=float(np.max(np.abs(combined - gg))),
                l2_error=float(np.linalg.norm(combined - gg)), atol=1e-5, rtol=1e-4, passed=True))
            arrays[endpoint + '/query_gradients'] = gradients
            arrays[endpoint + '/global_gradient'] = gg
            arrays[endpoint + '/logits'] = logits
            print(endpoint, json.dumps(dict(errors=[r['responsibility_errors'] for r in rows],
                shared_cosines=[p['cosine'] for p in details['shared']['pairs']])), flush=True)
        final_pairs = results['final']['parameter_groups']['shared']['pairs']
        minimum = min(p['cosine'] for p in final_pairs if p['nonzero_norms'])
        summary = dict(final_min_pair_cosine=minimum,
                       final_responsibility_errors_total=sum(q['responsibility_errors'] for q in results['final']['queries']),
                       local_conflict_signal=minimum <= -.1,
                       cosine_scope='Combined shared parameters excluding four one-hot columns')
        for path, expected in inputs.items():
            assert sha(path) == expected, 'Input changed: ' + path
        np.savez_compressed(run / 'gradients.npz', **arrays)
        write(run / 'result.json', dict(summary=summary, endpoints=results, parameter_layout=layout,
            train_frames=7562, disagreement_examples=1165, training_steps=0,
            inputs_unchanged=True, limits=['Local gradients do not prove negative transfer or harmless sharing.',
            'Raw responsibility errors are in-sample; no task cutoff or DEV result is evaluated.',
            'Full TRAIN global mean is a diagnostic aggregate, not a replay of mini-batch optimizer dynamics.']))
        outputs = {p.name: sha(p) for p in run.iterdir() if p.is_file()}
        write(run / 'receipt.json', dict(status='PASS', training_steps=0, seconds=time.perf_counter() - tick,
              inputs=inputs, code_sha256=code, outputs=outputs, backend='CUDA forward/backward, CPU scalar summaries'))
        print('PASS', json.dumps(summary), flush=True)
    except Exception:
        write(run / 'receipt.json', dict(status='FAIL', training_steps=0, inputs=inputs, code_sha256=code,
              error=traceback.format_exc(), outputs={p.name: sha(p) for p in run.iterdir() if p.is_file()}))
        raise
    finally:
        if model is not None:
            model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); main(a.root, a.run)
