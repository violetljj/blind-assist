"""TRAIN-only fixed-checkpoint gradient and leave-region-out feature diagnosis.

No optimizer, parameter update, threshold selection, or DEV/EVAL access.
All TRAIN count-logit gradients; 128 fixed balanced HEAD_ONLY examples for
parameter gradients; all exclusive-range HEAD_ONLY examples for frozen 1-NN.
Nearest neighbours exclude the entire query region. A shuffled-label control
and RGB-mean control limit interpretation; this is not a trained probe or proof
that unavailable information cannot be decoded by another method.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel, near_from_counts
from body_query_data import QueryRGB, truth, read, write, sha, fresh_output
from body_query_train import digest
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tools'))
from research_backend import torch_observation


def relation(a, b):
    a, b = a.detach().flatten().double(), b.detach().flatten().double()
    na, nb = a.norm(), b.norm()
    return dict(norm_alert=float(na), norm_count_weighted=float(nb),
                cosine=float(a.dot(b)/(na*nb)) if na*nb > 0 else None,
                count_to_alert=float(nb/na) if na > 0 else None)


def nn_probe(x, y, regions):
    z = F.normalize(x.float(), dim=1)
    similarity = z @ z.T
    different = torch.tensor(regions[:, None] != regions[None, :], device=x.device)
    assert different.any(1).all()
    similarity.masked_fill_(~different, -torch.inf)
    index = similarity.argmax(1).cpu().numpy()
    shuffled = np.random.default_rng(17).permutation(y)
    def score(labels):
        p = labels[index]
        return dict(correct=int((p == labels).sum()), total=len(labels),
                    balanced_accuracy=float(np.mean([(p[labels == c] == c).mean() for c in (0, 1)])))
    return dict(actual=score(y), shuffled=score(shuffled),
                per_region={str(r):dict(correct=int((y[index][regions == r] == y[regions == r]).sum()),
                                       total=int((regions == r).sum())) for r in np.unique(regions)},
                neighbor_index=index.tolist())


def run(a):
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    out = fresh_output(a.output)
    started = time.perf_counter()
    model = None
    handles = []
    try:
        manifest_sha = sha(a.cache/'manifest.json')
        checkpoint_sha = sha(a.run/'NEW-step2000.pt')
        assert checkpoint_sha == read(a.run/'fit-complete.json')['checkpoint_sha256']
        assert manifest_sha == read(a.run/'protocol.json')['cache_sha256']
        data = QueryRGB(a.cache, 'train')
        rec, target = truth(a.cache, 'train', training=True)
        assert data.ids == rec['sample_indices'] and len(data.ids) == 5000
        y = target['counts'].reshape(-1, 2, 2, 3).sum(-1) >= 3
        exclusive = y[:, 1, 0] ^ y[:, 1, 1]
        eligible = exclusive & (target['near'][:, 0] == 0) & (target['near'][:, 1] == 1)
        ids = np.flatnonzero(eligible)
        labels = y[ids, 1, 1].astype(int)
        regions = np.array([rec['records'][i]['region_id'] for i in ids])
        rng = np.random.default_rng(17)
        chosen = np.concatenate([rng.choice(ids[labels == v], 64, replace=False) for v in (0, 1)])
        write(out/'protocol.json', dict(scope='Consumed TRAIN-only Development diagnostic', fits=0,
              optimizer_steps=0, checkpoint_sha256=checkpoint_sha, cache_sha256=manifest_sha,
              source_sha256=sha(Path(__file__)), gradient_examples=chosen.tolist(), feature_examples=ids.tolist(),
              loss='Original mean alert BCE and .25 mean count CE; support has no path to count logits',
              feature_rule='Cosine 1-NN excluding same region; raw mean and query embedding; no fitted decoder',
              stop='One checkpoint; no training, threshold tuning, successor or DEV/EVAL access'))

        # Output-space derivatives use the saved exact count probabilities.
        saved = np.load(a.run/'NEW-train.npz', allow_pickle=False)
        assert np.all(saved['counts'] > 0)
        logits = torch.tensor(np.log(saved['counts']), device='cuda', requires_grad=True)
        gt = torch.tensor(target['counts'], device='cuda')
        alert_gt = torch.tensor(target['near'], device='cuda', dtype=torch.float32)
        n = near_from_counts(logits)
        parity = float((n.sigmoid()-torch.tensor(saved['near'], device='cuda')).detach().abs().max())
        assert parity < 2e-6, parity
        la = F.binary_cross_entropy_with_logits(n, alert_gt)
        lc = -.25*logits.log_softmax(-1).gather(-1, gt[..., None]).mean()
        ga = torch.autograd.grad(la, logits, retain_graph=True)[0]
        gc = torch.autograd.grad(lc, logits)[0]
        output_grad = {}
        for name, mask in [('all', np.ones(5000, bool)), ('head_near_only', eligible & y[:, 1, 0]),
                           ('head_far_only', eligible & y[:, 1, 1])]:
            m = torch.tensor(mask, device='cuda')
            output_grad[name] = {}
            for d, segment in enumerate(('near', 'far')):
                sl = slice(6+d*3, 9+d*3)
                aa, cc = ga[m, sl], gc[m, sl]
                # descent on class0 versus a class3 logit: positive raises nonempty evidence
                push_a, push_c = aa[..., 0]-aa[..., 3], cc[..., 0]-cc[..., 3]
                nonzero = (push_a.abs() > 1e-12) & (push_c.abs() > 1e-12)
                output_grad[name][segment] = dict(**relation(aa, cc), frames=int(mask.sum()),
                    opposing_cells=int(((push_a*push_c < 0) & nonzero).sum()), nonzero_cells=int(nonzero.sum()),
                    alert_raise_cells=int((push_a > 1e-12).sum()), count_lower_cells=int((push_c < -1e-12).sum()),
                    combined_raise_cells=int((push_a+push_c > 1e-12).sum()))

        model = BodyQueryModel(a.pretrained, 'B').cuda()
        model.load_state_dict(torch.load(a.run/'NEW-step2000.pt', map_location='cuda', weights_only=True))
        model.eval()
        before = digest(model.state_dict())
        parameters = [(k, p) for k, p in model.named_parameters() if k.startswith(('query_', 'deep_projection.', 'detail.', 'backbone.'))]
        accum = [[torch.zeros_like(p) for _, p in parameters] for _ in range(2)]
        for batch in np.array_split(chosen, 4):
            n, _, c = model(data.tensor(batch, 'cuda'))
            losses = [F.binary_cross_entropy_with_logits(n, torch.tensor(target['near'][batch], device='cuda', dtype=torch.float32)),
                      -.25*c.log_softmax(-1).gather(-1, torch.tensor(target['counts'][batch, :, None], device='cuda')).mean()]
            for j, loss in enumerate(losses):
                gradients = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=(j == 0))
                for total, grad in zip(accum[j], gradients):
                    total.add_(grad.detach()/4)
        parameter_grad = {}
        for prefix in ('query_point.', 'query_readout.', 'deep_projection.', 'detail.', 'backbone.'):
            selected = [i for i, (k, _) in enumerate(parameters) if k.startswith(prefix)]
            parameter_grad[prefix] = relation(*[torch.cat([g[i].flatten() for i in selected]) for g in accum])
        del accum, gradients
        write(out/'gradients.json', dict(output_gradients=output_grad, parameter_gradients=parameter_grad))

        captured = {}
        handles.append(model.query_point.register_forward_pre_hook(lambda module, args: captured.update(raw=args[0].detach())))
        handles.append(model.query_readout.register_forward_pre_hook(lambda module, args: captured.update(embedding=args[0].detach())))
        features = {k:[] for k in ('raw_near', 'raw_far', 'embedding_near', 'embedding_far', 'rgb_mean')}
        max_parity = 0.
        with torch.inference_mode():
            # Match historical batch size AND composition to avoid batch-shape
            # dependent CUDA rounding; select diagnostic rows only afterwards.
            for begin in range(0, len(data.ids), 32):
                batch = np.arange(begin, min(begin+32, len(data.ids)))
                x = data.tensor(batch, 'cuda')
                n, _, c = model(x)
                max_parity = max(max_parity, float((c.softmax(-1)-torch.tensor(saved['counts'][batch], device='cuda')).abs().max()))
                keep = torch.tensor(eligible[batch], device='cuda')
                raw = captured['raw'][..., :64]
                mask = model.query_valid[None, :, :, None]
                raw = (raw*mask).sum(2)/mask.sum(2).clamp_min(1)
                for d, segment in enumerate(('near', 'far')):
                    sl = slice(6+d*3, 9+d*3)
                    features['raw_'+segment].append(raw[keep, sl].flatten(1))
                    features['embedding_'+segment].append(captured['embedding'][keep, sl].flatten(1))
                features['rgb_mean'].append(x[keep].mean((2, 3)))
        assert max_parity < 2e-5, max_parity
        features = {k:torch.cat(v) for k,v in features.items()}
        probes = {k:nn_probe(v, labels, regions) for k,v in features.items()}
        assert digest(model.state_dict()) == before
        assert sha(a.run/'NEW-step2000.pt') == checkpoint_sha and sha(a.cache/'manifest.json') == manifest_sha
        torch.cuda.synchronize()
        write(out/'result.json', dict(output_gradients=output_grad, parameter_gradients=parameter_grad,
              feature_probes=probes, feature_classes=np.bincount(labels).tolist(), regions=np.unique(regions).tolist(),
              cached_alert_parity=parity, frozen_inference_count_parity=max_parity,
              backend=asdict(torch_observation(model=model, output=(n,c))), seconds=time.perf_counter()-started,
              state_unchanged=True, fits=0, optimizer_steps=0))
        np.savez_compressed(out/'features.npz', **{k:v.cpu().numpy() for k,v in features.items()}, ids=ids, labels=labels, regions=regions)
        write(out/'receipt.json', dict(status='PASS', result_sha256=sha(out/'result.json'),
              protocol_sha256=sha(out/'protocol.json'), features_sha256=sha(out/'features.npz'),
              scope='TRAIN only; fixed checkpoint; no weight updates'))
        print('PASS', out, flush=True)
    except Exception as exc:
        write(out/'failure.json', dict(error=repr(exc)))
        raise
    finally:
        for handle in handles:
            handle.remove()
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('cache', 'run', 'pretrained', 'output'):
        parser.add_argument('--'+key, required=True, type=Path)
    run(parser.parse_args())
