"""One same-initialization expanded-source B fit, against unchanged original B."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel
from body_query_data import QueryRGB, truth, sha, write, read, fresh_output
from body_query_train import digest, metrics, predict, benchmark, INITIAL_SHA
from city_data import pixel_support_bce
from city_dev_selection import select_threshold
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'tools'))
from research_backend import torch_observation

OLD_SHA = 'dd6fab0bddf436477e9076bb47a0df8914b4ddd21dcf0ea9897143643460726b'


def strata(pred, target, record):
    p = 1-pred['counts'][:, :, 0]; y = target['counts'] > 0
    output = {}
    for h, head in enumerate(('BODY', 'HEAD')):
        for distance, name in enumerate(('near', 'far')):
            s = slice(h*6+distance*3, h*6+distance*3+3)
            pos, fired = y[:, s], p[:, s] >= .5
            tp, fn = int((pos & fired).sum()), int((pos & ~fired).sum())
            output[head+'_'+name] = dict(TP=tp, FN=fn, FP=int((~pos & fired).sum()),
                                        TN=int((~pos & ~fired).sum()), recall=tp/(tp+fn) if tp+fn else None)
    return output


def run(a):
    assert torch.cuda.is_available(), 'CUDA required'
    assert sha(a.initial) == INITIAL_SHA and sha(a.old) == OLD_SHA
    assert read(a.cache/'manifest.json')['source_index_sha256'] == '88dd9687fbefe2432ac05ac57d54510ec1f0d0e65faac646f324365ae380166e'
    historical = read(a.old.parent/'protocol.json')['source_sha256']
    for name in ('body_query_model.py','city_data.py','decoupled_model.py','representation_model.py'):
        assert sha(Path(__file__).with_name(name)) == historical[name], 'Historical model/loss source changed: '+name
    out = fresh_output(a.output); start = time.perf_counter()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    train = QueryRGB(a.cache, 'train'); record, target = truth(a.cache, 'train', training=True)
    assert len(train.ids) == 2500 and train.ids == record['sample_indices']
    schedule = np.random.default_rng(17).integers(0, 2500, (2000, 32))
    np.save(out/'schedule.npy', schedule, allow_pickle=False)
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    model = BodyQueryModel(a.pretrained, 'B').cuda()
    model.initialize_g13(torch.load(a.initial, map_location='cpu', weights_only=True))
    initial_digest = digest(model.state_dict())
    assert initial_digest == read(a.old.parent/'B-fit-complete.json')['initial_digest']
    protocol = dict(seed=17, steps=2000, batch=32, lr=1e-5, weight_decay=1e-4,
                    cache_sha256=sha(a.cache/'manifest.json'), old_sha256=sha(a.old),
                    initial_sha256=sha(a.initial), initial_digest=initial_digest,
                    source_sha256=sha(Path(__file__)), schedule_sha256=sha(out/'schedule.npy'),
                    brief_sha256=sha(Path(__file__).with_name('BODY_QUERY_EXPANDED_B_PROTOCOL_20260909.md')),
                    dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in
                        ('body_query_model.py','body_query_data.py','body_query_train.py','city_data.py','city_dev_selection.py','decoupled_model.py','representation_model.py')},
                    torch=torch.__version__, device=torch.cuda.get_device_name(),
                    trained_frames=2500, expected_presentations_per_frame=25.6)
    write(out/'protocol.json', protocol)
    before = digest(dict(model.named_buffers())); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    targets = {k:torch.from_numpy(v).cuda() for k,v in target.items()}
    # Keep the larger source in uint8; convert only each sampled batch.
    rgb = torch.from_numpy(train.rgb.copy()).cuda()
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm): layer.eval()
    history = []; fit_start = time.perf_counter(); completed_steps = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        for step, indices in enumerate(schedule, 1):
            ids = torch.from_numpy(indices).cuda()
            n,s,c = model(rgb[ids].permute(0,3,1,2).float()/255.)
            ln = F.binary_cross_entropy_with_logits(n, targets['near'][ids].float())
            ls = pixel_support_bce(s, targets['support'][ids])
            lc = -c.log_softmax(-1).gather(-1, targets['counts'][ids].long().unsqueeze(-1)).mean()
            loss = ln + .25*ls + .25*lc
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if step == 1:
                write(out/'actual-backend.json', asdict(torch_observation(model=model, output=(n,s,c))))
                grads = {name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
                assert not any(k.startswith('near.') for k in grads)
                write(out/'first-gradients.json', grads)
            optimizer.step(); completed_steps = step
            if step % 100 == 0:
                row = dict(step=step, near=float(ln.detach()), support=float(ls.detach()), query=float(lc.detach()), loss=float(loss.detach()))
                history.append(row); write(out/'progress.json', row); print(row, flush=True)
        torch.cuda.synchronize()
        assert digest(dict(model.named_buffers())) == before
        torch.save(model.state_dict(), out/'NEW-step2000.pt')
        write(out/'fit-complete.json', dict(steps=2000, seconds=time.perf_counter()-fit_start, history=history,
              initial_digest=initial_digest, checkpoint_sha256=sha(out/'NEW-step2000.pt'), peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del model, optimizer, rgb, targets, n,s,c,ln,ls,lc,loss
        torch.cuda.empty_cache()
        evaluate_saved(a, out)
        write(out/'receipt.json', dict(status='PASS', fits=1, steps=2000, seconds=time.perf_counter()-start,
              result_sha256=sha(out/'result.json'), selection_sha256=sha(out/'selection.json'), backend='CUDA', device=torch.cuda.get_device_name()))
    except Exception as exc:
        write(out/'failure.json', dict(error=repr(exc), completed_steps=completed_steps)); raise


def evaluate_saved(a, out):
    selection = {}; result = dict(arms={})
    checkpoints = {'OLD':a.old, 'NEW':out/'NEW-step2000.pt'}
    for arm, checkpoint in checkpoints.items():
        model = BodyQueryModel(a.pretrained, 'B').cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        data = QueryRGB(a.cache, 'dev'); rec, gt = truth(a.cache, 'dev')
        pred = predict(model, data); np.savez_compressed(out/f'{arm}-dev.npz', **pred)
        heads = [select_threshold(pred['near'][:,h], gt['near'][:,h], min_count=48) for h in range(2)]
        selection[arm] = dict(thresholds=[v['threshold'] for v in heads], heads=heads, checkpoint_sha256=sha(checkpoint))
        result['arms'][arm] = dict(dev=metrics(pred,gt,rec,selection[arm]['thresholds']))
        result['arms'][arm]['dev']['query_strata'] = strata(pred,gt,rec)
        del model
    write(out/'selection.json', selection); frozen = sha(out/'selection.json')
    for arm, checkpoint in checkpoints.items():
        model = BodyQueryModel(a.pretrained, 'B').cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        for role in ('train','eval'):
            data = QueryRGB(a.cache,role); rec,gt = truth(a.cache,role)
            pred = predict(model,data); np.savez_compressed(out/f'{arm}-{role}.npz',**pred)
            result['arms'][arm][role] = metrics(pred,gt,rec,selection[arm]['thresholds'])
            result['arms'][arm][role]['query_strata'] = strata(pred,gt,rec)
        result['arms'][arm]['runtime'] = benchmark(model,data)
        del model
    assert sha(out/'selection.json') == frozen
    assert sha(a.cache/'manifest.json') == read(out/'protocol.json')['cache_sha256']
    write(out/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache','pretrained','initial','old','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
