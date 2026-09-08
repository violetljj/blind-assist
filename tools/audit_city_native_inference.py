"""Engineering replay of frozen G10/G13 inference, with no fitting or selection."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import evaluate_city_native_route as route
import numpy as np
from PIL import Image
import torch
from research_backend import torch_observation


def images(paths, historical_transfer=False):
    arrays = []
    for path in paths:
        with Image.open(path) as im:
            assert im.size == (640, 360)
            arrays.append(np.array(im.convert('RGB').resize((256, 144), Image.Resampling.BOX)))
    values = torch.from_numpy(np.stack(arrays))
    if historical_transfer:
        return values.cuda().permute(0, 3, 1, 2).float().div_(255.)
    return values.permute(0, 3, 1, 2).float().div_(255.).cuda()


@torch.inference_mode()
def run(output, match_batches=False, historical_transfer=False):
    output = output.resolve()
    assert output.is_relative_to(route.ARTIFACTS.resolve()) and not output.exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    config = route.configuration()  # All six original checkpoint hashes checked.
    capture = route.NF / 'diversity-20260907/main/capture'
    data = route.read(capture / 'model/dataset.json')
    validation = [s for s in data['samples'] if s['split'] == 'val']
    selected = validation[:16]
    frame_paths = {f['sample_index']: capture / 'model' / f['rgb_path'] for f in data['frames']}
    historical_paths = [frame_paths[s['frame_indices'][0]] for s in selected]
    city_root = route.NF / 'city-native-validation-20260908'
    city_ids = [6, 7, 14, 15, 28, 30]
    city_paths = [city_root / f'final-route-v1/model/sample/{i:04d}.png' for i in city_ids]
    cached = np.load(city_root / 'fixed-models-v1/predictions.npz', allow_pickle=False)
    sources = {
        'g10': route.NF / 'diversity-20260907/main/learned',
        'g13': route.NF / 'decoupled-20260908/main/learned',
    }
    protocol = dict(scope='ENGINEERING_REPLAY_CONSUMED_VALIDATION_AND_CITY_CACHE',
        question='Does the city adapter reproduce historical scores and show native-city gate collapse?',
        selection='First 16 original VAL samples in saved order; six previously reported reliable city target views',
        score_absolute_tolerance=1e-4, support_absolute_tolerance=1e-4,
        optimizer_steps=0, checkpoint_configuration=config,
        historical_ids=[s['sample_id'] for s in selected], city_ids=city_ids,
        rgb_sha256={str(p): route.sha(p) for p in historical_paths + city_paths},
        audit_source_sha256=route.sha(Path(__file__)),
        no_intervention='No threshold or gate replacement; hooks observe existing activations only',
        match_original_batch_context=match_batches,
        historical_cuda_conversion=historical_transfer)
    # Check RGB identities against original training receipts before replay.
    receipts = {k: route.read(v / 'receipt.json') for k, v in sources.items()}
    for name, receipt in receipts.items():
        cache = receipt['cache_receipts']['new'] if name == 'g10' else receipt['caches']['training']
        assert route.sha(capture / 'model/dataset.json') == cache['input_sha256']['model/dataset.json']
        for sample, path in zip(selected, historical_paths):
            assert route.sha(path) == cache['rgb_sha256'][str(sample['frame_indices'][0])]
    protocol['model_source_checks'] = {}
    for name, files in {'g10': ['diversity_model.py', 'whisker_model.py'],
                        'g13': ['decoupled_model.py', 'representation_model.py']}.items():
        checks = {f: route.sha(route.SOURCE / f) == receipts[name]['source_sha256'][f] for f in files}
        assert all(checks.values()), checks
        protocol['model_source_checks'][name] = checks
    output.mkdir(parents=True)
    route.write(output / 'protocol.json', protocol)
    started = time.perf_counter()
    tensors = {'historical_val': images(historical_paths), 'city': images(city_paths)}
    records, outputs = [], {}
    try:
        for name in route.METHODS:
            parent = sources[name] / ('reference' if name == 'g13' else '')
            original = route.read(parent / 'predictions.json')
            indices = [original['sample_ids'].index(s['sample_id']) for s in selected]
            arm = 'expanded_region' if name == 'g10' else 'decoupled'
            for record in config[name]['weights']:
                seed = record['seed']
                model = (route.DiversityModel(region=True) if name == 'g10' else
                         route.DecoupledModel(route.NF / 'representation-20260908/pretrained'))
                model.load_state_dict(torch.load(record['path'], map_location='cpu', weights_only=True), strict=True)
                model = model.cuda().eval()
                assert not any(m.training for m in model.modules())
                for domain, rgb in tensors.items():
                    selected_positions = list(range(len(rgb)))
                    batch_size = len(rgb)
                    if match_batches:
                        if domain == 'historical_val':
                            batch_size = 64 if name == 'g10' else 32
                            context = validation[:batch_size]
                            context_paths = [frame_paths[s['frame_indices'][0]] for s in context]
                            rgb = images(context_paths, historical_transfer)
                            selected_positions = list(range(16))
                        else:
                            batch_size = 16
                            rgb = images([city_root / f'final-route-v1/model/sample/{i:04d}.png' for i in range(32)])
                            selected_positions = city_ids
                    pooled = []
                    hook = model.near[0].register_forward_hook(lambda m, a, value: pooled.append(value.detach()))
                    try:
                        pairs = [model(batch) for batch in rgb.split(batch_size)]
                        near = torch.cat([p[0] for p in pairs])[selected_positions]
                        support = torch.cat([p[1] for p in pairs])[selected_positions]
                    finally:
                        hook.remove()
                    pooled_selected = torch.cat([p.reshape(-1, 2, *p.shape[1:]) for p in pooled])[selected_positions]
                    prob, spatial = near.sigmoid().cpu().numpy(), support.sigmoid().cpu().numpy()
                    outputs[f'{name}_{seed}_{domain}_near'] = prob
                    outputs[f'{name}_{seed}_{domain}_support'] = spatial
                    ref = (np.asarray(original['arms'][arm]['seeds'][str(seed)]['normal'])[indices, :2]
                           if domain == 'historical_val' else cached[f'{name}_seed{seed}_near'][city_ids])
                    error = float(np.abs(prob - ref).max())
                    bias = model.near[2].bias.detach().sigmoid().cpu().numpy()
                    row = dict(method=name, seed=seed, domain=domain,
                        score_max_absolute_error=error, score_parity_pass=error <= 1e-4,
                        near_range=[prob.min(0).tolist(), prob.max(0).tolist()],
                        support_quantiles=np.quantile(spatial, [0, .5, .95, 1]).tolist(),
                        support_fraction_ge_half=float((spatial >= .5).mean()),
                        pooled_gated_abs_mean=float(pooled_selected.abs().mean()),
                        near_bias_sigmoid=bias.tolist(),
                        max_distance_to_bias_probability=float(np.abs(prob-bias).max()),
                        backend=asdict(torch_observation(model=model, output=(near, support))))
                    if domain == 'city':
                        spatial_error = float(np.abs(spatial-cached[f'{name}_seed{seed}_support'][city_ids]).max())
                        row.update(support_max_absolute_error=spatial_error, support_parity_pass=spatial_error <= 1e-4)
                    records.append(row)
                del model
        torch.cuda.synchronize()
        result = dict(status='PASS' if all(r['score_parity_pass'] and r.get('support_parity_pass', True) for r in records) else 'PARITY_FAILURE',
            records=records, elapsed_seconds=time.perf_counter()-started, optimizer_steps=0,
            note='Parity on selected original VAL samples verifies adapter execution, not entire historical benchmark metrics.')
        np.savez_compressed(output / 'replay.npz', **outputs)
        route.write(output / 'result.json', result)
        print(route.json.dumps(result))
    finally:
        cached.close()
        tensors.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-batches', action='store_true')
    parser.add_argument('--historical-transfer', action='store_true')
    args = parser.parse_args()
    run(args.output, args.match_batches, args.historical_transfer)
