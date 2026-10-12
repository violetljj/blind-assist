"""Continuation label-blind M3/augmented/historical-V shared-feature inference.

No geometry or labels are loaded. Every model sees one shared prepared batch;
Five original M3 seeds, three augmented and three retained V checkpoints see
identical prepared tensors. Each family's fixed raw mean is then smoothed by
the evaluator. V is a historical-data control, not a matched yaw-only ablation.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import cnh_temporal_readout_model as M
import cnh_temporal_readout_train as V
import cnh_extrinsic_aug_train as T

ANGULAR_AXES = (-15, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
OUT = T.OUT/'continuation-r1'


def infer(root=OUT, split='calibration', seeds=(0, 1, 2), device='cuda'):
    root = Path(root); seeds = tuple(int(s) for s in seeds)
    if seeds != (0, 1, 2) or root.resolve() == T.OUT.resolve():
        raise ValueError('Continuation uses fixed seeds0/1/2 in its distinct output tree')
    if split not in ('calibration', 'evaluation', 'angular'):
        raise ValueError('Unknown predeclared feature split')
    _, deadline = T.contract(root, later=split != 'calibration' or seeds != (0,))
    if split != 'calibration':
        final = json.loads((root/'evaluator/final_calibration.json').read_text(encoding='utf8'))
        gate_path = root/'evaluator/pilot_gate.json'
        if final.get('status') != 'COMPLETE' or final.get('seeds') != [0, 1, 2]:
            raise PermissionError('Final ensemble3 calibration must seal before evaluation feature access')
        if final['plan_sha256'] != T.sha(root/'PLAN.json') or final['revision_gate_sha256'] != T.sha(gate_path):
            raise ValueError('Final calibration revision identity mismatch')
        if final['V_calibration_sha256'] != T.sha(root/'evaluator/V_calibration.json'):
            raise ValueError('Independent V calibration identity mismatch')
    folder = root/'features'/split
    feature_receipt = json.loads((folder/'receipt.json').read_text(encoding='utf8'))
    if feature_receipt.get('status') != 'COMPLETE':
        raise ValueError('Feature split must be sealed complete before inference')
    if split == 'angular' and tuple(feature_receipt['axes_degrees']) != ANGULAR_AXES:
        raise ValueError('Angular feature optical-axis order differs from frozen unique axes')
    paths = sorted(folder.glob('unit*.npy'))
    if not paths:
        raise ValueError('No unit feature matrices')
    if 'units' in feature_receipt and [int(p.stem[4:]) for p in paths] != list(feature_receipt['units']):
        raise ValueError('Feature receipt units differ from available matrices')
    stage = 'ensemble3'
    dest = root/'scores'/split/stage
    vdest = root/'scores'/split/'V3'
    dest.mkdir(parents=True, exist_ok=True)
    vdest.mkdir(parents=True, exist_ok=True)
    if any((d/'receipt.json').exists() or any(d.glob('unit*.npz')) for d in (dest, vdest)):
        raise FileExistsError('Inference payloads are immutable; no partial result overwrite')
    V.configure(device); began = time.monotonic(); plan_hash = T.sha(root/'PLAN.json')
    source_hash = T.sources(); source_hash[str(Path(__file__).resolve())] = T.sha(__file__)
    model_paths = [T.ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{s}.pt' for s in range(5)]
    model_paths += [root/'models/M3_aug'/f'seed{s}/model.pt' for s in seeds]
    for seed, p in zip(seeds, model_paths[5:8]):
        receipt = json.loads((p.parent/'training_receipt.json').read_text(encoding='utf8'))
        if receipt.get('status') != 'COMPLETE' or receipt['model_sha256'] != T.sha(p):
            raise ValueError(f'Unsealed augmented seed {seed}')
    v_paths = [T.ROOT/'artifacts.local/work/cnh-temporal-readout-20261004/models/V'/f'seed{s}/model.pt' for s in (0, 1, 2)]
    for p in v_paths:
        receipt = json.loads((p.parent/'training_receipt.json').read_text(encoding='utf8'))
        if receipt.get('status') != 'COMPLETE' or receipt['model_sha256'] != T.sha(p) or receipt['parameters'] != 46097:
            raise ValueError('Historical V checkpoint changed or wrong architecture')
    model_paths += v_paths
    model_hashes = {str(p.resolve()): T.sha(p) for p in model_paths}
    nets = []; results = []; v_results = []; cached = None; inputs = None
    try:
        for p in model_paths:
            net = M.CVR()
            net.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
            nets.append(net.to(device).eval())
        for p in paths:
            T.check_deadline(deadline); started = time.monotonic()
            file_hash = T.sha(p); stat = p.stat()
            inputs = np.load(p, mmap_mode='r', allow_pickle=False)
            axis_count = 12 if split == 'angular' else 3
            if inputs.dtype != np.float16 or inputs.ndim != 7 or inputs.shape[0] != axis_count or inputs.shape[2:] != (13, 3, 24, 17, 33):
                raise ValueError('Expected fp16 [optical_axes,C,13,3,24,17,33]')
            if split == 'angular' and inputs.shape[1] != 28:
                raise ValueError('Angular configs must retain all seven deltas times four photon replicas')
            shape = inputs.shape; flat = inputs.reshape(-1, 3, 24, 17, 33)
            if split == 'angular':
                config_ids = np.arange(28, dtype=np.int32)
            else:
                unit_receipt = json.loads(p.with_suffix('.json').read_text(encoding='utf8'))
                declared = unit_receipt.get('config_ids', unit_receipt.get('configs'))
                if declared is None:
                    declared = feature_receipt.get('config_ids', feature_receipt.get('configs'))
                if declared is None:
                    raise ValueError('Natural feature receipt must declare original config IDs')
                config_ids = np.asarray(declared, dtype=np.int32)
                if config_ids.shape != (shape[1],) or len(set(config_ids.tolist())) != shape[1]:
                    raise ValueError('Natural original config IDs must match feature rows')
            cache_info = dict(used=False, bytes=int(flat.nbytes), reason='cpu_or_memory_budget')
            free, total = torch.cuda.mem_get_info(device) if device.startswith('cuda') else (0, 0)
            if device.startswith('cuda') and flat.nbytes <= free-2*1024**3 and flat.nbytes <= .7*total:
                cached = torch.from_numpy(flat).to(device, copy=True)
                # Exact dtype/values parity is checked on the first and last rows.
                parity_ids = np.array([0, len(flat)-1])
                cpu_values = torch.from_numpy(np.array(flat[parity_ids], copy=True)).to(device)
                if not torch.equal(cached[torch.as_tensor(parity_ids, device=device)], cpu_values):
                    raise ValueError('Inference feature cache parity failed')
                del cpu_values
                cache_info.update(used=True, reason='cached_exact_fp16_bytes', bitwise_sample_parity=True,
                                  free_before=int(free), total=int(total), reserve_bytes=2*1024**3)
            raw = np.empty((len(nets), len(flat), 2), np.float32)
            reference_raw = np.empty((len(flat), 2), np.float32)
            augmented_raw = np.empty((len(flat), 2), np.float32)
            v_raw = np.empty((len(flat), 2), np.float32)
            with torch.inference_mode():
                for offset in range(0, len(flat), T.BATCH):
                    T.check_deadline(deadline); end = min(offset+T.BATCH, len(flat))
                    voxels = cached[offset:end] if cached is not None else torch.from_numpy(np.array(flat[offset:end], copy=True)).to(device)
                    prepared = M.prepare_voxels(voxels)
                    batch_outputs = []
                    for index, net in enumerate(nets):
                        logits = net(prepared).float(); batch_outputs.append(logits)
                        values = logits.cpu().numpy()
                        if not np.isfinite(values).all():
                            raise FloatingPointError('Nonfinite raw inference logits')
                        raw[index, offset:end] = values
                    # Match retained M3 GPU-stack mean before CPU transfer.
                    reference_raw[offset:end] = torch.stack(batch_outputs[:5]).mean(0).cpu().numpy()
                    augmented_raw[offset:end] = torch.stack(batch_outputs[5:8]).mean(0).cpu().numpy()
                    v_raw[offset:end] = torch.stack(batch_outputs[8:11]).mean(0).cpu().numpy()
            reference = reference_raw.reshape(*shape[:3], 2)
            augmented = raw[5:8].reshape(len(seeds), *shape[:3], 2)
            mean = augmented_raw.reshape(*shape[:3], 2)
            v_by_seed = raw[8:11].reshape(3, *shape[:3], 2)
            v_mean = v_raw.reshape(*shape[:3], 2)
            if p.stat().st_size != stat.st_size or p.stat().st_mtime_ns != stat.st_mtime_ns:
                raise ValueError('Sealed feature changed during inference')
            output = dest/(p.stem+'.npz')
            metadata = dict(sensors=np.array(['S', 'L', 'R'])) if split != 'angular' else dict(
                axes_degrees=np.array(ANGULAR_AXES, np.int32),
                variants=np.arange(28, dtype=np.int32)//4, replicas=np.arange(28, dtype=np.int32)%4)
            np.savez_compressed(output, reference=reference, augmented_by_seed=augmented,
                                augmented_mean=mean, seeds=np.array(seeds, np.int32),
                                configs=config_ids, **metadata,
                                frames=np.arange(3, 16, dtype=np.int32), unit=np.array(int(p.stem[4:]), np.int32))
            result = dict(unit=int(p.stem[4:]), rows=len(flat), configs=shape[1], config_ids=config_ids.tolist(), seconds=time.monotonic()-started,
                          feature_path=str(p.resolve()), feature_sha256=file_hash, output_sha256=T.sha(output),
                          gpu_input_cache=cache_info, label_access=False)
            T.write(output.with_suffix('.json'), result); results.append(result)
            voutput = vdest/(p.stem+'.npz')
            np.savez_compressed(voutput, reference=reference, augmented_by_seed=v_by_seed,
                                augmented_mean=v_mean, seeds=np.array([0, 1, 2], np.int32),
                                configs=config_ids, **metadata, frames=np.arange(3, 16, dtype=np.int32),
                                unit=np.array(int(p.stem[4:]), np.int32))
            v_result = dict(result, output_sha256=T.sha(voutput), model_family='Historical V',
                            caveat='Different training units, scene/noise seeds, yaw and mirror; not yaw-only ablation')
            T.write(voutput.with_suffix('.json'), v_result); v_results.append(v_result)
            cached = None; flat = inputs = None
            del raw, augmented, reference, mean, prepared, voxels, reference_raw, augmented_raw, batch_outputs, logits, v_raw, v_by_seed, v_mean
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            print(f'{split}/{stage} {p.stem}: {result["seconds"]:.2f}s', flush=True)
        if T.sha(root/'PLAN.json') != plan_hash or any(T.sha(p) != h for p, h in model_hashes.items()):
            raise ValueError('Plan or model changed during inference')
        if any(T.sha(p) != h for p, h in source_hash.items()):
            raise ValueError('Inference implementation changed while running')
        result = dict(status='COMPLETE', split=split, stage=stage, augmented_seeds=list(seeds),
                      units=[r['unit'] for r in results], rows=sum(r['rows'] for r in results),
                      seconds=time.monotonic()-began, results=results, label_access=False,
                      schema='reference[A,C,13,Q]; augmented_by_seed[K,A,C,13,Q]; augmented_mean[A,C,13,Q]; A=3 S/L/R natural or 12 named optical axes angular',
                      ensemble='Fixed seed set; mean raw logits, then evaluator original causal five-score smooth',
                      model_sha256=model_hashes, source_sha256=source_hash, plan_sha256=plan_hash,
                      feature_receipt_sha256=T.sha(folder/'receipt.json'))
        T.write(dest/'receipt.json', result)
        v_receipt = dict(result, stage='V3', results=v_results, model_family='Historical V',
                         threshold='Independent calibration threshold using the same physical-clear matching rule',
                         role='Descriptive control only; excluded from primary augmented-model judgment',
                         caveat='Different training units, scene/noise seeds, yaw and mirror; not yaw-only ablation')
        T.write(vdest/'receipt.json', v_receipt)
        return result
    finally:
        cached = inputs = None; nets.clear()
        if 'net' in locals():
            del net
        if 'voxels' in locals():
            del voxels
        if 'prepared' in locals():
            del prepared
        if 'batch_outputs' in locals():
            del batch_outputs
        if 'logits' in locals():
            del logits
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--split', choices=('calibration', 'evaluation', 'angular'), default='calibration')
    parser.add_argument('--seeds', default='0,1,2'); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(json.dumps(infer(args.root, args.split, tuple(int(s) for s in args.seeds.split(',')), args.device), indent=2), flush=True)
