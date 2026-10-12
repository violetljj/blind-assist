"""Authorized continuation: fit only missing seeds1/2 under its new frozen PLAN.

Original stopped-run inputs, seed0 checkpoint and failure remain read-only.
Training arithmetic delegates to the inspected exact V recipe; receipts and
checkpoints are written exclusively into continuation-r1.
"""
import argparse
import json
from pathlib import Path

import cnh_extrinsic_aug_train as T

OUT = T.OUT/'continuation-r1'


def train(root=OUT, seed=1, device='cuda'):
    root = Path(root)
    if root.resolve() == T.OUT.resolve() or seed not in (1, 2):
        raise ValueError('Continuation may fit only seed1/2 in its distinct output tree')
    own_hash = T.sha(__file__)
    result = T.train(root, seed, device)
    if T.sha(__file__) != own_hash:
        raise ValueError('Continuation driver changed during training')
    T.write(root/'models/M3_aug'/f'seed{seed}'/'continuation_receipt.json', dict(
        status='COMPLETE', seed=seed, driver_sha256=own_hash,
        training_receipt_sha256=T.sha(root/'models/M3_aug'/f'seed{seed}'/'training_receipt.json'),
        old_failure_preserved=True, plan_sha256=result['plan_sha256'], model_sha256=result['model_sha256']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, choices=(1, 2), default=1); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); print(json.dumps(train(args.root, args.seed, args.device), indent=2), flush=True)
