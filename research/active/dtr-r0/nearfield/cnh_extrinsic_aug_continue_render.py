"""Isolated, explicitly authorized continuation of the stopped augmentation run.

Reuse frozen physical/photon/projection code without changing old outputs.
Fresh evaluation and retained angular photons require revised pilot PASS and
the three-seed final calibration seal in this continuation's own artifact tree.
"""
import argparse
from pathlib import Path
import time

import cnh_extrinsic_aug_data as D
import cnh_extrinsic_aug_eval_subset as E
import cnh_extrinsic_aug_angular_features as A

ROOT = D.ROOT
SOURCE_OUT = D.OUT
OUT = SOURCE_OUT/'continuation-r1'
ORIGINAL_PLAN = D.plan

# All write-bearing module roots must move together. Old renderer/feature
# functions remain byte-identical and access their patched module globals.
D.OUT = OUT
E.OUT = OUT
A.OUT = OUT


def continuation_plan():
    p = ORIGINAL_PLAN()
    assert p['deadline_unix'] == 1791213092, 'Continuation deadline must be 2026-10-05T15:11:32Z'
    return p


def require_evaluation():
    p = continuation_plan()
    gate_path = OUT/p['calibration']['pilot_gate_path']
    final_path = OUT/'evaluator/final_calibration.json'
    score_path = OUT/'scores/calibration/ensemble3/receipt.json'
    gate = D.read(gate_path)
    final = D.read(final_path)
    scores = D.read(score_path)
    assert gate.get('judgment') == 'PASS', 'Revised continuation pilot must PASS'
    assert final.get('status') == 'COMPLETE' and final.get('seeds') == [0, 1, 2], 'Seal final three-seed calibration before evaluation'
    assert final.get('plan_sha256') == D.sha(OUT/'PLAN.json'), 'Final calibration belongs to another protocol'
    assert final.get('revision_gate_sha256') == D.sha(gate_path), 'Final calibration must use this revised pilot gate'
    assert scores.get('status') == 'COMPLETE' and scores.get('augmented_seeds') == [0, 1, 2], 'Three-seed calibration inference incomplete'
    assert scores.get('plan_sha256') == D.sha(OUT/'PLAN.json'), 'Calibration scores belong to another protocol'
    return p


D.plan = continuation_plan
D.require_evaluation = require_evaluation


def sources():
    return {str(Path(module.__file__).resolve()): D.sha(module.__file__)
            for module in (D, E, A)} | {str(Path(__file__).resolve()): D.sha(__file__)}


def prepare():
    """Source/metadata checks only; never opens evaluation observations."""
    p = continuation_plan()
    assert D.OUT == E.OUT == A.OUT == OUT
    assert p['natural_evaluation_units'] == list(range(99000, 99096))
    for count in (4, 8, 12, 16, 20, 24, 32, 40):
        for unit in p['natural_evaluation_units']:
            ids = E.configs_for(unit, count)
            assert len(ids) == len(set(ids)) == count and ids == sorted(ids)
            assert all(0 <= config < 40 for config in ids)
    assert A.AXES == [-15., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45.]
    D.save(OUT/'engineering/render_wrapper_cpu.json', dict(status='PASS',
        module_output_roots=[str(module.OUT) for module in (D, E, A)],
        source_sha256=sources(), plan_sha256=D.sha(OUT/'PLAN.json'),
        original_plan_sha256=D.sha(SOURCE_OUT/'PLAN.json'),
        old_artifacts_overwritten=False, eval_payload_opened=False,
        sampling='Uniform without replacement using frozen seed2026100609+unit'))


def run(stage, count, reason):
    if stage == 'prepare':
        prepare()
        return
    require_evaluation()
    D.setup()
    source_hashes = sources()
    if stage in ('evaluation', 'evaluation-angular'):
        E.run(count, reason)
    if stage in ('angular', 'evaluation-angular'):
        # Source split manifests are public metadata; observation payloads are
        # opened inside A.run only after the strict revised continuation gate.
        pilot_plan = D.read(A.DUAL/'PLAN.json')
        envelope_plan = D.read(A.ANGULAR.parent/'PLAN.json')
        units = sorted(set(pilot_plan['evaluation_units']) & set(envelope_plan['A']['units']))
        A.run(units)
    assert sources() == source_hashes, 'Source changed during continuation rendering'
    D.save(OUT/'engineering'/f'render-{stage}-complete.json', dict(status='COMPLETE',
        stage=stage, source_sha256=source_hashes, closed_unix=time.time(),
        plan_sha256=D.sha(OUT/'PLAN.json'), config_count_per_unit=count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'evaluation', 'angular', 'evaluation-angular'])
    parser.add_argument('--count', type=int, choices=range(1, 41), default=40)
    parser.add_argument('--reason', default='Full40 fits runtime budget before any evaluation outcomes')
    args = parser.parse_args()
    try:
        run(args.stage, args.count, args.reason)
    except BaseException as error:
        D.save(OUT/'failures'/f'render-{args.stage}-{time.time_ns()}.json',
            dict(status='FAILED', error=repr(error), unix=time.time()))
        raise
