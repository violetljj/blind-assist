"""Freeze sensor-only UE inputs and rerun causal perception without launching UE.

Replay keeps the recorded poses and issued plans. It cannot evaluate the actual
trajectory/contact outcome of a different motion controller. Dataset identity is
content addressed and checked before every run; optional hardlinks are never
modified or chmod'ed, because that would modify the source inode too.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

REPO = Path(__file__).resolve().parents[4]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def exact(value, keys):
    if set(value) != set(keys.split()):
        raise ValueError(f'Unexpected sensor schema keys: {set(value) ^ set(keys.split())}')


def contained(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError('Input path escapes sensory root')
    return path


def validate_episode(item):
    exact(item, 'episode_id route_frame plan_path frames')
    if not re.fullmatch(r'episode_[0-9]{4}', item['episode_id']):
        raise ValueError('Episode identifiers must be opaque, not evaluator scenario names')
    exact(item['route_frame'], 'center_xy_m z_origin_m forward_xy right_xy')
    for i, row in enumerate(item['frames']):
        exact(row, 'sample_index time_s rgb_path depth_path camera_transform wearer_transform command_velocity plan_path')
        if row['sample_index'] != i:
            raise ValueError('A dataset must contain the full causal prefix from sample zero')
        for name in ('camera_transform', 'wearer_transform'):
            exact(row[name], 'x y z pitch yaw roll')
        exact(row['command_velocity'], 'x y z')


def export_dataset(source_run, output, *, hardlink=False):
    """Read descriptors from model/ only; no evaluator or worker outputs copied."""
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    if (source_run / 'owner.lock').exists():
        raise ValueError('Source capture is still owned by a running job')
    model_root = source_run / 'model'
    header = read(model_root / 'manifest.json')
    exact(header['calibration'], 'width height horizontal_fov_degrees depth_max_m')
    episodes = [read(p) for p in sorted(model_root.glob('episode_[0-9][0-9][0-9][0-9]/episode.json'))]
    if not episodes:
        raise ValueError('No completed sensory episode descriptors')
    references = set()
    for item in episodes:
        validate_episode(item)
        references.add(item['plan_path'])
        for row in item['frames']:
            references.update(row[k] for k in ('rgb_path', 'depth_path', 'plan_path'))
    # Validate plans before creating output; only declared navigation waypoints
    # are future-looking (issued intentions, never future observations/truth).
    for relative in references:
        path = contained(model_root, relative)
        if path.suffix == '.json':
            plan = read(path)
            exact(plan, 'schema_version coordinate_frame plan_id session_id issued_at_s valid_from_s expires_at_s time_parameterized_waypoints receipt_sha256')
            for point in plan['time_parameterized_waypoints']:
                exact(point, 'time_s forward_m right_m')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in sorted(references):
        source = contained(model_root, relative)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            os.link(source, target)
        else:
            shutil.copyfile(source, target)
        hashes[relative] = sha(target)
    manifest = {'schema_version': 'ue-fixed-sensory-v1', 'calibration': header['calibration'], 'episodes': episodes}
    write(output / 'manifest.json', manifest)
    hashes['manifest.json'] = sha(output / 'manifest.json')
    identity = {'schema': 'ue-fixed-sensory-integrity-v1', 'files': hashes,
                'source_model_manifest_sha256': sha(model_root / 'manifest.json'),
                'source_episode_descriptor_sha256': {p.parent.name: sha(p) for p in sorted(model_root.glob('episode_*/episode.json'))},
                'episodes': len(episodes), 'frames': sum(len(e['frames']) for e in episodes),
                'storage': 'hardlinks_hash_checked' if hardlink else 'independent_copies_hash_checked',
                'authority': 'CONSUMED_SYNTHETIC_DEVELOPMENT_FIXED_INPUT',
                'trajectory': 'RECORDED_ONLY_NO_COUNTERFACTUAL_MOTION_AUTHORITY'}
    write(output / 'integrity.json', identity)
    (output / 'integrity.sha256').write_text(sha(output / 'integrity.json'), encoding='ascii')
    verify_dataset(output)
    return identity


def verify_dataset(root):
    root = Path(root).resolve()
    if sha(root / 'integrity.json') != (root / 'integrity.sha256').read_text().strip():
        raise ValueError('Dataset integrity receipt changed')
    receipt = read(root / 'integrity.json')
    for relative, expected in receipt['files'].items():
        if sha(contained(root, relative)) != expected:
            raise ValueError(f'Frozen sensory input changed: {relative}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(receipt['files']) | {'integrity.json', 'integrity.sha256'}:
        raise ValueError('Unexpected file in sensory dataset')
    for episode in read(root / 'manifest.json')['episodes']:
        validate_episode(episode)
    return receipt


def causal_prefixes(episode, max_frames=None):
    """The predictor receives all observations up to t and none after t."""
    limit = min(len(episode.observations), max_frames or len(episode.observations))
    for count in range(1, limit + 1):
        yield replace(episode, observations=episode.observations[:count])


def replay_dataset(dataset, output, *, max_frames=None, episode_ids=None, weights=None, mode='perception',
                   engine='incremental'):
    if mode != 'perception':
        raise ValueError('Fixed input replay cannot create/evaluate a new motion trajectory; run UE closed-loop')
    if engine not in ('incremental', 'batch-prefix'):
        raise ValueError('Unknown prediction engine')
    started = time.perf_counter()
    receipt = verify_dataset(dataset)
    import ue_dtr_replay as replay
    from ue_replay_cache import cached_replay_inputs
    contract = replay.load_contract(Path(dataset))
    episodes = [e for e in contract.episodes if episode_ids is None or e.episode_id in episode_ids]
    if not episodes or (episode_ids and set(episode_ids) != {e.episode_id for e in episodes}):
        raise ValueError('Unknown or empty episode selection')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'yolo-config')
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA backend required for this GPU-first model replay')
    weights = Path(weights or replay.detector.DEFAULT_MODEL).resolve(strict=True)
    model_hash = sha(weights)
    model = YOLO(str(weights), task='segment').to('cuda:0')
    names = replay.detector.model_names(model)
    rows, inference_s, predictor_s, engine_stats = [], 0., 0., {}
    for episode in episodes:
        candidates = []
        incremental = replay.IncrementalX73(episode.episode_id, episode.route_frame, contract.calibration) if engine == 'incremental' else None
        limit = min(len(episode.observations), max_frames or len(episode.observations))
        for observation in episode.observations[:limit]:
            source = {'episode_id': episode.episode_id, 'sample_index': observation.sample_index,
                      'time_s': observation.time_s, 'world_frame': observation.world_frame,
                      'frame_id': f'{episode.episode_id}/{observation.sample_index:06d}'}
            tick = time.perf_counter()
            candidates.append(replay.detector.infer_one(model, names,
                replay.detector.ImageInput(observation.rgb.path, source, observation.rgb.sha256),
                model_path=weights, model_sha256=model_hash, device='0', run_kind='FIXED_SENSORY_REPLAY'))
            torch.cuda.synchronize()
            inference_s += time.perf_counter() - tick
            tick = time.perf_counter()
            if incremental is not None:
                row = incremental.update(observation, candidates[-1])
            elif len(candidates) == 1:
                row = {'episode_id': episode.episode_id, 'sample_index': observation.sample_index,
                       'time_s': observation.time_s, 'event': 'WARMUP', 'route_risk': False,
                       'risk_state': 'UNKNOWN_INSUFFICIENT_HISTORY'}
            else:
                with cached_replay_inputs():
                    prediction = replay.predict_episode(replace(episode, observations=episode.observations[:len(candidates)]), candidates, contract.calibration)
                row = replay.compact_rows(episode.episode_id, prediction)[-1]
            predictor_s += time.perf_counter() - tick
            rows.append(dict(row, input_prefix_frames=len(candidates)))
        if incremental is not None:
            engine_stats[episode.episode_id] = incremental.stats
        print(f'replayed {episode.episode_id}: {len(candidates)} frames with {engine}', flush=True)
    report = {'schema': 'ue-fixed-perception-replay-v1', 'status': 'COMPLETE', 'frames': len(rows),
              'dataset_integrity_sha256': sha(Path(dataset) / 'integrity.json'),
              'dataset_frames': receipt['frames'], 'selected_episodes': [e.episode_id for e in episodes],
              'max_frames_per_episode': max_frames, 'model_sha256': model_hash,
              'device': str(next(model.model.parameters()).device), 'gpu': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'elapsed_s': time.perf_counter() - started,
              'inference_s': inference_s, 'predictor_s': predictor_s,
              'prediction_engine': engine, 'incremental_stats': engine_stats,
              'ue_launched': False, 'evaluator_truth_opened': False,
              'authority': 'CONSUMED_SYNTHETIC_DEVELOPMENT_FIXED_INPUT',
              'trajectory': 'RECORDED_ONLY_NO_COUNTERFACTUAL_MOTION_AUTHORITY'}
    write(output / 'replay.json', report)
    (output / 'frames.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--source-run', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--hardlink', action='store_true')
    verify = commands.add_parser('verify')
    verify.add_argument('--dataset', type=Path, required=True)
    run = commands.add_parser('replay')
    run.add_argument('--dataset', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--episode', action='append')
    run.add_argument('--max-frames', type=int)
    run.add_argument('--weights', type=Path)
    run.add_argument('--mode', choices=['perception'], default='perception')
    run.add_argument('--engine', choices=['incremental', 'batch-prefix'], default='incremental')
    args = parser.parse_args()
    if hasattr(args, 'output') and not args.output.resolve().is_relative_to((REPO / 'artifacts.local').resolve()):
        parser.error('Output must remain under canonical artifacts.local')
    if getattr(args, 'max_frames', None) is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if args.command == 'export':
        result = export_dataset(args.source_run, args.output, hardlink=args.hardlink)
    elif args.command == 'verify':
        result = verify_dataset(args.dataset)
    else:
        result = replay_dataset(args.dataset, args.output, max_frames=args.max_frames,
                                episode_ids=args.episode, weights=args.weights, mode=args.mode, engine=args.engine)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
