"""Reuse completed, unchanged open-loop controls for a controller-only revision.

This is disclosed Development reuse, never fresh confirmation. Immutable sensor
files are hard-linked; the new run has its own manifests, worker and evaluation.
"""
import json
import os
from pathlib import Path
import shutil


def reuse_open_loop(baseline, output, identity):
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    old = json.loads((baseline/'identity.json').read_text())
    run = json.loads((baseline/'run.json').read_text())
    evaluation = json.loads((baseline/'evaluation.json').read_text())
    if run['status'] != 'COMPLETE' or run['completed_episodes'] != 16:
        raise ValueError('Reuse requires a complete sixteen-episode baseline')
    if not evaluation['all_open_loop_contrasts_pass']:
        raise ValueError('Baseline controls must have passed actual trajectory evaluation')
    if old['cases'] or identity['cases'] or old['map_sha256'] != identity['map_sha256']:
        raise ValueError('Reuse requires the same full catalog and map')
    for name, digest in old['sources'].items():
        if name != 'street_live_policy.py' and identity['sources'].get(name) != digest:
            raise ValueError('Only the motion policy may change for reuse: '+name)
    episodes = [p for p in sorted((baseline/'evaluator/episodes').glob('*.json'))
                if json.loads(p.read_text())['arm'] == 'OPEN_LOOP']
    if len(episodes) != 8:
        raise ValueError('Eight complete open-loop controls required')
    destination = output/'evaluator/episodes'
    destination.mkdir(parents=True, exist_ok=True)
    reused = []
    for path in episodes:
        episode = json.loads(path.read_text())
        if not episode['completed']:
            raise ValueError('Incomplete baseline episode')
        eid = episode['episode_id']
        shutil.copytree(baseline/'model'/eid, output/'model'/eid, copy_function=os.link)
        shutil.copy2(path, destination/path.name)
        reused.append({'episode_id': eid, 'scenario_id': episode['scenario_id'],
                       'frames': len(episode['frames'])})
    receipt = {'authority': 'REUSED_SYNTHETIC_DEVELOPMENT_CONTROLS',
               'baseline': str(baseline), 'baseline_map_sha256': old['map_sha256'],
               'reused_open_loop_episodes': 8, 'new_assisted_episodes_expected': 8,
               'episodes': reused, 'sensor_files': 'immutable hardlinks to completed baseline'}
    (output/'baseline-reuse.json').write_text(json.dumps(receipt, indent=2))
    return receipt
