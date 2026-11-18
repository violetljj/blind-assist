"""Fit the registered edge readout exclusively on the original development material."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import shallow_paired_areas_20261118 as prior
from shallow_edge_method_20261118 import VARIANTS, MIN_AREA, edge_variants, largest_area
from shallow_valid_view_20261118 import valid_view

replay = prior.replay
OUT = prior.OUT.parent / 'shallow-edge-20261118/dev'
METHOD = Path(__file__).with_name('shallow_edge_method_20261118.py')


def measure_source(name):
    manifest = prior.PREVIOUS / prior.SOURCES[name] / 'manifest.json'
    cache = prior.OUT / 'cache' / replay.sha(manifest)[:16]
    labels_path = prior.PREVIOUS / 'scoring/labels' / (name + '.json')
    native = prior.by_frame(replay.inputs(manifest)[1])
    frames = prior.by_frame(replay.read(cache / 'index.json')['rows'])
    labels = replay.read(labels_path)
    receipt = replay.read(prior.OUT / (name + '-areas-camera-valid.json'))
    obj_by_frame, empty_by_frame = defaultdict(list), defaultdict(list)
    events = {e['id']: e for e in receipt['events']}
    windows = {w['id']: w for w in receipt['windows']}
    for case in labels['cases']:
        e = events[case['encounter_id']]
        obs = {(r['i'],r['object_id']): r for r in e['observations']}
        for entry in case['observations']:
            if entry.get('near', True):
                obj = obs[entry['i'],case['object_id']]
                obj_by_frame[entry['seq'],entry['i']].append((entry,obj))
    for window in labels['windows']:
        rows = {r['i']:r for r in windows[window['id']]['rows']}
        for entry in window['frames']:
            if entry['i'] in rows:
                empty_by_frame[entry['seq'],entry['i']].append(rows[entry['i']])
    for key in sorted(obj_by_frame.keys() | empty_by_frame.keys()):
        source = native[key]
        with np.load(cache / frames[key]['file'],allow_pickle=False) as p:
            z = p['z']
            view = valid_view(name,source,z.shape)
            bgr = replay.image(source,tuple(reversed(z.shape)))
            masks, area = edge_variants(bgr,z,p['K'],p['probability'],view)
        total = {v:largest_area(ids,area) for v,ids in masks.items()}
        for row in empty_by_frame[key]:
            row['edge'] = total
        for entry,row in obj_by_frame[key]:
            path = labels_path.parent / entry['reference_footprint']
            assert replay.sha(path) == entry['reference_mask_sha256']
            roi = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if roi.shape != area.shape:
                roi = cv2.resize(roi,tuple(reversed(area.shape)), interpolation=cv2.INTER_NEAREST)
            row['edge'] = total
            row['edge_target'] = {v:largest_area(ids,area,roi > 0) for v,ids in masks.items()}
    replay.save(OUT / (name + '.json'),receipt)
    return receipt


def direct(row):
    return row['old'] or row['total']['literal'] >= MIN_AREA


def count(events,windows,variant=None,threshold=MIN_AREA):
    hits, false_samples, false_windows = {}, [], []
    for e in events:
        hit = set()
        for r in e['observations']:
            if (r['old_target'] or r['target']['literal'] >= MIN_AREA or
                (variant is not None and r['edge_target'][variant] >= threshold)):
                hit.add(r['object_id'])
        hits[e['id']] = sorted(hit)
    for w in windows:
        false = [r['i'] for r in w['rows'] if direct(r) or
                 (variant is not None and r['edge'][variant] >= threshold)]
        if false:
            false_windows.append(w['id'])
            false_samples.extend((w['id'],i) for i in false)
    silent = [e for e in events if e['band']=='shallow' and e['pass_old_alert_count']==0]
    assert len(silent)==5 and len(windows)==8 and sum(len(w['rows']) for w in windows)==55
    return dict(hits=hits,first_events=[e['id'] for e in silent if hits[e['id']]],
                n_first=sum(bool(hits[e['id']]) for e in silent),
                n_first_objects=sum(len(hits[e['id']]) for e in silent),
                false_samples=false_samples, false_windows=false_windows)


def develop():
    cv2.setNumThreads(1)
    receipts = [measure_source(s) for s in prior.SOURCES]
    events = [e for d in receipts for e in d['events']]
    windows = [w for d in receipts for w in d['windows']]
    base = count(events,windows)
    assert len(base['false_windows'])==1 and len(base['false_samples'])==6 and base['n_first']==2
    results = {}
    for v in VARIANTS:
        strongest = max(r['edge'][v] for w in windows for r in w['rows'] if not direct(r))
        threshold = max(MIN_AREA, float(np.nextafter(strongest, np.inf)))
        result = count(events,windows,v,threshold)
        assert result['false_samples'] == base['false_samples']
        results[v] = dict(threshold_m2=threshold,raw=count(events,windows,v),matched=result)
    winner = max(VARIANTS,key=lambda v:results[v]['matched']['n_first'])
    report = dict(registration_sha256=replay.sha(OUT/'registration.txt'),method_sha256=replay.sha(METHOD),
                  direct=base,variants=results,selected=winner,threshold_m2=results[winner]['threshold_m2'],
                  selection='Max correctly attributed events among five previously silent shallow passes at exact development direct-5cm false trace; ties as registered.')
    replay.save(OUT/'freeze.json',report)
    for v,d in results.items():
        print(v,d['threshold_m2'], d['matched']['n_first'], 'raw',d['raw']['n_first'],
              len(d['raw']['false_windows']),len(d['raw']['false_samples']))
    print('FROZEN',winner,report['threshold_m2'],report['method_sha256'],flush=True)


if __name__ == '__main__':
    develop()
