"""Metadata-only inventory of admitted existing sources; no pixel/model access."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import time

from mz5_ensemble_readout import read, write, sha


def frequencies(rows, key):
    return dict(sorted(Counter(str(r[key]) for r in rows).items()))


def main(root, output):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    work = root/'artifacts.local/work'
    definitions = [('old5000', 'body-query-5000-20260909/dataset-v1', 5000),
                   ('relation10000', 'body-query-10000-20260909/final-dataset-v2', 10000),
                   ('distance5000', 'body-query-distance-5000-20260909/final-dataset-v1', 5000)]
    inventory, all_rows, selected, inputs = {}, {}, [], {}
    for name, relative, expected in definitions:
        folder = work/relative
        manifest = read(folder/'manifest.json')
        assert manifest['status'] == 'PASS'
        for filename in ['manifest.json', 'index.json', 'summary.json']:
            path = folder/filename
            inputs[str(path.resolve())] = sha(path)
            if filename != 'manifest.json':
                assert inputs[str(path.resolve())] == manifest[filename.split('.')[0]+'_sha256']
        # Verify explicit provenance links without recursively reading payloads.
        bindings = {}
        for key, value in manifest.items():
            if isinstance(value, str) and key+'_sha256' in manifest:
                path = Path(value)
                if path.is_absolute():
                    digest = sha(path)
                    assert digest == manifest[key+'_sha256'], (name, key)
                    inputs[str(path.resolve())] = digest
                    bindings[key] = str(path.resolve())
        index = read(folder/'index.json')
        raw = index['records'] if name == 'distance5000' else index['frames']
        assert len(raw) == expected
        rows, ranges, regions = [], [], []
        for i, row in enumerate(raw):
            distance = name == 'distance5000'
            assert row['accepted'] if distance else row['status'] == 'PASS'
            normalized = dict(dataset=name, index=i,
                rgb=row['rgb_path' if distance else 'rgb_file'], rgb_sha=row['rgb_sha256'],
                native=row['native_path' if distance else 'native_file'], native_sha=row['native_sha256'],
                site=row['site_id'], group=row['pair_id' if distance else 'group_id'],
                family=row['family'], condition='HEAD_ONLY' if distance else row['condition'],
                role=row['source_partition'].upper()+'_ONLY' if distance else row['source_role'],
                camera=row['camera'])
            rows.append(normalized)
            ranges.append(row['endpoint' if distance else 'declared_range'])
            regions.append(row['region_id'])
        existing = {key: sum(Path(r[key]).is_file() for r in rows) for key in ['rgb', 'native']}
        assert existing == dict(rgb=expected, native=expected), (name, existing)
        all_rows[name] = rows
        inventory[name] = dict(path=str(folder.resolve()), frames=len(rows), status='PASS',
            roles=frequencies(rows, 'role'), families=frequencies(rows, 'family'),
            conditions=frequencies(rows, 'condition'), declared_range=dict(Counter(ranges)),
            regions=dict(Counter(regions)), sites=len({r['site'] for r in rows}),
            groups=len({r['group'] for r in rows}), unique_rgb_hashes=len({r['rgb_sha'] for r in rows}),
            group_semantics='Rigid near/far assembly pair' if name == 'distance5000' else 'Site/family relation variant group',
            existing_paths=existing, payload_hashes_recomputed=False, provenance_bindings=bindings,
            calibration=manifest.get('label_contract', {}).get('calibration',
                dict(width=640, height=360, hfov_degrees=100, height_m=1.7,
                     source='distance source protocol and admission; not new pixel inspection')))
        if name != 'old5000':
            chosen = [r for r in rows if r['role'] == 'DEV_ONLY']
            assert len(chosen) == (2000 if name == 'relation10000' else 1000)
            selected.extend(chosen)
            inventory[name]['selected_DEV'] = dict(frames=len(chosen), sites=len({r['site'] for r in chosen}),
                groups=len({r['group'] for r in chosen}), families=frequencies(chosen, 'family'),
                conditions=frequencies(chosen, 'condition'))
    overlap = {}
    for a, b in combinations(all_rows, 2):
        overlap[a+'__'+b] = dict(rgb_hash_intersection=len({r['rgb_sha'] for r in all_rows[a]} & {r['rgb_sha'] for r in all_rows[b]}),
            shared_site_ids=len({r['site'] for r in all_rows[a]} & {r['site'] for r in all_rows[b]}))
    usage_paths = [work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json',
        work/'body-query-10000-b-20260909/cache-v1/manifest.json',
        work/'body-query-10000-b-20260909/distance-analysis-v1/receipt.json',
        work/'body-query-context-distance-20260909/run-v1/protocol.json',
        work/'body-query-context-distance-20260909/run-v1/receipt.json']
    for path in usage_paths:
        inputs[str(path.resolve())] = sha(path)
    notes = [
        'old5000 supplies frozen MZ1/MZ5 training and development/evaluation cache; not new data.',
        'relation10000 supplies shared frozen B encoder cache; saved DEV role does not establish unseen backbone exposure.',
        'distance5000 already scored OLD/NEW B and LOCAL/JOINT decoders on all5000 with zero fitting; consumed Development.',
        'Distance near/far pairs share500 relation sites/assets but use different translated physical endpoints.',
        'Distance has only HEAD_ONLY intended positive endpoints: HEAD range attribution/BODY false activation, no HEAD_ANY-negative or BODY-positive denominator.',
        'Crossbar and oblique rod are thin supported horizontal/diagonal members, not the MZ6 vertical pole configuration.',
        'Distance anchors near1.0-1.3m/far2.1-2.6m are placement anchors, not nearest visible distances or a boundary sweep.',
        'Distance background animation not locked;160 reviewed frames retain BACKGROUND_VISUAL_LIMITATION_RETAINED.',
        'Inventory inspects all20000 metadata rows only. No protected EVAL model outcomes, RGB/native pixels, new training or collection. UNKNOWN is not CLEAR.',
        'Selected3000 are predeclared existing-data Development replay, not independent confirmation. Group by sites/configurations, not adjacent samples.'
    ]
    assert len(selected) == 3000
    write(output/'selected.json', selected)
    write(output/'inventory.json', dict(status='PASS', total_frames=20000, datasets=inventory,
        pairwise_overlap=overlap, total_unique_rgb_hashes=len({r['rgb_sha'] for rows in all_rows.values() for r in rows}),
        selected_frames=len(selected), selected_order='relationDEV source index then distanceDEV source index',
        usage_evidence=[str(p.resolve()) for p in usage_paths], notes=notes))
    write(output/'receipt.json', dict(status='PASS', backend='CPU_METADATA_ONLY', seconds=time.perf_counter()-start,
        source_sha256=sha(Path(__file__)), inputs=inputs,
        outputs={name: sha(output/name) for name in ['selected.json','inventory.json']}))
    print('PASS', dict(total=20000, selected=len(selected), overlap=overlap), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.output)
