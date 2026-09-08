"""Merge native HLOD metadata batches and resolve source-actor DAG leaves offline.

The graph retains original edge transforms. Only all-identity transforms are
declared composed; component/instance rendering and visibility remain unverified.
"""
import argparse
import hashlib
import json
from pathlib import Path


def identity_transform(value):
    return (value.get('translation_cm') == [0, 0, 0] and value.get('scale') == [1, 1, 1]
            and value.get('rotation_xyzw') in ([0, 0, 0, 1], [0, 0, 0, -1]))


def merge_graph(descriptors, reports):
    expected = {r['actor_package'] for r in descriptors}
    nodes, conflicting = {}, set()
    for report in reports:
        for node in report['hlod']:
            key = node['proxy_actor_package']
            if key in nodes and nodes[key] != node:
                conflicting.add(key)
            nodes[key] = node
    missing = sorted(expected - nodes.keys())
    unknown = sorted(k for k, v in nodes.items() if v['status'] == 'UNVERIFIED')
    incomplete, cycles = set(), set()
    cache, leaves, nonidentity, mismatches = {}, {}, [], []
    for parent, node in nodes.items():
        for i, source in enumerate(node['sources']):
            nested = source['package'] in expected
            if nested != source['nested_hlod_source']:
                mismatches.append(dict(parent=parent, edge=i, source=source['package']))
            if not all(identity_transform(source.get(k, {})) for k in
                       ('container_transform', 'editor_only_parent_transform')):
                nonidentity.append(dict(parent=parent, edge=i))

    def resolve(package, visiting):
        if package in visiting:
            cycles.add(package)
            return set()
        if package in cache:
            return cache[package]
        if package not in nodes:
            incomplete.add(package)
            return set()
        result = set()
        for source in nodes[package]['sources']:
            if source['package'] in expected or source['nested_hlod_source']:
                result.update(resolve(source['package'], visiting | {package}))
            else:
                # Preserve distinct container instances of a shared actor package.
                key = '|'.join(source[k] for k in ('world_package', 'package', 'container_id', 'actor_instance_guid'))
                if key in leaves and leaves[key] != source:
                    conflicting.add(key)
                leaves[key] = source
                result.add(key)
        cache[package] = result
        return result
    for package in sorted(expected):
        resolve(package, set())
    complete = not (missing or unknown or conflicting or incomplete or cycles or mismatches)
    return dict(schema='city-hlod-source-graph-v1', status='SOURCE_GRAPH_COMPLETE' if complete else 'UNVERIFIED',
                visible_background_isolation='UNVERIFIED', descriptor_hlod_count=len(expected),
                exported_hlod_count=len(nodes), missing_packages=missing, unresolved_nodes=unknown,
                conflicting_nodes_or_leaves=sorted(conflicting), unresolved_nested_packages=sorted(incomplete),
                cycle_packages=sorted(cycles), nested_flag_mismatches=mismatches,
                nonidentity_edges=nonidentity,
                transform_composition='ALL_RECORDED_TRANSFORMS_IDENTITY' if not nonidentity else 'UNVERIFIED_NONIDENTITY_COMPOSITION',
                source_edge_count=sum(len(n['sources']) for n in nodes.values()),
                nested_edge_count=sum(s['package'] in expected for n in nodes.values() for s in n['sources']),
                leaf_identity_count=len(leaves), leaf_package_count=len({s['package'] for s in leaves.values()}),
                nodes=nodes, leaves=leaves, reachable_leaf_ids={k: sorted(v) for k, v in cache.items()},
                limitation='Actor-level package/container/GUID graph, not mesh component-instance mapping, pixel visibility or source admission')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    canonical = (Path(__file__).resolve().parents[1]/'artifacts.local').resolve()
    if a.output.exists() or not a.output.resolve().is_relative_to(canonical):
        raise ValueError('Fresh canonical output required')
    paths = [a.directory/'baseline.json'] + sorted(a.directory.glob('batch-*.json'))
    load = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
    descriptor_path = a.directory/'descriptor-identities.json'
    result = merge_graph(load(descriptor_path), (load(path) for path in paths))
    paths.insert(0, descriptor_path)
    result['input_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status','descriptor_hlod_count','exported_hlod_count',
                      'source_edge_count','nested_edge_count','leaf_identity_count','leaf_package_count','transform_composition')}))


if __name__ == '__main__':
    main()
