"""Deterministic source-only site proposals from retained native floor scouts."""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path


HEADINGS = ((0, 1, 0), (90, 0, 1), (180, -1, 0), (-90, 0, -1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surface(row, meshes):
    if not row.get('hit') or not 0.4 <= row.get('z', -999) <= 1.3:
        return None
    name = (row.get('component_path', '') + ' ' + meshes.get(row.get('component_path'), '')).lower()
    if any(t in name for t in ('tree', 'wall', 'bench', 'stairs', 'building', 'water', 'roof', 'grill')):
        return None
    if any(t in name for t in ('sidewalk', 'floor', 'pavement')):
        return 2
    if 'ground_collision' in name:
        return 1
    if 'road_' in name:
        return 0
    return None


def candidates(rows, meshes, step=3.0):
    def key(x, y):
        return round(x, 5), round(y, 5)
    grid = {key(r['x'], r['y']): r for r in rows}
    out = []
    for r in rows:
        quality = surface(r, meshes)
        if quality is None:
            continue
        for yaw, dx, dy in HEADINGS:
            # Camera, ahead 3m, and lateral 3m probes. This is sparse evidence,
            # never swept-volume clearance or semantic pedestrian admission.
            points = [r] + [grid.get(key(r['x'] + step*a, r['y'] + step*b))
                           for a, b in ((dx, dy), (-dy, dx), (dy, -dx))]
            if any(p is None or surface(p, meshes) is None for p in points):
                continue
            span = max(p['z'] for p in points) - min(p['z'] for p in points)
            if span > 0.1000001:
                continue
            out.append(dict(camera_xy_m=[r['x'], r['y']], floor_z_m=r['z'], yaw_deg=yaw,
                            surface_preference=quality, local_height_span_m=span,
                            source_floor_camera=r, source_floor_forward3=points[1],
                            source_floor_lateral3=points[2:]))
    return out


def select(pool, count, separation=6.0):
    selected = []
    headings = collections.Counter()
    remaining = list(pool)
    while remaining and len(selected) < count:
        def distance(c):
            return min((math.dist(c['camera_xy_m'], p['camera_xy_m']) for p in selected), default=30.0)
        remaining = [c for c in remaining if distance(c) >= separation - 1e-7]
        if not remaining:
            break
        # Underrepresented cardinal headings first, then known floor/sidewalk,
        # spatial coverage, flatness, and stable coordinate tie breaking.
        best = max(remaining, key=lambda c: (-headings[c['yaw_deg']], c['surface_preference'],
                   min(distance(c), 30.0), -c['local_height_span_m'],
                   -c['camera_xy_m'][0], -c['camera_xy_m'][1], -c['yaw_deg']))
        selected.append(best)
        headings[best['yaw_deg']] += 1
    return selected


def generate(repo, output, primary=25, reserves=15):
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    output.relative_to((repo / 'artifacts.local').resolve())
    if output.exists():
        raise FileExistsError(output)
    source_root = repo / 'artifacts.local/nearfield/city-crossregion-v1-20260908/scouting'
    dense_root = repo / 'artifacts.local/work/city-crossregion-worker-20260908/dense-terminal-v3/dense-source-scouts-v3'
    sources = [(f'big{i:02}', 'train' if i <= 4 else 'dev' if i == 5 else 'eval',
                source_root / f'big_candidate_{i:02}-v4') for i in range(1, 8)]
    sources += [(f'dense{i:02}', split, dense_root / f'dense_candidate_{i:02}')
                for i, split in ((1, 'train'), (7, 'dev'), (2, 'eval'))]
    primary_map = json.loads((sources[0][2] / 'source/spec.json').read_bytes())['map_file']
    regions = []
    for name, split, capture in sources:
        spec_file = capture / 'source/spec.json'
        floor_file = capture / 'evaluator/source-floor-grid.json'
        inventory_file = capture / 'evaluator/hlod-membership.json'
        spec = json.loads(spec_file.read_bytes())
        receipt = json.loads((capture / 'receipt.json').read_bytes())
        if receipt['status'] != 'PASS' or not receipt['source_unchanged']:
            raise ValueError(f'Unsuccessful or modified source: {name}')
        grid = json.loads(floor_file.read_bytes())
        if grid['config']['step_m'] != 3:
            raise ValueError('Expected retained 3m grid')
        inventory = json.loads(inventory_file.read_bytes())
        meshes = {c['component_path']: c.get('static_mesh_asset', '')
                  for a in inventory.get('loaded_actor_paths', []) for c in a.get('primitive_components', [])}
        pool = candidates(grid['rows'], meshes)
        sites = select(pool, primary + reserves)
        for rank, site in enumerate(sites, 1):
            site.update(site_id=f'{name}_site_{rank:03}', rank=rank, status='CANDIDATE_NOT_ADMITTED')
            site['pose_sha256'] = hashlib.sha256(json.dumps(
                {k: site[k] for k in ('camera_xy_m', 'floor_z_m', 'yaw_deg')},
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        regions.append(dict(region_id=name, split=split, map_asset=spec['map_asset'],
            primary_map_file=primary_map, map_sha256=spec['map_sha256'],
            template_spec_file=str(spec_file), template_spec_sha256=sha(spec_file),
            floor_grid_file=str(floor_file), floor_grid_sha256=sha(floor_file),
            primitive_inventory_file=str(inventory_file), primitive_inventory_sha256=sha(inventory_file),
            source_receipt_file=str(capture / 'receipt.json'), source_receipt_sha256=sha(capture / 'receipt.json'),
            original_scout_split=spec['cases'][0].get('split'),
            world_partition_region_m=spec['world_partition_region_m'],
            eligible_heading_candidates=len(pool), eligible_distinct_xy=len({tuple(p['camera_xy_m']) for p in pool}),
            primary_sites=sites[:primary], reserve_sites=sites[primary:],
            status='CANDIDATE_NOT_ADMITTED' if len(sites) >= primary else 'INSUFFICIENT_CANDIDATES'))
    payload = dict(schema='body-query-5000-site-candidates-v1', status='CANDIDATE_NOT_ADMITTED',
        scope='Source-only sparse floor proposals; no model outcomes, route admission, dense clearance, or formal visible-background isolation.',
        selection=dict(primary_per_region=primary, reserves_requested=reserves, minimum_xy_separation_m=6,
                       headings_deg=[0, 90, 180, -90], maximum_local_height_span_m=0.1,
                       floor_z_range_m=[0.4, 1.3], probes='camera, forward3m, lateral +/-3m',
                       priority='balanced headings, known floor/sidewalk, ground collision, road; farthest coverage; flatness',
                       camera_height_policy='New generator adds one fixed eye height to floor_z_m.',
                       role_policy='New split assignment; retained source scouts were already visually inspected.'),
        generator_sha256=sha(__file__), regions=regions)
    payload['primary_site_count'] = sum(len(r['primary_sites']) for r in regions)
    payload['planned_frames'] = payload['primary_site_count'] * 20
    if payload['primary_site_count'] != len(sources) * primary:
        payload['status'] = 'INSUFFICIENT_CANDIDATES'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return payload


def generate_directed(repo, existing_file, review_file, region_id, output, count=20):
    """Propose unobserved poses near reviewed outdoor anchors, never admit them."""
    output = Path(output).resolve()
    output.relative_to((Path(repo) / 'artifacts.local').resolve())
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(Path(existing_file).read_bytes())
    review = json.loads(Path(review_file).read_bytes())
    region = copy.deepcopy(next(r for r in original['regions'] if r['region_id'] == region_id))
    used = region['primary_sites'] + region['reserve_sites']
    reviewed = set(review['reviewed_sites'])
    native_bad = review['region_reviews'][region_id]['native_rejected_sites']
    visual_bad = review['rejected_sites']
    # Original outdoor anchors only; supplemental wall-only views are not anchors.
    anchors = [s for s in used if s['rank'] <= 40 and s['site_id'] in reviewed
               and s['site_id'] not in native_bad and s['site_id'] not in visual_bad]
    rejected = [s for s in used if s['site_id'] in visual_bad]
    assert anchors and rejected and {s['site_id'] for s in used} <= reviewed
    for key in ('floor_grid', 'primitive_inventory', 'template_spec', 'source_receipt'):
        assert sha(region[key + '_file']) == region[key + '_sha256'], key
    inventory = json.loads(Path(region['primitive_inventory_file']).read_bytes())
    meshes = {c['component_path']: c.get('static_mesh_asset', '')
              for a in inventory['loaded_actor_paths'] for c in a.get('primitive_components', [])}
    floor = json.loads(Path(region['floor_grid_file']).read_bytes())
    assert floor['config']['step_m'] == 3
    pool = candidates(floor['rows'], meshes)

    def nearest(xy, sites):
        site = min(sites, key=lambda s: (math.dist(xy, s['camera_xy_m']), s['site_id']))
        return dict(site_id=site['site_id'], distance_m=math.dist(xy, site['camera_xy_m']))

    directed = []
    for candidate in pool:
        xy = candidate['camera_xy_m']
        forward = [candidate['source_floor_forward3'][k] for k in ('x', 'y')]
        a, f = nearest(xy, anchors), nearest(forward, anchors)
        bad, fbad = nearest(xy, rejected), nearest(forward, rejected)
        prior = nearest(xy, used)
        if (prior['distance_m'] < 6 - 1e-7 or max(a['distance_m'], f['distance_m']) > 12 + 1e-7
                or min(bad['distance_m'], fbad['distance_m']) < 12 - 1e-7
                or a['distance_m'] > bad['distance_m'] or f['distance_m'] > fbad['distance_m']):
            continue
        candidate['directed_evidence'] = dict(camera_anchor=a, forward3_anchor=f,
            camera_rejected_anchor=bad, forward3_rejected_anchor=fbad, nearest_previous_pose=prior)
        directed.append(candidate)
    selected = select(directed, count)
    start = max(s['rank'] for s in used) + 1
    for rank, site in enumerate(selected, start):
        site.update(site_id=f'{region_id}_site_{rank:03}', rank=rank, status='CANDIDATE_NOT_ADMITTED')
        site['pose_sha256'] = hashlib.sha256(json.dumps(
            {k: site[k] for k in ('camera_xy_m', 'floor_z_m', 'yaw_deg')},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    region.update(primary_sites=selected, reserve_sites=[], eligible_heading_candidates=len(directed),
        eligible_distinct_xy=len({tuple(c['camera_xy_m']) for c in directed}),
        status='CANDIDATE_NOT_ADMITTED' if len(selected) == count else 'INSUFFICIENT_CANDIDATES')
    payload = dict(schema='body-query-5000-site-candidates-v1', status=region['status'],
        scope='Source-only directed proposals from existing floor grid and reviewed source views. '
              'Anchor proximity is not outdoor membership, empty-view admission, dense clearance, or background isolation.',
        selection=dict(minimum_xy_separation_m=6, prior_pose_count=len(used),
            maximum_anchor_distance_m=12, minimum_visual_rejected_anchor_distance_m=12,
            proximity_applies_to='camera and forward3m; original dual-eligible anchors only',
            anchor_site_ids=[s['site_id'] for s in anchors], rejected_anchor_site_ids=[s['site_id'] for s in rejected],
            headings_deg=[0, 90, 180, -90], requested_count=count,
            previous_sites_file=str(Path(existing_file).resolve()), previous_sites_sha256=sha(existing_file),
            visual_review_file=str(Path(review_file).resolve()), visual_review_sha256=sha(review_file)),
        generator_sha256=sha(__file__), regions=[region], primary_site_count=len(selected),
        planned_frames=len(selected)*20)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.repo, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'primary_site_count', 'planned_frames')}))
