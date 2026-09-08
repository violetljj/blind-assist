"""Audit accepted City metadata and create a region-disjoint Development split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from summarize_city_groups import paired_groups

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def counts(rows):
    return dict(frames=len(rows),
        near_joint=dict(Counter(''.join(map(str,r['body_head'])) for r in rows)),
        distance_states={h:dict(Counter(r['distance_states'][i] for r in rows))
                         for i,h in enumerate(('BODY','HEAD'))})


def audit(capture, output, verify_payloads=False):
    root = capture.resolve(strict=True)
    out = output.resolve()
    artifacts = (REPO/'artifacts.local').resolve()
    require(root.is_relative_to(artifacts) and out.is_relative_to(artifacts) and out != artifacts,
            'Inputs and outputs must be under artifacts.local')
    require(not out.exists(), 'Refuse to overwrite audit')
    paths = {n:root/n for n in ('evaluator/spec.json','receipt.json','world-verification.json',
                               'group-summary.json','completion.json')}
    spec, receipt, world, quality, completion = [read(p) for p in paths.values()]
    require(all(x['status']=='PASS' for x in (receipt,world,quality,completion)), 'Capture QA must pass')
    require(world['source_spec_sha256']==quality['source_spec_sha256']==sha(paths['evaluator/spec.json']), 'Spec identity mismatch')
    require(world['receipt_sha256']==sha(paths['receipt.json']), 'Receipt mismatch')
    require(quality['world_verification_sha256']==sha(paths['world-verification.json']), 'World report mismatch')
    cases = spec['cases']
    groups = paired_groups(cases)
    require(len(world['rows'])==len(quality['rows'])==len(cases)==completion['frames'], 'Frame counts mismatch')
    require(receipt['frame_count']==quality['frame_count']==len(cases) and quality['group_count']==len(groups), 'Group counts mismatch')
    require(receipt['source_unchanged'] is True, 'Map mutated')
    # Fixed by region names before looking at outcomes. Adjacent one-metre poses
    # remain together; two regions cannot supply three independent partitions.
    assignment = {'west_sidewalk':'train','plaza':'test'}
    split_rows = []
    region_rows = defaultdict(list)
    by_asset, by_variant, by_distance = defaultdict(list), defaultdict(list), defaultdict(list)
    poses = defaultdict(set)
    duplicates = defaultdict(list)
    low_support, missing_center, report_flags = [], [], []
    verified = 0
    for i,(case,w,q) in enumerate(zip(cases,world['rows'],quality['rows'])):
        require(w['sample_index']==q['index']==i and w['name']==case['name'], 'Row alignment mismatch')
        require(q['group_id']==case['group_id'] and q['variant']==case['variant_id'], 'Group alignment mismatch')
        require(q['body_head']==w['body_head_visible_targets'] and q['distance_states']==[d['state'] for d in w['distance_evidence']], 'Label mismatch')
        region = case['group_id'].split('__')[0]
        require(region in assignment, 'Unknown region; declare split explicitly')
        region_rows[region].append(q)
        by_asset[q['asset']].append(q)
        by_variant[q['variant']].append(q)
        center = cases[groups[case['group_id']]['center']]
        anchor_distance = round(center['objects'][0]['center_m'][0]-case['camera']['x'],6)
        by_distance[str(anchor_distance)].append(q)
        camera = case['camera']
        poses[region].add(tuple(camera[k] for k in ('x','y','z','pitch','yaw','roll')))
        duplicates[w['rgb_sha256']].append(i)
        split_rows.append(dict(sample_index=i, group_id=case['group_id'], region=region,
            split=assignment[region], rgb_path=f'model/sample/{i:04d}.png', rgb_sha256=w['rgb_sha256'],
            mask_path=w['mask_path'], mask_sha256=w['mask_sha256']))
        for h,n in enumerate(w['visible_pixels_per_height']):
            if 0 < n < 8:
                low_support.append(dict(index=i,head=('BODY','HEAD')[h],pixels=n))
        if q['variant']=='center' and all(s=='NO_VISIBLE_SUPPORT' for s in q['distance_states']):
            missing_center.append(i)
        if q['unknown_fraction']>.5 or q['near_black_fraction']>.95 or q['near_white_fraction']>.95:
            report_flags.append(i)
        if verify_payloads:
            for rel, digest in ((f'model/sample/{i:04d}.png',w['rgb_sha256']),
                                (f'evaluator/native/{i:04d}.npy',w['native_sha256']),
                                (w['mask_path'],w['mask_sha256'])):
                require(sha(root/rel)==digest, 'Payload hash mismatch: '+rel)
                verified += 1
    require(set(region_rows)==set(assignment), 'Both declared regions required')
    group_splits = defaultdict(set)
    for r in split_rows:
        group_splits[r['group_id']].add(r['split'])
    require(all(len(v)==1 for v in group_splits.values()), 'Triplet leakage')
    cross_duplicates = [ids for ids in duplicates.values() if len({split_rows[i]['split'] for i in ids})>1]
    require(not cross_duplicates, 'Exact RGB duplicates cross partitions')
    minimum_xy = min(math.hypot(a[0]-b[0],a[1]-b[1]) for a in poses['west_sidewalk'] for b in poses['plaza'])
    split = dict(schema='city-region-development-split-v1', assignment=assignment,
        validation='NO_CITY_VALIDATION_PARTITION; use separately declared validation data, never tune on test',
        authority='Same-map region-disjoint Development; inspected capture is not blind or unseen-world evidence',
        input_hashes={n:sha(p) for n,p in paths.items()}, frames=split_rows,
        minimum_cross_split_camera_xy_m=minimum_xy, cross_split_exact_rgb_duplicates=0)
    result = dict(status='PASS', schema='city-audit-v1', backend='CPU',
        backend_reason='TASK_NOT_GPU_SUITABLE_METADATA_AND_FILE_HASHES',
        source_hashes=split['input_hashes'], code_sha256=sha(Path(__file__)),
        groups=len(groups), total=counts(quality['rows']),
        by_region={k:counts(v) for k,v in region_rows.items()},
        by_asset={k:counts(v) for k,v in by_asset.items()},
        by_variant={k:counts(v) for k,v in by_variant.items()},
        by_anchor_distance_m={k:counts(v) for k,v in by_distance.items()},
        camera_poses={k:len(v) for k,v in poses.items()},
        review=dict(inherited_observations=quality['observations'], severe_rgb_or_unknown_flags=report_flags,
            positive_support_under_8_pixels=low_support, center_no_visible_support_indices=missing_center,
            scope='Metadata QA; no object identity/occlusion oracle. NO_VISIBLE_SUPPORT is not certified free space'),
        exact_rgb_duplicate_clusters=[ids for ids in duplicates.values() if len(ids)>1],
        raw_payload_files_rehashed=verified,
        split_counts={s:dict(frames=sum(r['split']==s for r in split_rows),
                            groups=sum(v=={s} for v in group_splits.values())) for s in ('train','test')},
        minimum_cross_split_camera_xy_m=minimum_xy,
        readiness='HEAD_DANGER_SUPERVISION_MISSING' if not counts(quality['rows'])['distance_states']['HEAD'].get('DANGER') else 'REVIEW_CLASS_COUNTS')
    out.mkdir(parents=True)
    for name,value in (('audit.json',result),('split.json',split)):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-payloads',action='store_true')
    a=p.parse_args()
    r=audit(a.capture,a.output,a.verify_payloads)
    print(json.dumps({k:r[k] for k in ('status','groups','total','split_counts','readiness')}))
