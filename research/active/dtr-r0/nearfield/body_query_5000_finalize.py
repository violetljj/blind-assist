"""Bind complete expanded collection chunks into a readable source index."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from body_query_collection_labels import sha,summarize

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,nargs='+',required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--visual-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(root) and not a.output.exists()
    plan=read(a.plan);expected={}
    for chunk in plan['outputs']:
        assert sha(chunk['spec'])==chunk['sha256']
        for case in read(chunk['spec'])['cases']:
            assert case['name'] not in expected
            expected[case['name']]=case
    assert len(expected)==5000
    rows=[];bindings=[];sites={};groups={};rgb_roles=defaultdict(set);observed=set();label_contract=None
    for folder in a.labels:
        folder=folder.resolve();m=read(folder/'manifest.json');receipt=read(folder/'receipt.json')
        contract=dict(source_sha256=m['source_sha256'],calibration=m['calibration'])
        if label_contract is None:label_contract=contract
        assert contract==label_contract,'Mixed label-generation contracts'
        assert receipt['manifest_sha256']==sha(folder/'manifest.json')
        assert m['status']=='PASS' and m['complete_group_acceptance']
        assert sha(folder/'result.json')==m['result_sha256']
        for path,h in m['inputs'].items():assert sha(path)==h,'Changed capture input: '+path
        for relative,h in m['arrays'].items():
            path=(folder/relative).resolve();assert path.is_relative_to(folder) and sha(path)==h
        result=read(folder/'result.json');captures={}
        for row in result['records']:
            cpath=Path(row['capture']);key=str(cpath)
            if key not in captures:captures[key]=read(cpath/'source/spec.json')
            case=captures[key]['cases'][row['sample_index']]
            assert case==expected[case['name']] and case['name'] not in observed
            observed.add(case['name'])
            assert case['group_id']==row['group_id'] and row['intent_matches']
            record=dict(row,frame_id=len(rows),source_frame_id=row['frame_id'],
                label_file=str(folder/row['labels']['path']),site_id=case['site_id'],camera=case['camera'],
                rgb_file=str(cpath/f"model/sample/{row['sample_index']:04d}.png"),
                native_file=str(cpath/f"evaluator/native/{row['sample_index']:04d}.npy"),
                local_geometry_sha256=case['local_geometry_sha256'],original_fixture_group_id=case['original_fixture_group_id'])
            identity=(case['region_id'],case['camera']['x'],case['camera']['y'])
            assert sites.setdefault(case['site_id'],identity)==identity
            assert groups.setdefault(case['group_id'],case['source_role'])==case['source_role']
            rgb_roles[row['rgb_sha256']].add(row['source_role']);rows.append(record)
        bindings.append(dict(folder=str(folder),manifest_sha256=sha(folder/'manifest.json')))
    assert len(rows)==5000 and len(groups)==1000 and len(sites)==250
    assert len({v[1:] for v in sites.values()})==250
    assert all(len(v)==1 for v in rgb_roles.values()),'Identical RGB across roles'
    stats=summarize(rows);assert stats['complete_group_acceptance']
    region_sites={r:len({v['site_id'] for v in rows if v['region_id']==r}) for r in {v['region_id'] for v in rows}}
    assert len(region_sites)==10 and set(region_sites.values())=={25}
    assert all(n==20 for n in Counter(r['site_id'] for r in rows).values())
    separation={}
    for region in region_sites:
        positions=[v[1:] for v in sites.values() if v[0]==region]
        separation[region]=min(math.dist(a,b) for i,a in enumerate(positions) for b in positions[i+1:])
    assert min(separation.values())>=6.-1e-7
    review=read(a.visual_review);by_name={r['name']:r for r in rows};reviewed=set()
    assert set(review['region_reviews'])==set(region_sites)
    for region,item in review['region_reviews'].items():
        assert len(item['frames'])==40
        assert {f['sample_index'] for f in item['frames']}==set(range(20))|set(range(480,500))
        for frame in item['frames']:
            assert frame['name'] not in reviewed
            actual=by_name[frame['name']]
            assert actual['region_id']==region and actual['rgb_sha256']==frame['rgb_sha256']
            assert frame['visual_status']=='NO_VISUAL_DEFECT_IDENTIFIED'
            reviewed.add(frame['name'])
    a.output.mkdir(parents=True)
    index=a.output/'index.json';index.write_text(json.dumps(dict(frames=rows),indent=2))
    result=dict(status='PASS',frames=5000,complete_groups=1000,unique_xy_sites=250,region_sites=region_sites,
        unique_rgb_hashes=len(rgb_roles),repeated_rgb_frames=5000-len(rgb_roles),
        unique_geometry_state_hashes=len({r['local_geometry_sha256'] for r in rows}),
        role_frames=dict(Counter(r['source_role'] for r in rows)),
        minimum_xy_separation_m=separation,counts=stats['counts'],
        headings=dict(Counter(str(r['camera']['yaw']) for r in rows)),
        region_HEAD_range_coverage=stats['region_HEAD_range_coverage'],
        rejected_groups=0,model_inference_frames=0,training_steps=0,
        selected_visual_review_frames=len(reviewed),
        authority='SOURCE_COLLECTION_ONLY; controlled shared-asset Development; no strict background isolation')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    (a.output/'manifest.json').write_text(json.dumps(dict(status='PASS',source_chunks=bindings,
        label_contract=label_contract,
        plan=str(a.plan.resolve()),plan_sha256=sha(a.plan),
        visual_review=str(a.visual_review.resolve()),visual_review_sha256=sha(a.visual_review),
        index_sha256=sha(index),summary_sha256=sha(a.output/'summary.json'),code_sha256=sha(__file__)),indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
