"""Source-only empty-view checks and explicit review selection, never model selection."""
import argparse
import copy
import json
from pathlib import Path
from body_query_collection_labels import verify_capture,floor_acceptance,sha

def inspect(root,region):
    spec,world,hashes,probes=verify_capture(root.resolve())
    sites={s['site_id']:s for s in region['primary_sites']+region['reserve_sites']}
    rows=[]
    for i,(case,row) in enumerate(zip(spec['cases'],world['rows'])):
        assert case['site_id'] in sites
        floor=floor_acceptance(case,probes)
        reasons=[]
        if not floor['accepted']:reasons.append('CAMERA_FLOOR_MISMATCH')
        if row['body_head_visible_targets']!=[0,0]:reasons.append('EXISTING_NEAR_BODY_HEAD_WITNESS')
        if row['valid_depth_pixels']<640*360*.3:reasons.append('INSUFFICIENT_KNOWN_DEPTH')
        rows.append(dict(site_id=case['site_id'],sample_index=i,native_candidate=not reasons,reasons=reasons,
            floor=floor,near=row['body_head_visible_targets'],valid_depth_pixels=row['valid_depth_pixels'],
            rgb=str(root/f'model/sample/{i:04d}.png'),rgb_sha256=row['rgb_sha256']))
    return rows,hashes

def main():
    p=argparse.ArgumentParser();p.add_argument('--sites',type=Path,required=True)
    p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--review',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(root) and not a.output.exists()
    source=json.loads(a.sites.read_text());regions={r['region_id']:r for r in source['regions']}
    review=json.loads(a.review.read_text()) if a.review else None
    records=[];accepted=[];grouped={}
    for capture in a.captures:
        region_id=json.loads((capture/'source/spec.json').read_text())['cases'][0]['region_id']
        region=regions[region_id];rows,hashes=inspect(capture,region)
        item=grouped.setdefault(region_id,dict(region_id=region_id,rows=[],inputs={}))
        item['rows']+=rows;item['inputs'].update(hashes)
    for region_id,item in grouped.items():
        region=regions[region_id];rows=item['rows']
        assert len({r['site_id'] for r in rows})==len(rows),'Duplicate source probe site'
        item['native_candidates']=sum(r['native_candidate'] for r in rows)
        if review:
            assert region_id in review['reviewed_regions']
            assert {r['site_id'] for r in rows} <= set(review['reviewed_sites']), 'Unreviewed source probe'
            rejected=review.get('rejected_sites',{})
            ids=[r['site_id'] for r in rows if r['native_candidate'] and r['site_id'] not in rejected][:25]
            assert len(ids)==25,(region_id,len(ids),'Need more source candidates')
            pool={s['site_id']:s for s in region['primary_sites']+region['reserve_sites']}
            ready=copy.deepcopy(region);ready['accepted_sites']=[pool[i] for i in ids];accepted.append(ready)
            item['selected_sites']=ids
        records.append(item)
    result=dict(status='SOURCE_REVIEW_SELECTED' if review else 'NATIVE_SCREEN_ONLY',regions=accepted if review else [],
        inspections=records,sites_sha256=sha(a.sites),review_sha256=sha(a.review) if review else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({r['region_id']:r['native_candidates'] for r in records}))

if __name__=='__main__':main()
