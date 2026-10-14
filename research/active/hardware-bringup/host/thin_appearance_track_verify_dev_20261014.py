"""Focused observation, clock, counter, immutable-anchor and archive checks."""
import hashlib,json,gzip
from collections import Counter
from pathlib import Path
import thin_appearance_track_dev_20261014 as T
from atom_camera import ROOT

OUT=T.OUT


def main():
    producer=T.read(T.IN/'frames.jsonl');lookup={(r['key'],r['rgb_index']):r for r in producer}
    specs=[(OUT/'fast_partial'/f'tracking_{a}.jsonl',OUT/'fast_partial/anchors.jsonl',False) for a in ('local_lab','local_strip','anchor_strip')]
    specs += [(OUT/'tracking_anchor_finite.jsonl',OUT/'anchors_finite.jsonl',False),
              (OUT/'tracking_local_lab_partial.jsonl',OUT/'anchors_partial.jsonl',False),
              (OUT/'h8_anchor/tracking_anchor_strip.jsonl',OUT/'h8_anchor/anchors.jsonl',False),
              (OUT/'tracking_offline_family.jsonl',None,True)]
    stats=[];checks=0;first_ep=0;flow_count=0
    for path,anchorpath,offline in specs:
        rows=T.read(path);obj={};epi={};anchor={};images={};originalflows={}
        anchor_records={} if anchorpath is None else {(r['arm'],r['key'],r['track_id']):r for r in T.read(anchorpath)}
        for row in rows:
            raw=lookup[row['key'],row['rgb_index']]
            assert row['device_capture_ns']==raw['device_capture_ns'] and row['rgb_file']==raw['rgb_file']
            assert len(row['candidates'])==len(raw['candidates'])
            for a,b in zip(raw['candidates'],row['candidates']):assert all(a[k]==b[k] for k in a);checks+=1
            grouped={}
            for c in row['candidates']+row['flow_observations']:
                grouped.setdefault(c['object_id'],[]).append(c)
                if c['observation_kind']=='FLOW_SUPPORTED':
                    assert c['flow_support']['pixel_supported'] and c['current_depth_support']=='UNKNOWN_NOT_MEASURED';flow_count+=1
                    if offline:
                        assert c['association']['source']=='SOURCE_FLOW_NAMESPACE_ONLY'
                        assert c['anchor_status']=='SOURCE_FLOW_DESCRIPTOR_UNKNOWN'
                        assert c['family_id'] not in {n['family_id'] for n in row['candidates']}
            for oi,cc in grouped.items():
                if not offline:assert len(cc)==1
                else:assert sum(c['counted_new_image'] for c in cc)==1
                c=cc[0];identity=row['key'],oi;ep=(*identity,c['episode_id']);t=row['device_capture_ns']
                oo=obj.setdefault(identity,set());ee=epi.setdefault(ep,set());fresh=not ee;oo.add(t);ee.add(t)
                if fresh:
                    assert c['episode_independent_hits']==1 and not c['stable'];first_ep+=1
                for c in cc:
                    assert c['object_independent_hits']==len(oo) and c['episode_independent_hits']==len(ee)
                    assert c['hits']==c['independent_hits']==c['episode_independent_hits']
                    if c['anchor_status'] in ('AVAILABLE','OFFLINE_COMPONENT_AVAILABLE'):
                        a=anchor.setdefault(identity,(c['anchor_sha256'],c['anchor_descriptor_id']))
                        assert a==(c['anchor_sha256'],c['anchor_descriptor_id'])
                        if not offline:
                            ar=anchor_records[row['arm'],row['key'],oi]
                            assert ar['anchor_sha256']==c['anchor_sha256'] and T.desc_hash(ar['descriptor'])==ar['anchor_sha256']
            images[row['key']]=images.get(row['key'],0)+1
        stats.append(dict(path=str(path.relative_to(OUT)),frames=len(rows),per_clip=images,sha256=T.sha(path),
                          native=sum(len(r['candidates']) for r in rows),flow=sum(len(r['flow_observations']) for r in rows),
                          complete_source_clips=[k for k,n in images.items() if n==sum(r['key']==k for r in producer)],
                          full_654_complete=len(rows)==654))
    # Native offline geometry/proposals stay identical through the contract fix.
    before=T.read(OUT/'offline_pre_repair/tracking_offline_family.jsonl');after=T.read(OUT/'tracking_offline_family.jsonl')
    for a,b in zip(before,after):
        for field in ('candidates','flow_observations'):
            assert len(a[field])==len(b[field])
            for c,d in zip(a[field],b[field]):
                assert all(c[k]==d[k] for k in ('x1','y1','x2','y2','rank'))
                if field=='flow_observations':assert c['flow_support']==d['flow_support']
    old_edges=json.loads((OUT/'offline_pre_repair/offline_family_edges.json').read_text())
    new_edges=json.loads((OUT/'offline_family_edges.json').read_text())
    def edge(d):return {k:v for k,v in d.items() if k not in ('previous_node','current_node')}
    assert [edge(d) for d in old_edges]==[edge(d) for d in new_edges]
    archives=[]
    for folder in ('partial_slow','h8_three_arm_partial'):
        for rec in json.loads((OUT/folder/'archive_manifest.json').read_text()):
            p=OUT/folder/rec['gzip_file']
            assert T.sha(p)==rec['gzip_sha256']
            with gzip.open(p,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==rec['raw_sha256']
            archives.append(dict(path=str(p.relative_to(OUT)),status='LOSSLESS_VERIFIED'))
    codes=[Path(T.__file__),Path(T.Fast.__file__),Path(T.Partial.__file__),
           Path(__file__),Path(__file__).with_name('thin_appearance_track_offline_dev_20261014.py'),
           Path(__file__).with_name('thin_appearance_track_archive_dev_20261014.py')]
    receipt=dict(status='PASS',native_original_field_checks=checks,first_episode_observation_checks=first_ep,
                 actual_flow_support_checks=flow_count,source_input_sha256=T.sha(T.IN/'frames.jsonl'),
                 codes={str(p.relative_to(ROOT)):T.sha(p) for p in codes},outputs=stats,archives=archives,
                 offline_repair_actual_geometry_and_flow_support_unchanged=True,
                 offline_repair_native_edges_unchanged_except_node_numbers=True,
                 unknown_flow_never_inherits_native_component=True,one_independent_observation_per_object_per_image=True,
                 labels_read=False,gpu_s=0)
    T.save('final_verification.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
