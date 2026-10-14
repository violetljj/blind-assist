"""Explicit offline reciprocal fragment-family diagnostic on real observations.

Full clip lookahead, label-blind. No pixels, candidates, gaps or observations
are invented. Composite association axes never become observation segments.
"""
import copy,json,time,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import thin_appearance_track_dev_20261014 as T
import thin_appearance_track_fast_dev_20261014 as Fast
import thin_appearance_features_dev_20261014 as F
from atom_camera import ROOT

OUT=ROOT/'artifacts.local/work/thin-appearance-dev-20261014/tracking'
INPUT=OUT/'fast_partial/tracking_anchor_strip.jsonl'
CONFIG=dict(session='073635',family_normal_px=3.,family_direction_deg=3.,family_overlap=.5,
            temporal_normal_px=6.,temporal_direction_deg=8.,temporal_min_overlap=.1,
            unique_margin=2.,max_object_gap_s=2.,episode_gap_s=.5,
            immutable_component_anchor=True,reciprocal_two_transition_witness=True,
            lookahead_frames=69,pixel_source='ONLY_PREEXISTING_CURRENT_NATIVE_AND_REAL_FLOW')


def geometry(a,b):
    pa,ma,u,la=T.C.geom(a);pb,mb,v,lb=T.C.geom(b)
    angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),0.,1.))))
    normal=abs(float((mb-ma)@np.array([-u[1],u[0]])))
    overlap=max(0.,min(max(pa@u),max(pb@u))-max(min(pa@u),min(pb@u)))/max(1.,min(la,lb))
    return normal,angle,overlap


def appearance(a,b,guard=False):
    q=T.compatible(a,b,'anchor_finite' if guard else 'anchor_strip')
    return q


def groupable(a,da,b,db):
    n,r,o=geometry(a,b)
    return n<=3. and r<=3. and o>=.5 and appearance(da,db)['accepted']


def controls():
    image=np.full((200,200,3),190,np.uint8);image[:,98:103]=(65,95,70)
    prep=F.prepare_image(image)
    a=dict(x1=100.,y1=30.,x2=100.,y2=175.)
    b=dict(x1=100.,y1=70.,x2=100.,y2=140.)
    parallel=dict(x1=112.,y1=30.,x2=112.,y2=175.)
    cross=dict(x1=60.,y1=70.,x2=150.,y2=140.)
    disjoint=dict(x1=100.,y1=180.,x2=100.,y2=195.)
    da,db=F.extract_appearance(prep,a),F.extract_appearance(prep,b)
    assert groupable(a,da,b,db)
    assert not groupable(a,da,parallel,da)
    assert not groupable(a,da,cross,da)
    assert not groupable(a,da,disjoint,da)
    return dict(status='PASS',same_line_crop_grouped=True,near_parallel_not_grouped=True,
                crossing_not_grouped=True,empty_interval_not_bridged=True)


def main():
    start=time.perf_counter();ctrl=controls()
    rows=[r for r in T.read(INPUT) if r['key']==CONFIG['session']];assert len(rows)==70
    featurefile=T.FEATURES/'features.jsonl'
    ds={(r['key'],r['rgb_index']):{d['rank']:d['feature'] for d in r['features']} for r in T.read(featurefile)}
    nodes=[];families=[];native_node={};flow_node={};family_merges=0
    for fi,row in enumerate(rows):
        groups=[]
        for ci,c in enumerate(row['candidates']):
            d=ds[row['key'],row['rgb_index']][c['rank']]
            chosen=next((g for g in groups if all(groupable(c,d,row['candidates'][k],ds[row['key'],row['rgb_index']][row['candidates'][k]['rank']]) for k in g)),None)
            if chosen is None:groups.append([ci])
            else:chosen.append(ci);family_merges+=1
        fn=[]
        for group in groups:
            ci=max(group,key=lambda k:(bool(ds[row['key'],row['rgb_index']][row['candidates'][k]['rank']].get('quality',{}).get('stripe_reliable')),T.C.geom(row['candidates'][k])[3]))
            c=row['candidates'][ci];d=ds[row['key'],row['rgb_index']][c['rank']]
            node=dict(index=len(nodes),frame=fi,rgb_index=row['rgb_index'],time_ns=row['device_capture_ns'],
                      native_members=list(group),flow_members=[],representative=c,descriptor=d)
            nodes.append(node);fn.append(node['index'])
            for k in group:native_node[fi,k]=node['index']
        for ci,c in enumerate(row['flow_observations']):
            # A current descriptor was not exported by the source replay.
            # Spatial overlap cannot turn that UNKNOWN into appearance proof,
            # and native component joins cannot silently relabel real flows.
            ni=len(nodes);nodes.append(dict(index=ni,frame=fi,rgb_index=row['rgb_index'],time_ns=row['device_capture_ns'],
              native_members=[],flow_members=[ci],representative=c,descriptor=dict(status='UNKNOWN',reason='CURRENT_FLOW_DESCRIPTOR_NOT_RETAINED')))
            flow_node[fi,ci]=ni
        families.append(fn)
    proposals=[]
    for fi in range(1,len(rows)):
        gap=(rows[fi]['device_capture_ns']-rows[fi-1]['device_capture_ns'])/1e9
        if not 0<gap<=2.:continue
        aa,bb=families[fi-1],families[fi]
        costs=np.full((len(aa),len(bb)),1e6);detail={}
        for i,ai in enumerate(aa):
            a=nodes[ai]
            for j,bj in enumerate(bb):
                b=nodes[bj];n,r,o=geometry(a['representative'],b['representative'])
                if n>6. or r>8. or o<.1:continue
                q=appearance(a['descriptor'],b['descriptor'],True)
                if not q['accepted']:continue
                cost=n+.2*r+.3*q['lab_core_delta']+.2*q['lab_contrast_delta']+3*(1-q['normalized_shape_similarity'])
                costs[i,j]=cost;detail[i,j]=dict(previous_node=ai,current_node=bj,previous_rgb_index=a['rgb_index'],
                    current_rgb_index=b['rgb_index'],normal_distance_px=n,direction_deg=r,visible_overlap=o,appearance=q,
                    cost=cost,capture_gap_s=gap,episode_continuous=gap<=.5,gap_observations_added=0,
                    current_support='REAL_AUTONOMOUS_NATIVE_MEMBERS',motion_scope='offline coherent supporting-axis displacement; no same-physical-point or gyro claim')
        if not len(aa) or not len(bb):continue
        for i in range(len(aa)):
            j=int(np.argmin(costs[i]));cost=costs[i,j]
            if cost>=1e5 or int(np.argmin(costs[:,j]))!=i:continue
            other=[costs[i,k] for k in range(len(bb)) if k!=j and costs[i,k]<1e5]
            other += [costs[k,j] for k in range(len(aa)) if k!=i and costs[k,j]<1e5]
            margin=None if not other else float(min(other)-cost)
            d=detail[i,j];d.update(unique_margin=margin,accepted=False,reason='PENDING_OFFLINE_TWO_TRANSITION_WITNESS')
            if margin is not None and margin<2.:d['reason']='AMBIGUOUS_FAMILY_NO_OBJECT_JOIN'
            proposals.append(d)
    eligible=[d for d in proposals if d['reason']=='PENDING_OFFLINE_TWO_TRANSITION_WITNESS']
    parents=list(range(len(nodes)));members={n['index']:[n['index']] for n in nodes}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for edge in eligible:
        a,b=edge['previous_node'],edge['current_node']
        witness=any(d is not edge and (d['current_node']==a or d['previous_node']==b) for d in eligible)
        if not witness:edge['reason']='NO_OFFLINE_TWO_TRANSITION_WITNESS';continue
        ra,rb=find(a),find(b)
        if ra==rb:continue
        merged=members[ra]+members[rb]
        if len({nodes[k]['frame'] for k in merged})!=len(merged):edge['reason']='CONCURRENT_FAMILY_IDENTITY_CONFLICT';continue
        anchor=nodes[min(merged,key=lambda k:(nodes[k]['frame'],k))]['descriptor']
        if not all(appearance(anchor,nodes[k]['descriptor'],True)['accepted'] for k in merged):
            edge['reason']='IMMUTABLE_COMPONENT_ANCHOR_INCOMPATIBLE';continue
        parents[rb]=ra;members[ra]=merged;del members[rb];edge.update(accepted=True,reason='OFFLINE_RECIPROCAL_FAMILY_WITH_TEMPORAL_WITNESS')
    native_roots=sorted(r for r,mm in members.items() if any(nodes[k]['native_members'] for k in mm))
    object_ids={r:i+1 for i,r in enumerate(native_roots)}
    flow_source_ids=sorted({c['track_id'] for row in rows for c in row['flow_observations']})
    flow_ids={source:len(native_roots)+i+1 for i,source in enumerate(flow_source_ids)}
    component_anchor={r:min(members[r],key=lambda k:(nodes[k]['frame'],k)) for r in native_roots}
    episodes={};times={};objects={};object_times={};results=[]
    for fi,row in enumerate(rows):
        result=copy.deepcopy(row);result.update(arm='offline_family',lookahead_frames=69,offline=True,source_tracking_arm='anchor_strip',
             offline_association_scope='hypothesis on existing real members; supporting axes are not observation spans')
        result['families']=[];result['offline_edges']=[d for d in proposals if nodes[d['current_node']]['frame']==fi]
        bynode={}
        for kind,indexer in (('candidates',native_node),('flow_observations',flow_node)):
            for ci,c in enumerate(result[kind]):
                ni=indexer[fi,ci];root=find(ni)
                oi=object_ids[root] if kind=='candidates' else flow_ids[c['track_id']]
                bynode.setdefault(ni,[]).append((kind,ci,c))
                for k in ('anchor_descriptor_id','anchor_sha256','anchor_rgb_index','anchor_status'):
                    if k in c:c['appearance_source_'+k]=c[k]
                if kind=='candidates':
                    anchor_node=nodes[component_anchor[root]];d=anchor_node['descriptor']
                    known=d['status']=='AVAILABLE'
                    c.update(anchor_descriptor_id=f'offline_node:{anchor_node["index"]}' if known else None,
                             anchor_sha256=T.desc_hash(d) if known else None,
                             anchor_rgb_index=anchor_node['rgb_index'] if known else None,
                             anchor_status='OFFLINE_COMPONENT_AVAILABLE' if known else 'UNKNOWN_NOT_ESTABLISHED')
                else:
                    c.update(anchor_descriptor_id=None,anchor_sha256=None,anchor_rgb_index=None,anchor_status='SOURCE_FLOW_DESCRIPTOR_UNKNOWN',
                             identity_status='SOURCE_REAL_FLOW_HYPOTHESIS_ONLY_NO_NATIVE_COMPONENT_JOIN')
                c.update(appearance_source_track_id=c['track_id'],appearance_source_episode_id=c['episode_id'],
                         appearance_source_association=c['association'],track_id=oi,object_id=oi,family_id=ni,
                         lookahead_frames=69,offline=True,association=dict(source='OFFLINE_RECIPROCAL_FAMILY_GRAPH' if kind=='candidates' else 'SOURCE_FLOW_NAMESPACE_ONLY',node=ni,object_id=oi))
        for ni,actual in bynode.items():
            oi=actual[0][2]['object_id'];t=row['device_capture_ns'];last=objects.get(oi)
            if last is None:episodes[oi]=0;times[oi]=[]
            elif t-last>.5e9:episodes[oi]+=1;times[oi]=[]
            objects[oi]=t;times[oi].append(t);object_times.setdefault(oi,[]).append(t)
            object_hits=len(object_times[oi])
            for j,(kind,ci,c) in enumerate(actual):
                stable=kind=='candidates' and len(times[oi])>=3
                c.update(episode_id=episodes[oi],object_independent_hits=object_hits,object_established=object_hits>=3,
                         episode_independent_hits=len(times[oi]),hits=len(times[oi]),independent_hits=len(times[oi]),
                         episode_stable=stable,stable=stable,counted_new_image=j==0,
                         family_member_support='THIS_ACTUAL_MEMBER_SEGMENT_ONLY')
            result['families'].append(dict(family_id=ni,object_id=oi,episode_id=episodes[oi],
                actual_support_members=[dict(kind=kind,index=ci,segment=[c[k] for k in ('x1','y1','x2','y2')]) for kind,ci,c in actual],
                independent_observations_this_image=1,composite_segment_is_observation=False))
        results.append(result)
    path=OUT/'tracking_offline_family.jsonl'
    with path.open('w',encoding='utf8') as f:
        for r in results:f.write(json.dumps(r,allow_nan=False)+'\n')
    assert sum(len(r['candidates']) for r in results)==562
    for a,b in zip(rows,results):
        for field in ('candidates','flow_observations'):
            assert len(a[field])==len(b[field])
            for c,d in zip(a[field],b[field]):assert all(c[k]==d[k] for k in ('x1','y1','x2','y2','rank'))
        assert len({f['object_id'] for f in b['families']})==len(b['families'])
    receipt=dict(complete=True,frames=70,native_candidates=562,original_flow_observations=sum(len(r['flow_observations']) for r in rows),
                 current_native_family_merges=family_merges,family_nodes=len(nodes),object_hypotheses=len(native_roots)+len(flow_ids),
                 native_object_components=len(native_roots),source_flow_identity_hypotheses=len(flow_ids),
                 edge_proposals=len(proposals),edges_accepted=sum(d['accepted'] for d in proposals),
                 gap_edges_accepted=sum(d['accepted'] and not d['episode_continuous'] for d in proposals),
                 rejection_reasons=dict(Counter(d['reason'] for d in proposals if not d['accepted'])),config=CONFIG,
                 flow_identity_contract='source-flow namespace retained separately; no UNKNOWN flow relabelled by native component',
                 source_sha256=T.sha(__file__),tracking_input_sha256=T.sha(INPUT),feature_cache_sha256=T.sha(featurefile),
                 output_sha256=T.sha(path),labels_read=False,controls=ctrl,wall_s=time.perf_counter()-start,gpu_s=0)
    T.save('receipt_offline_family.json',receipt);T.save('offline_family_edges.json',proposals);print(json.dumps(receipt))

if __name__=='__main__':main()
