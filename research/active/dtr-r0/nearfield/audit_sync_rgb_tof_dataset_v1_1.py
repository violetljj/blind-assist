"""Independent v1.1 geometry/input checks; eval query/model payloads excluded."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')


def preflight(root):
    import sync_rgb_tof_dataset_v1_1 as impl
    started=time.perf_counter()
    rng=np.random.default_rng(4011)
    points=np.column_stack([rng.uniform(-1,1,300),rng.uniform(-1,1,300),rng.uniform(.05,3,300)])
    counts=np.arange(1,len(points)+1)
    pose=np.eye(4);pose[:3,3]=[.05,-.03,.01]
    k=np.array([[80.,0,48.],[0,83.,35.],[0,0,1.]])
    shape=(72,96);y,x=np.indices(shape)
    reference=np.full(shape,np.inf);source=np.zeros(shape,int)
    for p,c in zip(points,counts):
        p=p-pose[:3,3]
        if p[2]<=0:continue
        # Physical metric camera-plane disk distance, not implementation pixel-radius grouping.
        hit=((x-k[0,2])/k[0,0]*p[2]-p[0])**2+((y-k[1,2])/k[1,1]*p[2]-p[1])**2<=.02**2
        hit &= p[2]<reference
        reference[hit]=p[2];source[hit]=c
    observed,support=impl.render(points,counts,pose,k,shape)
    assert np.array_equal(np.isfinite(observed),np.isfinite(reference))
    assert np.allclose(observed[np.isfinite(reference)],reference[np.isfinite(reference)])
    assert np.array_equal(support,source)
    coords=np.floor(points/.02).astype(np.int64)
    keys=impl.voxel_key(points)
    assert len(np.unique(keys))==len(np.unique(coords,axis=0))
    pool={'band0':np.array([.01]),'band1':np.array([-.02]),'band2':np.array([.03])}
    native=np.array([[.3,.799,.8,1.499,1.5,2.999,3.,0.,np.nan]])
    perturbed=impl.perturb(native,pool,0)
    expected=np.array([[.31,.809,.78,1.479,1.53,3.029,np.inf,np.inf,np.inf]])
    assert np.allclose(perturbed,expected,equal_nan=True)
    out=dict(status='PASS',render_surfels=len(points),image_shape=shape,
             tests=['Independent metric disk/zbuffer and winning-source support','2cm voxel key uniqueness','Fixed half-open residual bands including exact3m exclusion'],
             elapsed_s=time.perf_counter()-started,eval_method_payloads_opened=0)
    save(root/'independent_preflight.json',out);return out


def inputs(root,old,surface_only=False):
    from cnh_pose_expected_only import S
    started=time.perf_counter()
    plan=load(root/'PLAN.json')
    assert digest(root/'PLAN.json')=='ccf3a6e12040aa32ce4adae7967ce03fefb23b1bff8f358f74841b38ca87bc4a'
    assert digest(root/'frozen_readout_seal.json')==digest(old/'frozen_readout_seal.json')
    source=load(root/'source_manifest.json')
    assert digest(root/'source_manifest.json')==load(root/'source_seal.json')['source_manifest_sha256']
    source_total=sum(len(e['frames']) for e in source['entries'])
    source_missing=sum(not r['usable_pose'] for e in source['entries'] for r in e['frames'])
    assert source_total==4827 and source_missing==26
    for t in load(root/'source_final_summary.json')['transform_records']:
        if t['status']==200:assert digest(t['path'])==t['sha256']
    src={str(e['capture']):e for e in source['entries']}
    oldm=load(old/'dataset_manifest.json');original={r['source_id']:(e,r) for e in oldm['entries'] for r in e['frames']}
    surfaces=load(root/'surface_manifest.json')['entries'] if (root/'surface_manifest.json').exists() else [load(p) for p in sorted((root/'surfaces').glob('*.json'))]
    surface_counts=[]
    for s in surfaces:
        assert digest(s['path'])==s['sha256']
        with np.load(s['path'],allow_pickle=False) as f:
            p=f['points'];n=f['source_frame_count'];key=f['voxel_key']
        q=np.floor(p/.02).astype(np.int64)+(1<<20)
        independently_packed=(q[:,0].astype(np.uint64)<<42)|(q[:,1].astype(np.uint64)<<21)|q[:,2].astype(np.uint64)
        assert np.array_equal(independently_packed,key)
        assert len(np.unique(key))==len(key)
        assert np.all((n>=1)&(n<=s['frames_used']))
        entry=src[str(s['capture'])]
        allowed={str(s['capture'])}
        if entry.get('transform_path'):
            target=np.load(entry['transform_path'],allow_pickle=False)
            assert np.allclose(target[3],[0,0,0,1])
            for other in source['entries']:
                if str(other['visit_id'])!=str(entry['visit_id']) or not other.get('transform_path'):continue
                transform=np.load(other['transform_path'],allow_pickle=False)
                assert np.allclose(transform[3],[0,0,0,1])
                # Forward equality on independent homogeneous basis validates direction.
                combined=target@np.linalg.inv(transform)
                assert np.allclose(combined@transform,target)
                allowed.add(str(other['capture']))
        seen={r['source_id'] for r in s['frames']}
        assert len(seen)==len(s['frames'])
        assert all(sid.split('_')[0] in allowed for sid in seen)
        expected_source_ids={f['source_id'] for e in source['entries'] if str(e['capture']) in allowed for f in e['frames']}
        assert seen==expected_source_ids
        surface_counts.append(dict(capture=s['capture'],voxels=len(p),frames_used=s['frames_used'],min_support=int(n.min()) if len(n) else 0,max_support=int(n.max()) if len(n) else 0,allowed_captures=sorted(allowed)))
    if surface_only:
        out=dict(status='PASS',surfaces=surface_counts,source_frames=source_total,source_pose_missing=source_missing,
                 elapsed_s=time.perf_counter()-started,eval_query_method_payloads_opened=0)
        save(root/'independent_surface_audit.json',out);return out
    assert sum(s['frames_used'] for s in surfaces)==source_total-source_missing
    with (root/'input_coverage_gate.csv').open(encoding='utf-8-sig') as f:table=list(csv.DictReader(f))
    index={(r['frame_id'],r['arm'],int(r['K'])):r for r in table}
    syn=load(root/'synthesis_manifest.json')['frames'];assert len(syn)==384
    directions,_=S.angular_rays(16);checked=0;near_total=unknown_total=0;counts={}
    for frame in syn:
        e,r=original[frame['source_id']]
        assert (frame['role'],frame['visit_id'])==(e['role'],e['visit_id'])
        assert frame['grid_timestamp_s']==r['grid_timestamp_s'] and frame['rgb_timestamp_s']==r['rgb_timestamp_s']
        with np.load(frame['geometry_path'],allow_pickle=False) as g:
            z=g['depth'];k=g['K'];pose=g['grid_pose']
            u=np.rint(k[0,0]*directions[...,0]/directions[...,2]+k[0,2]).astype(int)
            v=np.rint(k[1,1]*directions[...,1]/directions[...,2]+k[1,2]).astype(int)
            inside=(u>=0)&(u<z.shape[1])&(v>=0)&(v<z.shape[0]);values=z[v.clip(0,z.shape[0]-1),u.clip(0,z.shape[1]-1)]
            hit=inside&np.isfinite(values)&(values>0)
            near=(hit&(values>=.3)&(values<.8)).sum(-1);unknown=(~hit).sum(-1)
            assert np.array_equal(near,g['near_subray_hits']) and np.array_equal(unknown,g['unknown_subrays'])
            assert np.all(g['source_frame_count'][np.isfinite(z)]>=1)
            farocov=hit.mean(-1)
        near_total+=int(near.sum());unknown_total+=int(unknown.sum())
        assert len(frame['arms'])==12
        assert frame['common_available']==(bool(frame['source_available']) and all(a['valid'] for a in frame['arms']))
        for arm in frame['arms']:
            assert digest(arm['path'])==arm['sha256']
            with np.load(arm['path'],allow_pickle=False) as f:
                h=f['hist'];bg=f['background'];cov=f['coverage']
                assert np.array_equal(h,f['counts']-bg)
                seed=int(np.random.SeedSequence([20261011,e['accepted_ordinal'],r['window_id'],r['window_frame'],arm['repeat']]).generate_state(1)[0])
                assert int(f['seed'])==seed
                if arm['arm'].startswith('faro_'):assert np.allclose(cov,farocov)
                peak=h.argmax(-1);height=np.take_along_axis(h,peak[...,None],-1)[...,0];back=np.take_along_axis(bg,peak[...,None],-1)[...,0]
                snr=height/np.sqrt(np.maximum(height,0)+2*back+1)
                geom=int((cov>=.75).sum());joint=int(((cov>=.75)&(snr>=3)).sum())
            record=index[frame['frame_id'],arm['arm'],arm['repeat']]
            assert geom==int(record['geometry_pass_zones']) and joint==int(record['joint_pass_zones'])
            assert int(record['near_subray_hits'])==int(near.sum()) and int(record['unknown_subrays'])==int(unknown.sum())
            for role in ['all',frame['role']]:
                total=counts.setdefault((role,arm['arm'],arm['repeat']),[0,0,0])
                total[0]+=1;total[1]+=geom>=52;total[2]+=joint>=52
            checked+=1
    assert checked==len(table)==4608
    summary=load(root/'coverage_summary.json')
    for r in summary['rows']:
        assert counts[r['role'],r['arm'],r['K']]==[r['frames'],r['geometry_80zone_frames'],r['joint_80zone_frames']]
    primary=counts['all','faro_rho030_ambient1',0]
    assert summary['primary_gate']['PASS']==(primary[2]>=308)
    fit=load(root/'residual_fit.json')
    assert len(fit['frames'])==18 and fit['eval_frames_used']==0
    assert all(r['role'] in ('train','cal') for r in fit['frames'])
    with np.load(root/'residual_pool.npz',allow_pickle=False) as f:
        for j in range(3):assert len(f[f'band{j}'])==sum(r['paired_band_pixels'][j] for r in fit['frames'])
    out=dict(status='PASS',elapsed_s=time.perf_counter()-started,frames=384,input_arm_noise_rows_recomputed=checked,
             primary_gate=dict(frames=primary[0],geometry_80zone_frames=primary[1],joint_80zone_frames=primary[2],PASS=primary[2]>=308),
             surfaces=surface_counts,near_subray_hits=near_total,unknown_subrays=unknown_total,
             source_frames=source_total,surface_used_frames=source_total-source_missing,
             residual_fit_frames=18,eval_residual_fit_frames=0,eval_query_method_payloads_opened=0,
             scope='All4608CNH input coverage/SNR and hashes incl authorized eval input integrity; no eval query or model metrics')
    save(root/'independent_input_audit.json',out);return out


def methods(root,old):
    from audit_sync_rgb_tof_feasibility_dev import interval
    started=time.perf_counter()
    with (root/'train_cal_fusion_queries.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==256*27*6*2*6 and all(r['role'] in ('train','cal') for r in rows)
    common={f['frame_id']:f['common_available'] for f in load(root/'synthesis_manifest.json')['frames'] if f['role']!='eval'}
    assert all((r['source_available']=='True')==common[r['frame']] for r in rows)
    idx={(r['frame'],r['noise_arm'],int(r['repeat']),r['query'],r['arm']):r for r in rows}
    assert len(idx)==len(rows)
    refs={r['source_id']:r for r in load(old/'train_cal_reference_roster.json')['rows']}
    queries=load(old/'dataset_manifest.json')['queries']
    pred={name:{r['rgb_path']:r for r in load(old/'rgb_inference'/model/'predictions.json')['rows']} for name,model in [('dav','dav2'),('uni','unidepth')]}
    seen=set();checked=0;selected=[]
    for frame in load(root/'synthesis_manifest.json')['frames']:
        if frame['role']=='eval' or frame['visit_id'] in seen or not frame['source_available']:continue
        seen.add(frame['visit_id']);selected.append(frame['frame_id']);r=refs[frame['source_id']]
        k=np.asarray(r['depth_K']);shape=tuple(r['depth_shape']);y,x=np.indices(shape)
        rx=(x-k[0,2])/k[0,0];ry=(y-k[1,2])/k[1,1];edge=np.tan(np.deg2rad(22.5))
        fov=(abs(rx)<=edge)&(abs(ry)<=edge);ix=np.clip(np.floor((rx+edge)/(2*edge)*8).astype(int),0,7);iy=np.clip(np.floor((ry+edge)/(2*edge)*8).astype(int),0,7)
        depths={}
        for name in pred:
            pr=pred[name][r['rgb_path']]
            assert digest(pr['path'])==pr['sha256']
            with np.load(pr['path'],allow_pickle=False) as f:depths[name]=f['depth']
        with np.load(r['reference_path'],allow_pickle=False) as f:labels=f['labels']
        for arm in frame['arms']:
            with np.load(arm['path'],allow_pickle=False) as f:
                h=f['hist'];peak=h.argmax(-1);height=np.take_along_axis(h,peak[...,None],-1)[...,0];back=np.take_along_axis(f['background'],peak[...,None],-1)[...,0]
                snr=height/np.sqrt(np.maximum(height,0)+2*back+1);valid=fov&(f['coverage'][iy,ix]>=.75)&(snr[iy,ix]>=3)
                z=(peak[iy,ix]+.5)*.3002784/np.sqrt(1+rx*rx+ry*ry)
                xyz=np.stack([rx[valid]*z[valid],ry[valid]*z[valid],z[valid],np.ones(int(valid.sum()))])
                p=np.linalg.inv(f['rgb_pose'])@f['grid_pose']@xyz;uv=k@p[:3];u,v=np.rint(uv[:2]/uv[2]).astype(int)
                ok=(p[2]>0)&(u>=0)&(u<shape[1])&(v>=0)&(v<shape[0]);tof=np.full(shape,np.inf)
                np.minimum.at(tof.ravel(),v[ok]*shape[1]+u[ok],p[2,ok])
            for j,q in enumerate(queries):
                lo,hi,domain=interval(rx,ry,q);near=q['high'][2]<=.8;masks={}
                for name,z,cut,active in [('tof',tof,0.,True),('dav',depths['dav'],.24403834342956543,not near),('uni',depths['uni'],.09616100788116455,near)]:
                    masks[name]=domain&np.isfinite(z)&(z>0)&(np.minimum(z-lo,hi-z)>=cut)&active
                rgb=masks['uni' if near else 'dav'];masks.update(rgb_banded=rgb,OR=masks['tof']|rgb,AND=masks['tof']&rgb)
                for method,mask in masks.items():
                    rr=idx[frame['frame_id'],arm['arm'],arm['repeat'],q['name'],method];n,w=int(mask.sum()),int((mask&(labels[j]==1)).sum())
                    assert (n,w)==(int(rr['support_pixels']),int(rr['positive_pixels']))
                    assert (rr['support']=='True')==(n>=16) and (rr['witness']=='True')==(w>=16)
                    checked+=1
    buckets={}
    for r in rows:
        for subset in ['all_grid']+(['all_arm_common'] if r['source_available']=='True' else []):
            key=(subset,r['noise_arm'],int(r['repeat']),r['arm'],r['band'])
            b=buckets.setdefault(key,dict(queries=0,POS=0,W=0,FREE=0,F=0,UNKNOWN=0,U=0));b['queries']+=1
            if r['state']=='POSITIVE':b['POS']+=1;b['W']+=r['witness']=='True'
            elif r['state']=='FREE_ON_SAMPLED_RAYS':b['FREE']+=1;b['F']+=r['support']=='True'
            else:assert r['state']=='UNKNOWN';b['UNKNOWN']+=1;b['U']+=r['support']=='True'
    compact=load(root/'train_cal_compact.json')['rows']
    for r in compact:
        b=buckets[r['subset'],r['noise_arm'],r['K'],r['method'],r['band']]
        assert all(r[k]==v for k,v in b.items())
    out=dict(status='PASS',elapsed_s=time.perf_counter()-started,all_table_rows_aggregated=len(rows),compact_rows_checked=len(compact),pixel_rows_recomputed=checked,
             selected_frame_ids=selected,sampling='First source-available frame of each train/cal visit; all6arms xK2 x27queries x6methods',eval_query_method_payloads_opened=0)
    save(root/'independent_methods_audit.json',out);return out


def geometry_checks(root,old):
    from PIL import Image
    from sync_rgb_tof_dataset_v1 import pose_at
    started=time.perf_counter()
    previous={f['frame_id']:f for f in load(old/'synthesis_manifest.json')['frames']}
    new={f['frame_id']:f for f in load(root/'synthesis_manifest.json')['frames']}
    expected={(r['frame_id'],r['band']):r for r in load(root/'original51_geometry_consistency.json')['rows']}
    pools={};frame_counts={};nframes=0
    for e in load(old/'dataset_manifest.json')['entries']:
        trajectory=np.loadtxt(e['trajectory_path'])
        for i,r in enumerate(e['frames']):
            fid=f"{e['capture']}_{r['window_id']}_{i:03d}"
            if not previous[fid]['source_available']:continue
            srcpose,_=pose_at(trajectory,r['highres_timestamp_s']);targetpose,_=pose_at(trajectory,r['grid_timestamp_s'])
            z=np.asarray(Image.open(r['highres_depth_path']),float)/1000.
            y,x=np.nonzero(np.isfinite(z)&(z>0));d=z[y,x]
            xyz=np.linalg.inv(np.asarray(r['highres_K']))@np.stack([x*d,y*d,d])
            p=np.linalg.inv(targetpose)@srcpose@np.vstack([xyz,np.ones(len(d))])
            k=np.asarray(r['depth_K']);projected=k@p[:3];u,v=np.rint(projected[:2]/projected[2]).astype(int);shape=tuple(r['depth_shape'])
            ok=(p[2]>0)&(u>=0)&(u<shape[1])&(v>=0)&(v<shape[0]);original=np.full(np.prod(shape),np.inf)
            np.minimum.at(original,v[ok]*shape[1]+u[ok],p[2,ok]);original=original.reshape(shape)
            with np.load(new[fid]['geometry_path'],allow_pickle=False) as f:reconstructed=f['depth']
            paired=np.isfinite(original)&np.isfinite(reconstructed)&(original>0)&(reconstructed>0)
            for lo,hi in [(0.3,.8),(.8,1.5),(1.5,3.)]:
                band=f'{lo:g}-{hi:g}m';mask=paired&(original>=lo)&(original<hi);err=reconstructed[mask]-original[mask]
                row=expected[fid,band]
                assert len(err)==row['paired_pixels']
                if len(err):
                    assert np.isclose(np.median(err),row['median_signed_m'],atol=1e-9)
                    assert np.isclose(np.median(abs(err)),row['median_abs_m'],atol=1e-9)
                    assert np.isclose(np.quantile(abs(err),.9),row['p90_abs_m'],atol=1e-9)
                for role in [e['role'],'all']:
                    for b in [band,'0.3-3m_all_bands']:
                        pools.setdefault((role,b),[]).append(err)
            nframes+=1;frame_counts[e['role']]=frame_counts.get(e['role'],0)+1
    assert nframes==51 and frame_counts=={'train':14,'cal':4,'eval':33}
    summary=[]
    for (role,band),values in sorted(pools.items()):
        vals=np.concatenate(values)
        summary.append(dict(role=role,band=band,paired_pixels=len(vals),median_signed_m=float(np.median(vals)) if len(vals) else None,
                            median_abs_m=float(np.median(abs(vals))) if len(vals) else None,p90_abs_m=float(np.quantile(abs(vals),.9)) if len(vals) else None))
    out=dict(status='PASS',frames=nframes,split_frames=frame_counts,rows=summary,elapsed_s=time.perf_counter()-started,
             scope='Pooled pixel-level reconstruction self-consistency on original51 source frames, includingeval inputgeometry; source frames contributed to reconstruction, not independent accuracy or query/method evaluation',eval_query_method_payloads_opened=0)
    save(root/'independent_geometry_consistency_pooled.json',out);return out


def frozen_tables(root):
    started=time.perf_counter();rows=load(root/'frozen_inference/train_cal_description.json')['rows']
    assert len(rows)==64 and all(r['role'] in ('train','cal') and r['arm'].startswith('faro_') for r in rows)
    summaries=load(root/'frozen_inference/train_cal_summary.json')['rows']
    for s in summaries:
        subset=[r for r in rows if r['role']==s['role'] and r['arm']==s['arm']]
        height={'HEAD':0,'BODY':1}[s['height']]
        assert s['noise_sequences']==len(subset) and s['slot_denominator']==sum(r['frames'] for r in subset)
        for k in ['m3_alarm_slots','old5_strong_slots','s_light_slots','s_any_alarm_slots']:
            value=sum(r['head_body_'+k][height] for r in subset)
            assert value==s[k] and np.isclose(value/s['slot_denominator'],s[k+'_fraction'])
    out=dict(status='PASS',raw_descriptive_sequences=64,summary_rows=len(summaries),elapsed_s=time.perf_counter()-started,
             scope='Frozen train/cal descriptive counts aggregated independently; eval model outputs not run or read',eval_query_method_payloads_opened=0)
    save(root/'independent_frozen_summary_audit.json',out);return out


def report_check(root,path):
    started=time.perf_counter();body=path.read_text(encoding='utf8')
    compact=load(root/'train_cal_compact.json')['rows'];index={(r['subset'],r['noise_arm'],str(r['K']),r['method'],r['band']):r for r in compact}
    subset=None;cells_checked=0;frozen_checked=0
    frozen=load(root/'frozen_inference/train_cal_summary.json')['rows']
    for line in body.splitlines():
        if line=='### all_grid':subset='all_grid'
        if line=='### all_arm_common':subset='all_arm_common'
        cells=[c.strip() for c in line.split('|')[1:-1]]
        if subset and len(cells)==6 and cells[0] in {k[1] for k in index}:
            for band,value in zip(['0.3-0.8m','0.8-1.5m','1.5-3m'],cells[3:]):
                r=index[subset,cells[0],cells[1],cells[2],band]
                assert value==f"{r['W']}/{r['POS']}；{r['F']}/{r['FREE']}；{r['U']}/{r['UNKNOWN']}"
                cells_checked+=1
        if len(cells)==7 and cells[0] in ('train','cal') and cells[1].startswith('faro_'):
            rr={r['height']:r for r in frozen if r['role']==cells[0] and r['arm']==cells[1]}
            assert int(cells[2])==rr['HEAD']['slot_denominator']==rr['BODY']['slot_denominator']
            for key,value in zip(['m3_alarm_slots','old5_strong_slots','s_light_slots','s_any_alarm_slots'],cells[3:]):
                assert value==f"{rr['HEAD'][key]}/{rr['BODY'][key]}"
                frozen_checked+=1
    assert cells_checked==216 and frozen_checked==32
    out=dict(status='PASS',report_sha256=digest(path),method_cells=cells_checked,frozen_cells=frozen_checked,
             elapsed_s=time.perf_counter()-started,text_review='80/80FAIL103of384; geometry184; original51pooled long tails;5wholevisit7components;eval input-only and residual18; no parameter relaxation; budgets within caps',
             eval_query_method_payloads_opened=0)
    save(root/'report_review_audit.json',out);return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--preflight',action='store_true');ap.add_argument('--old',type=Path);ap.add_argument('--surface-only',action='store_true');ap.add_argument('--methods',action='store_true');ap.add_argument('--geometry-checks',action='store_true');ap.add_argument('--frozen-tables',action='store_true');ap.add_argument('--report',type=Path);a=ap.parse_args()
    if a.preflight:print(json.dumps(preflight(a.root)))
    elif a.report:print(json.dumps(report_check(a.root,a.report)))
    elif a.frozen_tables:print(json.dumps(frozen_tables(a.root)))
    elif a.geometry_checks:print(json.dumps(geometry_checks(a.root,a.old)))
    elif a.methods:print(json.dumps(methods(a.root,a.old)))
    elif a.old:print(json.dumps(inputs(a.root,a.old,a.surface_only)))


if __name__=='__main__':main()
