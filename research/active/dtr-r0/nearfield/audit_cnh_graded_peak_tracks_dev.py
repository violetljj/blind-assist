"""Independent scalar native-peak / causal one-to-one track audit (Development)."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009/data'
OUT = ROOT/'artifacts.local/work/cnh-graded-peak-track-dev-20261010/tracks'
BIAS = ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'


def move(point, matrix):
    return point @ matrix[:3, :3].T + matrix[:3, 3]


def reference(c, motion, names):
    """A separate scalar state machine: no production matching/features calls."""
    size=len(c['amplitude']); features=np.full((size,13,2,len(names)),np.nan,np.float32)
    ids=np.full((size,13,2,8),-1,np.int16)
    residual=np.full(ids.shape,np.nan,np.float32); gaps=np.zeros(ids.shape,np.int8)
    selected=np.full((size,13,2),-1,np.int16)
    paths=np.full((size,13,2,5,3),np.nan,np.float32)
    native=np.full((size,13,2,5),-1,np.int16); amplitudes=np.zeros(native.shape,np.float32)
    for n in range(size):
        for q in range(2):
            history={}; next_id=0
            for f in range(13):
                lower=max(0,f-4); candidates=[r for r in range(8) if c['valid'][n,f,q,r]]
                edges=[]
                for tid, observations in history.items():
                    old=observations[-1]
                    if f-old['f']>3:
                        continue
                    previous=move(old['point'],motion[f+3,old['f']+3])
                    for r in candidates:
                        distance=float(np.linalg.norm(previous-c['xyz'][n,f,q,r]))
                        if distance<=.45:
                            edges.append((distance,tid,r))
                ownership={}; used=set()
                for distance,tid,r in sorted(edges):
                    if tid not in used and r not in ownership:
                        ownership[r]=(tid,distance); used.add(tid)
                current=[]
                for r in candidates:
                    if r in ownership:
                        tid,distance=ownership[r]; gap=f-history[tid][-1]['f']-1
                    else:
                        tid=next_id; next_id+=1; history[tid]=[]; distance=np.nan; gap=0
                    o=dict(f=f,r=r,point=c['xyz'][n,f,q,r],amp=float(c['amplitude'][n,f,q,r]),
                           inner=float(c['inner_share'][n,f,q,r]),native=int(c['native_index'][n,f,q,r]),d=distance,gap=gap)
                    history[tid].append(o); ids[n,f,q,r]=tid; residual[n,f,q,r]=distance; gaps[n,f,q,r]=gap
                    w=[o for o in history[tid] if lower<=o['f']<=f]
                    current.append((r,tid,w,sum(o['amp'] for o in w),sum(o['amp']*o['inner'] for o in w)))
                h=c['amplitude'][n,lower:f+1,q,0]
                values=dict(untracked_peakmax5=float(max(h)),untracked_top1sum5=float(sum(h)),
                            untracked_positive_frames5=int(sum(h>0)),current_top8sum=float(sum(c['amplitude'][n,f,q])),
                            candidate_count=len(candidates),window_available_frames=f-lower+1,left_boundary_warmup=int(f<4))
                if current:
                    r,tid,w,total,inside=max(current,key=lambda v:(v[3],-v[0])); first,last=w[0],w[-1]
                    ds=[o['d'] for o in w if np.isfinite(o['d'])]
                    extent=c['support_high_xyz'][n,f,q,r]-c['support_low_xyz'][n,f,q,r]
                    values.update(tracked_sum=total,tracked_mean=total/len(w),tracked_max=max(o['amp'] for o in w),
                        hits=len(w),span=f-first['f']+1,max_gap=max([b['f']-a['f']-1 for a,b in zip(w,w[1:])] or [0]),
                        last_gap=last['gap'],matched_distance_residual=sum(ds)/len(ds) if ds else np.nan,
                        last_distance_residual=last['d'],depth_change=float(first['point'][2]-last['point'][2]),
                        compensated_depth_change=float(move(first['point'],motion[f+3,first['f']+3])[2]-last['point'][2]),
                        inner_share=inside/total,inner_weighted_sum=inside,current_peak=last['amp'],current_rank=r+1,
                        current_membership=float(c['membership'][n,f,q,r]),current_depth_m=float(last['point'][2]),
                        current_x_m=float(last['point'][0]),current_y_m=float(last['point'][1]),
                        support_x_extent_m=float(extent[0]),support_y_extent_m=float(extent[1]),support_z_extent_m=float(extent[2]))
                    ir,it,iw,isum,iinside=max(current,key=lambda v:(v[4],-v[0]))
                    values.update(inner_best_tracked_sum=iinside,inner_best_hits=len(iw),inner_best_span=iw[-1]['f']-iw[0]['f']+1,
                        inner_best_max_gap=max([b['f']-a['f']-1 for a,b in zip(iw,iw[1:])] or [0]),
                        inner_best_share=iinside/isum,inner_best_current_rank=ir+1)
                    selected[n,f,q]=tid
                    for o in w:
                        offset=o['f']-lower; paths[n,f,q,offset]=move(o['point'],motion[f+3,o['f']+3])
                        native[n,f,q,offset]=o['native']; amplitudes[n,f,q,offset]=o['amp']
                features[n,f,q]=[values.get(name,np.nan) for name in names]
    return dict(features=features,valid=np.isfinite(features),track_id=ids,match_residual=residual,
                previous_missing_frames=gaps,best_track_id=selected,best_track_path_xyz=paths,
                best_track_path_native_index=native,best_track_path_amplitude=amplitudes)


def ray_nodes():
    edge=np.float32(np.tan(np.pi/8)); width=np.float32(8*.0375348)
    result=[]
    for y in range(8):
        for x in range(8):
            for b in range(16):
                nodes=[]
                for oy in (1/6,.5,5/6):
                    for ox in (1/6,.5,5/6):
                        direction=np.array([-edge+np.float32(x+ox)*(2*edge/8),-edge+np.float32(y+oy)*(2*edge/8),1],np.float32)
                        nodes.append(direction/np.linalg.norm(direction)*np.float32((b+.5)*width))
                result.append(nodes)
    return np.asarray(result,np.float32)


def check_equal(actual,expected,stats):
    for key, value in expected.items():
        np.testing.assert_array_equal(np.isfinite(actual[key]),np.isfinite(value))
        if value.dtype.kind=='f':
            np.testing.assert_allclose(actual[key],value,atol=4e-6,rtol=4e-6,equal_nan=True)
            good=np.isfinite(value)
            if good.any(): stats['max_absolute_difference']=max(stats['max_absolute_difference'],float(abs(actual[key][good]-value[good]).max()))
        else:
            np.testing.assert_array_equal(actual[key],value)
        stats['scalar_checks']+=value.size


def toy(kind):
    shape=(1,13,2,8)
    c=dict(valid=np.zeros(shape,bool),amplitude=np.zeros(shape,np.float32),xyz=np.full((*shape,3),np.nan,np.float32),
           support_low_xyz=np.full((*shape,3),np.nan,np.float32),support_high_xyz=np.full((*shape,3),np.nan,np.float32),
           native_index=np.full(shape,-1,np.int16),membership=np.zeros(shape,np.float32),inner_share=np.zeros(shape,np.float32))
    motion=np.broadcast_to(np.eye(4,dtype=np.float32),(16,16,4,4)).copy()
    frames=range(13) if kind!='intermittent' else (0,3,7)
    for f in frames:
        count=2 if kind=='competition' else 1
        for q in range(2):
            for r in range(count):
                point=np.array([r*.3 if f==0 or kind!='competition' else .12+r*.03,q*.5,1],np.float32)
                if kind=='public_motion': point[2]+=f*.08
                c['valid'][0,f,q,r]=True; c['amplitude'][0,f,q,r]=.3 if f%2 else 1
                c['xyz'][0,f,q,r]=point; c['support_low_xyz'][0,f,q,r]=point-.01
                c['support_high_xyz'][0,f,q,r]=point+.01; c['native_index'][0,f,q,r]=r
                c['membership'][0,f,q,r]=1; c['inner_share'][0,f,q,r]=1-r*.5
    if kind=='public_motion':
        for current in range(16):
            for old in range(16): motion[current,old,2,3]=(current-old)*.08
    return c,dict(motion=motion)


def audit(output=OUT):
    began=time.monotonic(); destination=output/'audit'; destination.mkdir(parents=True,exist_ok=True)
    stats=dict(status='PASS',scalar_checks=0,max_absolute_difference=0.,representative_observations=6,query_frames=156,
               CPU_seconds_cap=120,GPU_seconds=0,toys=[],prefix=[])
    # Import only to compare production code with independent scalar reference.
    import cnh_graded_peak_tracks_dev as P
    nodes=ray_nodes(); bias=np.load(BIAS).astype(np.float32)
    for split in ('cal','validation'):
        d=SOURCE/split
        with np.load(d/'geometry.npz') as a: sensor=a['sensor']; query=a['public_query']
        with np.load(d/'physics.npz') as a: ambient=a['ambient'].astype(np.float32)
        with np.load(output/f'{split}_tracks.npz') as a: saved={k:a[k] for k in a.files}
        with np.load(output/f'{split}_public_geometry.npz') as a: geometry={k:a[k] for k in a.files}
        motion=np.empty((16,16,4,4),sensor.dtype)
        for f in range(16):
            for old in range(16):
                motion[f,old]=(query[f]@np.linalg.inv(sensor[f]))@(sensor[old]@np.linalg.inv(query[old]))
        np.testing.assert_allclose(geometry['motion'],motion,atol=3e-7,rtol=3e-7)
        hist=np.load(d/'hist.npy',mmap_mode='r'); histories=np.load(d/'histories.npy',mmap_mode='r')
        for i,k in ((0,0),(127,1),(383,3)):
            raw=np.array(hist[i,k],copy=True).astype(np.float32)
            z=((raw-bias)/np.sqrt(np.maximum(16*ambient[...,None]+np.maximum(bias,0),1e-9))).astype(np.float16)
            np.testing.assert_array_equal(z[8:16],histories[(i*4+k)*13+12])
            logs=np.sign(z.astype(np.float32))*np.log1p(abs(z.astype(np.float32)))
            for fj,f in enumerate(range(3,16)):
                # public_query maps current native sensor coordinates into query
                # coordinates. World sensor pose cancels for a current frame.
                matrix=query[f].astype(np.float32)
                positions=nodes@matrix[:3,:3].T+matrix[:3,3]
                np.testing.assert_allclose(geometry['node_xyz'][fj],positions,atol=5e-7,rtol=3e-7)
                for q,(lo,hi) in enumerate(((-.2,.42),(.42,.9))):
                    mask=(abs(positions[...,0])<=.4)&(positions[...,1]>=lo)&(positions[...,1]<=hi)&(positions[...,2]>=.3)&(positions[...,2]<=3)
                    inner=mask&(abs(positions[...,0])<=.3)
                    np.testing.assert_array_equal(geometry['expanded_nodes'][fj,q],mask)
                    np.testing.assert_array_equal(geometry['inner_nodes'][fj,q],inner)
                    weights=mask.mean(-1,dtype=np.float32); positive=np.maximum(logs[f].reshape(1024)*weights,0)
                    peaks=[]
                    for zone in range(64):
                        for b in range(16):
                            ix=zone*16+b
                            if positive[ix]>0 and positive[ix]>(positive[ix-1] if b else -np.inf) and positive[ix]>=(positive[ix+1] if b<15 else -np.inf):
                                peaks.append((float(positive[ix]),ix))
                    best=sorted(peaks,key=lambda v:(-v[0],v[1]))[:8]
                    for rank,(amp,index) in enumerate(best):
                        assert saved['candidate_valid'][i,k,fj,q,rank]
                        assert saved['candidate_native_index'][i,k,fj,q,rank]==index
                        assert saved['candidate_native_zone'][i,k,fj,q,rank]==index//16
                        assert saved['candidate_native_bin'][i,k,fj,q,rank]==index%16
                        np.testing.assert_allclose(saved['candidate_amplitude'][i,k,fj,q,rank],amp,atol=2e-6,rtol=2e-6)
                        np.testing.assert_allclose(saved['candidate_xyz'][i,k,fj,q,rank],positions[index,4],atol=4e-7,rtol=4e-7)
                        np.testing.assert_allclose(saved['candidate_support_low_xyz'][i,k,fj,q,rank],positions[index,mask[index]].min(0),atol=4e-7,rtol=4e-7)
                        np.testing.assert_allclose(saved['candidate_support_high_xyz'][i,k,fj,q,rank],positions[index,mask[index]].max(0),atol=4e-7,rtol=4e-7)
                        np.testing.assert_allclose(saved['candidate_membership'][i,k,fj,q,rank],weights[index],atol=1e-7)
                        assert saved['candidate_inner_share'][i,k,fj,q,rank]==np.float32(inner[index].sum()/mask[index].sum())
                        stats['scalar_checks']+=17
                    assert int(saved['candidate_valid'][i,k,fj,q].sum())==len(best)
            candidates={key.removeprefix('candidate_'):value[i,k][None] for key,value in saved.items() if key.startswith('candidate_')}
            expected=reference(candidates,motion,P.NAMES)
            check_equal({key:saved[key][i,k][None] for key in expected},expected,stats)
            # Mutate future hist observations, not merely postcomputed features.
            cutoff=7; altered=z[None].copy(); altered[:,cutoff+1:]=np.float16(-13)
            future=P.extract_tracks(P.current_candidates(altered,geometry),geometry)
            for key in expected:
                np.testing.assert_array_equal(future[key][:,:cutoff-2],saved[key][i,k][None,:cutoff-2])
            stats['prefix'].append(dict(split=split,scene_index=i,replica=k,through_frame=cutoff,exact=True))
        # Whole payload checks: no duplicate track assignment, gap cap and no past8 double counting.
        ids=saved['track_id']; valid=saved['candidate_valid']
        sorted_ids=np.sort(np.where(valid,ids,-1),axis=-1)
        assert not np.any((sorted_ids[...,:-1]==sorted_ids[...,1:])&(sorted_ids[...,:-1]>=0))
        assert np.all(saved['previous_missing_frames'][valid]<=2)
        assert np.all(saved['match_residual'][np.isfinite(saved['match_residual'])]<=.450001)
        hit=saved['features'][...,P.NAMES.index('hits')]
        np.testing.assert_array_equal(hit,(saved['best_track_path_native_index']>=0).sum(-1))
        np.testing.assert_allclose(saved['features'][...,P.NAMES.index('tracked_sum')],saved['best_track_path_amplitude'].sum(-1),atol=3e-6,rtol=3e-6)
        assert np.all(hit<=np.minimum(np.arange(1,14),5)[None,None,:,None])
        np.testing.assert_array_equal(saved['valid'],np.isfinite(saved['features']))
        stats['scalar_checks']+=valid.size+hit.size*3
        if time.monotonic()-began>=120: raise TimeoutError('Audit 120s CPU wall cap reached')
    for kind in ('static','intermittent','competition','public_motion'):
        c,g=toy(kind); actual=P.extract_tracks(c,g); expected=reference(c,g['motion'],P.NAMES)
        check_equal(actual,expected,stats)
        if kind=='intermittent':
            assert actual['track_id'][0,0,0,0]==actual['track_id'][0,3,0,0]
            assert actual['track_id'][0,7,0,0]!=actual['track_id'][0,3,0,0]
            assert actual['features'][0,3,0,P.NAMES.index('hits')]==2
            assert actual['features'][0,3,0,P.NAMES.index('max_gap')]==2
            assert actual['features'][0,7,0,P.NAMES.index('hits')]==1
        if kind=='public_motion':
            np.testing.assert_allclose(actual['features'][...,P.NAMES.index('compensated_depth_change')],0,atol=3e-7)
        stats['toys'].append(dict(kind=kind,reference_exact_or_tolerance=True))
        np.savez_compressed(destination/f'toy_{kind}.npz',**actual,**{f'candidate_{key}':value for key,value in c.items()},motion=g['motion'])
    stats.update(CPU_seconds=time.monotonic()-began,audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tracker_source_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),command=[sys.executable,*sys.argv],
        evidence_scope='Representative raw normalization/native peaks/XYZ bounds and independent all track-feature state machine; whole payload one-to-one/gap/window-count/sum invariants; future raw mutation prefix and static/intermittent/competition/motion toys.',
        limitation='Association correctness establishes declared observed-peak mechanics only, not physical object identity, obstacle existence, coverage or free space.')
    (destination/'result.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf8')
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<22),b''): h.update(block)
        return h.hexdigest()
    inputs=[output/f'{split}_{kind}.npz' for split in ('cal','validation') for kind in ('tracks','public_geometry')]
    receipt=dict(audit_source_sha256=stats['audit_source_sha256'],tracker_source_sha256=stats['tracker_source_sha256'],
        command=stats['command'],GPU_seconds=0,CPU_seconds=time.monotonic()-began,
        input_sha256={str(p.relative_to(output)):sha(p) for p in inputs},
        output_sha256={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='execution_receipt.json'})
    (destination/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(stats))


if __name__=='__main__':
    audit()
