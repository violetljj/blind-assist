"""Evaluator-only support retention on consumed frozen-e2e HEAD horizontal bars.

The separate top8/joint and all-bin M3/local paths are examined in parallel.
No model forward, threshold selection, new photons, training or protected data.
"""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
FROZEN = WORK/'cnh-frozen-e2e-20261010'
MAPS = WORK/'cnh-bar-local-readout-dev-20261009'
MAP_GEOMETRY = WORK/'cnh-aligned-shapes-dev-20261008/geometry.npz'
BIAS = WORK/'cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<23), b''): h.update(block)
    return h.hexdigest()


def describe(values):
    v=np.asarray([x for x in values if x is not None], float)
    v=v[np.isfinite(v)]
    return dict(n=len(v),minimum=float(v.min()) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,
                maximum=float(v.max()) if len(v) else None)


def fraction(a,b):
    return float(a/b) if b>0 else None


def signedlog(x):
    return np.sign(x)*np.log1p(abs(x))


def write_csv(path, rows):
    with path.open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(oracle, out, cap):
    began=time.monotonic()
    out=out.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Canonical artifacts.local required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'chain_receipt.json').exists() or (out/'chain_frames.csv').exists():
        raise FileExistsError('Preserve existing chain outputs')
    def check():
        if time.monotonic()-began>=cap: raise TimeoutError('chain CPU wall cap reached')
    with np.load(oracle,allow_pickle=False) as a:
        ids=a['scene_ids']; target=a['expected_target'].reshape(len(ids),16,1024)
        present=a['expected_present'].reshape(len(ids),16,1024)
        direct=a['direct'].reshape(len(ids),16,1024).astype(bool)
        sensor=a['sensor']; query=a['public_query']; ambient=a['ambient']
    if target.min() < -1e-9: raise ValueError('Target emission must be nonnegative')
    dark=present-target
    with np.load(FROZEN/'data/hold/geometry.npz') as a:
        np.testing.assert_array_equal(sensor,a['sensor']); np.testing.assert_array_equal(query,a['public_query'])
        categories=a['category']
    with np.load(MAP_GEOMETRY) as a:
        np.testing.assert_array_equal(sensor,a['sensor']); np.testing.assert_array_equal(query,a['public_query'])
    map_plan=read(MAPS/'PLAN.json')
    map_parity=read(MAPS/'result_run.json')
    if map_parity['status']!='COMPLETE': raise ValueError('Original map run incomplete')
    bound_sources={}
    for name in ('cnh_cvr_projection.py','cnh_bar_projection_diagnostic_dev.py'):
        source=Path(__file__).with_name(name)
        logical=source.relative_to(ROOT).as_posix()
        digest=map_plan['source_sha256'][logical]
        if sha(source)!=digest: raise ValueError('Map configuration/source binding changed: '+name)
        bound_sources[logical]=digest
    local_source=Path(__file__).with_name('cnh_bar_local_readout_dev.py')
    local_snapshot=MAPS/'source_snapshot.py'
    local_logical=local_source.relative_to(ROOT).as_posix()
    if sha(local_snapshot)!=map_plan['source_sha256'][local_logical]:
        raise ValueError('Original local map source snapshot drift')
    def projection_ast(path):
        module=ast.parse(path.read_text(encoding='utf8'))
        return ast.dump(next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Projection'))
    if projection_ast(local_source)!=projection_ast(local_snapshot):
        raise ValueError('Projection implementation changed since map creation')
    bound_sources[local_logical]=dict(archived_sha256=sha(local_snapshot),current_sha256=sha(local_source),
        Projection_class_AST_equal=True,other_changes='Later scan finite-scale fix and commentary; map class identical')
    import cnh_graded_peak_tracks_dev as P
    import cnh_counterfactual_eval_dev as E
    from cnh_cvr_projection import query_masks
    geom=P.native_geometry(sensor,query)
    mask=query_masks()[0].ravel().astype(float)
    bias=np.load(BIAS).astype(np.float32)
    den=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9)).reshape(16,1024)
    if den.shape!=(16,1024): raise ValueError('Unexpected ambient geometry')
    # Match native normalization rounding; the sparse map is ideal arithmetic on
    # original FP32 geometry coefficients, followed by the retained voxel FP16.
    norm_present=((present-bias.reshape(1,1,1024))/den).astype(np.float16).astype(float)
    norm_dark=((dark-bias.reshape(1,1,1024))/den).astype(np.float16).astype(float)
    norm_target=target/den
    hist=np.load(FROZEN/'data/hold/hist.npy',mmap_mode='r')[ids].copy()
    z=((hist.astype(np.float32)-bias)/den.reshape(16,8,8,16)).astype(np.float16)
    candidates=P.current_candidates(z.reshape(-1,16,8,8,16),geom)
    with np.load(FROZEN/'data/hold/scores.npz') as a:
        m3_raw=a['m3_raw'][ids]; local_raw=a['local_raw'][ids]
    m3=E.smooth(m3_raw); local=E.smooth(local_raw)
    author=read(FROZEN/'scene_rows.json')['hold']
    selected=[author[int(n)] for n in ids]
    assert all(r['shape_family']=='horizontal' and r['group']==0 and r['placement']=='contact' for r in selected)
    frame_rows=[]; projections=[]; input_paths=[Path(oracle),MAP_GEOMETRY,BIAS,MAPS/'PLAN.json',MAPS/'result_run.json',local_snapshot,
        FROZEN/'data/hold/hist.npy',
        FROZEN/'data/hold/geometry.npz',FROZEN/'data/hold/scores.npz',FROZEN/'scene_rows.json']
    for fj,f in enumerate(range(3,16)):
        check(); start=max(0,f-7); length=f-start+1
        path=MAPS/f'map_f{f}.npz'; p=sparse.load_npz(path).tocsr(); input_paths.append(path)
        assert p.shape==(len(mask),length*1024)
        current_map=p[:,-1024:]
        # Native-bin integrated public HEAD sensitivity is a descriptor; it is
        # not a network attribution or an optimal detector.
        query_sensitivity=np.asarray(mask@p).reshape(length,1024)
        total_p=np.asarray(p@norm_present[:,start:f+1].reshape(len(ids),-1).T).T.astype(np.float16).astype(float)
        total_d=np.asarray(p@norm_dark[:,start:f+1].reshape(len(ids),-1).T).T.astype(np.float16).astype(float)
        current_p=np.asarray(current_map@norm_present[:,f].T).T.astype(np.float16).astype(float)
        current_d=np.asarray(current_map@norm_dark[:,f].T).T.astype(np.float16).astype(float)
        target_projected=np.asarray(p@norm_target[:,start:f+1].reshape(len(ids),-1).T).T
        linear_query=target_projected@mask
        log_query=(signedlog(total_p)-signedlog(total_d))@mask
        log_current=(signedlog(current_p)-signedlog(current_d))@mask
        for i,n in enumerate(ids):
            emission=target[i,f]; energy=emission.sum(); member=geom['membership'][fj,0]
            target_peak=int(np.argmax(np.where(direct[i,f],norm_target[i,f],-np.inf))) if direct[i,f].any() else None
            for k in range(hist.shape[1]):
                score=np.maximum(signedlog(z[i,k,f].astype(np.float32)).ravel()*member,0).reshape(64,16)
                peak=(score>0)&(score>np.c_[np.full(64,-np.inf),score[:,:-1]])&(score>=np.c_[score[:,1:],np.full(64,-np.inf)])
                top=np.zeros(1024,bool)
                ci=i*hist.shape[1]+k
                ix=candidates['native_index'][ci,fj,0]; valid=candidates['valid'][ci,fj,0]
                top[ix[valid]]=True
                positive=z[i,k,f].ravel()>0
                row=dict(scene=int(n),replica=k,frame=f,size_variant=selected[i]['size_variant'],rho=selected[i]['rho'],
                    background_id=selected[i]['background_id'],target_expected_counts=float(energy),
                    target_reference_z_direct_peak=float(norm_target[i,f,target_peak]) if target_peak is not None else None,
                    opaque0_reference_z_at_target_peak=float(norm_dark[i,f,target_peak]) if target_peak is not None else None,
                    present_reference_z_at_target_peak=float(norm_present[i,f,target_peak]) if target_peak is not None else None,
                    observed_reference_z_at_target_peak=float(z[i,k,f].ravel()[target_peak]) if target_peak is not None else None,
                    direct_bins=int(direct[i,f].sum()),direct_positive_bins=int((direct[i,f]&positive).sum()),
                    direct_gated_bins=int((direct[i,f]&(member>0)).sum()),direct_local_peak_bins=int((direct[i,f]&peak.ravel()).sum()),
                    direct_top8_bins=int((direct[i,f]&top).sum()),
                    gate_energy_fraction=fraction((emission*member).sum(),energy),
                    positive_gate_energy_fraction=fraction((emission*member*positive).sum(),energy),
                    local_peak_energy_fraction=fraction((emission*member*peak.ravel()).sum(),energy),
                    top8_energy_fraction=fraction((emission*member*top).sum(),energy),
                    projected_target_current_mass=float((norm_target[i,f]*query_sensitivity[-1]).sum()),
                    projected_target_past8_mass=float(linear_query[i]),
                    projected_target_current_fraction=fraction((norm_target[i,f]*query_sensitivity[-1]).sum(),norm_target[i,f].sum()),
                    projected_target_past8_fraction=fraction(linear_query[i],norm_target[i,start:f+1].sum()),
                    expected_voxel_signedlog_target_delta_tot=float(log_query[i]),
                    expected_voxel_signedlog_target_delta_current=float(log_current[i]),
                    signedlog_delta_per_linear_query_mass=fraction(log_query[i],linear_query[i]),
                    m3_raw=float(m3_raw[i,k,fj,0]),m3_smooth=float(m3[i,k,fj,0]),
                    m3_threshold_margin=float(m3[i,k,fj,0]-E.M3_THETA),
                    local_raw=float(local_raw[i,k,fj,0]),local_smooth=float(local[i,k,fj,0]),
                    local_threshold_margin=float(local[i,k,fj,0]-E.OLD_LOCAL))
                if energy>0:
                    assert row['top8_energy_fraction']<=row['local_peak_energy_fraction']+1e-9<=row['positive_gate_energy_fraction']+2e-9<=row['gate_energy_fraction']+3e-9
                frame_rows.append(row)
        # Accumulation is linear before the M3 signedlog; count is geometry-only
        # and thus its target-presence contrast is identically zero.
        for window in (1,2,4,8):
            begin=max(start,f-window+1); keep=slice(begin-start,length)
            mass=(norm_target[:,begin:f+1]*query_sensitivity[keep][None]).sum((1,2))
            for i,n in enumerate(ids):
                projections.append(dict(scene=int(n),frame=f,window_frames=f-begin+1,
                    requested_window=window,target_projected_query_mass=float(mass[i]),
                    target_native_mass=float(norm_target[i,begin:f+1].sum()),
                    projected_fraction=fraction(mass[i],norm_target[i,begin:f+1].sum()),
                    projected_mass_per_available_frame=float(mass[i]/(f-begin+1)),
                    target_count_channel_delta=0.))
    write_csv(out/'chain_frames.csv',frame_rows); write_csv(out/'chain_accumulation.csv',projections)
    event_rows=[]
    for i,n in enumerate(ids):
        for k in range(hist.shape[1]):
            rr=[r for r in frame_rows if r['scene']==n and r['replica']==k and r['frame']<=13]
            timely=bool((m3[i,k,:11,0]>=E.M3_THETA).any()); any_alarm=bool((m3[i,k,:,0]>=E.M3_THETA).any())
            seen=[r['frame'] for r in rr if r['direct_top8_bins']>0]
            event_rows.append(dict(scene=int(n),replica=k,size_variant=selected[i]['size_variant'],rho=selected[i]['rho'],
                outcome='timely' if timely else 'late' if any_alarm else 'silent',
                first_target_compatible_top8_frame=min(seen) if seen else None,target_compatible_top8_timely_frames=len(seen),
                max_timely_m3_margin=float(m3[i,k,:11,0].max()-E.M3_THETA),
                max_timely_local_margin=float(local[i,k,:11,0].max()-E.OLD_LOCAL),
                max_timely_top8_energy_fraction=max((r['top8_energy_fraction'] for r in rr if r['top8_energy_fraction'] is not None),default=None),
                max_timely_projected_target_current_mass=max(r['projected_target_current_mass'] for r in rr),
                max_timely_projected_target_past8_mass=max(r['projected_target_past8_mass'] for r in rr),
                max_timely_projected_target_past8_fraction=max((r['projected_target_past8_fraction'] for r in rr if r['projected_target_past8_fraction'] is not None),default=None),
                max_timely_expected_voxel_signedlog_delta_tot=max(r['expected_voxel_signedlog_target_delta_tot'] for r in rr)))
    write_csv(out/'chain_events.csv',event_rows)
    grouped=defaultdict(list)
    for r in event_rows:
        grouped[f"size{r['size_variant']}/all"].append(r)
        grouped[f"size{r['size_variant']}/{r['outcome']}"].append(r)
    summary={}
    fields=list(event_rows[0])[5:]
    for key,rr in grouped.items():
        summary[key]=dict(events=len(rr),unique_scenes=len({r['scene'] for r in rr}),
            no_target_compatible_top8=sum(r['target_compatible_top8_timely_frames']==0 for r in rr),
            metrics={name:describe([r[name] for r in rr]) for name in fields})
    check()
    receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,CPU_cap_seconds=cap,
        physical_scenes=len(ids),noise_events=len(event_rows),frame_records=len(frame_rows),summary=summary,
        topology='parallel: native all-bin->M3/local and native all-bin->ordinary; gated radial maxima top8->joint',
        thresholds=dict(M3_logit_smooth=E.M3_THETA,old5_raised_M3=E.OLD_RAISED,local_patch_reference_z=E.OLD_LOCAL),
        threshold_units='SNR cannot be compared with learned logits; local is projected-patch reference-normalized score, not native-bin SNR',
        legacy_detection_snr5_usage='Only derive_readout/older proxy peak detector; frozen current chain uses full bins and strictly-positive gated radial top8, no native SNR>=5 gate',
        limitations='Consumed ideal simulated hold, explanatory reuse; target-compatible bin may contain background/noise. Energy retention and expected sparse-map FP16/signedlog descriptors are not exact CUDA voxel reconstruction, network attribution, physical hardware limit or new held-out utility. Cached scores unchanged.',
        target_attribution='same opaque geometry rho endpoints; background occlusion cancels; count channel has zero contrast',
        map_configuration_source_sha256=bound_sources,
        map_original_parity=map_parity['projection_parity'],
        inputs_sha256={str(p):sha(p) for p in input_paths},source_sha256=sha(__file__),
        training=0,model_forward=0,new_noise=0,protected_access=0,persistent_resources=0)
    (out/'chain_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status='COMPLETE',seconds=receipt['seconds'],summary=summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-npz',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cap',type=float,default=120.)
    args=parser.parse_args()
    with threadpool_limits(limits=2): run(args.oracle_npz,args.out,args.cap)
