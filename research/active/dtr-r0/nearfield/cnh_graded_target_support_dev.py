"""Evaluator-only target support in parallel ToF readout branches.

Frozen noisy observations/scores remain unchanged. Expected target emission is
the difference between two opaque, geometrically identical rho endpoints, not
between a target-present and target-absent world. No learned forward pass.
"""
from collections import defaultdict
from pathlib import Path
import csv
import json
import os
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
SOURCE = WORK/'cnh-counterfactual-dev-20261009'
PRIOR_ATTEMPT = WORK/'cnh-graded-target-support-dev-20261010'
OUT = PRIOR_ATTEMPT/'repair3'
TRACK = WORK/'cnh-graded-peak-track-dev-20261010/tracks'
PREVIOUS = WORK/'cnh-graded-remaining-detection-dev-20261010'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<23), b''): h.update(block)
    return h.hexdigest()


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def ray_target(box, direction, origins, background):
    """Same slab/tie rule as the frozen opaque target renderer."""
    parallel = abs(direction) < 1e-14
    inverse = np.divide(1., direction, out=np.zeros_like(direction), where=~parallel)
    x = (np.asarray(box['lo'])-origins)*inverse
    y = (np.asarray(box['hi'])-origins)*inverse
    near, far = np.where(parallel, -np.inf, np.minimum(x,y)), np.where(parallel, np.inf, np.maximum(x,y))
    enter, leave = near.max(-1), far.min(-1)
    distance = np.where(enter > 1e-10, enter, leave)
    visible = (leave >= np.maximum(enter,0)) & (distance > 1e-10)
    visible &= ~(parallel & ((origins < box['lo']) | (origins > box['hi']))).any(-1)
    visible &= distance <= background
    return distance, visible


def projection_weights(sensor, query, geom):
    """Query-integrated linear sensitivity of the all-bin M3 projection.

    FP32 indexing/weights match the original implementation; no voxel model,
    FP16 features or signed-log nonlinear activation is reconstructed here.
    """
    import torch
    import cnh_cvr_projection as P
    _, points = P.grid()
    points = torch.as_tensor(points.reshape(-1,3), dtype=torch.float32, device='cuda')
    volumes = torch.as_tensor(P.cell_volumes().reshape(-1), dtype=torch.float32, device='cuda')
    masks = torch.as_tensor(P.query_masks().reshape(2,-1), dtype=torch.float32, device='cuda')
    mask_nodes = masks.repeat_interleave(P.SUB**3, dim=1)
    weights = np.zeros((13,2,16,1024), np.float32)
    center_soft = np.zeros_like(weights)
    centers = geom['node_xyz'][0,:,4]  # original bin centers in query at f3
    # Recover sensor centers independently of the query translation/rotation.
    centers = (centers-query[3,:3,3]) @ query[3,:3,:3]
    boxes = np.array([[-.3,-.2,.3,.3,.42,3],[-.3,.42,.3,.3,.9,3]])
    for fj,f in enumerate(range(3,16)):
        for h in range(max(0,f-7),f+1):
            transform = query[f] @ np.linalg.inv(sensor[f]) @ sensor[h]
            t = torch.as_tensor(transform, dtype=torch.float32, device='cuda')
            xyz = (points-t[:3,3]) @ t[:3,:3]
            radius = torch.linalg.vector_norm(xyz, dim=1)
            ij = torch.floor((xyz[:,:2]/xyz[:,2:3].clamp_min(1e-30)+P.EDGE)/(2*P.EDGE)*8).long()
            bins = torch.floor(radius/P.WIDTH).long()
            valid = (xyz[:,2] > 0) & (ij >= 0).all(1) & (ij < 8).all(1) & (bins >= 0) & (bins < 16)
            index = (ij[:,1].clamp(0,7)*8+ij[:,0].clamp(0,7))*16+bins.clamp(0,15)
            mass = valid*(float(np.prod(P.STEP))/(P.SUB**3))/volumes[index]
            for q in range(2):
                result = torch.zeros(1024, dtype=torch.float32, device='cuda')
                result.scatter_add_(0,index,mass*mask_nodes[q])
                weights[fj,q,h] = result.cpu().numpy()
            cxyz = centers @ transform[:3,:3].T + transform[:3,3]
            for q,box in enumerate(boxes):
                outside = np.maximum(box[:3]-cxyz,0)+np.maximum(cxyz-box[3:],0)
                center_soft[fj,q,h] = np.exp(-outside.sum(-1)/.05)
    torch.cuda.synchronize()
    return weights, center_soft


def describe(a):
    a = np.asarray(a,float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), min=float(a.min()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def run():
    started = time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve prior diagnostic attempt')
    OUT.mkdir(parents=True,exist_ok=True)
    inherited = read(PREVIOUS/'PLAN.json')['inputs_sha256']
    paths = [SOURCE/'scene_rows.json', SOURCE/'data_source_manifest.json', SOURCE/'data_render_receipt.json',
             PREVIOUS/'events.csv', Path(__file__),
             Path(__file__).with_name('cnh_cvr_projection.py'),
             Path(__file__).with_name('cnh_graded_corridor_features_dev.py')]
    for split in ('cal','validation'):
        paths += [SOURCE/'data'/split/'geometry.npz', SOURCE/'data'/split/'hist.npy',
                  SOURCE/'data'/split/'physics.npz', TRACK/f'{split}_tracks.npz', TRACK/f'{split}_public_geometry.npz']
    inputs = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(OUT/'PLAN.json',dict(task='CNH_GRADED_TARGET_SUPPORT_DEV_20261010',base_commit='1311e202',
        authorization='User 推进 target-native-bin/readout retention diagnosis and fixed raw-burst comparison',
        lane='EXPLORE consumed ideal simulated Development',
        goal='Distinguish visible target/native evidence, gate/local-maximum/top8 retention and parallel all-bin projection/center-token history support',
        budget=dict(target_CPU_main_command_wall_seconds=480,target_GPU_stage_wall_seconds=180,
                    target_verification_CPU_command_wall_seconds=180,integration_CPU_command_wall_seconds=180,
                    independent_raw_comparison_CPU_command_wall_seconds=300,contract_review_CPU_command_wall_seconds=60,
                    total_CPU_command_wall_seconds=1200),
        baseline='Frozen baseline/both/head50 all3seeds both splits, contact height, f3..13timely f14..15late',
        topology='All native bins feed M3/local finite-volume projection AND ordinary center-token readout; a separate gated local-max top8 feature bank feeds joint addition. No serial top8->M3 claim.',
        expected_target='Identical opaque target geometry at rho=.65 vs rho=0 endpoints, difference times actualrho/.65; same blocked background/crosstalk cancel. No photon subtraction, noise resampling or target-absent difference.',
        direct_support='Sub16 renderer first-visible target rays at raw range bin floor((distance-rangezero)/.0375348), coarsebin raw//8, originalzone; ignore tiny pulse tails as identity',
        fractions='Continuous expected target emission energy retained by public expanded gate, observed positive score, radial localmaximum and actual cached top8. Total emission includes target surfaces outside query; report physical inner hit fraction separately.',
        projection='Query-mask integrated original FP32 finite-volume linear sensitivity applied to target emission/normalization and actual observations; past8 geometry reexpressed in current query; not actual M3 signedlog features or causal network attribution',
        token='Actual ordinary branch keeps all bins; center softmembership and gated signedlog history mean are descriptor diagnostics, no model forward',
        adjustable_scope='Implementation/schema repair, scoped backend/geometry checks, descriptive comparisons only',
        stop='Complete fixed all-contact diagnostics and declared burst comparisons within budgets; no new model/threshold/data/pitch/hardware/notification scan',
        decision_check='Per seed/policy/split height384contacts and family96;1event resolution. All outcomes/weak and highrho controls retained; no target-posterior training or success gate',
        backend='Existing frozen FP64 CUDA renderer; GPU suitable projection maps, CPU small joins and descriptive stats',
        inputs_sha256=inputs, inherited_identity='Inherited scores/grades remain as previous PLAN; directly used paths pinned separately',
        new_noise_draws=0,fit=0,model_forward=0,Android_changes=0,
        repair_of=PRIOR_ATTEMPT.relative_to(ROOT).as_posix(),
        repair_reason='Archived renderer scripts use repository-relative paths and cannot be imported after relocation. Verify original bindings before using original locations; no scientific criteria change.',
        prior_failed_CPU_command_wall_seconds=50,
        prior_failed_GPU_stage_wall_seconds_upper_bound=47,
        zero_emission_repair='No-target-emission frames keep fractions null; check fraction ordering only when total>0. Direct-support counts still include zero frames.',
        DLL_repair='Register installed nvidia/cublas/bin for CuPy CUDA12, as original data producer does; Torch uses CUDA13. Preserve repair1 failure and close handle.'))
    env_before = {k:os.environ.get(k) for k in ('CUPY_CACHE_DIR','TEMP','TMP')}
    engines = []
    dll_handle = None
    gpu_started = None
    try:
        for k in env_before:
            p = OUT/'runtime'/k.lower(); p.mkdir(parents=True,exist_ok=True); os.environ[k] = str(p)
        # The scripts use repository-relative paths: verify originals before import.
        # The geometry dependency remains the original frozen v5 artifact source.
        manifest = read(SOURCE/'data_source_manifest.json')
        for original,binding in manifest.items():
            if sha(SOURCE/binding['snapshot']) != binding['sha256']: raise ValueError('Frozen renderer snapshot drift')
            if sha(Path(original)) != binding['sha256']: raise ValueError('Original frozen renderer binding drift')
        import cnh_location_reference_gpu as R
        import cnh_graded_corridor_features_dev as C
        import cnh_graded_evidence_dev as G
        import torch
        cuda_dll = Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
        dll_handle = os.add_dll_directory(str(cuda_dll))
        gpu_started = time.monotonic()
        def check():
            if time.monotonic()-gpu_started+47 >= 180: raise TimeoutError('Cumulative target GPU stage180s cap')
            if time.monotonic()-started+50 >= 480: raise TimeoutError('Cumulative target main480s cap')
        authors = read(SOURCE/'scene_rows.json')
        all_events = list(csv.DictReader((PREVIOUS/'events.csv').open(encoding='utf8')))
        summaries, frame_records, parity, render_metadata = {}, [], [], []
        projection = None
        for split in ('cal','validation'):
            directory = SOURCE/'data'/split
            with np.load(directory/'geometry.npz',allow_pickle=False) as a:
                sensor,query,category,ids = (a[k] for k in ('sensor','public_query','category','scene_ids'))
            with np.load(directory/'physics.npz',allow_pickle=False) as a: ambient = a['ambient']
            with np.load(TRACK/f'{split}_public_geometry.npz',allow_pickle=False) as a:
                geom = {k:a[k] for k in ('membership','node_xyz')}
                np.testing.assert_array_equal(a['sensor'],sensor)
                np.testing.assert_array_equal(a['public_query'],query)
            if projection is None:
                projection,soft = projection_weights(sensor,query,geom)
                np.savez_compressed(OUT/'public_weights.npz',projection=projection,center_soft=soft,sensor=sensor,public_query=query)
            else:
                np.testing.assert_array_equal(sensor,old_sensor); np.testing.assert_array_equal(query,old_query)
            old_sensor,old_query = sensor,query
            selected = np.flatnonzero((category == 'contact').any(-1))
            expected_target = np.zeros((384,16,1024),np.float64)
            expected_present = np.zeros_like(expected_target)
            direct = np.zeros((384,16,1024),bool)
            visible_counts = np.zeros((384,16),np.int32)
            inner_ray_fraction = np.zeros((384,13,2),float)
            bgids = sorted({authors[split][n]['background_id'] for n in selected})
            for bgid in bgids:
                check()
                positions = [n for n in selected if authors[split][n]['background_id'] == bgid]
                first = authors[split][positions[0]]
                engine = R.ExpectedRenderer(sensor,first['background_boxes'])
                engines.append(engine)
                render_metadata.append(dict(split=split,background_id=int(bgid),**engine.metadata))
                direction = engine.cp.asnumpy(engine.directions).reshape(16,-1,3)
                background = engine.cp.asnumpy(engine.distance).reshape(16,-1)
                origins = np.broadcast_to(sensor[:,:3,3][:,None],direction.shape)
                rays_per_zone = 256
                zone = np.tile(np.repeat(np.arange(64),rays_per_zone),16).reshape(16,-1)
                frame = np.broadcast_to(np.arange(16)[:,None],zone.shape)
                for offset,endpoints in engine.iter_render([authors[split][n]['target_box'] for n in positions],candidate_batch=8,deadline_check=check):
                    for j,(low,high) in enumerate(endpoints):
                        n = positions[offset+j]; author = authors[split][n]
                        difference = high-low
                        if difference.min() < -1e-10: raise ValueError('Opaque rho endpoint target emission became negative')
                        expected_target[n] = np.maximum(difference,0).reshape(16,1024)*author['rho']/R.ENDPOINT_RHO
                        expected_present[n] = (low+author['rho']/R.ENDPOINT_RHO*difference).reshape(16,1024)
                        distance, visible = ray_target(author['target_box'],direction,origins,background)
                        raw_bin = np.floor(np.where(visible,distance,0)/R.SENSOR.RAW_BIN_M).astype(int)
                        visible &= (raw_bin >= 0) & (raw_bin < 128)
                        cell = (frame*1024+zone*16+raw_bin//8)
                        direct[n].reshape(-1)[cell[visible]] = True
                        visible_counts[n] = visible.sum(-1)
                        xyz = origins+direction*np.where(visible,distance,0)[...,None]
                        for fj,f in enumerate(range(3,16)):
                            qxyz = xyz[f] @ (query[f] @ np.linalg.inv(sensor[f]))[:3,:3].T + (query[f] @ np.linalg.inv(sensor[f]))[:3,3]
                            for q,(yl,yh) in enumerate(((-.2,.42),(.42,.9))):
                                inside = (abs(qxyz[:,0]) <= .3) & (qxyz[:,1] >= yl) & (qxyz[:,1] <= yh) & (qxyz[:,2] >= .3) & (qxyz[:,2] <= 3)
                                inner_ray_fraction[n,fj,q] = float((visible[f]&inside).sum()/max(visible[f].sum(),1))
                        if len(parity) < (1 if split == 'cal' else 2):
                            frames = [3,13]
                            scene = dict(poses=sensor[frames],boxes=author['boxes'])
                            ref = R.R.expected(scene)['expectation']
                            np.testing.assert_allclose(expected_present[n,frames].reshape(2,8,8,16),ref,rtol=1e-10,atol=1e-9)
                            hit = R.S.raycast_boxes(sensor[13,:3,3],direction[13],author['boxes'])
                            np.testing.assert_array_equal(visible[13], (hit['object_id'] == 0) & (hit['distance'] < 128*R.SENSOR.RAW_BIN_M))
                            parity.append(dict(split=split,scene=int(n),frames=frames,max_expected_abs=float(abs(expected_present[n,frames].reshape(2,8,8,16)-ref).max()),ray_exact=True))
                engine.close(); engines.remove(engine)
            check()
            bias = np.load(C.BIAS).astype(np.float32)
            den = np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9)).reshape(16,1024)
            hist = np.load(directory/'hist.npy',mmap_mode='r',allow_pickle=False)
            z = C.normalize(np.array(hist[selected],copy=True),ambient,bias).astype(np.float32).reshape(len(selected),4,16,1024)
            logs = np.sign(z)*np.log1p(abs(z))
            with np.load(TRACK/f'{split}_tracks.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'],ids)
                cand_index,cand_valid = a['candidate_native_index'],a['candidate_valid']
                cand_amp = a['candidate_amplitude']
            np.savez_compressed(OUT/f'{split}_target_geometry.npz',scene_ids=ids,selected_scene_ids=selected,
                expected_target=expected_target,expected_present=expected_present,direct=direct,visible_counts=visible_counts,inner_ray_fraction=inner_ray_fraction)
            metrics = {}
            for local,n in enumerate(selected):
                q = int(np.flatnonzero(category[n] == 'contact')[0])
                for k in range(4):
                    for fj,f in enumerate(range(3,16)):
                        check()
                        t = expected_target[n,f]; total = t.sum()
                        m = geom['membership'][fj,q]
                        score = np.maximum(logs[local,k,f]*m,0).reshape(64,16)
                        peak = (score > 0) & (score > np.c_[-np.inf*np.ones(64),score[:,:-1]]) & (score >= np.c_[score[:,1:],-np.inf*np.ones(64)])
                        rank = np.argsort(-np.where(peak,score,-np.inf).ravel(),kind='stable')[:8]
                        ranked_valid = peak.ravel()[rank]
                        np.testing.assert_array_equal(cand_valid[n,k,fj,q],ranked_valid)
                        np.testing.assert_array_equal(cand_index[n,k,fj,q][ranked_valid],rank[ranked_valid])
                        np.testing.assert_allclose(cand_amp[n,k,fj,q][ranked_valid],score.ravel()[rank[ranked_valid]],rtol=2e-6,atol=1e-6)
                        top = np.zeros(1024,bool); top[rank[ranked_valid]] = True
                        gate = m > 0; obs_positive = z[local,k,f] > 0
                        begin = max(0,f-7); norm_t = expected_target[n,begin:f+1]/den[begin:f+1]
                        pw = projection[fj,q,begin:f+1]
                        sm = soft[fj,q,begin:f+1]
                        entry = dict(split=split,scene=int(n),replica=k,height=('HEAD','BODY')[q],frame=f,
                            shape_family=authors[split][n]['shape_family'],rho=authors[split][n]['rho'],placement=authors[split][n]['placement'],
                            target_visible_rays=int(visible_counts[n,f]),physical_inner_ray_fraction=float(inner_ray_fraction[n,fj,q]),direct_bins=int(direct[n,f].sum()),
                            direct_positive_bins=int((direct[n,f]&obs_positive).sum()),direct_gated_bins=int((direct[n,f]&gate).sum()),
                            direct_gated_positive_bins=int((direct[n,f]&gate&obs_positive).sum()),direct_local_peak_bins=int((direct[n,f]&peak.ravel()).sum()),
                            direct_top8_bins=int((direct[n,f]&top).sum()),target_expected_counts=float(total),
                            gate_energy_fraction=float((t*m).sum()/total) if total > 0 else None,
                            positive_gate_energy_fraction=float((t*m*obs_positive).sum()/total) if total > 0 else None,
                            local_peak_energy_fraction=float((t*m*peak.ravel()).sum()/total) if total > 0 else None,
                            top8_energy_fraction=float((t*m*top).sum()/total) if total > 0 else None,
                            top8_of_local_peak_energy_fraction=float((t*m*top).sum()/(t*m*peak.ravel()).sum()) if (t*m*peak.ravel()).sum() > 0 else None,
                            center_soft_current_energy_fraction=float((t*soft[fj,q,f]).sum()/total) if total > 0 else None,
                            center_soft_past8_energy_fraction=float((expected_target[n,begin:f+1]*sm).sum()/expected_target[n,begin:f+1].sum()) if expected_target[n,begin:f+1].sum() > 0 else None,
                            projected_target_current_mass=float((norm_t[-1]*pw[-1]).sum()),projected_target_past8_mass=float((norm_t*pw).sum()),
                            projected_target_current_fraction=float((norm_t[-1]*pw[-1]).sum()/norm_t[-1].sum()) if norm_t[-1].sum() > 0 else None,
                            projected_target_past8_fraction=float((norm_t*pw).sum()/norm_t.sum()) if norm_t.sum() > 0 else None,
                            observed_direct_signedlog_sum=float(logs[local,k,f][direct[n,f]].sum()),
                            observed_center_gated_past8_signedlog_mean=float((logs[local,k,begin:f+1]*sm).sum()/(f-begin+1)),
                            observed_projected_query_past8_signed_mass=float((z[local,k,begin:f+1]*pw).sum()))
                        assert entry['direct_top8_bins'] <= entry['direct_local_peak_bins'] <= entry['direct_gated_positive_bins'] <= entry['direct_positive_bins']
                        if total > 0:
                            assert entry['top8_energy_fraction'] <= entry['local_peak_energy_fraction']+1e-10 <= entry['positive_gate_energy_fraction']+2e-10 <= entry['gate_energy_fraction']+3e-10
                        frame_records.append(entry); metrics[n,k,f] = entry
            # Join frozen event outcomes AFTER all evaluator geometry/support has been measured.
            numeric = [name for name in frame_records[-1] if name not in ('split','scene','replica','height','frame','shape_family','rho','placement')]
            group = defaultdict(list)
            for e in all_events:
                if e['split'] != split: continue
                n,k = int(e['scene']),int(e['replica'])
                rows = [metrics[n,k,f] for f in range(3,14)]
                record = dict(e,**{f'{key}_timely_max':max((r[key] for r in rows if r[key] is not None),default=None) for key in numeric},
                    direct_positive_frames=sum(r['direct_positive_bins'] > 0 for r in rows),direct_top8_frames=sum(r['direct_top8_bins'] > 0 for r in rows))
                key = f"{split}/{e['seed']}/{e['policy']}/{e['height']}"
                for suffix in ('all',f"family:{e['shape_family']}",f"family_rho:{e['shape_family']}:{e['rho']}",f"placement:{e['placement']}"):
                    for outcome in ('all',e['outcome']): group[f'{key}/{suffix}/{outcome}'].append(record)
            for key,rr in group.items():
                summaries[key] = dict(events=len(rr),no_direct_positive=sum(r['direct_positive_frames'] == 0 for r in rr),
                    no_direct_top8=sum(r['direct_top8_frames'] == 0 for r in rr),
                    fields={name:describe([r[name] for r in rr if r[name] is not None]) for name in [f'{n}_timely_max' for n in numeric]+['direct_positive_frames','direct_top8_frames']})
            print(f'{split} COMPLETE contact_scenes={len(selected)} frames={len(metrics)}',flush=True)
        G.write_csv(OUT/'frames.csv',frame_records)
        save(OUT/'summary.json',summaries)
        save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-started,GPU_stage_wall_seconds=time.monotonic()-gpu_started,
            backend='Frozen FP64 CuPy renderer and FP32 Torch projection maps; CPU readout statistics',render_metadata=render_metadata,
            frames=len(frame_records),summary_groups=len(summaries),renderer_ray_parity=parity,input_hashes=inputs,
            source_sha256=sha(Path(__file__)),new_noise_draws=0,fit=0,model_forward=0))
        print('COMPLETE',round(time.monotonic()-started,3),flush=True)
    except BaseException as error:
        save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started))
        raise
    finally:
        for engine in engines: engine.close()
        if dll_handle is not None: dll_handle.close()
        if gpu_started is not None:
            import torch
            torch.cuda.empty_cache()
        for k,value in env_before.items():
            if value is None: os.environ.pop(k,None)
            else: os.environ[k] = value
        save(OUT/f'execution_{time.time_ns()}.json',dict(seconds=time.monotonic()-started,source_sha256=sha(Path(__file__)),persistent_resources=0))


def timing_alignment():
    """Post-hoc timing descriptions only, not a runtime feature or policy."""
    started = time.monotonic()
    frames = defaultdict(list)
    for r in csv.DictReader((OUT/'frames.csv').open(encoding='utf8')):
        frames[r['split'],r['scene'],r['replica']].append(r)
    events = list(csv.DictReader((PREVIOUS/'events.csv').open(encoding='utf8')))
    output = {}
    for seed in ('2026100955','2026100956','2026100957'):
        for height in ('HEAD','BODY'):
            for family in ('horizontal','sign_edge'):
                for outcome in ('timely','late','silent'):
                    rows = [r for r in events if r['split']=='validation' and r['seed']==seed and r['policy']=='head50'
                            and r['height']==height and r['shape_family']==family and r['outcome']==outcome]
                    if not rows: continue
                    stats = {name:float(np.median([float(r[name]) for r in rows]))
                             for name in ('score_f3_margin','score_max_timely_margin')}
                    first,counts,gaps = [],[],[]
                    for r in rows:
                        seen = [int(x['frame']) for x in frames[r['split'],r['scene'],r['replica']]
                                if int(x['frame'])<=13 and int(x['direct_top8_bins'])>0]
                        first.append(min(seen)); counts.append(len(seen))
                        if int(r['first_any'])>=0: gaps.append(int(r['first_any'])-min(seen))
                    stats.update(events=len(rows),first_target_top8_median=float(np.median(first)),
                                 target_top8_frames_min=min(counts),target_top8_frames_median=float(np.median(counts)),
                                 first_target_top8_to_alert_median=float(np.median(gaps)) if gaps else None)
                    output[f'{seed}/{height}/{family}/{outcome}'] = stats
    save(OUT/'first_score_alignment.json',output)
    save(OUT/'timing_receipt.json',dict(seconds=time.monotonic()-started,source_sha256=sha(Path(__file__)),
        inputs_sha256={p.name:sha(p) for p in (OUT/'frames.csv',PREVIOUS/'events.csv')},posthoc=True,policy_changes=0))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
        timing_alignment()
