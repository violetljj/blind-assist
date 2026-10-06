"""Frozen-score CPU geometry diagnostic; no model execution or photon rendering."""
from __future__ import annotations
import argparse,csv,importlib.util,time
from pathlib import Path
import numpy as np
import cnh_tristate_dev as R
import cnh_tristate_dev_r2 as R2
import cnh_tristate_event_dev as E

OUT=R.WORK/'cnh-mode2-seam-dev-20261006'
GEOM=R.WORK/'cnh-track-a-v5-20260928/data/source/cnh_track_a_geometry.py'


def geometry_module():
    spec=importlib.util.spec_from_file_location('frozen_seam_geometry',GEOM)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def first_hit_visible(points,boxes,origin,target_id=0):
    """Opaque AABB first intersection; surface-centroid rays, no response synthesis."""
    direction=points-origin
    best=np.full(len(points),np.inf);owner=np.full(len(points),-1,int)
    for bid,box in enumerate(boxes):
        lo=np.asarray(box['lo'])-origin;hi=np.asarray(box['hi'])-origin
        parallel=np.abs(direction)<1e-14
        a=np.divide(lo,direction,out=np.full_like(direction,-np.inf),where=~parallel)
        b=np.divide(hi,direction,out=np.full_like(direction,np.inf),where=~parallel)
        lower=np.minimum(a,b);upper=np.maximum(a,b)
        outside=parallel&((lo>1e-10)|(hi< -1e-10))
        enter=lower.max(1);leave=upper.min(1);distance=np.where(enter>1e-9,enter,leave)
        hit=~outside.any(1)&(leave>=np.maximum(enter,1e-9))&(distance<best-1e-10)
        best[hit]=distance[hit];owner[hit]=bid
    return (owner==target_id)&(np.abs(best-1.)<=1e-7)


def visibility(points,weights,sensor):
    vals=[];seen=[];bearing=[];margin=[]
    for angle in (0,-15,15):
        pose=sensor@R.extrinsic(angle);local=(points-pose[:3,3])@pose[:3,:3]
        az=np.rad2deg(np.arctan2(local[:,0],local[:,2]));vertical=np.rad2deg(np.arctan2(np.abs(local[:,1]),local[:,2]))
        fov=(local[:,2]>0)&(np.abs(az)<=22.5+1e-10)&(vertical<=22.5+1e-10)&(np.linalg.norm(local,axis=1)<=16*8*.0375348+1e-10)
        seen.append(fov);bearing.append(float(np.average(az,weights=weights)))
        margin.append(float(np.average(22.5-np.abs(az),weights=weights)))
        vals.append(float(weights[fov].sum()/weights.sum()))
    union=float(weights[seen[1]|seen[2]].sum()/weights.sum())
    return dict(single_fraction=vals[0],L_fraction=vals[1],R_fraction=vals[2],union_fraction=union,
        max_branch_fraction=max(vals[1:]),split_fraction=union-max(vals[1:]),
        single_bearing_deg=bearing[0],L_bearing_deg=bearing[1],R_bearing_deg=bearing[2],
        L_horizontal_margin_deg=margin[1],R_horizontal_margin_deg=margin[2],
        best_horizontal_margin_deg=max(margin[1:]))


def target_geometry(boxes,travel,sensor,queries,G):
    world=G.box_mesh(boxes[0]['lo'],boxes[0]['hi'])
    local=(world-travel[:3,3])@travel[:3,:3];pieces=[]
    for q in queries:
        yl,yh=R.HEIGHTS[int(q)]
        pieces.append(G.clip_triangles(local,[-.30,yl,.3],[.30,yh,3.]))
    clipped=np.concatenate(pieces)
    if not len(clipped):return dict(status='NO_DESIGNATED_TARGET_SURFACE_IN_CONTACT_QUERIES')
    pp,ww=G.surface_quadrature(clipped,max_edge=.05)
    points=pp@travel[:3,:3].T+travel[:3,3]
    visible=first_hit_visible(points,boxes,sensor[:3,3])
    area=float(ww.sum());unoccluded=float(ww[visible].sum())
    if not visible.any():return dict(status='DESIGNATED_TARGET_OCCLUDED',area_m2=area,unoccluded_area_m2=0.)
    v=visibility(points[visible],ww[visible],sensor)
    center=np.average(points[visible],axis=0,weights=ww[visible])
    cent=(center-sensor[:3,3])@sensor[:3,:3]
    return dict(status='EVALUABLE',area_m2=area,unoccluded_area_m2=unoccluded,unoccluded_fraction=unoccluded/area,
        centroid_world=center.tolist(),centroid_head_azimuth_deg=float(np.rad2deg(np.arctan2(cent[0],cent[2]))),quadrature_points=len(pp),**v)


def full_target_geometry(boxes,travel,sensor,queries,G):
    """Target height slices without cropping its lateral/forward extent to query.

    Some exact-deadline contacts enter the corridor after the last causal frame.
    Retain the original query-clipped diagnostic as a distinct secondary view.
    """
    local=(G.box_mesh(boxes[0]['lo'],boxes[0]['hi'])-travel[:3,3])@travel[:3,:3]
    pieces=[]
    for q in queries:
        yl,yh=R.HEIGHTS[int(q)];pieces.append(G.clip_triangles(local,[-np.inf,yl,-np.inf],[np.inf,yh,np.inf]))
    clipped=np.concatenate(pieces)
    if not len(clipped):return dict(status='NO_TARGET_SURFACE_IN_CONTACT_HEIGHTS')
    pp,ww=G.surface_quadrature(clipped,max_edge=.05);points=pp@travel[:3,:3].T+travel[:3,3]
    clear=first_hit_visible(points,boxes,sensor[:3,3]);area=float(ww.sum())
    if not clear.any():return dict(status='DESIGNATED_TARGET_OCCLUDED',area_m2=area,unoccluded_area_m2=0.)
    points,ww=points[clear],ww[clear];v=visibility(points,ww,sensor)
    horizontal=[]
    for name,angle in zip(('single','L','R'),(0,-15,15)):
        pose=sensor@R.extrinsic(angle);loc=(points-pose[:3,3])@pose[:3,:3]
        az=np.rad2deg(np.arctan2(loc[:,0],loc[:,2]));h=(loc[:,2]>0)&(np.abs(az)<=22.5+1e-10)
        horizontal.append(h);v[name+'_horizontal_fraction']=float(ww[h].sum()/ww.sum())
    v['union_horizontal_fraction']=float(ww[horizontal[1]|horizontal[2]].sum()/ww.sum())
    v['max_branch_horizontal_fraction']=max(v['L_horizontal_fraction'],v['R_horizontal_fraction'])
    center=np.average(points,axis=0,weights=ww);angles=[]
    for angle in (0,-15,15):
        pose=sensor@R.extrinsic(angle);loc=(center-pose[:3,3])@pose[:3,:3];angles.append(float(np.rad2deg(np.arctan2(loc[0],loc[2]))))
    return dict(status='EVALUABLE',area_m2=area,unoccluded_area_m2=float(ww.sum()),unoccluded_fraction=float(ww.sum()/area),
        centroid_world=center.tolist(),centroid_azimuth_SLR_deg=angles,quadrature_points=len(pp),**v)


def supplement():
    tick=time.monotonic();check_time()
    R.save(OUT/'FULL_TARGET_PLAN.json',dict(reason='13 query-clipped target geometries absent at last causal frame; inspect full physical target at contact heights to answer requested target FOV, without replacing originals',
        source_sha256=R.sha(__file__),baseline_events_sha256=R.sha(OUT/'events.json'),scope='Same313events/scores/currenttruepose; noquerycrop, retained height slices, same5cmfirsthitquadrature; oldqueryclip secondary unchanged',
        restrictions='No score/model/threshold change; no inference/render/train/futureextension',deadline_unix=R.read(OUT/'PLAN.json')['deadline_unix']))
    records=R.read(OUT/'events.json');rows=R.read(E.R3/'rows.json');G=geometry_module();manifest={}
    for path in (R.MARGIN/'scene_manifest.json',R.FUSION/'natural97000/scene_manifest.json'):
        manifest.update({(s['unit'],s['config']):s for s in R.read(path)})
    truth={};full=[]
    for r in records:
        check_time();row=rows[r['row_index']];sensor,travel,_=R2.metadata(row)
        if row['batch']>=98000:
            if row['unit'] not in truth:truth[row['unit']]=R.read((R.AUG if row['batch']==98000 else R.CONT)/f'truth/{"calibration" if row["batch"]==98000 else "evaluation"}/unit{row["unit"]}.json')
            boxes=truth[row['unit']]['scenes'][row['config']]['boxes']
        else:boxes=manifest[row['unit'],row['config']]['boxes']
        g=full_target_geometry(boxes,travel[r['frame']],sensor[r['frame']],r['contact_queries'],G)
        full.append(dict(row_index=r['row_index'],unit=r['unit'],config=r['config'],batch=r['batch'],depth=r['contact_depth'],dual_timely=r['dual_timely'],single_timely=r['single_timely'],geometry=g))
    R.save(OUT/'full_target_events.json',full)
    metrics=['single_fraction','L_fraction','R_fraction','union_fraction','max_branch_fraction','split_fraction','L_bearing_deg','R_bearing_deg','best_horizontal_margin_deg',
             'single_horizontal_fraction','L_horizontal_fraction','R_horizontal_fraction','max_branch_horizontal_fraction','union_horizontal_fraction']
    masks={'dual_miss':np.array([not r['dual_timely'] for r in full]),'dual_timely':np.array([r['dual_timely'] for r in full]),
        'dual_miss_single_timely':np.array([not r['dual_timely'] and r['single_timely'] for r in full])}
    for depth in range(3):
        for timely in (False,True):masks[f'depth{depth}/dual_{"timely" if timely else "miss"}']=np.array([r['depth']==depth and r['dual_timely']==timely for r in full])
    valid=np.array([r['geometry']['status']=='EVALUABLE' for r in full]);groups={}
    for name,mask in masks.items():
        rr=[r for r,m in zip(full,mask) if m];gg=[r['geometry'] for r,m in zip(full,mask&valid) if m]
        groups[name]=dict(events=len(rr),geometry_n=len(gg),geometry_status={s:sum(r['geometry']['status']==s for r in rr) for s in {r['geometry']['status'] for r in rr}},
            metrics={m:dict(mean=float(np.mean([g[m] for g in gg])) if gg else None,q10_q50_q90=np.quantile([g[m] for g in gg],[.1,.5,.9]).tolist() if gg else [None]*3) for m in metrics},
            split_positive=sum(g['split_fraction']>1e-10 for g in gg),
            centroid_azimuth_SLR_deg=np.quantile([g['centroid_azimuth_SLR_deg'] for g in gg],[.1,.5,.9],axis=0).tolist() if gg else None)
    with np.load(E.OUT/'ledger.npz') as z:units=np.unique(z['unit'])
    with np.load(R.WORK/'cnh-turn-factor-dev-20261006/ledger.npz') as z:boot=z['bootstrap']
    iv=np.searchsorted(units,[r['unit'] for r in full]);paired={}
    def meanboot(values,mask):
        den=boot@np.bincount(iv,weights=mask,minlength=len(units));num=boot@np.bincount(iv,weights=np.where(mask,values,0),minlength=len(units))
        return np.divide(num,den,out=np.full(1000,np.nan),where=den>0)
    for metric in metrics:
        v=np.array([r['geometry'].get(metric,np.nan) for r in full]);delta=meanboot(v,masks['dual_miss']&valid)-meanboot(v,masks['dual_timely']&valid);ok=np.isfinite(delta)
        paired[metric]=dict(miss_minus_timely_ci95=np.quantile(delta[ok],[.025,.975]).tolist() if ok.any() else [None,None],defined=int(ok.sum()))
    R.save(OUT/'full_target_result.json',dict(groups=groups,paired=paired,seconds=time.monotonic()-tick,
        events_sha256=R.sha(OUT/'full_target_events.json'),limits='Physical target at contact heights, no currentquerycrop; centroid area geometry not photons/SNR; firsthit only boxes; samples from same Development'))
    (OUT/'source/final_analysis.py').write_bytes(Path(__file__).read_bytes())
    print({k:groups[k] for k in ('dual_miss','dual_timely','dual_miss_single_timely')},flush=True)


def freeze(start):
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[E.OUT/'ledger.npz',E.OUT/'result.json',E.R3/'online.npz',E.R3/'rows.json',R.MARGIN/'scene_manifest.json',R.FUSION/'natural97000/scene_manifest.json',GEOM,Path(__file__)]
    R.save(OUT/'PLAN.json',dict(phase='EXPLORE consumed simulated Development',start_unix=start,deadline_unix=start+3600,budget_cpu_seconds=3600,
        goal='Confirm exact head-to-travel query condition, fix chapter mode confounding, evaluate whether mode2 geometry/scores support dual seam hypothesis; table and commit',
        restrictions='No synthetic future extension, inference, training, rendering, model/query/threshold changes; existing CPU metadata and branch scores only',
        population='All313 mode2 contact scene episodes, last saved output<=exact.9m deadline; dual timely287/miss26; single same episode paired',
        target='Designated boxes[0] physical surfaces clipped to CURRENT true travel original M3 query boxx+/-.30,z.3..3 and deadline contact-query heights. Target is not guaranteed unique cause of scene-level event/alarm; target absence/occlusion separately missing, no substitution.',
        visibility='Frozen geometry triangle clipping then area-weighted5cm longest-edge centroid quadrature; opaque AABB first hit across all boxes, co-located sensors. Fractions: unoccluded query-relevant target surface visible in current rectangularFOV±22.5deg and native radial range4.8044544m. Denominator common target surface first-hit unoccluded area, not zones/photons/SNR.',
        angles='True sensor_center @ fixed extrinsic Rx10 Ry(+/-15) Rx-10, not simple localYaw; weighted unoccluded surface azimuth and horizontal edge margin. No future poses.',
        scores='Saved raw five-seedM3 then original causal5 smooth per branch/HEAD/BODY before any max. Last output per-branch scene max and contact-query max; peak up to deadline to distinguish timely earlier alarms. Original threshold fixed.',
        comparison='Describe miss26 vs timely287 and paired single on same events; split_fraction=union-max(L,R); depth and batch descriptive to avoid shallow/mode confounding; no fitted moderate-score or edge cutoff',
        bootstrap='1000 whole-unit paired draws same frozen strata batch/mode/turn and seed2026100617; descriptive miss-minus-timely means and paired single-maxbranch; null if no denominator',
        decision='All missed branch peaks below threshold is definitional. Seam support requires spatial loss/split pattern, not mere subthresholdscores; zero split or ample singlebranchFOV weakens simple split hypothesis. No mechanism confirmation.',
        inputs={str(p.relative_to(R.ROOT)):R.sha(p) for p in paths}))
    (OUT/'source').mkdir();(OUT/'source/analysis.py').write_bytes(Path(__file__).read_bytes())


def check_time():
    if time.time()>R.read(OUT/'PLAN.json')['deadline_unix']:raise TimeoutError('CPU task1h elapsed; preserve outputs')


def run():
    tick=time.monotonic();plan=R.read(OUT/'PLAN.json')
    for p,h in plan['inputs'].items():assert R.sha(R.ROOT/p)==h,p
    with np.load(E.OUT/'ledger.npz') as z:d={k:z[k] for k in z.files}
    with np.load(E.R3/'online.npz') as z:frozen=z['score']
    rows=R.read(E.R3/'rows.json');ids=np.flatnonzero(d['contact']&(d['mode']==2));deadline=E.causal_index(d['fraction'])
    G=geometry_module();manifest={}
    for path in (R.MARGIN/'scene_manifest.json',R.FUSION/'natural97000/scene_manifest.json'):
        manifest.update({(s['unit'],s['config']):s for s in R.read(path)})
    raw_cache={};truth_cache={};records=[];hashes={};all_scores=[]
    for i in ids:
        check_time();row=rows[i];batch,u,c=row['batch'],row['unit'],row['config'];fi=int(deadline[i]);frame=fi+3
        if u not in raw_cache:
            path=R.score_sources()[batch]/f'unit{u}.npz'
            hashes[str(path.relative_to(R.ROOT))]=R.sha(path)
            with np.load(path) as z:
                raw=z['reference'] if batch>=98000 else z['raw'];sm=R.smooth(raw)
                raw_cache[u]=(z['configs'].tolist(),sm)
            if batch>=98000:
                path=(R.AUG if batch==98000 else R.CONT)/f'truth/{"calibration" if batch==98000 else "evaluation"}/unit{u}.json'
                truth_cache[u]=R.read(path);hashes[str(path.relative_to(R.ROOT))]=R.sha(path)
        configs,sm=raw_cache[u];ci=configs.index(c);branch=sm[1:,ci] if batch>=97000 else sm[:,ci]
        np.testing.assert_array_equal(branch.max((0,2)),frozen[i,:,1])
        qscore=d['query_score'][i,:,0,:];single_last=float(qscore[fi].max());single_peak=float(qscore[:fi+1].max())
        sensor,travel,_=R2.metadata(row)
        if batch>=98000:boxes=truth_cache[u]['scenes'][c]['boxes']
        else:boxes=manifest[u,c]['boxes']
        queries=np.flatnonzero(d['contact_query'][i]);geo=target_geometry(boxes,travel[frame],sensor[frame],queries,G)
        last=branch[:,fi,:].max(1);peak=branch[:,:fi+1,:].max((1,2))
        dual_timely=bool((peak>=R.THRESHOLD).any());single_timely=bool(single_peak>=R.THRESHOLD)
        assert dual_timely==bool(d['states'][1,-1,i]==0);assert single_timely==bool(d['states'][0,-1,i]==0)
        record=dict(row_index=int(i),batch=batch,unit=u,config=c,turn=row['turn'],frame=frame,contact_depth=int(d['depth'][i]),
            contact_queries=queries.tolist(),dual_timely=dual_timely,single_timely=single_timely,
            L_last=float(last[0]),R_last=float(last[1]),L_peak=float(peak[0]),R_peak=float(peak[1]),
            single_last=single_last,single_peak=single_peak,
            L_contact_last=float(branch[0,fi,queries].max()),R_contact_last=float(branch[1,fi,queries].max()),geometry=geo)
        records.append(record);all_scores.append(branch)
    R.save(OUT/'events.json',records)
    np.savez_compressed(OUT/'branch_scores.npz',scores=np.asarray(all_scores),row_index=ids,deadline_index=deadline[ids])
    metrics=('single_fraction','L_fraction','R_fraction','union_fraction','max_branch_fraction','split_fraction','L_bearing_deg','R_bearing_deg','best_horizontal_margin_deg')
    score_metrics=('L_last','R_last','L_peak','R_peak','single_last','single_peak','L_contact_last','R_contact_last')
    def quantile(v):
        v=np.asarray(v,float);v=v[np.isfinite(v)];return dict(n=len(v),mean=float(v.mean()) if len(v) else None,q10_q50_q90=np.quantile(v,[.1,.5,.9]).tolist() if len(v) else [None]*3)
    def summary(selected):
        rr=[records[k] for k in selected];visible=[r['geometry'] for r in rr if r['geometry']['status']=='EVALUABLE']
        return dict(events=len(rr),single_timely=sum(r['single_timely'] for r in rr),
            geometry_status={s:sum(r['geometry']['status']==s for r in rr) for s in sorted(set(r['geometry']['status'] for r in rr))},
            geometry={m:quantile([g[m] for g in visible]) for m in metrics},scores={m:quantile([r[m] for r in rr]) for m in score_metrics},
            both_last_subthreshold=sum(r['L_last']<R.THRESHOLD and r['R_last']<R.THRESHOLD for r in rr),
            both_peak_subthreshold=sum(r['L_peak']<R.THRESHOLD and r['R_peak']<R.THRESHOLD for r in rr),
            depth_counts={str(k):sum(r['contact_depth']==k for r in rr) for k in range(3)})
    groups={'dual_miss':np.flatnonzero([not r['dual_timely'] for r in records]),'dual_timely':np.flatnonzero([r['dual_timely'] for r in records]),
            'dual_miss_single_timely':np.flatnonzero([not r['dual_timely'] and r['single_timely'] for r in records]),
            'dual_miss_single_miss':np.flatnonzero([not r['dual_timely'] and not r['single_timely'] for r in records])}
    for depth in range(3):
        for timely in (False,True):groups[f'depth{depth}/dual_{"timely" if timely else "miss"}']=np.flatnonzero([r['contact_depth']==depth and r['dual_timely']==timely for r in records])
    for batch in sorted({r['batch'] for r in records}):
        for timely in (False,True):groups[f'batch{batch}/dual_{"timely" if timely else "miss"}']=np.flatnonzero([r['batch']==batch and r['dual_timely']==timely for r in records])
    results={k:summary(ix) for k,ix in groups.items() if len(ix)}
    # Reuse the frozen paired whole-unit draws without generating new choices.
    with np.load(R.WORK/'cnh-turn-factor-dev-20261006/ledger.npz') as z:boot=z['bootstrap']
    units,inv=np.unique(d['unit'],return_inverse=True);iv=inv[ids];valid=np.array([r['geometry']['status']=='EVALUABLE' for r in records])
    paired={}
    def bootmean(values,mask):
        den=boot@np.bincount(iv,weights=mask,minlength=len(units));num=boot@np.bincount(iv,weights=np.where(mask,values,0),minlength=len(units))
        return np.divide(num,den,out=np.full(1000,np.nan),where=den>0)
    for m in metrics:
        values=np.array([r['geometry'].get(m,np.nan) for r in records]);bm=bootmean(values,valid&~np.array([r['dual_timely'] for r in records]));bt=bootmean(values,valid&np.array([r['dual_timely'] for r in records]));delta=bm-bt;ok=np.isfinite(delta)
        paired[m]=dict(miss_minus_timely_ci95=np.quantile(delta[ok],[.025,.975]).tolist() if ok.any() else [None,None],defined=int(ok.sum()))
    values=np.array([r['geometry'].get('single_fraction',np.nan)-r['geometry'].get('max_branch_fraction',np.nan) for r in records])
    for name in ('dual_miss','dual_timely','dual_miss_single_timely'):
        mask=np.zeros(len(records),bool);mask[groups[name]]=True;mask&=valid;bv=bootmean(values,mask);ok=np.isfinite(bv)
        paired[name+'/single_minus_maxbranch']=dict(mean=float(values[mask].mean()) if mask.any() else None,ci95=np.quantile(bv[ok],[.025,.975]).tolist() if ok.any() else [None,None],defined=int(ok.sum()))
    R.save(OUT/'result.json',dict(status='DESCRIPTIVE_COMPLETE',events=len(records),threshold=R.THRESHOLD,groups=results,paired=paired,
        seconds=time.monotonic()-tick,input_hashes=hashes,source_sha256=R.sha(__file__),
        limits=['Same consumed simulated Development; mode2 confounds turning and head-forward','First-hit geometric target visibility is not signal strength or model recognition','Scene-level timely alarm may correspond to background or other height/object','Target clipping uses current straight query, not future curved region','Last score can be below threshold after an earlier timely alarm; peaks also reported','Neither split nor subthresholdscore alone confirms a fusion mechanism']))
    with (OUT/'events.csv').open('x',encoding='utf8',newline='') as f:
        names=['unit','config','batch','turn','frame','contact_depth','dual_timely','single_timely',*score_metrics,'geometry_status',*metrics]
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in records:w.writerow({k:(r['geometry']['status'] if k=='geometry_status' else r.get(k,r['geometry'].get(k,''))) for k in names})
    print({k:results[k] for k in ('dual_miss','dual_timely','dual_miss_single_timely','dual_miss_single_miss')},flush=True)


def report():
    r=R.read(OUT/'result.json');f=R.read(OUT/'full_target_result.json');audit=R.read(OUT/'independent_audit.json')
    lines=['# mode2视场交界与报警查询诊断（2026-10-06）','',
        '**本数据不支持“目标被两路水平视场交界切分、各路只看到一部分”的简单解释。** mode2的26个dual漏报，指定目标在接触相关高度的无遮挡表面均有一条支路覆盖全部水平范围；完整矩形视场的双路并集与较好单支路可见比例逐事件相同，没有几何切分增量。20/26同事件single及时。不能据此确认漏报机制，也不能排除垂直视场、信号强度、噪声或历史对齐/读出失配。','',
        'EXPLORE，用户新授权CPU≤1h，现有已消费模拟Development；不训练、不推理、不渲染、不延长路径。冻结M3、报警阈值0.8557642486787612、因果平滑与0.9m截止。所有指标描述本数据，不作新确认或部署效果。','',
        '## 查询条件核实','',
        '`cnh_cvr_pilot.relative_transforms`先取`q=inv(travel[f])@sensor[f]`，两个输入都是真实模拟位姿；随后`q@inv(noisy[f])@noisy[past]`保留估计历史误差。95000/96000单路由此物化；后续dual和97000单路采用`public_query`预定头部运动指令，它在此配方上精确等于真实当前head-to-travel；98000/99000保存的query也显式与该真变换核对一致。因此“报警查询得到精确当前相对行进方向信息”成立；“所有预测代码直接读取真实travel”不准确，且它不是未来标签输入。','',
        '审计旧批240unit×16帧及现代保存144unit，query最大绝对差4.44e−16，当前帧噪声相消后的三支路变换误差≤5.01e−16。报警查询没有覆盖门的过去1秒方向滞后，但部署需要估计head-to-travel；历史8帧对齐仍使用noisy姿态。已补入[章节局限](../thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)。代码入口：[pilot](cnh_cvr_pilot.py)、[public_query](cnh_m3_pose_ensemble.py)、[支路predict](cnh_dual_sensor_envelope_natural.py)、[现代metadata](cnh_extrinsic_aug_data.py)。','',
        '## 模式混淆与曲线停止','',
        '|模式|接触事件|single未及时|dual未及时|最大τsingle未知时间|最大τdual未知时间|','|---|---:|---:|---:|---:|---:|',
        '|mode0：恒偏15°|294|33|2|97.06%|0.94%|','|mode1：扫视±20°|353|8|3|52.28%|1.79%|','|mode2：转弯且相对头部偏角恒0°|313|10|26|0.16%|2.56%|','',
        '漏报为截止前从未报警的episode，计数不随τ变化。mode2是唯一持续头部正视行进方向的模式；mode1可瞬间跨过0°，不构成独立稳定正视对照。single在mode0漏得更多，但其33个漏报全部被门归为unknown；单路静默集中于mode2主要反映这项筛选，不能概括为转弯更难。dual在mode2确实多漏16个，但转弯与正视未被独立操纵。','',
        '用户明确不批准按原角速度延长轨迹，曲线两臂未执行并停止：mode2最大τ门几乎放行；换门不会产生及时报警，最多改变unknown/静默且增加负担；报警查询已有精确当前相对方向；弯曲检查与当前直线接触真值还不一致。原[姿态报告](CNH_TURN_FACTOR_DEV_20261006.md)已记录理由，原载荷与NOT_EVALUABLE保留。','',
        '## 逐事件配对与目标几何','',
        '主表使用mode2全部313接触episode，取截止前最后已产生输出frame13或14。目标是生成器指定`boxes[0]`；选截止接触的HEAD/BODY高度片，**不裁剪目标的当前横向/前向范围**，以免把在截止时才进入直线查询的目标丢掉。对这些物理表面先用全部boxes做不透明首命中/遮挡检查，再在共同无遮挡表面分母上计算各视场覆盖面积比例。5cm最长边三角形细分的面积加权重心为近似积分；它不是图像像素、sensor zone、回波能量、SNR或识别概率。','',
        '支路姿态为`sensor_center @ Rx(+10) @ Ry(±15) @ Rx(−10)`，FOV水平/垂直半角22.5°、原传感径向范围4.8044544m。正前方约为各支路15°离轴，仍有约7.5°水平余量及15°重叠，不能预设“正前方正好在边界”。','',
        '|mode2事件|single及时|single未及时|合计|','|---|---:|---:|---:|',
        '|dual未及时|20|6|26|','|dual及时|283|4|287|','|合计|303|10|313|','',
        '同事件single报警不是指定目标已被识别的证明：原事件允许任一高度/背景报警计及时。新诊断保留这一场景级口径，没有把box0几何改成新的事件真值。','',
        '|组|目标几何分母|L/R目标方位角中位数（°）|single / L / R视场内面积均值|双路并集 / 较好支路均值|较好支路水平余量中位数（°）|','|---|---:|---|---|---|---:|']
    for name,label in [('dual_miss','dual未及时'),('dual_timely','dual及时'),('dual_miss_single_timely','dual未及时且single及时')]:
        g=f['groups'][name];m=g['metrics']
        pct=lambda k:f"{m[k]['mean']*100:.2f}%"
        lines.append(f"|{label}|{g['geometry_n']}/{g['events']}|{m['L_bearing_deg']['q10_q50_q90'][1]:.2f} / {m['R_bearing_deg']['q10_q50_q90'][1]:.2f}|{pct('single_fraction')} / {pct('L_fraction')} / {pct('R_fraction')}|{pct('union_fraction')} / {pct('max_branch_fraction')}|{m['best_horizontal_margin_deg']['q10_q50_q90'][1]:.2f}|")
    lines+=['','方位角是目标共同无遮挡表面的面积加权方位角，再按事件取中位数；符号保留物理L=−15°、R=+15°，不按镜像换名。余量为22.5°−|方位角|在目标表面加权后的较好支路值，不是逐点最坏余量。','',
        '仅考虑水平视场时，26/26漏报和287/287及时事件均有一支路覆盖100%目标相关表面的水平范围。完整视场下，全部313事件`union_fraction−max_branch_fraction=0`；26漏报中位数可见比例64.31%，287及时为68.56%。漏报−及时面积均值差−4.99pp，整unit分层bootstrap95%区间[−13.14,+2.40]跨零；较好支路水平余量差区间[−0.23,+0.42]°也跨零。不能把总体几何差归为机制。','',
        '|侵入深度|dual漏报 / 该深度事件|漏报组较好支路可见面积均值|及时组对应均值|','|---|---:|---:|---:|']
    for depth,label in enumerate(('0–2cm','2–5cm','>5cm')):
        miss=f['groups'][f'depth{depth}/dual_miss'];timely=f['groups'][f'depth{depth}/dual_timely']
        lines.append(f"|{label}|{miss['events']} / {miss['events']+timely['events']}|{100*miss['metrics']['max_branch_fraction']['mean']:.2f}%|{100*timely['metrics']['max_branch_fraction']['mean']:.2f}%|")
    lines+=['','浅/中侵入占25/26漏报；组间深度构成差别明显。深组漏报仅1例，其均值不能支撑稳定分层结论。逐批描述保留于result，不把左右方向差当转向因果。','',
        '另保留初始当前查询裁片诊断：当帧M3 boxx±.30,z.3..3裁片中，漏报16/26、及时284/287有指定目标表面，缺失10/3不能当目标不可见；可评价300例的union−max仍均为0。发现当前裁片缺失后，先记录FULL_TARGET_PLAN再补上上述完整横向/前向物理目标表面分母，旧结果未覆盖。','',
        '## 保存的逐支路分数','',
        '逐支路raw M3已保存，未重推理。先按每支路/高度独立因果5帧1/2/4/8/16平滑，再取HEAD/BODY最大；表为截止前最后输出logit的P10 / 中位数 / P90，阈值θ=0.855764。不同于概率，不能凭“中等”称呼推断应当相加。','',
        '|组|L末帧logit|R末帧logit|single同事件末帧logit|L/R截止前峰值中位数|','|---|---|---|---|---|']
    fmt=lambda a:' / '.join(f'{v:.3f}' for v in a)
    for name,label in [('dual_miss','dual未及时26'),('dual_timely','dual及时287'),('dual_miss_single_timely','dual未及时且single及时20')]:
        s=r['groups'][name]['scores']
        lines.append(f"|{label}|{fmt(s['L_last']['q10_q50_q90'])}|{fmt(s['R_last']['q10_q50_q90'])}|{fmt(s['single_last']['q10_q50_q90'])}|{s['L_peak']['q10_q50_q90'][1]:.3f} / {s['R_peak']['q10_q50_q90'][1]:.3f}|")
    lines+=['','26漏报的两路末帧及截止前峰值均未过阈，这是按dual漏报定义必然的性质，不能单独证明late-max融合损失。两路末帧中位数−1.939/−1.714，峰值中位数−1.039/−0.576；已有分数没有“两个部分都接近阈值、合起来就该过”的直接证据。及时287组有1例末帧两路都低，但较早已报警，因此仅看末帧不能重新判漏报。contact-query末帧分数另存events/result，主表保持任一查询报警的原口径。','',
        '## 决定与边界','',
        '在本数据上，简单水平视场交界切分假设不受支持，不能凭这一猜测优先宣称前融合能恢复目标。前融合/偏角扫描仍可作为另外的待验证方法，但本轮未启动。最直接的解混淆数据设计是增加“直行且持续正视”，并与转弯且正视匹配深度、几何、噪声和controls；可并入最终统一确认的事前配方，当前NOT_RUN。','',
        '本诊断只是当前保存帧的box0几何与冻结分数：未衡量回波/SNR或历史目标支持，不解释所有接触物体，也不证明视场内M3应正确识别。矩形FOV的垂直限制、信号方向性、噪声、历史对齐与学习读出仍未隔离。所有数据来自已消费模拟Development，无真机/人体交互/安全结论。','',
        '## 复现、核验与交付','',
        f"主诊断墙钟{r['seconds']:.2f}s，补充完整目标几何墙钟{f['seconds']:.2f}s，均为CPU，未启动GPU工作。3项独立几何控制测试通过。独立审计重建全部逐支路因果分数、模式计数、配对事件及分位数；313例首命中/Cartesian FOV重算差≤1.42e−14。对26漏报细化到2.5cm，split仍为0、较好支路水平比例仍为1；dual并集/较好支路面积比例最多变化1.335pp，方位/余量最多变化0.0332°，主表保持5cm原结果。",
        '几何差区间采用既有1000配对整unit分层draws（batch/mode/左右转，seed2026100617）；不是把313事件当独立样本。源码`cnh_mode2_seam_dev.py freeze/run/supplement/report`，固定输出目录只新写载荷，不能原地重跑覆盖。','',
        '[PLAN](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/PLAN.json) · [初始完整结果](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/result.json) · [完整目标结果](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/full_target_result.json) · [逐事件CSV](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/events.csv) · [支路分数](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/branch_scores.npz) · [完整目标逐事件](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/full_target_events.json) · [查询条件审计](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/query_contract_audit.json) · [独立审计](../../../../artifacts.local/work/cnh-mode2-seam-dev-20261006/independent_audit.json)']
    Path(__file__).with_name('CNH_MODE2_SEAM_DEV_20261006.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'source/final_analysis.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run','supplement','report']);p.add_argument('--start',type=float,default=time.time());a=p.parse_args()
    if a.stage=='freeze':freeze(a.start)
    elif a.stage=='run':run()
    elif a.stage=='supplement':supplement()
    else:report()
