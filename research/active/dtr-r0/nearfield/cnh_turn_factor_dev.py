"""CPU-only pose/path factor diagnostic on frozen consumed Development.

Saved future support is audited before any extrapolation. Oracle pose changes
the gate only; it does not repair the model or change its alert queries.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import time
import numpy as np
import cnh_tristate_dev as R
import cnh_tristate_dev_r2 as R2
import cnh_tristate_dev_r3_geometry as G
import cnh_tristate_event_dev as E

OUT=R.WORK/'cnh-turn-factor-dev-20261006'
ARMS=('estimated/straight','true/straight')


def future_length(travel,frame):
    return float(np.linalg.norm(np.diff(travel[frame:,[0,2],3],axis=0),axis=1).sum())


def true_unit(row):
    sensor,travel,_=R2.metadata(row)
    gates=[];counts=[];missing=[]
    for frame in R.FRAMES:
        ev=G.gates(G.prepared(sensor,int(frame)),[3.])
        gates.append(ev['passed'][0]);counts.append(ev['mask_count'][0]);missing.append(ev['missing_count'][0])
    return dict(unit=row['unit'],gate=np.asarray(gates),mask_count=np.asarray(counts),missing_count=np.asarray(missing),
                future_length=np.asarray([future_length(travel,int(f)) for f in R.FRAMES]))


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[E.OUT/'ledger.npz',E.OUT/'result.json',E.R3/'online.npz',E.R3/'rows.json',Path(__file__),Path(G.__file__),Path(R2.__file__),Path(R.__file__)]
    R.save(OUT/'PLAN.json',dict(phase='EXPLORE consumed simulated Development',start_unix=time.time(),
        budget_cpu_seconds=10800,goal='Turning residual silent-miss pose/path factors; tables, paired whole-unit bootstrap, commit',
        baseline_commit='fd55b74dcf9567be17903fa527da11f56a94da04',
        unchanged='M3 scores, alarm threshold, original r3 m3, 20 tau grid,960 events,3473 controls and causal deadline',
        pose='estimated saved r3 poses vs true sensor center; same past1s displacement direction, nominal pitch-10, history0.6s and original lattice/sentinels. True retains direction lag.',
        region='straight0.9..2.1m,+/-0.29m; true future curved region requires full2.1m saved forward path. Missing support is NOT_EVALUABLE, never unknown/safe/error.',
        path_extension='PENDING_USER_AUTHORIZATION; not executed by this plan',
        fifth_arm='NOT_EVALUABLE_WITHOUT_INFERENCE: nonlinear query-dependent M3; only two fixed-query logits saved',
        variant='A no hold; B cannot alter event partition, omitted to isolate gate effect',
        select='For each sensor/gate choose all-event silent<=10 grid point with minimal all-control unknown time; group rows use same global selected point. Not runtime tuning.',
        bootstrap='1000 paired whole-unit draws using same batch/mode/turn strata and seed2026100617 as event analysis; reselect per replicate at silent rate10/960; no superiority gate',
        decision='Timely counts invariant. Compare changes in silent/unknown and cost; no readout recovery or hardware claim. A single event=0.10417pp globally; mode2=0.31949pp.',
        restrictions='CPU only no training/render/inference/hardware; no changes to stopped readout or thresholds',
        inputs={str(p.relative_to(R.ROOT)):R.sha(p) for p in paths}))
    (OUT/'source').mkdir()
    for p in paths[-4:]: (OUT/'source'/p.name).write_bytes(p.read_bytes())


def gates():
    tick=time.monotonic();plan=R.read(OUT/'PLAN.json')
    for p,h in plan['inputs'].items(): assert R.sha(R.ROOT/p)==h,p
    rows=R.read(E.R3/'rows.json'); unique={r['unit']:r for r in rows}
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(true_unit,unique.values(),chunksize=4))
    byunit={v['unit']:v for v in results}
    with np.load(E.R3/'online.npz') as z: estimated=z['gate'][:,:,1,:]
    truth=np.stack([byunit[r['unit']]['gate'] for r in rows])
    future=np.stack([byunit[r['unit']]['future_length'] for r in rows])
    # Recompute a full estimated unit as a direct inheritance check.
    with np.load(E.R3/'online.npz') as z: poses=z['poses'][:40]
    parity=np.array([[G.gates(G.prepared(p,int(f)),[3.])['passed'][0] for f in R.FRAMES] for p in poses])
    np.testing.assert_array_equal(parity,estimated[:40])
    if (OUT/'gates.npz').exists():raise FileExistsError('Retain existing gates')
    np.savez_compressed(OUT/'gates.npz',gate=np.stack((estimated,truth)),future_length=future,
        mask_count=np.stack([byunit[r['unit']]['mask_count'] for r in rows]),missing_count=np.stack([byunit[r['unit']]['missing_count'] for r in rows]))
    R.save(OUT/'gate_receipt.json',dict(seconds=time.monotonic()-tick,units=len(unique),estimated_parity_frames=int(parity.size/2),
        saved_future_ge2_1_nonwarm_frames=int((future[:,2:]>=2.1-1e-10).sum()),nonwarm_frames=int(future[:,2:].size),
        source_hashes=plan['inputs'],gate_sha256=R.sha(OUT/'gates.npz')))
    print('gates complete',round(time.monotonic()-tick,2),flush=True)


def bootstrap(units,rows):
    uu,inv=np.unique(units,return_inverse=True);rng=np.random.default_rng(2026100617);boot=np.zeros((1000,len(uu)),np.int16)
    strat=np.array([f"{rows[np.flatnonzero(units==u)[0]]['batch']}/{rows[np.flatnonzero(units==u)[0]]['mode']}/{rows[np.flatnonzero(units==u)[0]]['turn']}" for u in uu])
    for s in np.unique(strat):
        ids=np.flatnonzero(strat==s);draw=rng.integers(0,len(ids),size=(1000,len(ids)))
        for k in range(1000):boot[k,ids]=np.bincount(draw[k],minlength=len(ids))
    return uu,inv,boot


def analyze():
    tick=time.monotonic();rows=R.read(E.R3/'rows.json')
    with np.load(E.OUT/'ledger.npz') as z:d={k:z[k] for k in z.files}
    with np.load(E.R3/'online.npz') as z:score=z['score'];taus=z['thresholds']
    with np.load(OUT/'gates.npz') as z:gate=z['gate'];future=z['future_length']
    uu,inv,boot=bootstrap(d['unit'],rows);n=len(rows);control=d['control'];contact=d['contact']
    masks={'all':np.ones(n,bool),'mode2':d['mode']==2,'turn_left':d['turn']=='left','turn_right':d['turn']=='right'}
    states=[];costs=[];curves={};samples={};selected={};totals={}
    def counts(values,mask):return np.bincount(inv,weights=np.where(mask,values,0),minlength=len(uu))
    def ci(values):
        v=np.asarray(values);v=v[np.isfinite(v)];return np.quantile(v,[.025,.975]).tolist() if len(v) else [None,None]
    for j,geometry in enumerate(ARMS):
        for a,sensor in enumerate(('single','dual')):
            key=sensor+'/'+geometry;ss=[];cc=[];risk=[];cost=[]
            for tau in taus:
                state,_,unknown,_=E.partition(score[...,a],gate[j,...,a],tau,d['fraction'],contact)
                ss.append(state);cc.append(E.burden(unknown)['seconds'])
            ss=np.asarray(ss);cc=np.asarray(cc);states.append(ss);costs.append(cc)
            for group,mask in masks.items():
                eventmask=contact&mask;cm=control&mask;den=int(eventmask.sum());cden=int(cm.sum())*2
                ed=boot@counts(np.ones(n),eventmask);cd=boot@counts(np.full(n,2.),cm)
                curve=[];bs=[];bc=[]
                for k,tau in enumerate(taus):
                    bn=boot@counts(ss[k]==2,eventmask);cn=boot@counts(cc[k],cm)
                    br=np.divide(bn,ed,out=np.full(1000,np.nan),where=ed>0);bv=np.divide(cn,cd,out=np.full(1000,np.nan),where=cd>0)
                    curve.append(dict(index=k,tau=float(tau),events=den,controls=cden//2,timely=int(((ss[k]==0)&eventmask).sum()),
                        unknown_miss=int(((ss[k]==1)&eventmask).sum()),silent=int(((ss[k]==2)&eventmask).sum()),
                        silent_rate_ci95=ci(br),unknown_time=float(cc[k,cm].sum()/cden) if cden else None,unknown_ci95=ci(bv)))
                    bs.append(br);bc.append(bv)
                curves.setdefault(group,{})[key]=curve
                if group=='all':risk=np.array(bs);cost=np.array(bc)
            feasible=[p for p in curves['all'][key] if p['silent']<=10]
            selected[key]=min(feasible,key=lambda p:(p['unknown_time'],p['index']))['index'] if feasible else None
            feasible=(risk<=10/960+1e-12)&np.isfinite(cost);v=np.where(feasible,cost,np.inf).min(0);v[~np.isfinite(v)]=np.nan
            samples[key]=dict(matched=v,max_silent=risk[-1],max_unknown=cost[-1])
            totals[key]=dict(max_tau=curves['all'][key][-1],budget10=None if selected[key] is None else curves['all'][key][selected[key]])
    np.testing.assert_array_equal(np.stack(states[:2]),d['states'][:2])
    for a in range(2):np.testing.assert_array_equal(np.stack(states)[a]==0,np.stack(states)[a+2]==0)
    paired={}
    for sensor in ('single','dual'):
        old=samples[sensor+'/'+ARMS[0]];new=samples[sensor+'/'+ARMS[1]]
        paired[sensor]={}
        for metric in old:
            delta=new[metric]-old[metric];paired[sensor][metric]=dict(ci95=ci(delta),defined=int(np.isfinite(delta).sum()))
    ix=E.causal_index(d['fraction']);rr=np.flatnonzero(contact)
    counts_record=dict(events=int(contact.sum()),controls=int(control.sum()),units=len(uu),
        curved_saved_path_status='NOT_EVALUABLE: no nonwarm saved frame has2.1m actual future path',
        event_last_frame_future_lengths={str(f+3):sorted(np.unique(np.round(future[rr[ix[rr]==f],f],8)).tolist()) for f in np.unique(ix[rr])})
    R.save(OUT/'result.json',dict(status='POSE_FACTOR_COMPLETE_PATH_PENDING',counts=counts_record,curves=curves,selected_indices=selected,
        summary=totals,paired_true_minus_estimated=paired,tau_grid=taus.tolist(),seconds=time.monotonic()-tick,
        limits=['Consumed simulated Development; oracle gate only','Timely unchanged; no readout improvement measured','True sensor pose still uses past1s direction and retains lag','Curved true future region unavailable without new continuation assumption','Budget selection is descriptive, bootstrap reselects per draw']))
    np.savez_compressed(OUT/'ledger.npz',states=np.stack(states),unknown_seconds=np.stack(costs),arms=np.asarray([s+'/'+g for g in ARMS for s in ('single','dual')]),
        unit=d['unit'],mode=d['mode'],turn=d['turn'],contact=contact,control=control,bootstrap=boot)
    with (OUT/'curves.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['group','arm','index','tau','events','controls','timely','unknown_miss','silent','unknown_time']);w.writeheader()
        for group,arms in curves.items():
            for arm,points in arms.items():
                for p in points:w.writerow(dict(group=group,arm=arm,**{k:p[k] for k in w.fieldnames if k not in ('group','arm')}))
    print(totals,flush=True)


def report():
    r=R.read(OUT/'result.json');receipt=R.read(OUT/'gate_receipt.json')
    lines=['# 转弯残余静默因素诊断（2026-10-06）','',
        '**替换真实姿态未减少转弯静默；现有真实未来轨迹不足以完成路径因素拆分。** 本轮只完成2×2中的直线两臂，曲线两臂为NOT_EVALUABLE。不能据此判定读出是唯一原因，也不能按“两因素都不明显”停止路径问题。','',
        'EXPLORE，复用`fd55b74d`的已消费模拟Development，CPU≤3h；不训练、不渲染、不推理。冻结M3、原报警阈值0.8557642486787612、r3 original/m=3°、20点τ网格与事件/control定义。A无保持；B不改变事件分拆，本轮不重复列成本变体。','',
        '## 因素的实际定义与缺失','',
        '估计姿态直接复用r3 online的poses；真实姿态为sensor_center（含头部姿态），不是travel姿态。两者均用过去1秒位移的R2.direction和原nominal pitch−10°；真实臂仍保留转弯方向滞后。FOV历史0.6s、径向限制、世界5cm格点与边缘sentinels、非空收缩核心均沿用r3，只有位姿输入替换。报警查询保持原0.3–3.0m×±0.30m两高度，因此及时必然不变，门只改变unknown/clear。','',
        '真实travel仅保存frame0–15，共2.4m路径。排除预热后的frame5–15，最长剩余路径1.6m，全部168960个有效方向候选帧均不足2.1m；960个事件的最后因果输出frame13/14之后仅剩0.32/0.16m。不能截短区域、补零或把缺失当unknown/漏报。按原角速度延长可以做合成诊断，但它不是已保存真实未来路径，需要另明确该假设；本轮尚未执行。','',
        '|覆盖门因素|直线0.9–2.1m×±0.29m|同宽真实未来曲线|','|---|---|---|',
        '|估计姿态|已完成，原r3基线全量复现|NOT_EVALUABLE：远域真实未来轨迹缺失|',
        '|真实姿态|已完成，oracle诊断|NOT_EVALUABLE：远域真实未来轨迹缺失|','',
        '第5臂不可由原分数重建：[M3](cnh_cvr_pilot.py)把查询mask作为第4/5输入通道送入Conv3d，再参与masked max/mean及非线性head；保存raw/query_score仅是固定HEAD/BODY两个logits。保存观测可供未来新投影/推理，但本轮不启动。','',
        '## 计数和负担','',
        '三项顺序为及时 / 无法判断漏报 / 静默。最大τ为网格19；预算选点在全体960事件上要求静默≤10，再取3473条controls unknown时间最小的点，各分组使用同一个全体选点，没有分组重调。每条control为2.0s，累计6946s；分组控制数在表内。unknown时间不是连续真实行走或人体交互成本。','',
        '|组（事件/controls）|传感器 / 姿态|最大τ三项|最大τ unknown时间|≤10选点index|选点三项|选点unknown时间|','|---|---|---|---:|---:|---|---:|']
    for group in ('all','mode2','turn_left','turn_right'):
        for sensor in ('single','dual'):
            for geometry in ARMS:
                key=sensor+'/'+geometry;p=r['curves'][group][key][-1];i=r['selected_indices'][key];b=r['curves'][group][key][i]
                tri=lambda v:f"{v['timely']} / {v['unknown_miss']} / {v['silent']}"
                lines.append(f"|{group}（{p['events']}/{p['controls']}）|{sensor} / {'估计' if geometry.startswith('estimated') else '真实'}|{tri(p)}|{p['unknown_time']*100:.2f}%|{i}|{tri(b)}|{b['unknown_time']*100:.2f}%|")
    lines+=['','1000次配对整unit分层bootstrap，保留384个unit的全部configs/frames；沿用原batch/mode/左右转分层与seed2026100617。下表为真实−估计的unknown负担差（pp），预算行每次重采样重新按10/960风险率选择网格；全部1000/1000次有定义。','',
        '|传感器|最大τ差 [95%描述区间]|≤10重选差 [95%描述区间]|','|---|---|---|']
    for sensor in ('single','dual'):
        old=r['summary'][sensor+'/'+ARMS[0]];new=r['summary'][sensor+'/'+ARMS[1]];pair=r['paired_true_minus_estimated'][sensor]
        fmt=lambda delta,interval:f'{delta*100:+.2f} [{interval[0]*100:+.2f}, {interval[1]*100:+.2f}]'
        lines.append(f"|{sensor}|{fmt(new['max_tau']['unknown_time']-old['max_tau']['unknown_time'],pair['max_unknown']['ci95'])}|{fmt(new['budget10']['unknown_time']-old['budget10']['unknown_time'],pair['matched']['ci95'])}|")
    lines+=['','dual预算选点从index11改到10，实际静默10→8，并非相同风险率精确匹配。负担39.77%→44.61%，差+4.83pp区间[−1.34,+20.93]跨零；不能写真实姿态显著变差。single负担60.59%→59.01%，但转弯静默10与对应事件划分不变。最大τ dual多出1个全体静默，转弯26不变；更准确的几何门放行原unknown，不意味着读出变好。','',
        '## 下一决定与边界','',
        '本数据不支持“位姿估计误差是转弯残余的主要可修复因素”；真实臂没有消除过去方向公式的滞后，因此不能扩大成“方向估计无关”。路径因素尚未评价，不能把未评价当阴性结果，也不能直接归因读出漏掉。下一步若继续本问题，先明确未来轨迹延续假设，再做标明合成假设的曲线门；若要求严格真实未来，需新的更长轨迹。重力变体、IMU曲线部署方法、交互提示、文献查新及统一确认批本轮NOT_RUN。','',
        '左右转来源仍混合批次/镜像配方；oracle姿态只诊断这批模拟Development，不可部署，不证明真机、人体动作或安全。旧失败及载荷保留。','',
        '## 复现与验证','',
        f"姿态门阶段墙钟{receipt['seconds']:.2f}s（4个CPU进程）；统计阶段墙钟{r['seconds']:.2f}s，均未使用GPU。墙钟不是累计CPU时间；主计算不到一分钟，明显低于CPU3h上限。",
        '原估计门额外重算40条序列×13帧×2路逐元素一致；原A事件状态全量一致；四臂及时不变断言通过。沿用8项事件截止/负担边界测试通过；独立计数/选点检查见audit。','',
        '[PLAN](../../../../artifacts.local/work/cnh-turn-factor-dev-20261006/PLAN.json) · [完整JSON与区间](../../../../artifacts.local/work/cnh-turn-factor-dev-20261006/result.json) · [CSV](../../../../artifacts.local/work/cnh-turn-factor-dev-20261006/curves.csv) · [门账本](../../../../artifacts.local/work/cnh-turn-factor-dev-20261006/gates.npz) · [事件账本](../../../../artifacts.local/work/cnh-turn-factor-dev-20261006/ledger.npz) · [独立审计](../../../../artifacts.local/work/cnh-turn-factor-dev-20261006/independent_audit.json)', '',
        '源码`cnh_turn_factor_dev.py`：先freeze，再gates/analyze/report；载荷采用新文件写入，不原地覆盖。freeze源码保留在source，生成报告后的最终源码另存source/final_analysis.py。']
    target=Path(__file__).with_name('CNH_TURN_FACTOR_DEV_20261006.md')
    target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'source/final_analysis.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','gates','analyze','report']);a=p.parse_args()
    globals()[a.stage]()
