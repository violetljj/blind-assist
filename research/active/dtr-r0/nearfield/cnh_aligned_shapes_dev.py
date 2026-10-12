"""Finite aligned AABB shape capability map; frozen M3, no training."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import cnh_aligned_boundary_dev as B

ROOT=B.ROOT
OUT=ROOT/'artifacts.local/work/cnh-aligned-shapes-dev-20261008'
K=4
FAMILIES=('horizontal','vertical','protrusion','sign_edge')
BASE_BG=[dict(lo=[-8.,1.65,-8.],hi=[8.,1.80,9.],rho=.30),
         dict(lo=[-8.,-3.,4.2],hi=[8.,1.65,4.4],rho=.35)]


def save(name,value):
    B.save(OUT/name,value)


def layouts():
    for name,inner in (('in_1cm',.29),('in_4cm',.26),('in_12cm',.18),('pass',.35),('clear',.45)):
        for side in (-1,1):yield name,inner,side
    yield 'center',None,0


def scene_rows():
    rows=[]
    def add(family,placement,side,x,width,yr,depth,rho,variant,context='plain'):
        xl,xh=(-width/2,width/2) if side==0 else ((x,x+width) if side>0 else (-x-width,-x))
        bg=[dict(b) for b in BASE_BG]
        if context=='large_face':
            a,b=(.60,1.2) if side>0 else (-1.2,-.60)
            bg.append(dict(lo=[a,-.30,.73],hi=[b,1.65,1.73],rho=.65))
        group='HEAD' if yr[1]<=.42 else 'BODY' if yr[0]>=.42 else 'BOTH'
        rows.append(dict(id=len(rows),family=family,placement=placement,side=side,
            group=group,variant=variant,context=context,rho=rho,
            lo=[xl,yr[0],.65],hi=[xh,yr[1],.65+depth],background=bg))
    for placement,x,side in layouts():
        for length in (.30,.90):
            for thick in (.04,.10):
                for y in (.10,.65):
                    for rho in (.25,.65):
                        add('horizontal',placement,side,x,length,(y-thick/2,y+thick/2),thick,rho,
                            f'length{length:g}_thick{thick:g}')
        for width in (.04,.10):
            for yr,span in (((-.05,.35),'HEAD_40cm'),((.45,.85),'BODY_40cm'),((-.05,1.65),'full_170cm')):
                for rho in (.25,.65):
                    add('vertical',placement,side,x,width,yr,width,rho,span+f'_width{width:g}')
        for y in (.10,.65):
            for rho in (.25,.65):
                for context in ('plain','large_face'):
                    # Protrusion attaches to an optional side face outside +/- .40.
                    # center case remains side-attached; two sides are explicit.
                    if side==0:
                        for s in (-1,1):add('protrusion',placement,s,0.,.65,(y-.10,y+.10),.08,rho,'height20cm',context)
                    else:add('protrusion',placement,side,x,.65-x,(y-.10,y+.10),.08,rho,'height20cm',context)
        for y in (.10,.65):
            for rho in (.25,.65):
                for width,depth,view in ((.30,.02,'front_plate'),(.02,.30,'edge_plate')):
                    add('sign_edge',placement,side,x,width,(y-.10,y+.10),depth,rho,view)
    return rows


def imports():
    A,G,R,normal=B.imports()
    A.OUT=OUT/'runtime'
    import cnh_all_object_geometry as T
    return A,G,R,normal,T


def solid_categories(boxes):
    """Independent axis-aligned volume intersection, matching open y/z."""
    result=[]
    for yl,yh in ((-.20,.42),(.42,.90)):
        def hit(w,closed=False):
            return any((b['hi'][0]>=-w and b['lo'][0]<=w if closed else
                        b['hi'][0]>-w+1e-8 and b['lo'][0]<w-1e-8) and
                       b['hi'][1]>yl+1e-8 and b['lo'][1]<yh-1e-8 and
                       b['hi'][2]>.30+1e-8 and b['lo'][2]<3.-1e-8 for b in boxes)
        result.append('contact' if hit(.30) else 'pass' if hit(.40,closed=True) else 'clear')
    return result


def translated(boxes,z):
    d=np.array([0.,0.,z])
    return [dict(lo=(np.asarray(b['lo'])-d).tolist(),hi=(np.asarray(b['hi'])-d).tolist(),rho=b['rho']) for b in boxes]


def prepare():
    began=time.monotonic();A,G,R,normal,T=imports();rows=scene_rows()
    sensor,query=B.poses(-10.);categories=[];mismatch=0
    for r in rows:
        first=None
        for f in B.FRAMES:
            bg=translated(r['background'],sensor[f,2,3])
            target=translated([r],sensor[f,2,3])[0];boxes=[target,*bg]
            surface=[T.classify_boxes(boxes,q,np.eye(4))['all_category'] for q in (0,1)]
            solid=solid_categories(boxes);mismatch+=int(surface!=solid)
            assert surface==solid,(r['id'],int(f),surface,solid)
            assert solid_categories(bg)==['clear','clear'],(r['id'],'background not clear')
            assert all(T.classify_boxes(bg,q,np.eye(4))['all_category']=='clear' for q in (0,1))
            if first is None:first=surface
            assert surface==first,(r['id'],'time-varying truth')
            if time.monotonic()-began>180:raise TimeoutError('prepare cap')
        categories.append(first)
    cat=np.asarray(categories)
    paths=[Path(__file__),Path(B.__file__),*A.M3_MODELS]
    paths += [Path(m.__file__).resolve() for n,m in list(sys.modules.items())
              if n.startswith('cnh_') and getattr(m,'__file__',None)]
    plan=dict(task='CNH_ALIGNED_SHAPES_DEV_20261008',lane='EXPLORE finite shape proxies',
        authorization='User 继续 after three obstacle families and sign-edge supplement proposal',
        scope='Straight yaw-aligned, exact relative poses; frozen -10 pitch/M3/old theta; no training, angle search, device or heading work',
        budgets_wall_seconds=dict(prepare=180,render_infer=600,analysis_checks_signal=300),
        scene_rows=rows,replicas=K,pitch_deg=-10.,theta=B.THETA,frames=B.FRAMES.tolist(),
        contact_query_events=int((cat=='contact').sum()*K),
        contact_physical_events=int((cat=='contact').any(1).sum()*K),
        category_scene_counts=dict(Counter('/'.join(c) for c in categories)),
        family_scene_counts=dict(Counter(r['family'] for r in rows)),
        input='Frozen M3 seed0..4 mean logits, exact pose projection, 8x8x16 CNH, original float64 last5 exponential smoothing',
        proxies='AABBs only: square rods, rectangular plates/cabinet patches; no cylinders, slanted branches or natural shape claim. Dimensions and balanced grid not obstacle population frequencies.',
        random='SeedSequence[2026100827,scene_id,replica]; K4 shotnoise replicates not independent participants',
        time='16 raw frames dt.2s, .16m/frame; common target front3.05-.16*f; all13outputs; timely f3..f13(front.97>=.9), f14/15 late. All13 clear alarm slots plus segment/clip costs, no actual reminders/minute.',
        truth='Original surface contact/pass/clear plus independent solid parity every output; background-only both queries clear; full columns contact both heights explicitly count two query events but one physical event. No mismatch, no partial ranking; incomplete cells NOT_EVALUABLE with full original denominator.',
        decision_check='Capability map rather than improvement trial: denominators frozen from geometry, K4 event minimum1; no pass gate or superiority. Describe family/height/rho/placement cells and paired large-face context. For subsequent mechanism work prioritize actual missed events in the user main families; distinguish never-seen vs seen-missed, then conditional signal evidence, without declaring cause or retuning. Sign/cube supplementary; no family-average population ranking.',
        stop='Complete one frozen-pitch queue or cap; no new training/improvement arm. Preserve old failed recipes.',
        deliverables='Physical payload/first-hit/target-signal, category and event CSV, family maps, independent focused review, signal supplement, report and scoped commit/push',
        source_sha256={B.logical_path(p):B.sha(p) for p in sorted(set(paths),key=str)})
    save('PLAN.json',plan)
    np.savez_compressed(OUT/'geometry.npz',category=cat,sensor=sensor,public_query=query)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    save('prepare_result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
        scenes=len(rows),surface_solid_mismatch=mismatch,output_frame_checks=len(rows)*13))
    print('PREPARED',len(rows),plan['family_scene_counts'],plan['category_scene_counts'],flush=True)


def run():
    began=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());
    spent=sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('run_failure_*.json'))
    def check():
        if spent+time.monotonic()-began>p['budgets_wall_seconds']['render_infer']:raise TimeoutError('render/infer cap')
    for name,digest in p['source_sha256'].items():
        assert B.sha(ROOT/name)==digest,('Frozen source changed',name)
    A,G,R,normal,T=imports();A.setup_gpu();engine=None;rows=p['scene_rows'];n=len(rows)
    try:
        engine=A.Engine();sensor,query=B.poses(-10.)
        expectation=np.empty((n,16,8,8,16),np.float64);target_signal=np.empty((n,16),np.float64)
        contexts={json.dumps(r['background'],sort_keys=True):r['background'] for r in rows};parity=[];backends=[]
        for key,bg in contexts.items():
            ids=[i for i,r in enumerate(rows) if json.dumps(r['background'],sort_keys=True)==key]
            renderer=G.ExpectedRenderer(sensor,bg)
            try:
                ambient=renderer.ambient.copy();backends.append(renderer.metadata)
                for begin,ends in renderer.iter_render([rows[i] for i in ids],candidate_batch=4,pose_batch=16,deadline_check=check):
                    for j in range(len(ends)):
                        i=ids[begin+j];alpha=rows[i]['rho']/G.ENDPOINT_RHO
                        expectation[i]=ends[j,0]+alpha*(ends[j,1]-ends[j,0])
                        target_signal[i]=(alpha*(ends[j,1]-ends[j,0])).sum((1,2,3))
                # At least one actual long/plate/protrusion target per renderer context.
                probes=[ids[0],ids[-1]]
                if len(ids)>100:
                    probes+= [next(i for i in ids if rows[i]['family']==family) for family in FAMILIES]
                for i in set(probes):
                    ref=R.expected(dict(poses=sensor[[0,3,13,15]],boxes=[rows[i],*bg]))
                    diff=float(np.max(np.abs(ref['expectation']-expectation[i,[0,3,13,15]])))
                    np.testing.assert_allclose(ref['expectation'],expectation[i,[0,3,13,15]],atol=1e-8,rtol=1e-11)
                    parity.append(dict(scene=i,max_abs=diff))
            finally:renderer.close()
        raw=np.empty((n,K,13,2),np.float32);hist=np.empty((n,K,16,8,8,16),np.int32)
        for begin in range(0,n,4):
            check();zs=[]
            for i in range(begin,min(begin+4,n)):
                for k in range(K):
                    seed=int(np.random.SeedSequence([2026100827,i,k]).generate_state(1)[0])
                    hist[i,k]=R.sample(expectation[i],ambient,seed)[0]
                    zs.append(normal(hist[i,k][None],ambient[None])[0])
            z=np.stack(zs);N=len(z);pn=np.broadcast_to(sensor,(N,16,4,4));pq=np.broadcast_to(query,(N,16,4,4))
            fv=[engine.features(z,pn,pq,int(f)) for f in B.FRAMES]
            raw[begin:begin+N//K]=engine.predict(np.stack(fv,1).reshape(N*13,3,24,17,33)).reshape(N//K,K,13,2)
            if begin%40==0:print('SCENES',min(begin+4,n),'/',n,'wall',round(time.monotonic()-began,1),flush=True)
        file=OUT/'physical.npz'
        if file.exists():raise FileExistsError('Preserve existing output')
        np.savez_compressed(file,raw=raw,hist=hist,ambient=ambient,expectation=expectation,target_signal=target_signal)
        check();save('run_result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
            payload_sha256=B.sha(file),backend=backends,renderer_parity=parity))
    except BaseException as e:
        save('run_failure_'+str(time.time_ns())+'.json',dict(status='FAILED',error=repr(e),seconds=time.monotonic()-began));raise
    finally:
        if engine is not None:engine.projector=None;engine.nets=[];engine.torch.cuda.empty_cache()


def analyze():
    began=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());rr=json.loads((OUT/'run_result.json').read_text())
    assert B.sha(OUT/'physical.npz')==rr['payload_sha256']
    save('analysis_source_manifest.json',dict(source_sha256=B.sha(__file__),
        plan_source_sha256=p['source_sha256'][B.logical_path(Path(__file__))],
        changes='Pass solid parity fixed to closed x boundary; cohort avoids exact .40; added descriptive geometry-matched context pairs. Frozen inputs/seeds/model unchanged.'))
    A,G,R,normal,T=imports();rows=p['scene_rows'];sensor,query=B.poses(-10.)
    with np.load(OUT/'physical.npz') as z:scores=B.smooth(z['raw']);signal=z['target_signal']
    cat=np.load(OUT/'geometry.npz')['category'];m=B.metrics(scores,cat,B.THETA)
    directions,_=R.S.angular_rays(16);directions=directions.reshape(-1,3)@sensor[0,:3,:3].T
    rays=np.zeros((len(rows),16),np.int32)
    for i,r in enumerate(rows):
        for f in range(16):
            rays[i,f]=np.count_nonzero(R.S.raycast_boxes(sensor[f,:3,3],directions,[r,*r['background']])['object_id']==0)
        if time.monotonic()-began>180:raise TimeoutError('analysis stage cap; shared300')
    seen=rays[:,:14].any(1);groups=[]
    for family in FAMILIES:
        for placement in ('in_1cm','in_4cm','in_12cm','center','pass','clear'):
            for context in (('plain','large_face') if family=='protrusion' else ('plain',)):
                for q in (0,1):
                    for rho in (.25,.65):
                        ix=np.array([r['family']==family and r['placement']==placement and r['context']==context and r['rho']==rho for r in rows])
                        if not ix.any():continue
                        alarm=scores[ix]>=B.THETA;contact=cat[ix,q]=='contact';clear=(cat[ix]=='clear').all(1)
                        timely=m['timely'][ix,:,q];late=alarm[:,:,11:,q].any(2)&contact[:,None]&~timely
                        groups.append(dict(family=family,placement=placement,context=context,height=('HEAD','BODY')[q],rho=rho,
                            scenes=int(ix.sum()),contact_scenes=int(contact.sum()),contact_events=int(contact.sum()*K),
                            timely=int(timely.sum()),late=int(late.sum()),
                            never_seen_contact_scenes=int((contact&~seen[ix]).sum()),
                            seen_missed_events=int((~timely[contact&seen[ix]]).sum()),
                            clear_slots=int(alarm[clear].any(-1).sum()),clear_slot_denominator=int(clear.sum()*K*13)))
    context_pairs=paired_context(rows,scores,cat)
    result=dict(status='COMPLETE',scope=p['lane'],scenes=len(rows),replicas=K,
        contact_query_events=p['contact_query_events'],contact_physical_events=p['contact_physical_events'],
        overall=B.serial_metrics(m),groups=groups,context_pairs=context_pairs,seconds_before_plot=time.monotonic()-began,
        decision='RETAIN_M3_AND_PITCH; describe shape-specific weaknesses, no training authorized by this run')
    np.savez_compressed(OUT/'evaluated.npz',category=cat,scores=scores,first_target_rays=rays,timely=m['timely'])
    with (OUT/'groups.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
    with (OUT/'event_ledger.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['scene','replica','family','placement','context','variant','intended_group','rho','query','category','timely','late','first_alarm_frame','seen_before_deadline','near_visible_frames','target_expected_counts_before_deadline'])
        for i,r in enumerate(rows):
            for k in range(K):
                for q in (0,1):
                    frames=np.flatnonzero(scores[i,k,:,q]>=B.THETA)
                    late=cat[i,q]=='contact' and not m['timely'][i,k,q] and (scores[i,k,11:,q]>=B.THETA).any()
                    w.writerow([i,k,r['family'],r['placement'],r['context'],r['variant'],r['group'],r['rho'],q,cat[i,q],int(m['timely'][i,k,q]),int(late),int(B.FRAMES[frames[0]]) if len(frames) else '',int(seen[i]),int((rays[i,11:14]>0).sum()),float(signal[i,:14].sum())])
    plot(groups);save('result.json',result);save('analysis_result.json',dict(seconds=time.monotonic()-began,source_sha256=B.sha(__file__)))
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)


def paired_context(rows,scores,cat):
    buckets={}
    for r in rows:
        if r['family']!='protrusion':continue
        key=json.dumps({k:v for k,v in r.items() if k not in ('id','context','background')},sort_keys=True)
        buckets.setdefault(key,{})[r['context']]=r['id']
    pairs=[[p['plain'],p['large_face']] for p in buckets.values()]
    assert all(len(p)==2 for p in buckets.values())
    a=np.array([x[0] for x in pairs]);b=np.array([x[1] for x in pairs]);assert np.array_equal(cat[a],cat[b])
    ma=B.metrics(scores[a],cat[a],B.THETA);mb=B.metrics(scores[b],cat[b],B.THETA)
    paired=B.compare(ma,mb,cat[a])
    for p in paired:
        p['gain_keys']=[[pairs[i],k] for i,k in p['gain_keys']]
        p['loss_keys']=[[pairs[i],k] for i,k in p['loss_keys']]
        p['contact_scene_pairs']=[pairs[i] for i in p.pop('contact_scene_ids')]
    return dict(scope='Same geometry pairs, different random noise draws; descriptive context comparison, no pure causal attribution',
                scene_pairs=pairs,plain=B.serial_metrics(ma),large_face=B.serial_metrics(mb),paired=paired)


def plot(groups):
    import os
    os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    placements=('in_1cm','in_4cm','in_12cm','center');labels=('1 cm entry','4 cm entry','12 cm entry','center/cross')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,family in zip(axs.flat,FAMILIES):
        image=np.full((4,4),np.nan);counts={}
        for gi,(q,rho) in enumerate((('HEAD',.25),('HEAD',.65),('BODY',.25),('BODY',.65))):
            for pi,placement in enumerate(placements):
                gs=[g for g in groups if g['family']==family and g['context']=='plain' and g['height']==q and g['rho']==rho and g['placement']==placement]
                n=sum(g['contact_events'] for g in gs);t=sum(g['timely'] for g in gs)
                if n:image[gi,pi]=t/n;counts[gi,pi]=(t,n)
        im=ax.imshow(image,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
        ax.set_xticks(range(4),labels,rotation=15);ax.set_yticks(range(4),['HEAD dark','HEAD bright','BODY dark','BODY bright'])
        ax.set_title(family+' (AABB proxy; plain context)')
        for (gi,pi),(t,n) in counts.items():ax.text(pi,gi,f'{t}/{n}',ha='center',va='center')
    fig.colorbar(im,ax=axs,label='Target-height timely fraction / K4 noise repeats')
    fig.suptitle('Aligned straight / exact poses / pitch -10 / frozen M3 / original threshold\nFinite shape grid; pass/clear and paired large-face costs in CSV')
    fig.savefig(OUT/'shape_capability.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','analyze'))
    args=parser.parse_args();dict(prepare=prepare,run=run,analyze=analyze)[args.stage]()
