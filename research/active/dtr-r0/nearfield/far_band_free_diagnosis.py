"""Consumed Development far-band diagnostics; never trains or alters decisions."""
import argparse, csv, hashlib, html, json, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from rgb_body_query_reference_eval import rays, ray_interval

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
WORK = REPO / 'artifacts.local/work'
OUT = WORK / 'far-band-free-diagnosis-dev-20261011'
PLAN = HERE / 'FAR_BAND_FREE_DIAGNOSIS_PLAN_DEV_20261011.json'
FREE = 'FREE_ON_SAMPLED_RAYS'

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvwrite(p,rows):
    with Path(p).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def nth(z,en,ex,dm,n=16):
    v=np.minimum(z[dm]-en[dm],ex[dm]-z[dm]);v=v[np.isfinite(v)]
    return float(np.partition(v,len(v)-n)[len(v)-n]) if len(v)>=n else None
def predictions(root,model):
    return {r['source_id']:r for suffix in ('predictions.json','sealed_eval/predictions.json')
            for r in load(root/'rgb_inference'/model/suffix)['rows']}
def cached(root):
    result={}
    for line in (root/'eval/per_query.jsonl').open(encoding='utf8'):
        r=json.loads(line)
        if r['source']!='native_perturbed' or r['band']!='1.5-3m':continue
        decisions={m:next(v for k,v in r['predictions'].items() if '/'+m+'/' in k) for m in ('logit','rgb')}
        result[(r['frame_id'],r['query_id'])]=dict(**decisions,scores=r['scores'])
    return result
def contexts():
    v1=WORK/'sync-rgb-tof-dataset-v1-dev-20261011';v11=WORK/'sync-rgb-tof-dataset-v1-1-dev-20261011';v2=WORK/'sync-fusion-confirm-v2-dev-20261011'
    result=[]
    for name,base,geom,ref,dec in [('v2_eval',v2,v2,'eval_reference_roster.json',cached(v2)),('v1_1_all',v1,v11,'reference_roster.json',cached(WORK/'sync-fusion-v1-dev-20261011'))]:
        rr=load(base/ref);gg={r['source_id']:r for r in load(geom/'synthesis_manifest.json')['frames']}
        result.append(dict(name=name,rows=rr['rows'],queries=rr['queries'],geom=gg,dec=dec,pred={m:predictions(base,m) for m in ('dav2','unidepth')}))
    return result
def select():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection.json').exists(), 'Preserve selection'
    inputs=[PLAN];allrows=[];seed=load(PLAN)['sampling']['seed'];rng=np.random.default_rng(seed)
    for ctx in contexts():
        for fr in ctx['rows']:
            dp=ctx['pred']['dav2'][fr['source_id']]
            with np.load(dp['path']) as a:z=a['depth']
            rx,ry=rays(fr['depth_K'],z.shape)
            for j,q in enumerate(ctx['queries']):
                if q['low'][2]!=1.5 or q['high'][2]!=3.:continue
                state=fr['queries'][j]['state'];decision=ctx['dec'].get((fr['source_id'],q['name']))
                en,ex,dm=ray_interval(rx,ry,q);score=nth(z,en,ex,dm)
                rgb=decision['rgb'] if ctx['name']=='v2_eval' else score is not None and score>=.24403834342956543
                logit=decision['logit'] if decision else None
                group='a' if state==FREE and (rgb or logit) else 'b' if state==FREE else 'c' if state=='POSITIVE' else 'unknown'
                allrows.append(dict(dataset=ctx['name'],visit_id=str(fr['visit_id']),capture=str(fr['capture']),split=fr['role'],source_id=fr['source_id'],query_id=q['name'],query_index=j,state=state,pool_group=group,rgb_support=bool(rgb),logit_support=logit,dav_margin16_m=score))
    selected=[dict(r,group='a') for r in allrows if r['pool_group']=='a'];shortages=[]
    for dataset in sorted({r['dataset'] for r in allrows}):
        aa=[r for r in selected if r['dataset']==dataset];target=Counter(r['visit_id'] for r in aa)
        for group in ('b','c'):
            pools={v:sorted([r for r in allrows if r['dataset']==dataset and r['visit_id']==v and r['pool_group']==group],key=lambda r:(r['source_id'],r['query_id'])) for v in sorted({r['visit_id'] for r in allrows if r['dataset']==dataset})}
            chosen=[];remaining={}
            for v,pool in pools.items():
                perm=rng.permutation(len(pool));n=min(target[v],len(pool));chosen.extend(pool[i] for i in perm[:n]);remaining[v]=[pool[i] for i in perm[n:]]
                if n<target[v]:shortages.append(dict(dataset=dataset,group=group,visit_id=v,target=target[v],available=len(pool),shortfall=target[v]-n))
            while len(chosen)<len(aa) and any(remaining.values()):
                v=max(remaining,key=lambda v:(len(remaining[v]),v));chosen.append(remaining[v].pop())
            selected.extend(dict(r,group=group) for r in chosen)
    selected.sort(key=lambda r:(r['dataset'],r['source_id'],r['query_index']))
    csvwrite(OUT/'selection.csv',selected);csvwrite(OUT/'far_query_inventory.csv',allrows)
    frames=[dict(dataset=d,source_id=s,capture=next(r['capture'] for r in selected if r['dataset']==d and r['source_id']==s)) for d,s in sorted({(r['dataset'],r['source_id']) for r in selected})]
    save(OUT/'selection.json',dict(plan_sha256=sha(PLAN),seed=seed,rows=selected,frames=frames,shortages=shortages,pool_counts={d:dict(Counter(r['pool_group'] for r in allrows if r['dataset']==d)) for d in sorted({r['dataset'] for r in allrows})},selected_counts=dict(Counter(r['dataset']+'/'+r['group'] for r in selected)),command_wall_s=time.perf_counter()-started,GPU_s=0))
    print(json.dumps(load(OUT/'selection.json')['selected_counts']))

def reproject(z,K,sp,tp,TK,shape):
    yy,xx=np.where(np.isfinite(z)&(z>0));zz=z[yy,xx]
    xyz=np.stack(((xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz),1)
    T=np.linalg.inv(tp)@sp;p=xyz@T[:3,:3].T+T[:3,3];p=p[p[:,2]>0]
    u=np.rint(TK[0,0]*p[:,0]/p[:,2]+TK[0,2]).astype(int);v=np.rint(TK[1,1]*p[:,1]/p[:,2]+TK[1,2]).astype(int)
    ok=(u>=0)&(u<shape[1])&(v>=0)&(v<shape[0]);out=np.full(np.prod(shape),np.inf);np.minimum.at(out,v[ok]*shape[1]+u[ok],p[ok,2]);return out.reshape(shape)
def quantiles(a,prefix):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return {prefix+'_'+s:float(np.quantile(a,q)) if len(a) else None for s,q in [('p10',.1),('median',.5),('p90',.9),('min',0),('max',1)]}
def depth_stats(z,en,ex,dm,prefix):
    valid=dm&np.isfinite(z)&(z>0);inside=valid&(z>=en)&(z<=ex);before=valid&(z<en);after=valid&(z>ex);n=int(dm.sum())
    return {prefix+'_valid':int(valid.sum()),prefix+'_inside':int(inside.sum()),prefix+'_before':int(before.sum()),prefix+'_after':int(after.sum()),prefix+'_missing_fraction':1-int(valid.sum())/n,**quantiles(z[after],prefix+'_after_z_m'),**quantiles(z[after]-ex[after],prefix+'_after_exit_gap_m')}
def rules(r):
    label=r['native_valid']<64 or (r['confidence2_fraction'] is not None and r['confidence2_fraction']<.9) or (r['state']==FREE and r['faro_inside']>=16)
    geom=r['edge_fraction']>0 or r['domain_pixels']<64 or r['dav_top16_short_fraction']>.5
    nativefree=r['native_after']==r['domain_pixels']
    strong=nativefree and r['confidence2_fraction'] is not None and r['confidence2_fraction']>=.9 and r['faro_after']/r['domain_pixels']>=.9 and r['faro_inside']==0 and r['dav_inside']>=16
    return dict(label_suspect=bool(label),geometry_risk=bool(geom),model_error_strong=bool(strong),model_disagreement_unconfirmed=bool(nativefree and r['dav_inside']>=16 and not strong),fusion_only=bool(r['logit_support'] and not r['rgb_support']),unresolved=not(label or geom or strong))

def panel(path,fr,q,r,en,ex,dm,native,faro,dav,uni,confidence):
    rgb=np.asarray(Image.open(fr['rgb_path']).convert('RGB'));h,w=dm.shape;yy,xx=np.where(dm);bbox=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
    x0,y0,x1,y1=bbox;pad=12;box=(max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad))
    valid=dm&np.isfinite(native)&(native>0);missing=dm&~valid;overlay=rgb.copy();overlay[valid]=(rgb[valid]*.65+np.array([30,230,160])*.35).astype('uint8');overlay[missing]=(rgb[missing]*.4+np.array([255,30,150])*.6).astype('uint8')
    im=Image.fromarray(overlay);dr=ImageDraw.Draw(im);dr.rectangle((x0,y0,x1-1,y1-1),outline='yellow',width=2)
    # Projection of all eight AABB vertices, clipped to the image.
    K=np.asarray(fr['depth_K']);corners=[(K[0,0]*x/z+K[0,2],K[1,1]*y/z+K[1,2]) for x in (q['low'][0],q['high'][0]) for y in (q['low'][1],q['high'][1]) for z in (q['low'][2],q['high'][2])]
    dr.rectangle((max(0,min(x for x,y in corners)),max(0,min(y for x,y in corners)),min(w-1,max(x for x,y in corners)),min(h-1,max(y for x,y in corners))),outline='orange',width=1)
    out=Image.new('RGB',(1200,650),'#101823');d=ImageDraw.Draw(out)
    def label(x,y,text,fill='white'):d.text((x,y),text,fill=fill)
    label(18,12,f"{r['dataset']} {r['source_id']} {r['query_id']} group={r['group']}")
    out.paste(Image.fromarray(rgb).crop(box).resize((370,278)),(20,48));out.paste(im.crop(box).resize((370,278)),(410,48))
    cm=np.zeros((h,w,3),np.uint8);cm[:]=[38,48,60]
    if confidence is not None:
        for val,color in [(0,[230,65,65]),(1,[235,170,55]),(2,[50,200,170])]:cm[dm&(confidence==val)]=color
    else:cm[dm]=[100,100,100]
    out.paste(Image.fromarray(cm).crop(box).resize((370,278)),(800,48))
    label(20,332,'RGB crop');label(410,332,'Yellow: ray-domain; orange: box projection; green valid / magenta missing');label(800,350,'Confidence 0 red / 1 amber / 2 green; grey unavailable')
    # Fixed representative profile: rank16 DAV ray, horizontal pixels across its row.
    inds=np.flatnonzero(dm);m=np.minimum(dav-en,ex-dav).ravel()[inds];order=np.argsort(m);pi=inds[order[-16]] if len(order)>=16 else inds[0];row,col=divmod(int(pi),w)
    label(20,372,f"Profile: row y={row}, DAV rank16 pixel x={col}; optical Z metres (0-8). Query interval grey. Missing = gap.")
    left,top,width,height=50,410,1100,190
    def xy(x,z):return left+x/(w-1)*width,top+height-min(8.,max(0.,z))/8*height
    for x in range(w):
        if dm[row,x]:d.line([xy(x,en[row,x]),xy(x,ex[row,x])],fill='#384455')
    for zval in range(9):
        _,y=xy(0,zval);d.line([(left,y),(left+width,y)],fill='#25313c');label(20,y-4,str(zval))
    for name,z,color in [('LiDAR',native,'#44dda7'),('FARO',faro,'#ffd161'),('DAV2',dav,'#fa729e'),('UniDepth',uni,'#79a8ff')]:
        prev=None
        for x in range(w):
            if dm[row,x] and np.isfinite(z[row,x]) and z[row,x]>0:
                p=xy(x,z[row,x])
                if prev is not None:d.line([prev,p],fill=color,width=2)
                prev=p
            else:prev=None
        label(55+280*['LiDAR','FARO','DAV2','UniDepth'].index(name),615,name,color)
    d.line([(xy(col,0)[0],top),(xy(col,0)[0],top+height)],fill='white')
    out.save(path)

def diagnose():
    started=time.perf_counter();selection=load(OUT/'selection.json');assert selection['plan_sha256']==sha(PLAN)
    confpath=OUT/'confidence/manifest.json';confs=load(confpath).get('rows',[]) if confpath.exists() else [];confby={r['source_id']:r for r in confs if r.get('path')}
    rows=[];inputs={};panels=[];(OUT/'panels').mkdir(exist_ok=True);byframe=defaultdict(list)
    for r in selection['rows']:byframe[(r['dataset'],r['source_id'])].append(r)
    def bind(p,expected=None):
        p=str(p);digest=sha(p)
        if expected:assert digest==expected, p
        inputs[p]=digest
    bind(PLAN);bind(Path(__file__));bind(OUT/'selection.json')
    for root,files in [(WORK/'sync-fusion-confirm-v2-dev-20261011',['eval/per_query.jsonl','eval_reference_roster.json','synthesis_manifest.json','rgb_inference/dav2/predictions.json','rgb_inference/dav2/sealed_eval/predictions.json','rgb_inference/unidepth/predictions.json','rgb_inference/unidepth/sealed_eval/predictions.json']), (WORK/'sync-fusion-v1-dev-20261011',['eval/per_query.jsonl']), (WORK/'sync-rgb-tof-dataset-v1-dev-20261011',['reference_roster.json','rgb_inference/dav2/predictions.json','rgb_inference/dav2/sealed_eval/predictions.json','rgb_inference/unidepth/predictions.json','rgb_inference/unidepth/sealed_eval/predictions.json']), (WORK/'sync-rgb-tof-dataset-v1-1-dev-20261011',['synthesis_manifest.json'])]:
        for name in files:bind(root/name)
    for ctx in contexts():
        frames={r['source_id']:r for r in ctx['rows']}
        for (dataset,sid),selected in byframe.items():
            if dataset!=ctx['name']:continue
            if time.perf_counter()-started>600:raise TimeoutError('Diagnostic allocation')
            fr=frames[sid];gp=ctx['geom'][sid]['geometry_path'];bind(fr['reference_path'],fr['reference_sha256']);bind(fr['native_depth_path'],fr['native_depth_sha256']);bind(fr['rgb_path'],fr['rgb_sha256']);bind(gp)
            with np.load(fr['reference_path']) as a:native=a['depth'];K=a['depth_K'];labels=a['labels'];mx=a['map_x'];my=a['map_y'];observed=a['observed']
            assert native.shape==(192,256);raw=np.asarray(Image.open(fr['native_depth_path']),float)/1000;np.testing.assert_allclose(native,raw,atol=1e-6,rtol=0)
            yy,xx=np.indices(native.shape);assert np.array_equal(mx,xx) and np.array_equal(my,yy) and observed.all()
            with np.load(gp) as a:
                faro=reproject(a['depth'],a['K'],a['grid_pose'],a['rgb_pose'],K,native.shape) if np.isfinite(a['grid_pose']).all() and np.isfinite(a['rgb_pose']).all() else np.full(native.shape,np.inf)
            rgb={}
            for model in ('dav2','unidepth'):
                pred=ctx['pred'][model][sid];bind(pred['path'],pred['sha256'])
                with np.load(pred['path']) as a:rgb[model]=a['depth']
            confidence=None
            if sid in confby:
                cr=confby[sid];bind(cr['path'],cr['sha256']);confidence=np.asarray(Image.open(cr['path']));assert confidence.shape==native.shape and set(np.unique(confidence))<={0,1,2}
            rx,ry=rays(K,native.shape);radial=np.sqrt(1+rx*rx+ry*ry);h,w=native.shape
            for sr in selected:
                r=dict(sr);q=ctx['queries'][r['query_index']];en,ex,dm=ray_interval(rx,ry,q);n=int(dm.sum());physical=(ex-en)*radial;axial=ex-en;coords=np.argwhere(dm);r['domain_pixels']=n
                r.update(depth_stats(native,en,ex,dm,'native'));r.update(depth_stats(faro,en,ex,dm,'faro'));r.update(depth_stats(rgb['dav2'],en,ex,dm,'dav'));r.update(depth_stats(rgb['unidepth'],en,ex,dm,'uni'))
                for m,prefix in [('dav2','dav'),('unidepth','uni')]:
                    z=rgb[m];pair=dm&np.isfinite(native)&(native>0)&np.isfinite(z)&(z>0);r.update(quantiles((z-native)[pair],prefix+'_minus_native_m'));r.update(quantiles(np.abs(z-native)[pair],prefix+'_absdiff_native_m'));r[prefix+'_margin16_m']=nth(z,en,ex,dm)
                after=dm&np.isfinite(native)&(native>ex);r.update(quantiles((native*radial)[after],'native_after_ray_distance_m'));r.update(quantiles(axial[dm],'intersection_axial_m'));r.update(quantiles(physical[dm],'intersection_ray_m'))
                margins=np.minimum(rgb['dav2'][dm]-en[dm],ex[dm]-rgb['dav2'][dm]);top=np.argsort(margins)[-16:];r['dav_top16_short_fraction']=float((physical[dm][top]<.10).mean());r['dav_top16_intersection_median_m']=float(np.median(physical[dm][top]));r['dav_margin15_minus17_m']=nth(rgb['dav2'],en,ex,dm,15)-nth(rgb['dav2'],en,ex,dm,17)
                y,x=coords[:,0],coords[:,1];edge=(x<.05*w)|(x>=.95*w)|(y<.05*h)|(y>=.95*h);r['edge_fraction']=float(edge.mean());r['top_fraction']=float((y<.1*h).mean());r['bottom_fraction']=float((y>=.9*h).mean());r['image_center_x_fraction']=float(x.mean()/w);r['image_center_y_fraction']=float(y.mean()/h)
                r['confidence_status']='AVAILABLE' if confidence is not None else 'UNAVAILABLE';r['confidence_path']=confby[sid]['path'] if confidence is not None else None
                for val in range(3):r['confidence'+str(val)+'_pixels']=int((confidence[dm]==val).sum()) if confidence is not None else None
                r['confidence2_fraction']=r['confidence2_pixels']/n if confidence is not None else None;r.update(rules(r))
                if r['state']==FREE:assert r['native_after']==n and r['native_inside']==0 and r['native_missing_fraction']==0
                r['faro_coverage_fraction']=r['faro_valid']/n;r['faro_confirms_empty']=r['faro_after']/n>=.9 and r['faro_inside']==0;r['sparse_native']=r['native_valid']<64;r['low_confidence']=r['confidence2_fraction'] is not None and r['confidence2_fraction']<.9;r['faro_surface_present']=r['faro_inside']>=16;r['faro_surface_conflict']=r['state']==FREE and r['faro_surface_present']
                r['panel_path']=None
                if r['group']=='a':
                    p=OUT/'panels'/f"{dataset}_{sid}_{r['query_id']}.png";panel(p,fr,q,r,en,ex,dm,native,faro,rgb['dav2'],rgb['unidepth'],confidence);r['panel_path']=str(p.resolve());panels.append(r)
                rows.append(r)
    csvwrite(OUT/'per_query_diagnosis.csv',rows)
    with (OUT/'per_query_diagnosis.jsonl').open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    cats=['label_suspect','geometry_risk','model_error_strong','model_disagreement_unconfirmed','fusion_only','unresolved','sparse_native','low_confidence','faro_surface_conflict','faro_confirms_empty','faro_surface_present']
    summary=[]
    for dataset in ['all','v1_1_all','v2_eval']:
        for group in ('a','b','c'):
            rr=[r for r in rows if r['group']==group and (dataset=='all' or r['dataset']==dataset)];n=len(rr)
            summary.append(dict(dataset=dataset,group=group,n=n,visits=len({r['visit_id'] for r in rr}),categories={c:dict(count=sum(r[c] for r in rr),denominator=n,rate=sum(r[c] for r in rr)/n if n else None) for c in cats},confidence_available=sum(r['confidence_status']=='AVAILABLE' for r in rr),faro_coverage_median=float(np.median([r['faro_coverage_fraction'] for r in rr])) if rr else None,confidence2_fraction_median=float(np.median([r['confidence2_fraction'] for r in rr if r['confidence2_fraction'] is not None])) if any(r['confidence2_fraction'] is not None for r in rr) else None,intersection_ray_median_m=float(np.median([r['intersection_ray_m_median'] for r in rr])) if rr else None,dav_absdiff_native_median_m=float(np.median([r['dav_absdiff_native_m_median'] for r in rr if r['dav_absdiff_native_m_median'] is not None])) if rr else None))
    byvisit=[];contrasts=[]
    for dataset in ['v1_1_all','v2_eval']:
        for visit in sorted({r['visit_id'] for r in rows if r['dataset']==dataset}):
            for group in ('a','b','c'):
                rr=[r for r in rows if r['dataset']==dataset and r['visit_id']==visit and r['group']==group]
                if rr:byvisit.append(dict(dataset=dataset,visit_id=visit,group=group,n=len(rr),**{c:sum(r[c] for r in rr)/len(rr) for c in cats}))
        for comparator in ('b','c'):
            pairs=[]
            for visit in sorted({r['visit_id'] for r in rows if r['dataset']==dataset}):
                pair=[next((r for r in byvisit if r['dataset']==dataset and r['visit_id']==visit and r['group']==g),None) for g in ('a',comparator)]
                if all(pair):pairs.append(pair)
            contrasts.append(dict(dataset=dataset,comparator=comparator,common_visits=[p[0]['visit_id'] for p in pairs],unit='unweighted mean of per-visit a minus control rates; descriptive, unmatched counts',differences_pp={c:100*float(np.mean([p[0][c]-p[1][c] for p in pairs])) if pairs else None for c in cats}))
    csvwrite(OUT/'per_visit.csv',byvisit)
    save(OUT/'summary.json',dict(status='COMPLETE',plan_sha256=sha(PLAN),implementation_sha256=sha(Path(__file__)),selection_sha256=sha(OUT/'selection.json'),groups=summary,visit_contrasts=contrasts,command_wall_s=time.perf_counter()-started,GPU_s=0,rows=len(rows),panels=len(panels)))
    save(OUT/'input_hashes.json',dict(files=[dict(path=p,sha256=s) for p,s in inputs.items()],plan_sha256=sha(PLAN)))
    cards=[]
    for r in panels:
        reasons=', '.join(c for c in cats[:6] if r[c]);rel='panels/'+Path(r['panel_path']).name
        cards.append(f'<article data-reasons="{html.escape(reasons)}"><h2>{html.escape(r["dataset"]+" / "+r["source_id"]+" / "+r["query_id"])}</h2><p>{html.escape(reasons)} · native rays {r["native_valid"]} · confidence2 {r["confidence2_fraction"]} · FARO in {r["faro_inside"]}, coverage {r["faro_coverage_fraction"]:.3f} · DAV in {r["dav_inside"]} · RGB/logit {r["rgb_support"]}/{r["logit_support"]}</p><img loading="lazy" src="{html.escape(rel)}"><label>人工备注（本页本地保存）：<input data-key="{html.escape(rel)}"></label></article>')
    page='''<!doctype html><meta charset="utf-8"><title>远带 FREE 诊断</title><style>body{background:#101823;color:#dde7ef;font:16px system-ui;margin:24px}article{border-top:1px solid #455;margin:24px 0}img{width:100%;max-width:1200px}input{width:65%;padding:8px}select{padding:8px}</style><h1>远带 FREE 误支持逐 query 抽查</h1><p>全 a 组；原图低分辨率256×192，放大不增加细节。绿为native有效，粉为缺测；灰色query轴向区间，深度线缺口为UNKNOWN。影像位置不是语义地面/天花板。规则允许多标签，FARO冲突不自动证明错标。</p><select id="filter"><option value="">全部</option>'''+''.join(f'<option>{c}</option>' for c in cats[:6])+'</select>'+''.join(cards)+'''<script>document.querySelector('#filter').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.reasons.includes(e.target.value));document.querySelectorAll('input').forEach(x=>{x.value=localStorage.getItem(x.dataset.key)||'';x.onchange=()=>localStorage.setItem(x.dataset.key,x.value)})</script>'''
    (OUT/'index.html').write_text(page,encoding='utf8');print(json.dumps(dict(rows=len(rows),panels=len(panels),seconds=time.perf_counter()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','diagnose']);a=p.parse_args();select() if a.mode=='select' else diagnose()
