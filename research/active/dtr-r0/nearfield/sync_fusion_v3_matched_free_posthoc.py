"""Cache-only posthoc whole-tie oracle; no fitting or working-point changes."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
SRC=REPO/'artifacts.local/work/sync-fusion-confirm-v3-dev-20261011'
OUT=REPO/'artifacts.local/work/sync-fusion-v3-matched-free-posthoc-dev-20261011'
PLAN=HERE/'SYNC_FUSION_V3_MATCHED_FREE_POSTHOC_PLAN_DEV_20261011.json'
METHODS=('D_prime','B_prime','audit_b','rgb','tof','or','and')
ARMS=('native_perturbed','faro_rho030_ambient1','faro_rho030_ambient10')
BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m')
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvout(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def threshold(x):return 'disabled' if not np.isfinite(x) else float(x)

def ties(s,y,v,nv=12):
    ii=np.flatnonzero(np.isfinite(s));ii=ii[np.argsort(-s[ii],kind='stable')]
    ends=np.r_[np.flatnonzero(np.diff(s[ii]))+1,len(ii)] if len(ii) else np.array([],int)
    cuts=np.r_[np.inf,s[ii][ends-1]]
    counts=[]
    for lab in (1,0,-1):
        a=np.zeros((nv,len(ii)),np.int32)
        if len(ii):a[v[ii],np.arange(len(ii))]=(y[ii]==lab)
        counts.append(np.column_stack((np.zeros(nv,int),np.cumsum(a,axis=1)[:,ends-1])))
    return cuts,counts
def exact(cuts,W,F,U):
    # Last threshold at each F has maximum W (counts are monotonic).
    last=np.r_[np.flatnonzero(np.diff(F)),len(F)-1]
    best=np.maximum(np.searchsorted(F,F[last],side='left'),np.searchsorted(W,W[last],side='left'))
    return [(int(F[i]),int(W[i]),int(U[i]),float(cuts[i])) for i in best]
def point(curve,cap):return curve[np.searchsorted([r[0] for r in curve],cap,side='right')-1]
def boot_counts(counts,weights,den,rate):
    W,F,U=[weights@c for c in counts]
    cap=np.floor(rate*(weights@den)).astype(int)
    # nearest F <= cap; last index also has max W at this F.
    indices=(F<=cap[:,None]).sum(axis=1)-1
    ar=np.arange(len(weights))
    fw=F[ar,indices];ww=W[ar,indices]
    indices=((F==fw[:,None])&(W==ww[:,None])).argmax(axis=1)
    return W[ar,indices],F[ar,indices],indices

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'PLAN.json',load(PLAN))
    c=npz(SRC/'eval/cache.npz');cal=npz(SRC/'cal/cache.npz');seal=load(SRC/'cal/cal_seal.json')
    assert sha(SRC/'eval/cache.npz')==load(SRC/'eval/cache_seal.json')['cache_sha256']
    assert sha(SRC/'cal/cache.npz')==seal['cache_sha256']
    visits=load(SRC/'eval/cache_seal.json')['visit_ids'];vi={str(v):i for i,v in enumerate(visits)}
    v=np.array([vi[str(x)] for x in c['visit_id']]);rng=np.random.default_rng(20261011)
    draw=rng.integers(0,12,(2000,12));weights=np.array([np.bincount(d,minlength=12) for d in draw],np.int32)
    tables=[];curves=[];fixed=[];boots=[];visitrows=[];drift=[];driftv=[];replicates={};selections={}
    for arm in ARMS:
      for b in range(3):
        mask=(c['arm']==arm)&(c['band']==b);y=c['label'][mask];vv=v[mask]
        P=int(sum(y==1));N=int(sum(y==0));den=np.bincount(vv[y==0],minlength=12);posden=np.bincount(vv[y==1],minlength=12)
        dat={};pooledcuts={}
        for m in METHODS:
          cuts,counts=ties(c[m+'_score'][mask],y,vv);W,F,U=[a.sum(0) for a in counts]
          curve=exact(cuts,W,F,U);dat[m]=(cuts,counts,curve)
          base=dict(arm=arm,band=b,method=m,POS=P,FREE=N)
          tables.extend(dict(**base,tie=i,threshold=threshold(ct),W=int(ww),F=int(ff),U=int(uu)) for i,(ct,ww,ff,uu) in enumerate(zip(cuts,W,F,U)))
          curves.extend(dict(**base,F=f,W=w,U=u,threshold=threshold(ct)) for f,w,u,ct in curve)
          for lev in ('0','4','67','1%','2%','5%','10%'):
            cap=int(lev) if '%' not in lev else int(np.floor(N*float(lev[:-1])/100))
            f,w,u,ct=point(curve,cap)
            fixed.append(dict(**base,level=lev,cap=cap,F=f,W=w,U=u,threshold=threshold(ct),status='EVALUABLE' if N and P else 'N/E_RATE'))
            if lev in ('2%','5%'):pooledcuts[m,lev]=ct
          for rate in (.02,.05):
            bw,bf,ix=boot_counts(counts,weights,den,rate);key=f'{arm}__{b}__{m}__{rate}'
            replicates[key+'__W']=bw;replicates[key+'__F']=bf;replicates[key+'__tie']=ix
          cc=next(x for x in seal['cells'] if x['method']==m and x['band']==b)
          for role,cache in (('cal',cal),('eval',c)):
            mm=(cache['arm']==arm)&(cache['band']==b);yy=cache['label'][mm];pr=cache[m+'_pred'][mm]
            drift.append(dict(arm=arm,band=b,method=m,role=role,threshold=cc['threshold'],target=cc['target'],F=int(sum(pr&(yy==0))),FREE=int(sum(yy==0)),W=int(sum(pr&(yy==1))),POS=int(sum(yy==1))))
            for visit in sorted(set(cache['visit_id'][mm])):
              vm=mm&(cache['visit_id']==visit);yl=cache['label'][vm];pp=cache[m+'_pred'][vm]
              driftv.append(dict(arm=arm,band=b,method=m,role=role,visit_id=str(visit),F=int(sum(pp&(yl==0))),FREE=int(sum(yl==0)),W=int(sum(pp&(yl==1))),POS=int(sum(yl==1))))
        for rate in (.02,.05):
          valid=(weights@den>0)&(weights@posden>0);lev=f'{round(rate*100)}%'
          for comparator in ('rgb','or'):
            key=lambda m:f'{arm}__{b}__{m}__{rate}__W'
            delta=replicates[key('D_prime')]-replicates[key(comparator)]
            pp=100*delta[valid]/(weights@posden)[valid]
            dp=point(dat['D_prime'][2],int(np.floor(rate*N)));cp=point(dat[comparator][2],int(np.floor(rate*N)))
            ci=np.quantile(delta[valid],[.025,.975]) if valid.any() else [None,None]
            pi=np.quantile(pp,[.025,.975]) if valid.any() else [None,None]
            boots.append(dict(arm=arm,band=b,level=lev,comparator=comparator,delta_W=dp[1]-cp[1],D_F=dp[0],comparison_F=cp[0],POS=P,FREE=N,lower=ci[0],upper=ci[1],lower_pp=pi[0],upper_pp=pi[1],valid=int(valid.sum()),replicates=2000))
          for j,visit in enumerate(visits):
            local=vv==j;ny=int(sum(y[local]==0));py=int(sum(y[local]==1));res={}
            for m in ('D_prime','rgb','or'):
              cc,co=ties(c[m+'_score'][mask][local],y[local],np.zeros(sum(local),int),1)
              cw,cf,cu=[a.sum(0) for a in co];res[m]=point(exact(cc,cw,cf,cu),int(np.floor(rate*ny)))
            row=dict(arm=arm,band=b,level=lev,visit_id=str(visit),POS=py,FREE=ny,status='EVALUABLE' if ny and py else 'N/E_RATE')
            for m in res:
              row[m+'_W']=res[m][1];row[m+'_F']=res[m][0]
              row[m+'_pooled_W']=int(sum(local&(y==1)&(c[m+'_score'][mask]>=pooledcuts[m,lev])))
            row['D_ge_rgb']=bool(res['D_prime'][1]>=res['rgb'][1]) if ny and py else ''
            visitrows.append(row)
        print('curves',arm,b,'wall',time.perf_counter()-start,flush=True)
    # Pooled six-arm cal selection, followed by per-arm drift decomposition.
    for cc in seal['cells']:
      drift.append(dict(arm='CAL_POOLED_SIX_ARMS',band=cc['band'],method=cc['method'],role='cal',threshold=cc['threshold'],target=cc['target'],F=cc['F'],FREE=cc['FREE_denominator'],W=cc['W'],POS=cc['POS_denominator']))
    for name,rows in [('all_ties.csv',tables),('curves.csv',curves),('fixed_points.csv',fixed),('bootstrap.csv',boots),('visit_consistency.csv',visitrows),('drift.csv',drift),('drift_visit.csv',driftv)]:csvout(name,rows)
    np.savez_compressed(OUT/'bootstrap_draws.npz',draw=draw,**replicates)
    loss,groups,lossstats=loss_analysis(c)
    csvout('loss_queries.csv',loss);csvout('loss_groups.csv',groups)
    primary=[r for r in fixed if r['arm']==ARMS[0] and r['band']==1]
    aa=all(next(r['W'] for r in primary if r['level']==lev and r['method']=='D_prime')>=next(r['W'] for r in primary if r['level']==lev and r['method']==m) for lev in ('1%','2%','5%') for m in ('rgb','or'))
    vc={lev:sum(r['D_ge_rgb'] is True for r in visitrows if r['arm']==ARMS[0] and r['band']==1 and r['level']==lev) for lev in ('2%','5%')}
    decision=load(PLAN)['decision']['if_both' if aa and min(vc.values())>=8 else 'otherwise']
    plot(curves,fixed)
    inputs=[SRC/'eval/cache.npz',SRC/'cal/cache.npz',SRC/'cal/cal_seal.json',SRC/'features/eval.npz',PLAN]
    summary=dict(decision=decision,rule_a=aa,visit_success=vc,primary_points=primary,bootstrap=boots,loss=lossstats,plan_sha256=sha(PLAN),source_inputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in inputs],seconds=time.perf_counter()-start,GPU_s=0)
    save(OUT/'summary.json',summary);print(json.dumps(dict(decision=decision,visits=vc,seconds=summary['seconds']),ensure_ascii=False))

def loss_analysis(c):
    from rgb_body_query_reference_eval import rays,ray_interval
    feat=npz(SRC/'features/eval.npz');assert all(np.array_equal(feat[k],c[k]) for k in ('arm','frame_id','query_id','band'))
    take=(c['arm']==ARMS[0])&(c['band']==1)&(c['label']==1)&c['rgb_pred'];ii=np.flatnonzero(take)
    lost=~c['D_prime_pred'][ii];assert lost.sum()==126
    roster=load(SRC/'public_roster.json');pub={r['source_id']:r for r in roster['rows']};qs={q['name']:q for q in roster['queries']}
    preds={r['source_id']:r for rel in ('predictions.json','sealed_eval/predictions.json') for r in load(SRC/'rgb_inference/dav2'/rel)['rows']}
    synth={r['source_id']:r for r in load(SRC/'synthesis_manifest.json')['frames']}
    obsseal=load(SRC/'frozen_readout_seal.json');snrcache={};dcache={};loss=[];bound={}
    expected={str(Path(r['path']).resolve()):r['sha256'] for r in load(SRC/'feature_manifest.json')['inputs']}
    def bind(path):
      path=Path(path).resolve();value=sha(path);assert value==expected[str(path)],str(path);bound[str(path)]=value
    score=c['rgb_score'][ii];quart=np.quantile(score,[.25,.5,.75]);cut=next(x['threshold'] for x in load(SRC/'cal/cal_seal.json')['cells'] if x['method']=='rgb' and x['band']==1)
    for idx,is_loss in zip(ii,lost):
      fid=str(c['frame_id'][idx]);qid=str(c['query_id'][idx]);q=qs[qid]
      if fid not in dcache:
        r=pub[fid];bind(preds[fid]['path']);z=npz(preds[fid]['path'])['depth'];rx,ry=rays(r['depth_K'],z.shape);dcache[fid]=(z,rx,ry)
      z,rx,ry=dcache[fid];entry,exit,dom=ray_interval(rx,ry,q);finite=dom&np.isfinite(z)&(z>0);inside=finite&(z>=entry)&(z<=exit)
      support=int(inside.sum());zmed=float(np.median(z[inside])) if support else None
      margins=np.minimum(z[finite]-entry[finite],exit[finite]-z[finite])
      raw=float(np.partition(margins,len(margins)-16)[len(margins)-16]) if len(margins)>=16 else -np.inf
      assert np.isclose(raw,c['rgb_score'][idx],rtol=0,atol=1e-12), (fid,qid,raw,c['rgb_score'][idx])
      X=feat['X'][idx];unknown=float(X[3]);nfinite=int(round(dom.sum()*(1-unknown)));after=float(X[6]);before=float(X[4]);ins=float(X[5])
      # Evidence proxies, never equate absence/UNKNOWN with FREE.
      empty='finite_all_after' if X[9]==2 and unknown<1 and ins==0 and before==0 and after>0 else 'finite_outside_mixed_or_before' if X[9]==2 and unknown<1 and ins==0 else 'inside_or_unknown'
      if fid not in snrcache:
        vals=[]
        for a in synth[fid]['arms']:
          if a['arm']!=ARMS[0]:continue
          bind(a['path']);zz=npz(a['path']);h=zz['hist'];pk=h.argmax(-1);height=np.take_along_axis(h,pk[...,None],-1)[...,0];bg=np.take_along_axis(zz['background'],pk[...,None],-1)[...,0]
          snr=height/np.sqrt(np.maximum(height,0)+2*bg+1);accepted=(zz['coverage']>=obsseal['tof_zone_coverage_min'])&(snr>=obsseal['tof_peak_snr_min'])
          vals.append((float(snr.max()),float(snr[accepted].max()) if accepted.any() else 0.,int(accepted.sum())))
        snrcache[fid]=np.mean(vals,axis=0)
      snrmax,acceptedmax,naccepted=snrcache[fid]
      row=dict(visit_id=str(c['visit_id'][idx]),frame_id=fid,query_id=qid,loss=bool(is_loss),rgb_score=float(c['rgb_score'][idx]),rgb_cut_margin=float(c['rgb_score'][idx]-cut),rgb_quartile=int(np.searchsorted(quart,c['rgb_score'][idx],side='right')+1),rgb_support_rays=support,rgb_support_bin='0-15' if support<16 else '16-63' if support<64 else '64-255' if support<256 else '256+',rgb_strong=bool(c['rgb_score'][idx]>=quart[1] and support>=16),tof_unknown=unknown,tof_unknown_bin='0-.25' if unknown<.25 else '.25-.75' if unknown<.75 else '.75-1',tof_support_Kmean=float(X[2]),tof_support_bin='0' if X[2]==0 else '0-16' if X[2]<16 else '16+',tof_valid_K=int(X[9]),tof_score=threshold(c['tof_score'][idx]),tof_empty_proxy=empty,tof_peak_snr_max_Kmean=float(snrmax),tof_accepted_peak_snr_max_Kmean=float(acceptedmax),tof_accepted_zones_Kmean=float(naccepted),tof_peak_bin='<3' if acceptedmax<3 else '3-10' if acceptedmax<10 else '10+',rgb_support_median_Z_m=zmed,rgb_band_position='no_inside' if zmed is None else 'lower_third' if zmed<.8+.7/3 else 'middle_third' if zmed<.8+1.4/3 else 'upper_third',query_xy=qid.split('_z')[0],evidence_group=str(c['evidence_group'][idx]))
      loss.append(row)
    save(OUT/'loss_input_receipts.json',dict(feature_manifest_sha256=sha(SRC/'feature_manifest.json'),inputs=[dict(path=p,sha256=h) for p,h in bound.items()],all_1609_rgb_raw_margins_recomputed=True))
    csvout('loss126.csv',[r for r in loss if r['loss']])
    groups=[]
    for field in ('rgb_quartile','rgb_support_bin','rgb_strong','tof_unknown_bin','tof_support_bin','tof_empty_proxy','tof_peak_bin','rgb_band_position','query_xy','evidence_group','visit_id'):
      for val in sorted(set(str(r[field]) for r in loss)):
        sub=[r for r in loss if str(r[field])==val];n=sum(r['loss'] for r in sub)
        groups.append(dict(dimension=field,value=val,loss=n,RGB_supported_POS=len(sub),loss_fraction=n/len(sub),share_of_126=n/126))
    strong=[r for r in loss if r['rgb_strong']];nstrong=sum(r['loss'] for r in strong)
    return loss,groups,dict(total_loss=126,RGB_supported_POS=len(loss),margin_quartile_edges=quart.tolist(),strong_definition='upper half RGB raw margin among original RGB-supported native middle POS and >=16 inside rays',strong_losses=nstrong,strong_den=len(strong),other_losses=126-nstrong,other_den=len(loss)-len(strong),query_empty_proxy='Both K have >=16 finite rays; mean inside and before fractions zero and after positive => finite_all_after. This covers observed rays only, not full query emptiness.',peak_scope='K-mean maximum over all 8x8 frame zones, not spatial query peak; query-local support is separate.',distance_scope='Median RGB predicted inside-ray axial Z, not independent physical target range')

def plot(curves,fixed):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=False)
    colors=dict(zip(METHODS,('#c93636','#dc8c34','#8c60b5','#2676bc','#777777','#299a60','#62bdb7')))
    for i,arm in enumerate(ARMS):
      for b in range(3):
        ax=axes[i,b]
        for m in METHODS:
          rows=[r for r in curves if r['arm']==arm and r['band']==b and r['method']==m]
          n=rows[0]['FREE'];p=rows[0]['POS'];rows=[r for r in rows if not n or r['F']/n<=.105]
          ax.step([100*r['F']/n if n else r['F'] for r in rows],[100*r['W']/p if p else 0 for r in rows],where='post',label=m,color=colors[m],linewidth=1.5,marker='o' if len(rows)==1 else None,markersize=4)
        ax.set_title(arm+'\n'+BANDS[b]+f' POS={p}, FREE={n}',fontsize=9);ax.grid(alpha=.25)
        ax.set_xlabel('FREE support (%)' if n else 'FREE count (rate N/E)');ax.set_ylabel('Witness / POS (%)');ax.set_ylim(0,102)
        if n:ax.set_xlim(0,10)
        else:ax.set_xlim(-.5,.5);ax.set_xticks([0]);ax.text(.03,.05,'FREE denominator = 0: rate N/E',transform=ax.transAxes,fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.968),ncol=7)
    fig.suptitle('POSTHOC eval-label oracle: whole ties, no deployment threshold',y=.997,fontsize=12);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'matched_free_curves.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
