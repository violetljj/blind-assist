"""Cache-only conditional vertical-clear signed raw-bin observation diagnostic."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('E:/linnan/linnan');WORK=ROOT/'artifacts.local/work';OUT=WORK/'cnh-bar-vertical-evidence-dev-20261009/observation'
SHAPE=WORK/'cnh-aligned-shapes-dev-20261008';CACHE=WORK/'cnh-bar-cached-diagnostic-dev-20261008';HERE=ROOT/'research/active/dtr-r0/nearfield'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf8')
def base_background(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_BG' for t in n.targets))
    return eval(compile(ast.Expression(node),str(path),'eval'),{'__builtins__':{},'dict':dict})

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve observation PLAN')
    start=time.monotonic();inputs=[SHAPE/'PLAN.json',SHAPE/'physical.npz',SHAPE/'geometry.npz',SHAPE/'source_snapshot.py',CACHE/'PLAN.json',CACHE/'source.py',CACHE/'background_reference.npz']
    plan=dict(task='CNH_BAR_VERTICAL_CLEAR_OBSERVATION_DEV_20261009',CPU_cumulative_wall_cap_seconds=180,
        scope='All24 originalverticalclear scenes, all16 rawframes, all13 output windows f3..15. Oracle conditional same-background contrast only; no new noise, inference, GPU, training or policy feature/gate.',
        cache='Present mean from original shape physical.expectation. Absent mean from existing originalBASE_BG16pose background_reference. Verify retained BASE_BG, frozen compute source, nominalpose and exactambient identity before reuse.',
        delta='Unquantized coarse signed histogram mean contrast mu_present−mu_absent. Positive/negative mean-bin changes can combine target return and occluded background within one coarsebin; they are not separately observed target/background photons.',
        variance='V=.5*(mu_present+mu_absent)+16ambient perrawbin. d2=sum(delta²/V) is conditional Gaussian variance-standardized observation proxy, not exactdiscrimination or an information-loss chain.',
        zones='Original sensor8×8 zones, ordering [rawframe,y,x,radial16]; slope boundaries uniform over±tan22.5deg. Radial16bins width8*.0375348m. Originalbins retained; positive/negative split before any angular/radial pooling. Marginaltables are descriptive, not additive acrossalternativegroupings.',
        windows='Each outputend f3..15 includes max(0,end−7)..end inclusive, exactsamepast8 asrepresentation. Include early4/5/6/7 histories. Ideal/query±3 observation identicalbecause truephysicalsensor/hist unchanged.',
        stop='Complete scopedcache-only diagnostic and focusedcheck or cumulative180s; preserve failures, no extra renders if cacheverified.',
        inputs_sha256={str(p.relative_to(ROOT)).replace(chr(92),'/'):sha(p) for p in inputs},source_sha256=sha(__file__))
    save(OUT/'PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());save(OUT/'prepare_result.json',dict(seconds=time.monotonic()-start))

def run():
    if (OUT/'mean_delta.npz').exists():raise FileExistsError('Preserve observation payload')
    start=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());prior=json.loads((OUT/'prepare_result.json').read_text())['seconds']
    old=json.loads((SHAPE/'PLAN.json').read_text());cached=json.loads((CACHE/'PLAN.json').read_text());rows=old['scene_rows'];ids=np.array([r['id'] for r in rows if r['family']=='vertical' and r['placement']=='clear'])
    assert len(ids)==24 and ids.tolist()==list(range(368,380))+list(range(412,424))
    frozenbg=base_background(SHAPE/'source_snapshot.py');assert all(rows[i]['background']==frozenbg for i in ids)
    code=(CACHE/'source.py').read_text();assert 'bg=R.expected(dict(poses=sensor,boxes=S.BASE_BG))' in code and 'sensor,_=B.poses(-10.)' in code
    assert sha(CACHE/'source.py')==cached['source_sha256']['research/active/dtr-r0/nearfield/cnh_bar_cached_diagnostic_dev.py']
    assert sha(HERE/'cnh_aligned_boundary_dev.py')==cached['source_sha256']['research/active/dtr-r0/nearfield/cnh_aligned_boundary_dev.py']
    assert sha(SHAPE/'physical.npz')==cached['input_sha256']['artifacts.local/work/cnh-aligned-shapes-dev-20261008/physical.npz']
    with np.load(SHAPE/'geometry.npz') as z:sensor=z['sensor'];query=z['public_query'];cat=z['category'];assert (cat[ids]=='clear').all()
    a=np.deg2rad(-10.);c,s=np.cos(a),np.sin(a);rotation=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
    nominal=np.repeat(np.eye(4)[None],16,axis=0);nominal[:,:3,:3]=rotation;nominal[:,2,3]=np.arange(16)*.16-2.4
    expectedquery=nominal.copy();expectedquery[:,:3,3]=0
    np.testing.assert_allclose(sensor,nominal,rtol=0,atol=1e-15);np.testing.assert_allclose(query,expectedquery,rtol=0,atol=1e-15)
    with np.load(SHAPE/'physical.npz') as z:mu=z['expectation'][ids];ambient=z['ambient']
    with np.load(CACHE/'background_reference.npz') as z:bg=z['expectation'];np.testing.assert_array_equal(ambient,z['ambient'])
    assert mu.shape==(24,16,8,8,16) and bg.shape==(16,8,8,16) and ambient.shape==(16,8,8)
    delta=mu-bg[None];variance=.5*(mu+bg[None])+16*ambient[None,...,None];assert (variance>0).all()
    info=delta**2/variance;positive=np.maximum(delta,0);negative=np.maximum(-delta,0);pinfo=np.where(delta>0,info,0);ninfo=np.where(delta<0,info,0)
    np.savez_compressed(OUT/'mean_delta.npz',scene_ids=ids,mu_present=mu,mu_absent=bg,delta=delta,variance=variance,ambient=ambient,sensor=sensor,public_query=query,
        radial_edges=np.arange(17)*8*.0375348,angular_slope_edges=np.linspace(-np.tan(np.pi/8),np.tan(np.pi/8),9),
        positive_delta=positive,negative_delta_abs=negative,positive_d2=pinfo,negative_d2=ninfo)
    windows=[];ranges=[];angles=[];frameangles=[];frameranges=[]
    def totals(d,p,n,pi,ni):return dict(signed_mean_difference=float(d.sum()),positive_mean_difference=float(p.sum()),negative_mean_difference_abs=float(n.sum()),positive_d2=float(pi.sum()),negative_d2=float(ni.sum()))
    for j,i in enumerate(ids):
        row=rows[i]
        for f in range(16):
            for y in range(8):
                for x in range(8):frameangles.append(dict(scene=int(i),frame=f,zone_y=y,zone_x=x,**totals(delta[j,f,y,x],positive[j,f,y,x],negative[j,f,y,x],pinfo[j,f,y,x],ninfo[j,f,y,x])))
            for b in range(16):frameranges.append(dict(scene=int(i),frame=f,radial_bin=b,range_low_m=b*8*.0375348,range_high_m=(b+1)*8*.0375348,**totals(delta[j,f,:,:,b],positive[j,f,:,:,b],negative[j,f,:,:,b],pinfo[j,f,:,:,b],ninfo[j,f,:,:,b])))
        for end in range(3,16):
            begin=max(0,end-7);sl=slice(begin,end+1);d=delta[j,sl];p=positive[j,sl];n=negative[j,sl];pi=pinfo[j,sl];ni=ninfo[j,sl]
            windows.append(dict(scene=int(i),end_frame=end,start_frame=begin,histories=end-begin+1,side=row['side'],height_group=row['group'],variant=row['variant'],rho=row['rho'],**totals(d,p,n,pi,ni)))
            for y in range(8):
                for x in range(8):angles.append(dict(scene=int(i),end_frame=end,start_frame=begin,zone_y=y,zone_x=x,**totals(d[:,y,x],p[:,y,x],n[:,y,x],pi[:,y,x],ni[:,y,x])))
            for b in range(16):ranges.append(dict(scene=int(i),end_frame=end,start_frame=begin,radial_bin=b,range_low_m=b*8*.0375348,range_high_m=(b+1)*8*.0375348,**totals(d[:,:,:,b],p[:,:,:,b],n[:,:,:,b],pi[:,:,:,b],ni[:,:,:,b])))
        if prior+time.monotonic()-start>180:raise TimeoutError('cumulative180s')
    for filename,data in (('past8.csv',windows),('past8_angles.csv',angles),('past8_ranges.csv',ranges),('frame_angles.csv',frameangles),('frame_ranges.csv',frameranges)):
        with (OUT/filename).open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summaries=[]
    for end in (3,12,13,15):
        sl=slice(max(0,end-7),end+1)
        for group in ('HEAD','BODY','BOTH'):
            ix=np.array([rows[i]['group']==group for i in ids]);summaries.append(dict(end_frame=end,group=group,scenes=int(ix.sum()),
                positive_d2_sum=float(pinfo[ix,sl].sum()),negative_d2_sum=float(ninfo[ix,sl].sum()),conditional_d_median=float(np.median(np.sqrt(info[ix,sl].sum((1,2,3,4))))),
                positive_range_bin_d2=pinfo[ix,sl].sum((0,1,2,3)).tolist(),negative_range_bin_d2=ninfo[ix,sl].sum((0,1,2,3)).tolist(),
                positive_angle_zone_d2=pinfo[ix,sl].sum((0,1,4)).tolist(),negative_angle_zone_d2=ninfo[ix,sl].sum((0,1,4)).tolist()))
    save(OUT/'result.json',dict(status='PASS',seconds=time.monotonic()-start,prior_prepare_seconds=prior,cumulative_seconds=prior+time.monotonic()-start,
        scenes=ids.tolist(),all_scene_count=24,rawframes=16,output_windows=13,background_contexts=1,new_expectations_rendered=0,new_draws=0,new_inference=0,
        identities=dict(frozen_BASE_BG=True,cached_background_computation_source=True,physical_payload_identity=True,nominal_true_sensor_and_public_query=True,ambient_BITWISE=True),
        row_counts=dict(past8=len(windows),past8_angles=len(angles),past8_ranges=len(ranges),frame_angles=len(frameangles),frame_ranges=len(frameranges)),summary=summaries,
        limitations='Oracle fixed-background mean contrast of24 consumedDevelopment clear proxies. Coarsebin sign is net mean change, not separate photon identity. Queryrotation leaves observation unchanged; projection/readout differences need separate measurements. Conditional d2 proxy is not validated classifier/noise-generalization or monotonic information chain.'))
    print('COMPLETE24clear vertical signed bins, CPU seconds',prior+time.monotonic()-start,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
