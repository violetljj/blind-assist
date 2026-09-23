"""Independent rendered cuboid labels, saved readout and event metric audit."""
import os,sys,json
from pathlib import Path
import numpy as np
from local_rescue_audit import read,write,sha,measure
REPO=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
ROOT=REPO/'artifacts.local/evidence/ba-local-rescue-fresh-audit-20260923';E=ROOT.parent
S=E/'ba-local-rescue-fresh-20260923'
def stage(s):return S.with_name(S.name+'-'+s)
def main():
    assert read(Path(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL']))['state']=='running'
    for name,digest in read(ROOT/'plan/source-seal.json')['files'].items():
        assert sha(REPO/name)==digest,name
    checks=0
    spec=read(S/'plan/spec.json');protocol=read(S/'plan/protocol.json')
    launch=read(stage('capture')/'launch-receipt.json');receipt=read(stage('capture')/'receipt.json')
    gatepath=E/'ba-local-rescue-gate-20260923-run/gate.json';gate=read(gatepath)
    assert launch['spec_sha256']==receipt['spec_sha256']==protocol['spec_sha256']==sha(S/'plan/spec.json')
    assert launch['protocol_sha256']==receipt['protocol_sha256']==sha(S/'plan/protocol.json')
    assert protocol['input_hashes'][gatepath.relative_to(REPO).as_posix()]==sha(gatepath)
    assert receipt['source_unchanged'] and receipt['task_actors_released'] and read(stage('capture')/'process-release.json')['released']
    for f,h in protocol['code_hashes'].items():assert sha(HERE/f)==h;checks+=1
    for f,h in protocol['input_hashes'].items():assert sha(REPO/f)==h;checks+=1
    prep=stage('prepared');pred=stage('predictions');gated=stage('gated');evaluated=stage('evaluated')
    manifest=read(prep/'materialization.json')
    for f,h in manifest['hashes'].items():assert sha(prep/f)==h;checks+=1
    ps=read(pred/'prediction-seal.json');gs=read(gated/'prediction-seal.json')
    assert ps['evaluation_labels_opened'] is False and gs['labels_opened'] is False
    assert gs['gate_sha256']==sha(gatepath) and gs['decisions_sha256']==sha(gated/'decisions.json')
    assert gs['input_prediction_seal_sha256']==sha(pred/'prediction-seal.json')
    assert gs['expected_labels_sha256']==sha(prep/'labels/evaluation.npz')
    for f,h in ps['hashes'].items():assert sha(pred/f)==h;checks+=1
    lab=np.load(prep/'labels/evaluation.npz');geo=read(stage('capture')/'evaluator/geometry.json')
    rows=read(evaluated/'rows.json');ids=read(pred/'identities.json');prob=np.load(pred/'probabilities.npz')['local']
    features=np.load(pred/'features.npz')['local'];baseline=read(pred/'baseline.json')
    decisions=read(gated/'decisions.json');report=read(evaluated/'report.json')
    edges=np.array([.3,.75,1.25,1.75,2.25,2.75,3],np.float32)
    queries=np.array([[x-.3,x+.3,lo,hi] for lo,hi in ((.42,.9),(-.2,.42)) for x in (-.3,0,.3)],np.float32)
    assert len(rows)==len(ids)==len(geo)==len(spec['cases'])==576
    assert len({r['clip_id'] for r in rows})==48
    assert np.array_equal(lab['indices'],np.arange(576))
    geometry_classes=[];bounds_checked=0
    for i,(r,g,c) in enumerate(zip(rows,geo,spec['cases'],strict=True)):
        assert r['id']==g['id']==c['name']==ids[i]['id']
        camera=g['actual_camera_location_m']
        assert max(abs(camera[j]-c['camera'][axis]) for j,axis in enumerate(('x','y','z')))<=.002
        bounds=[]
        for actual,planned in zip(g['objects'],c['objects'],strict=True):
            center=actual['render_bounds_center_m'];half=actual['render_bounds_extent_m']
            assert max(abs(center[j]-planned['center_m'][j]) for j in range(3))<=.002
            assert max(abs(2*half[j]-planned['size_m'][j]) for j in range(3))<=.002
            assert actual['mesh_path'].split('.')[-1]=='Cube';bounds_checked+=1
            # Evaluate against actual camera pose; axes: lateral=worldY, down=-worldZ, forward=worldX.
            middle=[center[1]-camera[1],camera[2]-center[2],center[0]-camera[0]]
            extent=[half[1],half[2],half[0]]
            bounds.append(([middle[j]-extent[j] for j in range(3)],[middle[j]+extent[j] for j in range(3)]))
        cls=[];dist=[]
        for xl,xh,yl,yh in queries:
            near=[max(lo[2],float(edges[0])) for lo,hi in bounds if hi[0]>=xl and lo[0]<=xh and hi[1]>=yl and lo[1]<=yh and hi[2]>=edges[0] and lo[2]<=edges[-1]]
            distance=np.float32(min(near)) if near else np.float32(np.nan)
            klass=min(5,sum(distance>e for e in edges[1:])) if near else 6
            cls.append(klass);dist.append(distance)
        assert np.array_equal(cls,lab['classes'][i]),(i,cls,lab['classes'][i]);checks+=6
        assert np.allclose(dist,lab['distances'][i],atol=2e-5,equal_nan=True);checks+=6
        geometry_classes.append(cls)
        truth=bool((np.array(cls)[[1,4]]<6).any()) if lab['valid'][i,[1,4]].all() else None
        assert r['truth']==truth;checks+=1
        q=1 if prob[i,1]>=prob[i,4] else 4
        local=bool(baseline[i]['alert'] or prob[i,q]>=ps['thresholds']['local'])
        scalar=bool(baseline[i]['alert'] or (local and float(features[i,q,939])<=gate['cutoff']))
        two=bool(baseline[i]['alert'] or (prob[i,q]>=ps['thresholds']['local'] and r['frame_in_clip']>0 and max(prob[i-1,[1,4]])>=ps['thresholds']['local']))
        for arm,alert in dict(A_current=bool(baseline[i]['alert']),local=local,ambiguity_gate=scalar,two_frames=two).items():
            assert r['predictions'][arm]['alert']==alert and decisions['flags'][i][arm]==alert;checks+=1
            assert r['predictions'][arm]['unknown']==baseline[i]['unknown'];checks+=1
        assert decisions['features'][i]['possible_fraction']==float(features[i,q,939]);checks+=1
    assert lab['valid'].all()
    results={}
    for arm,saved in report['metrics']['arms'].items():
        m=measure(rows,arm)
        for k,v in m['counts'].items():assert saved['frames']['all_known'][k]==v;checks+=1
        assert m['events']==[(e['clip_id'],e['start_frame'],e['end_frame'],e['first_alert_time_s']) for e in saved['events']]
        assert m['segments']==[(e['clip_id'],e['start_frame'],e['end_frame']) for e in saved['false_alert_segments']]
        results[arm]=m
    inc=[r for r in rows if r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
    summary=dict(incremental_TP=sum(r['truth'] is True for r in inc),incremental_FP=sum(r['truth'] is False for r in inc),retained_TP=sum(r['truth'] is True and r['predictions']['ambiguity_gate']['alert'] for r in inc),removed_FP=sum(r['truth'] is False and not r['predictions']['ambiguity_gate']['alert'] for r in inc))
    pairs=zip(results['local']['events'],results['ambiguity_gate']['events'],strict=True);pairs=list(pairs)
    summary.update(lost_events=sum(a[3] is not None and b[3] is None for a,b in pairs),delayed_events=sum(a[3] is not None and b[3] is not None and b[3]>a[3]+1e-9 for a,b in pairs))
    for k,v in summary.items():assert report['summary'][k]==v;checks+=1
    # A small deterministic re-extraction checks public representation serialization and slot alignment.
    sys.path.insert(0,str(HERE));from inherit_spatial_model import extract
    rgb=np.load(prep/'observations/rgb.npy',mmap_mode='r');tof=np.load(prep/'observations/tof.npy',mmap_mode='r')
    sampled=[0,147,294,441]
    for i in sampled:assert np.array_equal(extract(rgb[i],tof[i])[:,1],features[i]);checks+=1
    from tof_corridor_calibration import score_frame,decide
    for i in range(576):
        values=np.where(tof[i,:,1]==1,tof[i,:,0]*8,np.nan);boxes=np.rint(tof[i,:,2:]*[192,256,192,256]).astype(int)
        recalculated=decide(score_frame(boxes,values),.4071309640537889)
        for k in ('alert','unknown','ambiguous','valid_zones','definite_zones'):assert recalculated[k]==baseline[i][k];checks+=1
        assert abs(recalculated['score']-baseline[i]['score'])<1e-12
    output=dict(status='PASS',checks=checks,frames=576,queries=3456,rendered_cuboids=bounds_checked,public_feature_frames_reextracted=sampled,summary=summary,arms=results,scope='New same-generator controlled instances, not independent scene or hardware validation',label_limit='Cuboid union and distances reconstructed from rendered bounds/actual camera; validity mask verified by materialization hash but native pixel attribution not independently regenerated.')
    write(ROOT.with_name(ROOT.name+'-run')/'result.json',output)
    print(json.dumps(dict(status='PASS',checks=checks,summary=summary,metrics={a:dict(**m['counts'],events=m['detected_events'],total_events=len(m['events']),segments=len(m['segments'])) for a,m in results.items()})))
if __name__=='__main__':main()

