"""V2 frozen RGB predictions and observation-only v1 query features.

No evaluator/reference file is opened. New eval observations are built before
calibration; fusion model inference belongs to the once-only evaluator.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from sync_fusion_features_v1 import FEATURES, FULL_FEATURES, SOURCES, BANDS, MISSING_MARGIN, features_for_depth, quality
from sync_rgb_tof_dataset_v1 import peak_readout
from rgb_body_query_reference_eval import rays, ray_interval

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,o): Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n',encoding='utf8')

def protocol(a):
    assert sha(a.protocol)==a.protocol_sha256
    relative=a.protocol.resolve().relative_to(a.repo.resolve()).as_posix()
    committed=subprocess.check_output(['git','show',f'{a.protocol_commit}:{relative}'],cwd=a.repo)
    assert hashlib.sha256(committed).hexdigest()==a.protocol_sha256, 'Protocol was not committed before new observations'
    p=load(a.protocol)
    assert tuple(p['features']['names'])==FEATURES
    a.root.mkdir(parents=True,exist_ok=True)
    code_seal=a.root/'observation_code_seal.json'
    paths=[Path(__file__),Path(__file__).with_name('sync_fusion_features_v1.py'),Path(__file__).with_name('sync_rgb_tof_v1_rgb_infer.py'),Path(__file__).with_name('sync_rgb_tof_dataset_v1.py'),Path(__file__).with_name('rgb_depth_backbone_adapters.py'),Path(__file__).with_name('rgb_body_query_query_calibration_probe.py')]
    identity=dict(protocol_sha256=a.protocol_sha256,protocol_commit=a.protocol_commit,files=[dict(path=str(x.resolve()),sha256=sha(x)) for x in paths])
    if code_seal.exists():assert load(code_seal)==identity, 'Observation recipe changed after first execution'
    else:save(code_seal,identity)
    return p

def infer(a):
    protocol(a)
    import sync_rgb_tof_v1_rgb_infer as frozen
    public=a.root/'public_roster.json';roster=load(public)
    admission=load(a.root/'admission_seal.json')
    assert sha(public)==admission['public_roster_sha256'], 'Final data admission observer hash differs'
    assert admission['plan_sha256']==a.protocol_sha256
    assert all(r['role'] in ('cal','eval') for r in roster['rows'])
    assert len(roster['rows'])==576, 'Wait for the final 18-visit observer roster'
    visits={str(r['visit_id']) for r in roster['rows']};assert len(visits)==18
    assert visits=={str(v) for v in admission['accepted_visit_ids']}
    assert len({str(r['visit_id']) for r in roster['rows'] if r['role']=='cal'})==6
    assert len({str(r['visit_id']) for r in roster['rows'] if r['role']=='eval'})==12
    assert all(sum(str(r['visit_id'])==v for r in roster['rows'])==32 for v in visits)
    # The v2 public roster adds observer timing/capture metadata. The unchanged
    # v1 inference contract receives its exact permitted public projection.
    from rgb_body_query_residual_target import PUBLIC
    permitted=PUBLIC|{'role','cohort','visit_id','window_id','timestamp_s','frame_id','source_id'}
    infer_roster=a.root/'rgb_public_roster.json'
    projected={'rows':[{k:v for k,v in r.items() if k in permitted} for r in roster['rows']]}
    if infer_roster.exists():assert load(infer_roster)==projected
    else:save(infer_roster,projected)
    seal_path=a.root/'rgb_observation_seal.json'
    seal=dict(public_roster_sha256=sha(infer_roster),original_public_roster_sha256=sha(public),accepted_visit_ids=sorted({str(r['visit_id']) for r in roster['rows']}),
              protocol_sha256=a.protocol_sha256,protocol_commit=a.protocol_commit,reference_read=False)
    if seal_path.exists():assert load(seal_path)==seal
    else:save(seal_path,seal)
    args=argparse.Namespace(repo=a.repo,runroot=a.root/'rgb_inference',roster=infer_roster,plan_seal=seal_path,budget_s=a.budget_s)
    return frozen.run(args)

def build_features(a):
    started=time.perf_counter();protocol(a)
    output_manifest=a.root/'feature_manifest.json'
    if output_manifest.exists():raise FileExistsError(output_manifest)
    inputs=[];groups={};admission=[]
    def record(p):inputs.append(dict(path=str(Path(p).resolve()),sha256=sha(p)))
    public_path=a.root/'public_roster.json';pub=load(public_path);queries=pub['queries'];public={r['source_id']:r for r in pub['rows']};assert len(queries)==27
    synth_path=a.root/'synthesis_manifest.json';synth=load(synth_path)
    seal_path=a.root/'frozen_readout_seal.json';seal=load(seal_path)
    assert seal['tof_peak_snr_min']==3 and seal['tof_zone_coverage_min']==.75 and seal['query_pixel_support_min']==16
    gate_path=a.root/'input_coverage_gate.csv'
    with gate_path.open(encoding='utf8',newline='') as f:
        allowed={r['frame_id'] for r in csv.DictReader(f) if r['arm']=='faro_rho030_ambient1' and r['K']=='0' and int(r['joint_pass_zones'])>=52}
    for p in (a.protocol,public_path,synth_path,seal_path,gate_path):record(p)
    centers=np.array([(np.asarray(q['low'])+q['high'])/2 for q in queries]);axes=[sorted(set(centers[:,j])) for j in range(3)]
    predictions={}
    for model in ('dav2','unidepth'):
        indexed={}
        for relative in ('predictions.json','sealed_eval/predictions.json'):
            p=a.root/'rgb_inference'/model/relative;record(p)
            assert load(p)['status']=='COMPLETE', 'Incomplete RGB observations cannot be silently selected'
            for r in load(p)['rows']:indexed[r['source_id']]=r
        predictions[model]=indexed
    for frame in synth['frames']:
        if time.perf_counter()-started>=a.budget_s:raise TimeoutError('Feature command-wall cap')
        role=frame['role'];assert role in ('cal','eval');r=public[frame['source_id']];K=np.asarray(r['depth_K']);shape=tuple(r['depth_shape']);rx,ry=rays(K,shape);intervals=[ray_interval(rx,ry,q) for q in queries]
        rgb={}
        for model in ('dav2','unidepth'):
            prediction=predictions[model][frame['source_id']];record(prediction['path'])
            with np.load(prediction['path'],allow_pickle=False) as f:z=f['depth']
            assert z.shape==shape;rgb[model]=[features_for_depth(z,*interval) for interval in intervals]
        for source in SOURCES:
            admit=source!='faro_rho030_ambient1' or frame['frame_id'] in allowed
            admission.append(dict(source=source,role=role,visit_id=frame['visit_id'],frame_id=frame['frame_id'],admitted=admit))
            if not admit:continue
            observations=[];qualities=[]
            for repeat in (0,1):
                obs=next(x for x in frame['arms'] if x['arm']==source and x['repeat']==repeat);record(obs['path'])
                with np.load(obs['path'],allow_pickle=False) as f:
                    z=peak_readout(f['hist'],f['background'],f['coverage'],f['grid_K'],shape,f['grid_pose'],K,shape,f['rgb_pose'],seal)[0] if np.isfinite(f['grid_pose']).all() else np.full(shape,np.inf)
                    qualities.append(quality(f['hist'],f['background'],f['coverage']))
                observations.append([features_for_depth(z,*interval) for interval in intervals])
            for j,q in enumerate(queries):
                score0,v0=observations[0][j];score1,v1=observations[1][j];both=np.isfinite(score0) and np.isfinite(score1)
                tof=(score0+score1)/2 if both else -np.inf;tv=(v0+v1)/2;tv[0]=tof if both else MISSING_MARGIN;tv[1]=float(not both)
                validK=int(np.isfinite(score0))+int(np.isfinite(score1));dav,dv=rgb['dav2'][j];uni,uv=rgb['unidepth'][j]
                band=0 if q['high'][2]<=.8 else 1 if q['high'][2]<=1.5 else 2;rgbscore=uni if band==0 else dav;domain=int(intervals[j][2].sum())
                vector=np.concatenate((tv,[validK,abs(score0-score1) if both else 0.],np.mean(qualities,axis=0),dv,uv,[band],[axes[axis].index(centers[j,axis]) for axis in range(3)],[domain,domain/np.prod(shape)]))
                assert len(vector)==len(FULL_FEATURES) and np.isfinite(vector).all()
                vector=vector[[FULL_FEATURES.index(n) for n in FEATURES]];cols=[i for i,n in enumerate(FEATURES) if n.endswith('_m')];vector[cols]=np.clip(vector[cols],-10,10)
                groups.setdefault((source,role),[]).append(dict(X=vector,frame_id=frame['source_id'],sensor_frame_id=frame['frame_id'],source_id=frame['source_id'],query_id=q['name'],query_index=j,visit_id=str(frame['visit_id']),capture=str(frame['capture']),band=BANDS[band],band_index=band,tof_score=tof,rgb_score=rgbscore,or_score=max(tof,rgbscore),and_score=min(tof,rgbscore)))
        save(a.root/'feature_progress.json',dict(last_frame=frame['frame_id'],elapsed_s=time.perf_counter()-started,rows=sum(map(len,groups.values()))))
    assert len(synth['frames'])==len(public), 'Every public frame must remain in the ToF sequence'
    outputs=[]
    for source in SOURCES:
        folder=a.root/'features'/source;folder.mkdir(parents=True,exist_ok=True)
        # Previously consumed eval4 is now training. No old cal visit is used.
        prior=[]
        for oldrole in ('train','eval'):
            p=a.old_fusion/'features'/source/f'{oldrole}.npz';record(p)
            with np.load(p,allow_pickle=False) as f:
                assert tuple(f['feature_names'])==FEATURES;prior.append({k:f[k].copy() for k in f.files})
        train={k:(np.asarray(FEATURES) if k=='feature_names' else np.concatenate([x[k] for x in prior],axis=0)) for k in prior[0]}
        assert len(set(train['visit_id']))<=10
        for role in ('train','cal','eval'):
            path=folder/f'{role}.npz'
            if path.exists():raise FileExistsError(path)
            if role=='train':arrays=train
            else:
                rows=groups.get((source,role),[]);arrays={'X':np.stack([r['X'] for r in rows]) if rows else np.empty((0,len(FEATURES))),'feature_names':np.asarray(FEATURES)}
                for key in ('frame_id','sensor_frame_id','source_id','query_id','visit_id','capture','band'):arrays[key]=np.asarray([r[key] for r in rows],dtype=str)
                for key in ('query_index','band_index'):arrays[key]=np.asarray([r[key] for r in rows],np.int16)
                for key in ('tof_score','rgb_score','or_score','and_score'):arrays[key]=np.asarray([r[key] for r in rows],np.float64)
            np.savez_compressed(path,**arrays)
            outputs.append(dict(source=source,role=role,path=str(path.resolve()),sha256=sha(path),rows=len(arrays['X']),frames=len(set(arrays['frame_id'])),visits=sorted(set(arrays['visit_id']))))
    save(a.root/'feature_inputs.json',inputs);save(a.root/'feature_admission.json',dict(rule='FARO fixed K0 joint_pass_zones>=52; native all frames',rows=admission))
    save(output_manifest,dict(status='COMPLETE',outputs=outputs,feature_names=FEATURES,dimension=20,labels_read=False,reference_paths_read=False,K2_rows_independent=False,observation_only=True,protocol_sha256=a.protocol_sha256,protocol_commit=a.protocol_commit,seconds=time.perf_counter()-started))
    print(json.dumps(dict(seconds=time.perf_counter()-started,outputs=outputs)),flush=True)
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('infer','features'),required=True)
    for name in ('root','repo','protocol','old-fusion'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);p.add_argument('--protocol-commit',required=True);p.add_argument('--budget-s',type=float,required=True)
    args=p.parse_args();raise SystemExit(0 if (infer(args) if args.mode=='infer' else build_features(args)) else 1)
