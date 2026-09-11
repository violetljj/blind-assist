"""Independent scalar score after sealed MZ76 run; no model or RGB loading."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from mz58_score import metrics,scalar_metrics
from mz74_score import candidate,exchange
from tof_return_sensitivity import constrain
from tof_model_adapter import legacy_batch_inputs

PROFILES=('IDEAL','CLOSEST_REPORTED_PROXY','FARTHEST_REPORTED_PROXY')
HEADS=('MZ70/DIVERSE','MZ76/BASE','MZ76/RELIABILITY')
METHODS=('MZ37','OLD_NEG/UNION')+tuple(h+'/'+kind for h in HEADS for kind in ('candidate','UNION'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def exact(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def score(root,task):
    inputs={}
    def bind(p,h=None):
        p=Path(p).resolve();v=sha(p);assert h is None or v==h,str(p);inputs[str(p)]=v;return p
    go=read(bind(task/'score-go-v1.json'));assert go['status']=='ROOT_SCORE_GO' and go['run_exit_code']==0 and go['scorer_sha256']==sha(__file__)
    rp=task/'run-v1';run=read(bind(rp/'receipt.json',go['run_receipt_sha256']));assert run['status']=='PASS'
    assert run['training_steps_per_arm']==256 and run['new_cutoffs']==0 and run['evaluation_frames']==2048 and run['evaluation_profiles']==3 and run['initial_parity_frames']==32
    out=task/'score-v1';out.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        bind(__file__)
        for name in ['mz58_score.py','mz74_score.py','tof_model_adapter.py','tof_return_sensitivity.py']:bind(Path(__file__).with_name(name))
        for p,h in run['inputs'].items():bind(p,h)
        for name,h in run['outputs'].items():bind(rp/name,h)
        sel=read(bind(task/'selection.json'));refs={};archives={}
        for key,name in [('70','mz70-diverse-learning-20260911'),('74','mz74-return-survival-20260911'),('75','mz75-return-slot-control-20260911')]:
            p=task.parent/name/'run-v1';r=read(bind(p/'receipt.json'));refs[key]=(p,r);archives[key]=np.load(bind(p/'predictions.npz',r['outputs']['predictions.npz']))
        cut=np.load(bind(refs['70'][0]/'DIVERSE-cutoff.npy',refs['70'][1]['outputs']['DIVERSE-cutoff.npy']))
        result={};pairs={};audit=dict(candidate_scalars=0,metric_bits=0,baseline_tables=0,packet_scalars=0,new_native_winner_attribution=False)
        try:
            with np.load(rp/'predictions.npz') as new:
                for source,name in [('mz61','mz61-geometry-source-20260911'),('mz67','mz67-topology-source-20260911')]:
                    base=task.parent/name;idx=read(bind(base/'source-index.json'));ref=idx['combined']['evaluator.npz'];ii=np.asarray(sel['sources'][source]['held_ids'],np.int64)
                    exact(ii,new[source+'/original_indices']);ids=new[source+'/frame_ids'];np.testing.assert_array_equal(ids,sel['sources'][source]['held_frame_ids']);assert len(ii)==1024
                    with np.load(bind(base/ref['path'],ref['sha256'])) as z:
                        exact(z['frame_ids'][ii],ids);truth=z['truth'][ii];known=z['known'][ii]
                    original=archives['70'];rr=original[source+'/IDEAL/ranges'][ii];vv=original[source+'/IDEAL/valid'][ii]
                    result[source]={};pairs[source]={}
                    for profile in PROFILES:
                        p=source+'/'+profile+'/';obs=dict(ranges=rr,valid=vv) if profile=='IDEAL' else constrain(rr,vv,profile)[0];obs,_=legacy_batch_inputs(obs['ranges'],obs['valid'])
                        exact(obs['ranges'],new[p+'ranges']);exact(obs['valid'],new[p+'valid']);audit['packet_scalars']+=obs['ranges'].size
                        for head in HEADS:
                            calc=candidate(new[p+head+'/raw'],new[p+head+'/support'],new[p+'MZ37'],cut)
                            np.testing.assert_array_equal(calc,new[p+head+'/candidate']);audit['candidate_scalars']+=calc.size
                            np.testing.assert_array_equal(np.maximum(new[p+'OLD_NEG/UNION'],calc),new[p+head+'/UNION'])
                        key='70' if profile=='IDEAL' else '74' if profile.startswith('CLOSEST') else '75';prior=archives[key]
                        previous=source+'/'+('FAR_PACKED_SLOT0' if key=='75' else profile)+'/'
                        rows=np.arange(1024) if key=='75' else ii
                        if key=='75':exact(prior[source+'/frame_ids'],ids)
                        for method in ('MZ37','OLD_NEG/UNION','MZ70/DIVERSE/candidate','MZ70/DIVERSE/UNION'):
                            value=new[p+method];expected=prior[previous+method][rows]
                            np.testing.assert_allclose(value,expected,atol=2e-5,rtol=1e-6);np.testing.assert_array_equal(value>=0,expected>=0);audit['baseline_tables']+=1
                        values={m:new[p+m] for m in METHODS};table={m:metrics(v,truth,known) for m,v in values.items()}
                        for m,v in values.items():assert table[m]==scalar_metrics(v,truth,known);audit['metric_bits']+=v.size
                        result[source][profile]=table;pairs[source][profile]={}
                        for after,before in [('MZ76/RELIABILITY','MZ76/BASE'),('MZ76/RELIABILITY','MZ70/DIVERSE'),('MZ76/BASE','MZ70/DIVERSE')]:
                            tag=after+'_vs_'+before;pair=exchange(values[after+'/UNION'],values[before+'/UNION'],truth,known,ids,np.arange(1024))
                            gain=(values[after+'/UNION']>=0)&(values[before+'/UNION']<0)&truth&known
                            pair['gain_ALREADY_OLD_NEG_positive']=(gain&(values['OLD_NEG/UNION']>=0)).sum(0).tolist()
                            pairs[source][profile][tag]=pair
        finally:
            for archive in archives.values():archive.close()
        write(out/'result.json',dict(status='PASS',query_order=['BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR'],sources=result,interpretation='Consumed two-source continuation with fixed cutoffs. Neighbor geometry is not hardware quality; no model promotion.'))
        write(out/'paired-events.json',pairs);write(out/'audit.json',dict(status='PASS',**audit))
        lines=['# MZ76 fixed-cut continuation results','','Each source:1,024 HELD frames,1,024 positive query bits and3,072 negative bits if all known. Totals count query events.','','| Source | Profile | Frozen TP/FP | BASE TP/FP | RELIABILITY TP/FP |','|---|---|---:|---:|---:|']
        for source,conditions in result.items():
            for profile,table in conditions.items():
                cells=['%d/%d'%(sum(table[h+'/UNION']['tp']),sum(table[h+'/UNION']['fp'])) for h in HEADS];lines.append('| '+source+' | '+profile+' | '+' | '.join(cells)+' |')
        lines+=['','All per-query/UNKNOWN tables and exact bidirectional events are retained; no new cutoff, native winner claim or model promotion.']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        for p,h in inputs.items():assert sha(p)==h,p
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},seconds=time.perf_counter()-start,model_inferences=0,new_cutoffs=0))
        print('SCORE PASS',time.perf_counter()-start)
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs));raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',type=Path,required=True);args=parser.parse_args();score(args.root.resolve(),args.task.resolve())
