"""Read-only frozen sim input closure and paired contract-v4 evaluation.

Only the new artifact directory is written. The original hold is already
consumed Development; neither this check nor its fixed forward selects cuts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[4]
NEAR=ROOT/'research/active/dtr-r0/nearfield'
OLD=ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
OUT=ROOT/'artifacts.local/work/local-signed-bias-contract-dev-20261011'
BIAS=ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'
if str(NEAR) not in sys.path:sys.path.insert(0,str(NEAR))

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve contract-check receipt '+str(path))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def bindings():
    paths=[OLD/'data/hold/hist.npy',OLD/'data/hold/ambient.npy',OLD/'data/hold/geometry.npz',BIAS,
        OLD/'scene_rows.json',OLD/'background_rows.json',OLD/'sealed_candidates.json',OLD/'sealed_calibration.json',
        OLD/'main_real_manifest.json',OLD/'hold_scores/no_bin0_scores.npz',OLD/'hold_grades_notifications.npz',
        OLD/'simulation_evaluation.json',NEAR/'cnh_no_bin0_dr_eval_20261011.py',NEAR/'cnh_no_bin0_dr_source_20261011.py',
        Path(__file__),Path(__file__).with_name('local_signed_bias_adapter.py')]
    manifest=read(OLD/'main_real_manifest.json')
    entries=[*manifest['m3_paths'],*manifest['ordinary_paths'],*(entry for pair in manifest['s_paths'] for entry in pair)]
    for entry in entries:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'],path;paths.append(path)
    return {str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in paths}

def compare_inputs(limit_scenes=None,check=lambda:None):
    from local_signed_bias_adapter import sim_adapter,native_evidence
    import cnh_aligned_boundary_dev as A
    *_,old_normal=A.imports()
    hist=np.load(OLD/'data/hold/hist.npy',mmap_mode='r');ambient=np.load(OLD/'data/hold/ambient.npy')
    bias=np.load(BIAS);n=len(hist) if limit_scenes is None else min(limit_scenes,len(hist))
    presence=np.empty((n,hist.shape[1],16),bool);changed_cells=0;max_abs=0.;sim_changed=0
    zero_raw_frames=0;background_frames=0;examples=[]
    for begin in range(0,n,64):
        check();raw=np.asarray(hist[begin:min(n,begin+64)]);count=int(np.prod(raw.shape[:2]))
        flat=raw.reshape(count,16,8,8,16);amb=np.broadcast_to(ambient,(count,*ambient.shape))
        old=old_normal(flat,amb)
        adapted=sim_adapter(flat,amb,bias)
        if adapted.shape!=old.shape or adapted.dtype!=np.float16:raise ValueError('Original FP16 sim contract required')
        sim_changed+=int(np.count_nonzero(adapted!=old))
        old[...,0]=0
        new,positive=native_evidence(adapted,(0,))
        oldpositive=(old>0).any(axis=(-3,-2,-1))
        if np.asarray(positive).shape!=oldpositive.shape:raise ValueError('Presence must bind each native frame')
        difference=new!=old;changed_cells+=int(difference.sum())
        max_abs=max(max_abs,float(np.max(np.abs(new.astype(np.float32)-old.astype(np.float32)))))
        presence[begin:min(n,begin+64)]=positive.reshape(-1,hist.shape[1],16)
        zero_raw_frames+=int((flat==0).all(axis=(-3,-2,-1)).sum())
        background_frames+=int((flat==bias).all(axis=(-3,-2,-1)).sum())
        if difference.any() and len(examples)<8:
            for clip,frame in np.argwhere(difference.any(axis=(-3,-2,-1)))[:8-len(examples)]:
                examples.append(dict(scene=int(begin+clip//hist.shape[1]),replica=int(clip%hist.shape[1]),frame=int(frame)))
        check()
    history=np.stack([presence[:,:,max(0,f-7):f+1].any(-1) for f in range(3,16)],-1)
    result=dict(physical_scenes=n,replicas=hist.shape[1],native_frames=int(presence.size),
        native_positive_frames=int(presence.sum()),native_nonpositive_frames=int((~presence).sum()),
        history_slots=int(history.size),history_no_positive_slots=int((~history).sum()),
        zero_raw_frames=zero_raw_frames,exact_background_frames=background_frames,
        sim_adapter_changed_cells=sim_changed,masked_input_changed_cells=changed_cells,max_abs_input_difference=max_abs,
        changed_examples=examples,inputs_identical=changed_cells==0,
        grade_gate_identical=bool(history.all()),all_cached_branches_reusable=bool(changed_cells==0 and history.all()),
        original_dtype='FP16 signed normalization; bin0 masked before every extractor',
        no_model_cut_or_seed_selection=True)
    return result,presence,history

def profile(check=lambda:None):
    # Warm imports separately so startup is not multiplied by all 1536 scenes.
    startup=time.monotonic()
    import cnh_aligned_boundary_dev as A
    A.imports()
    import local_signed_bias_adapter  # noqa: F401
    cold_import_seconds=time.monotonic()-startup
    began=time.monotonic();closure,_,_=compare_inputs(5,check)
    closure.update(seconds=time.monotonic()-began,clips=10,
        cold_import_seconds=cold_import_seconds,
        linear_estimate_full_input_seconds=(time.monotonic()-began)*1536/5,
        estimate_limit='Linear warm input-only estimate; score archive/HGB/statistics I/O extra; no forward or outcome read')
    save(OUT/'sim_profile.json',closure);print(json.dumps(closure),flush=True)

def cpu(check=lambda:None):
    import cnh_no_bin0_dr_eval_20261011 as E
    began=time.monotonic();hashes=bindings();closure,presence,history=compare_inputs(None,check)
    np.savez_compressed(OUT/'sim_presence.npz',native_positive=presence,history_positive=history)
    closure.update(bindings_sha256=hashes,seconds=time.monotonic()-began)
    save(OUT/'sim_input_closure.json',closure)
    if not closure['all_cached_branches_reusable']:
        raise RuntimeError('Input/presence changed: root must authorize fresh affected-score recomputation; no blind cache reuse')
    with np.load(OLD/'hold_scores/no_bin0_scores.npz') as saved:scores={key:saved[key].copy() for key in saved.files}
    with np.load(OLD/'hold_grades_notifications.npz') as saved:
        keys=list(saved['keys']);baseline=saved['grades'][keys.index('original_old5')].copy();cached=saved['grades'][keys.index('no_bin0_S')].copy()
    rows=read(OLD/'scene_rows.json')['hold'];cuts=read(OLD/'main_real_manifest.json')['cuts']
    with np.load(OLD/'data/hold/geometry.npz') as saved:category=saved['category']
    rawstrong=E.strong(scores,cuts)
    light=E.T.predict_s(OLD/'no_bin0',E.s_features(scores,cuts)).mean(0)
    grade=E.TM.apply_gate(rawstrong,light,cuts['s_tau'])
    grade=np.where(history[...,None],grade,0).astype(np.int8)
    np.testing.assert_array_equal(grade,cached)
    stats,notice=E.describe(grade,baseline,category,rows)
    oldstats=read(OLD/'simulation_evaluation.json');reference=oldstats['no_bin0_S'];default=oldstats['old_default_S']
    assert stats==reference,'Scores/grades identical must reproduce every saved integer and weighted cost'
    gain=sum(stats[q]['net'] for q in ('HEAD','BODY'));timely=sum(stats[q]['timely'] for q in ('HEAD','BODY'))
    reference_timely=sum(reference[q]['timely'] for q in ('HEAD','BODY'))
    fc=stats['far_clear']['notified_clips'];fcden=stats['far_clear']['denominator']
    acceptance=dict(gain=gain,gain_floor=136,gain_pass=bool(gain>=136),
        original_default_far_clear=default['far_clear']['notified_clips'],far_clear_denominator=fcden,
        far_clear_clips=fc,original_default_plus_1pp_pass=bool(fc<=default['far_clear']['notified_clips']+.01*fcden),
        timely_loss_vs_193=reference_timely-timely,timely_loss_cap=10,timely_small_change_pass=bool(reference_timely-timely<=10),
        far_clear_increment_vs_193=fc-reference['far_clear']['notified_clips'],far_clear_increment_cap=15,
        far_clear_small_change_pass=bool(fc-reference['far_clear']['notified_clips']<=15))
    np.savez_compressed(OUT/'sim_grades_notifications.npz',grade=grade,notifications=notice,category=category)
    save(OUT/'sim_evaluation.json',dict(status='COMPLETE',role='Consumed same-hold Development contract repair',
        statistics=stats,acceptance=acceptance,grades_identical=True,weighted_cost_identical=True,
        cached_score_reuse='Complete every-cell native input equality and every-window evidence-gate equality; fixed fresh forward remains required',
        cut_changes=0,training=0,protected_access=0,seconds=time.monotonic()-began,bindings_sha256=hashes))
    print(json.dumps(acceptance),flush=True)

def forward(check=lambda:None):
    import cnh_no_bin0_dr_eval_20261011 as E
    from local_signed_bias_adapter import sim_adapter,native_evidence
    closure=read(OUT/'sim_input_closure.json');assert closure['all_cached_branches_reusable']
    for path,digest in closure['bindings_sha256'].items():assert sha(ROOT/path)==digest,path
    hist=np.load(OLD/'data/hold/hist.npy',mmap_mode='r')[:8];ambient=np.load(OLD/'data/hold/ambient.npy');bias=np.load(BIAS)
    flat=hist.reshape(16,16,8,8,16)
    z,positive=native_evidence(sim_adapter(flat,np.broadcast_to(ambient,(16,*ambient.shape)),bias),(0,))
    assert positive.all()
    with np.load(OLD/'data/hold/geometry.npz') as g:sensor,query=g['sensor'],g['public_query']
    fresh=E.infer(z.reshape(8,2,16,8,8,16),sensor,query,arms=('no_bin0',),check=check)['no_bin0']
    with np.load(OLD/'hold_scores/no_bin0_scores.npz') as saved:old={key:saved[key][:,:8] if key=='ordinary_raw' else saved[key][:8] for key in fresh}
    detail={}
    for key,values in fresh.items():
        if values.dtype==bool:
            np.testing.assert_array_equal(values,old[key]);detail[key]=dict(exact=True)
        else:
            # Same FP16 feature contract, deterministic CUDA FP32 computation.
            atol=1e-4 if key in ('m3_raw','ordinary_raw') else 1e-5
            np.testing.assert_allclose(values,old[key],atol=atol,rtol=1e-5,equal_nan=True)
            finite=np.isfinite(values)&np.isfinite(old[key]);delta=np.abs(values[finite]-old[key][finite])
            detail[key]=dict(max_abs=float(delta.max()) if len(delta) else 0.,atol=atol,rtol=1e-5)
    np.savez_compressed(OUT/'sim_forward_first8.npz',**fresh)
    save(OUT/'sim_forward_receipt.json',dict(status='PASS',fixed_scenes=list(range(8)),physical_scenes=8,replicas=2,
        frame_slots=13,model_arm='main no_bin0 frozen',no_cut_selection=True,comparisons=detail,
        source_input_closure_sha256=sha(OUT/'sim_input_closure.json')))
    print(json.dumps(detail),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('profile','cpu','forward'),required=True)
    parser.add_argument('--seconds',type=float);args=parser.parse_args()
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    cap=min(args.seconds or (35 if args.stage=='forward' else 300),35 if args.stage=='forward' else 300)
    def check():
        if time.monotonic()-began>=cap-1:raise TimeoutError('Allocated sim-check command-wall budget reached')
    try:
        with threadpool_limits(limits=2):globals()[args.stage](check)
        save(OUT/(f'sim_{args.stage}_command_'+str(time.time_ns())+'.json'),dict(status='COMPLETE',seconds=time.monotonic()-began,
            stage=args.stage,budget_class='GPU' if args.stage=='forward' else 'CPU'))
    except BaseException as exc:
        save(OUT/(f'sim_{args.stage}_failure_'+str(time.time_ns())+'.json'),dict(status='FAILED',seconds=time.monotonic()-began,
            stage=args.stage,error=repr(exc),budget_class='GPU' if args.stage=='forward' else 'CPU'))
        raise
