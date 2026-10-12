"""Frozen dark-4cm bar FP16 reconstruction; cache-only, no model loading/inference.

Mean-vector probes describe coordinate rounding, not statistical separability.
Real K4 reconstruction uses original integer histograms; absent A has no hist.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

OUT = B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008/fp16'
BACKGROUND = B.ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008/background_reference.npz'
ENDS = (12, 13)


def save(name, value):
    B.save(OUT/name, value)


def inputs():
    # B.imports establishes the original frozen geometry source before imports.
    A, G, R, normal = B.imports()
    import cnh_displacement_ceiling as D
    from cnh_cvr_v2_materialize import BatchedProjector
    import torch
    return A, normal, D, BatchedProjector, torch


def selected(rows, cat):
    ids = [i for i, r in enumerate(rows) if r['family']=='horizontal' and
           r['rho']==.25 and 'thick0.04' in r['variant'] and 'contact' in cat[i]]
    pairs = {}
    for i in ids:
        r = rows[i]
        if r['placement']=='center':
            continue
        pairs[i] = next(j for j, b in enumerate(rows) if b['family']==r['family'] and
            b['placement']=='pass' and all(b[k]==r[k] for k in ('side','group','variant','rho')))
    return ids, pairs


def prepare():
    A, normal, D, Projector, torch = inputs()
    plan0 = json.loads((S.OUT/'PLAN.json').read_text(encoding='utf8'))
    with np.load(S.OUT/'geometry.npz') as z:
        ids, pairs = selected(plan0['scene_rows'], z['category'])
    assert len(ids)==28 and len(pairs)==24 and len(set(pairs.values()))==8
    paths = [S.OUT/'PLAN.json', S.OUT/'physical.npz', S.OUT/'geometry.npz', BACKGROUND, D.BIAS]
    sources = {Path(__file__), *[Path(m.__file__).resolve() for n,m in list(sys.modules.items())
                if n.startswith('cnh_') and getattr(m,'__file__',None)]}
    plan = dict(task='CNH_BAR_FP16_DIAGNOSTIC_DEV_20261008',lane='EXPLORE consumed simulation cache',
        authorization='User: directly execute same-window A/B projection comparison and FP16 reconstruction',
        scope='Frozen -10 pitch, past8 f12/f13; no sampling, training or M3 inference/model initialization',
        budget_wall_seconds=300, budget_start='First run entry; failed run seconds cumulative; programming/prepare excluded',
        stop='Complete diagnostics or cumulative cap; preserve all failures and existing payloads',
        decision_check='Measure actual coordinate rounding and deterministic A/B mean contrast retention; no alarm improvement gate or separability interpretation',
        contact_scene_ids=ids, B_pass_pairs={str(k):v for k,v in pairs.items()},
        sample_scene_ids=sorted(set(ids)|set(pairs.values())), sample_replicas=4,
        A_sample_status='NOT_RUN_NO_ABSENT_HIST: background_reference contains expected means only',
        B_center_status='NOT_EVALUABLE_NO_CENTER_PASS: 4 center contacts have no matched side pass',
        mean_probe='FP32 normalized expected histogram, then original first FP16, original A.Engine.project FP32, second FP16. Conditional deterministic coordinate contrast, not d2/d or information loss.',
        semantics='cnt depends only on poses and cannot carry A/B mean contrast. tot/lst jointly stored; per-coordinate norm/error is not covariance-whitened separability. Rounding is input-dependent, no independent uniform variance assumption. Signed-log1p is invertible in ideal reals; not assigned information loss.',
        input_sha256={B.logical_path(p):B.sha(p) for p in paths},
        source_sha256={B.logical_path(p):B.sha(p) for p in sorted(sources,key=str)})
    save('PLAN.json',plan)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',len(ids),len(pairs),len(plan['sample_scene_ids']),flush=True)


def rounding(before, after):
    before=np.asarray(before,np.float64); after=np.asarray(after,np.float64)
    err=after-before
    norm=float(np.linalg.norm(before.reshape(-1)))
    return dict(coordinates=int(before.size), changed=int(np.count_nonzero(err)),
        nonzero_before=int(np.count_nonzero(before)), zeroed=int(np.count_nonzero((before!=0)&(after==0))),
        error_max_abs=float(np.abs(err).max()), error_l2=float(np.linalg.norm(err.reshape(-1))),
        relative_l2=float(np.linalg.norm(err.reshape(-1))/norm) if norm else 0.,
        error_mean=float(err.mean()), nonfinite=int((~np.isfinite(before)).sum()+(~np.isfinite(after)).sum()))


def contrast(before1,before0,after1,after0):
    a=np.asarray(before1,np.float64)-np.asarray(before0,np.float64)
    b=np.asarray(after1,np.float64)-np.asarray(after0,np.float64)
    v=rounding(a,b)
    v.update(merged_nonzero_contrast=int(np.count_nonzero((a!=0)&(b==0))),
             contrast_nonzero_before=int(np.count_nonzero(a)),contrast_nonzero_after=int(np.count_nonzero(b)),
             contrast_norm_before=float(np.linalg.norm(a.reshape(-1))),
             contrast_norm_after=float(np.linalg.norm(b.reshape(-1))))
    return v


def csv_write(name, rows):
    with (OUT/name).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run():
    began=time.monotonic(); plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    spent=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('run_failure_*.json'))
    def check():
        if spent+time.monotonic()-began>plan['budget_wall_seconds']:
            raise TimeoutError('Cumulative FP16 diagnostic 300s wall cap')
    engine=None;torch=None
    try:
        for name, digest in {**plan['input_sha256'],**plan['source_sha256']}.items():
            assert B.sha(B.ROOT/name)==digest, ('changed frozen input/source',name)
        A, normal, D, Projector, torch = inputs()
        A.OUT=OUT/'runtime'; A.setup_gpu()
        dll=Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
        dll_handle=os.add_dll_directory(str(dll)) if os.name=='nt' and dll.is_dir() else None
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available()
        engine=object.__new__(A.Engine);engine.torch=torch;engine.projector=Projector()
        rows=json.loads((S.OUT/'PLAN.json').read_text(encoding='utf8'))['scene_rows']
        ids=plan['contact_scene_ids']; pairs={int(k):v for k,v in plan['B_pass_pairs'].items()}
        sample_ids=plan['sample_scene_ids']; sample_index={i:j for j,i in enumerate(sample_ids)}
        with np.load(S.OUT/'physical.npz') as z:
            hist=z['hist'][sample_ids]; mu=z['expectation'][sample_ids]; ambient=z['ambient']
        with np.load(S.OUT/'geometry.npz') as z:
            sensor=z['sensor'];query=z['public_query']
        with np.load(BACKGROUND) as z:
            bg=z['expectation']; np.testing.assert_array_equal(z['ambient'],ambient)
        bias=np.load(D.BIAS).astype(np.float32)
        den=np.sqrt(np.maximum(16*np.asarray(ambient,np.float32)[...,None]+np.maximum(bias,0),1e-9))
        # These operation orders exactly match normalized_z, including FP32 cast.
        def z32(x):return (np.asarray(x,np.float32)-bias)/den
        sample32=z32(hist);sample16=sample32.astype(np.float16)
        parity=normal(hist.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(len(sample_ids)*4,*ambient.shape)))
        np.testing.assert_array_equal(sample16.reshape(parity.shape),parity)
        mean32=z32(np.concatenate((mu,bg[None])));mean16=mean32.astype(np.float16)
        # Archive full pre/post first-rounding values; all projected pathways are past8.
        all32=np.concatenate((sample32.reshape(-1,16,8,8,16),mean32))
        all16=np.concatenate((sample16.reshape(-1,16,8,8,16),mean16))
        projected32=[];projected_after_z=[]
        for end in ENDS:
            ix=np.arange(end-7,end+1)
            mats=(query[end]@np.linalg.inv(sensor[end]))[None]@sensor[ix]
            arms=[]
            for values in (all32,all16):
                arr=[]
                for first in range(0,len(values),4):
                    check();sub=values[first:first+4,ix]
                    with torch.no_grad():
                        out=engine.project(sub,np.broadcast_to(mats,(len(sub),8,4,4)))
                    arr.append(out.cpu().numpy());del out
                arms.append(np.concatenate(arr))
            projected32.append(arms[0]);projected_after_z.append(arms[1])
            print('PROJECTED',end,'wall',round(time.monotonic()-began,2),flush=True)
        v32=np.stack(projected32,1);v_z16=np.stack(projected_after_z,1);v16=v_z16.astype(np.float16)
        sample_rows=[];probe_rows=[];feature_rows=[];n_sample=len(sample_ids)*4
        for j,i in enumerate(sample_ids):
            for k in range(4):
                sample=j*4+k
                for ei,end in enumerate(ENDS):
                    ix=np.arange(end-7,end+1)
                    meta=dict(scene=i,replica=k,end=end,past8_start=end-7,role='contact' if i in ids else 'B_pass')
                    sample_rows.append({**meta,**rounding(sample32[j,k,ix],sample16[j,k,ix])})
                    for stage,a,b in (('second_feature_round',v_z16[sample,ei],v16[sample,ei]),
                                      ('first_z_propagated',v32[sample,ei],v_z16[sample,ei]),
                                      ('both_rounds',v32[sample,ei],v16[sample,ei])):
                        for channel, ci in (('tot',0),('cnt',1),('lst',2),('joint_tot_lst',[0,2])):
                            feature_rows.append({**meta,'stage':stage,'channel':channel,**rounding(a[ci],b[ci])})
        for i in ids:
            j=sample_index[i];q=n_sample+j
            for label,reference in (('A_absent',n_sample+len(sample_ids)),
                                    ('B_pass',n_sample+sample_index[pairs[i]] if i in pairs else None)):
                if reference is None:continue
                mr=reference-n_sample
                for ei,end in enumerate(ENDS):
                    ix=np.arange(end-7,end+1)
                    meta=dict(scene=i,reference_scene=pairs.get(i) if label=='B_pass' else 'absent',
                              height=rows[i]['group'],variant=rows[i]['variant'],placement=rows[i]['placement'],
                              contrast=label,end=end,past8_start=end-7)
                    probe_rows.append({**meta,'stage':'first_z_round','channel':'hist_window',
                        **contrast(mean32[j,ix],mean32[mr,ix],mean16[j,ix],mean16[mr,ix])})
                    for stage, before, after in (('first_z_propagated',v32,v_z16),
                                                ('second_feature_round',v_z16,v16),('both_rounds',v32,v16)):
                        for channel,ci in (('tot',0),('cnt',1),('lst',2),('joint_tot_lst',[0,2])):
                            probe_rows.append({**meta,'stage':stage,'channel':channel,
                                **contrast(before[q,ei,ci],before[reference,ei,ci],after[q,ei,ci],after[reference,ei,ci])})
        payload=OUT/'reconstruction.npz'; assert not payload.exists()
        np.savez_compressed(payload,sample_scene_ids=sample_ids,ends=ENDS,hist=hist,ambient=ambient,bias=bias,
                            sample_z32=sample32,sample_z16=sample16,mean_z32=mean32,mean_z16=mean16,
                            projection_z32=v32,projection_z16=v_z16,feature_fp16=v16,
                            sensor=sensor,query=query)
        csv_write('sample_z_rounding.csv',sample_rows);csv_write('sample_feature_rounding.csv',feature_rows)
        csv_write('mean_contrast_rounding.csv',probe_rows)
        check()
        result=dict(status='COMPLETE',seconds=time.monotonic()-began,cumulative_seconds=spent+time.monotonic()-began,
            backend=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                         projection='original cnh_active_scan_dev.Engine.project FP32; object.__new__, no networks'),
            sample_unique_contact_scenes=len(ids),sample_unique_B_pass_scenes=len(sample_ids)-len(ids),
            contact_K4=len(ids)*4,B_paired_K4=len(pairs)*4,A_absent_samples='NOT_RUN_NO_ABSENT_HIST',
            normalization_parity='BITWISE_EQUAL against frozen normalized_z, all 144 cached K4',
            mean_A_pairs=28,mean_B_pairs=24,frames=list(ENDS),source_snapshot_sha256=B.sha(OUT/'source_snapshot.py'),
            payload_sha256=B.sha(payload),nonfinite=int((~np.isfinite(all32)).sum()+(~np.isfinite(all16)).sum()+
                (~np.isfinite(v32)).sum()+(~np.isfinite(v_z16)).sum()+(~np.isfinite(v16)).sum()),
            limitations=plan['semantics'])
        assert result['nonfinite']==0
        save('result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as e:
        save('run_failure_'+str(time.time_ns())+'.json',dict(status='FAILED',error=repr(e),seconds=time.monotonic()-began))
        raise
    finally:
        if engine is not None:
            engine.projector=None
            for k in ('_pts32','_vol32'):
                if hasattr(engine,k):delattr(engine,k)
            del engine
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize();torch.cuda.empty_cache()
            save('release_'+str(time.time_ns())+'.json',dict(torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(),pid=os.getpid()))


def focused_check():
    # Recompute all published rounding/contrast metrics from archived pre/post values.
    p=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    r=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    assert r['payload_sha256']==B.sha(OUT/'reconstruction.npz')
    with np.load(OUT/'reconstruction.npz') as z:
        ids=z['sample_scene_ids'].tolist();index={i:j for j,i in enumerate(ids)};ns=len(ids)*4
        s32=z['sample_z32'];s16=z['sample_z16'];m32=z['mean_z32'];m16=z['mean_z16']
        a=z['projection_z32'];b=z['projection_z16'];c=z['feature_fp16']
        np.testing.assert_array_equal(s32.astype(np.float16),s16)
        np.testing.assert_array_equal(m32.astype(np.float16),m16)
        np.testing.assert_array_equal(b.astype(np.float16),c)
        np.testing.assert_array_equal(a[:,:,1],b[:,:,1])
        count=0
        def verify(row,expected):
            nonlocal count
            for key,value in expected.items():
                np.testing.assert_allclose(float(row[key]),value,rtol=1e-12,atol=1e-15)
            count+=1
        with (OUT/'sample_z_rounding.csv').open(encoding='utf8') as f:
            for row in csv.DictReader(f):
                j=index[int(row['scene'])];k=int(row['replica']);end=int(row['end']);ix=np.arange(end-7,end+1)
                verify(row,rounding(s32[j,k,ix],s16[j,k,ix]))
        channels={'tot':0,'cnt':1,'lst':2,'joint_tot_lst':[0,2]}
        stages={'first_z_propagated':(a,b),'second_feature_round':(b,c),'both_rounds':(a,c)}
        with (OUT/'sample_feature_rounding.csv').open(encoding='utf8') as f:
            for row in csv.DictReader(f):
                q=index[int(row['scene'])]*4+int(row['replica']);ei=ENDS.index(int(row['end']));ci=channels[row['channel']]
                before,after=stages[row['stage']];verify(row,rounding(before[q,ei,ci],after[q,ei,ci]))
        with (OUT/'mean_contrast_rounding.csv').open(encoding='utf8') as f:
            for row in csv.DictReader(f):
                j=index[int(row['scene'])];q=ns+j;ei=ENDS.index(int(row['end']));end=ENDS[ei]
                ref=ns+len(ids) if row['contrast']=='A_absent' else ns+index[int(row['reference_scene'])]
                if row['stage']=='first_z_round':
                    ix=np.arange(end-7,end+1);mr=ref-ns
                    v=contrast(m32[j,ix],m32[mr,ix],m16[j,ix],m16[mr,ix])
                else:
                    before,after=stages[row['stage']];ci=channels[row['channel']]
                    v=contrast(before[q,ei,ci],before[ref,ei,ci],after[q,ei,ci],after[ref,ei,ci])
                verify(row,v)
        assert count==5096, count
        save('focused_check.json',dict(status='PASS',recomputed_metric_rows=count,cast_parity='BITWISE',
                                     cnt_input_independence='BITWISE',payload_sha256=r['payload_sha256']))
        print('FOCUSED_CHECK_PASS',count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','check'))
    action=p.parse_args().action
    {'prepare':prepare,'run':run,'check':focused_check}[action]()
