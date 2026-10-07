"""Physical sensor-yaw simulation: new photons, poses and frozen-M3 scores.

Consumed Development with artificial head-yaw profiles, not measured head motion.
Native position-chord E1 and torso/gait direction states remain frozen.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import time
import numpy as np
import cnh_torso_native_motion_dev as M
import cnh_torso_native_replay_dev as N
import cnh_torso_bias_replay_dev as B

OUT = M.OUT / 'head_yaw'
NAMES = ('zero', 'const_pos', 'const_neg', 'pulse_pos', 'pulse_neg')
ARMS = ('exact', 'e1', 'torso', 'gait')
OLD_ARMS = ('exact', 'e1', 'torso', 'corrected_gait')
UNITS = list(range(99000, 99096))
BUDGET = {'prepare': 120., 'run': 2400., 'analyze': 180.}


def perturbation(name, n=600):
    t = np.arange(n) / 60.
    if name == 'zero': return np.zeros(n)
    if name not in NAMES: raise ValueError(name)
    sign = 1. if name.endswith('pos') else -1.
    if name.startswith('const_'): return np.full(n, sign * 15.)
    return np.where((t >= 2.) & (t <= 4.), sign * 10. * (1. - np.cos(np.pi * (t - 2.))), 0.)


def rotate_sensor(sensor, delta_deg):
    sensor = np.asarray(sensor, float)
    delta = np.broadcast_to(np.asarray(delta_deg, float), sensor.shape[:-2])
    r = np.broadcast_to(np.eye(3), sensor.shape[:-2] + (3,3)).copy()
    a = np.radians(delta); c, s = np.cos(a), np.sin(a)
    r[...,0,0] = c; r[...,0,2] = s
    r[...,2,0] = -s; r[...,2,2] = c
    new = sensor.copy(); new[...,:3,:3] = r @ sensor[...,:3,:3]
    new[delta == 0] = sensor[delta == 0]
    return new


def unit_inputs(f, name):
    delta = f['sign'][:,None] * perturbation(name)[f['native_frame_index']]
    sensor = rotate_sensor(f['sensor'], delta)
    relative = np.linalg.inv(f['sensor']) @ sensor
    queries = {a:f[o+'_query'] @ relative for a,o in zip(ARMS,OLD_ARMS)}
    for a,o in zip(ARMS,OLD_ARMS): queries[a][delta == 0] = f[o+'_query'][delta == 0]
    noisy = np.stack([N.RC.noisy_for(p, int(f['unit']), c) for c,p in enumerate(sensor)])
    return sensor, noisy, queries, delta


@contextmanager
def stage(name):
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = OUT/'attempts'; receipts.mkdir(exist_ok=True)
    spent = sum(B.A.read(p)['seconds'] for p in receipts.glob(name+'*.json'))
    tick = time.monotonic(); status = 'FAILED'; error = None
    def check():
        if spent + time.monotonic()-tick >= BUDGET[name]:
            raise TimeoutError(f'{name} cumulative wall budget {BUDGET[name]}s')
    try:
        check(); yield check; check(); status = 'COMPLETE'
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        B.save(receipts/f'{name}{time.time_ns()}.json', dict(stage=name,status=status,
            error=error,seconds=time.monotonic()-tick,previous_seconds=spent,budget=BUDGET[name]))


def prepare():
    with stage('prepare') as check:
        paths = [M.OUT/'bank.npz', M.OUT/'ledger.npz', M.OUT/'replay_result.json',
                 M.OUT/'replay_plan.json', N.HERE/'cnh_torso_head_yaw_dev.py',
                 N.HERE/'cnh_torso_gait_bias_dev.py', N.HERE/'cnh_real_head_confirm.py',
                 N.HERE/'cnh_extrinsic_aug_data.py', *B.A.M3_MODELS]
        plan = dict(task='CNH_TORSO_HEAD_YAW_DEV_20261007',lane='EXPLORE consumed Development',
            authorization='User 推进 after reviewed roundtable next-step recommendation',
            goal='Paired physical sensor yaw rerender to challenge frozen gait versus raw torso and E1',
            units=UNITS,names=NAMES,arms=ARMS,budgets_phase_wall_seconds=BUDGET,
            profiles='constant +/-15deg; native2..4s +/-10*(1-cos(pi*(age-2)))deg, peak20deg; sign mirrors source. Artificial stress profiles, not a measured population distribution.',
            pairing='Identical boxes/true positions/labels/deadlines. Ry(delta) left rotates true sensor orientation, translation unchanged. Rebuild all queries as oldQ@inv(oldSensor)@newSensor, regenerate noisy pose using same seed, rerender once per condition/scene shared by all arms.',
            photons='D.render original unit/config/branch seeds; no copied pulse recovery exposures. Zero/pulse pre-onset identical-pose causal prefix checked. Same seed does not promise recovery exposures identical after different Poisson draw consumption.',
            e1='Full native60Hz ideal head-position past60frame chord; physical yaw does not change this direction. New query, new noisy projection, photons and raw scores rerun; do not replace with reset noisy5Hz E1.',
            states='Original fullclip torso/gait and initialization preserved; no estimator perturbation or window reset; exact evaluator arm included only as diagnostic upper comparator.',
            workpoint='Each condition new E1 selects whole-tie <=2.5% all13 strict-control outputs; other arms matched to its actual integer count with residual reported. Secondary original zero thresholds. Descriptive evaluation workpoints, not deployment calibration.',
            coverage='Separate sparse interface fixture first; does not certify full physical coverage or integrate a full three-state pipeline.',
            stop='Phase cumulative cap or missing input/unresolved zero/prefix check: preserve partial/failure; no partial ranking, added challenges, retuning or expanded cap. Stop on completed queue.',
            adjustments='Implementation repairs only within phase caps; frozen M3/gait and cohort/profiles unchanged.',
            deliverables='Plan/source hashes, zero/prefix receipts, new observations/poses/queries/raw, paired results and loss traces, focused checks/audit, report and scoped master delivery.',
            limits='Artificial physical sensor yaw in future-conditioned simulated scenes with native proxy torso/ideal position origin, not real head-motion data or glasses/hardware. Existing overlapping consumed sources, participant mapping unknown. No CI or noninferiority claim. No new coverage/UNKNOWN safety proof; 1..2s event fixture still pending.',
            hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths})
        dest = OUT/'PLAN.json'
        if dest.exists():
            if B.A.read(dest) != B.json_value(plan): raise ValueError('Frozen head-yaw plan changed')
        else: B.save(dest,plan)
        source = B.A.read(M.OUT/'replay_result.json')['provenance']['unit_sha256']
        for u in UNITS:
            check()
            if B.A.sha(M.OUT/'units'/f'unit{u}.npz') != source[str(u)]: raise ValueError('Source unit changed')
        bank,_ = M.load_bank()
        # Rebuild the native E1 from position observations, avoiding a baseline switch.
        head = bank['head']; idx = np.maximum(np.arange(head.shape[1])-60,0)
        rebuilt = M.T.B.yaw(head - head[:,idx])
        np.testing.assert_allclose(M.T.B.wrap(rebuilt-bank['e1']),0,atol=1e-10,rtol=0)
        B.save(OUT/'prepare_result.json',dict(status='COMPLETE',source_units_verified=len(UNITS),e1_position_chord_rebuilt=True))
        print('PREPARE COMPLETE',flush=True)


def observations(f, sensor, check):
    import cnh_extrinsic_aug_data as D
    hist=[]; ambient=[]; zs=[]; backends=[]
    for c,box in enumerate(f['boxes_json']):
        check(); h,amb,z,backend = D.render(json.loads(str(box)),sensor[c],list(B.A.ANGLES),int(f['unit']),c,{})
        hist.append(h);ambient.append(amb);zs.append(z);backends.append(json.dumps(B.json_value(backend),separators=(',',':')))
    return dict(hist=np.stack(hist,1),ambient=np.stack(ambient,1),z=np.stack(zs,1),renderer_backend=np.array(backends))


def infer(rn, z, noisy, queries, check):
    count=len(queries); C=len(noisy); result=np.empty((count,3,C,13,2),np.float32)
    for branch,angle in enumerate(B.A.ANGLES):
        check(); ex=rn.eng.N.extrinsic(angle)
        raw=rn.raw(np.concatenate([z[branch]]*count),np.concatenate([noisy@ex]*count),np.concatenate([q@ex for q in queries]))
        result[:,branch]=raw.reshape(count,C,13,2)
    return result


def run():
    with stage('run') as check:
        coverage=B.A.read(OUT/'coverage'/'results.json')
        if coverage['status'] != 'PASS': raise ValueError('Coverage interface fixture unresolved')
        B.A.OUT=OUT/'runtime'; rn=None
        try:
            rn=B.H.Runner();check()
            if not (OUT/'zero_check.json').exists():
                with np.load(M.OUT/'units'/f'unit{UNITS[0]}.npz') as f:
                    sensor,noisy,qs,_=unit_inputs(f,'zero'); ob=observations(f,sensor,check)
                    equality={k:bool(np.array_equal(ob[k],f[k])) for k in ('hist','ambient','z')}
                    np.testing.assert_array_equal(noisy,f['noisy'])
                    for a,o in zip(ARMS,OLD_ARMS):np.testing.assert_allclose(qs[a],f[o+'_query'],atol=1e-12,rtol=0)
                    raw=infer(rn,ob['z'],noisy,list(qs.values()),check)
                    rawdiff={a:float(np.abs(raw[i]-f[o+'_raw']).max()) for i,(a,o) in enumerate(zip(ARMS,OLD_ARMS))}
                    ok=all(equality.values()) and all(v==0 for v in rawdiff.values())
                    B.save(OUT/'zero_check.json',dict(status='PASS' if ok else 'FAIL',unit=UNITS[0],observations_equal=equality,raw_max_abs=rawdiff,noisy_equal=True))
                    if not ok: raise ValueError('Zero rerender/inference mismatch')
            elif B.A.read(OUT/'zero_check.json')['status']!='PASS':raise ValueError('Unresolved saved zero check')
            for u in UNITS:
                check(); dest=OUT/'units'/f'unit{u}.npz'
                if dest.exists():continue
                tick=time.monotonic();values={}; prefixes=[]
                with np.load(M.OUT/'units'/f'unit{u}.npz') as f:
                    for name in NAMES[1:]:
                        sensor,noisy,qs,delta=unit_inputs(f,name)
                        ob=observations(f,sensor,check)
                        raw=infer(rn,ob['z'],noisy,list(qs.values()),check)
                        if name.startswith('pulse_'):
                            # Frames through age2s precede any nonzero yaw.
                            prefix=f['native_frame_index']<=120
                            for key in ('hist','ambient','z'):
                                np.testing.assert_array_equal(ob[key][:,prefix],f[key][:,prefix])
                            np.testing.assert_array_equal(noisy[prefix],f['noisy'][prefix])
                            for i,(a,o) in enumerate(zip(ARMS,OLD_ARMS)):
                                np.testing.assert_allclose(qs[a][prefix],f[o+'_query'][prefix],atol=1e-12,rtol=0)
                                output_prefix=prefix[:,3:]
                                np.testing.assert_array_equal(raw[i][:,output_prefix],f[o+'_raw'][:,output_prefix])
                            prefixes.append(dict(name=name,identical_input_frames=int(prefix.sum()),identical_raw_outputs_per_branch_arm=int(prefix[:,3:].sum())))
                        values.update({name+'/'+k:v for k,v in ob.items()})
                        values.update({name+'/sensor':sensor,name+'/noisy':noisy,name+'/delta_deg':delta})
                        for i,a in enumerate(ARMS):values[name+'/'+a+'_raw']=raw[i];values[name+'/'+a+'_query']=qs[a]
                    values['unit']=u
                check();B.atomic_npz(dest,**values)
                B.save(OUT/'prefix_checks'/f'unit{u}.json',dict(status='PASS',unit=u,checks=prefixes,seconds=time.monotonic()-tick))
                print('head_yaw unit',u,'seconds',round(time.monotonic()-tick,3),flush=True)
        finally:
            if rn is not None:rn.eng.torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));args=p.parse_args();globals()[args.stage]()
