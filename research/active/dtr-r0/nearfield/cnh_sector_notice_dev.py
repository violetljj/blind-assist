"""Head-relative spatial notices on cached physical-yaw Development photons.

Queries/notice selection never receive evaluator geometry or future direction.
Shared privileged pelvis-origin adapter remains from the practical baselines.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import cnh_torso_head_yaw_dev as Y
import cnh_torso_ema_compare_dev as E
import cnh_torso_bias_replay_dev as B
import cnh_real_head_confirm as RC

OUT = B.A.ROOT / 'artifacts.local/work/cnh-sector-notice-dev-20261008'
ANGLES = (-20., 0., 20.)
LABELS = ('LEFT', 'CENTER', 'RIGHT')
BUDGET = dict(prepare=180., run=1800., analyze=600.)


def sector_queries(noisy, displacement):
    """Yaw-only target axes at noisy head yaw + fixed angle; common origin.

    displacement is the inherited trueHead-idealPelvis origin difference;
    no future heading, labels, boxes, scores or true sensor rotation enters.
    """
    import cnh_cvr_pilot as CP
    qs = []
    for angle in ANGLES:
        q = np.broadcast_to(RC.rel_query(-angle), noisy.shape).copy()
        for c in range(len(noisy)):
            for f in range(16):
                heading = RC.A_yaw(noisy[c, f]) + angle
                q[c, f, :3, 3] = CP.rotation(-heading, 'y') @ displacement[c, f]
        qs.append(q)
    return qs


def emitted(score, theta, policy):
    """score[N,13,3]; one command per selected label (no audio playback)."""
    if policy == 'all_sectors':
        return score >= theta
    if policy != 'top1':
        raise ValueError(policy)
    selected = score.argmax(-1)  # exact ties: LEFT,CENTER,RIGHT fixed order
    out = np.zeros(score.shape, bool)
    np.put_along_axis(out, selected[..., None], (score.max(-1) >= theta)[..., None], -1)
    return out


def select_threshold(score, budget, policy):
    values = score.max(-1) if policy == 'top1' else score
    th = B.integer_budget_threshold(values, budget)
    return dict(threshold=th['threshold'], target_notices=budget,
        actual_notices=th['actual_fa_count'], unused_notices=th['residual_fa_count'],
        boundary_tie_count=th['boundary_tie_count'], selection='All3840 replay windows, whole ties, no event/localization objective')


def events(notice, truth, contact, deadline):
    """First main-window command is primary; later correct commands separate."""
    n = len(notice); idx = np.arange(13)[None, :]
    eligible = (idx >= 2) & (idx <= deadline[:, None]) & contact[:, None]
    active = notice.any(-1) & eligible
    first = np.where(active.any(-1), active.argmax(-1), -1)
    timely = first >= 0
    at_first = np.zeros((n, 3), bool); correct = np.zeros(n, bool)
    clean = np.zeros(n, bool); unique_correct = np.zeros(n, bool)
    rr = np.flatnonzero(timely); ff = first[rr]
    at_first[rr] = notice[rr, ff]
    compatible = notice[rr, ff] & truth[rr, ff]
    wrong = notice[rr, ff] & ~truth[rr, ff]
    correct[rr] = compatible.any(-1)
    clean[rr] = compatible.any(-1) & ~wrong.any(-1)
    unique_correct[rr] = clean[rr] & (truth[rr, ff].sum(-1) == 1)
    ever = ((notice & truth).any(-1) & eligible).any(-1)
    return dict(timely=timely, first=first, first_supported=correct,
        first_clean=clean, first_unique=unique_correct, ever_supported=ever,
        first_labels=at_first)


def decision(diffs):
    if all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
        return 'SUPPORT_DEDUP_CHECK'
    if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
        return 'LOWER_CURRENT_SECTOR_READOUT_PRIORITY'
    return 'RETAIN_MIXED_OR_ZERO_NO_AUTOMATIC_FOLLOWUP'


@contextmanager
def stage(name):
    receipts = OUT/'attempts'; receipts.mkdir(parents=True, exist_ok=True)
    spent = sum(B.A.read(p)['seconds'] for p in receipts.glob(name+'*.json'))
    tick = time.monotonic(); status = 'FAILED'; error = None
    def check():
        if spent + time.monotonic()-tick >= BUDGET[name]:
            raise TimeoutError(f'{name} cumulative wall cap {BUDGET[name]} seconds')
    try:
        check(); yield check; check(); status = 'COMPLETE'
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        B.save(receipts/f'{name}{time.time_ns()}.json', dict(stage=name, status=status,
            error=error, seconds=time.monotonic()-tick, previous_seconds=spent, budget=BUDGET[name]))


def prepare():
    with stage('prepare') as check:
        paths = [Path(__file__), Y.OUT/'result.json', E.OUT/'result.json', E.OUT/'ledger.npz',
            *B.A.M3_MODELS]
        plan = dict(task='CNH_SECTOR_NOTICE_DEV_20261008', lane='EXPLORE consumed Development',
            authorization='User 推进 after coarse-direction recommendation',
            goal='Timely correct head-relative information at the same emitted-command count as E1/EMA',
            units=Y.UNITS, names=Y.NAMES, sectors=LABELS, angle_degrees=ANGLES,
            budgets_phase_wall_seconds=BUDGET,
            baseline='Cached physical-yaw E1 and frozen EMA; same M3, photons and smoothing, practical schemes under unequal direction inputs',
            candidate='Fixed noisy-head-relative yaw axes -20/0/+20deg; no travel estimator; pitch-10deg, common inherited ideal pelvis-origin translation adapter',
            geometry='Evaluator-only all contact boxes at original interpolated0.9m reference; yaw-only current true sensor coordinates, x<0 LEFT; boundaries +/-10deg, footprint set and ambiguity retained',
            emission='Primary top1, secondary all_sectors. Each label each output costs one command; dual maxes branches before emission. No cooldown, audio timing, distance claim or user response model. Same-command-times label removal is exact detection ablation.',
            budget='Per condition/config E1 old actual-FA124/4992 working point anchors total notice count on all3840 windows/all13 outputs. Match EMA and candidate to this count with whole ties and residuals. Descriptive evaluation workpoints, no deployment calibration.',
            metrics='Timely first main-window notice (outputs2..deadline inclusive), first supported footprint, first clean/all labels supported, unique supported subset, later-any supported separate; full229 denominator; strict-clear cost/384 separately, other3227 not called false alarms',
            decision_check='229 contact events; one paired event minimum change, baseline headroom at new notice budget unknown. Compare first_clean notices to max(E1,EMA) timely counts per sensor and four challenges separately, no summed cells. Baseline does not claim localization.',
            decision_rule='For primary top1, four first_clean-minus-max(E1,EMA) timely count diffs: all>=0 and some>0 supports dedup; all<=0 and some<0 lowers this readout; mixed/all-zero retains without automatic followup. Geometric footprint support is not object attribution, user benefit, significance or noninferiority.',
            adjustable_scope='Implementation repairs and recording defects; fixed queries/readouts/cohort. No training, rerender, new NAT batch or truth-selected tuning. Dedup only after primary support and evaluable continuous two-obstacle sequences.',
            stop='Complete queue or cumulative phase cap/unresolved source/parity failure; preserve failures/partials, no partial ranking or cap expansion',
            deliverables='Cached-query inference, independent geometry evaluator, paired output ledger/CSV, focused tests and report/current/run log/master delivery',
            hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths})
        B.save(OUT/'PLAN.json', plan)
        parent=B.A.read(Y.OUT/'result.json')['provenance']['unit_sha256']
        native=B.A.read(Y.M.OUT/'replay_result.json')['provenance']['unit_sha256']
        for u in Y.UNITS:
            check()
            if B.A.sha(Y.OUT/'units'/f'unit{u}.npz') != parent[str(u)]: raise ValueError('Yaw input changed')
            if B.A.sha(Y.M.OUT/'units'/f'unit{u}.npz') != native[str(u)]: raise ValueError('Native input changed')
        B.save(OUT/'prepare_result.json', dict(status='COMPLETE', units_verified=96,
            geometry_source_sha256=B.A.sha(Path(__file__).with_name('cnh_sector_geometry_dev.py'))))


def run():
    with stage('run') as check:
        if not (OUT/'prepare_result.json').exists(): raise ValueError('Prepare first')
        B.A.OUT=OUT/'runtime'; rn=None
        try:
            rn=B.H.Runner(); check()
            torch=rn.eng.torch
            B.save(OUT/f'backend{time.time_ns()}.json', dict(device=str(rn.eng.device) if hasattr(rn.eng,'device') else 'cuda',
                torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                cuda_device=torch.cuda.get_device_name(), projection='fused FP32 CUDA', inference='AMP FP16 five frozen seeds'))
            for u in Y.UNITS:
                check(); dest=OUT/'units'/f'unit{u}.npz'
                if dest.exists(): continue
                tick=time.monotonic(); values=dict(unit=u)
                with np.load(Y.M.OUT/'units'/f'unit{u}.npz') as old, np.load(Y.OUT/'units'/f'unit{u}.npz') as phy:
                    for name in Y.NAMES:
                        f=old if name=='zero' else phy; prefix='' if name=='zero' else name+'/'
                        noisy=f[prefix+'noisy']; sensor=f[prefix+'sensor']; z=f[prefix+'z']
                        queries=sector_queries(noisy, sensor[...,:3,3]-old['travel'][...,:3,3])
                        values[name+'/raw']=Y.infer(rn,z,noisy,queries,check)
                        values[name+'/query']=np.asarray(queries)
                    if u==Y.UNITS[0]:
                        ref=Y.infer(rn,phy['const_pos/z'],phy['const_pos/noisy'],[phy['const_pos/e1_query']],check)[0]
                        np.testing.assert_array_equal(ref,phy['const_pos/e1_raw'])
                        B.save(OUT/'first_unit_parity.json',dict(status='PASS',unit=u,max_abs=0.))
                check(); B.atomic_npz(dest,**values)
                seconds=time.monotonic()-tick
                B.save(OUT/'progress'/f'unit{u}.json',dict(unit=u,seconds=seconds))
                if u==Y.UNITS[0]: B.save(OUT/'cost_projection.json',dict(first_unit_seconds=seconds, projected96_seconds=seconds*96, first_unit_reused=True))
                print('SECTOR unit',u,'seconds',round(seconds,3),flush=True)
        finally:
            if rn is not None: rn.eng.torch.cuda.empty_cache()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('prepare','run'))
    args=parser.parse_args(); globals()[args.stage]()
