"""Active scanning prompts: single-sensor UNKNOWN triggers a simulated head turn.

EXPLORE, consumed simulation Development (batches 98000/99000). Frozen M3,
alarm threshold, r3 coverage gate (m=3 deg) and event/control definitions.
Arms: stored passive single/dual; open-loop head-aligned single/dual
(mode0/1 re-rendered, mode2 already aligned); closed-loop single prompting.
The simulated user response is an assumed model, not measured behaviour.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'artifacts.local/work/cnh-active-scan-dev-20261006'
AUG = ROOT/'artifacts.local/work/cnh-extrinsic-aug-20261006'
CONT = AUG/'continuation-r1'
R3 = ROOT/'artifacts.local/work/cnh-tristate-dev-r3-20261006'
M3_MODELS = [ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{s}.pt' for s in range(5)]
UNITS = list(range(98000, 98048))+list(range(99000, 99096))
ANGLES = (0., -15., 15.)
NOISE_PREFIX, PHOTON_PREFIX, PROMPT_PHOTON_PREFIX = 2026100607, 2026100606, 2026100661
POLICY = dict(trigger_frames=2, refractory_frames=8, delay_frames=3, hold_frames=5,
              max_step_deg=12., target_yaw_deg=0., earliest_gate_frame=5)

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda: s.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), f'preserve {p}'
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def check_deadline():
    if time.time() > read(OUT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('active-scan wall budget reached')


def setup_gpu():
    for k, f in (('TEMP', 'tmp'), ('TMP', 'tmp'), ('CUPY_CACHE_DIR', 'cupy-cache')):
        (OUT/f).mkdir(parents=True, exist_ok=True); os.environ[k] = str(OUT/f)


def scripted_head(mode):
    if mode == 1:
        return 20*np.sin(np.linspace(-np.pi/2, np.pi/2, 16))
    return np.full(16, 15. if mode == 0 else 0.)


def travel_yaw(mode):
    return np.linspace(-20., 0., 16) if mode == 2 else np.zeros(16)


def poses_for(unit, config, head):
    """Unmirrored construction identical to CP.motion_metadata, head yaw supplied."""
    import cnh_cvr_pilot as CP
    from cnh_track_a_readout import noisy_poses
    assert Path(sys.modules['cnh_track_a_readout'].__file__).resolve().parent.name == 'source'
    mode = unit % 3; yaw = travel_yaw(mode)
    position = np.zeros((16, 3))
    for i in range(1, 16):
        position[i] = position[i-1]+.16*(CP.rotation((yaw[i-1]+yaw[i])/2, 'y')@np.array([0., 0., 1.]))
    position -= position[-1]
    n = len(head)
    sensor = np.repeat(np.eye(4)[None], n, 0); travel = np.repeat(np.eye(4)[None], 16, 0); query = sensor.copy()
    for i in range(16):
        travel[i, :3, :3] = CP.rotation(yaw[i], 'y'); travel[i, :3, 3] = position[i]
    for i in range(n):
        sensor[i, :3, :3] = CP.rotation(yaw[i]+head[i], 'y')@CP.rotation(-10, 'x'); sensor[i, :3, 3] = position[i]
        query[i, :3, :3] = CP.rotation(head[i], 'y')@CP.rotation(-10, 'x')
    seed = int(np.random.SeedSequence([NOISE_PREFIX, unit, config, 0]).generate_state(1)[0])
    noisy = noisy_poses(sensor, seed, dt=.2)
    return sensor, travel, noisy, query


def mirrored(unit):
    return (unit//3) % 2 == 1


def mirror(x, flag):
    R = np.diag([-1., 1., 1., 1.])
    return R@np.asarray(x)@R if flag else np.asarray(x).copy()


class Engine:
    def __init__(self):
        setup_gpu()
        import cnh_extrinsic_aug_data  # noqa: F401  frozen-source import order (inserts SOURCE)
        import torch
        dll = Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
        if os.name == 'nt' and dll.is_dir():
            self._dll = os.add_dll_directory(str(dll)); os.environ['PATH'] = str(dll)+os.pathsep+os.environ['PATH']
        import cnh_cvr_pilot as CP
        import cnh_dual_sensor_envelope_natural as N
        from cnh_cvr_v2_materialize import BatchedProjector
        from cnh_cvr_projection import query_masks
        self.torch = torch; torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self.projector = BatchedProjector(); self.N = N
        self.masks = torch.as_tensor(query_masks(), device='cuda')
        self.nets = []
        for p in M3_MODELS:
            n = CP.CVR().cuda().eval(); n.load_state_dict(torch.load(p, map_location='cpu', weights_only=True)); self.nets.append(n)

    def project(self, z, matrices):
        """FP32 copy of N.Predictor.paired_project (FP64 is ~3x slower on this GPU).
        Validated against stored FP64 passive scores; see fp32_parity.json."""
        from cnh_cvr_projection import SUB, SHAPE, EDGE, WIDTH
        torch = self.torch; p = self.projector; c, l = matrices.shape[:2]
        if not hasattr(self, '_pts32'):
            self._pts32 = p.points.float(); self._vol32 = p.volumes.float()
        t = torch.as_tensor(matrices.reshape(c*l, 4, 4), dtype=torch.float32, device='cuda')
        points = torch.bmm(self._pts32[None]-t[:, None, :3, 3], t[:, :3, :3])
        radius = torch.linalg.vector_norm(points, dim=2)
        xy = points[:, :, :2]/points[:, :, 2:3].clamp_min(1e-30)
        ij = torch.floor((xy+EDGE)/(2*EDGE)*8).long(); bins = torch.floor(radius/WIDTH).long()
        valid = (points[:, :, 2] > 0)&(ij >= 0).all(2)&(ij < 8).all(2)&(bins >= 0)&(bins < 16)
        index = (ij[:, :, 1].clamp(0, 7)*8+ij[:, :, 0].clamp(0, 7))*16+bins.clamp(0, 15)
        weight = valid*p.voxel_volume/(SUB**3)/self._vol32[index]
        values = torch.as_tensor(z, dtype=torch.float32, device='cuda').reshape(c, l, 1024)
        mass = torch.gather(values, 2, index.reshape(c, l, -1))*weight.reshape(c, l, -1)
        evidence = mass.reshape(c, l, -1, SUB**3).sum(3).reshape(c, l, *SHAPE)
        coverage = valid.reshape(c, l, -1, SUB**3).float().mean(3).reshape(c, l, *SHAPE)
        total = evidence.sum(1); count = coverage.sum(1)
        return torch.stack((total, count, evidence[:, -1]), 1)

    def features(self, z, noisy, query, frame):
        """z[C,16,8,8,16], noisy/query[C,16,4,4] (already branch-extrinsic) -> fp16 voxels[C,3,24,17,33]."""
        ix = np.arange(max(0, frame-7), frame+1); out = []
        for b in range(0, len(z), 4):
            nn, qq = noisy[b:b+4], query[b:b+4]
            m = (qq[:, frame]@np.linalg.inv(nn[:, frame]))[:, None]@nn[:, ix]
            out.append(self.project(z[b:b+4][:, ix], m).cpu().numpy().astype(np.float16))
        return np.concatenate(out)

    def predict(self, voxels):
        from cnh_temporal_readout_model import prepare_voxels
        torch = self.torch; out = []
        with torch.inference_mode():
            for b in range(0, len(voxels), 64):
                x = prepare_voxels(torch.from_numpy(np.ascontiguousarray(voxels[b:b+64])).cuda(), self.masks)
                out.append(torch.stack([n(x).float() for n in self.nets]).mean(0).cpu().numpy())
        return np.concatenate(out)

    def render_sequence(self, boxes, sensor, angles, unit, config):
        """Whole-sequence render with the stored photon seeds (bitwise passive parity)."""
        import cnh_extrinsic_aug_data as D
        hist, ambient, z, backend = D.render(boxes, sensor, list(angles), unit, config, {})
        return z, backend

    def render_frame(self, boxes, pose, unit, config, branch, frame):
        import cnh_location_reference_gpu as G
        import cnh_displacement_ceiling_render as R
        from cnh_temporal_readout_data import normalized_z
        e = G.ExpectedRenderer(pose[None], boxes)
        try:
            ex = e.background_expected(return_device=False)
            amb = e.ambient.copy()
        finally:
            e.close()
        seed = int(np.random.SeedSequence([PROMPT_PHOTON_PREFIX, unit, config, branch, frame]).generate_state(1)[0])
        hist = R.sample(ex, amb, seed)[0]
        return normalized_z(hist, amb)[0]


def gate_seq(est, frames=range(3, 16)):
    import cnh_tristate_dev_r3_geometry as G3
    return np.array([G3.gates(G3.prepared(est, int(f)), [3.])['passed'][0] for f in frames])  # [13,2] single,dual


def smooth_online(raw):
    """raw[...,k,2] for frames 3..3+k-1 -> smoothed last frame [...,2] (R.smooth weights)."""
    k = raw.shape[-2]; begin = max(0, k-5); w = 2.**np.arange(k-begin)
    return np.sum(raw[..., begin:k, :]*w[:, None], axis=-2)/w.sum()


_R3 = {}


def r3_gate(unit, C):
    if not _R3:
        rows = read(R3/'rows.json')
        with np.load(R3/'online.npz') as z:
            g = z['gate'][:, :, 1, :]
        _R3.update({(r['unit'], r['config']): g[i] for i, r in enumerate(rows)})
    return np.stack([_R3[unit, c] for c in range(C)])


def stored(unit):
    split = 'calibration' if unit < 99000 else 'evaluation'
    base = AUG if unit < 99000 else CONT
    obs = np.load(base/f'observations/{split}/unit{unit}.npz')
    sc = np.load(CONT/f'scores/{split}/ensemble3/unit{unit}.npz')
    return obs, sc


def run_unit(eng, unit, delay=None):
    import cnh_margin_confirm as MC
    import cnh_extrinsic_aug_data as D
    import cnh_tristate_dev as R
    sensitivity = delay is not None
    dest = OUT/(f'units_delay{delay}' if sensitivity else 'units')/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); mode = unit % 3; flag = mirrored(unit)
    obs, sc = stored(unit)
    scenes = MC.scenes_for(unit); C = len(scenes)
    boxes = [D.mirror_boxes(s['boxes'], flag) for s in scenes]
    exs = [eng.N.extrinsic(a) for a in ANGLES]
    head0 = scripted_head(mode)
    # passive contract check (stored noisy/query/sensor equal to our construction)
    ps, pt, pn, pq = (mirror(x, flag) for x in poses_for(unit, 0, head0))
    np.testing.assert_allclose(ps, obs['sensor_center'], atol=1e-12); np.testing.assert_allclose(pq, obs['public_query'], atol=1e-12)
    passive_noisy = np.stack([mirror(poses_for(unit, c, head0)[2], flag) for c in range(C)])
    np.testing.assert_allclose(passive_noisy, obs['noisy_center'], atol=1e-12)
    passive_gate = r3_gate(unit, C)  # frozen r3 gate (margin m=3), parity verified on earlier units
    result = dict(unit=unit, mode=mode, mirror=flag, passive_gate=passive_gate,
                  passive_raw=sc['reference'].astype(np.float32))
    # ---- open-loop head-aligned arm (mode2 is already aligned: copy passive) ----
    if sensitivity:
        pass
    elif mode in (0, 1):
        head = np.zeros(16); zs = []; noisys = []
        sensor, travel, _, query = (mirror(x, flag) for x in poses_for(unit, 0, head))
        for c in range(C):
            check_deadline()
            z, backend = eng.render_sequence(boxes[c], sensor, ANGLES, unit, c); zs.append(z)
            noisys.append(mirror(poses_for(unit, c, head)[2], flag))
        z = np.stack(zs, 1)  # [3,C,16,...]
        noisy = np.stack(noisys)
        raw = np.empty((3, C, 13, 2), np.float32)
        for s, ex in enumerate(exs):
            nn = noisy@ex; qq = np.repeat((query@ex)[None], C, 0)
            vox = np.stack([eng.features(z[s], nn, qq, int(f)) for f in range(3, 16)], 1)
            raw[s] = eng.predict(vox.reshape(-1, 3, 24, 17, 33)).reshape(C, 13, 2)
        result.update(aligned_raw=raw, aligned_gate=np.stack([gate_seq(noisy[c]) for c in range(C)]))
    else:
        result.update(aligned_raw=result['passive_raw'], aligned_gate=passive_gate)
    # ---- closed-loop single prompting ----
    # Frames before a sequence's head first diverges are bitwise identical to the stored
    # passive single arm (verified 0 abs diff on the first 18 full-recompute units), so
    # stored raw/gate are reused there; only diverged sequences are recomputed.
    zS = obs['z1'][0].astype(np.float16).copy()  # [C,16,8,8,16]
    head = np.repeat(head0[None], C, 0).astype(float)
    prompts = np.zeros((C, 16), bool); fresh = np.zeros((C, 16), bool); diverged = np.zeros(C, bool)
    raw = np.empty((C, 13, 2), np.float32); gate = np.zeros((C, 13), bool); state = np.zeros((C, 13), np.int8)
    unknown_run = np.zeros(C, int); last_prompt = np.full(C, -99)
    noisy = passive_noisy.copy(); query = np.repeat(pq[None], C, 0)
    P = dict(POLICY, delay_frames=delay) if sensitivity else POLICY
    for f in range(16):
        check_deadline()
        for c in range(C):
            # head at f responds to prompts issued at <= f-delay
            active = np.flatnonzero(prompts[c, :max(0, f-P['delay_frames']+1)])
            if f > 0 and len(active):
                t = active[-1]; start = t+P['delay_frames']; prev = head[c, f-1]
                goal = P['target_yaw_deg'] if f < start+P['hold_frames'] else head0[f]
                head[c, f] = prev+np.clip(goal-prev, -P['max_step_deg'], P['max_step_deg'])
            moved = abs(head[c, f]-head0[f]) > 1e-9
            diverged[c] |= moved
            if diverged[c]:
                s_, _, n_, q_ = (mirror(x, flag) for x in poses_for(unit, c, head[c, :f+1]))
                noisy[c, :f+1] = n_; query[c, :f+1] = q_[:f+1]
                if moved:
                    fresh[c, f] = True
                    zS[c, f] = eng.render_frame(boxes[c], s_[f], unit, c, 0, f)
        if f < 3:
            continue
        k = f-3
        raw[:, k] = result['passive_raw'][0, :, k]; gate[:, k] = passive_gate[:, k, 0]
        idx = np.flatnonzero(diverged)
        if len(idx):
            raw[idx, k] = eng.predict(eng.features(zS[idx], noisy[idx], query[idx], f))
            for c in idx:
                gate[c, k] = gate_seq(noisy[c, :f+1], [f])[0, 0]
        score = smooth_online(raw[:, :k+1]).max(-1)
        alarm = score >= R.THRESHOLD
        unk = ~alarm & ~gate[:, k]
        state[:, k] = np.where(alarm, 2, np.where(unk, 1, 0))
        evaluable = f >= P['earliest_gate_frame']
        unknown_run = np.where(unk & evaluable, unknown_run+1, 0)
        fire = (unknown_run >= P['trigger_frames'])&(f-last_prompt >= P['refractory_frames'])&(f <= 14)
        prompts[fire, f] = True; last_prompt[fire] = f
    result.update(prompt_raw=raw, prompt_gate=gate, prompt_state=state, prompts=prompts, head=head, fresh=fresh,
                  seconds=time.monotonic()-tick)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **result)
    print('unit', unit, 'mode', mode, 'prompts', int(prompts.sum()), 'fresh', int(fresh.sum()), round(time.monotonic()-tick, 1), 's', flush=True)


def fp32check(eng):
    import cnh_tristate_dev as R
    thresholds = np.load(R3/'online.npz')['thresholds']; rows = []
    for unit in (99001, 99002, 99003, 98001):
        obs, sc = stored(unit)
        for s, a in enumerate(ANGLES):
            ex = eng.N.extrinsic(a); nn = obs['noisy_center']@ex; qq = np.repeat((obs['public_query']@ex)[None], 40, 0)
            vox = np.stack([eng.features(obs['z1'][s], nn, qq, f) for f in range(3, 16)], 1)
            raw = eng.predict(vox.reshape(-1, 3, 24, 17, 33)).reshape(40, 13, 2); ref = sc['reference'][s]
            s1, s0 = R.smooth(raw).max(-1), R.smooth(ref).max(-1)
            flips = int(sum(((s1 <= t) != (s0 <= t)).sum() for t in thresholds)+((s1 >= R.THRESHOLD) != (s0 >= R.THRESHOLD)).sum())
            rows.append(dict(unit=unit, branch=s, max_abs_logit=float(np.abs(raw-ref).max()), state_flips_over_20tau_and_alarm=flips, frames=int(s1.size)))
            print(rows[-1], flush=True)
    save(OUT/'fp32_parity.json', dict(rows=rows, note='FP32 projection vs stored FP64 passive features+M3; any flips are counted, not hidden'))


def freeze(hours):
    now = time.time()
    save(OUT/'PLAN.json', dict(task='CNH_ACTIVE_SCAN_DEV_20261006', lane='EXPLORE consumed simulation Development',
        authorization='User 2026-10-06 "你来做" after Claude proposed active scanning; Claude executes',
        goal='Can coverage-driven head-turn prompts let a single sensor approach passive dual on the event/burden tradeoff, and at what prompt cost?',
        units='98000-98047 + 99000-99095 (144 units, 40 configs each); events/controls = existing exact 0.9 m ledger subset',
        frozen='M3 5-seed reference, alarm 0.8557642486787612, causal 5-frame smoothing, r3 gate m=3 deg, 20-point tau grid, event/control/burden definitions',
        arms=dict(passive='stored single/dual', aligned='open-loop head yaw 0 relative to travel all frames; mode2 identical to passive; same photon/noise seeds',
                  prompt='closed-loop single; frames identical to passive until head diverges; diverged frames rendered per frame with new photon seeds'),
        policy=POLICY|dict(state='unknown = no alarm and single gate fails (tau = alarm threshold); prompts only at frames 6..14',
                           response='after delay, head offset moves toward 0 deg at <=12 deg/frame (60 deg/s), holds 5 frames (1.0 s), then returns to scripted yaw at same rate',
                           privileged='alarm query uses exact current head-to-travel (existing convention); prompt direction assumes travel direction known'),
        metrics='timely/unknown-miss/silent at deadline (subset denominators), control unknown time, prompts per simulated minute on controls and all sequences, per mode; tau curves and silent-budget matched points (descriptive)',
        stop='Aligned arm is the zero-delay upper bound; if aligned single does not cut mode0/1 unknown burden by >=20 pp, skip prompt interpretation as useless',
        budget_wall_hours=hours, started_unix=now, deadline_unix=now+hours*3600,
        source_sha256=sha(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['freeze', 'run', 'one', 'fp32check']); ap.add_argument('--hours', type=float, default=6.)
    ap.add_argument('--unit', type=int); ap.add_argument('--delay', type=int)
    ap.add_argument('--reverse', action='store_true')
    a = ap.parse_args()
    if a.stage == 'freeze':
        return freeze(a.hours)
    sys.path.insert(0, str(HERE))
    eng = Engine()
    if a.stage == 'fp32check':
        return fp32check(eng)
    for u in ([a.unit] if a.stage == 'one' else (UNITS[::-1] if a.reverse else UNITS)):
        run_unit(eng, u, a.delay)


if __name__ == '__main__':
    main()
