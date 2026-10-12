"""FP64 CUDA zero-noise CNH templates with shared first-visible background.

Physical quadrature, opaque ray replacement, four angular quarters, pulse,
neighbour exchange and residual crosstalk match the frozen R.expected path.
No observation, target identity, evaluator label or RNG enters this renderer.
The preaggregated pulse uses linearity; engineering parity is checked before
scientific use. Candidate and pose tiles bound device workspace.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import cnh_displacement_ceiling_render as R

S = R.S
SENSOR = sys.modules[S.synthesize_response.__module__]
ENDPOINT_RHO = .65

CUDA_SOURCE = r'''
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
__device__ bool slab(const double *box, const double *o, const double *d,
                     double *range, double *cosine) {
    double enter=-INFINITY, leave=INFINITY;
    int enter_axis=0, leave_axis=0;
    bool outside=false;
    for (int j=0;j<3;++j) {
        bool parallel=fabs(d[j])<1e-14;
        if (parallel) {
            outside |= o[j]<box[j] || o[j]>box[j+3];
        } else {
            double x=(box[j]-o[j])/d[j], y=(box[j+3]-o[j])/d[j];
            double near=fmin(x,y), far=fmax(x,y);
            if (near>enter) {enter=near;enter_axis=j;}
            if (far<leave) {leave=far;leave_axis=j;}
        }
    }
    double t=enter>1e-10?enter:leave;
    if (outside || !(leave>=fmax(enter,0.0)) || !(t>1e-10)
            || !isfinite(t)) return false;
    *range=t; *cosine=fabs(d[enter>1e-10?enter_axis:leave_axis]);
    return true;
}
extern "C" __global__ void background_hits(
    const double *poses,const double *rays,const double *boxes,int nbox,
    int nray,long long size,double *directions,double *distance,
    double *rho,double *cosine,int *identity) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size)return;
    int p=i/nray,r=i%nray;
    const double *pose=poses+p*16;
    double o[3]={pose[3],pose[7],pose[11]},d[3];
    double norm=0;
    for(int j=0;j<3;++j){
        d[j]=pose[j*4]*rays[r*3]+pose[j*4+1]*rays[r*3+1]
                 +pose[j*4+2]*rays[r*3+2];
        norm+=d[j]*d[j];
    }
    norm=sqrt(norm);
    for(int j=0;j<3;++j){d[j]/=norm;directions[i*3+j]=d[j];}
    double best=INFINITY,bc=0,br=0;int bi=-1;
    for(int b=0;b<nbox;++b){
        double t,c;
        if(slab(boxes+b*7,o,d,&t,&c) && t<best){
            best=t;bc=c;br=boxes[b*7+6];bi=b;
        }
    }
    distance[i]=best;rho[i]=br;cosine[i]=bc;identity[i]=bi;
}
extern "C" __global__ void candidate_histograms(
    const double *poses,const double *directions,const double *background_d,
    const double *background_rho,const double *background_cos,
    const double *targets,const double *fraction,const double *weight,
    const int *quarter,int npose,int pose_start,int nray,int sub,int ncand,
    double signal,double rhohi,double rangezero,double binwidth,
    double *h0,double *hhi) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long size=(long long)ncand*npose*nray;
    if(i>=size)return;
    int ray=i%nray,pt=(i/nray)%npose,cand=i/(nray*npose);
    int p=pt+pose_start;
    long long hit=(long long)p*nray+ray;
    const double *pose=poses+p*16;
    double o[3]={pose[3],pose[7],pose[11]};
    const double *d=directions+hit*3;
    double target_d,target_cos;
    bool take=slab(targets+cand*6,o,d,&target_d,&target_cos)
                     && target_d<=background_d[hit];
    double range=take?target_d:background_d[hit];
    if(!isfinite(range)||range<=0)return;
    int bin=(int)floor((range-rangezero)/binwidth);
    if(bin<0||bin>=128)return;
    double cosine=take?target_cos:background_cos[hit];
    double r0=take?0.:background_rho[hit];
    double rh=take?rhohi:background_rho[hit];
    double safe=fmax(range,.05);
    // Retain the frozen quarter multiplication order.
    double e0=(signal/4.)*(r0*4.*fraction[ray])*cosine*weight[ray]/(safe*safe);
    double eh=(signal/4.)*(rh*4.*fraction[ray])*cosine*weight[ray]/(safe*safe);
    int zone=ray/(sub*sub),q=quarter[ray];
    long long index=((((long long)cand*npose+pt)*4+q)*64+zone)*128+bin;
    if(e0!=0)atomicAdd(h0+index,e0);
    if(eh!=0)atomicAdd(hhi+index,eh);
}
extern "C" __global__ void candidate_delta_histograms(
    const double *poses,const double *directions,const double *background_d,
    const double *background_rho,const double *background_cos,
    const double *targets,const double *fraction,const double *weight,
    const int *quarter,int npose,int pose_start,int nray,int sub,int ncand,
    double signal,double rhohi,double rangezero,double binwidth,
    double *h0,double *hhi) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long size=(long long)ncand*npose*nray;
    if(i>=size)return;
    int ray=i%nray,pt=(i/nray)%npose,cand=i/(nray*npose);
    int p=pt+pose_start;
    long long hit=(long long)p*nray+ray;
    const double *pose=poses+p*16;
    double o[3]={pose[3],pose[7],pose[11]};
    const double *d=directions+hit*3;
    double target_d,target_cos;
    if(!slab(targets+cand*6,o,d,&target_d,&target_cos)
            || target_d>background_d[hit])return;
    int zone=ray/(sub*sub),q=quarter[ray];
    long long index=((((long long)cand*npose+pt)*4+q)*64+zone)*128;
    // Opaque foreground replaces background, including when its rho is zero.
    double bg=background_d[hit];
    if(isfinite(bg) && bg>0){
        int bin=(int)floor((bg-rangezero)/binwidth);
        if(bin>=0 && bin<128){
            double safe=fmax(bg,.05);
            double e=(signal/4.)*(background_rho[hit]*4.*fraction[ray])
                     *background_cos[hit]*weight[ray]/(safe*safe);
            if(e!=0){atomicAdd(h0+index+bin,-e);atomicAdd(hhi+index+bin,-e);}
        }
    }
    int bin=(int)floor((target_d-rangezero)/binwidth);
    if(bin>=0 && bin<128){
        double safe=fmax(target_d,.05);
        double e=(signal/4.)*(rhohi*4.*fraction[ray])*target_cos*weight[ray]/(safe*safe);
        if(e!=0)atomicAdd(hhi+index+bin,e);
    }
}
'''


def _boxes(boxes):
    values = np.asarray([[*b['lo'], *b['hi'], b['rho']] for b in boxes], dtype=np.float64)
    if not len(values) or values.shape[1:] != (7,) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite physical background boxes required')
    if np.any(values[:, :3] >= values[:, 3:6]) or np.any((values[:, 6] < 0) | (values[:, 6] > 1)):
        raise ValueError('Invalid background geometry/rho')
    return values


def _targets(candidates):
    values = np.asarray([[*b['lo'], *b['hi']] if isinstance(b, dict) else b
                         for b in candidates], dtype=np.float64)
    if not len(values) or values.shape[1:] != (6,) or not np.isfinite(values).all() or np.any(values[:, :3] >= values[:, 3:]):
        raise ValueError('Finite nonempty candidate AABBs[lo3,hi3] required')
    return values


class ExpectedRenderer:
    """Reuse background for fixed poses; candidate rho is unknown, not read.

    iter_render yields (begin, endpoints[C,2,P,8,8,16]) with rho endpoints0/.65.
    P must be the caller's fixed anchor/past concatenation (e.g.94 for window8).
    No window mixture or class mixture is performed by this physical renderer.
    """
    def __init__(self, poses, background, *, sub=16, device=0):
        import cupy as cp
        if sub != 16:
            raise ValueError('Frozen sub16 quadrature required')
        self.cp, self.sub = cp, sub
        self.device = cp.cuda.Device(device)
        self.device.use()
        poses = np.ascontiguousarray(poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
            raise ValueError('Finite physical poses[P,4,4] required')
        self.npose = len(poses)
        parameters, _ = S.nominal_parameters()
        if parameters.output_gain != 1 or parameters.noise_scale != 1:
            raise ValueError('Frozen gain1/noiseScale1 required')
        self.params = parameters
        rays, weights = S.angular_rays(sub)
        self.nray = int(rays.size//3)
        w = weights.reshape(8, 8, sub, sub)
        fractions = np.empty_like(w)
        normalized = np.empty_like(w)
        quarters = np.empty(w.shape, dtype=np.int32)
        half = sub//2
        for qy in range(2):
            for qx in range(2):
                ys, xs = slice(qy*half, (qy+1)*half), slice(qx*half, (qx+1)*half)
                part = w[:, :, ys, xs]
                mass = part.reshape(8, 8, -1).sum(-1)
                frac = mass/weights.sum(-1)
                fractions[:, :, ys, xs] = frac[:, :, None, None]
                normalized[:, :, ys, xs] = part/mass[:, :, None, None]
                quarters[:, :, ys, xs] = qy*2+qx
        self.poses = cp.asarray(poses)
        self.fractions, self.weights, self.quarters = (cp.asarray(a.reshape(-1))
            for a in (fractions, normalized, quarters))
        self.module = cp.RawModule(code=CUDA_SOURCE, options=('--std=c++11', '--fmad=false'),
                name_expressions=('background_hits', 'candidate_histograms', 'candidate_delta_histograms'))
        boxes = cp.asarray(_boxes(background))
        n = self.npose*self.nray
        self.directions = cp.empty((n, 3), dtype=cp.float64)
        self.distance = cp.empty(n, dtype=cp.float64)
        self.rho = cp.empty(n, dtype=cp.float64)
        self.cosine = cp.empty(n, dtype=cp.float64)
        self.identity = cp.empty(n, dtype=cp.int32)
        self.module.get_function('background_hits')(((n+255)//256,), (256,),
            (self.poses, cp.asarray(rays.reshape(-1, 3)), boxes, np.int32(len(background)),
             np.int32(self.nray), np.int64(n), self.directions, self.distance,
             self.rho, self.cosine, self.identity))
        matrix = SENSOR._pulse_matrix(parameters)
        # Linear coarse radial aggregation of the frozen pulse, without wrap.
        self.pulse = cp.asarray(matrix.reshape(128, 16, 8).sum(-1))
        xtalk_bin = int(np.floor((parameters.crosstalk_range_m-parameters.range_zero_m)/SENSOR.RAW_BIN_M))
        xtalk = np.zeros(16, dtype=np.float64)
        if 0 <= xtalk_bin < 128:
            xtalk = (parameters.signal_counts/4*parameters.crosstalk_fraction
                     *matrix[xtalk_bin]).reshape(16, 8).sum(-1)
        self.xtalk = cp.asarray(xtalk)
        self.ambient = np.full((self.npose, 8, 8), parameters.ambient_counts*parameters.output_gain)
        cp.cuda.get_current_stream().synchronize()
        props = cp.cuda.runtime.getDeviceProperties(device)
        self.metadata = dict(backend='CuPy raw CUDA FP64 AABB and atomic raw histogram; FP64 coarse pulse; frozen quarter electronics',
            device=props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
            device_id=int(device), cupy_version=cp.__version__, dtype='float64', sub=sub,
            sensor_params=asdict(parameters), endpoint_rhos=[0., ENDPOINT_RHO],
            compile_flags=['--std=c++11', '--fmad=false'], physical_poses=self.npose,
            background_hit_device_bytes=sum(a.nbytes for a in (self.directions, self.distance, self.rho, self.cosine, self.identity)))
        self._background_expected = None

    def background_expected(self, *, return_device=True, pose_batch=32):
        """Pure background expectation from cached hits, with no target AABB."""
        cp = self.cp
        if self._background_expected is None:
            result = cp.empty((self.npose, 8, 8, 16), dtype=cp.float64)
            zones = cp.arange(self.nray, dtype=cp.int64)//(self.sub*self.sub)
            ray_index = self.quarters.astype(cp.int64)*64+zones
            for ps in range(0, self.npose, int(pose_batch)):
                np_ = min(int(pose_batch), self.npose-ps)
                start, stop = ps*self.nray, (ps+np_)*self.nray
                distance = self.distance[start:stop].reshape(np_, self.nray)
                valid = cp.isfinite(distance) & (distance > 0)
                bins = cp.floor((cp.where(valid, distance, self.params.range_zero_m)-self.params.range_zero_m)
                                /SENSOR.RAW_BIN_M).astype(cp.int64)
                inside = valid & (bins >= 0) & (bins < 128)
                safe = cp.maximum(cp.where(valid, distance, 1.), .05)
                rho = self.rho[start:stop].reshape(np_, self.nray)
                cosine = self.cosine[start:stop].reshape(np_, self.nray)
                energy = ((self.params.signal_counts/4)*(rho*4*self.fractions[None])*cosine
                          *self.weights[None]/(safe*safe))
                index = ((cp.arange(np_, dtype=cp.int64)[:, None]*256+ray_index[None])*128
                         +cp.clip(bins, 0, 127))
                histogram = cp.zeros((np_, 4, 8, 8, 128), dtype=cp.float64)
                cp.add.at(histogram.reshape(-1), index.reshape(-1), cp.where(inside, energy, 0.).reshape(-1))
                signal = histogram@self.pulse
                old = signal.copy()
                leak = self.params.neighbour_leak/4
                signal[..., 1:, :, :] += leak*(old[..., :-1, :, :]-old[..., 1:, :, :])
                signal[..., :-1, :, :] += leak*(old[..., 1:, :, :]-old[..., :-1, :, :])
                signal[..., :, 1:, :] += leak*(old[..., :, :-1, :]-old[..., :, 1:, :])
                signal[..., :, :-1, :] += leak*(old[..., :, 1:, :]-old[..., :, :-1, :])
                signal += self.xtalk
                result[ps:ps+np_] = (signal[:, 0]+signal[:, 1])+(signal[:, 2]+signal[:, 3])
            self._background_expected = result
        return self._background_expected if return_device else cp.asnumpy(self._background_expected)

    def iter_render(self, candidates, *, candidate_batch=4, pose_batch=32, deadline_check=None,
                    return_device=False, delta=True):
        cp = self.cp
        candidates = _targets(candidates)
        if candidate_batch < 1 or pose_batch < 1:
            raise ValueError('Positive tile sizes required')
        background = self.background_expected(return_device=True, pose_batch=pose_batch) if delta else None
        for begin in range(0, len(candidates), int(candidate_batch)):
            if deadline_check is not None:
                deadline_check()
            targets = cp.asarray(np.ascontiguousarray(candidates[begin:begin+int(candidate_batch)]))
            count = len(targets)
            allocator = cp if return_device else np
            endpoints = allocator.empty((count, 2, self.npose, 8, 8, 16), dtype=allocator.float64)
            for ps in range(0, self.npose, int(pose_batch)):
                if deadline_check is not None:
                    deadline_check()
                np_ = min(int(pose_batch), self.npose-ps)
                shape = (count, np_, 4, 8, 8, 128)
                h0, hh = cp.zeros(shape, dtype=cp.float64), cp.zeros(shape, dtype=cp.float64)
                n = count*np_*self.nray
                kernel = 'candidate_delta_histograms' if delta else 'candidate_histograms'
                self.module.get_function(kernel)(((n+255)//256,), (256,),
                    (self.poses, self.directions, self.distance, self.rho, self.cosine,
                     targets, self.fractions, self.weights, self.quarters,
                     np.int32(np_), np.int32(ps), np.int32(self.nray), np.int32(self.sub), np.int32(count),
                     np.float64(self.params.signal_counts), np.float64(ENDPOINT_RHO),
                     np.float64(self.params.range_zero_m), np.float64(SENSOR.RAW_BIN_M), h0, hh))
                for ri, histogram in enumerate((h0, hh)):
                    signal = histogram@self.pulse
                    old = signal.copy()
                    leak = self.params.neighbour_leak/4
                    signal[..., 1:, :, :] += leak*(old[..., :-1, :, :]-old[..., 1:, :, :])
                    signal[..., :-1, :, :] += leak*(old[..., 1:, :, :]-old[..., :-1, :, :])
                    signal[..., :, 1:, :] += leak*(old[..., :, :-1, :]-old[..., :, 1:, :])
                    signal[..., :, :-1, :] += leak*(old[..., :, 1:, :]-old[..., :, :-1, :])
                    if not delta:
                        signal += self.xtalk
                    # qy/qx addition retains each quarter's independent spatial exchange.
                    coarse = (signal[:, :, 0]+signal[:, :, 1])+(signal[:, :, 2]+signal[:, :, 3])
                    if delta:
                        coarse += background[None, ps:ps+np_]
                    endpoints[:, ri, ps:ps+np_] = coarse if return_device else cp.asnumpy(coarse)
                    del signal, old, coarse
                del h0, hh
            if not bool(allocator.isfinite(endpoints).all()) or bool(allocator.any(endpoints < 0)):
                raise FloatingPointError('Invalid nonnegative physical expectation')
            yield begin, endpoints

    def render(self, candidates, **kwargs):
        allocator = self.cp if kwargs.get('return_device', False) else np
        return allocator.concatenate([block for _, block in self.iter_render(candidates, **kwargs)], axis=0)

    def close(self):
        for name in ('poses','fractions','weights','quarters','module','directions','distance','rho','cosine','identity','pulse','xtalk','_background_expected'):
            setattr(self, name, None)
        self.cp.cuda.get_current_stream().synchronize()
        self.cp.get_default_memory_pool().free_all_blocks()


def engineering_check(output, *, deadline_check=None, receipt_name='gpu_engineering_check.json'):
    """Constructed geometry only; no scientific scores or target truth read."""
    from cnh_displacement_ceiling_evaluate import skellam_logpmf_signed
    import cnh_unknown_target_gpu as G
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/receipt_name).exists():
        raise FileExistsError('Retain prior engineering receipt; do not overwrite')
    started = time.monotonic()
    if deadline_check is not None:
        deadline_check()
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[0, :3, :3] = S.ry(15)@S.rx(-10)
    poses[1, :3, :3] = S.ry(-15)@S.rx(-10)
    poses[2, :3, :3] = S.ry(0)@S.rx(0)
    poses[3, :3, :3] = S.ry(2.35)@S.rx(-10)
    poses[:, 2, 3] = [-1.35, -.9, -.8, -.5]
    background = [dict(lo=[.36, -.3, 1.3], hi=[.7, 1.3, 1.5], rho=.55),
        dict(lo=[-8., 1.65, -8.], hi=[8., 1.8, 9.], rho=.45),
        dict(lo=[-8., -3., 4.5], hi=[8., 2., 4.7], rho=.35)]
    candidates = []
    for side in (-1, 1):
        for group, (yc, height) in enumerate(((.08, .36), (.67, .34))):
            for width, h, depth in ((.11, height, .15), (.02, .04, .04)):
                for front in (.63, 1.36):
                    a = .285
                    xl, xh = (a, a+width) if side == 1 else (-a-width, -a)
                    candidates.append(dict(lo=[xl, yc-h/2, front], hi=[xh, yc+h/2, front+depth]))
    # An exact target/background AABB tie exercises target-first ordering.
    candidates.append(dict(lo=background[0]['lo'], hi=background[0]['hi']))
    tick = time.monotonic()
    engine = ExpectedRenderer(poses, background)
    initialization = time.monotonic()-tick
    metadata = engine.metadata
    try:
        tick = time.monotonic()
        endpoints = engine.render(candidates, candidate_batch=4, pose_batch=4, deadline_check=deadline_check)
        render_s = time.monotonic()-tick
        pure_reference = R.expected(dict(poses=poses, boxes=background))
        pure_background = engine.background_expected(return_device=False)
        background_max = float(np.max(np.abs(pure_background-pure_reference['expectation'])))
        np.testing.assert_allclose(pure_background, pure_reference['expectation'], atol=1e-9, rtol=1e-12)
        maximum, score_max, object_checks = 0., 0., 0
        rng = np.random.default_rng(2026100501)
        rhos = [0., .22+(.65-.22)/6, .435, ENDPOINT_RHO]
        tick = time.monotonic()
        for ci, candidate in enumerate(candidates):
            for rho in rhos:
                if deadline_check is not None:
                    deadline_check()
                boxes = [dict(**candidate, rho=rho), *background]
                reference = R.expected(dict(poses=poses, boxes=boxes))
                expectation = endpoints[ci, 0]+rho/ENDPOINT_RHO*(endpoints[ci, 1]-endpoints[ci, 0])
                delta = float(np.max(np.abs(reference['expectation']-expectation)))
                maximum = max(maximum, delta)
                np.testing.assert_allclose(expectation, reference['expectation'], atol=1e-9, rtol=1e-12)
                np.testing.assert_array_equal(engine.ambient, reference['ambient'])
                # Independent signed photon draws are solely an engineering check.
                if ci in (0, 7, len(candidates)-1):
                    ambient = 8*reference['ambient'][..., None]
                    observed = (rng.poisson(reference['expectation']+ambient)-rng.poisson(np.broadcast_to(ambient, expectation.shape))).astype(np.int32)
                    frozen = skellam_logpmf_signed(observed, reference['expectation'], ambient).sum(axis=(-3, -2, -1))
                    fast = G.skellam_logpmf_signed_gpu(observed, expectation, ambient).sum(axis=(-3, -2, -1))
                    err = float(np.max(np.abs(fast-frozen)))
                    score_max = max(score_max, err)
                    np.testing.assert_allclose(fast, frozen, atol=1e-8, rtol=0)
                    object_checks += 1
        cpu_parity_s = time.monotonic()-tick
        # Engineering timing repeats the same constructed targets and poses.
        tick = time.monotonic()
        repeated = engine.render(candidates, candidate_batch=4, pose_batch=4, deadline_check=deadline_check)
        repeat_s = time.monotonic()-tick
        np.testing.assert_allclose(repeated, endpoints, atol=1e-9, rtol=1e-12)
        receipt = dict(status='PASS', role='Engineering constructed backgrounds/targets only; no cohort scoring',
            candidates=len(candidates), physical_poses=len(poses), endpoint_rhos=[0., ENDPOINT_RHO],
            checked_rhos=rhos, full_expectation_max_abs=maximum, exact_frame_ll_max_abs=score_max,
            exact_likelihood_geometry_cases=object_checks, ambient_equal=True,
            pure_background_max_abs=background_max, candidate_implementation='delta first-visible target replacement only',
            initialization_s=initialization, first_render_s=render_s, warmed_render_s=repeat_s,
            frozen_cpu_parity_s=cpu_parity_s, total_s=time.monotonic()-started,
            template_pose_endpoints=len(candidates)*len(poses)*2,
            metadata=metadata, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_renderer_sha256=R.source_sha256(), limits=['FP64 regrouping parity is numerical, not bitwise',
                'Small constructed timing is not a full-scene runtime forecast',
                'No original observation or scientific class mixture scored'])
        (output/receipt_name).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf8')
        return receipt
    finally:
        engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=Path)
    parser.add_argument('--receipt-name', default='gpu_engineering_check.json')
    args = parser.parse_args()
    def check_deadline():
        if args.budget:
            record = json.loads(args.budget.read_text(encoding='utf8'))
            if time.time() >= record['deadline_unix_s']:
                raise TimeoutError('Authorized measurement wallclock budget reached')
    print(json.dumps(engineering_check(args.output, deadline_check=check_deadline, receipt_name=args.receipt_name), indent=2), flush=True)
