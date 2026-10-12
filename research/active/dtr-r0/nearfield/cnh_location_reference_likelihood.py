"""Tiled FP64 exact signed-Skellam evidence for a location-mixture reference.

Inputs are photons, ambient rates and zero-noise endpoint templates only.
Each candidate remains fixed across its whole causal window. A shared background
log likelihood is subtracted cellwise before candidate marginalization. This
common additive constant cancels exactly from every class likelihood ratio;
cells whose expectation equals the background need no repeated series sum.
No numerical tolerance is used to decide whether two expectations are equal.
Importing this module does not initialize CUDA.
"""
from __future__ import annotations

import time
import numpy as np
import cnh_unknown_target_gpu as G
from cnh_readout_pilot2_reference import PAST, START

RHOS = np.array([.22+(.65-.22)*(i+.5)/3 for i in range(3)], np.float64)
ENDPOINT_RHO = .65
_MODULE = None

CUDA_SOURCE = G.CUDA_SOURCE.split('extern "C" __global__ void skellam_elementwise')[0] + r'''
extern "C" __global__ void background_cells(
    const int *observed,const double *expected,const double *ambient,
    double *result,long long size,int nk,int npast,double tol,int terms,int *error) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size)return;
    int cell=i%1024, pose=(i/1024)%npast;
    long long eidx=i%((long long)nk*npast*1024);
    result[i]=exact_logpmf(observed[i],expected[eidx],
                          8.0*ambient[pose*64+cell/16],tol,terms,error,i);
}
extern "C" __global__ void candidate_cells(
    const int *observed,const double *endpoints,const double *ambient,
    const double *background_expected,const double *background_ll,
    const double *alphas,double *result,long long size,int nk,int npast,
    double tol,int terms,int *error) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size)return;
    int cell=i%1024;
    long long j=i/1024;
    int pose=j%npast; j/=npast;
    int k=j%nk; j/=nk;
    int observed_variant=j%7; j/=7;
    int reflectance=j%3; int candidate=j/3;
    long long local=((long long)k*npast+pose)*1024+cell;
    long long ep0=(long long)candidate*2*nk*npast*1024+local;
    long long ep1=ep0+(long long)nk*npast*1024;
    double e0=endpoints[ep0], ehi=endpoints[ep1];
    double expected=e0+(ehi-e0)*alphas[reflectance];
    double base=background_expected[local];
    if(expected==base) {result[i]=0.0;return;}
    long long obs=((long long)observed_variant*nk*npast+k*npast+pose)*1024+cell;
    result[i]=exact_logpmf(observed[obs],expected,
             8.0*ambient[pose*64+cell/16],tol,terms,error,i)-background_ll[obs];
}
'''


def _runtime():
    global _MODULE
    import cupy as cp
    if _MODULE is None:
        _MODULE = cp.RawModule(code=CUDA_SOURCE, options=('--std=c++11',),
                              name_expressions=('background_cells', 'candidate_cells'))
    return cp, _MODULE


class ExactWindowScorer:
    """Cache one unit; ``scores`` returns relative LL[C,3,7,K,13], timing.

    hist is signed int32 [7,K,16,8,8,16], ambient [16,8,8]. Endpoint input is
    contiguous cupy.float64 [C,2,K*94,8,8,16], endpoint rho=0/.65. Optional
    background expectation is cupy.float64 [K*94,8,8,16]. Set it once, before
    scoring scientific candidates. Returned LL excludes the same background
    constant for every candidate, reflectance and hypothesis in each window.
    A four-candidate tile at K=4 uses about 260 MiB cell workspace.
    """
    likelihood_is_relative = True

    def __init__(self, hist, ambient, background_expected=None):
        hist, ambient = np.asarray(hist), np.asarray(ambient, dtype=np.float64)
        if (hist.ndim != 6 or hist.shape[0] != 7 or hist.shape[2:] != (16,8,8,16)
                or hist.shape[1] not in (1,2,4)):
            raise ValueError('hist must have [7,K,16,8,8,16], K=1/2/4')
        if ambient.shape != (16,8,8) or not np.isfinite(ambient).all() or np.any(ambient<0):
            raise ValueError('Finite nonnegative ambient[16,8,8] required')
        cp, self.module = _runtime()
        self.cp, self.k, self.npast = cp, hist.shape[1], len(PAST)
        self.hist = cp.asarray(G._observations(hist[:,:,PAST]).reshape(7,self.k,self.npast,1024))
        self.ambient = cp.asarray(np.ascontiguousarray(ambient[PAST]))
        self.alphas = cp.asarray(RHOS/ENDPOINT_RHO)
        self.background_expected = self.background_ll = self.background_windows = None
        self.background_seconds = 0.0
        if background_expected is not None:
            self.set_background(background_expected)

    def set_background(self, expected):
        if self.background_expected is not None:
            raise RuntimeError('Background cache is immutable for a scoring unit')
        cp = self.cp
        if not isinstance(expected, cp.ndarray) or expected.dtype != cp.float64:
            raise ValueError('Background must be a cupy.float64 array')
        if expected.shape != (self.k*self.npast,8,8,16):
            raise ValueError('Background axes must be [K*94,8,8,16]')
        self.background_expected = cp.ascontiguousarray(expected)
        # Renderer endpoints guarantee finite nonnegative rates; check the cache
        # once rather than rechecking every repeated candidate tile.
        if not bool(cp.all(cp.isfinite(expected))) or bool(cp.any(expected<0)):
            raise ValueError('Nonfinite or negative background expectation')
        cp.cuda.get_current_stream().synchronize()
        tick = time.perf_counter()
        self.background_ll = cp.empty(self.hist.shape, cp.float64)
        error = cp.zeros(2, cp.int32)
        self.module.get_function('background_cells')(((self.hist.size+255)//256,), (256,),
            (self.hist,self.background_expected,self.ambient,self.background_ll,
             np.int64(self.hist.size),np.int32(self.k),np.int32(self.npast),
             np.float64(G.RELATIVE_TOLERANCE),np.int32(G.MAX_TERMS),error))
        G._raise_kernel_error(cp,error)
        frames = self.background_ll.sum(axis=-1,dtype=cp.float64)
        self.background_windows = cp.stack([frames[...,a:b].sum(axis=-1,dtype=cp.float64)
                                           for a,b in zip(START[:-1],START[1:])],axis=-1)
        cp.cuda.get_current_stream().synchronize()
        self.background_seconds = time.perf_counter()-tick

    def scores(self, endpoints, background_expected=None):
        """Return (numpy LL[C,3,7,K,13], timing); no expected CPU transfer."""
        if background_expected is not None:
            self.set_background(background_expected)
        if self.background_expected is None:
            raise RuntimeError('A shared background expectation must be set before scoring')
        cp = self.cp
        if (not isinstance(endpoints,cp.ndarray) or endpoints.dtype != cp.float64
                or endpoints.ndim != 6 or endpoints.shape[1:] != (2,self.k*self.npast,8,8,16)
                or not endpoints.flags.c_contiguous or len(endpoints)<1):
            raise ValueError('Contiguous cupy.float64 endpoints[C,2,K*94,8,8,16] required')
        shape = (len(endpoints),3,7,self.k,self.npast,1024)
        workspace_bytes = int(np.prod(shape,dtype=np.int64))*8
        if workspace_bytes > 2*1024**3:
            raise ValueError('Candidate tile exceeds 2 GiB likelihood workspace')
        cp.cuda.get_current_stream().synchronize()
        tick = time.perf_counter()
        cells = cp.empty(shape,cp.float64)
        error = cp.zeros(2,cp.int32)
        self.module.get_function('candidate_cells')(((cells.size+255)//256,), (256,),
            (self.hist,endpoints,self.ambient,self.background_expected,self.background_ll,
             self.alphas,cells,np.int64(cells.size),np.int32(self.k),np.int32(self.npast),
             np.float64(G.RELATIVE_TOLERANCE),np.int32(G.MAX_TERMS),error))
        G._raise_kernel_error(cp,error)
        frames = cells.sum(axis=-1,dtype=cp.float64)
        del cells
        windows = cp.stack([frames[...,a:b].sum(axis=-1,dtype=cp.float64)
                            for a,b in zip(START[:-1],START[1:])],axis=-1)
        result = cp.asnumpy(windows)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite relative scene likelihood; preserve denominator')
        return result,dict(likelihood_s=time.perf_counter()-tick,
                           cell_workspace_bytes=workspace_bytes,
                           candidate_count=len(endpoints),background_s=self.background_seconds,
                           exact_equal_cache=True,relative_log_likelihood=True)

    def close(self):
        """Release task-owned GPU arrays; leave the shared device pool alone."""
        self.hist = self.ambient = self.alphas = None
        self.background_expected = self.background_ll = self.background_windows = None


def engineering_check():
    """Constructed photons: axis, cache, full-window and class-ratio parity."""
    from cnh_displacement_ceiling_evaluate import skellam_logpmf_signed as cpu
    from scipy.special import logsumexp
    cp, _ = _runtime()
    rng = np.random.default_rng(2026100419)
    k, c = 1, 2
    ambient = rng.uniform(.05,2,(16,8,8))
    background = rng.uniform(0,10,(k*len(PAST),8,8,16))
    endpoints = np.broadcast_to(background[None,None],(c,2,*background.shape)).copy()
    # Sparse deterministic changes cover opaque-zero, rho-dependent signals,
    # all variants/poses/candidates without creating a scientific result.
    endpoints[0,0,:,0,0,:4] *= .2
    endpoints[0,1,:,0,0,:4] += 4
    endpoints[1,0,:,7,7,8:] *= .7
    endpoints[1,1,:,7,7,8:] += 7
    hist = (rng.poisson(10+8*ambient[...,None],size=(7,k,16,8,8,16))
            -rng.poisson(8*ambient[...,None],size=(7,k,16,8,8,16))).astype(np.int32)
    started = time.perf_counter()
    scorer = ExactWindowScorer(hist,ambient,cp.asarray(background))
    try:
        actual,timing = scorer.scores(cp.asarray(endpoints))
        base = cpu(hist[:,:,PAST],background.reshape(k,len(PAST),8,8,16)[None],
                   8*ambient[PAST][None,None,...,None]).sum(axis=(-3,-2,-1))
        expected = np.empty_like(actual)
        for candidate in range(c):
            for r,alpha in enumerate(RHOS/.65):
                e = endpoints[candidate,0]+alpha*(endpoints[candidate,1]-endpoints[candidate,0])
                ll = cpu(hist[:,:,PAST],e.reshape(k,len(PAST),8,8,16)[None],
                         8*ambient[PAST][None,None,...,None]).sum(axis=(-3,-2,-1))
                difference = ll-base
                expected[candidate,r] = np.stack([difference[...,a:b].sum(axis=-1)
                                                  for a,b in zip(START[:-1],START[1:])],axis=-1)
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=3e-12)
        a = logsumexp(actual[0],axis=0)-logsumexp(actual[1],axis=0)
        b = logsumexp(expected[0],axis=0)-logsumexp(expected[1],axis=0)
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=3e-12)
        return dict(status='PASS',shape=list(actual.shape),
                    max_relative_ll_abs_error=float(np.max(np.abs(actual-expected))),
                    max_llr_abs_error=float(np.max(np.abs(a-b))),
                    elapsed_s=time.perf_counter()-started,timing=timing,
                    background_subtraction_common_to_all_candidates=True)
    finally:
        scorer.close()
