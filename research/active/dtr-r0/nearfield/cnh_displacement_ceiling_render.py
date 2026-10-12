"""Frozen CNH expectation and exact aggregate shot noise for displacement probes.

No model or label input. Each coarse 8x8x16 bin aggregates four angular
quadrants times eight raw radial bins (32 terms), not a retained 128-bin count.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import sys

import numpy as np

import cnh_proposal_attribution_scenes as S

SCHEMA = 'cnh.displacement.coarse32-skellam.v1'


def source_sha256():
    """Current task renderer and actual imported frozen sensor dependencies."""
    modules = (S, sys.modules[S.synthesize_response.__module__],
               sys.modules[S.reference_parameters.__module__])
    paths = (Path(__file__).resolve(), *(Path(m.__file__).resolve() for m in modules))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def expected(scene, sub=16):
    """Coarse expectation/ambient and first-visible IDs, with fixed physical poses.

    expectation[N,8,8,16] includes residual crosstalk, and is the signed
    histogram mean at gain1. ambient[N,8,8] is four-quarter ambient per raw
    radial bin; one returned coarse radial bin has Poisson background8*ambient.
    object_id[N,8,8,sub*sub] retains original zone/subray ordering (-1=miss).
    sub32 only refines the geometric quadrature, preserving photon budgets.
    """
    if sub not in (16,32):
        raise ValueError('Only declared main16/sensitivity32 quadrature is supported')
    directions, weights = S.angular_rays(sub)
    poses = np.asarray(scene['poses'],dtype=np.float64)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected finite nonempty sensor pose sequence')
    hits = [S.raycast_boxes(p[:3,3],directions@p[:3,:3].T,scene['boxes']) for p in poses]
    shape = (len(poses),8,8,sub,sub)
    distance = np.stack([h['distance'] for h in hits]).reshape(shape)
    reflectance = np.stack([h['rho'] for h in hits]).reshape(shape)
    cosine = np.stack([h['cos'] for h in hits]).reshape(shape)
    object_id = np.stack([h['object_id'] for h in hits]).astype(np.int64)
    w = weights.reshape(8,8,sub,sub)
    parameters,_ = S.nominal_parameters()
    if parameters.output_gain!=1. or parameters.noise_scale!=1.:
        raise ValueError('Aggregate count sampler requires frozen gain1/noiseScale1')
    quarter = replace(parameters,signal_counts=parameters.signal_counts/4,
                      ambient_counts=parameters.ambient_counts/4,noise_scale=0.)
    fine = np.empty((len(poses),16,16,16),np.float64)
    fine_ambient = np.empty((len(poses),16,16),np.float64)
    half = sub//2
    for qy in range(2):
        for qx in range(2):
            ys,xs = slice(half*qy,half*(qy+1)),slice(half*qx,half*(qx+1))
            qw = w[:,:,ys,xs].reshape(8,8,half*half)
            fraction = qw.sum(-1)/weights.sum(-1)
            ray_shape = (len(poses),8,8,half*half)
            rho = reflectance[:,:,:,ys,xs].reshape(ray_shape)*(4*fraction[None,:,:,None])
            if rho.max()>1:
                raise ValueError('Quarter reflectance scaling exceeds frozen sensor domain')
            quiet = S.synthesize_response(distance[:,:,:,ys,xs].reshape(ray_shape),rho,
                cosine[:,:,:,ys,xs].reshape(ray_shape),qw,params=quarter,seed=0)
            fine[:,qy::2,qx::2] = quiet['histogram'].reshape(len(poses),8,8,16,8).sum(-1)
            fine_ambient[:,qy::2,qx::2] = quiet['ambient']
    mean = fine.reshape(len(poses),8,2,8,2,16).sum(axis=(2,4))
    ambient = fine_ambient.reshape(len(poses),8,2,8,2).sum(axis=(2,4))
    if not np.isfinite(mean).all() or np.any(mean<0) or not np.isfinite(ambient).all() or np.any(ambient<0):
        raise ValueError('Invalid frozen coarse Poisson expectation/background')
    return dict(schema=SCHEMA,expectation=mean,ambient=ambient,object_id=object_id,
                sensor_params=asdict(parameters),sub=sub,
                count_semantics='8x8x16; each radial bin sums4 angular quadrants x8 raw radial bins=32 aggregate terms')


def sample(expectation, ambient, seed):
    """Exact marginal aggregate Skellam law; independent counts/background draws.

    At frozen noiseScale1/gain1, signed=Poisson(E+8ambient)-Poisson(8ambient).
    This is equal in distribution to frozen quarter/radial aggregation, not
    bitwise equal to its per-quarter RNG ordering for the same seed.
    Returns signed,counts,ambient_estimate as int32 coarse32aggregate arrays.
    """
    mean = np.asarray(expectation,dtype=np.float64)
    ambient = np.asarray(ambient,dtype=np.float64)
    if mean.ndim!=4 or mean.shape[1:]!=(8,8,16) or ambient.shape!=mean.shape[:-1]:
        raise ValueError('Expected coarse mean[N,8,8,16],ambient[N,8,8]')
    if not np.isfinite(mean).all() or np.any(mean<0) or not np.isfinite(ambient).all() or np.any(ambient<0):
        raise ValueError('Need finite nonnegative Poisson means')
    rng = np.random.default_rng(int(seed))
    background_mean = np.broadcast_to(8*ambient[...,None],mean.shape)
    background = rng.poisson(background_mean)
    counts = rng.poisson(mean+background_mean)
    signed = counts-background
    limit = np.iinfo(np.int32)
    if any(x.min()<limit.min or x.max()>limit.max for x in (signed,counts,background)):
        raise OverflowError('Aggregate count exceeds int32 storage')
    return signed.astype(np.int32),counts.astype(np.int32),background.astype(np.int32)


def check():
    # Two engineering exposures, not a scientific cohort or model run.
    poses = np.repeat(np.eye(4)[None],2,axis=0)
    poses[:,:3,:3] = S.ry(15)@S.rx(-10)
    poses[0,2,3] = -.16
    scene = dict(poses=poses,boxes=[dict(lo=[.29,-.1,1.2],hi=[.39,.26,1.35],rho=.5),
        dict(lo=[-8,-3,4.5],hi=[8,2,4.7],rho=.35)])
    base = expected(scene,16)
    reference = S.render(scene,123,noise_scale=0)
    np.testing.assert_allclose(base['expectation'],reference['hist'],rtol=0,atol=1e-9)
    np.testing.assert_array_equal(base['ambient'],reference['ambient'])
    refined = expected(scene,32)
    assert base['object_id'].shape==(2,8,8,256) and refined['object_id'].shape==(2,8,8,1024)
    np.testing.assert_array_equal(base['ambient'],refined['ambient'])
    a,b,c = sample(base['expectation'],base['ambient'],42)
    assert a.dtype==b.dtype==c.dtype==np.int32
    np.testing.assert_array_equal(a,b-c)
    repeated = sample(base['expectation'],base['ambient'],42)
    for x,y in zip((a,b,c),repeated):
        np.testing.assert_array_equal(x,y)
    zero = np.zeros_like(base['expectation'])
    signed,counts,background = sample(zero,np.zeros_like(base['ambient']),0)
    assert not signed.any() and not counts.any() and not background.any()
    print('PASS frozen sub16 expectation parity<=1e-9, sub32 axes/budget, coarse32aggregate count identity and RNG repeatability')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('check',))
    args=parser.parse_args();check()
