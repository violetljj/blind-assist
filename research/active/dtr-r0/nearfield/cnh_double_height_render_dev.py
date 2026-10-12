"""Isolated single-branch renderer for paired H/B/HB physical interventions.

Source loading is CPU-only. GPU initialization occurs only on entering the
DoubleHeightRunner context; this module never starts an experiment itself.
"""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'cnh-double-height-dev-20261007'
HEIGHTS = {'H':(-.10,.26), 'B':(.50,.84), 'HB':(-.10,.84)}
PHOTON_PREFIX = 2026100606
_SOURCE_CACHE = {}


def source_paths(unit):
    """Actual immutable source files, including the normalization bias."""
    unit = int(unit)
    if 98000 <= unit < 98048:
        folder, split = WORK/'cnh-extrinsic-aug-20261006', 'calibration'
    elif 99000 <= unit < 99096:
        folder, split = WORK/'cnh-extrinsic-aug-20261006/continuation-r1', 'evaluation'
    else:
        raise ValueError('Only the existing 98000/99000 Development units are supported')
    return dict(observations=folder/f'observations/{split}/unit{unit}.npz',
        truth=folder/f'truth/{split}/unit{unit}.json',
        head_query_scores=WORK/f'cnh-adaptive-query-dev-20261007/units/unit{unit}.npz',
        normalization_bias=WORK/'cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy')


def source_code_paths():
    """Direct renderer/inference sources for the main driver's freeze manifest."""
    here = Path(__file__).resolve().parent
    return {name:here/name for name in ('cnh_double_height_render_dev.py',
        'cnh_extrinsic_aug_data.py','cnh_location_reference_gpu.py','cnh_displacement_ceiling_render.py',
        'cnh_displacement_ceiling.py','cnh_temporal_readout_data.py','cnh_temporal_readout_model.py',
        'cnh_heading_uncertainty_dev.py','cnh_active_scan_dev.py','cnh_fused_projection.py','cnh_tristate_dev.py')}


def clear_source_cache():
    _SOURCE_CACHE.clear()


def query_with_error(public_query, errors):
    """Exactly the stored adaptive-query current-yaw rotation; no new estimator."""
    query = np.asarray(public_query).copy(); errors = np.asarray(errors)
    if query.shape != (16,4,4) or errors.shape != (16,):
        raise ValueError('16 poses and 16 stored head errors required')
    for f in range(16):
        angle = np.deg2rad(float(errors[f])); c,s = np.cos(angle),np.sin(angle)
        rotation = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        query[f,:3,:3] = rotation@public_query[f,:3,:3]
    return query


def load_source(unit, config, *, cache=None):
    """Load original branch=0 data and saved head-query outputs, no Torch/CuPy.

    The default cache keeps only the most recently requested unit. All returned
    arrays/boxes are independent copies, so caller intervention cannot alter it.
    """
    import cnh_tristate_dev as R
    unit,config = int(unit),int(config); paths=source_paths(unit)
    cache = _SOURCE_CACHE if cache is None else cache
    if cache.get('unit') != unit:
        with np.load(paths['observations']) as z:
            obs={key:z[key] for key in ('hist','ambient','z1','sensor_center','travel','noisy_center','public_query','configs')}
        with np.load(paths['head_query_scores']) as z:
            scores={key:z[key] for key in ('head_c_err','head_c_raw')}
        truth=json.loads(paths['truth'].read_text(encoding='utf8'))
        cache.clear();cache.update(unit=unit,obs=obs,scores=scores,truth=truth)
    obs,scores,truth=cache['obs'],cache['scores'],cache['truth']
    ids=np.flatnonzero(np.asarray(obs['configs'])==config)
    if len(ids)!=1: raise ValueError('config absent or duplicated in source observations')
    ci=int(ids[0]);scenes=[s for s in truth['scenes'] if int(s['config'])==config]
    if len(scenes)!=1: raise ValueError('config absent or duplicated in physical truth')
    public=np.asarray(obs['public_query'])
    if public.ndim==4: public=public[ci]
    err=scores['head_c_err'][ci].copy();raw=scores['head_c_raw'][0,ci].copy()
    result=dict(unit=unit,config=config,config_index=ci,boxes=copy.deepcopy(scenes[0]['boxes']),
        scene=copy.deepcopy(scenes[0]),source_paths=paths,
        hist=obs['hist'][0,ci].copy(),ambient=obs['ambient'][0,ci].copy(),z1=obs['z1'][0,ci].copy(),
        sensor_center=obs['sensor_center'].copy(),travel=obs['travel'].copy(),
        public_query=public.copy(),head_c_err=err,nn=obs['noisy_center'][ci].copy(),
        qq=query_with_error(public,err),raw=raw,smooth=R.smooth(raw),branch=0,
        photon_seed=int(np.random.SeedSequence([PHOTON_PREFIX,unit,config,0]).generate_state(1)[0]))
    expected={'hist':(16,8,8,16),'ambient':(16,8,8),'z1':(16,8,8,16),
        'sensor_center':(16,4,4),'travel':(16,4,4),'nn':(16,4,4),'qq':(16,4,4),'raw':(13,2),'smooth':(13,2)}
    for name,shape in expected.items():
        if result[name].shape!=shape or not np.isfinite(result[name]).all():
            raise ValueError(f'Invalid source {name}: expected finite {shape}, got {result[name].shape}')
    return result


def variant_boxes(source, tag):
    boxes=copy.deepcopy(source['boxes'])
    lo,hi=HEIGHTS[tag]
    boxes[0]['lo'][1]=lo;boxes[0]['hi'][1]=hi
    return boxes


def _canonical(value):
    return json.dumps(value,sort_keys=True,default=lambda x:np.asarray(x).tolist())


def source_result(source, tag='source'):
    return {**{k:source[k].copy() for k in ('hist','ambient','z1','nn','qq','raw','smooth')},
            'backend':dict(kind='reused immutable original single-sensor branch0 and head_c outputs',
                           tag=tag,unit=source['unit'],config=source['config'],photon_seed=source['photon_seed'])}


class DoubleHeightRunner:
    """One task-owned GPU engine; close restores local patches/environment."""
    def __init__(self, out=OUT):
        self.out=Path(out).resolve();self.runner=None;self._active=False;self._env={}
        self._setup_old=None;self._tempdir_old=None;self._pycache_old=None;self.cleanup_receipt={}

    def _set_caches(self):
        mapping={'TEMP':'tmp','TMP':'tmp','CUPY_CACHE_DIR':'cupy-cache','CUDA_CACHE_PATH':'cuda-cache',
            'TORCH_EXTENSIONS_DIR':'torch-extensions','TORCHINDUCTOR_CACHE_DIR':'torchinductor-cache',
            'TRITON_CACHE_DIR':'triton-cache','NUMBA_CACHE_DIR':'numba-cache','MPLCONFIGDIR':'mpl-cache',
            'PYTHONPYCACHEPREFIX':'pycache'}
        for key,folder in mapping.items():
            target=self.out/folder;target.mkdir(parents=True,exist_ok=True)
            if key not in self._env:self._env[key]=os.environ.get(key)
            os.environ[key]=str(target)
        tempfile.tempdir=str(self.out/'tmp');sys.pycache_prefix=str(self.out/'pycache')

    def __enter__(self):
        if self._active:raise RuntimeError('Runner context already active')
        self._active=True;self._env['PATH']=os.environ.get('PATH')
        self._tempdir_old=tempfile.tempdir;self._pycache_old=sys.pycache_prefix
        try:
            self._set_caches()
            import cnh_active_scan_dev as A
            self.A=A;self._setup_old=A.setup_gpu;A.setup_gpu=self._set_caches
            import cnh_extrinsic_aug_data as D
            import cnh_heading_uncertainty_dev as HU
            self.D=D
            # Keep a partial runner reachable for cleanup if initialization fails.
            self.runner=HU.Runner.__new__(HU.Runner)
            self.runner.__init__()
            return self
        except BaseException:
            self.close();raise

    def _require_open(self):
        if not self._active or self.runner is None:raise RuntimeError('Enter the DoubleHeightRunner context first')

    def render_observation(self, source, boxes, tag):
        """Real single-sensor raycast/electronics/sample; no inference yet."""
        self._require_open();start=time.monotonic()
        hist,ambient,z1,backend=self.D.render(copy.deepcopy(boxes),source['sensor_center'],[0.],
            source['unit'],source['config'],{'photon_seed_prefix':PHOTON_PREFIX})
        return dict(hist=hist[0],ambient=ambient[0],z1=z1[0],nn=source['nn'].copy(),qq=source['qq'].copy(),
            backend=dict(kind='new physical branch0 rendering',tag=tag,unit=source['unit'],config=source['config'],
                         photon_seed=source['photon_seed'],render_seconds=time.monotonic()-start,renderer=backend))

    def infer_observations(self, items):
        """Batch existing rendered items across configuration axis; modifies only items."""
        import cnh_tristate_dev as R
        self._require_open()
        if not items:return items
        start=time.monotonic()
        raw=self.runner.raw(np.stack([v['z1'] for v in items]),
            np.stack([v['nn'] for v in items]),np.stack([v['qq'] for v in items]))
        for i,item in enumerate(items):
            item['raw']=raw[i];item['smooth']=R.smooth(raw[i])
            item['backend']['inference_batch_size']=len(items)
            item['backend']['inference_batch_seconds']=time.monotonic()-start
        return items

    def render_variant(self, source, boxes, tag, *, allow_reuse=True):
        if allow_reuse and _canonical(boxes)==_canonical(source['boxes']):return source_result(source,tag)
        return self.infer_observations([self.render_observation(source,boxes,tag)])[0]

    def anchor_parity(self, source, *, ambient_atol=1e-6,z1_atol=.002,raw_atol=.005):
        """Explicit unchanged-anchor check; never changes thresholds or tolerances."""
        start=time.monotonic()
        actual=self.render_variant(source,source['boxes'],'unchanged_anchor_parity',allow_reuse=False)
        tolerances={'hist':0.,'ambient':ambient_atol,'z1':z1_atol,'raw':raw_atol,'smooth':raw_atol,'nn':0.,'qq':0.}
        checks={}
        for key,tolerance in tolerances.items():
            left,right=np.asarray(actual[key]),np.asarray(source[key])
            same_shape=left.shape==right.shape
            delta=float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)))) if same_shape else None
            checks[key]=dict(exact=bool(np.array_equal(left,right)),max_abs=delta,atol=float(tolerance),
                             passed=bool(same_shape and np.isfinite(left).all() and delta<=tolerance))
        return dict(passed=all(v['passed'] for v in checks.values()),unit=source['unit'],config=source['config'],
            checks=checks,seconds=time.monotonic()-start,backend=actual['backend'])

    def close(self):
        if not self._active:return
        errors=[];eng=getattr(self.runner,'eng',None);torch=getattr(eng,'torch',None)
        cp=getattr(getattr(self.runner,'F',None),'cp',None)
        try:
            if torch is not None:torch.cuda.synchronize()
            if cp is not None:cp.cuda.get_current_stream().synchronize()
        except Exception as e:errors.append(repr(e))
        if self.runner is not None:
            self.runner.F=None;self.runner.eng=None
        if eng is not None:
            eng.nets=[];eng.masks=None;eng.projector=None
            dll=getattr(eng,'_dll',None)
            if dll is not None:dll.close()
        eng=None;self.runner=None;gc.collect()
        try:
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
            if torch is not None:torch.cuda.empty_cache()
        except Exception as e:errors.append(repr(e))
        finally:
            if self._setup_old is not None:self.A.setup_gpu=self._setup_old
            for key,value in self._env.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            tempfile.tempdir=self._tempdir_old;sys.pycache_prefix=self._pycache_old
            self._env={};self._active=False;clear_source_cache()
        self.cleanup_receipt=dict(closed=True,cleanup_errors=errors)

    def __exit__(self,exc_type,exc,tb):
        self.close();return False
