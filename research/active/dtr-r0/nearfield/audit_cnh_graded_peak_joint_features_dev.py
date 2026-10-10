"""Independent current top8 descriptors and exact audited-track lineage check."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/cnh-graded-peak-track-dev-20261010/tracks'
OUTPUT=ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/features'
NAMES=('rank0_peak_log','top8_peak_log_sum','inner_weighted_peak_log_sum','inner_weighted_peak_log_max',
       'strongest_inner_peak_log','inner_weighted_sum_share','inner_supported_candidate_count',
       'rank0_depth_m','rank0_x_m','rank0_y_m','rank0_support_x_extent_m','rank0_support_y_extent_m','rank0_support_z_extent_m',
       'strongest_inner_depth_m','strongest_inner_x_m','strongest_inner_y_m','strongest_inner_share',
       'strongest_inner_support_x_extent_m','strongest_inner_support_y_extent_m','strongest_inner_support_z_extent_m',
       'rank0_minus_inner_depth_m','rank0_to_inner_peak_log_ratio')


def direct(a):
    shape=a['candidate_amplitude'].shape[:-1]
    result=np.full((*shape,22),np.nan,np.float32)
    amplitude=a['candidate_amplitude']; valid=a['candidate_valid']; share=a['candidate_inner_share']
    xyz=a['candidate_xyz']; low=a['candidate_support_low_xyz']; high=a['candidate_support_high_xyz']
    count=np.zeros(shape,np.int32); total=np.zeros(shape,np.float32); inside=np.zeros(shape,np.float32)
    maximum=np.zeros(shape,np.float32); chosen=np.full(shape,-1,np.int8); best=np.full(shape,-np.inf,np.float32)
    for rank in range(8):
        allow=valid[...,rank]; supported=allow&(share[...,rank]>0)
        total+=np.where(allow,amplitude[...,rank],0)
        weighted=np.where(allow,amplitude[...,rank]*share[...,rank],0)
        inside+=weighted; maximum=np.maximum(maximum,weighted); count+=supported
        improve=supported&(amplitude[...,rank]>best)
        chosen[improve]=rank; best[improve]=amplitude[...,rank][improve]
    present=valid.any(-1); inner=count>0
    result[...,1]=np.where(present,total,np.nan)
    result[...,6]=np.where(present,count,np.nan)
    result[...,2]=np.where(inner,inside,np.nan); result[...,3]=np.where(inner,maximum,np.nan)
    result[...,5]=np.divide(inside,total,out=np.full(shape,np.nan,np.float32),where=inner&(total>0))
    rank0=valid[...,0]
    result[...,0]=np.where(rank0,amplitude[...,0],np.nan)
    extent0=high[...,0,:]-low[...,0,:]
    for output,axis in ((7,2),(8,0),(9,1)):
        result[...,output]=np.where(rank0,xyz[...,0,axis],np.nan)
    for axis in range(3): result[...,10+axis]=np.where(rank0,extent0[...,axis],np.nan)
    for rank in range(8):
        mask=chosen==rank
        result[...,4][mask]=amplitude[...,rank][mask]
        for output,axis in ((13,2),(14,0),(15,1)):
            result[...,output][mask]=xyz[...,rank,axis][mask]
        result[...,16][mask]=share[...,rank][mask]
        extent=high[...,rank,:]-low[...,rank,:]
        for axis in range(3): result[...,17+axis][mask]=extent[...,axis][mask]
    result[...,20]=result[...,7]-result[...,13]
    result[...,21]=np.divide(result[...,0],result[...,4],out=np.full(shape,np.nan,np.float32),where=inner&(result[...,4]>0))
    return result,np.isfinite(result)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<22),b''): h.update(block)
    return h.hexdigest()


def audit():
    began=time.monotonic(); destination=OUTPUT/'audit'
    if (destination/'result.json').exists(): raise FileExistsError('Preserve existing joint feature audit')
    destination.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',CPU_seconds_cap=30,GPU_seconds=0,checks=0,max_absolute_difference=0.,splits={})
    for split in ('cal','validation'):
        with np.load(SOURCE/f'{split}_tracks.npz') as a: old={key:a[key] for key in a.files}
        with np.load(OUTPUT/f'{split}_features.npz') as a: new={key:a[key] for key in a.files}
        expected,valid=direct(old)
        np.testing.assert_array_equal(new['current_names'],NAMES)
        np.testing.assert_array_equal(new['temporal_names'],(*NAMES,*(f'track_{n}' for n in old['names'])))
        np.testing.assert_array_equal(new['current_valid'],valid)
        np.testing.assert_array_equal(np.isfinite(new['current_features']),valid)
        np.testing.assert_allclose(new['current_features'],expected,atol=3e-6,rtol=3e-6,equal_nan=True)
        result['max_absolute_difference']=max(result['max_absolute_difference'],float(abs(new['current_features'][valid]-expected[valid]).max()))
        np.testing.assert_array_equal(new['temporal_features'][...,:22],new['current_features'])
        np.testing.assert_array_equal(new['temporal_valid'][...,:22],valid)
        np.testing.assert_array_equal(new['temporal_features'][...,22:],old['features'])
        np.testing.assert_array_equal(new['temporal_valid'][...,22:],old['valid'])
        for key in ('scene_ids','scene_uids','replica','frames','queries','cache_row_index'):
            np.testing.assert_array_equal(new[key],old[key]); result['checks']+=old[key].size
        assert np.all(np.isnan(new['current_features'][~valid]))
        counts=expected[...,6]
        zero=np.isfinite(counts)&(counts==0)
        assert np.all(~valid[...,2][zero]) and np.all(~valid[...,5][zero])
        result['checks']+=expected.size*2+new['temporal_features'].size*2
        result['splits'][split]=dict(query_slots=int(np.prod(expected.shape[:-1])),invalid_current_features=int((~valid).sum()),
            valid_zero_inner_count=int(zero.sum()),strongest_inner_not_rank0=int((valid[...,4]&(expected[...,4]!=expected[...,0])).sum()))
        if time.monotonic()-began>=30: raise TimeoutError('Joint feature audit30s cap reached')
    # Missing-inner and strongest-original-amplitude semantics are not guaranteed
    # to occur in the real retained top8 list, so test only these explicit cases.
    import cnh_graded_peak_joint_features_dev as P
    shape=(3,8)
    a=dict(candidate_amplitude=np.zeros(shape,np.float32),candidate_valid=np.zeros(shape,bool),
           candidate_inner_share=np.zeros(shape,np.float32),candidate_xyz=np.full((*shape,3),np.nan,np.float32),
           candidate_support_low_xyz=np.full((*shape,3),np.nan,np.float32),candidate_support_high_xyz=np.full((*shape,3),np.nan,np.float32))
    for slot in (0,1):
        a['candidate_valid'][slot,:2]=True; a['candidate_amplitude'][slot,:2]=(4,3)
        a['candidate_xyz'][slot,:2]=((.1,.3,1),(.2,.4,2))
        a['candidate_support_low_xyz'][slot,:2]=a['candidate_xyz'][slot,:2]-.01
        a['candidate_support_high_xyz'][slot,:2]=a['candidate_xyz'][slot,:2]+.01
    a['candidate_inner_share'][1,:2]=(.1,1)
    expected,valid=direct(a); actual,mask=P.build(a)
    np.testing.assert_array_equal(mask,valid); np.testing.assert_array_equal(actual,expected)
    assert valid[0,6] and expected[0,6]==0 and not valid[0,4]
    assert expected[1,4]==4 and expected[1,3]==3 and expected[1,13]==1
    assert not valid[2].any()
    np.savez_compressed(destination/'missing_and_inner_choice_toy.npz',expected=expected,valid=valid,**a)
    result.update(CPU_seconds=time.monotonic()-began,toy_cases=['retained positive peaks without inner support: count0valid, inner descriptors missing',
        'strongest inner uses original amplitude4, while weighted maximum3 belongs to another peak','no positive retained candidates: descriptors missing'],
        audit_source_sha256=sha(Path(__file__)),feature_source_sha256=sha(Path(P.__file__)),command=[sys.executable,*sys.argv],
        limitation='Audits declared public observations and exact lineage; peak membership is not target existence, physical intrusion, coverage or free evidence.')
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    inputs=[base/f'{split}_{kind}.npz' for base,kind in ((SOURCE,'tracks'),(OUTPUT,'features')) for split in ('cal','validation')]
    receipt=dict(CPU_seconds=time.monotonic()-began,GPU_seconds=0,input_sha256={str(p.relative_to(ROOT/'artifacts.local')):sha(p) for p in inputs},
        audit_source_sha256=result['audit_source_sha256'],feature_source_sha256=result['feature_source_sha256'],
        output_sha256={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='execution_receipt.json'},command=result['command'])
    (destination/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__': audit()
