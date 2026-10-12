"""Descriptive +/-22.5 degree public geometry on retained natural96000.

One CPU process; no rendering, truth boxes, labels, model inference or fitting.
Compare with the already sealed +/-15 degree coverage, at identical query keys.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import cnh_dual_gated_geometry as G

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005/diagnostic'
SOURCE=OUT/'coverage_96000.npz'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def save(path,value):
    path=Path(path);assert not path.exists(),'Existing evidence preserved: '+str(path)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def distribution(a):
    a=np.asarray(a,float).ravel()
    return dict(n=len(a),minimum=float(a.min()),q10=float(np.quantile(a,.1)),q25=float(np.quantile(a,.25)),
        median=float(np.median(a)),q75=float(np.quantile(a,.75)),q90=float(np.quantile(a,.9)),maximum=float(a.max()))

def run():
    from cnh_cvr_pilot import motion_metadata
    tick=time.monotonic();started=time.time();source_hash=sha(SOURCE);geometry_hash=sha(G.__file__)
    assert read(OUT/'coverage_result.json')['coverage_script_sha256']==geometry_hash
    assert not (OUT/'geometry_splay22p5.json').exists()
    with np.load(SOURCE) as data:old={key:data[key].copy() for key in data.files}
    assert np.array_equal(old['query'].reshape(-1,2),np.tile([0,1],(3840,1)))
    for key in ('unit','config'):np.testing.assert_array_equal(old[key][::2],old[key][1::2])
    units=old['unit'][::2];configs=old['config'][::2];c15=np.stack((old['coverage'][::2],old['coverage'][1::2]),-1)
    assert c15.shape==(3840,2,13,2)
    c22=[];axes15=[];axes22=[]
    for k,(unit,config) in enumerate(zip(units,configs)):
        noisy=motion_metadata(int(unit),int(config))[2]
        q22=G.estimated_query_poses(noisy,angles=(-22.5,22.5));c22.append(G.query_coverage(q22))
        q15=G.estimated_query_poses(noisy,angles=(-15,15))
        # These axes are measured relative to the causal estimated travel tangent.
        axes15.append(q15[...,:3,2]);axes22.append(q22[...,:3,2])
        if (k+1)%400==0:print('splay22p5 geometry',k+1,'/',len(units),'seconds',round(time.monotonic()-tick,1),flush=True)
    c22=np.asarray(c22);axes15=np.asarray(axes15);axes22=np.asarray(axes22)
    assert np.isfinite(c22).all() and (c22>=0).all() and (c22<=1).all()
    stats={};optics={}
    for label,coverage,axes in [('dual15',c15,axes15),('dual22p5',c22,axes22)]:
        stats[label]={};optics[label]={}
        for mode in range(3):
            keep=units%3==mode;stats[label][f'mode{mode}']={};optics[label][f'mode{mode}']={}
            for b,branch in enumerate(('L','R')):
                ax=axes[keep,b];yaw=np.rad2deg(np.arctan2(ax[...,0],ax[...,2]));pitch=np.rad2deg(np.arctan2(ax[...,1],np.hypot(ax[...,0],ax[...,2])))
                optics[label][f'mode{mode}'][branch]=dict(optical_yaw_relative_estimated_travel_deg=distribution(yaw),downward_optical_elevation_deg=distribution(pitch))
                for q,query in enumerate(('HEAD','BODY')):
                    values=coverage[keep,b,:,q].ravel()
                    stats[label][f'mode{mode}'][f'{query}/{branch}']=dict(**distribution(values),
                        eligible={str(c):dict(numerator=int((values>=c).sum()),denominator=len(values),rate=float((values>=c).mean())) for c in (.3,.5,.7)})
    assert sha(SOURCE)==source_hash and sha(G.__file__)==geometry_hash
    payload=OUT/'geometry_splay22p5.npz';assert not payload.exists()
    np.savez_compressed(payload,coverage22p5=c22,axes15=axes15,axes22p5=axes22,unit=units,config=configs,frames=G.FRAMES,
        query=['HEAD','BODY'],sensor=['L','R'])
    result=dict(status='COMPLETE',scope='Descriptive geometry only; not a candidate, calibration, score or main verdict arm',
        units=96,configs_per_unit=40,configurations=3840,frames_per_configuration=13,sensor_query_frame_cells_per_splay=int(c22.size),
        nominal_installation_splay_deg=dict(dual15=[-15,15],dual22p5=[-22.5,22.5]),
        query=dict(x=[-.29,.29],z=[.9,2.5],HEAD=[-.2,.42],BODY=[.42,.9]),
        coverage=stats,optical_axes=optics,elapsed_seconds=time.monotonic()-tick,started_unix=started,completed_unix=time.time(),
        checks=['Identical 3840 config and HEAD/BODY keys to sealed +/-15 coverage','Finite [0,1] coverage values','Input and causal geometry hashes unchanged'],
        provenance=dict(input_sha256=source_hash,geometry_script_sha256=geometry_hash,script_sha256=sha(__file__),payload_sha256=sha(payload)),
        limits=['No new +/-22.5 photon render, M3 logits, threshold calibration or alarm performance. Excluded from main verdict.',
            'Causal 0.6s translation-tangent estimate, declared world gravity, same zero-baseline installation and config-specific estimated-pose errors.',
            'Optical yaw/elevation are relative to estimated travel direction. This is simulated public geometry, not measured hardware.',
            'Fractions pool serial frames for description, not independent statistical trials. Each mode has1280 configs,13 frames.'])
    save(OUT/'geometry_splay22p5.json',result)
    print('COMPLETE',round(result['elapsed_seconds'],2),'seconds',flush=True)
    for label in stats:
        print(label,'mode2',json.dumps(stats[label]['mode2']),flush=True)

if __name__=='__main__':run()
