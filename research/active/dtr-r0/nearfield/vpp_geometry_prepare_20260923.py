"""Frozen public RGB/ToF hardlink subset and scoped VPP execution governance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[4]
ART=REPO/'artifacts.local'
ROOT=ART/'evidence/ba-vpp-geometry-20260923'
PREPARED=ART/'evidence/ba-vpp-geometry-20260923-prepared'
OLD_RGB=ART/'evidence/ba-foundation-geometry-20260923-rgb'
WORK=ART/'work/vpp-geometry-20260923'
PROTOCOL='research/active/dtr-r0/nearfield/VPP_GEOMETRY_PROTOCOL_20260923.md'
CAPTURES={'mz101':ART/'work/mz101-stereo-tof-spatial-20260912/capture-v1',
          'mz102':ART/'work/mz102-stereo-support-20260912/capture-v1'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def materialize():
    journal=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])
    assert journal['state']=='running' and journal['reuse_preflight']['status']=='PASS'
    assert not PREPARED.exists() or not any(PREPARED.iterdir()), 'Preserve prior attempt'
    assert PREPARED.resolve().is_relative_to(ART.resolve())
    plan=read(ROOT/'plan/materialization-seal.json')
    for path,digest in plan['inputs'].items():assert sha(Path(path))==digest,path
    assert sha(__file__)==plan['producer_sha256']
    metadata=read(OLD_RGB/'materialization.json')
    assert metadata['status']=='PASS' and metadata['frames']==576 and not metadata['native_or_tof_copied']
    assert sha(OLD_RGB/'rgb-inputs.json')==metadata['input_manifest_sha256']
    rgb_source=read(OLD_RGB/'rgb-inputs.json')
    assert rgb_source['authority']=='RGB_AND_CALIBRATION_ONLY'
    rig=rgb_source['rig'];assert rig==dict(width=640,height=360,hfov_deg=70.,baseline_m=.10,tof_hfov_deg=45.)
    assert len(rgb_source['frames'])==576
    captures={panel:read(path/'receipt.json') for panel,path in CAPTURES.items()}
    for panel,receipt in captures.items():
        assert receipt['status']=='PASS'
        assert sha(CAPTURES[panel]/'mz101_stereo_capture.py')==receipt['source_sha256']
        assert sha(CAPTURES[panel]/'spec.json')==receipt['spec_sha256']
    rgbs=[];tofs=[];hashes={};seen=set()
    for row in rgb_source['frames']:
        assert set(row)=={'panel','id','left','right','left_sha256','right_sha256'}
        panel,id=row['panel'],row['id'];assert panel in CAPTURES
        assert id and not any(x in id for x in ('/','\\','..'))
        assert (panel,id) not in seen;seen.add((panel,id))
        r=dict(panel=panel,id=id);t=dict(panel=panel,id=id)
        for name,suffix in [('left','left.png'),('right','right.png'),('ranges','tof-range.npy'),('valid','tof-valid.npy')]:
            is_rgb=name in ('left','right')
            original=Path(row[name]) if is_rgb else CAPTURES[panel]/'frame'/id/suffix
            expected=captures[panel]['hashes'][f'frame/{id}/{suffix}']
            assert sha(original)==expected
            if is_rgb:assert expected==row[name+'_sha256']
            destination=PREPARED/('rgb' if is_rgb else 'tof')/panel/id/suffix
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.link(original,destination)
            assert sha(destination)==expected
            item=r if is_rgb else t
            item[name]=str(destination);item[name+'_sha256']=expected
            hashes[destination.relative_to(PREPARED).as_posix()]=expected
        ranges=np.load(t['ranges'],allow_pickle=False);valid=np.load(t['valid'],allow_pickle=False)
        assert ranges.shape==valid.shape==(64,) and valid.dtype==np.bool_
        assert np.isfinite(ranges[valid]).all() and ((ranges[valid]>=0)&(ranges[valid]<=4.00001)).all()
        assert np.isnan(ranges[~valid]).all()
        rgbs.append(r);tofs.append(t)
    write(PREPARED/'rgb-inputs.json',dict(rig=rig,frames=rgbs,authority='RGB_AND_CALIBRATION_ONLY'))
    write(PREPARED/'tof-inputs.json',dict(rig=rig,frames=tofs,authority='TOF_AND_CALIBRATION_ONLY'))
    for filename in ('rgb-inputs.json','tof-inputs.json'):hashes[filename]=sha(PREPARED/filename)
    write(PREPARED/'receipt.json',dict(status='PASS',frames=576,rgb_images=1152,tof_arrays=1152,
        authority='PUBLIC_RGB_TOF_CALIBRATION_ONLY_HARDLINK_SUBSET',native_or_pose_or_labels_copied=False,
        source_receipts={panel:sha(path/'receipt.json') for panel,path in CAPTURES.items()},
        source_rgb_manifest_sha256=sha(OLD_RGB/'rgb-inputs.json'),plan_sha256=sha(ROOT/'plan/materialization-seal.json'),
        ray_calibration=dict(origin='EXACT_LEFT_CAMERA_ORIGIN_NO_BASELINE_OFFSET',row_major=True,
            azimuth_deg='-22.5+(column+0.5)*45/8',elevation_deg='22.5-(row+0.5)*45/8',
            vector='normalize([1,tan(azimuth),tan(elevation)]) in forward/right/up camera axes',
            range='radial metres; original native collision rays, not hardware noise',maximum_m=4),
        hashes=hashes,backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',scope='IO_ETL hashes and hardlinks')))
    print('VPP_PUBLIC_SOURCE_PASS',len(rgbs),len(hashes),flush=True)


def inp(alias,path,role):return dict(alias=alias,path=str(path),role=role,purpose='matched-original-FoundationStereo-VPP-public-ToF')


def save_spec(stage,command,inputs,outputs):
    write(ROOT/(stage+'-run-spec.json'),dict(schema='blindassist-asset-run-v1',id='vpp-geometry-20260923-'+stage,
        route='ue-vpp-geometry',question='Does public 64-ray ToF VPP improve matched historical original FoundationStereo geometry?',
        evaluator=PROTOCOL,evidence_boundary='Consumed historical rendered Development, no current-A or hardware claim',
        reuse=dict(mode='diagnostic',query='MZ101 MZ102 public64ToF virtual pattern projection stereo geometry'),
        inputs=inputs,outputs=outputs,result_output='result',command=[str(x) for x in command]))


def output(path):return dict(alias='result',path=str(path),role='result',required=True)


def register(path,status):
    sys.path.insert(0,str(REPO/'tools/data'))
    import asset_catalog as cat
    con=cat.open_catalog(ART/cat.DEFAULT_DATABASE_RELATIVE)
    locator=path.relative_to(ART).as_posix()
    old=con.execute('SELECT * FROM assets WHERE locator=?',(locator,)).fetchone();con.close()
    if old is not None:return dict(locator=locator,action='existing',evidence_status=old['evidence_status'])
    result=subprocess.run([sys.executable,str(REPO/'tools/data/asset_catalog.py'),'register',str(path),
        '--kind','research-bundle','--asset-class','research-evidence','--evidence-status',status,
        '--storage-status','active','--owner','dtr-r0-nearfield','--retention-reason','VPP matched public ToF stereo diagnostic',
        '--claim-ceiling','Consumed Development only'],capture_output=True,text=True,check=True)
    return dict(locator=locator,action='registered',output=result.stdout)


def setup():
    assert (REPO/PROTOCOL).exists(), 'Root owns protocol; wait until present'
    WORK.mkdir(parents=True,exist_ok=True)
    sources=[OLD_RGB/'rgb-inputs.json',OLD_RGB/'materialization.json']
    for path in CAPTURES.values():sources.extend([path/'receipt.json',path/'spec.json',path/'mz101_stereo_capture.py'])
    write(ROOT/'plan/materialization-seal.json',dict(producer_sha256=sha(__file__),inputs={str(p):sha(p) for p in sources}))
    contracts=[]
    for locator,status,scopes in [
        ('evidence/ba-vpp-geometry-20260923','source_material',[('plan','configuration')]),
        ('evidence/ba-vpp-geometry-20260923-prepared','development_consumed',[('rgb','observation'),('tof','observation'),('rgb-inputs.json','observation'),('tof-inputs.json','observation'),('receipt.json','configuration')]),
        ('work/vpp-geometry-20260923','source_material',[('vppstereo/vpp_standalone.py','configuration')]),
        ('evidence/ba-vpp-geometry-20260923-frontend-v1','development_consumed',[('.','evaluator')]),
        ('evidence/ba-vpp-geometry-20260923-evaluation-v1','development_consumed',[('.','evaluator')])]:
        contracts.append(dict(locator=locator,source_family='ue-historical-stereo-mz101-mz102',evidence_status=status,
            evidence=PROTOCOL,allowed_inputs=[dict(relative_path=p,role=r) for p,r in scopes]))
    policy=REPO/'data/ue-reuse-policy.json';text=policy.read_text(encoding='utf-8');old=read(policy)
    assert not {c['locator'] for c in contracts}&{c['locator'] for c in old['contracts']}
    marker='"contracts": [';assert text.count(marker)==1
    insertion='\n'+',\n'.join('    '+json.dumps(c,indent=2).replace('\n','\n    ') for c in contracts)+','
    updated=text.replace(marker,marker+insertion,1);json.loads(updated)
    policy.write_text(updated,encoding='utf-8')
    write(ROOT/'own-policy-contracts.json',contracts)
    write(ROOT/'policy-insertion.json',dict(policy_sha256=sha(policy),locators=[c['locator'] for c in contracts]))
    plan=inp('plan',ROOT/'plan','configuration')
    material_inputs=[plan,inp('rgb_manifest',OLD_RGB/'rgb-inputs.json','observation'),
        inp('rgb_observations',OLD_RGB/'observations','observation'),inp('rgb_metadata',OLD_RGB/'materialization.json','configuration')]
    material_inputs.extend(inp('capture_'+panel,path,'evaluator') for panel,path in CAPTURES.items())
    outputs=[dict(alias=alias,path=str(PREPARED/name),role='observation',required=True)
        for alias,name in [('rgb','rgb'),('tof','tof'),('rgb_manifest','rgb-inputs.json'),('tof_manifest','tof-inputs.json')]]
    outputs.append(output(PREPARED/'receipt.json'))
    save_spec('materialize',[sys.executable,Path(__file__),'materialize'],material_inputs,outputs)
    producer_inputs=[plan,inp('rgb',PREPARED/'rgb','observation'),inp('tof',PREPARED/'tof','observation'),
        inp('rgb_manifest',PREPARED/'rgb-inputs.json','observation'),inp('tof_manifest',PREPARED/'tof-inputs.json','observation'),
        inp('public_receipt',PREPARED/'receipt.json','configuration'),inp('vpp_code',WORK/'vppstereo/vpp_standalone.py','configuration'),
        inp('model',ART/'work/foundation-geometry-20260923/weights/23-51-11','configuration'),
        inp('source',ART/'work/mz103-depth-frontend-20260912/FoundationStereo','configuration')]
    for stage,suffix in [('canary','canary-v1'),('full','frontend-v1')]:
        dest=ART/('evidence/ba-vpp-geometry-20260923-'+suffix)
        command=[sys.executable,REPO/'research/active/dtr-r0/nearfield/vpp_geometry_infer_20260923.py',
            '--config',ROOT/'plan/config.json','--inputs',PREPARED/'rgb-inputs.json',
            '--tof',PREPARED/'tof-inputs.json','--output',dest]
        if stage=='canary':command.append('--canary')
        save_spec(stage,command,producer_inputs,[output(dest/'receipt.json')])
    previous=read(ART/'evidence/ba-foundation-geometry-20260923/evaluate-run-spec.json')
    eval_inputs=[plan,inp('candidate',ART/'evidence/ba-vpp-geometry-20260923-frontend-v1','evaluator'),
        inp('rgb_manifest',PREPARED/'rgb-inputs.json','observation'),inp('rgb',PREPARED/'rgb','observation'),
        inp('original_foundation',ART/'evidence/ba-foundation-geometry-20260923-frontend-v1','evaluator'),
        inp('original_evaluation',ART/'evidence/ba-foundation-geometry-20260923-evaluation-v1','evaluator')]
    eval_inputs += [p for p in previous['inputs'] if p['alias'] not in ('plan','candidate','rgb_manifest','rgb')]
    dest=ART/'evidence/ba-vpp-geometry-20260923-evaluation-v1'
    save_spec('evaluate',[sys.executable,REPO/'research/active/dtr-r0/nearfield/vpp_geometry_evaluate_20260923.py',
        '--candidate',ART/'evidence/ba-vpp-geometry-20260923-frontend-v1','--inputs',PREPARED/'rgb-inputs.json',
        '--output',dest],eval_inputs,[output(dest/'summary.json')])
    write(ROOT/'input-registration.json',[register(ROOT,'source_material'),register(WORK,'source_material')])
    print('VPP_GOVERNANCE_PREPARED; producer/evaluator commands are root-finalized drafts',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('setup','materialize','register-prepared'));a=p.parse_args()
    if a.command=='setup':setup()
    elif a.command=='materialize':materialize()
    else:write(ROOT/'prepared-registration.json',register(PREPARED,'development_consumed'))
