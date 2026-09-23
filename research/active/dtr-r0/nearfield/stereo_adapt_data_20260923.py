"""Fresh grouped Development source specification, not training or evaluation.

One unchanged MZ101 renderer capture produces 72 layouts x 4 poses. Native
visible depth is retained without a 4m training-label truncation. Scene actor
geometry remains mixed evaluator metadata, never a public training feature.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
ART=REPO/'artifacts.local'
BASE=ART/'evidence/ba-stereo-adapt-20260923'
CAPTURE=ART/'evidence/ba-stereo-adapt-20260923-capture-v1'
SOURCE=Path(__file__).with_name('mz101_stereo_capture.py')
HELPER=Path(__file__).with_name('ue_capture_readiness.py')
RIG=dict(width=640,height=360,hfov_deg=70.,baseline_m=.1,tof_hfov_deg=45.)
FAMILIES=('thin_near','small_head','ordinary_positive','near_outside','far_background','empty')
SEEDS={'train':20260923041,'eval':20260923079}
POSES=(0.,.075,.15,.225)
WIDTHS={'BODY':.56,'HEAD':.36}
BANDS={'BODY':(-1.05,-.3),'HEAD':(-.3,.15)}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def obj(name,center,size,seed,grid):
    return dict(name=name,kind='cube',center_m=[round(v,9) for v in center],
        size_m=[round(v,9) for v in size],texture_seed=seed,texture_grid=grid)


def specification():
    frames=[];groups=[]
    for split,number in [('train',8),('eval',4)]:
        rng=random.Random(SEEDS[split])
        for family in FAMILIES:
            for index in range(number):
                layout=f'adapt_{split}_{family}_{index:02d}'
                appearance='textured' if index%2==0 else 'flat'
                camera_x=rng.uniform(-.12,.12);camera_y=rng.uniform(-.10,.10)
                front=rng.uniform(1.10,2.82);texture_seed=rng.randrange(10_000_000,99_999_999)
                grid=[rng.choice((5,7,9,11)),rng.choice((5,7,9,11))]
                objects=[]
                if family=='thin_near':
                    thickness=rng.uniform(.027,.062);depth=rng.uniform(.036,.095)
                    lateral=rng.choice((-1,1))*rng.uniform(.035,.125)
                    objects=[obj('thin_support',[camera_x+front+depth/2,camera_y+lateral,1.26],
                                 [depth,thickness,rng.uniform(1.30,1.47)],texture_seed,grid)]
                elif family=='small_head':
                    depth=rng.uniform(.055,.155);width=rng.uniform(.065,.18);height=rng.uniform(.055,.125)
                    objects=[obj('small_suspended',[camera_x+front+depth/2,camera_y+rng.uniform(-.075,.075),rng.uniform(1.60,1.74)],
                                 [depth,width,height],texture_seed,grid)]
                elif family=='ordinary_positive':
                    depth=rng.uniform(.12,.38);width=rng.uniform(.36,.78)
                    objects=[obj('broad_obstacle',[camera_x+front+depth/2,camera_y+rng.uniform(-.10,.10),1.13],
                                 [depth,width,rng.uniform(.52,.87)],texture_seed,grid)]
                elif family=='near_outside':
                    depth=rng.uniform(.07,.28);width=rng.uniform(.10,.31)
                    lateral=rng.choice((-1,1))*(.28+width/2+rng.uniform(.045,.13))
                    objects=[obj('near_lateral',[camera_x+front+depth/2,camera_y+lateral,1.06],
                                 [depth,width,rng.uniform(.36,.58)],texture_seed,grid)]
                elif family=='far_background':
                    far=rng.uniform(5.2,7.7);depth=rng.uniform(.08,.24)
                    objects=[obj('far_panel',[camera_x+far+depth/2,camera_y+rng.uniform(-.22,.22),1.64],
                                 [depth,rng.uniform(.6,1.8),rng.uniform(.4,1.25)],texture_seed,grid)]
                else:
                    # Geometry outside the widest 1m query avoids an identical
                    # empty-layout duplicate while leaving the central corridor empty.
                    lateral=rng.choice((-1,1))*rng.uniform(.95,1.45);depth=rng.uniform(.10,.28)
                    objects=[obj('remote_side_panel',[camera_x+front+depth/2,camera_y+lateral,1.30],
                                 [depth,rng.uniform(.12,.32),rng.uniform(.6,1.1)],texture_seed,grid)]
                ids=[]
                for pose,offset in enumerate(POSES):
                    id=f'{layout}_{pose:02d}';ids.append(id)
                    frames.append(dict(id=id,episode=layout,layout_id=layout,split=split,family=family,
                        appearance=appearance,time_s=pose*.25,
                        camera=dict(x=round(camera_x+offset,9),y=round(camera_y,9),z=1.7,yaw=0.,pitch=0.,roll=0.),
                        body_origin_m=[round(camera_x+offset,9),round(camera_y,9),0.],objects=copy.deepcopy(objects)))
                groups.append(dict(layout_id=layout,split=split,family=family,appearance=appearance,frame_ids=ids,
                    intended_union_positive=family in FAMILIES[:3]))
    return dict(schema='stereo-adapt-grouped-development-v1',rig=RIG,split_seeds=SEEDS,
        frames=frames,layout_groups=groups,
        background=dict(name='background',center_m=[11.4,0.,1.65],size_m=[.12,18.,9.],texture_grid=[23,17],texture_seed=23092391),
        floor=dict(center_m=[4.,0.,-.05],size_m=[24.,20.,.1]),
        sampling='POSED_4_HZ_FOUR_VIEWS_PER_LAYOUT_NOT_INDEPENDENT_FRAMES',
        split_scope='EXPLORE_DEVELOPMENT_GEOMETRY_LAYOUT_DISJOINT_SAME_RENDERER_NOT_PROTECTED_FINAL',
        label_scope='NATIVE_VISIBLE_AXIAL_DEPTH_UNTRUNCATED_NO_SCENE_GEOMETRY_TRAINING_INPUT',
        appearances='One fixed appearance per geometry layout; no counterpart split leakage',
        pose_scope='FIXED_ZERO_CAMERA_ROTATION_CONTROLLED_FORWARD_VIEWS_NOT_HEAD_MOTION_GENERALIZATION')


def signature(frame):
    camera=frame['camera']
    # Exclude names/texture seeds/absolute translations so geometry duplicates cannot hide.
    return tuple(sorted((tuple(obj['size_m']),tuple(round(obj['center_m'][j]-camera[k],8)
        for j,k in enumerate(('x','y','z')))) for obj in frame['objects']))


def positive(frame):
    camera=frame['camera'];answer=False
    for item in frame['objects']:
        center=item['center_m'];half=[x/2 for x in item['size_m']]
        x=center[0]-camera['x'];y=center[1]-camera['y'];z=center[2]-camera['z']
        for part,(lo,hi) in BANDS.items():
            answer |= (x+half[0]>=.5 and x-half[0]<=3 and y+half[1]>=-WIDTHS[part]/2 and
                y-half[1]<=WIDTHS[part]/2 and z+half[2]>=lo and z-half[2]<=hi)
    return bool(answer)


def validate(spec):
    assert spec==specification()
    frames=spec['frames'];groups=spec['layout_groups'];assert len(frames)==288 and len(groups)==72
    assert len({f['id'] for f in frames})==288
    by_id={f['id']:f for f in frames};signatures={};counts=Counter();balance=Counter()
    historical=set()
    for stem in ('mz101-stereo-tof-spatial-20260912','mz102-stereo-support-20260912'):
        old=read(ART/'work'/stem/'capture-v1/spec.json')
        historical.update(signature(f) for f in old['frames'])
    for group in groups:
        rows=[by_id[id] for id in group['frame_ids']];assert len(rows)==4
        assert len({r['split'] for r in rows})==1
        assert all(r['objects']==rows[0]['objects'] and r['appearance']==rows[0]['appearance'] for r in rows)
        for row in rows:
            assert positive(row)==group['intended_union_positive'],row['id']
            s=signature(row);assert s not in historical,'Exact historical geometry reused'
            assert s not in signatures,'Duplicate relative geometry across fresh frames/layouts'
            signatures[s]=row['split'];counts[row['split']]+=1;balance[(row['split'],positive(row))]+=1
    assert dict(counts)=={'train':192,'eval':96}
    assert balance==Counter({('train',True):96,('train',False):96,('eval',True):48,('eval',False):48})
    return dict(status='PASS',frames=288,layouts=72,frames_per_layout=4,
        split_frames=dict(counts),split_layouts={'train':48,'eval':24},
        intended_union_positive_frames={'train':96,'eval':48},intended_union_negative_frames={'train':96,'eval':48},
        relative_geometry_duplicates=0,historical_exact_geometry_duplicates=0,
        source_family_counts={split:{family:sum(f['split']==split and f['family']==family for f in frames) for family in FAMILIES} for split in SEEDS},
        caveat='Declared analytic scene balance; native visible valid labels and actual bounds must be checked after capture, no output-based exclusion')


def prepare():
    spec=specification();check=validate(spec)
    write(BASE/'plan/spec.json',spec);write(BASE/'plan/source-check.json',check)
    write(BASE/'plan/split-manifest.json',dict(schema='stereo-adapt-layout-split-v1',groups=spec['layout_groups'],
        train_ids=[r['id'] for r in spec['frames'] if r['split']=='train'],eval_ids=[r['id'] for r in spec['frames'] if r['split']=='eval']))
    write(BASE/'plan/capture-source-seal.json',dict(spec_sha256=sha(BASE/'plan/spec.json'),
        generator_sha256=sha(__file__),capture_source_sha256=sha(SOURCE),helper_sha256=sha(HELPER)))
    run=dict(schema='blindassist-asset-run-v1',id='stereo-adapt-20260923-capture-v1',route='ue-stereo-adapt',
        question='Materialize fresh fixed layout-disjoint stereo adaptation Development inputs',
        evaluator='research/active/dtr-r0/nearfield/STEREO_ADAPT_PROTOCOL_20260923.md',
        evidence_boundary='Fresh same-renderer controlled Development; train/eval geometry-layout disjoint, no protected final',
        reuse=dict(mode='diagnostic',query='unchanged stereo capture renderer new grouped adaptation layouts'),
        inputs=[dict(alias='plan',path=str(BASE/'plan'),role='configuration',purpose='frozen source geometry and group split'),
            dict(alias='project',path=str(ART/'unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject'),role='configuration',purpose='existing UE project runtime')],
        outputs=[dict(alias='result',path=str(BASE/'capture-result-v1.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,str(Path(__file__)),'capture','--result','{{output:result}}'])
    write(BASE/'capture-run-spec.json',run)
    print(json.dumps(check,indent=2))


def capture(result):
    journal=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL']);assert journal['state']=='running'
    seal=read(BASE/'plan/capture-source-seal.json')
    assert sha(__file__)==seal['generator_sha256'] and sha(SOURCE)==seal['capture_source_sha256']
    assert sha(HELPER)==seal['helper_sha256'] and sha(BASE/'plan/spec.json')==seal['spec_sha256']
    assert not CAPTURE.exists(),'Existing capture must remain intact'
    log=BASE/'capture-subprocess-v1.log'
    with log.open('x',encoding='utf-8') as f:
        outcome=subprocess.run([sys.executable,str(SOURCE),'--spec',str(BASE/'plan/spec.json'),'--output',str(CAPTURE)],
            cwd=REPO,stdout=f,stderr=subprocess.STDOUT,text=True)
    if outcome.returncode:
        write(result,dict(status='FAILED',returncode=outcome.returncode,log=str(log),capture=str(CAPTURE)))
        raise RuntimeError('Capture failed; retained log '+str(log))
    receipt=read(CAPTURE/'receipt.json')
    assert receipt['status']=='PASS' and receipt['frames']==288
    assert receipt['source_sha256']==seal['capture_source_sha256'] and receipt['spec_sha256']==seal['spec_sha256']
    for filename,h in receipt['hashes'].items():assert sha(CAPTURE/filename)==h
    write(result,dict(status='PASS',frames=288,capture=str(CAPTURE),receipt_sha256=sha(CAPTURE/'receipt.json'),
        manifest_sha256=sha(CAPTURE/'manifest.json'),source_check=read(BASE/'plan/source-check.json'),
        public_schema='left/right RGB; 64 radial ToF range+valid; calibration',
        training_label='native visible axial depth retained without4m truncation; evaluator geometry not public input',
        log=str(log)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','capture'));p.add_argument('--result',type=Path)
    a=p.parse_args();prepare() if a.command=='prepare' else capture(a.result)
