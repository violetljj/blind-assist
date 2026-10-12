"""Independent CPU-only contract checks for extrinsic augmentation B.

No candidate score, evaluator cohort or GPU is opened by the preflight stage.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-extrinsic-aug-20261006'
sys.pycache_prefix=str(OUT/'pycache')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(name,value):
    folder=OUT/'checks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def rotation(angle,axis='y'):
    a=np.deg2rad(angle);c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]]) if axis=='y' else np.array([[1,0,0],[0,c,-s],[0,s,c]])


def pose(yaw,pitch,position):
    p=np.eye(4);p[:3,:3]=rotation(yaw)@rotation(pitch,'x');p[:3,3]=position;return p


def extrinsic(angle):
    p=np.eye(4);p[:3,:3]=rotation(10,'x')@rotation(angle)@rotation(-10,'x');return p


def coordinate_preflight():
    rng=np.random.default_rng(2026100608);F=np.diag([-1.,1,1,1]);tested=0;maxerr=0.
    for headyaw in (-20.,0.,15.,20.):
        q=pose(headyaw,-10,[0,0,0]);noisy=np.stack([pose(float(rng.uniform(-35,35)),float(rng.uniform(-15,-5)),rng.normal(size=3)) for _ in range(16)])
        for angle in (-30.,-15.,0.,15.,30.):
            ex=extrinsic(angle);physical=q@ex
            np.testing.assert_allclose(F@physical@F,(F@q@F)@extrinsic(-angle),atol=1e-14,rtol=0)
            assert np.linalg.det((F@physical@F)[:3,:3])>1.-1e-14
            nb=noisy@ex;qb=q@ex
            for f in (3,8,15):
                past=np.arange(max(0,f-7),f+1)
                actual=qb@np.linalg.inv(nb[f])@nb[past]
                expected=q@np.linalg.inv(noisy[f])@noisy[past]@ex
                err=float(abs(actual-expected).max());maxerr=max(maxerr,err);assert err<1e-13
                # Joint world/local x reflection yields the identical global query.
                mirrored=(F@qb@F)@np.linalg.inv(F@nb[f]@F)@(F@nb[past]@F)
                np.testing.assert_allclose(mirrored,F@actual@F,atol=1e-13,rtol=0)
                tested+=1
    # Reflection swaps box endpoints without changing symmetric travel corridor membership.
    for _ in range(64):
        lo=rng.uniform([-1,-.4,.3],[1,.8,2.4]);hi=lo+rng.uniform(.01,.7,3)
        ml=lo.copy();mh=hi.copy();ml[0],mh[0]=-hi[0],-lo[0]
        assert np.all(mh>ml)
        assert (min(hi[0],.30)>max(lo[0],-.30))==(min(mh[0],.30)>max(ml[0],-.30))
    return dict(status='PASS',matrix_cases=tested,mirrored_box_cases=64,relative_transform_max_error=maxerr,
                contract='q_b=q_center@E, N_b=N_center@E all history, labels fixed global travel; F P F proper rotation',
                old_V=dict(rows=39936,natural_rows=27456,displacement_rows=12480,epochs=8,batch=64,lr=.0003,wd=.0001,scheduler='CosineAnnealingLR8',initialization='matching M3 seed',labels='actual half-width.30 all-object perframe contact positive / pass0-10 mask / clear negative',far_weight=2,domain_weight='each masked domain N/2'),
                old_recipe_sha256={name:sha(Path(__file__).with_name(name)) for name in ('cnh_temporal_readout_train.py','cnh_temporal_readout_data.py','cnh_temporal_readout_model.py','cnh_cvr_pilot.py','cnh_m3_pose_ensemble.py')},
                outcome_access=False,GPU_operations=False)


def reservation_audit():
    # Numeric reservations and exact unit filenames only; no score payloads.
    wanted=set(range(100000,100096))|{u for u in range(240000,240720) if u%3 in (0,1)}|set(range(98000,98048))|set(range(99000,99096))
    hits=[];roots=0;plans=0;files=0
    for folder in OUT.parent.glob('cnh*'):
        if folder==OUT or not folder.is_dir():continue
        roots+=1
        for name in ('PLAN.json','plan.json','request.json'):
            p=folder/name
            if p.exists():
                plans+=1;content=p.read_text(encoding='utf8')
                found={int(n) for n in re.findall(r'(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_.])',content)}
                found|={int(n) for n in re.findall(r'unit(\d+)\.(?:npz|npy|json)',content)}
                if wanted&found:hits.append(dict(path=str(p),units=sorted(wanted&found),kind='reservation'))
        # Existing filename inventory avoids recursively opening any data bytes.
        for p in folder.rglob('unit*.*'):
            m=re.fullmatch(r'unit(\d+)',p.stem)
            if m:
                files+=1
                if int(m[1]) in wanted:hits.append(dict(path=str(p),units=[int(m[1])],kind='exact unit filename'))
    assert not hits,hits
    return dict(status='PASS',wanted_units=len(wanted),roots=roots,plans=plans,unit_filenames=files,hits=hits,
                reserved_domains=dict(natural_train=[100000,100095],displacement_train=[240000,240719],calibration=[98000,98047],evaluation=[99000,99095]),payload_access=False)


def plan_check():
    from collections import Counter
    p=read(OUT/'PLAN.json');t=p['training'];c=p['calibration'];e=p['evaluation']
    assert p['deadline_unix']-p['start_unix']==4*3600
    assert t['expected_rows']==96*22*13+480*2*13==39936
    assert (t['epochs'],t['batch_size'],t['learning_rate'],t['weight_decay'],t['seeds'])==(8,64,.0003,.0001,[0,1,2])
    assert t['dtype']=='float32' and not t['tf32'] and t['zero_yaw_fraction']==.25
    assert e['no_access_before_pilot_pass'] and c['pilot_gate_path']=='evaluator/pilot_gate.json'
    cells={}
    for name,bounds in [('natural_train',t['natural_units']),('calibration',c['units']),('evaluation',e['units'])]:
        units=range(bounds[0],bounds[1]+1);counts=Counter((u%3,(u//3)%2) for u in units)
        assert all(counts[(mode,0)]==counts[(mode,1)] for mode in range(3)),counts
        cells[name]={f'mode{m}_mirror{r}':n for (m,r),n in sorted(counts.items())}
    assert cells['evaluation']['mode2_mirror0']==cells['evaluation']['mode2_mirror1']==16
    old=ROOT/'artifacts.local/work/cnh-margin-labels-20261002/models/M3'
    models={str(old/f'model_seed{s}.pt'):sha(old/f'model_seed{s}.pt') for s in range(5)}
    return dict(status='PASS',plan_sha256=sha(OUT/'PLAN.json'),balanced_unit_modes=cells,
                original_M3_model_sha256=models,augmentation_is_old_V_recipe_not_original_M3_training_recipe=True,
                pilot_bias='Threshold and seed0 pilot gate use the same calibration set; optimistic exploratory screen',
                evaluation_payload_access=False,GPU_operations=False)


def architecture_check():
    import torch
    import cnh_cvr_pilot as CP
    import cnh_temporal_readout_model as M
    torch.set_num_threads(2)
    old=ROOT/'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed0.pt'
    stored=torch.load(old,map_location='cpu',weights_only=True);network=CP.CVR()
    network.load_state_dict(stored,strict=True)
    assert sum(p.numel() for p in network.parameters())==46097
    assert M.CVR is CP.CVR
    shape={k:list(v.shape) for k,v in network.state_dict().items()}
    source=Path(__file__).with_name('cnh_extrinsic_aug_train.py');tree=ast.parse(source.read_text(encoding='utf8'))
    text=source.read_text(encoding='utf8')
    assert 'V.network(\'V\', seed, device)' in text and 'M.prepare_voxels(batch[\'voxels\'])' in text
    assert 'CosineAnnealingLR(opt, EPOCHS)' in text and 'final_epoch_selection=True' in text
    assert all(s in text for s in ('EPOCHS, BATCH, LR, WD = 8, 64, 3e-4, 1e-4','teacher=False','precision=\'float32; fp16 raw voxels; TF32 disabled\''))
    del network,stored
    return dict(status='PASS',parameters=46097,model_shapes=shape,training_source_sha256=sha(source),original_initializer_sha256=sha(old),new_predictor_accepts_voxels_only=True,GPU_operations=False)


def data_check():
    from collections import Counter
    import cnh_extrinsic_aug_data as D
    p=D.plan();F=np.diag([-1.,1,1,1]);counts=Counter();seq=0;zero=0
    for domain,units,nconfig in [('natural',p['natural_train_units'],22),('displacement',p['train_units'],1)]:
        for index,u in enumerate(units):
            mirror=(u//3)%2 if domain=='natural' else (index//8)%2
            for c in range(nconfig):
                slot=index*22+c if domain=='natural' else 96*22+index
                angle=D.angle_for(u,c,0,slot,p)
                expected=0. if slot%4==0 else float(np.random.default_rng([2026100608,u,c,0]).uniform(-30,30))
                assert angle==expected and -30<=angle<=30
                counts[(domain,u%3,mirror,'zero' if angle==0 else 'nonzero')]+=1
                seq+=1;zero+=angle==0
    assert seq==2592 and zero==648
    # Actual source passes replica0 once then repeats the same physical angle for K.
    tree=ast.parse(Path(D.__file__).read_text(encoding='utf8'));source=Path(D.__file__).read_text(encoding='utf8')
    assert 'angle_for(unit, config, 0, sequence_index, p)' in source and 'aa = [angle]*repeats' in source
    assert 'paired_project = N.Predictor.paired_project' in source
    source_signatures={name:sha(Path(D.__file__).with_name(name)) for name in ('cnh_displacement_ceiling_render.py','cnh_temporal_readout_data.py','cnh_dual_sensor_envelope_natural.py','cnh_cvr_v2_materialize.py')}
    matrices=0;labels=0;maxerr=0.
    for u in (100000,100001,100002):
        scenes=D.natural_train_scenes(u)
        for c in (0,10,21):
            original=D.base_metadata(u,c,0,False,p);mirrored=D.base_metadata(u,c,0,True,p)
            for a,b in zip(original,mirrored):np.testing.assert_allclose(b,F@a@F,atol=1e-13,rtol=0)
            assert all(np.allclose(np.linalg.det(a[:,:3,:3]),1.,atol=1e-12) for a in mirrored)
            b=scenes[c]['boxes'];mb=D.mirror_boxes(b,True)
            for x,y in zip(b,mb):
                assert y['lo'][0]==-x['hi'][0] and y['hi'][0]==-x['lo'][0]
                assert y['lo'][1:]==x['lo'][1:] and y['hi'][1:]==x['hi'][1:]
            cats=D.categories(b,original[1]);mcats=D.categories(mb,mirrored[1]);np.testing.assert_array_equal(cats,mcats);labels+=cats.size
            for angle in (-30,0,30):
                ex=extrinsic(angle);noisy=mirrored[2]@ex;query=mirrored[3]@ex
                for f in (3,8,15):
                    h=np.arange(max(0,f-7),f+1);actual=query[f]@np.linalg.inv(noisy[f])@noisy[h]
                    expected=mirrored[3][f]@np.linalg.inv(mirrored[2][f])@mirrored[2][h]@ex
                    err=float(abs(actual-expected).max());maxerr=max(maxerr,err);assert err<1e-12;matrices+=1
    return dict(status='PASS',installation_sequences=seq,exact_zero_angles=zero,zero_fraction=zero/seq,
                angle_counts={f'{d}|mode{m}|mirror{r}|{a}':n for (d,m,r,a),n in sorted(counts.items())},
                shared_photon_replica_angle=True,actual_metadata_matrices=matrices,mirror_category_queries=labels,
                relative_matrix_max_error=maxerr,data_sha256=sha(D.__file__),admitted_dependency_sha256=source_signatures,
                known_design_consequence='Global displacement sequence mod4 zeros select mode0 only: mode0 zero50%, mode1 zero0%; complies with frozen overall25%, descriptive confounding retained',
                observation_boundary='Renderer uses boxes for simulated photons; projector receives z/noisy/public query only; true travel and boxes used for supervision and fixed-schedule assertion only',GPU_operations=False)


def evaluator_engineering_check():
    import cnh_extrinsic_aug_evaluate as E
    rng=np.random.default_rng(2026100611);cases=0
    for physical in (1,3,11):
        for _ in range(24):
            clear=rng.random(physical*2)>.2;baseline=np.full((physical*2,13),-2.);new=np.full_like(baseline,-4.)
            for j in range(physical):
                if rng.random()<.5:baseline[2*j+rng.integers(0,2),rng.integers(0,13)]=2.
                new[2*j+rng.integers(0,2),rng.integers(0,13)]=rng.integers(-3,4)
            fit=E.cost_threshold(new,baseline,clear);jc=clear.reshape(physical,2).all(1)
            if not jc.any():assert fit['status']=='NOT_EVALUABLE';continue
            budget=sum(bool(jc[i] and (baseline[2*i:2*i+2]>=.8557642486787612).any()) for i in range(physical))
            maxima=[float(new[2*i:2*i+2].max()) for i in range(physical) if jc[i]]
            attainable=[float(np.nextafter(x,np.inf)) for x in set(maxima)]+[float(np.nextafter(new.min(),-np.inf))]
            allowed=[a for a in attainable if sum(x>=a for x in maxima)<=budget]
            expected=min(allowed)
            assert fit['threshold']==expected and fit['original_single_stops']==budget
            assert fit['calibration_physical_clear_n']==int(jc.sum())
            assert fit['new_dual_stops']==sum(x>=expected for x in maxima)
            cases+=1
    # Both queries clear is essential. A contact in the other height never enters primary cost.
    clear=np.array([True,False,True,True]);baseline=np.zeros((4,13));baseline[0,0]=2
    new=np.zeros((4,13));new[0,0]=4;fit=E.cost_threshold(new,baseline,clear)
    assert fit['calibration_physical_clear_n']==1 and fit['original_single_stops']==fit['new_dual_stops']==0
    return dict(status='PASS',independent_whole_tie_threshold_cases=cases,physical_pair_contact_exclusion=True,
                minimum_finite_attainable_cutoff=True,evaluator_sha256=sha(E.__file__),scientific_scores_opened=False,GPU_operations=False)


def source_audit():
    import cnh_extrinsic_aug_data as D
    import cnh_extrinsic_aug_train as T
    import cnh_extrinsic_aug_evaluate as E
    from collections import Counter
    p=read(OUT/'PLAN.json');sources={}
    for name in ('cnh_extrinsic_aug_data.py','cnh_extrinsic_aug_train.py','cnh_extrinsic_aug_predict.py','cnh_extrinsic_aug_evaluate.py'):
        path=Path(__file__).with_name(name);ast.parse(path.read_text(encoding='utf8'));sources[name]=sha(path)
    final=read(OUT/'DATA_EXECUTION_FINAL.json')
    assert final['script_sha256']==final['snapshot_sha256']==sources['cnh_extrinsic_aug_data.py']
    assert sha(OUT/'source/cnh_extrinsic_aug_data-final.py')==final['script_sha256']
    assert sources['cnh_extrinsic_aug_data.py']==read(OUT/'checks/data.json')['data_sha256']
    original=read(OUT/'checks/coordinates.json')['old_recipe_sha256']
    for name,h in original.items():assert sha(Path(__file__).with_name(name))==h
    for name,h in read(OUT/'checks/plan.json')['original_M3_model_sha256'].items():assert sha(name)==h
    blocked={}
    assert not (OUT/p['calibration']['pilot_gate_path']).exists()
    for name,fn in [('generation',D.require_evaluation),('later_training',lambda:T.contract(OUT,later=True)),('evaluator',lambda:E.require_pilot_pass(OUT))]:
        try:fn()
        except (FileNotFoundError,PermissionError,ValueError,AssertionError):blocked[name]=True
        else:raise AssertionError('Missing pilot gate allowed '+name)
    return dict(status='PASS',final_data_source_amendment_checked=True,new_sources_sha256=sources,
                old_sources_and_five_models_preserved=True,strict_pre_pilot_evaluation_denied=blocked,
                supervision='same .30 global travel labels/categories, contact/clear hard targets only, pass masked, far2, eachdomainN/2',
                deployment_inputs='fp16 3-channel projection from photons + estimated whole trajectory + declared public head command + known extrinsic; fixed query masks; no truth/teacher predictor input',
                replica_install_angle_shared=True,mirror_rotation='F R F properdet+1; boxes+position+noisy+publicquery reflected together',
                data_receipt_interface='rows and samples39936 now provided; trainer verifies39936',
                repaired_interfaces=['Gate paths unified to PLAN calibration.pilot_gate_path; judgment PASS common to consumers','Predictor score phases seed0/ensemble3 now explicit evaluator load paths','Training receipt rows39936 added before render/training source freeze'],
                evaluation_score_access=False,GPU_operations=False)


def training_inputs_check():
    import cnh_extrinsic_aug_data as D
    from cnh_temporal_readout_data import normalized_z
    p=D.plan();folder=OUT/'inputs/train';receipt=read(folder/'receipt.json')
    assert receipt['status']=='COMPLETE' and receipt['rows']==receipt['samples']==39936
    assert receipt['plan_sha256']==sha(OUT/'PLAN.json')
    with np.load(folder/'rows.npz') as z:rows={k:z[k] for k in z.files}
    arrays={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ('labels','mask','weights','voxels')}
    assert arrays['voxels'].dtype==np.float16 and arrays['voxels'].shape==(39936,3,24,17,33)
    for k in ('labels','mask','weights'):assert arrays[k].dtype==np.float32 and arrays[k].shape==(39936,2)
    labels=np.char.startswith(rows['category'],'contact').astype(np.float32)
    mask=(rows['category']!='pass0-10cm').astype(np.float32)
    rawweights=mask*np.where((rows['front']>=1.6)&(rows['front']<2.6),2.,1.)[:,None]
    np.testing.assert_array_equal(labels,arrays['labels']);np.testing.assert_array_equal(mask,arrays['mask'])
    masses={};weight_error=0.;payloads=0;offset=0;metadata_matrices=0
    for domain,numeric,units,configs,repeats in [('natural',0,p['natural_train_units'],22,1),('displacement',1,p['train_units'],1,2)]:
        choose=rows['domain']==numeric;mass=float(rawweights[choose].sum());assert abs(mass-receipt['raw_loss_mass'][str(numeric)])<1e-9
        expected=(rawweights[choose].astype(np.float32)*np.float32(19968./mass))
        np.testing.assert_array_equal(expected,arrays['weights'][choose]);masses[domain]=float(arrays['weights'][choose].sum(dtype=np.float64))
        assert abs(masses[domain]-19968)<.01
        for index,u in enumerate(units):
            path=OUT/'train_units'/domain/f'unit{u}.npz';r=read(path.with_suffix('.json'));truth=read(OUT/'truth/train'/domain/f'unit{u}.json');count=configs*repeats*13
            assert r['samples']==count and r['unit']==truth['unit']==u and r['plan_sha256']==sha(OUT/'PLAN.json')
            block=slice(offset,offset+count);assert np.all(rows['unit'][block]==u)
            np.testing.assert_array_equal(rows['frame'][block],np.tile(np.arange(3,16),configs*repeats))
            expected_mirror=((u//3)%2 if domain=='natural' else (index//8)%2)==1
            assert r['mirror']==truth['mirror']==expected_mirror and np.all(rows['mirror'][block]==expected_mirror)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['labels'],labels[block]);np.testing.assert_array_equal(z['mask'],mask[block])
                np.testing.assert_array_equal(z['weights'],rawweights[block].astype(np.float32))
                assert z['hist'].dtype==np.int32 and z['z1'].dtype==np.float16
                np.testing.assert_array_equal(z['z1'],normalized_z(z['hist'],z['ambient']))
                angles=z['angles'];assert len(angles)==configs*repeats
                for config in range(configs):
                    slot=index*22+config if domain=='natural' else 96*22+index
                    angle=D.angle_for(u,config,0,slot,p)
                    assert all(angles[config*repeats+j]==angle for j in range(repeats))
                    assert np.all(rows['angle_deg'][offset+config*repeats*13:offset+(config+1)*repeats*13]==angle)
                # Every saved nominal public transform is independent of object supervision.
                for sequence in range(configs*repeats):
                    config=sequence//repeats;replica=sequence%repeats
                    _,travel,noisy,query=D.base_metadata(u,config,replica,expected_mirror,p);ex=extrinsic(float(angles[sequence]))
                    np.testing.assert_allclose(z['noisy'][sequence],noisy@ex,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(z['public_query'][sequence],query@ex,atol=1e-12,rtol=0)
                    metadata_matrices+=32
            assert sha(OUT/'truth/train'/domain/f'unit{u}.json')==r['truth_sha256']
            offset+=count;payloads+=1
    assert offset==39936 and payloads==576
    # Full small arrays are independently sealed; GB feature bytes are hashed once by their owner.
    for name in ('labels.npy','mask.npy','weights.npy','rows.npz'):assert sha(folder/name)==receipt['outputs'][name]
    return dict(status='PASS',rows=39936,training_unit_payloads=payloads,supervision_queries=79872,
                saved_public_noisy_matrices=metadata_matrices,domain_masked_weight_mass=masses,
                zero_angle_and_K_replica_contract_checked=True,mirror_schedule_checked=True,
                signed_int32_photons_to_fp16_z_parity=True,
                voxel_header='fp16 [39936,3,24,17,33]; full feature SHA provided by data owner and final parent',
                receipt_sha256=sha(folder/'receipt.json'),evaluation_payload_access=False,GPU_operations=False)


def independent_smooth(raw):
    values=np.asarray(raw,np.float64);result=np.empty_like(values)
    for t in range(13):
        indices=np.arange(max(0,t-4),t+1);weights=np.exp2(indices-indices[0])
        result[...,t,:]=np.sum(values[...,indices,:]*weights[:,None],axis=-2)/float(weights.sum())
    return result


def independent_metrics(scores,threshold,g,mask,baseline,baseline_threshold):
    hit=scores>=threshold;stopped=hit.any(1);first=hit.argmax(1)
    front=g['frame_ranges'][np.arange(len(scores)),first];timely=stopped&(front>=.9)
    bhit=baseline>=baseline_threshold;bstop=bhit.any(1);bf=bhit.argmax(1)
    bt=bstop&(g['frame_ranges'][np.arange(len(scores)),bf]>=.9)
    clear=mask&g['clear_all'];joint=g['clear_all'].reshape(-1,2).all(1)&mask.reshape(-1,2).all(1)
    cell={}
    for label,select,counted in [('query_clear',clear,stopped),('physical_clear',joint,stopped.reshape(-1,2).any(1))]:
        n=int(select.sum());s=int((select&counted).sum());minutes=n*2.6/60
        cell[label]=dict(n=n,stops=s,proxy_minutes=minutes,rate_per_proxy_minute=s/minutes if n else None)
    censored=mask&~g['covered'];cell['censored']=dict(n=int(censored.sum()),stopped=int((censored&stopped).sum()))
    for label,categories in [('shallow0_2',['contact0-2cm']),('mid2_5',['contact2-5cm']),('shallow0_5',['contact0-2cm','contact2-5cm']),('deep',['contact>5cm'])]:
        select=mask&g['covered']&np.isin(g['ref_category'],categories);n=int(select.sum());num=int((select&timely).sum())
        rescue=int((select&timely&~bt).sum());loss=int((select&~timely&bt).sum())
        cell[label]=dict(n=n,timely=num,rate=num/n if n else None,stopped=int((select&stopped).sum()),rescue=rescue,loss=loss,net=rescue-loss,baseline_timely=int((select&bt).sum()))
    return cell


def compare_nested(actual,expected):
    for key,value in expected.items():
        if isinstance(value,dict):compare_nested(actual[key],value)
        elif isinstance(value,float):assert abs(actual[key]-value)<1e-11,(key,actual[key],value)
        else:assert actual[key]==value,(key,actual[key],value)


def pilot_check():
    selection=read(OUT/'evaluator/seed0_calibration.json');gate=read(OUT/'evaluator/pilot_gate.json');units=list(range(98000,98048))
    assert selection['seeds']==gate['seeds']==[0] and selection['phase']=='calibration/seed0'
    with np.load(OUT/'geometry/calibration.npz') as z:g={k:z[k] for k in z.files}
    reference=[];augmented=[];payloads=0
    for unit in units:
        path=OUT/'scores/calibration/seed0'/f'unit{unit}.npz';receipt=read(path.with_suffix('.json'))
        assert receipt['output_sha256']==sha(path)
        with np.load(path) as z:
            assert z['unit']==unit and z['seeds'].tolist()==[0]
            assert z['reference'].shape==z['augmented_mean'].shape==(3,40,13,2)
            np.testing.assert_array_equal(z['augmented_mean'],z['augmented_by_seed'][0])
            reference.append(independent_smooth(z['reference']).transpose(1,3,0,2).reshape(80,3,13))
            augmented.append(independent_smooth(z['augmented_mean']).transpose(1,3,0,2).reshape(80,3,13))
        payloads+=1
    np.testing.assert_array_equal(g['unit'],np.repeat(units,80));np.testing.assert_array_equal(g['config'],np.tile(np.repeat(np.arange(40),2),48));np.testing.assert_array_equal(g['query'],np.tile([0,1],1920))
    m3=np.concatenate(reference);new=np.concatenate(augmented)
    arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=new[:,0],DUAL_NEW=new[:,1:].max(1));threshold=selection['threshold']
    joint=g['clear_all'].reshape(-1,2).all(1);baseline_physical=(arms['S_M3']>=.8557642486787612).any(1).reshape(-1,2).any(1)
    budget=int((joint&baseline_physical).sum());maxima=arms['DUAL_NEW'].reshape(-1,2,13).max((1,2))[joint]
    candidates=np.concatenate((np.nextafter(np.unique(maxima),np.inf),[np.nextafter(arms['DUAL_NEW'].min(),-np.inf)]))
    allowed=[float(c) for c in candidates if np.count_nonzero(maxima>=c)<=budget]
    assert threshold==min(allowed) and selection['original_single_stops']==budget
    turn=g['mode']==2;allrows=np.ones(3840,bool)
    for name,scores in arms.items():
        t=threshold if name.endswith('NEW') else .8557642486787612
        expected_all=independent_metrics(scores,t,g,allrows,arms['S_M3'],.8557642486787612)
        expected_turn=independent_metrics(scores,t,g,turn,arms['DUAL_M3'],.8557642486787612)
        compare_nested(selection['cells'][name]['all'],expected_all);compare_nested(selection['cells'][name]['turn'],expected_turn)
    before=selection['cells']['DUAL_M3']['turn'];after=selection['cells']['DUAL_NEW']['turn']
    nclear=before['query_clear']['n'];nc=before['query_clear']['stops'];nshallow=before['shallow0_2']['n']
    evaluable=nclear>0 and nc>0 and nshallow>0;net=after['shallow0_2']['timely']-before['shallow0_2']['timely']
    passed=evaluable and after['query_clear']['stops']<=.7*nc and net>0
    expected='PASS' if passed else 'FAIL' if evaluable else 'NOT_EVALUABLE'
    assert gate['judgment']==expected and gate['threshold']==threshold and gate['turn_shallow0_2_timely_net']==net
    assert gate['calibration_sha256']==sha(OUT/'evaluator/seed0_calibration.json')
    if not passed:
        assert not (OUT/'scores/evaluation').exists() and not (OUT/'features/evaluation').exists()
        assert not (OUT/'models/M3_aug/seed1/model.pt').exists() and not (OUT/'models/M3_aug/seed2/model.pt').exists()
    return dict(status='PASS',scientific_calibration_units=payloads,query_episodes=3840,physical_sequences=1920,
                threshold=threshold,joint_clear_n=int(joint.sum()),baseline_single_physical_stops=budget,
                pilot_judgment=expected,turn_dual_M3_clear=nc,turn_dual_new_clear=after['query_clear']['stops'],turn_shallow_n=nshallow,turn_shallow_net=net,
                original_smoothing_rebuilt=True,all_four_arms_all_and_turn_cells_checked=True,
                gate_sha256=sha(OUT/'evaluator/pilot_gate.json'),evaluation_payload_access=False,GPU_operations=False)


def post_gate_check():
    gate=read(OUT/'evaluator/pilot_gate.json');selection=read(OUT/'evaluator/seed0_calibration.json')
    descriptor=read(OUT/'evaluator/calibration_directions.json')
    assert gate['judgment']=='FAIL' and descriptor['pilot_judgment']=='FAIL'
    assert descriptor['threshold']==selection['threshold'] and descriptor['seeds']==[0]
    assert descriptor['pilot_gate_sha256']==sha(OUT/'evaluator/pilot_gate.json')
    assert descriptor['calibration_sha256']==sha(OUT/'evaluator/seed0_calibration.json')
    with np.load(OUT/'geometry/calibration.npz') as z:g={k:z[k] for k in z.files}
    references=[];augmented=[]
    for unit in range(98000,98048):
        with np.load(OUT/f'scores/calibration/seed0/unit{unit}.npz') as z:
            references.append(independent_smooth(z['reference']).transpose(1,3,0,2).reshape(80,3,13))
            augmented.append(independent_smooth(z['augmented_mean']).transpose(1,3,0,2).reshape(80,3,13))
    m3=np.concatenate(references);new=np.concatenate(augmented)
    arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=new[:,0],DUAL_NEW=new[:,1:].max(1));counts={}
    for direction in ('left','right'):
        mask=(g['mode']==2)&(g['turn']==direction);assert mask.reshape(-1,2).all(1).sum()==320
        counts[direction]={}
        for arm,score in arms.items():
            threshold=selection['threshold'] if arm.endswith('NEW') else .8557642486787612
            expected=independent_metrics(score,threshold,g,mask,arms['S_M3'],.8557642486787612)
            compare_nested(descriptor['cells'][arm]['turn_'+direction],expected)
            other=independent_metrics(score,threshold,g,mask,arms['DUAL_M3'],.8557642486787612)
            compare_nested(descriptor['versus_dual_M3'][arm]['turn_'+direction],other)
            counts[direction][arm]=dict(query_clear=expected['query_clear'],physical_clear=expected['physical_clear'],shallow0_2=expected['shallow0_2'])
    shallow={name:selection['cells'][name]['turn']['shallow0_2'] for name in arms}
    assert all(c['timely']==c['n']==6 and c['net']==0 for c in shallow.values())
    old_models=read(OUT/'checks/plan.json')['original_M3_model_sha256']
    for path,digest in old_models.items():assert sha(path)==digest
    near=Path(__file__).parent;old_sources=read(OUT/'checks/coordinates.json')['old_recipe_sha256']
    for name,digest in old_sources.items():assert sha(near/name)==digest
    absent=['features/evaluation','features/angular','inputs/evaluation','truth/evaluation','scores/evaluation','scores/angular','geometry/evaluation.npz','models/M3_aug/seed1/model.pt','models/M3_aug/seed2/model.pt','evaluator/final_calibration.json']
    for relative in absent:assert not (OUT/relative).exists(),relative
    return dict(status='PASS',calibration_directions=counts,old_five_models_unchanged=True,old_recipe_sources_unchanged=True,
                turn_shallow_timely_all_four_arms='6/6; rescue/loss/net all zero',strict_improvement_reachable=False,
                interpretation='Frozen pilot FAIL ends this run; old dual timely6/6 is the exact ceiling, so lack of strict increase is not efficacy negative evidence.',
                absent_evaluation_or_extra_seed_paths=absent,descriptor_sha256=sha(OUT/'evaluator/calibration_directions.json'),
                evaluation_payload_access=False,GPU_operations=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['coordinates','ids','plan','architecture','data','threshold','source','train-inputs','pilot','post-gate']);args=p.parse_args();tick=time.monotonic()
    value={'coordinates':coordinate_preflight,'ids':reservation_audit,'plan':plan_check,'architecture':architecture_check,'data':data_check,'threshold':evaluator_engineering_check,'source':source_audit,'train-inputs':training_inputs_check,'pilot':pilot_check,'post-gate':post_gate_check}[args.stage]();value['seconds']=time.monotonic()-tick
    value['checker_sha256']=sha(__file__);ast.parse(Path(__file__).read_text(encoding='utf8'));record(args.stage+'.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
