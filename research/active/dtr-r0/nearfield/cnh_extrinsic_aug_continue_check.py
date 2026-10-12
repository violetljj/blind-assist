"""Independent CPU checks of the authorized post-pilot continuation."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
PARENT=ROOT/'artifacts.local/work/cnh-extrinsic-aug-20261006'
OUT=PARENT/'continuation-r1'
sys.pycache_prefix=str(OUT/'pycache/review')

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def record(name,value):
    path=OUT/'checks'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def plan_check():
    old=read(PARENT/'PLAN.json');new=read(OUT/'PLAN.json');revision=read(OUT/'PLAN_REVISION.json')
    old_gate=read(PARENT/'evaluator/pilot_gate.json');gate=read(OUT/'evaluator/pilot_gate.json')
    assert sha(PARENT/'PLAN.json')=='86a33853de8d02315fcd16a0a07f0a737480aa41d947457da2012c7bfd35a9ee'
    assert sha(PARENT/'evaluator/pilot_gate.json')=='e22b16e5c86f5f44289ea2601563cee3dba93fe39509b89342613b25acf6ed84'
    assert revision['new_plan_sha256']==sha(OUT/'PLAN.json') and revision['original_pilot_sha256']==sha(PARENT/'evaluator/pilot_gate.json')
    assert old_gate['judgment']=='FAIL' and gate['judgment']=='PASS' and gate['original_judgment']=='FAIL'
    assert revision['old_predicate']=='net>0' and revision['new_predicate']=='net>=0'
    assert gate['turn_query_clear_reduction']>=.3 and gate['turn_shallow0_2_timely_net']>=0
    for key in ('turn_query_clear_reduction','turn_shallow0_2_timely_net','turn_shallow0_2_n','threshold','calibration_sha256','score_inputs_sha256'):
        assert gate[key]==old_gate[key]
    assert gate['revision_sha256']==sha(OUT/'PLAN_REVISION.json') and gate['plan_sha256']==sha(OUT/'PLAN.json')
    assert canonical(old['evaluation'])==canonical(new['evaluation'])
    assert new['deadline_unix']-new['start_unix']==9000 and new['deadline_unix']==1791213092
    assert new['training']==old['training'] and new['descriptive_V']['in_primary_gates'] is False
    for child,parent in [('inputs/train','inputs/train'),('features/calibration','features/calibration'),('models/M3_aug/seed0','models/M3_aug/seed0')]:
        assert (OUT/child).resolve()==(PARENT/parent).resolve()
    for relative in ('geometry/calibration.npz','geometry/calibration.json'):
        assert sha(OUT/relative)==sha(PARENT/relative)
    for name,entry in read(OUT/'training_input_hashes.json').items():
        path=OUT/'inputs/train'/f'{name}.npy';st=path.stat()
        assert st.st_size==entry['size'] and st.st_mtime_ns==entry['mtime_ns']
    preserved=OUT/'source/PARENT_PLAN_143110bf.json'
    assert sha(preserved)==sha(PARENT/'PLAN.json')
    return dict(status='PASS',plan_sha256=sha(OUT/'PLAN.json'),revision_gate_sha256=sha(OUT/'evaluator/pilot_gate.json'),
                old_fail_and_plan_unchanged=True,evaluation_json_exactly_preserved=True,readonly_reuse_targets_verified=True,
                training_inputs_sealed_size_mtime_only=True,attribution='Historical V is a descriptive recipe control, not an isolated yaw ablation.',
                evaluation_payload_access=False,GPU_operations=False)

def v_models_check():
    import torch
    import cnh_cvr_pilot as CP
    torch.set_num_threads(2);models={};receipts={}
    for seed in (0,1,2):
        folder=ROOT/'artifacts.local/work/cnh-temporal-readout-20261004/models/V'/f'seed{seed}'
        path=folder/'model.pt';r=read(folder/'training_receipt.json')
        assert r['status']=='COMPLETE' and r['arm']=='V' and r['seed']==seed
        assert (r['rows'],r['epochs'],r['batch_size'],r['learning_rate'],r['weight_decay'])==(39936,8,64,.0003,.0001)
        assert r['parameters']==46097 and r['initialization']=='M3 matching seed' and r['final_epoch_selection']
        assert r['baseline_sha256']==sha(ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{seed}.pt')
        assert sha(path)==r['model_sha256']
        net=CP.CVR();state=torch.load(path,map_location='cpu',weights_only=True);net.load_state_dict(state,strict=True)
        assert sum(p.numel() for p in net.parameters())==46097 and all(torch.isfinite(v).all() for v in state.values())
        models[str(path)]=sha(path);receipts[str(folder/'training_receipt.json')]=sha(folder/'training_receipt.json')
        del net,state
    return dict(status='PASS',models_sha256=models,receipts_sha256=receipts,same_architecture_parameters=46097,
                same_V_training_recipe=True,matched_training_scenes_or_photons=False,strict_yaw_only_ablation=False,
                evaluation_payload_access=False,GPU_operations=False)

def source_check():
    import inspect
    import cnh_extrinsic_aug_evaluate as original
    near=Path(__file__).parent
    paths=[near/f'cnh_extrinsic_aug_continue_{name}.py' for name in ('train','predict','render','evaluate')]
    sources={str(p):sha(p) for p in paths};texts={p.stem:p.read_text(encoding='utf8') for p in paths}
    for text in texts.values():ast.parse(text)
    predict=texts['cnh_extrinsic_aug_continue_predict']
    assert predict.index("if split != 'calibration':")<predict.index("folder = root/'features'/split")
    for required in ("final.get('status') != 'COMPLETE'", "final.get('seeds') != [0, 1, 2]", "final['plan_sha256']", "final['revision_gate_sha256']", "final['V_calibration_sha256']"):
        assert required in predict
    assert 'torch.stack(batch_outputs[:5]).mean(0)' in predict and 'torch.stack(batch_outputs[5:8]).mean(0)' in predict and 'torch.stack(batch_outputs[8:11]).mean(0)' in predict
    assert predict.count('prepared = M.prepare_voxels(voxels)')==1
    render=texts['cnh_extrinsic_aug_continue_render']
    assert 'D.OUT = OUT' in render and 'E.OUT = OUT' in render and 'A.OUT = OUT' in render
    for required in ("final.get('seeds') == [0, 1, 2]", "final.get('plan_sha256')", "final.get('revision_gate_sha256')", "scores.get('augmented_seeds') == [0, 1, 2]"):
        assert required in render
    evaluator=texts['cnh_extrinsic_aug_continue_evaluate']
    assert "canonical(plan['evaluation'])!=canonical(original['evaluation'])" in evaluator
    assert "E.cost_threshold(aug[:,1:].max(1),m3[:,0],g['clear_all'])" in evaluator
    assert "E.cost_threshold(v[:,1:].max(1),m3[:,0],g['clear_all'])" in evaluator
    assert "out=configure(out);require_final_calibration(out);main=E.evaluate_angular(out,'angular/ensemble3')" in evaluator
    assert 'main=E.finalize(out);result=dict(main)' in evaluator
    assert "original_pilot_judgment='FAIL',revised_pilot_judgment='PASS'" in evaluator
    assert "Positive net means V timely count exceeds augmented at each independently calibrated threshold" in evaluator
    assert "if root.resolve() == T.OUT.resolve() or seed not in (1, 2)" in texts['cnh_extrinsic_aug_continue_train']
    expected=read(PARENT/'evaluator/THRESHOLD_CONTRACT.json')['threshold_function_sha256']
    assert hashlib.sha256(inspect.getsource(original.cost_threshold).encode()).hexdigest()==expected
    for name,digest in read(PARENT/'checks/coordinates.json')['old_recipe_sha256'].items():assert sha(near/name)==digest
    return dict(status='PASS',source_sha256=sources,final_calibration_required_before_evaluation_feature_access=True,
                same_prepared_tensor_and_raw_mean_family_slices=True,unchanged_final_evaluation_rule=True,
                V_independent_physical_cost_threshold_excluded_from_primary_gate=True,
                no_old_output_root_writes=True,evaluation_payload_access=False,GPU_operations=False)

def independent_cost_threshold(new_dual,reference_single,clear):
    joint=np.asarray(clear,bool).reshape(-1,2).all(1)
    budget=int((joint&(reference_single>=.8557642486787612).any(1).reshape(-1,2).any(1)).sum())
    maxima=new_dual.reshape(-1,2,13).max((1,2))[joint]
    choices=np.r_[np.nextafter(np.unique(maxima),np.inf),np.nextafter(new_dual.min(),-np.inf)]
    theta=min(float(c) for c in choices if np.count_nonzero(maxima>=c)<=budget)
    return theta,budget,int(joint.sum()),int((maxima>=theta).sum())

def final_calibration_check():
    import cnh_extrinsic_aug_check as B
    selection=read(OUT/'evaluator/final_calibration.json');v_selection=read(OUT/'evaluator/V_calibration.json')
    assert selection['status']==v_selection['status']=='COMPLETE'
    assert selection['seeds']==v_selection['seeds']==[0,1,2]
    assert selection['V_calibration_sha256']==sha(OUT/'evaluator/V_calibration.json')
    for value in (selection,v_selection):
        assert value['plan_sha256']==sha(OUT/'PLAN.json') and value['revision_gate_sha256']==sha(OUT/'evaluator/pilot_gate.json')
    with np.load(OUT/'geometry/calibration.npz') as z:g={k:z[k] for k in z.files}
    np.testing.assert_array_equal(g['unit'],np.repeat(np.arange(98000,98048),80))
    np.testing.assert_array_equal(g['config'],np.tile(np.repeat(np.arange(40),2),48))
    np.testing.assert_array_equal(g['query'],np.tile([0,1],1920))
    references=[];families={};mean_error={};sealed_hashes={}
    for phase,label in [('ensemble3','augmented'),('V3','V')]:
        raw=[];reference=[];largest_error=0.;receipt=read(OUT/f'scores/calibration/{phase}/receipt.json')
        assert receipt['status']=='COMPLETE' and receipt['augmented_seeds']==[0,1,2] and receipt['plan_sha256']==sha(OUT/'PLAN.json')
        for path,digest in receipt['model_sha256'].items():assert sha(path)==digest
        for u in range(98000,98048):
            path=OUT/f'scores/calibration/{phase}/unit{u}.npz'
            with np.load(path) as z:
                assert int(z['unit'])==u and z['seeds'].tolist()==[0,1,2]
                assert z['configs'].tolist()==list(range(40)) and z['frames'].tolist()==list(range(3,16)) and z['sensors'].tolist()==['S','L','R']
                assert z['reference'].shape==z['augmented_mean'].shape==(3,40,13,2)
                by_seed=z['augmented_by_seed'];assert by_seed.shape==(3,3,40,13,2) and np.isfinite(by_seed).all()
                error=np.abs(z['augmented_mean']-by_seed.astype(np.float64).mean(0))
                assert np.all(error<=4*np.finfo(np.float32).eps*np.maximum(1,np.abs(by_seed).max(0)))
                largest_error=max(largest_error,float(error.max()))
                with np.load(PARENT/f'scores/calibration/seed0/unit{u}.npz') as parent:
                    np.testing.assert_array_equal(z['reference'],parent['reference'])
                raw.append(B.independent_smooth(z['augmented_mean']).transpose(1,3,0,2).reshape(80,3,13))
                reference.append(B.independent_smooth(z['reference']).transpose(1,3,0,2).reshape(80,3,13))
            sealed_hashes[str(path)]=sha(path)
        families[label]=np.concatenate(raw);references.append(np.concatenate(reference));mean_error[label]=largest_error
    np.testing.assert_array_equal(references[0],references[1]);m3=references[0];costs={}
    for label,value in [('augmented',selection),('V',v_selection)]:
        dual=families[label][:,1:].max(1);theta,budget,n,stops=independent_cost_threshold(dual,m3[:,0],g['clear_all'])
        assert theta==value['threshold'] and budget==value['original_single_stops'] and n==value['calibration_physical_clear_n'] and stops==value['new_dual_stops']
        costs[label]=dict(threshold=theta,baseline_single_stops=budget,physical_joint_clear_n=n,family_dual_stops=stops)
        for path,digest in value['input_sha256'].items():assert sha(path)==digest
    arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=families['augmented'][:,0],DUAL_NEW=families['augmented'][:,1:].max(1),S_V=families['V'][:,0],DUAL_V=families['V'][:,1:].max(1))
    masks=dict(all=np.ones(3840,bool),mode2=g['mode']==2,turn_left=(g['mode']==2)&(g['turn']=='left'),turn_right=(g['mode']==2)&(g['turn']=='right'))
    cells={}
    for arm,scores in arms.items():
        theta=selection['threshold'] if arm.endswith('NEW') else v_selection['threshold'] if arm.endswith('_V') else .8557642486787612
        cells[arm]={name:B.independent_metrics(scores,theta,g,mask,arms['S_M3'],.8557642486787612) for name,mask in masks.items()}
    return dict(status='PASS',costs=costs,calibration_cells=cells,raw_three_seed_mean_max_abs_error=mean_error,
                M3_reference_exact_parent_seed0_parity=True,raw_smoothing_independently_rebuilt=True,
                score_payloads=96,score_sha256=sealed_hashes,evaluation_payload_access=False,GPU_operations=False)

def natural_evaluation_check():
    import cnh_extrinsic_aug_check as B
    selection=read(OUT/'evaluator/final_calibration.json');vc=read(OUT/'evaluator/V_calibration.json')
    assert selection['status']==vc['status']=='COMPLETE' and selection['seeds']==vc['seeds']==[0,1,2]
    result=read(OUT/'evaluator/natural_six_arm_result.json');main=read(OUT/'evaluator/natural_result.json')
    assert result['status']==main['status']=='COMPLETE' and result['calibration_sha256']==sha(OUT/'evaluator/final_calibration.json')
    with np.load(OUT/'evaluator/natural_six_arm_ledger.npz') as z:g={k:z[k] for k in z.files}
    units=np.unique(g['unit']);np.testing.assert_array_equal(units,np.arange(99000,99096))
    expected_keys=[];references=[];families={};configs_by_unit={};common_count=None
    for phase,family in [('ensemble3','NEW'),('V3','V')]:
        scores=[];refs=[];batch=read(OUT/f'scores/evaluation/{phase}/receipt.json')
        assert batch['status']=='COMPLETE' and batch['augmented_seeds']==[0,1,2] and batch['plan_sha256']==sha(OUT/'PLAN.json')
        for unit in units:
            with np.load(OUT/f'scores/evaluation/{phase}/unit{unit}.npz') as z:
                ids=z['configs'].tolist();count=len(ids)
                assert count==len(set(ids)) and 0<count<=40 and set(ids)<=set(range(40))
                if common_count is None:common_count=count
                assert count==common_count and int(z['unit'])==unit and z['seeds'].tolist()==[0,1,2]
                assert z['reference'].shape==z['augmented_mean'].shape==(3,count,13,2)
                if family=='NEW':
                    truth=read(OUT/f'truth/evaluation/unit{unit}.json');assert [s['config'] for s in truth['scenes']]==ids
                    assert truth['unit']==unit and truth['mode']==unit%3 and truth['mirror']==bool((unit//3)%2)
                    expected_keys.extend((int(unit),int(c),q) for c in ids for q in (0,1));configs_by_unit[str(unit)]=ids
                else:assert configs_by_unit[str(unit)]==ids
                scores.append(B.independent_smooth(z['augmented_mean']).transpose(1,3,0,2).reshape(count*2,3,13))
                refs.append(B.independent_smooth(z['reference']).transpose(1,3,0,2).reshape(count*2,3,13))
        references.append(np.concatenate(refs));families[family]=np.concatenate(scores)
    assert list(zip(g['unit'],g['config'],g['query']))==expected_keys
    assert result['configs_by_unit']==configs_by_unit and result['config_count']==common_count and result['query_episodes']==96*common_count*2
    np.testing.assert_array_equal(references[0],references[1]);m3=references[0]
    arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=families['NEW'][:,0],DUAL_NEW=families['NEW'][:,1:].max(1),S_V=families['V'][:,0],DUAL_V=families['V'][:,1:].max(1))
    for name,values in arms.items():np.testing.assert_array_equal(g[name],values)
    masks=dict(all=np.ones(len(g['unit']),bool),mode0=g['mode']==0,mode1=g['mode']==1,mode2=g['mode']==2,turn_left=(g['mode']==2)&(g['turn']=='left'),turn_right=(g['mode']==2)&(g['turn']=='right'))
    rebuilt={}
    for name,scores in arms.items():
        theta=selection['threshold'] if name.endswith('NEW') else vc['threshold'] if name.endswith('_V') else .8557642486787612
        rebuilt[name]={}
        for group,mask in masks.items():
            cell=B.independent_metrics(scores,theta,g,mask,arms['S_M3'],.8557642486787612);rebuilt[name][group]=cell
            B.compare_nested(result['cells'][name][group],cell)
        if name.endswith('_V'):
            augmented=arms[{'S_V':'S_NEW','DUAL_V':'DUAL_NEW'}[name]]
            for group,mask in masks.items():
                B.compare_nested(result['paired_V_vs_augmented'][name][group],B.independent_metrics(scores,theta,g,mask,augmented,selection['threshold']))
    base=rebuilt['S_M3'];dual=rebuilt['DUAL_NEW'];single=rebuilt['S_NEW']
    def cost(a,b,factor):return a['stops']<=factor*b['stops'] if a['n'] and b['n'] else None
    checks=dict(dual_physicalclear_1p15=cost(dual['all']['physical_clear'],base['all']['physical_clear'],1.15),
                dual_turn_queryclear_1p5=cost(dual['mode2']['query_clear'],base['mode2']['query_clear'],1.5),
                dual_shallow0_5_net_minus2=dual['all']['shallow0_5']['net']>=-2 if dual['all']['shallow0_5']['n'] else None,
                dual_turn_shallow0_2_net_minus1=dual['mode2']['shallow0_2']['net']>=-1 if dual['mode2']['shallow0_2']['n'] else None,
                single_shallow0_2_net_minus2=single['all']['shallow0_2']['net']>=-2 if single['all']['shallow0_2']['n'] else None,
                single_deep_net_minus2=single['all']['deep']['net']>=-2 if single['all']['deep']['n'] else None,
                single_physicalclear_1p1=cost(single['all']['physical_clear'],base['all']['physical_clear'],1.1))
    assert main['checks']==checks
    for direction in ('left','right'):
        assert int(masks['turn_'+direction].sum())==16*common_count*2
    return dict(status='PASS',units=96,config_count=common_count,physical_sequences=96*common_count,query_episodes=96*common_count*2,
                six_arm_cells=rebuilt,checks=checks,actual_config_ids=configs_by_unit,all_raw_scores_smoothing_and_ledger_exact=True,
                V_paired_differences_checked=True,calibration_selection_unchanged=True,GPU_operations=False)

def angular_evaluation_check():
    import cnh_extrinsic_aug_check as B
    selection=read(OUT/'evaluator/final_calibration.json');vc=read(OUT/'evaluator/V_calibration.json')
    assert selection['status']==vc['status']=='COMPLETE' and selection['seeds']==vc['seeds']==[0,1,2]
    result=read(OUT/'evaluator/angular_six_arm_result.json')
    assert result['status']=='COMPLETE' and result['calibration_sha256']==sha(OUT/'evaluator/final_calibration.json')
    with np.load(OUT/'evaluator/angular_six_arm_ledger.npz') as z:ledger={k:z[k] for k in z.files}
    import cnh_dual_sensor_evaluate as D
    old=D.load_baseline();units=ledger['units'];assert len(units)==12
    indices=[old['units'].tolist().index(int(u)) for u in units];scenes=[old['scenes'][i] for i in indices]
    assert all(s['mode']==0 for s in scenes)
    covered=old['covered'][indices];cats=old['all_categories'][indices]
    np.testing.assert_array_equal(ledger['covered'],covered)
    clear=np.broadcast_to((cats=='clear').all(2)[:,:,None,:],(12,7,4,2)).reshape(-1)
    ranges=np.broadcast_to(np.asarray([s['front_range_m'] for s in scenes])[:,None,None,None,:],(12,7,4,2,13)).reshape(-1,13)
    np.testing.assert_array_equal(clear,ledger['clear_query']);np.testing.assert_array_equal(ranges,ledger['ranges'])
    axes=[-15,-5,0,5,10,15,20,25,30,35,40,45];lookup={a:i for i,a in enumerate(axes)};families={};references=[]
    for phase,family in [('ensemble3','NEW'),('V3','V')]:
        values=[];ref=[]
        assert read(OUT/f'scores/angular/{phase}/receipt.json')['status']=='COMPLETE'
        for u in units:
            with np.load(OUT/f'scores/angular/{phase}/unit{int(u)}.npz') as z:
                assert int(z['unit'])==u and z['seeds'].tolist()==[0,1,2] and z['axes_degrees'].tolist()==axes
                assert z['configs'].tolist()==list(range(28)) and z['variants'].tolist()==(np.arange(28)//4).tolist() and z['replicas'].tolist()==(np.arange(28)%4).tolist()
                assert z['reference'].shape==z['augmented_mean'].shape==(12,28,13,2)
                ref.append(B.independent_smooth(z['reference']).reshape(12,7,4,13,2))
                values.append(B.independent_smooth(z['augmented_mean']).reshape(12,7,4,13,2))
        references.append(np.stack(ref));families[family]=np.stack(values)
    np.testing.assert_array_equal(references[0],references[1]);families['M3']=references[0]
    masks={}
    for group in ('all','opposite','same'):
        shallow=np.zeros((12,7,4,2),bool);deep=shallow.copy();eligible=shallow.copy()
        for i,s in enumerate(scenes):
            if group=='all' or (group=='same')==bool(s['fov_in']):
                shallow[i,:2,:,s['group']]=covered[i,:2,None];deep[i,2,:,s['group']]=covered[i,2];eligible[i,:3,:,s['group']]=True
        masks[group]=(shallow.reshape(-1),deep.reshape(-1),eligible.reshape(-1))
    assert int(masks['opposite'][0].sum())==48
    def event(scores,theta):
        alarm=scores>=theta;stopped=alarm.any(1);first=alarm.argmax(1);front=ranges[np.arange(len(scores)),first]
        return dict(alarm=alarm,stopped=stopped,first_index=first,first_range=front,timely=stopped&(front>=.9))
    checks={};counts={}
    for psi in (0,10,15,20,25,30):
        arms={}
        for family,values in families.items():
            arms['S_'+family]=values[:,lookup[psi]].transpose(0,1,2,4,3).reshape(-1,13)
            arms['DUAL_'+family]=values[:,[lookup[psi-15],lookup[psi+15]]].max(1).transpose(0,1,2,4,3).reshape(-1,13)
        base=event(arms['S_M3'],.8557642486787612);counts[str(psi)]={}
        for arm,score in arms.items():
            theta=selection['threshold'] if arm.endswith('NEW') else vc['threshold'] if arm.endswith('_V') else .8557642486787612
            ev=event(score,theta);np.testing.assert_array_equal(ledger[f'psi{psi}_{arm}'],score)
            for key,value in ev.items():np.testing.assert_array_equal(ledger[f'psi{psi}_{arm}_{key}'],value)
            cell={}
            for group,(shallow,deep,eligible) in masks.items():
                item={}
                for label,den in [('shallow',shallow),('deep',deep)]:
                    n=int(den.sum());num=int((den&ev['timely']).sum());resc=int((den&ev['timely']&~base['timely']).sum());loss=int((den&~ev['timely']&base['timely']).sum())
                    item[label]=dict(timely=num,n=n,rate=num/n if n else None,rescue=resc,loss=loss,net=resc-loss,baseline_timely=int((den&base['timely']).sum()))
                missing=eligible&~np.broadcast_to(covered[:,:,None,None],(12,7,4,2)).reshape(-1)
                item['censored']=dict(n=int(missing.sum()),stopped=int((missing&ev['stopped']).sum()));cell[group]=item
            joint=clear.reshape(-1,2).all(1)
            for name,den,stopped in [('query_clear',clear,ev['stopped']),('physical_clear',joint,ev['stopped'].reshape(-1,2).any(1))]:
                n=int(den.sum());s=int((den&stopped).sum());minutes=n*2.6/60
                cell[name]=dict(stops=s,n=n,proxy_minutes=minutes,rate_per_proxy_minute=s/minutes if n else None)
            B.compare_nested(result['sequence'][str(psi)][arm],cell);counts[str(psi)][arm]=cell['opposite']['shallow']
            if arm.endswith('_V'):
                augmented=event(arms[{'S_V':'S_NEW','DUAL_V':'DUAL_NEW'}[arm]],selection['threshold'])
                for group,(shallow,deep,eligible) in masks.items():
                    for label,den in [('shallow',shallow),('deep',deep)]:
                        resc=int((den&ev['timely']&~augmented['timely']).sum());loss=int((den&~ev['timely']&augmented['timely']).sum())
                        expected=dict(rescue=resc,loss=loss,net=resc-loss,n=int(den.sum()),V_timely=int((den&ev['timely']).sum()),augmented_timely=int((den&augmented['timely']).sum()))
                        B.compare_nested(result['paired_V_vs_augmented'][str(psi)][arm][group][label],expected)
        if psi<=25:
            c=counts[str(psi)]['DUAL_NEW'];checks[str(psi)]=c['timely']>=.9*c['n'] if c['n'] else None
    assert checks==result['tolerance_checks']
    assert result['tolerance_psi_le25_pass']==(None if any(v is None for v in checks.values()) else all(checks.values()))
    return dict(status='PASS',opposite_denominator=48,opposite_counts=counts,tolerance_checks=checks,
                all_six_arm_raw_scores_events_cells_and_pairing_checked=True,original_retained_membership_checked=True,GPU_operations=False)

def judgment_check():
    natural=read(OUT/'checks/natural-evaluation.json');angular=read(OUT/'checks/angular-evaluation.json')
    assert natural['status']==angular['status']=='PASS'
    checks=dict(natural['checks']);values=angular['tolerance_checks'].values()
    checks['angular_opposite_shallow_psi_le25']=None if any(v is None for v in values) else all(values)
    turn=[checks['dual_turn_queryclear_1p5'],checks['dual_turn_shallow0_2_net_minus1']]
    expected='NOT_EVALUABLE' if any(v is None for v in checks.values()) else 'EXTRINSIC_AUG_SUPPORTED_SIM' if all(checks.values()) else 'TRADEOFF' if all(turn) else 'NOT_SUPPORTED'
    main=read(OUT/'evaluator/result.json');continuation=read(OUT/'evaluator/continuation_result.json')
    assert main['checks']==continuation['checks']==checks and main['judgment']==continuation['judgment']==expected
    assert continuation['original_pilot_judgment']=='FAIL' and continuation['revised_pilot_judgment']=='PASS' and continuation['V_descriptive_only']
    assert sha(PARENT/'PLAN.json')=='86a33853de8d02315fcd16a0a07f0a737480aa41d947457da2012c7bfd35a9ee'
    assert sha(PARENT/'evaluator/pilot_gate.json')=='e22b16e5c86f5f44289ea2601563cee3dba93fe39509b89342613b25acf6ed84'
    for path,digest in read(PARENT/'checks/plan.json')['original_M3_model_sha256'].items():assert sha(path)==digest
    for path,digest in read(OUT/'checks/v-models.json')['models_sha256'].items():assert sha(path)==digest
    for relative in ('final_calibration.json','V_calibration.json'):
        value=read(OUT/'evaluator'/relative)
        assert value['plan_sha256']==sha(OUT/'PLAN.json') and value['revision_gate_sha256']==sha(OUT/'evaluator/pilot_gate.json')
    return dict(status='PASS',judgment=expected,checks=checks,old_pilot_fail_and_original_models_preserved=True,
                V_descriptive_excluded_from_main_judgment=True,resource_release='All reviewer commands are finite CPU-only; no worker or GPU allocation retained.',GPU_operations=False)

def old_angular_reference_check():
    old_root=ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005/fusion'
    old=read(old_root/'angular_result.json');new=read(OUT/'evaluator/angular_six_arm_result.json')
    assert old['status']==new['status']=='COMPLETE' and old['units']==new['units'] and old['psi_deg']==new['psi_deg']
    curves={};largest_error=0.;comparisons=0
    with np.load(old_root/'angular_ledger.npz') as before,np.load(OUT/'evaluator/angular_six_arm_ledger.npz') as after:
        for old_name,new_name in [('single','S_M3'),('OR','DUAL_M3')]:
            curve=[]
            for psi in old['psi_deg']:
                previous=old['sequence'][str(psi)][old_name]['opposite']['shallow'];current=new['sequence'][str(psi)][new_name]['opposite']['shallow']
                assert previous['n']==current['n']==48 and previous['stops']==current['timely'];curve.append(current['timely'])
                a=before[f'psi{psi}_{old_name}'];b=after[f'psi{psi}_{new_name}'];error=float(np.abs(a-b).max())
                assert np.all(np.abs(a-b)<=64*np.finfo(np.float64).eps*np.maximum(1,np.abs(a)))
                largest_error=max(largest_error,error)
                for key in ('alarm','stopped','first_index','first_range','timely'):
                    np.testing.assert_array_equal(before[f'psi{psi}_{old_name}_{key}'],after[f'psi{psi}_{new_name}_{key}'])
                comparisons+=1
            curves[new_name]=curve
    assert curves['S_M3']==[47,47,13,2,0,0] and curves['DUAL_M3']==[45,47,47,45,47,13]
    return dict(status='PASS',source_result_sha256=sha(old_root/'angular_result.json'),source_ledger_sha256=sha(old_root/'angular_ledger.npz'),
                psi=old['psi_deg'],opposite_denominator=48,curves=curves,score_max_abs_difference=largest_error,
                paired_score_arrays=12,paired_event_arrays=60,all_alarm_stop_first_frame_distance_timely_arrays_exact=True,
                limit='Descriptive source-reuse check, not an additional outcome gate. Difference only at float64 summation rounding.',GPU_operations=False)

def main():
    stages={'plan':plan_check,'v-models':v_models_check,'source':source_check,'final-calibration':final_calibration_check,'natural-evaluation':natural_evaluation_check,'angular-evaluation':angular_evaluation_check,'old-angular-reference':old_angular_reference_check,'judgment':judgment_check}
    p=argparse.ArgumentParser();p.add_argument('stage',choices=stages);a=p.parse_args();tick=time.monotonic()
    result=stages[a.stage]()
    result.update(seconds=time.monotonic()-tick,checker_sha256=sha(__file__))
    ast.parse(Path(__file__).read_text(encoding='utf8'));record(a.stage+'.json',result)
    quiet={'calibration_cells','six_arm_cells','score_sha256','actual_config_ids'}
    print(json.dumps({k:v for k,v in result.items() if k not in quiet}),flush=True)

if __name__=='__main__':main()
