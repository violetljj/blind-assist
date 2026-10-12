"""Independent motion-path score/cost/cohort audit; no producer import.

Geometry extraction is audited separately. This audit independently reconstructs
grades, training population and column retention, whole-tie union-cost cuts,
timely/full-window outcomes and every reported shape/rho pairing.
"""
from pathlib import Path
import time
import numpy as np
import audit_cnh_graded_weak_mass_dev as V

C,G,E,A=V.C,V.G,V.E,V.A
OUT=C.ROOT/'artifacts.local/work/cnh-graded-motion-path-dev-20261010'
ARMS=('static_path','aligned_path')
CURRENT=C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
CONTROL=V.OUT


def descriptor(archive, bank):
    x=archive[bank+'_features'].astype(float)
    valid=archive[bank+'_valid']&np.isfinite(x)
    return np.concatenate((np.where(valid,x,np.nan),(~valid).astype(float)),-1)


def run():
    began=time.monotonic();audit=OUT/'verification'
    if (audit/'PLAN.json').exists():raise FileExistsError('Preserve independent motion audit attempt')
    V.save(audit/'PLAN.json',dict(task='MOTION_PATH_INDEPENDENT_AUDIT',
        CPU_command_wall_seconds_cap=120,producer_imported=False,
        scope='Twelve score-grade cells; six whole-tie cuts and union cost caps; twelve fit populations/column masks/model hashes; 360 shape/rho cohorts and paired full/timely versus head50/control/static path',
        source_sha256=C.sha(Path(__file__))))
    try:
        plan=C.read(OUT/'FIT_PLAN.json')
        for path,digest in plan['inputs_sha256'].items():V.close(C.sha(C.ROOT/path),digest,'input/'+path)
        V.close(C.sha(Path(__file__).with_name('cnh_graded_motion_path_dev.py')),plan['source_sha256'],'producer source')
        V.close(plan['params'],C.read(CURRENT/'PLAN.json')['params'],'fixed HGB recipe')
        data=C.load();thresholds=C.read(C.PARENT/'thresholds.json')
        models=C.read(OUT/'models.json');cuts=C.read(OUT/'calibrations.json')
        metrics=C.read(OUT/'metrics.json');cohorts=C.read(OUT/'cohorts.json')
        fit=np.array([r['background_id'] in (8,10) for r in data['cal']['rows']])
        V.close(C.read(OUT/'cal_partition.json'),dict(fit_background_ids=[8,10],calibrate_background_ids=[9,11]),'partition')
        current_x=descriptor(V.load_npz(CURRENT/'features/cal_features.npz'),'current')
        path_archive=V.load_npz(OUT/'features/cal_features.npz')
        path_x={}
        for arm,prefix in zip(ARMS,('static','aligned')):
            x=path_archive[prefix].astype(float);valid=path_archive[prefix+'_valid']
            assert np.isfinite(x[valid]).all(),arm+'/valid finite values'
            path_x[arm]=np.concatenate((np.where(valid,x,np.nan),(~valid).astype(float)),-1)
        V.close(path_archive['scene_ids'],data['cal']['scene_ids'],'training path feature scene identity')
        report={};calibration={};cells=groups=0
        for split,d in data.items():
            categories=d['category']
            archive=V.load_npz(OUT/f'{split}_grades.npz');scores=V.load_npz(OUT/f'{split}_scores.npz')
            keys=[f'{seed}/{arm}' for seed in G.SEEDS for arm in ARMS]
            for kind,item in (('grades',archive),('scores',scores)):
                V.close(item['keys'].tolist(),keys,split+'/'+kind+'/keys')
                V.close(item['scene_ids'],d['scene_ids'],split+'/'+kind+'/ids')
            saved=dict(zip(archive['keys'].tolist(),archive['grades']))
            scoremap=dict(zip(scores['keys'].tolist(),scores['scores']))
            parents=V.load_npz(V.PARENT/f'{split}_grades.npz');parents=dict(zip(parents['keys'].tolist(),parents['grades']))
            controls=V.load_npz(CONTROL/f'{split}_grades.npz');controls=dict(zip(controls['keys'].tolist(),controls['grades']))
            for si,seed in enumerate(G.SEEDS):
                th=thresholds[str(seed)];ordinary=d['candidates'][0,si]
                strong=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)|(ordinary>=th['addition'])
                base=np.where(strong,2,np.where(ordinary>=th['single'],1,0)).astype(np.int8)
                V.close(base,parents[f'{seed}/baseline'],split+'/'+str(seed)+'/baseline')
                head=parents[f'{seed}/head50'];ctrl=controls[f'{seed}/current_control']
                grades={}
                for arm in ARMS:
                    key=f'{seed}/{arm}';score=scoremap[key];cut=cuts[key]
                    theta=-np.inf if cut['nonbinding'] else cut['theta']
                    grade=base.copy();grade[(base==0)&np.isfinite(score)&(score>=theta)]=1
                    V.close(saved[key],grade,split+'/'+key+'/score-grade')
                    V.close(grade==2,strong,split+'/'+key+'/strong exact')
                    V.close(grade[base>0],base[base>0],split+'/'+key+'/base retained')
                    grades[arm]=grade
                for arm in ARMS:
                    if time.monotonic()-began>120:raise TimeoutError('Independent audit 120s cap reached')
                    key=f'{seed}/{arm}';pubkey=f'{split}/{key}';grade=grades[arm]
                    score=scoremap[key];cut=cuts[key];theta=-np.inf if cut['nonbinding'] else cut['theta']
                    comparisons=dict(head50=head,current_control=ctrl)
                    if arm=='aligned_path':comparisons['static_path']=grades['static_path']
                    expected=A.describe(grade,d,dict(prior_light=head>0,ordinary_OR=strong,M3=d['m3']>=E.M3_THETA,old_fusion=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)))
                    for name,value in expected.items():V.close(metrics[pubkey][name],value,pubkey+'/'+name)
                    V.close(metrics[pubkey]['cost'],V.flat_cost(grade,categories),pubkey+'/cost')
                    V.close(metrics[pubkey]['head50_cost'],V.flat_cost(head,categories),pubkey+'/head50 cost')
                    V.close(metrics[pubkey]['physical_vs_head50'],A.physical_pair(head>0,grade>0,categories),pubkey+'/physical')
                    for name,old in comparisons.items():V.close(metrics[pubkey]['paired_vs_'+name],A.paired(old>0,grade>0,categories),pubkey+'/paired/'+name)
                    report[pubkey]={}
                    for q,height in enumerate(E.HEIGHTS):
                        for family in ('all',*sorted({r['shape_family'] for r in d['rows']})):
                            for rho in ('all',.25,.65):
                                mask=np.array([(family=='all' or r['shape_family']==family) and (rho=='all' or r['rho']==rho) for r in d['rows']])
                                group=f'{height}/{family}/{rho}'
                                paired_reports={name:V.contact_report(old,grade,categories,q,mask) for name,old in comparisons.items()}
                                timing=paired_reports['head50']
                                V.close(cohorts[pubkey]['outcomes'][group],dict(events=timing['timely']['denominator'],**timing['after_outcomes']),pubkey+'/'+group+'/outcome')
                                report[pubkey][group]={'versus_'+name:r for name,r in paired_reports.items()}
                                for name,t in paired_reports.items():
                                    pair=t['timely'];value=dict(events=pair['denominator'],**{n:pair[n] for n in ('rescue','loss','earlier','later')})
                                    V.close(cohorts[pubkey]['paired_vs_'+name][group],value,pubkey+'/'+group+'/paired/'+name)
                                groups+=1
                    if split=='cal':
                        raw=E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_cal.npz','ordinary',seed,'cal','ideal')
                        scalar=C.build_score_features(raw,ordinary,th['single'])
                        training=np.concatenate((scalar,current_x,path_x[arm]),-1)
                        dimension=training.shape[-1]
                        V.close(dimension,plan['dimensions'][arm],key+'/declared dimensions')
                        category=categories[~fit];fixed=base[~fit];reference=head[~fit]
                        caps=V.flat_cost(reference,category);fixed_cost=V.flat_cost(fixed,category)
                        independent=V.cutoff_for_union(score[~fit],fixed,category,dict(clear=caps['clear_slots'],**{'pass':caps['pass_clips']}))
                        assert theta==independent,(key,theta,independent)
                        actual=V.flat_cost(grade[~fit],category)
                        assert actual['clear_slots']<=caps['clear_slots'] and actual['pass_clips']<=caps['pass_clips']
                        V.close(cut['reference_cost'],caps,key+'/caps');V.close(cut['fixed_base_cost'],fixed_cost,key+'/fixed cost');V.close(cut['actual_cost'],actual,key+'/actual cost')
                        V.close(cut['additional_caps'],dict(clear_slots=caps['clear_slots']-fixed_cost['clear_slots'],pass_clips=caps['pass_clips']-fixed_cost['pass_clips']),key+'/additional cap')
                        calibration[key]=dict(reference_cost=caps,actual_cost=actual,minimum_whole_tie_cutoff_exact=True)
                        for q,height in enumerate(E.HEIGHTS):
                            eligible=base[fit,...,q]==0;count=eligible.sum(-1)
                            yy=np.broadcast_to((categories[fit,q]=='contact')[:,None,None],eligible.shape)[eligible]
                            model=models[f'{key}/{height}']
                            V.close(model['rows'],int(eligible.sum()),key+'/'+height+'/fit rows')
                            V.close(model['contact_rows'],int(yy.sum()),key+'/'+height+'/labels')
                            V.close(model['eligible_scene_replica_groups'],int((count>0).sum()),key+'/'+height+'/groups')
                            V.close(model['weight_sum'],float(eligible.sum()),key+'/'+height+'/mean-one weights')
                            groups_present=count>0
                            contact=np.broadcast_to((categories[fit,q]=='contact')[:,None],count.shape)
                            probability=float(contact[groups_present].mean())
                            baseline_logit=float(np.log(probability/(1-probability)))
                            V.close(model['baseline_logit'],baseline_logit,key+'/'+height+'/equal-group weighted initial logit')
                            keep=np.isfinite(training[fit,...,q,:][eligible]).any(0)
                            V.close(model['input_dimensions'],dimension,key+'/'+height+'/input dimensions')
                            V.close(model['dimensions'],int(keep.sum()),key+'/'+height+'/retained dimensions')
                            V.close(model['retained_feature_indices'],np.flatnonzero(keep).tolist(),key+'/'+height+'/training-only mask')
                            V.close(model['all_missing_training_feature_indices'],np.flatnonzero(~keep).tolist(),key+'/'+height+'/all-missing columns')
                            assert model['iterations']<=100
                            V.close(C.sha(OUT/model['path']),model['sha256'],key+'/'+height+'/hash')
                    cells+=1
        V.close(len(models),12,'twelve fits');V.close(len(metrics),12,'twelve cells');V.close(len(cohorts),12,'twelve cohort cells')
        V.save(audit/'comparison.json',report);V.save(audit/'calibration.json',calibration)
        V.save(audit/'receipt.json',dict(status='PASS',seconds=time.monotonic()-began,cells=cells,cohorts=groups,
            fits=12,cuts=6,producer_imported=False,new_fit=0,new_prediction=0,full_window_timing_recomputed=True,
            training_only_column_retention_recomputed=True,source_sha256=C.sha(Path(__file__)),GPU_seconds=0))
        print(f'PASS cells{cells} cohorts{groups} fits12 cuts6 seconds{time.monotonic()-began:.3f}')
    except BaseException as error:
        V.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__':run()
