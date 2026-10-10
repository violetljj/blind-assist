"""Focused independent localized score/cost/cohort audit; no producer import."""
import argparse
from pathlib import Path
import time
import numpy as np
import audit_cnh_graded_weak_mass_dev as V

C,G,E,A=V.C,V.G,V.E,V.A
PARENT=V.PARENT
MASS=V.OUT


def run(out):
    began=time.monotonic();audit=out/'verification'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve localized audit attempt')
    V.save(audit/'PLAN.json',dict(task='LOCALIZED_WEAK_PROFILE_INDEPENDENT_AUDIT',
        inherited_verification_CPU_command_wall_seconds_cap=180,producer_imported=False,
        scope='Six new cells, three minimum cuts, complete grade/base/strong identities, 180 shape/rho cohorts and paired full/timely versus head50/control',
        source_sha256=C.sha(Path(__file__))))
    try:
        plan=C.read(out/'PLAN.json')
        for path,digest in plan['inputs_sha256'].items(): V.close(C.sha(C.ROOT/path),digest,'input/'+path)
        V.close(C.sha(Path(__file__).with_name('cnh_graded_weak_profile_dev.py')),plan['source_sha256'],'producer source')
        data=C.load();thresholds=C.read(C.PARENT/'thresholds.json')
        models=C.read(out/'models.json');cuts=C.read(out/'calibrations.json')
        metrics=C.read(out/'metrics.json');cohorts=C.read(out/'cohorts.json')
        fit=np.array([r['background_id'] in (8,10) for r in data['cal']['rows']])
        V.close(C.read(out/'cal_partition.json'),dict(fit_background_ids=[8,10],calibrate_background_ids=[9,11]),'partition')
        report={};calibration={};cells=0;groups=0
        # Eligible training-only empty-column repair: independently reconstruct
        # input validity, not a model forward pass or validation-selected mask.
        current=V.load_npz(C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/features/cal_features.npz')
        xx=current['current_features'].astype(float)
        vv=current['current_valid']&np.isfinite(xx)
        current_x=np.concatenate((np.where(vv,xx,np.nan),(~vv).astype(float)),-1)
        profile=V.load_npz(MASS/'localized/features/cal_features.npz')
        xx=profile['features'].astype(float);vv=profile['valid']
        profile_x=np.concatenate((np.where(vv,xx,np.nan),(~vv).astype(float)),-1)
        for split,d in data.items():
            categories=d['category']
            archive=V.load_npz(out/f'{split}_grades.npz');scores=V.load_npz(out/f'{split}_scores.npz')
            keys=[f'{seed}/weak_profile' for seed in G.SEEDS]
            for kind,item in (('grades',archive),('scores',scores)):
                V.close(item['keys'].tolist(),keys,split+'/'+kind+'/keys')
                V.close(item['scene_ids'],d['scene_ids'],split+'/'+kind+'/ids')
            parents=V.load_npz(PARENT/f'{split}_grades.npz');parents=dict(zip(parents['keys'].tolist(),parents['grades']))
            control=V.load_npz(MASS/f'{split}_grades.npz');control=dict(zip(control['keys'].tolist(),control['grades']))
            for si,seed in enumerate(G.SEEDS):
                if time.monotonic()-began>180:raise TimeoutError('Inherited audit180s cap reached')
                key=f'{seed}/weak_profile';pubkey=f'{split}/{key}';score=scores['scores'][si]
                th=thresholds[str(seed)];ordinary=d['candidates'][0,si]
                strong=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)|(ordinary>=th['addition'])
                base=np.where(strong,2,np.where(ordinary>=th['single'],1,0)).astype(np.int8)
                V.close(base,parents[f'{seed}/baseline'],pubkey+'/baseline')
                cut=cuts[key];theta=-np.inf if cut['nonbinding'] else cut['theta']
                grade=base.copy();grade[(base==0)&np.isfinite(score)&(score>=theta)]=1
                V.close(archive['grades'][si],grade,pubkey+'/score-grade')
                V.close(grade==2,strong,pubkey+'/strong exact');V.close(grade[base>0],base[base>0],pubkey+'/base retained')
                head=parents[f'{seed}/head50'];ctrl=control[f'{seed}/current_control']
                expected=A.describe(grade,d,dict(prior_light=head>0,ordinary_OR=strong,M3=d['m3']>=E.M3_THETA,old_fusion=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)))
                for name,value in expected.items():V.close(metrics[pubkey][name],value,pubkey+'/'+name)
                V.close(metrics[pubkey]['cost'],V.flat_cost(grade,categories),pubkey+'/cost')
                V.close(metrics[pubkey]['head50_cost'],V.flat_cost(head,categories),pubkey+'/head50 cost')
                V.close(metrics[pubkey]['physical_vs_head50'],A.physical_pair(head>0,grade>0,categories),pubkey+'/physical')
                for name,old in (('head50',head),('current_control',ctrl)):
                    V.close(metrics[pubkey]['paired_vs_'+name],A.paired(old>0,grade>0,categories),pubkey+'/paired/'+name)
                report[pubkey]={}
                for q,height in enumerate(E.HEIGHTS):
                    for family in ('all',*sorted({r['shape_family'] for r in d['rows']})):
                        for rho in ('all',.25,.65):
                            mask=np.array([(family=='all' or r['shape_family']==family) and (rho=='all' or r['rho']==rho) for r in d['rows']])
                            group=f'{height}/{family}/{rho}';timing=V.contact_report(head,grade,categories,q,mask)
                            V.close(cohorts[pubkey]['outcomes'][group],dict(events=timing['timely']['denominator'],**timing['after_outcomes']),pubkey+'/'+group+'/outcome')
                            report[pubkey][group]=dict(versus_head50=timing,versus_current_control=V.contact_report(ctrl,grade,categories,q,mask))
                            for name in ('head50','current_control'):
                                pair=report[pubkey][group]['versus_'+name]['timely']
                                V.close(cohorts[pubkey]['paired_vs_'+name][group],dict(events=pair['denominator'],**{n:pair[n] for n in ('rescue','loss','earlier','later')}),pubkey+'/'+group+'/paired/'+name)
                            groups+=1
                if split=='cal':
                    raw=E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_cal.npz','ordinary',seed,'cal','ideal')
                    scalar=C.build_score_features(raw,ordinary,th['single'])
                    training_tensor=np.concatenate((scalar,current_x,profile_x),-1)
                    assert training_tensor.shape[-1]==239
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
                        m=models[f'{key}/{height}']
                        V.close(m['rows'],int(eligible.sum()),key+'/'+height+'/fit rows');V.close(m['contact_rows'],int(yy.sum()),key+'/'+height+'/labels')
                        V.close(m['eligible_scene_replica_groups'],int((count>0).sum()),key+'/'+height+'/groups')
                        keep=np.isfinite(training_tensor[fit,...,q,:][eligible]).any(0)
                        V.close(m['input_dimensions'],239,key+'/'+height+'/input dimensions')
                        V.close(m['dimensions'],int(keep.sum()),key+'/'+height+'/retained dimensions')
                        V.close(m['retained_feature_indices'],np.flatnonzero(keep).tolist(),key+'/'+height+'/training-only mask')
                        V.close(m['all_missing_training_feature_indices'],np.flatnonzero(~keep).tolist(),key+'/'+height+'/all-missing columns')
                        V.close(C.sha(out/m['path']),m['sha256'],key+'/'+height+'/hash')
                cells+=1
        V.close(len(models),6,'six new fits');V.close(len(metrics),6,'six cells');V.close(len(cohorts),6,'six cohort cells')
        V.save(audit/'comparison.json',report);V.save(audit/'calibration.json',calibration)
        V.save(audit/'receipt.json',dict(status='PASS',seconds=time.monotonic()-began,cells=cells,cohorts=groups,
            fits=6,cuts=3,producer_imported=False,new_fit=0,new_prediction=0,full_window_timing_recomputed=True,
            source_sha256=C.sha(Path(__file__)),GPU_seconds=0))
        print(f'PASS cells{cells} cohorts{groups} fits6 cuts3 seconds{time.monotonic()-began:.3f}')
    except BaseException as error:
        V.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=MASS/'localized/repair1')
    run(parser.parse_args().output)
