"""Post-diagnostic localized public-depth readout; global-mass run is retained.

Uses a different spatial representation, not a threshold/parameter retry.
All training/cost conventions and matched-current checkpoint are inherited.
"""
from pathlib import Path
import pickle
import time
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_graded_weak_mass_dev as M

C,J,H,G,E,I = M.C,M.J,M.H,M.G,M.E,M.I
PROFILE=M.OUT/'localized'
OUT=PROFILE/'repair1'


def run():
    started=time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve localized profile comparison')
    inputs=[M.OUT/'TASK.json',M.OUT/'receipt.json',M.OUT/'metrics.json',M.OUT/'calibrations.json',
            M.OUT/'control_checkpoint_parity.json',Path(__file__),Path(M.__file__),PROFILE/'features/schema.json',PROFILE/'PLAN.json']
    for split in ('cal','validation'):
        inputs += [M.OUT/f'{split}_scores.npz',PROFILE/'features'/f'{split}_features.npz',J.OUT/'features'/f'{split}_features.npz',H.OUT/f'{split}_grades.npz']
    C.save(OUT/'PLAN.json',dict(task='CNH_GRADED_WEAK_PROFILE_DEV_20261010',amendment_of=M.OUT.relative_to(C.ROOT).as_posix(),
        lane='EXPLORE consumed ideal simulated Development, post-negative selected representation',
        problem='Global all-bin mass did not rescue low-rho horizontal contacts consistently; retain depth location rather than only pooled strength/duration',
        mechanism='Fixed12forward-depth bins [.3,3] equally spaced, inner/ring; current positive/negative and separate-before-mean history densities,96 features. Cached public mean supporting-node depth defines bin approximation.',
        scope='One localized bank; same fixed100tree HGB recipe,6new fits. Reuse current control fitted on identical cal8/10 original grade0 population and weights, parity proved exactly. No scan/tuning/hold access.',
        cost='Inherit cal9/11 lowest whole-tie shared cut with total union clear slots and purepass clips <= original head50; baseline strong and ordinary light retained',
        budget='Inherited total1050CPUcommand-wall/120GPUphase; no expansion. First features9.875s/GPU6.864s,fit/evaluation23s upper,audit2.340s deducted; localizedfit/eval under remaining337 of shared360s phase.',
        dimensions=239,stop='Complete this different localized representation and matched evaluation; preserve both failures/results without returning to the previous mass recipe',
        repair_of=PROFILE.relative_to(C.ROOT).as_posix(),repair_reason='Observed HGB binning failure on wholly NaN training columns. Drop columns with no finite value in eligible training population, separately per height; inference uses same retained columns. Missing indicators remain, no zero/free imputation or validation selection.',
        prior_failed_fit_command_wall_seconds_upper_bound=6,
        inputs_sha256={p.relative_to(C.ROOT).as_posix():C.sha(p) for p in inputs},source_sha256=C.sha(Path(__file__))))
    try:
        data=C.load();fit_mask,partition=C.split_cal(data['cal']['rows']);C.save(OUT/'cal_partition.json',partition)
        thresholds=C.read(C.PARENT/'thresholds.json')
        current,profile,parents,controls={},{},{},{}
        for split,d in data.items():
            with np.load(J.OUT/'features'/f'{split}_features.npz') as a:
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids']);current[split],current_names=J.descriptors(a,'current')
            with np.load(PROFILE/'features'/f'{split}_features.npz') as a:
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids']);x,v,names=a['features'].astype(float),a['valid'],a['names'].tolist()
                profile[split]=np.concatenate((np.where(v,x,np.nan),(~v).astype(float)),-1)
            with np.load(H.OUT/f'{split}_grades.npz') as a: parents[split]=dict(zip(a['keys'].tolist(),a['grades']))
            with np.load(M.OUT/f'{split}_scores.npz') as a: controls[split]=dict(zip(a['keys'].tolist(),a['scores']))
        C.save(OUT/'feature_names.json',dict(current=['ordinary_smooth_margin','ordinary_raw_slope5','ordinary_raw_detrended_fluctuation5']+current_names,
            extra=names+[n+'_missing' for n in names]))
        metrics,cohorts,cuts,models,summary={},{},{},{},[]
        grades_saved={s:[] for s in data};scores_saved={s:[] for s in data}
        for si,seed in enumerate(G.SEEDS):
            base,head50,tensors={},{},{}
            for split,d in data.items():
                base[split]=H.baseline(d,si,thresholds[str(seed)]);head50[split]=parents[split][f'{seed}/head50']
                raw=E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                s=C.build_score_features(raw,d['candidates'][0,si],thresholds[str(seed)]['single'])
                tensors[split]=np.concatenate((s,current[split],profile[split]),-1)
                assert tensors[split].shape[-1]==239
            score={s:np.empty(b.shape,float) for s,b in base.items()}
            for q,height in enumerate(E.HEIGHTS):
                if time.monotonic()-started>=331:raise TimeoutError('Remaining shared fit/eval budget reached')
                eligible=base['cal'][fit_mask,...,q]==0
                fit_values=tensors['cal'][fit_mask,...,q,:][eligible]
                keep=np.isfinite(fit_values).any(0)
                model,record=J.fit_height(tensors['cal'][...,keep],data['cal']['category'],base['cal'],fit_mask,q)
                path=OUT/f'models/{seed}_{height}.pickle';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(pickle.dumps(model,protocol=5))
                models[f'{seed}/weak_profile/{height}']=dict(**record,path=path.relative_to(OUT).as_posix(),sha256=C.sha(path),dimensions=int(keep.sum()),input_dimensions=239,
                    retained_feature_indices=np.flatnonzero(keep).tolist(),all_missing_training_feature_indices=np.flatnonzero(~keep).tolist())
                for split in data:
                    x=tensors[split][...,q,:][...,keep];score[split][...,q]=model.decision_function(x.reshape(-1,int(keep.sum()))).reshape(x.shape[:-1])
            cut=M.calibrate(score['cal'],base['cal'],head50['cal'],data['cal']['category'],~fit_mask)
            cuts[f'{seed}/weak_profile']=cut;theta=-np.inf if cut['nonbinding'] else cut['theta']
            control_cut=C.read(M.OUT/'calibrations.json')[f'{seed}/current_control']
            ct=-np.inf if control_cut['nonbinding'] else control_cut['theta']
            for split,d in data.items():
                control=np.where((base[split]==0)&(controls[split][f'{seed}/current_control']>=ct),1,base[split]).astype(np.int8)
                grade=np.where((base[split]==0)&np.isfinite(score[split])&(score[split]>=theta),1,base[split]).astype(np.int8)
                np.testing.assert_array_equal(grade==2,base[split]==2);assert np.all(grade>=base[split])
                refs=dict(prior_light=head50[split]>0,ordinary_OR=base[split]==2,M3=d['m3']>=E.M3_THETA,old_fusion=E.old_fusion(d['m3'],d['local']))
                report=G.describe(grade,d,refs)
                report.update(cost=M.costs(grade>0,d['category']),head50_cost=M.costs(head50[split]>0,d['category']),
                    paired_vs_head50=G.paired_timing(head50[split]>0,grade>0,d['category']),
                    paired_vs_current_control=G.paired_timing(control>0,grade>0,d['category']),
                    physical_vs_head50=H.B.physical_pair(head50[split]>0,grade>0,d['category']))
                key=f'{split}/{seed}/weak_profile';metrics[key]=report
                cohorts[key]=dict(outcomes=M.outcomes(grade,d['category'],d['rows']),
                    paired_vs_head50=M.paired_cohorts(head50[split],grade,d['category'],d['rows']),
                    paired_vs_current_control=M.paired_cohorts(control,grade,d['category'],d['rows']))
                summary.append(dict(split=split,seed=seed,arm='weak_profile',HEAD=report['any']['counts'][0],BODY=report['any']['counts'][1],
                    HEAD_rescue=report['paired_vs_head50'][0]['rescue'],HEAD_loss=report['paired_vs_head50'][0]['loss'],
                    BODY_rescue=report['paired_vs_head50'][1]['rescue'],BODY_loss=report['paired_vs_head50'][1]['loss'],**report['cost']))
                grades_saved[split].append((f'{seed}/weak_profile',grade));scores_saved[split].append((f'{seed}/weak_profile',score[split]))
            print('seed',seed,'COMPLETE',round(time.monotonic()-started,3),flush=True)
        for split,d in data.items():
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.array([k for k,v in grades_saved[split]]),grades=np.array([v for k,v in grades_saved[split]]),scene_ids=d['scene_ids'])
            np.savez_compressed(OUT/f'{split}_scores.npz',keys=np.array([k for k,v in scores_saved[split]]),scores=np.array([v for k,v in scores_saved[split]]),scene_ids=d['scene_ids'])
        G.write_csv(OUT/'summary.csv',summary)
        for name,v in [('metrics',metrics),('cohorts',cohorts),('calibrations',cuts),('models',models)]:C.save(OUT/f'{name}.json',v)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-started,fits=6,reused_control_fits=6,cells=6,
            source_sha256=C.sha(Path(__file__)),persistent_resources=0,new_backbone_forward=0,new_data=0))
        print('COMPLETE',round(time.monotonic()-started,3),flush=True)
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
