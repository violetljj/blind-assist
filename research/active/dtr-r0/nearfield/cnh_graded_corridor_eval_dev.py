"""Cal-only small linear light-alert score, fixed strong ordinary-OR.

Runtime feature construction is separate from truth joins. Both score-only and
spatial models use the same fit/calibration backgrounds and declared budgets.
This is consumed simulated Development exploration, not risk probability.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
import cnh_counterfactual_eval_dev as E
import cnh_graded_evidence_dev as G

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
PARENT = ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010'
OUT = ROOT/'artifacts.local/work/cnh-graded-corridor-dev-20261010'
MODELS = ('score_only','spatial_only','score_spatial')
POLICIES = ('filter','replace')
FRACTIONS = (.25,.50,.75)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    manifest = read(SOURCE/'eval_manifest.json')
    data = {}
    for spec in manifest['datasets']:
        if spec['branch'] != 'ideal':
            continue
        selected = dict(spec,scores=[j for j in spec['scores'] if j['arm']=='ordinary'])
        ds = E.load_dataset(selected,SOURCE,['ordinary'],list(G.SEEDS))
        data[spec['split']] = ds
    return data


def split_cal(rows):
    """One authoring background instance per family fits, the other calibrates."""
    ids = {}
    for row in rows:
        ids.setdefault(row['background_family'],set()).add(row['background_id'])
    if any(len(values)!=2 for values in ids.values()):
        raise ValueError('Expected two calibration instances per background family')
    fit_ids = {sorted(values)[0] for values in ids.values()}
    mask = np.array([r['background_id'] in fit_ids for r in rows])
    return mask,dict(fit_background_ids=sorted(fit_ids),
        calibrate_background_ids=sorted({r['background_id'] for r in rows}-fit_ids))


def fit_linear(x, category, strong, mask):
    """Regularized logistic score, equal eligible-weight per scene/K/query."""
    eligible = ~strong[mask]
    counts = eligible.sum(-2,keepdims=True)
    weights = np.broadcast_to(np.divide(1.,counts,out=np.zeros_like(counts,dtype=float),where=counts>0),eligible.shape)
    y = np.broadcast_to((category[mask]=='contact')[:,None,None,:],eligible.shape)[eligible].astype(float)
    xx, ww = x[mask][eligible],weights[eligible]
    if not len(xx) or len(np.unique(y))!=2:
        raise ValueError('Two fit classes required')
    ww /= ww.sum()
    mean = (xx*ww[:,None]).sum(0)
    scale = np.sqrt(((xx-mean)**2*ww[:,None]).sum(0))
    scale = np.where(scale>1e-8,scale,1.)
    xx = (xx-mean)/scale
    def objective(beta):
        z = beta[0]+xx@beta[1:]
        loss = (ww*(np.logaddexp(0.,z)-y*z)).sum()+.05*np.dot(beta[1:],beta[1:])
        r = ww*(expit(z)-y)
        grad = np.r_[r.sum(),xx.T@r+.1*beta[1:]]
        return loss,grad
    result = minimize(objective,np.zeros(xx.shape[-1]+1),method='L-BFGS-B',jac=True,
        options=dict(maxiter=100,ftol=1e-10,gtol=1e-6))
    if not result.success:
        raise ValueError(('Linear solver did not converge',result.message))
    return dict(beta=result.x.tolist(),mean=mean.tolist(),scale=scale.tolist(),iterations=int(result.nit),
        objective=float(result.fun),fit_rows=len(xx),fit_contact_rows=int(y.sum()),
        role='Cal-fit only; logistic ranking score, not calibrated physical risk probability')


def predict(x, model):
    beta = np.array(model['beta'])
    return beta[0]+((x-np.array(model['mean']))/np.array(model['scale']))@beta[1:]


def at_most(values, target):
    """Whole score ties; smallest cutoff whose finite flagged count<=target."""
    finite = np.sort(np.asarray(values).reshape(-1))
    finite = finite[np.isfinite(finite)]
    if not len(finite) or target >= len(finite):
        return -np.inf
    if target <= 0:
        return float(np.nextafter(finite[-1],np.inf))
    return float(np.nextafter(finite[len(finite)-target-1],np.inf))


def calibrate(score, eligible, category, mask, original_light, fraction):
    category = category[mask]
    eligible, score, original_light = eligible[mask],score[mask],original_light[mask]
    passed = (category=='pass').any(1)&~(category=='contact').any(1)
    clear = (category=='clear').all(1)
    baseline_pass = int(original_light[passed].any((-1,-2)).sum())
    baseline_clear = int(original_light[clear].any(-1).sum())
    target_pass, target_clear = int(np.floor(fraction*baseline_pass)),int(np.floor(fraction*baseline_clear))
    candidate = np.where(eligible,score,-np.inf)
    p = candidate[passed].max((-1,-2))
    c = candidate[clear].max(-1)
    theta = max(at_most(p,target_pass),at_most(c,target_clear))
    flags = eligible&(score>=theta)
    actual_pass = int(flags[passed].any((-1,-2)).sum())
    actual_clear = int(flags[clear].any(-1).sum())
    assert actual_pass<=target_pass and actual_clear<=target_clear
    return dict(theta=float(theta),fraction=fraction,baseline_light_pass_clips=baseline_pass,
        baseline_light_clear_slots=baseline_clear,target_light_pass_clips=target_pass,
        target_light_clear_slots=target_clear,actual_light_pass_clips=actual_pass,
        actual_light_clear_slots=actual_clear,pass_residual=actual_pass-target_pass,clear_residual=actual_clear-target_clear)


def build_score_features(raw, smooth, theta):
    slope,residual = G.trend(raw)
    return np.stack((smooth-theta,slope,residual),-1)


def spatial_features(archive):
    """Adapter filled against extractor's explicit runtime feature contract."""
    names = archive['names'].tolist()
    values = archive['features']
    # All extractor scalars are runtime-only, no authoring IDs or truth labels.
    # Missing numerical values are zero-imputed, with a corresponding indicator.
    usable = [i for i,n in enumerate(names) if n in SPATIAL_NAMES]
    if not usable:
        raise ValueError('Missing expected public spatial descriptors')
    if len(usable)!=len(SPATIAL_NAMES):
        raise ValueError('Incomplete fixed spatial descriptor contract')
    chosen = values[...,usable].astype(float)
    def field(name):
        return values[...,names.index(name)].astype(float)
    derived = [field('inner_current_positive_log_total')/np.maximum(field('inner_current_membership_mass'),1e-8),
        field('inner_past8_positive_log_total')/np.maximum(field('inner_past8_membership_mass'),1e-8),
        field('inner_current_top3_positive_peak_log_sum')/np.maximum(field('inner_current_positive_log_total'),1e-8),
        field('inner_current_top1_positive_peak_log')-field('inner_past8_top1_positive_peak_log'),
        field('inner_current_samebin_peak_log')-field('ring_current_samebin_peak_log')]
    chosen = np.concatenate((chosen,np.stack(derived,-1)),-1)
    selected_names = [names[i] for i in usable]+['inner_current_positive_density','inner_past8_positive_density',
        'inner_current_top3_concentration','inner_current_minus_past8_peak','inner_minus_ring_samebin_peak']
    missing = ~np.isfinite(chosen)
    answer = np.concatenate((np.where(missing,0.,chosen),missing.astype(float)),-1)
    return answer,selected_names+[name+'_missing' for name in selected_names]


# Fixed before feature extraction or validation readout, updated only for exact
# extractor naming (not selection based on outcome).
SPATIAL_NAMES = ('inner_current_top1_positive_peak_log','inner_current_top3_positive_peak_log_sum',
    'inner_past8_top1_positive_peak_log','inner_past8_top3_positive_peak_log_sum',
    'expanded_current_top1_positive_peak_log','ring_current_top1_positive_peak_log',
    'inner_current_positive_participation','inner_current_top1_extent_membership',
    'inner_current_top1_forward_depth_m','inner_current_samebin_peak_log',
    'ring_current_samebin_peak_log','inner_current_samebin_peak_share')


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve declared or completed analysis')
    plan = dict(task='CNH_GRADED_CORRIDOR_DEV_20261010',base_commit='20e014b5',
        authorization='User 继续 previous graded-evidence outcome',lane='EXPLORE consumed simulation',
        goal='Use public expanded corridor/local peaks to improve timely light-alert versus pass/clear cost',
        budget=dict(evaluation_CPU_command_seconds=900,features_CPU_command_seconds=600,features_GPU_stage_seconds=180),
        adjustable_scope='Implementation repair, public evidence extraction, fixed low-dimensional score fit and focused verification',
        stop='Complete declared models/workpoints and task report or phase caps; no old-run reopening, protected480/hardware or new draws',
        strong='Exactly frozen oldfusion OR ordinary>=oldORaddition, original thresholds unchanged',
        models=list(MODELS),policies=list(POLICIES),fractions=list(FRACTIONS),
        input_features='Score-only: ordinary single margin, past5 raw slope and detrendedRMS. Spatial: fixedpublic extent inner/expanded/ring sparse peaks and support; no truth input.',
        fit='Cal: firstsorted background_id per family fits, others choose cutoffs. Shared H/B logistic score per seed; equal nonstrong eligible weight per sceneKquery, L2=.1, max100 L-BFGS-B, no class resampling.',
        policies_meaning='Filter only old additional single-light slots; replace may add local-evidence light below single, and may lose old light. Both keep everystrong slot.',
        calibration='For each fit/model/policy/fraction, threshold minimizes above complete-score ties subject to light-passclip <= floor(fraction*oldlight-passclips) and lightjointclear-slot <= floor(fraction*oldlight-clear slots) on calibration instance half. Report residuals; validation not adjusted.',
        decision_check='Baseline has substantial lightpass cost and contact gains. Report full rescues/losses vsstrongOR and priorlight, and earliest time; no zero-loss gate or allseed-win rule. Better task/cost tradeoff, rather than score separability alone, motivates retention.',
        evidence='Cal reused for scalar fitting and thresholding with instance separation; oldmodel thresholds still used fulloldcal. Validation is previouslyconsumed Development, not independent confirmatory data.',
        backend='TASK_NOT_GPU_SUITABLE: small scalar feature regression/metadata statistics; extraction backend separately benchmarked',
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE/'eval_manifest.json',PARENT/'thresholds.json')},
        spatial_primitives=list(SPATIAL_NAMES),spatial_derived=['current/past8 positive density','current top3 concentration',
            'current-minus-past8 inner top1','samebin inner-minus-ring peak'],
        source_sha256=sha(Path(__file__)))
    save(OUT/'PLAN.json',plan)
    try:
        data = load()
        fit_mask,partitions = split_cal(data['cal']['rows'])
        save(OUT/'cal_partition.json',partitions)
        thresholds = read(PARENT/'thresholds.json')
        spatial = {}
        for split,ds in data.items():
            with np.load(OUT/'features'/f'{split}_features.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['scene_ids'],ds['scene_ids'])
                spatial[split],spatial_names = spatial_features(archive)
        save(OUT/'feature_contract.json',dict(score_names=['ordinary_single_margin','raw_past5_slope','raw_past5_detrended_RMS'],
            spatial_names=spatial_names,missing='Zero imputation plus explicit indicator; no missing->free rule'))
        models,calibrations,metrics,summary,ledger = {},{},{},[],[]
        saved = {split:[] for split in data}
        for si,seed in enumerate(G.SEEDS):
            if time.monotonic()-began>900:
                raise TimeoutError('Evaluation CPU command900s limit')
            th = thresholds[str(seed)]
            tensor = {}
            references = {}
            for split,ds in data.items():
                ordinary = ds['candidates'][0,si]
                raw = E.read_job(SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                base = build_score_features(raw,ordinary,th['single'])
                tensor[split] = dict(score_only=base,spatial_only=spatial[split],score_spatial=np.concatenate((base,spatial[split]),-1))
                strong = E.old_fusion(ds['m3'],ds['local'])|(ordinary>=th['addition'])
                light = ~strong&(ordinary>=th['single'])
                references[split] = dict(strong=strong,light=light,prior_any=strong|light)
            for model_name in MODELS:
                model = fit_linear(tensor['cal'][model_name],data['cal']['category'],references['cal']['strong'],fit_mask)
                models[f'{seed}/{model_name}'] = model
                scores = {split:predict(tensor[split][model_name],model) for split in data}
                for policy in POLICIES:
                    eligible = {split:(refs['light'] if policy=='filter' else ~refs['strong']) for split,refs in references.items()}
                    for fraction in FRACTIONS:
                        key = f'{seed}/{model_name}/{policy}/{fraction}'
                        record = calibrate(scores['cal'],eligible['cal'],data['cal']['category'],~fit_mask,
                                           references['cal']['light'],fraction)
                        calibrations[key] = record
                        for split,ds in data.items():
                            refs = references[split]
                            flags = eligible[split]&(scores[split]>=record['theta'])
                            grade = np.where(refs['strong'],2,np.where(flags,1,0)).astype(np.int8)
                            np.testing.assert_array_equal(grade==2,refs['strong'])
                            if policy=='filter':
                                assert np.all(~(grade>0)|refs['prior_any'])
                            comparison = dict(M3=ds['m3']>=E.M3_THETA,old_fusion=E.old_fusion(ds['m3'],ds['local']),
                                ordinary_OR=refs['strong'],ordinary_single=ds['candidates'][0,si]>=th['single'],prior_light=refs['prior_any'])
                            report = G.describe(grade,ds,comparison)
                            metrics[f'{split}/{key}'] = report
                            summary.append(dict(split=split,seed=seed,model=model_name,policy=policy,fraction=fraction,
                                HEAD=report['any']['counts'][0],BODY=report['any']['counts'][1],
                                HEAD_gain_prior=report['paired_any']['prior_light'][0]['rescue'],HEAD_loss_prior=report['paired_any']['prior_light'][0]['loss'],
                                BODY_gain_prior=report['paired_any']['prior_light'][1]['rescue'],BODY_loss_prior=report['paired_any']['prior_light'][1]['loss'],
                                HEAD_extra_strong=report['paired_any']['ordinary_OR'][0]['rescue'],BODY_extra_strong=report['paired_any']['ordinary_OR'][1]['rescue'],
                                clear_slots=report['any']['clear_slots'],clear_clips=report['any']['clear_clips'],pass_clips=report['any']['pass_clips'],
                                light_clear_slots=report['joint_costs']['clear']['light']['slots'],light_pass_slots=report['joint_costs']['pass']['light']['slots'],
                                light_pass_clips=report['joint_costs']['pass']['light']['clips']))
                            saved[split].append((key,grade))
                            if split=='validation':
                                before,after = G.first(refs['prior_any']),G.first(grade>0)
                                for n,row in enumerate(ds['rows']):
                                    for k in range(grade.shape[1]):
                                        for q,height in enumerate(E.HEIGHTS):
                                            ledger.append(dict(key=key,scene=int(ds['scene_ids'][n]),replica=k,height=height,
                                                category=ds['category'][n,q],family=row['shape_family'],rho=row['rho'],placement=row['placement'],
                                                background_family=row['background_family'],before_first=int(before[n,k,q]),after_first=int(after[n,k,q])))
        for split,records in saved.items():
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.asarray([r[0] for r in records]),grades=np.asarray([r[1] for r in records]))
        G.write_csv(OUT/'summary.csv',summary)
        G.write_csv(OUT/'event_ledger.csv',ledger)
        save(OUT/'models.json',models);save(OUT/'calibrations.json',calibrations);save(OUT/'metrics.json',metrics)
        save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),
            model_fits=len(models),event_rows=len(ledger),GPU_seconds=0,source_sha256=sha(Path(__file__)),
            invariants='Everystrong slot retained; filter subset; instance-separated cal fit/cutoff; all3seeds; finite coefficients and convergence'))
        print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),models=len(models))))
    except BaseException as error:
        save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began))
        raise


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
