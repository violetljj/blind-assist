"""Bound ideal projected separation without treating iteration caps as convergence."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import cnh_aligned_boundary_dev as B
from cnh_bar_projection_diagnostic_dev import OUT


def run():
    began=time.monotonic()
    result=json.loads((OUT/'projection_result.json').read_text())
    bounds=[]
    for numerical in result['numerics']:
        pair=json.loads((OUT/f"pair_f{numerical['end']}_{numerical['scene']}_{numerical['contrast']}.json").read_text())
        r=pair['metrics'];parts=numerical['components']
        # q is in the output row space even when LSMR reaches its iteration cap.
        # Cauchy-Schwarz gives (u.q)^2/||q||² <= ||Proj_row u||².
        lower=sum(c['lower_bound_d2'] for c in parts)
        # Input columns unseen by every voxel cannot enter that row space.
        upper=sum(c['covered_d2'] for c in parts)
        observation=sum(c['input_d2'] for c in parts)
        eps=max(1e-9,observation*1e-9)
        assert lower <= upper+eps <= observation+eps
        bounds.append(dict(scene=r['scene'],reference_scene=r['reference_scene'],height=r['height'],contrast=r['contrast'],
            end=r['end'],variant=r['variant'],placement=r['placement'],side=r['side'],
            observation_d=float(np.sqrt(observation)),joint_lower_d=float(np.sqrt(lower)),joint_upper_d=float(np.sqrt(upper)),
            retained_d2_lower=lower/observation,retained_d2_upper=upper/observation,
            interval_d2_fraction=(upper-lower)/observation,
            lost_d2_fraction_lower=1-upper/observation,lost_d2_fraction_upper=1-lower/observation,
            unavailable_input_d2=observation-upper,possible_additional_mixing_d2=upper-lower,
            lsmr_all_converged=r['converged'],estimate_joint_d=r['joint_d'],estimate_tot_d=r['tot_d']))
        for mode in ('positive','negative'):
            for label,key in (('input','input_d2'),('lower','lower_bound_d2'),('upper','covered_d2')):
                bounds[-1][f'{mode}_{label}_d2']=sum(c[mode][key] for c in parts)
        bounds[-1]['signed_cross_estimate_d2']=r['cross_joint_d2']
    fields=['observation_d','joint_lower_d','joint_upper_d','retained_d2_lower','retained_d2_upper',
            'interval_d2_fraction','lost_d2_fraction_lower','lost_d2_fraction_upper','unavailable_input_d2','possible_additional_mixing_d2']
    summaries=[]
    for end in (12,13):
        for height in ('HEAD','BODY'):
            for contrast in ('A_absent','B_pass'):
                rr=[r for r in bounds if r['end']==end and r['height']==height and r['contrast']==contrast]
                group=dict(end=end,height=height,contrast=contrast,n=len(rr),
                    **{k:dict(min=float(min(r[k] for r in rr)),median=float(np.median([r[k] for r in rr])),max=float(max(r[k] for r in rr))) for k in fields})
                group['signed_modes']={}
                for mode in ('positive','negative'):
                    values={label:sum(r[f'{mode}_{label}_d2'] for r in rr) for label in ('input','lower','upper')}
                    values['retained_lower_fraction']=values['lower']/values['input'] if values['input'] else None
                    values['retained_upper_fraction']=values['upper']/values['input'] if values['input'] else None
                    values['semantics']='Sum over geometries for a single signed mean mode, shared full covariance; not additive contributions to total after projection.'
                    group['signed_modes'][mode]=values
                summaries.append(group)
    with (OUT/'projection_bounds.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(bounds[0]));w.writeheader();w.writerows(bounds)
    B.save(OUT/'projection_bounds.json',dict(status='COMPLETE',pairs=len(bounds),summaries=summaries,
        maximum_interval_d2_fraction=max(r['interval_d2_fraction'] for r in bounds),
        minimum_retained_d2_lower=min(r['retained_d2_lower'] for r in bounds),
        numerical_contract='Lower: Cauchy-Schwarz for an actual row-space vector. Upper: standardized input norm on nonzero projection columns. Valid without declaring LSMR convergence; floating-point checks still apply. Medians are geometry summaries, not independent-trial inference.',
        projection_result_sha256=B.sha(OUT/'projection_result.json'),source_sha256=B.sha(__file__),seconds=time.monotonic()-began))
    (OUT/'projection_bounds_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(pairs=len(bounds),maximum_interval_d2_fraction=max(r['interval_d2_fraction'] for r in bounds),
        minimum_retained_d2_lower=min(r['retained_d2_lower'] for r in bounds)),ensure_ascii=False))


if __name__=='__main__':run()
