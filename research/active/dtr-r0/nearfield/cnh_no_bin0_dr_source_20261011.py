"""Fresh authoring-only no-bin0 DR hold, tradeoff recipe and exact identity audit."""
from pathlib import Path
import copy
import time
import numpy as np
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
import cnh_s_ensemble_tradeoff_source_20261010 as OLD
from cnh_cost_v2_source_20261010 import inventories, categories

K = 2
FAMILIES = ('no_bin0_dr_alternating_double_bridges','no_bin0_dr_stepped_twolevel_panels',
            'no_bin0_dr_skew_separated_tripod_posts','no_bin0_dr_offset_asymmetric_nested_backs')
DELTA = .00191
PARAMETERS = {k:tuple(float(v)+DELTA for v in value) if isinstance(value,tuple) else float(value)+DELTA
              for k,value in OLD.PARAMETERS['hold'].items()}

def sampling_seed(row, replica):
    words=np.frombuffer(bytes.fromhex(row['physical_key']),dtype='<u4').tolist()
    return int(np.random.SeedSequence([20261011317,*words,int(replica)]).generate_state(1)[0])

def backgrounds():
    rows=[]
    for fi,family in enumerate(FAMILIES):
        for instance in range(2):
            rho=.32391+.02791*instance+.00991*fi
            floor_y=1.62391+.00291*fi
            floor=D.box([-8,floor_y,-8],[8,floor_y+.18791,9],.35391+.01391*instance)
            wall_z=5.23391+.14791*fi+.19791*instance
            wall=D.box([-8,-3,wall_z],[8,floor_y,wall_z+.21391],.44391+.01191*fi)
            if fi==0:
                near=[D.box([.77391,-.63391,1.17391],[.86391,.77391,1.31391],rho),
                      D.box([1.26391,-.63391,1.17391],[1.35391,.77391,1.31391],rho),
                      D.box([.77391,-.63391,1.17391],[1.35391,-.48391,1.31391],rho),
                      D.box([-1.32391,.49391,2.23391],[-1.23391,1.37391,2.37391],rho),
                      D.box([-.86391,.49391,2.23391],[-.77391,1.37391,2.37391],rho),
                      D.box([-1.32391,.49391,2.23391],[-.77391,.64391,2.37391],rho)]
            elif fi==1:
                near=[D.box([.76391,-.56391,1.24391],[1.24391,-.42391,1.79391],rho),
                      D.box([.96391,-.18391,1.55391],[1.40391,-.04391,2.20391],rho),
                      D.box([1.10391,.23391,1.97391],[1.54391,.37391,2.58391],rho),
                      D.box([1.18391,-.42391,1.68391],[1.24391,1.34391,1.79391],rho),
                      D.box([-1.31391,.76391,2.17391],[-.79391,.91391,2.75391],rho)]
            elif fi==2:
                near=[D.box([.76391,-.61391,1.25391],[.84391,1.35391,1.36391],rho),
                      D.box([1.18391,-.53391,1.64391],[1.27391,1.27391,1.76391],rho),
                      D.box([.92391,-.46391,2.19391],[1.01391,1.39391,2.31391],rho),
                      D.box([-1.19391,.87391,2.77391],[-.76391,1.05391,3.08391],rho)]
            else:
                near=[D.box([.75391,-.63391,1.23391],[.86391,.92391,1.38391],rho),
                      D.box([.86391,-.63391,1.38391],[1.46391,-.46391,2.68391],rho),
                      D.box([1.30391,-.46391,2.50391],[1.46391,1.23391,2.68391],rho),
                      D.box([.86391,1.05391,1.97391],[1.30391,1.23391,2.50391],rho),
                      D.box([-1.44391,.07391,1.92391],[-.79391,.21391,2.57391],rho),
                      D.box([-1.44391,.21391,2.43391],[-1.30391,1.27391,2.57391],rho),
                      D.box([-1.30391,1.13391,2.57391],[-.76391,1.27391,3.12391],rho)]
            rows.append(dict(background_id=len(rows),background_family=family,split='hold',instance=instance,
                boxes=[floor,wall,*D.mirror_boxes(near,1 if instance==0 else -1)]))
    return rows

def scenes():
    original_parameters, original_backgrounds=OLD.PARAMETERS,OLD.backgrounds
    bg=backgrounds()
    try:
        OLD.PARAMETERS={'hold':PARAMETERS}
        OLD.backgrounds=lambda split:copy.deepcopy(bg)
        rows=OLD.scenes()['hold']
    finally:
        OLD.PARAMETERS,OLD.backgrounds=original_parameters,original_backgrounds
    for row in rows:
        row['scene_uid']=f"no-bin0-dr-20261011/hold/{row['scene_id']}"
    return rows

def prepare(out):
    began=time.monotonic();out=Path(out)
    if not (out/'PLAN.json').exists() or not (out/'sealed_calibration.json').exists():
        raise ValueError('Require task PLAN and negative calibration seal before fresh hold authoring')
    rows=scenes();inventory,oldkeys,oldfamilies,oldbackgrounds=inventories(out)
    keys={r['physical_key'] for r in rows};families={r['background_family'] for r in rows}
    bgs={D.physical_key(r['background_boxes']) for r in rows}
    assert len(rows)==len(keys)==1536
    assert not keys&oldkeys and not families&oldfamilies and not bgs&oldbackgrounds
    olddimensions,oldrhos=set(),set()
    for item in inventory['files']:
        OLD._target_values(C.read(C.ROOT/item['path']),olddimensions,oldrhos)
    dimensions,rhos=set(),set();OLD._target_values(rows,dimensions,rhos)
    assert not dimensions&olddimensions and not rhos&oldrhos
    _,_,_,B=D.frozen_imports();sensor,query=B.poses(-10.)
    category=categories(rows,sensor)
    support,visible,positive=OLD.target_support(rows,sensor)
    folder=out/'data/hold';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'geometry.npz',category=category,sensor=sensor,public_query=query,
        target_support=support,target_visible_now=visible,label_positive=positive,
        scene_ids=np.arange(len(rows)),scene_uids=np.array([r['scene_uid'] for r in rows]))
    C.save(out/'scene_rows.json',{'hold':rows})
    C.save(out/'background_rows.json',{'hold':backgrounds()})
    C.save(out/'inventory.json',inventory)
    result=dict(seconds=time.monotonic()-began,physical_scenes=len(rows),replicas=K,
        contact_events_per_height=[K*sum(r['placement']=='contact' and r['group']==q for r in rows) for q in range(2)],
        physical_overlap=0,background_overlap=0,family_overlap=0,target_dimensions_overlap=0,target_rho_overlap=0,
        families=sorted(families),parameters=PARAMETERS,inventory_file_count=inventory['file_count'],
        inventory_keys=inventory['unique_physical_keys'],calibration_sha256=C.sha(out/'sealed_calibration.json'),
        geometry_sha256=C.sha(folder/'geometry.npz'),source_sha256=C.sha(__file__))
    C.save(out/'fresh_hold_source_receipt.json',result)
    return result
