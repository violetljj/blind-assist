"""Authorized second readout pilot: immutable plan, paths, lifecycle, no deadline."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time

from cnh_temporal_readout import read, save, sha

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'
OUT = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
RUN = 'CNH_READOUT_PILOT2_20261005'
TRAIN_UNITS = read(OLD/'PLAN.json')['train_units']
EVAL_UNITS = [u for u in range(231001,231073) if u % 3 in (0,1)]
SMOKE_UNIT = 231999
FRAMES = list(range(3,16))
SEEDS = [0,1,2]


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, name in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache'),('MPLCONFIGDIR','mpl-cache')]:
        path = OUT/name; path.mkdir(exist_ok=True); os.environ[key] = str(path)


def load_plan():
    p = read(OUT/'PLAN.json')
    if p['run'] != RUN or p['eval_units'] != EVAL_UNITS or p['seeds'] != SEEDS:
        raise ValueError('Wrong pilot2 plan')
    if p['budget_seconds'] is not None:
        raise ValueError('Human explicitly removed the hard time limit')
    return p


def plan_sha(): return sha(OUT/'PLAN.json')


def start_budget():
    load_plan()
    path = OUT/'budget.json'
    if path.exists(): return read(path)
    value = dict(run=RUN, plan_sha256=plan_sha(), started_utc=datetime.now(timezone.utc).isoformat(),
                 compute_started_unix=time.time(), deadline_unix=None, budget_seconds=None,
                 authorization='Human: 不设置硬性时限，但是要尽量优化效率和时间',
                 scope='wall-clock from first pilot2 scientific computation; stop after scoped outcome, no automatic new experiments')
    save(path, value); return value


def check_budget(reserve_seconds=0):
    b = read(OUT/'budget.json')
    if b['plan_sha256'] != plan_sha() or b['deadline_unix'] is not None:
        raise ValueError('Lifecycle identity changed')
    return float('inf')


def input_folder(split):
    if split in ('train','calibration','evaluation'): return OLD/'inputs'/split
    return OUT/'inputs'/split


def freeze():
    setup()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Plan exists; resume without overwriting')
    candidate=set(EVAL_UNITS+[SMOKE_UNIT]); collisions=[]
    for folder in OUT.parent.glob('cnh*'):
        if folder == OUT or not folder.is_dir(): continue
        for name in ('PLAN.json','plan.json','request.json'):
            path=folder/name
            if not path.exists(): continue
            content=path.read_text(encoding='utf8')
            ids={int(x) for x in re.findall(r'(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_.])',content)}
            ids.update(int(x) for x in re.findall(r'unit(\d+)\.(?:npz|npy|json)',content))
            if ids & candidate: collisions.append(dict(path=str(path),units=sorted(ids&candidate)))
    if collisions: raise ValueError(collisions)
    import cnh_readout_pilot2_model as M
    p = read(OLD/'PLAN.json')
    for key in ('implementation_sha256_at_freeze','branch','descriptive','budget_reduction'):
        p.pop(key,None)
    p.update(run=RUN, created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Human approved revised pilot2 and removed hard time limit; optimize efficiency and time',
        budget_seconds=None, epochs=8,batch=64,lr=.0003,wd=.0001,
        eval_units=EVAL_UNITS, smoke_unit=SMOKE_UNIT, bootstrap=dict(n=1000,seed=2026100503,unit='whole paired scenes'),
        original_pilot_commit='3c5273a6', prior_plan_sha256=sha(OLD/'PLAN.json'),
        old_root=str(OLD), train_source_root=str(OLD),eval_source_root=str(OUT),
        render_workers=4, teachers_temperature=1.,
        arms=dict(V_retest='Frozen original V seed0/1/2; no retraining or natural recalibration',
                  VD='CVR fine tune matching M3 seed; original V recipe; masked target query label=(hard+sigmoid(D_LLR))/2; original weights unchanged',
                  T2='Explicit finite-cell/query overlap geometry with frame-zone/bin evidence, weighted query pooling, retained background context; random init'),
        T2_model_config=M.MODEL_CONFIG,
        T2_recipe=dict(epochs=20,batch=64,lr=.002,wd=.0001,scheduler='CosineAnnealingLR20; fixed final epoch'),
        T2_fit_gate=dict(loss='Final-checkpoint hard masked weighted BCE <=2 times original final V3 mean on identical train rows',
                         auc='pooled unweighted valid training query-row AUC >=.85; per-query descriptive',
                         fail='T2_UNFIT: no seed1/2, no three-seed candidate inference'),
        geometry_acceptance='Independent SUB3 finite-cell pulse field vs BatchedProjector abs<=1e-6; same discrete centroid <.025m; mass/query overlap/padding, left-right axes checked',
        VD_loss='Only valid displacement target query receives (hard+sigmoid(LLR))/2. Pass mask unchanged; other height and natural hard only. No additive doubling, no temperature tuning',
        reference='Known size/background/current anchor, noisy relative poses, class-balanced five-position exact Skellam8; teacher and fresh D share implementation; lambda0 at least2 scenes <1e-8',
        branch=dict(V_REPLICATED='V_retest-M3>=.015 AND pairedCIlo>0; otherwise V_NOT_REPLICATED',
                    CANDIDATE='arm-M3>=.045 AND pairedCIlo>0 AND clear<=M3+.1/min AND deep>=M3-.02; no automatic replacement',
                    TEACHER_EFFECTIVE='VD-V_retest>=.02 AND pairedCIlo>0',
                    GEOMETRY_RECIPE_EFFECTIVE='T2-V_retest>=.02 AND pairedCIlo>0; initialization, epochs, capacity differ',
                    OTHERWISE='Describe only; no subgroup or single-seed rescue'),
        efficiency='Reuse old voxels and natural scores; shared teacher templates and geometry cache; concurrent CPU rendering with independent GPU stages only when capacity allows; no hard deadline',
        interpretation='New random units within consumed generator family, not protected confirmation. Teacher privilege enters targets only. T2_UNFIT is fit failure, not representation impossibility. Simulated noise/poses only; no hardware claim',
        draft_sha256=sha(OUT/'PLAN_DRAFT.md'))
    save(OUT/'PLAN.json',p)
    save(OUT/'progress.json',dict(status='PLAN_FROZEN',stage='implementation/preflight',plan_sha256=plan_sha(),last_activity_utc=datetime.now(timezone.utc).isoformat()))
    print('PLAN FROZEN',plan_sha(),len(EVAL_UNITS),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','start-budget','status'],required=True)
    stage=parser.parse_args().stage
    if stage=='plan': freeze()
    elif stage=='start-budget': print(start_budget())
    else: print(dict(plan_sha256=plan_sha(),elapsed_seconds=time.time()-read(OUT/'budget.json')['compute_started_unix']))
