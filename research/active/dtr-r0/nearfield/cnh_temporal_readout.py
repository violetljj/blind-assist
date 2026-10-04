"""Frozen plan and compute budget for the V/T Development pilot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'
RUN='CNH_TEMPORAL_READOUT_PILOT_20261004'
TRAIN_UNITS=[u for u in range(220001,220721) if u%3 in (0,1)]
EVAL_UNITS=[u for u in range(221001,221073) if u%3 in (0,1)]
SMOKE_UNIT=229999
FRAMES=list(range(3,16))
SEEDS=[0,1,2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def load_plan():
    p=read(OUT/'PLAN.json')
    if p['run']!=RUN or p['seeds']!=SEEDS or p['eval_units']!=EVAL_UNITS:
        raise ValueError('Wrong frozen pilot plan')
    amendment=OUT/'PLAN_AMENDMENT.json'
    if amendment.exists():
        a=read(amendment)
        if a['original_plan_sha256']!=sha(OUT/'PLAN.json') or a['reason']!='budget_forecast':
            raise ValueError('Unrecognized plan amendment')
        changes=a['changes']
        if set(changes)-{'epochs','train_units'} or changes.get('epochs',p['epochs'])>p['epochs']:
            raise ValueError('Only prospectively shrinking training count/epochs is allowed')
        if not set(changes.get('train_units',p['train_units'])).issubset(p['train_units']):
            raise ValueError('Amendment introduced new units')
        p.update(changes)
    return p


def plan_sha():return sha(OUT/'PLAN.json')


def start_budget():
    path=OUT/'budget.json'
    if path.exists():return read(path)
    p=load_plan();now=time.time()
    b=dict(run=RUN,plan_sha256=plan_sha(),started_utc=datetime.now(timezone.utc).isoformat(),
           compute_started_unix=now,deadline_unix=now+p['budget_seconds'],budget_seconds=p['budget_seconds'],
           scope='wall-clock from first generated smoke through final measured evaluation; implementation/report/delivery separately reported')
    save(path,b);return b


def check_budget(reserve_seconds=0):
    path=OUT/'budget.json'
    if not path.exists():raise RuntimeError('Start the authorized compute budget before generation/training')
    b=read(path)
    if time.time()+reserve_seconds>=b['deadline_unix']:
        raise TimeoutError('Pilot compute budget reached; retain partial evidence, do not expand')
    return b['deadline_unix']-time.time()


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    for key,folder in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache'),('MPLCONFIGDIR','mpl-cache')]:
        target=OUT/folder;target.mkdir(exist_ok=True);os.environ[key]=str(target)


def freeze():
    setup()
    if (OUT/'PLAN.json').exists():raise FileExistsError('PLAN already frozen; resume without overwriting')
    candidate=set(TRAIN_UNITS+EVAL_UNITS+[SMOKE_UNIT]);collisions=[]
    for folder in OUT.parent.glob('cnh*'):
        if folder==OUT or not folder.is_dir():continue
        for name in ('PLAN.json','plan.json','request.json'):
            path=folder/name
            if path.exists():
                content=path.read_text(encoding='utf-8')
                ids={int(x) for x in re.findall(r'(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_.])',content)}
                ids.update(int(x) for x in re.findall(r'unit(\d+)\.(?:npz|npy|json)',content))
                if ids&candidate:collisions.append(dict(path=str(path),units=sorted(ids&candidate)))
    if collisions:raise ValueError(collisions)
    # These are exact candidate unit filenames; no dataset/observation bytes are scanned.
    import cnh_margin_confirm as MC
    import cnh_surface_distribution_train as ST
    import cnh_cvr_pilot as CP
    import cnh_sequence_observed_evaluate as OE
    import cnh_sequence_observed_geometry as OG
    import cnh_three_level_sequence as SE
    import cnh_displacement_ceiling_render as R
    import cnh_proposal_attribution_scenes as S
    from dataclasses import asdict
    input_files=list(MC.model_paths('M3'))+[MC.SS.WORK/'cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy',
                    OE.OUT/'geometry.npz',OE.OUT/'geometry_receipt.json',OE.OUT/'result.json',MC.OUT/'scene_manifest.json']
    for split,ids in [('train',range(93000,93096)),('calib',range(95000,95048)),('evaluation',range(96000,96096))]:
        base=MC.NR.OUT if split=='train' else MC.OUT
        input_files += [base/f'features/{split}/unit{u}.npz' for u in ids]
    params,_=S.nominal_parameters()
    scientific=dict(run=RUN,created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Human confirmed <=3h on2026-10-04 and requested efficiency optimization',
        role='EXPLORE consumed synthetic Development training pilot; fresh random evaluation units within same generator family, not protected confirmation',
        train_units=TRAIN_UNITS,eval_units=EVAL_UNITS,smoke_unit=SMOKE_UNIT,
        natural_train_units=list(range(93000,93096)),natural_calibration_units=list(range(95000,95048)),natural_evaluation_units=list(range(96000,96096)),
        seeds=SEEDS,epochs=8,batch=64,lr=.0003,wd=.0001,optimizer='AdamW',scheduler='CosineAnnealingLR T_max8; final epoch, no evaluation-based selection',
        frames=FRAMES,history=8,train_K=2,eval_K=4,train_delta_cm=[-25,10],eval_delta_cm=[1,2,5,-5,-10,-15,-20],
        train_scene_count=480,eval_scene_count=48,smoke_scene_count=1,scene_balance='mode0/1 xHEAD/BODY xnone/panel, same generator prior as displacement batch',
        noise_seed_prefix=2026100407,photon_seed_prefix=2026100406,scene_seed_prefix=2026100405,delta_sampling='firstdraw from independent scene RNG, fixed while nuisance resampled',
        noise_law='lambda1 original noisy_poses law, original sensor params/gain1/shot-noise law; no retuning',sensor_params=asdict(params),
        target_background_geometry='Draw train delta once uniformly then redraw nuisance scene if physical boxes interpenetrate; retain delta, record attempts. Evaluation uses original gap law and7variants.',
        labels='Perframe/perquery全物体surface_category: contact positive, onlypass0-10cm mask, clear negative. Exact outside10 is pass. Rebuild natural training labels, never inherit old0.33m enlarged labels.',
        input_contract='histories half z1, ambient nominal observation/calibration, estimated relative transforms and public query geometry only; truth/background/templates separated from network inputs',
        ambient_recovery='Old z1 has no storedambient; frozen law is ambient_counts*gain constant after4quarter aggregate; restore nominal calibration, not background truth or zero fill',
        current_query_geometry='Same nominal mount/scan geometry as frozen M3. No scene/global true position feature. Verify nominal current-sensor-to-query against existing metadata.',
        loss='masked BCE, both positive andclear examples far1.6<=front<2.6 weighted2, others1; no class balancing; domain rawweight=mask*far, each domain global sum normalized toN/2; batch loss=sum(BCE*weights)/batchN',
        far_weight=2,far_range_m=[1.6,2.6],domain_loss_mass={'natural':.5,'fresh':.5},
        arms=dict(V='Original CVR46,097 parameters, load frozenM3 seed0/1/2 respectively and finetune allweights',
                  T='66->64->64 sharedzone MLP; box6+queryemb8 zone attention +mean/max ->206->64; GRU64 time order;78->32->1. Random init. Approx51,794; <=5xM3.'),
        parameter_limit=230485,padding='Valid length4..8, left pad excluded from attention/mean/max/GRU; no zero-token hidden updates',
        scoring='Mean3seed rawlogits then same5logit causal[1,2,4,8,16] smooth. M3 original5seed mean andsame smoother; perseed all listed',
        primary='48equal scenes, within-scene targetposition AUC inside1/2 vsoutside10/15/20, K4,frames3..15,1.2<=front<2.1m; distinct from all-object clearfalse labels',
        bootstrap=dict(n=1000,seed=2026100403,unit='wholepaired scenes',interval='fixedmodels andcalibrated thresholds; no retraining uncertainty'),
        reference='Freshsame-batch exactSkellam8 D, entiretemplates at sensor[f]@inv(noisy[k,f])@noisy[k,past], five candidate positions1/2/-10/-15/-20, balancedclasses; knownbackground/target/currentanchor privileges',
        gap_ratio='(candidate-M3)/(D-M3), same macro difference and commonbootstrap. Do not clip negative/>1. denominator<=0 orabs<.01 orbootstrap crosses0 ->notstable ratio interval',
        natural_calibration='95000..95047 all-object full13 clear max score; mergedHEAD/BODY singlethreshold <=1firststop/proxyminute,whole ties,>=; V/T once; M3 retained original equivalentthreshold',
        threshold_frozen_M3=.8557642486787612,natural_M3_reference=dict(shallow_timely=26,shallow_n=31,deep_timely=162,deep_n=164,clear_stops=205,clear_proxy_minutes=194.35),
        guard=dict(clear_rate_upper=205/194.35+.1,deep_timely_rate_lower=162/164-.02,other_height='among all-windowclear, query!=targetgroup; descriptive'),
        sequence='Observed0.9m firstcrossing / censoring retained; same natural-calibrated thresholds applied to fullnewdisplacement sequences, report shallowtimely andoutside15/20 firststops',
        branch=dict(CANDIDATE='Perarm ensemble-M3 AUC>=.03 ANDpairedCIlo>0 ANDnaturalclear<=M3+.1/min ANDdeep>=M3-.02; suggestfreshconfirmation, do not replaceM3',
                    NOT_LEARNED='Bothensemble deltas<.015 -> stopthisarchitecture recipe andmove nextquestion tobackgroundprivilege; no universal unlearnability claim',
                    INTERMEDIATE='Otherwise report fordiscussion; noautomatic follow-up'),
        descriptive=dict(T_minus_V='>=.02 supports thisrepresentation training recipe, not purecausal representation effect',V_minus_M3='>=.03 supports originalstructure plusdata/target recipe'),
        efficiency='Reuse naturalVvoxels, share z/estimatedpose inputs acrossarms andseeds; lazy memmap, GPU token construction, batchinference; no repeatedprojection perepoch',
        precision='float32 primary training/inference, noTF32; mixedprecision only afterengineering smoke comparesfinite behavior andruntime disclosed, without new scientific arm',
        budget_seconds=10800,render_workers=4,budget_reduction='Prospective PLAN_AMENDMENT before affectedcomputation: shrinktrain count orcommonepoch only, never<3seeds/48eval/criteria. Resume sealedinputs, no overrun expansion.',
        smoke='Oneindependent scene oneepoch beforefullgeneration/training; shape,padding,parametercap,finiteloss/grad, inputtruth separation andVparity',
        interpretation='U andD separate privileges interventions, notjoint. T random vsV pretrained; NOT_LEARNED stops thisrecipe, not alltemporalmethods. Assumednoise/pose simulation only, nohardware or safety claim',
        old_input_sha256={str(path):sha(path) for path in input_files},
        dependency_sha256={str(Path(m.__file__)):sha(m.__file__) for m in [MC,ST,CP,OE,OG,SE,R,S]},
        implementation_sha256_at_freeze={str(path):sha(path) for path in Path(__file__).parent.glob('cnh_temporal_readout*.py')},
        original_diagnostic_commit='f8c5d0c1',old_result_preservation=True)
    save(OUT/'PLAN.json',scientific)
    print('PLAN FROZEN',plan_sha(),len(TRAIN_UNITS),len(EVAL_UNITS),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['plan','start-budget','status'])
    stage=parser.parse_args().stage
    if stage=='plan':freeze()
    elif stage=='start-budget':print(start_budget())
    else:print(dict(plan_sha256=plan_sha(),remaining_seconds=check_budget()))
