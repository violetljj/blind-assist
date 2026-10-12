"""Retained train-only shallow/contact versus full-clear matched ranking pairs.

Pairs are supervised examples from different sampled scenes, not controlled
physical interventions. Neither this module nor its caches is an inference input.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np

import cnh_query_mass_train as QM

OUT = QM.RT.ST.SS.WORK/'cnh-boundary-contrast-20261003'
RUN = 'CNH_BOUNDARY_CONTRAST_20261003'
FACTOR = QM.RT.ST.SS.WORK/'cnh-surface-factorization-probe-20261002/units'
UNITS = list(range(93000, 93096))
FRAMES = np.arange(3,16)
MIN_RANGE, MAX_RANGE, MAX_DELTA = .9, 2.1, .25
sha, read, save = QM.sha, QM.read, QM.save


def plan_sha():
    plan = read(OUT/'PLAN.json')
    expected = dict(run=RUN, training_units=UNITS, decision_frames=FRAMES.tolist(),
                    pair_min_range=MIN_RANGE, pair_max_range=MAX_RANGE, pair_max_delta=MAX_DELTA)
    if any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Boundary pair PLAN differs from recipe')
    if RUN not in (Path(__file__).parents[1]/'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('Missing boundary contrast run record')
    return sha(OUT/'PLAN.json')


def row_identity(rows):
    return hashlib.sha256(b''.join(rows[k].tobytes() for k in ('unit','config','frame'))).hexdigest()


def match_pairs(rows, positives, negatives):
    """Stable negative row ordering, reusable nearest match within fixed strata."""
    def key(row, query):
        return (int(rows['unit'][row])%3, int(rows['frame'][row]), int(query),
                int(rows['config'][row])>=10)
    buckets = defaultdict(list)
    for row,query,front in sorted(negatives,key=lambda x:(x[0],x[1])):
        buckets[key(row,query)].append((int(row),int(query),float(front)))
    pairs, missing, differences = [], [], []
    for row,query,front in sorted(positives,key=lambda x:(x[0],x[1])):
        candidates = buckets[key(row,query)]
        if not candidates:
            missing.append((int(row),int(query)))
            continue
        # min preserves the stable row order when distances tie.
        negative = min(candidates,key=lambda x:abs(front-x[2]))
        difference = abs(front-negative[2])
        if difference>MAX_DELTA:
            missing.append((int(row),int(query)))
            continue
        pairs.append((int(row),negative[0],int(query)))
        differences.append(float(difference))
    return np.asarray(pairs,dtype=np.int64).reshape(-1,3), missing, differences


def prepare():
    digest = plan_sha()
    rows, labels = QM.RT.ST.training_rows()
    identity = row_identity(rows)
    if len(rows['unit'])!=27456 or labels.shape!=(27456,2):
        raise ValueError('M3 training row axes differ')
    keys = list(zip(rows['unit'].tolist(),rows['config'].tolist(),rows['frame'].tolist()))
    if keys!=[(u,c,int(f)) for u in UNITS for c in range(22) for f in FRAMES]:
        raise ValueError('Training row order differs from canonical unit/config/frame')
    sources = {str(Path(p).resolve()):sha(p) for p in
               (__file__,QM.__file__,QM.RT.ST.__file__,QM.RT.ST.NT.__file__)}
    paths = {str(QM.RT.ML.OUT/'train_labels.npz'):sha(QM.RT.ML.OUT/'train_labels.npz')}
    positives, negatives = [], []
    categories, eligible_categories = Counter(), Counter()
    full_clear_counts = Counter()
    factor_ids = set()
    n_positive_other_target_query = 0
    for unit in UNITS:
        fp = FACTOR/f'unit{unit}.npz'
        receipt = read(fp.with_suffix('.json'))
        if receipt['status']!='COMPLETE' or receipt['unit']!=unit or receipt['output_sha256']!=sha(fp):
            raise ValueError('Factor unit receipt/cache mismatch')
        paths[str(fp)] = receipt['output_sha256']
        paths[str(fp.with_suffix('.json'))] = sha(fp.with_suffix('.json'))
        factor_ids.add(receipt['identity_sha256'])
        with np.load(fp,allow_pickle=False) as z:
            if int(z['unit'])!=unit or not np.array_equal(z['configs'],np.arange(22)) or not np.array_equal(z['frames'],FRAMES):
                raise ValueError('Factor unit/config/frame axes differ')
            category, front = z['category'],z['target_front_range']
            if category.shape!=(22,13,2) or front.shape!=(22,13) or not np.isfinite(front).all():
                raise ValueError('Factor category/range axes differ')
        native = QM.RT.NR.OUT/'features/train'/f'unit{unit}.npz'
        paths[str(native)] = sha(native)
        with np.load(native,allow_pickle=False) as z:
            if int(z['unit'])!=unit or str(z['split'])!='train' or list(zip(z['scene'].tolist(),z['frame'].tolist()))!=[(c,f) for c in range(22) for f in range(16)]:
                raise ValueError('Native observation unit/config/frame axes differ')
            if z['z1'].shape!=(352,8,8,16) or not np.isfinite(z['z1']).all():
                raise ValueError('Native observation shape/values differ')
            group = z['group']
            if not np.array_equal(group,(unit+np.arange(22))%2) or z['family'].tolist()!=['none']*10+['panel']*12:
                raise ValueError('Native target query/background metadata differs')
        full_clear = (category=='clear').all(axis=1)
        unit_rows = np.flatnonzero(rows['unit']==unit).reshape(22,13)
        categories.update(category.flatten().tolist())
        for c in range(22):
            full_clear_counts['all_query_episodes'] += int(full_clear[c].sum())
            full_clear_counts['target_query_episodes'] += int(full_clear[c,group[c]])
            for fi,frame in enumerate(FRAMES):
                if not MIN_RANGE<=front[c,fi]<=MAX_RANGE:
                    continue
                eligible_categories.update(category[c,fi].tolist())
                row = int(unit_rows[c,fi])
                for query in range(2):
                    record = (row,query,float(front[c,fi]))
                    if category[c,fi,query]=='contact0-2cm':
                        positives.append(record)
                        n_positive_other_target_query += int(query!=group[c])
                    if full_clear[c,query] and query==group[c]:
                        negatives.append(record)
    if len(factor_ids)!=1:
        raise ValueError('Mixed inherited factor identities')
    pairs, missing, differences = match_pairs(rows,positives,negatives)
    if not len(pairs):
        raise ValueError('No supported shallow/clear ranking pairs')
    positive,negative,query = pairs.T
    if not np.all(labels[positive,query]==1) or not np.all(labels[negative,query]==0):
        raise ValueError('Pair truth conflicts with original M3 frame BCE targets')
    reuse = Counter(zip(negative.tolist(),query.tolist()))
    strata = {}
    for condition,panel in (('none',False),('panel',True)):
        pcount = sum((int(rows['config'][r])>=10)==panel for r,q,f in positives)
        ncount = sum((int(rows['config'][r])>=10)==panel for r,q,f in negatives)
        matched = int(((rows['config'][positive]>=10)==panel).sum())
        strata[condition] = dict(eligible_positive=pcount,eligible_negative=ncount,matched_positive=matched,unmatched_positive=pcount-matched)
    receipt = dict(status='COMPLETE',plan_sha256=digest,row_identity_sha256=identity,
        train_row_n=len(labels),pairs_n=len(pairs),source_sha256=sources,input_sha256=paths,
        inherited_factor_identity_sha256=next(iter(factor_ids)),
        categories=dict(categories),eligibility=dict(min_front_range_m=MIN_RANGE,max_front_range_m=MAX_RANGE,
            eligible_frame_categories=dict(eligible_categories),positive_frames=len(positives),negative_frames=len(negatives),
            full_clear_episode_counts=dict(full_clear_counts),positive_other_target_height=n_positive_other_target_query),
        coverage=dict(paired_positive_frames=len(pairs),unpaired_positive_frames=len(missing),
            fraction=len(pairs)/len(positives),strata=strata,
            positive_units=len(np.unique(rows['unit'][positive])),negative_units=len(np.unique(rows['unit'][negative])),
            unique_negative_frame_queries=len(reuse),max_negative_reuse=max(reuse.values()),
            negative_reuse_histogram={str(k):v for k,v in sorted(Counter(reuse.values()).items())},
            range_delta_m_quantiles=np.quantile(differences,[0,.5,.95,1]).tolist()),
        original_M3_pair_labels=dict(positive_all_one=True,negative_all_zero=True,checked_query_labels=2*len(pairs)),
        recipe=dict(match='motion mode unit%3,frame,query,none/panel; nearest target-front range <=.25m; stable negative row tie break',
            negative_reuse=True,full_clear='all 13 sampled training frame categories are clear, and query is target height',
            shared_boundary_pair_BCE='Both CBASE and CCON add identical pair-query BCE batches of 8, weight0.5',
            contrast='CCON additionally uses hinge max(0,1-positive_logit+negative_logit),weight1',
            main='Unchanged original M3 frame BCE',
            interpretation='Supervised matched sample/ranking contrast across sampled scenes, not causal intervention; labels and target range never inference features'))
    done = OUT/'pairs_receipt.json'
    if done.exists():
        previous = read(done)
        if {k:v for k,v in previous.items() if k!='output_sha256'}!=receipt or previous['output_sha256']['pairs.npz']!=sha(OUT/'pairs.npz'):
            raise ValueError('Existing pair receipt/cache changed')
        return previous
    with (OUT/'pairs.npz').open('xb') as stream:
        np.savez_compressed(stream,positive=positive,negative=negative,query=query)
    receipt['output_sha256'] = {'pairs.npz':sha(OUT/'pairs.npz')}
    save(done,receipt)
    print('COMPLETE pairs',len(pairs),'/',len(positives),'eligible shallow positives;',len(negatives),'clear negatives',flush=True)
    return receipt


def check():
    # Independent scenes remain eligible; ties use stable row order, not RNG.
    rows = dict(unit=np.array([93000,93006,93003,93001,93006]),config=np.array([0,0,0,0,10]),frame=np.array([3,3,3,3,3]))
    # Use an exact binary tie so representational rounding is irrelevant.
    pairs,missing,deltas = match_pairs(rows,[(0,0,1.25),(0,1,1.2)],[(2,0,1.),(1,0,1.5),(3,1,1.2),(4,1,1.2)])
    assert pairs.tolist()==[[0,1,0]] and missing==[(0,1)] and deltas==[.25]
    repeated,_,_ = match_pairs(rows,[(0,0,1.25),(0,0,1.25)],[(1,0,1.5)])
    assert repeated.tolist()==[[0,1,0],[0,1,0]]
    print('PASS stable row ties, inclusive .25m limit, mode/background/query strata and reusable negatives; no cohort loaded')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('check','prepare'))
    args=parser.parse_args(); {'check':check,'prepare':prepare}[args.stage]()
