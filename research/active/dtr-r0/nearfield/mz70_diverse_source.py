"""MZ70 CPU source assembly and fair paired geometry substitution.

Schedule IDs select TRAIN by frozen role only, never event values or predictions.
FeatureStore.build is the separate GPU allocation boundary; prepare/smoke do not
decode RGB or load/encode model inputs. All native labels are loss/evaluator-only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path
import shutil
import time
import numpy as np
from mz45_object_transfer import Bindings
from mz59_source import read, write_new, sha, load_npz, FEATURE_SHAPE, FRAME_BYTES
from mz59_source import FeatureStore as FeatureStore59
from mz64_geometry_source import exact
from mz68_missing_input_source import prepare as prepare68
from mz69_topology_transfer import load_predictor, bound_ref

TASK='mz70-diverse-learning-20260911'
STEPS=4096
SEED=170
ARMS=('CONTROL','DIVERSE')
PROFILES=('IDEAL','MERGE_CLOSE','DROP_CLOSE','ALL_INVALID')
COHORTS=('DEV','relation10000','distance5000','rich','mz36','mz48','mz55','mz61','mz67')
SOURCES=('mz48','old','mz55','mz61','mz67')
SOURCE_TAGS={0:'mz61',1:'mz67'}
SOURCE_INDEX_SHA='23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b'
MZ68_SHA='2565c40a93862c6513962484fa45ebecb4e9aabec503f299c931c45e61f813eb'
MZ69_SHA='b65fa2e2c45c2014761fb03254267b38307c1ceaf49c846653da562e6ffdf259'

def make_schedule(previous, bundle):
    fit61=np.sort(np.asarray(bundle['mz61_groups']['fit'],np.int64))
    fit67=np.sort(np.asarray(bundle['mz67_groups']['fit'],np.int64))
    assert len(fit61)==len(fit67)==2048
    cycle=np.arange(STEPS,dtype=np.int64)%1536
    schedule=dict(profile_index=np.arange(STEPS,dtype=np.int64)%4,
        mz48=previous['mz48'][cycle].copy(),OLD_NEG=previous['OLD_NEG'][cycle].copy(),
        query=previous['query'][cycle].copy(),CONTROL=np.empty((STEPS,4),np.int64),
        DIVERSE=np.empty((STEPS,4),np.int64),DIVERSE_source=np.empty((STEPS,4),np.int64),
        mz61_fit_ids=fit61,mz67_fit_ids=fit67,
        mz48_fit_ids=previous['mz48_fit_ids'].copy(),old_train_ids=previous['old_train_ids'].copy())
    rng=np.random.default_rng(SEED)
    combined=np.concatenate([fit61,fit67])
    tags=np.repeat(np.arange(2,dtype=np.int64),2048)
    replacement=np.concatenate([fit61,fit61])
    for profile in range(4):
        permutation=rng.permutation(4096)
        schedule['DIVERSE'][profile::4]=combined[permutation].reshape(1024,4)
        schedule['DIVERSE_source'][profile::4]=tags[permutation].reshape(1024,4)
        schedule['CONTROL'][profile::4]=replacement[permutation].reshape(1024,4)
    return schedule


def validate_schedule(schedule, previous, bundle):
    expected=make_schedule(previous,bundle)
    assert set(schedule)==set(expected)
    for k in expected:
        assert schedule[k].dtype==np.int64,k
        exact(schedule[k],expected[k],k)
    for key,width in [('CONTROL',4),('DIVERSE',4),('DIVERSE_source',4),('mz48',4),('OLD_NEG',8),('query',8)]:
        assert schedule[key].shape==(STEPS,width),key
    exposure={k:np.zeros((2048,4),np.int64) for k in ['CONTROL_mz61','DIVERSE_mz61','DIVERSE_mz67']}
    for source in ['mz61','mz67']:
        fit=schedule[source+'_fit_ids'];groups=bundle[source+'_groups']
        assert len(np.unique(fit))==2048
        assert bundle['predictions'][source+'/known'][fit].all()
        assert all(bundle[source+'_records'][int(i)]['role']=='TRAIN_CANDIDATE' for i in fit)
        forbidden=np.r_[groups['calibration'],groups['heldout_geometry']]
        assert not np.isin(fit,forbidden).any()
        truth=bundle['native_labels'][source]['cell_truth']
        known=bundle['native_labels'][source]['cell_known']
        assert not (truth[fit] & ~known[fit,...,None]).any()
    fit61,fit67=schedule['mz61_fit_ids'],schedule['mz67_fit_ids']
    for p in range(4):
        control=schedule['CONTROL'][p::4].ravel()
        idx=schedule['DIVERSE'][p::4].ravel();tag=schedule['DIVERSE_source'][p::4].ravel()
        assert Counter(control)=={int(i):2 for i in fit61}
        assert Counter(idx[tag==0])=={int(i):1 for i in fit61}
        assert Counter(idx[tag==1])=={int(i):1 for i in fit67}
        np.testing.assert_array_equal(control[tag==0],idx[tag==0])
        np.testing.assert_array_equal(control[tag==1],fit61[np.searchsorted(fit67,idx[tag==1])])
        np.add.at(exposure['CONTROL_mz61'][:,p],np.searchsorted(fit61,control),1)
        np.add.at(exposure['DIVERSE_mz61'][:,p],np.searchsorted(fit61,idx[tag==0]),1)
        np.add.at(exposure['DIVERSE_mz67'][:,p],np.searchsorted(fit67,idx[tag==1]),1)
    assert (exposure['CONTROL_mz61']==2).all()
    assert (exposure['DIVERSE_mz61']==1).all() and (exposure['DIVERSE_mz67']==1).all()
    assert np.isin(schedule['mz48'],bundle['groups']['fit']).all()
    assert bundle['predictions']['mz48/known'][np.unique(schedule['mz48'])].all()
    assert np.isin(schedule['OLD_NEG'],schedule['old_train_ids']).all()
    assert ((schedule['query']>=0)&(schedule['query']<4)).all()
    assert not bundle['old_truth'][schedule['OLD_NEG'],schedule['query']].any()
    for cohort in COHORTS[:3]:
        assert not np.isin(schedule['OLD_NEG'],bundle['cohorts'][cohort]['global_ids']).any()
    return dict(status='PASS',seed=SEED,steps_per_arm=STEPS,profile_steps=[1024]*4,
        geometry_presentations_per_arm=16384,geometry_presentations_per_profile=4096,
        DIVERSE_presentations_per_source_per_profile=2048,CONTROL_every_frame_profile=2,
        DIVERSE_every_frame_profile=1,shared_source61_half_position_exact=True,
        replacement='sorted TRAIN ordinal bijection; no labels or predictions select order',
        old_negative_query_truth=False,replay_cycle_steps=1536,replay_steps=4096,
        no_calibration_or_heldout=True),exposure


def feature_index(bundle):
    schedule = bundle['schedule']
    ids = dict(mz48=np.unique(schedule['mz48']), old=np.unique(schedule['OLD_NEG']),
               mz55=np.empty(0,np.int64), mz61=np.unique(schedule['CONTROL']),
               mz67=np.unique(schedule['DIVERSE'][schedule['DIVERSE_source']==1]))
    allowed = dict(mz48=bundle['groups']['fit'], old=schedule['old_train_ids'],
                   mz55=bundle['mz55_groups']['fit'], mz61=bundle['mz61_groups']['fit'],
                   mz67=bundle['mz67_groups']['fit'])
    index = {}; offset = 0
    for source in SOURCES:
        assert np.isin(ids[source], allowed[source]).all()
        size = len(bundle['old_truth']) if source == 'old' else len(bundle['cohorts'][source]['frame_ids'])
        lookup = np.full(size, -1, np.int32)
        lookup[ids[source]] = np.arange(len(ids[source]), dtype=np.int32) + offset
        index[source + '_ids'], index[source + '_lookup'] = ids[source], lookup
        offset += len(ids[source])
    shape = (offset, *FEATURE_SHAPE)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False, shape=shape))
    frames, hits = {}, {}
    for cohort in COHORTS:
        rows = bundle['cohorts'][cohort]
        frames[cohort] = len(rows['frame_ids'])
        if cohort in COHORTS[:3]:
            cached = index['old_lookup'][rows['global_ids']]
        elif cohort in ('mz48', 'mz55', 'mz61', 'mz67'):
            cached = index[cohort + '_lookup']
        else:
            cached = np.full(frames[cohort], -1)
        hits[cohort] = int((cached >= 0).sum())
    misses = sum(frames.values()) - sum(hits.values())
    plan = dict(training_unique={source: len(ids[source]) for source in SOURCES},
        training_unique_total=offset, shape=list(shape), dtype='float32', bytes_per_frame=FRAME_BYTES,
        data_bytes=offset * FRAME_BYTES, npy_header_bytes=len(header.getvalue()),
        expected_file_bytes=offset * FRAME_BYTES + len(header.getvalue()), fixed_encoder_batch=16,
        evaluation_frames=frames, evaluation_cache_hits=hits, uncached_evaluation_frames=misses,
        planned_full_rgb_encodes=offset + misses, normalization='Original encoder values; runner applies frozen mean/std',
        scratch_owner=TASK, prior_full_cache_reused=False, calibration_or_heldout_in_training=False,
        baseline_replay_frames=0, extraction_unit='One per unique registered source/frame RGB reference',
        evaluation_call_contract='One batch outside arm/profile loops; reuse returned values for every arm/profile')
    return index, plan


def prepare(root, bind):
    started=time.perf_counter();root=Path(root);work=root/'artifacts.local/work'
    bundle=prepare68(root,bind)
    bind(Path(__file__))
    r68=read(bind(work/'mz68-missing-input-coverage-20260911/run-v1/receipt.json',MZ68_SHA))
    assert r68['status']=='PASS'
    for name in ['mz68_missing_input_source.py','mz68_missing_input_coverage.py']:
        refs=[(p,h) for p,h in r68['inputs'].items() if Path(p).name==name]
        assert len(refs)==1;bind(*refs[0])
    path68=work/'mz68-missing-input-coverage-20260911/run-v1'
    saved68=load_npz(bind(path68/'predictions.npz',r68['outputs']['predictions.npz']))
    for key,value in bundle['predictions'].items():exact(saved68[key],value,key)
    previous=load_npz(bind(path68/'schedule.npz',r68['outputs']['schedule.npz']))
    assert set(previous)==set(bundle['schedule'])
    for key,value in bundle['schedule'].items():exact(previous[key],value,key)
    bundle['predictions']=dict(saved68)
    source = work / 'mz67-topology-source-20260911'
    predictor, index_path = load_predictor(source, bind, SOURCE_INDEX_SHA)
    assert set(predictor) == {'frame_ids', 'ranges', 'valid', 'rgb_refs'}
    index = read(index_path); combined = index['combined']
    sr = read(bound_ref(source, combined['receipt.json'], bind))
    paths = {}
    for name in ('metadata.json', 'evaluator.npz', 'fullframe-cells.npz'):
        assert sr['outputs'][name] == combined[name]['sha256']
        paths[name] = bound_ref(source, combined[name], bind)
    meta = read(paths['metadata.json']); records, pairs = meta['records'], meta['pairs']
    labels, cells = load_npz(paths['evaluator.npz']), load_npz(paths['fullframe-cells.npz'])
    assert meta['source_role'] == 'CONSUMED_DEVELOPMENT' and len(records) == 4096
    for actual in ([r['frame_id'] for r in records], labels['frame_ids'], cells['frame_ids']):
        np.testing.assert_array_equal(actual, predictor['frame_ids'])
    np.testing.assert_array_equal([r['index'] for r in records], np.arange(4096))
    np.testing.assert_array_equal(cells['global_indices'], np.arange(4096))
    truth, known = labels['truth'], labels['known']
    counts, valid_counts = cells['fullframe_event_counts'], cells['valid_counts']
    assert truth.shape == known.shape == (4096, 4) and truth.dtype == known.dtype == bool
    assert counts.shape == (4096, 45, 80, 4) and valid_counts.shape == (4096, 45, 80)
    assert counts.dtype == valid_counts.dtype == np.uint8
    assert (valid_counts <= 64).all() and (counts <= valid_counts[..., None]).all()
    np.testing.assert_array_equal(counts.sum((1, 2)), [r['event_counts'] for r in records])
    np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, truth)
    np.testing.assert_array_equal(truth, [r['event_truth'] for r in records])
    np.testing.assert_array_equal(known, [[r['source_valid']] * 4 for r in records])
    roles = np.array([r['role'] for r in records])
    assert Counter(roles) == meta['roles'] == dict(TRAIN_CANDIDATE=2048, CALIBRATION=1024, HELDOUT_GEOMETRY=1024)
    groups = dict(fit=np.flatnonzero(roles == 'TRAIN_CANDIDATE'),
        calibration=np.flatnonzero(roles == 'CALIBRATION'), heldout_geometry=np.flatnonzero(roles == 'HELDOUT_GEOMETRY'))
    assert sorted(i for pair in pairs.values() for i in pair) == list(range(4096)) and len(pairs) == 2048
    geometry_roles = defaultdict(set)
    for i, row in enumerate(records):
        ref = predictor['rgb_refs'][i]
        assert row['rgb'] == ref.member and row['rgb_sha256'] == ref.sha256
        geometry_roles[row['geometry_id']].add(row['role'])
    assert all(len(assigned) == 1 for assigned in geometry_roles.values())
    for name, ids in pairs.items():
        assert len(ids) == 2 and {records[i]['pair_id'] for i in ids} == {name}
        assert len({roles[i] for i in ids}) == 1
        assert {records[i]['support_context'] for i in ids} == {'unsupported', 'supported'}
        np.testing.assert_array_equal(counts[ids[0]], counts[ids[1]])
        np.testing.assert_array_equal(truth[ids[0]], truth[ids[1]])
    for old in ('mz48', 'mz55', 'mz61'):
        assert not set(predictor['frame_ids']) & set(bundle['cohorts'][old]['frame_ids'])


    path69=work/'mz69-topology-transfer-20260911/run-v1'
    r69=read(bind(path69/'receipt.json',MZ69_SHA));assert r69['status']=='PASS'
    assert r69['frames']==4096 and r69['source_index_sha256']==SOURCE_INDEX_SHA
    refs=[(p,h) for p,h in r69['inputs'].items() if Path(p).name=='mz69_topology_transfer.py']
    assert len(refs)==1;bind(*refs[0])
    saved69=load_npz(bind(path69/'predictions.npz',r69['outputs']['predictions.npz']))
    exact(saved69['frame_ids'],predictor['frame_ids'])
    for key,value in saved69.items():
        assert 'mz67/'+key not in bundle['predictions']
        bundle['predictions']['mz67/'+key]=value
    bundle['predictions']['mz67/truth']=truth;bundle['predictions']['mz67/known']=known
    bundle['cohorts']['mz67']=predictor
    bundle.update(mz67_records=records,mz67_groups=groups,mz67_pairs=pairs,
        mz67_truth=truth,mz67_known=known,mz67_cell_counts=counts,mz67_valid_counts=valid_counts,
        mz67_cell_truth=counts>0,mz67_cell_known=valid_counts>0)
    bundle['native_labels']['mz67']=dict(cell_truth=bundle['mz67_cell_truth'],cell_known=bundle['mz67_cell_known'])
    bundle['inherited_mz64_schedule']=previous
    bundle['schedule']=make_schedule(previous,bundle)
    checks,exposure=validate_schedule(bundle['schedule'],previous,bundle)
    bundle['exposure']=exposure
    bundle['feature_index'],plan=feature_index(bundle)
    candidate=work/TASK/'source-preparation-v1/schedule.npz'
    if candidate.exists():
        stored=load_npz(bind(candidate))
        assert set(stored)==set(bundle['schedule'])
        for key in stored:exact(stored[key],bundle['schedule'][key],key)
    bundle['candidate_schedule_path']=candidate.resolve()
    bundle['source_info'].update(adapter=Path(__file__).name,scratch_owner=TASK,
        feature_plan=plan,training_unique=plan['training_unique'],training_unique_total=plan['training_unique_total'],
        feature_file_bytes=plan['expected_file_bytes'],schedule_seed=SEED,schedule_checks=checks,
        candidate_schedule_path=str(candidate.resolve()),candidate_schedule_sha256=sha(candidate) if candidate.exists() else None,
        schedule_design='seed170 sequential four-profile permutation of tagged sorted TRAIN; paired CONTROL ordinal replacement',
        replay_description='Original MZ64 1536-step mz48 first4/OLD_NEG8/query exact step%1536 for both arms',
        geometry_presentations=16384,mz48_presentations=16384,old_negative_presentations=32768,
        mz55_fit_presentations=0,exact_mz64_geometry_sequence=False,
        mz68_arrays_preserved=len(saved68),mz69_arrays_prefixed=len(saved69),
        mz68_preserved_keys=sorted(saved68),mz69_prefixed_keys=['mz67/'+k for k in sorted(saved69)],
        inherited_prediction_keys=sorted(bundle['predictions']),
        mz68_predictions_ref=dict(path=str((path68/'predictions.npz').resolve()),sha256=r68['outputs']['predictions.npz']),
        mz69_predictions_ref=dict(path=str((path69/'predictions.npz').resolve()),sha256=r69['outputs']['predictions.npz']),
        mz68_receipt_sha256=MZ68_SHA,mz69_receipt_sha256=MZ69_SHA,
        mz68_predictions_sha256=r68['outputs']['predictions.npz'],mz69_predictions_sha256=r69['outputs']['predictions.npz'],
        mz67_groups={k:len(v) for k,v in groups.items()},mz67_pairs=len(pairs),
        mz67_unknown_query_bits=int((~known).sum()),mz67_unknown_cells=int((valid_counts==0).sum()),
        native_labels_authority='Four source fullframe counts/known are loss/evaluator only',
        ALL_INVALID='Exactly one complete source TRAIN exposure per DIVERSE profile; two per CONTROL; packet zeroed in runner',
        source67_native_depth_reads=0,prepare_seconds=time.perf_counter()-started)
    bundle['source_info']['label_refs']['mz67']=dict(path=str(paths['fullframe-cells.npz'].resolve()),sha256=combined['fullframe-cells.npz']['sha256'])
    bundle['source_info']['source_index_refs']['mz67']=dict(path=str(index_path.resolve()),sha256=SOURCE_INDEX_SHA)
    bundle['source_info']['mz67_source_index_ref']=dict(path=str(index_path.resolve()),sha256=SOURCE_INDEX_SHA)
    for key in ['metadata.json','evaluator.npz']:
        bundle['source_info']['mz67_'+key.split('.')[0]+'_ref']=dict(path=str(paths[key].resolve()),sha256=combined[key]['sha256'])
    assert bundle['old_maps'] is None and plan['training_unique']['mz55']==0
    return bundle


class FeatureStore(FeatureStore59):
    """Five-source owner, with empty MZ55 fit cache; inherited encoder arithmetic and mmap close, no patching."""
    def __init__(self, base, store, bundle, task, bind):
        self.base, self.store, self.bundle, self.bind = base, store, bundle, bind
        self.task = Path(task).resolve()
        assert self.task == Path(bundle['source_info']['artifact_work']) / TASK
        self.index, self.plan = feature_index(bundle)
        assert self.plan == bundle['source_info']['feature_plan']
        for key, value in self.index.items(): exact(value, bundle['feature_index'][key], key)
        self.old_ids, self.mz48_ids, self.mz55_ids, self.mz61_ids, self.mz67_ids = (self.index[s + '_ids'] for s in ('old', 'mz48', 'mz55', 'mz61', 'mz67'))
        self.path = self.task / 'scratch-v1/training-full.npy'
        self.out = self.task / 'feature-plan-v1'
        self.full = None
        self.stats = dict(training_extracted_frames=0, full_extracted_frames=0, full_cache_eval_hits=0,
                          evaluation_extracted_frames=0, encoder_seconds=0., build_seconds=0., evaluation_seconds=0.)

    def build(self):
        assert self.full is None and not self.path.exists() and not self.out.exists()
        assert not self.path.parent.exists(), 'Preserve prior scratch/failure evidence'
        self.task.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(self.task).free > self.plan['expected_file_bytes'] + 1024 ** 3
        self.out.mkdir(); self.path.parent.mkdir(); tick = time.perf_counter()
        try:
            np.savez_compressed(self.out / 'feature-cache-index.npz', **self.index)
            write_new(self.out / 'plan.json', self.plan)
            self.full = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.float32, shape=tuple(self.plan['shape']))
            refs = []
            for source in SOURCES:
                original = self.bundle['old_rgb_refs'] if source == 'old' else self.bundle['cohorts'][source]['rgb_refs']
                refs.extend(original[int(i)] for i in self.index[source + '_ids'])
            for start in range(0, len(refs), 16):
                images = []
                try:
                    for ref in refs[start:start+16]:
                        images.append(self.store.load(ref))
                    self.full[start:start+len(images)] = self.extract(images)
                    self.stats['training_extracted_frames'] += len(images)
                finally:
                    for image in images: image.close()
                if start % 512 == 0 or start + len(images) == len(refs):
                    print('FEATURES', dict(frames=start+len(images), total=len(refs), seconds=time.perf_counter()-tick), flush=True)
            self.full.flush(); self.full._mmap.close(); self.full = None
            assert self.path.stat().st_size == self.plan['expected_file_bytes']
            self.full = np.load(self.path, mmap_mode='r', allow_pickle=False)
            assert self.full.mode == 'r' and not self.full.flags.writeable
            assert self.full.shape == tuple(self.plan['shape']) and self.full.dtype == np.float32
            assert self.stats['training_extracted_frames'] == self.plan['training_unique_total']
            self.cache_evidence = dict(path=str(self.path), bytes=self.path.stat().st_size, sha256=sha(self.path),
                feature_index_sha256=sha(self.out / 'feature-cache-index.npz'), owner=self.task.name,
                prior_full_cache_reused=False, readonly_after_build=True)
            self.stats['build_seconds'] = time.perf_counter() - tick
            write_new(self.out / 'receipt.json', dict(status='PASS', plan=self.plan, cache=self.cache_evidence,
                stats=self.stats, code_sha256=sha(__file__), inherited_encoder_code_sha256=sha(Path(__file__).with_name('mz59_source.py')),
                cleanup_owner='root after terminal score'))
            self.bind(self.out / 'feature-cache-index.npz', self.cache_evidence['feature_index_sha256'])
            self.bind(self.out / 'receipt.json')
            return self.stats['build_seconds']
        except BaseException:
            import traceback
            self.close()
            write_new(self.out / 'failure.json', dict(status='FAIL', error=traceback.format_exc(), stats=self.stats))
            raise

    def training(self, source, ids):
        assert self.full is not None and source in SOURCES
        ids = np.asarray(ids); lookup = self.index[source + '_lookup']
        assert ids.ndim == 1 and ids.dtype.kind in 'iu' and 0 < len(ids) <= 16
        assert ((ids >= 0) & (ids < len(lookup))).all()
        rows = lookup[ids]
        assert (rows >= 0).all(), 'Only registered training unique IDs may be read'
        return np.array(self.full[rows])

    def evaluation(self, cohort, ids):
        assert self.full is not None and cohort in COHORTS
        ids = np.asarray(ids); rows = self.bundle['cohorts'][cohort]
        assert ids.ndim == 1 and ids.dtype.kind in 'iu' and 0 < len(ids) <= 16
        assert ((ids >= 0) & (ids < len(rows['frame_ids']))).all()
        tick = time.perf_counter(); dense = np.empty((len(ids), *FEATURE_SHAPE), np.float32)
        images, missing = [], []
        try:
            for j, i in enumerate(ids):
                cached = -1
                if cohort in COHORTS[:3]:
                    cached = self.index['old_lookup'][int(rows['global_ids'][i])]
                elif cohort in ('mz48', 'mz55', 'mz61', 'mz67'):
                    cached = self.index[cohort + '_lookup'][i]
                if cached >= 0:
                    dense[j] = self.full[cached]
                    self.stats['full_cache_eval_hits'] += 1
                else:
                    missing.append(j); images.append(self.store.load(rows['rgb_refs'][int(i)]))
            if missing:
                dense[missing] = self.extract(images)
                self.stats['evaluation_extracted_frames'] += len(missing)
            return dense
        finally:
            for image in images:
                image.close()
            self.stats['evaluation_seconds'] += time.perf_counter() - tick


def smoke(root, output):
    """CPU source/schema/cache-plan assembly; no RGB decode or cache allocation."""
    import torch
    import traceback
    output=Path(output).resolve();task=(Path(root)/'artifacts.local/work'/TASK).resolve()
    assert output==task/'source-preparation-v1'
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'receipt.json').exists() and not (output/'schedule.npz').exists()
    bind=Bindings();features=None;bundle=None;started=time.perf_counter()
    assert not torch.cuda.is_initialized()
    try:
        bundle=prepare(root,bind)
        features=FeatureStore(None,None,bundle,task,bind)
        assert features.full is None and not features.path.exists()
        assert FeatureStore.extract is FeatureStore59.extract and FeatureStore.close is FeatureStore59.close
        rejected=[]
        for key in ['CONTROL','DIVERSE','DIVERSE_source','profile_index','mz48','OLD_NEG','query']:
            bad=dict(bundle['schedule']);bad[key]=bad[key].copy();bad[key].flat[0]=-12345
            try:validate_schedule(bad,bundle['inherited_mz64_schedule'],bundle)
            except AssertionError:rejected.append(key)
            else:raise AssertionError('Bad schedule accepted: '+key)
        try:FeatureStore(None,None,bundle,task.parent/'mz68-missing-input-coverage-20260911',bind)
        except AssertionError:rejected.append('wrong_cache_owner')
        else:raise AssertionError('Wrong cache owner accepted')
        for source in ['mz48','mz55','mz61','mz67']:
            groups=bundle['groups'] if source=='mz48' else bundle[source+'_groups']
            for name in ['calibration','heldout_site','heldout_geometry','nonfit_family']:
                if name in groups:assert (features.index[source+'_lookup'][groups[name]]<0).all()
        assert len(features.mz55_ids)==0 and len(features.mz61_ids)==len(features.mz67_ids)==2048
        # Source-order test changes labels while retaining valid role; scheduling must not inspect labels.
        alternative=dict(bundle,predictions=None,native_labels=None)
        regenerated=make_schedule(bundle['inherited_mz64_schedule'],alternative)
        for k,v in bundle['schedule'].items():exact(v,regenerated[k],k)
        np.savez_compressed(output/'schedule.npz',**bundle['schedule'])
        np.savez_compressed(output/'feature-cache-index.npz',**bundle['feature_index'])
        np.savez_compressed(output/'geometry-exposure.npz',**bundle['exposure'],
            mz61_fit_ids=bundle['schedule']['mz61_fit_ids'],mz67_fit_ids=bundle['schedule']['mz67_fit_ids'])
        bundle['source_info']['candidate_schedule_sha256']=sha(output/'schedule.npz')
        bind(output/'schedule.npz',bundle['source_info']['candidate_schedule_sha256'])
        write_new(output/'feature-plan.json',features.plan)
        write_new(output/'source-info.json',bundle['source_info'])
        bind.check()
        features.close();features=None
        assert not torch.cuda.is_initialized()
        write_new(output/'receipt.json',dict(status='PASS',inputs=bind.inputs,
            outputs={name:sha(output/name) for name in ['schedule.npz','feature-cache-index.npz',
                'geometry-exposure.npz','feature-plan.json','source-info.json']},
            source_info=bundle['source_info'],corruption_rejections=rejected,
            schedule_independent_of_event_values=True,full_per_frame_geometry_profile_exposure=True,
            no_other_replay_profile_coverage_claim=True,no_mz55_fit_cache=True,
            no_calibration_heldout_in_fit=True,old_predictions_preserved=len(bundle['predictions']),
            encoder_arithmetic_inherited=True,training_steps=0,model_loads=0,encoder_frames=0,
            rgb_decodes=0,raw_native_depth_reads=0,temporary_dense_bytes_created=0,
            cuda_initialized=False,mmap_handles_closed=True,backend='FROZEN_PROTOCOL_CPU_ONLY',
            seconds=time.perf_counter()-started))
        print('SOURCE PASS',dict(unique=features.plan if features is not None else bundle['source_info']['training_unique'],
            cache_bytes=bundle['source_info']['feature_file_bytes'],arrays=len(bundle['predictions']),
            seconds=time.perf_counter()-started),flush=True)
    except BaseException:
        write_new(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=bind.inputs))
        raise
    finally:
        if features is not None:features.close()
        if bundle is not None and bundle.get('old_maps') is not None:bundle['old_maps']._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();smoke(args.root.resolve(),args.output.resolve())

