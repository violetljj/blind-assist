"""Frozen runtime current-peak and temporal feature banks from audited tracks.

Only existing observations, public geometry descriptors and identity arrays are
loaded. No labels, authored boxes, outcome metrics, sampling or matching occurs.
Missing support is explicit NaN/validity, never an inferred free/negative state.
"""
import argparse
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import cnh_graded_peak_tracks_dev as T

ROOT = T.ROOT
SOURCE = T.OUTPUT
OUTPUT = ROOT / 'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/features'
CURRENT_NAMES = (
    'rank0_peak_log', 'top8_peak_log_sum', 'inner_weighted_peak_log_sum',
    'inner_weighted_peak_log_max', 'strongest_inner_peak_log',
    'inner_weighted_sum_share', 'inner_supported_candidate_count',
    'rank0_depth_m', 'rank0_x_m', 'rank0_y_m',
    'rank0_support_x_extent_m', 'rank0_support_y_extent_m', 'rank0_support_z_extent_m',
    'strongest_inner_depth_m', 'strongest_inner_x_m', 'strongest_inner_y_m',
    'strongest_inner_share', 'strongest_inner_support_x_extent_m',
    'strongest_inner_support_y_extent_m', 'strongest_inner_support_z_extent_m',
    'rank0_minus_inner_depth_m', 'rank0_to_inner_peak_log_ratio',
)


def build(candidate):
    """One current query slot's fixed top8 list; arrays may carry batch axes."""
    amp = candidate['candidate_amplitude']
    valid = candidate['candidate_valid'].astype(bool)
    share = candidate['candidate_inner_share']
    xyz = candidate['candidate_xyz']
    extent = candidate['candidate_support_high_xyz'] - candidate['candidate_support_low_xyz']
    if amp.shape[-1] != 8 or xyz.shape != (*amp.shape, 3):
        raise ValueError('Expected audited current top8 and XYZ')
    # A supported candidate is an observed positive peak with angular nodes in
    # the inner corridor. Membership does not prove object intrusion/coverage.
    iv = valid & (share > 0)
    has_peak = valid.any(-1)
    has_inner = iv.any(-1)
    chosen = np.argmax(np.where(iv, amp, -np.inf), axis=-1)
    take = lambda a: np.take_along_axis(a, chosen[..., None], -1)[..., 0]
    inner_amp = take(amp)
    inner_xyz = np.take_along_axis(xyz, chosen[..., None, None], -2)[..., 0, :]
    inner_extent = np.take_along_axis(extent, chosen[..., None, None], -2)[..., 0, :]
    total = np.where(valid, amp, 0).sum(-1)
    weighted = np.where(valid, amp * share, 0)
    weighted_sum = weighted.sum(-1)
    weighted_share = np.divide(weighted_sum, total, out=np.full_like(total, np.nan), where=total > 0)
    ratio = np.divide(amp[..., 0], inner_amp, out=np.full_like(total, np.nan), where=has_inner & (inner_amp > 0))
    values = [amp[..., 0], total, weighted_sum, weighted.max(-1), inner_amp,
              weighted_share, iv.sum(-1), xyz[..., 0, 2], xyz[..., 0, 0], xyz[..., 0, 1],
              extent[..., 0, 0], extent[..., 0, 1], extent[..., 0, 2],
              inner_xyz[..., 2], inner_xyz[..., 0], inner_xyz[..., 1], take(share),
              inner_extent[..., 0], inner_extent[..., 1], inner_extent[..., 2],
              xyz[..., 0, 2] - inner_xyz[..., 2], ratio]
    data = np.stack(values, -1).astype(np.float32)
    masks = np.broadcast_to(has_peak[..., None], data.shape).copy()
    masks[..., (0, 7, 8, 9, 10, 11, 12)] &= valid[..., 0, None]
    # Inner score and coordinate descriptors require observed inner support.
    # Count zero is still a valid count of this retained candidate list.
    masks[..., (2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21)] &= has_inner[..., None]
    masks &= np.isfinite(data)
    return np.where(masks, data, np.nan).astype(np.float32), masks


def declare(output):
    T.C.save(output/'PLAN.json', dict(task='CNH_GRADED_PEAK_JOINT_FEATURES_DEV_20261010',
        lane='EXPLORE consumed ideal simulated Development', CPU_command_wall_seconds_cap=120,
        GPU_seconds_cap=0, scope='Exactly two frozen banks, all existing cal/validation queries, once',
        goal='Current existence/support descriptors and public inner-corridor membership, with matched temporal descriptors',
        current_features=list(CURRENT_NAMES), temporal_features='current22 plus all audited35 track fields, prefix track_',
        definition='Strongest inner candidate maximizes original gated peak amplitude among inner_share>0 valid candidates, ties earlier existing rank',
        truth_boundary='Only track NPZ observations and source schema/PLAN audit identity; no category/authored boxes or outcome metrics',
        input_validity='Preserve NaN/valid; no imputation or normalization fit; inner-dependent descriptors missing when no retained candidate has inner node support',
        membership='Inner share fraction of expanded supporting angular nodes. Peak/track persistence and node extent do not prove target existence, intrusion or coverage.',
        new_sampling=0, raw_readout=0, matching=0, training=0, model_forward=0,
        source_sha256=T.C.sha(__file__)))


def run(output):
    began = time.monotonic()
    output = output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Use canonical artifacts.local')
    if (output/'PLAN.json').exists():
        raise FileExistsError('Preserve existing feature PLAN')
    output.mkdir(parents=True, exist_ok=True)
    declare(output)
    phase = 'load'
    try:
        temporal_names = (*CURRENT_NAMES, *(f'track_{name}' for name in T.NAMES))
        input_hashes = {}
        for split in ('cal', 'validation'):
            if time.monotonic()-began >= 120:
                raise TimeoutError('CPU command wall cap120s reached')
            phase = split
            source = SOURCE/f'{split}_tracks.npz'
            with np.load(source,allow_pickle=False) as a:
                candidate = {key:a[key] for key in ('candidate_amplitude','candidate_valid',
                    'candidate_inner_share','candidate_xyz','candidate_support_low_xyz','candidate_support_high_xyz')}
                track, track_valid, track_names = a['features'], a['valid'], a['names']
                identity = {key:a[key] for key in ('scene_ids','scene_uids','replica','frames','queries','cache_row_index')}
            np.testing.assert_array_equal(track_names, np.array(T.NAMES))
            current, current_valid = build(candidate)
            if current.shape != (384,4,13,2,22) or track.shape != (384,4,13,2,35):
                raise ValueError('Frozen split tensor shapes changed')
            temporal = np.concatenate((current, track),-1)
            temporal_valid = np.concatenate((current_valid,track_valid),-1)
            # Exact lineage checks: no slot mixing or altered track values.
            np.testing.assert_array_equal(temporal[...,22:],track)
            np.testing.assert_array_equal(temporal_valid[...,22:],track_valid)
            np.testing.assert_array_equal(current[...,0],candidate['candidate_amplitude'][...,0])
            np.testing.assert_array_equal(current[...,1],track[...,T.NAMES.index('current_top8sum')])
            assert np.all((current[...,5][current_valid[...,5]]>=0)&(current[...,5][current_valid[...,5]]<=1.000001))
            np.savez_compressed(output/f'{split}_features.npz',current_features=current,current_valid=current_valid,
                current_names=np.array(CURRENT_NAMES),temporal_features=temporal,temporal_valid=temporal_valid,
                temporal_names=np.array(temporal_names),**identity)
            T.C.save(output/f'{split}_summary.json',dict(current_shape=list(current.shape),temporal_shape=list(temporal.shape),
                invalid_current_by_feature=dict(zip(CURRENT_NAMES,(~current_valid).sum((0,1,2,3)).astype(int).tolist())),
                invalid_temporal_by_feature=dict(zip(temporal_names,(~temporal_valid).sum((0,1,2,3)).astype(int).tolist())),
                checks='Exact source rank0/top8 sum/35 track lineage; inner weighted-share bounded; identity copied exactly'))
            input_hashes[split] = T.C.sha(source)
        T.C.save(output/'schema.json',dict(current_names=CURRENT_NAMES,temporal_names=temporal_names,
            shape_axes=['scene384','replica4','frame13','query2','feature'],dtype='float32',valid_dtype='bool',
            input_tracks_sha256=input_hashes,input_track_schema_sha256=T.C.sha(SOURCE/'schema.json'),
            input_track_plan_sha256=T.C.sha(SOURCE/'PLAN.json'),
            input_track_audit_sha256=T.C.sha(SOURCE/'audit/result.json'),
            source_sha256=T.C.sha(__file__),physical_output=str(output),
            definitions=dict(rank0_peak_log='Rank0 existing current gated positive signed-log peak amplitude',
                top8_peak_log_sum='Sum of valid retained current peak amplitudes',
                inner_weighted_peak_log_sum='Sum amplitude*inner_share over current top8; missing if none has inner support',
                inner_weighted_peak_log_max='Maximum amplitude*inner_share over current top8; missing if none has inner support',
                strongest_inner_peak_log='Maximum original amplitude among valid candidates with inner_share>0, ties earlier rank',
                inner_weighted_sum_share='Weighted peak sum / all retained amplitudes sum; missing without observed inner support',
                inner_supported_candidate_count='Count of valid retained candidates with inner_share>0; zero is count, never free evidence',
                coordinate='Native center coordinates in current public query; depth=z, lateral=x, vertical=y',
                support_extent='Expanded-membership supporting-node max XYZ minus min XYZ for chosen native bin; not object size or physical coverage',
                strongest_inner_share='Chosen strongest inner candidate fraction of expanded angular support nodes in inner corridor',
                rank0_minus_inner_depth_m='Rank0 observed center z minus strongest inner center z; may have same bin/peak; not background visibility/free',
                rank0_to_inner_peak_log_ratio='Rank0 amplitude / strongest inner amplitude; not independent predictions',
                track_prefix='All35 source track fields unchanged, with explicit track_ prefix'),
            pose_identity='Inherited exact ideal public_query/sensor/calibration and native readout hashes from input track schema; no new pose assumptions',
            missing='NaN plus valid false; retained-count0 is valid description but missing geometry or candidates cannot infer negative/free',
            normalization='None. No cal/validation fitting in feature builder.'))
        T.C.save(output/'result.json',dict(status='COMPLETE',CPU_seconds=time.monotonic()-began,GPU_seconds=0,
            current_feature_count=22,temporal_feature_count=57,query_slots=79872,
            scope='Pure runtime descriptors from consumed ideal simulated Development; no target attribution'))
    except BaseException as error:
        T.C.save(output/f'failure_{time.time_ns()}.json',dict(phase=phase,error=repr(error),
            traceback=traceback.format_exc(),CPU_seconds=time.monotonic()-began))
        raise
    finally:
        T.C.save(output/'execution_receipt.json',dict(seconds=time.monotonic()-began,GPU_seconds=0,
            command=[sys.executable,*sys.argv],source_sha256=T.C.sha(__file__),
            outputs_sha256={p.name:T.C.sha(p) for p in output.iterdir() if p.is_file() and p.name!='execution_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
