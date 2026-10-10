"""Current native multi-peak tracks from public geometry on consumed Development.

No target attribution or coverage/free inference. Histograms are read once and
normalized once per observation. Tracks associate only distinct current frames,
not the already accumulated past8 readout. No category or authored box is read.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import cnh_graded_corridor_features_dev as C

ROOT = C.ROOT
OUTPUT = ROOT / 'artifacts.local/work/cnh-graded-peak-track-dev-20261010/tracks'
TOP = 8
WINDOW = 5
DISTANCE_M = .45
MISSING_FRAMES = 2
NAMES = (
    'tracked_sum', 'tracked_mean', 'tracked_max', 'hits', 'span',
    'max_gap', 'last_gap', 'matched_distance_residual', 'last_distance_residual',
    'depth_change', 'compensated_depth_change', 'inner_share', 'inner_weighted_sum',
    'current_peak', 'current_rank', 'current_membership', 'current_depth_m',
    'current_x_m', 'current_y_m', 'support_x_extent_m', 'support_y_extent_m',
    'support_z_extent_m', 'untracked_peakmax5', 'untracked_top1sum5',
    'untracked_positive_frames5', 'current_top8sum', 'candidate_count',
    'inner_best_tracked_sum', 'inner_best_hits', 'inner_best_span',
    'inner_best_max_gap', 'inner_best_share', 'inner_best_current_rank',
    'window_available_frames', 'left_boundary_warmup',
)


def native_geometry(sensor, query):
    """Fixed nine angular nodes/native bin and public center, current frame only."""
    edge = np.float32(np.tan(np.pi / 8))
    offsets = np.array([1 / 6, .5, 5 / 6], np.float32)
    slope = -edge + (np.arange(8, dtype=np.float32)[:, None] + offsets) * (2 * edge / 8)
    yy = np.broadcast_to(slope[:, None, :, None], (8, 8, 3, 3))
    xx = np.broadcast_to(slope[None, :, None, :], yy.shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), -1).reshape(64, 9, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    radii = (np.arange(16, dtype=np.float32) + .5) * C.WIDTH
    nodes = (rays[:, None] * radii[None, :, None, None]).reshape(1024, 9, 3)
    # Current native sensor coordinates use its public sensor-to-query map.
    # Sensor world poses enter only when moving a *past* query observation.
    transform = query[C.FRAMES]
    xyz = np.einsum('fij,bnj->fbni', transform[:, :3, :3].astype(np.float32), nodes)
    xyz += transform[:, None, None, :3, 3].astype(np.float32)
    x, y, z = np.moveaxis(xyz, -1, 0)
    expanded, inner = [], []
    for lo, hi in ((-.20, .42), (.42, .90)):
        vertical = (y >= lo) & (y <= hi) & (z >= .30) & (z <= 3)
        expanded.append(vertical & (abs(x) <= .40))
        inner.append(vertical & (abs(x) <= .30))
    em = np.stack(expanded, 1)
    im = np.stack(inner, 1)
    mass = em.mean(-1, dtype=np.float32)
    share = np.divide(im.sum(-1), em.sum(-1), out=np.zeros_like(mass), where=em.sum(-1) > 0)
    low = np.where(em[..., None], xyz[:, None], np.inf).min(-2)
    high = np.where(em[..., None], xyz[:, None], -np.inf).max(-2)
    return dict(membership=mass, inner_share=share, center_xyz=xyz[:, :, 4],
                support_low_xyz=low, support_high_xyz=high, node_xyz=xyz,
                expanded_nodes=em, inner_nodes=im,
                motion=query[:, None] @ np.linalg.inv(sensor[:, None]) @ sensor[None] @ np.linalg.inv(query[None]))


def current_candidates(z, geometry):
    """Positive gated radial maxima in 64 zones; top8 retains weaker candidates."""
    count = len(z)
    signed = np.sign(z.astype(np.float32)) * np.log1p(abs(z.astype(np.float32)))
    scores = signed[:, C.FRAMES].reshape(count, 13, 1, 64, 16)
    scores = np.maximum(scores * geometry['membership'][None].reshape(1, 13, 2, 64, 16), 0)
    pad = np.full((*scores.shape[:-1], 1), -np.inf, np.float32)
    left = np.concatenate((pad, scores[..., :-1]), -1)
    right = np.concatenate((scores[..., 1:], pad), -1)
    peak = ((scores > 0) & (scores > left) & (scores >= right)).reshape(count, 13, 2, 1024)
    amplitudes = scores.reshape(count, 13, 2, 1024)
    # Stable score/native-index lexicographic order makes equal-score choice explicit.
    ranked = np.argsort(-np.where(peak, amplitudes, -np.inf), axis=-1, kind='stable')[..., :TOP]
    valid = np.take_along_axis(peak, ranked, -1)
    amplitude = np.where(valid, np.take_along_axis(amplitudes, ranked, -1), 0)
    f = np.arange(13)[None, :, None, None]
    q = np.arange(2)[None, None, :, None]
    return dict(native_zone=np.where(valid, ranked // 16, -1).astype(np.int16),
                native_bin=np.where(valid, ranked % 16, -1).astype(np.int16),
                native_index=np.where(valid, ranked, -1).astype(np.int16), valid=valid,
                amplitude=amplitude.astype(np.float32),
                xyz=np.where(valid[..., None], geometry['center_xyz'][f, ranked], np.nan),
                support_low_xyz=np.where(valid[..., None], geometry['support_low_xyz'][f, q, ranked], np.nan),
                support_high_xyz=np.where(valid[..., None], geometry['support_high_xyz'][f, q, ranked], np.nan),
                membership=np.where(valid, geometry['membership'][f, q, ranked], 0),
                inner_share=np.where(valid, geometry['inner_share'][f, q, ranked], 0))


def transformed(point, motion):
    return motion[:3, :3] @ point + motion[:3, 3]


def extract_tracks(candidates, geometry):
    """One-to-one nearest association; retains gaps, starts at f3, no future access."""
    count = len(candidates['amplitude'])
    out = np.zeros((count, 13, 2, len(NAMES)), np.float32)
    validity = np.ones(out.shape, bool)
    identities = np.full((count, 13, 2, TOP), -1, np.int16)
    residuals = np.full((count, 13, 2, TOP), np.nan, np.float32)
    gaps = np.zeros((count, 13, 2, TOP), np.int8)
    selected = np.full((count, 13, 2), -1, np.int16)
    paths = np.full((count, 13, 2, WINDOW, 3), np.nan, np.float32)
    path_native = np.full((count, 13, 2, WINDOW), -1, np.int16)
    path_scores = np.zeros((count, 13, 2, WINDOW), np.float32)
    for n in range(count):
        for q in range(2):
            tracks = []
            for fj, frame in enumerate(C.FRAMES):
                begin = max(0, fj - WINDOW + 1)
                ranks = np.flatnonzero(candidates['valid'][n, fj, q])
                # Prune closed tracks, preserve track_id identity inside this episode.
                active = [t for t in tracks if fj - t['observations'][-1]['frame'] <= MISSING_FRAMES + 1]
                pairs = []
                for t in active:
                    last = t['observations'][-1]
                    moved = transformed(last['xyz'], geometry['motion'][frame, C.FRAMES[last['frame']]])
                    for rank in ranks:
                        d = float(np.linalg.norm(moved - candidates['xyz'][n, fj, q, rank]))
                        if d <= DISTANCE_M:
                            pairs.append((d, t['id'], int(rank)))
                assignments, used_tracks, used_ranks = {}, set(), set()
                for d, tid, rank in sorted(pairs):
                    if tid not in used_tracks and rank not in used_ranks:
                        assignments[rank] = (tracks[tid], d)
                        used_tracks.add(tid)
                        used_ranks.add(rank)
                now = []
                for rank in ranks:
                    rank = int(rank)
                    if rank in assignments:
                        t, d = assignments[rank]
                        gap = fj - t['observations'][-1]['frame'] - 1
                        residuals[n, fj, q, rank] = d
                    else:
                        t = dict(id=len(tracks), observations=[])
                        tracks.append(t)
                        d, gap = np.nan, 0
                    obs = dict(frame=fj, rank=rank, amplitude=float(candidates['amplitude'][n, fj, q, rank]),
                               xyz=candidates['xyz'][n, fj, q, rank].copy(),
                               inner_share=float(candidates['inner_share'][n, fj, q, rank]),
                               native_index=int(candidates['native_index'][n, fj, q, rank]),
                               residual=d, gap=gap)
                    t['observations'].append(obs)
                    identities[n, fj, q, rank] = t['id']
                    gaps[n, fj, q, rank] = gap
                    window = [o for o in t['observations'] if o['frame'] >= begin]
                    total = sum(o['amplitude'] for o in window)
                    inner_total = sum(o['amplitude'] * o['inner_share'] for o in window)
                    now.append((total, inner_total, -rank, t, window))
                history = candidates['amplitude'][n, begin:fj+1, q, 0]
                base = dict(untracked_peakmax5=float(history.max()), untracked_top1sum5=float(history.sum()),
                            untracked_positive_frames5=int((history > 0).sum()),
                            current_top8sum=float(candidates['amplitude'][n, fj, q].sum()),
                            candidate_count=len(ranks), window_available_frames=fj-begin+1,
                            left_boundary_warmup=int(fj < WINDOW-1))
                values = dict(base)
                if now:
                    total, inner_total, _, t, window = max(now, key=lambda v: (v[0], v[2]))
                    selected[n, fj, q] = t['id']
                    last, first = window[-1], window[0]
                    rank = last['rank']
                    residual = [o['residual'] for o in window if np.isfinite(o['residual'])]
                    maxgap = max((window[i]['frame']-window[i-1]['frame']-1 for i in range(1,len(window))), default=0)
                    moved_first = transformed(first['xyz'], geometry['motion'][frame,C.FRAMES[first['frame']]])
                    extent = candidates['support_high_xyz'][n, fj, q, rank] - candidates['support_low_xyz'][n, fj, q, rank]
                    values.update(tracked_sum=total, tracked_mean=total/len(window),
                        tracked_max=max(o['amplitude'] for o in window), hits=len(window),
                        span=last['frame']-first['frame']+1, max_gap=maxgap, last_gap=last['gap'],
                        matched_distance_residual=float(np.mean(residual)) if residual else np.nan,
                        last_distance_residual=last['residual'],
                        depth_change=float(first['xyz'][2]-last['xyz'][2]),
                        compensated_depth_change=float(moved_first[2]-last['xyz'][2]),
                        inner_share=inner_total/total, inner_weighted_sum=inner_total,
                        current_peak=last['amplitude'], current_rank=rank+1,
                        current_membership=float(candidates['membership'][n, fj, q, rank]),
                        current_depth_m=float(last['xyz'][2]), current_x_m=float(last['xyz'][0]), current_y_m=float(last['xyz'][1]),
                        support_x_extent_m=float(extent[0]), support_y_extent_m=float(extent[1]), support_z_extent_m=float(extent[2]))
                    itotal, isum, _, _, iw = max(now, key=lambda v: (v[1], v[2]))
                    values.update(inner_best_tracked_sum=isum, inner_best_hits=len(iw),
                                  inner_best_span=iw[-1]['frame']-iw[0]['frame']+1,
                                  inner_best_max_gap=max((iw[i]['frame']-iw[i-1]['frame']-1 for i in range(1,len(iw))),default=0),
                                  inner_best_share=isum/itotal, inner_best_current_rank=iw[-1]['rank']+1)
                    for o in window:
                        offset=o['frame']-begin
                        paths[n, fj, q, offset]=transformed(o['xyz'],geometry['motion'][frame,C.FRAMES[o['frame']]])
                        path_native[n, fj, q, offset]=o['native_index']
                        path_scores[n, fj, q, offset]=o['amplitude']
                for j, name in enumerate(NAMES):
                    out[n, fj, q, j] = values.get(name, np.nan)
                    validity[n, fj, q, j] = np.isfinite(out[n, fj, q, j])
    return dict(features=out, valid=validity, track_id=identities,
                match_residual=residuals, previous_missing_frames=gaps,
                best_track_id=selected, best_track_path_xyz=paths,
                best_track_path_native_index=path_native, best_track_path_amplitude=path_scores)


def run(output):
    began = time.monotonic()
    output = output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Use canonical artifacts.local')
    if (output/'PLAN.json').exists():
        raise FileExistsError('Preserve existing run')
    output.mkdir(parents=True,exist_ok=True)
    C.save(output/'PLAN.json',dict(task='CNH_GRADED_PEAK_TRACKS_DEV_20261010',
        CPU_command_wall_seconds_cap=300, GPU_seconds_cap=0, lane='EXPLORE consumed simulated Development',
        mechanism='Current expanded corridor positive radial local peaks, top8 per frame/query; public geometry one-to-one greedy nearest .45m, maximum2 missed frames, 5frame evidence window',
        start='f3..15 only; initial 4 frames left-boundary warmup; no inaccessible query frames used',
        match='All eligible previous-track/current-candidate distance pairs sorted (distance,track_id,rank), greedily one-to-one; no velocity/amplitude/target truth association',
        point='Native central angular ray (middle of nine nodes) at radial-bin center in current public query; supporting expanded nodes min/max XYZ saved separately',
        motion='query[current]@inv(sensor[current])@sensor[last]@inv(query[last])',
        selected='Only tracks with an actual current candidate; best by 5window positive amplitude sum, tie current lower rank; inner-weighted best also reported',
        controls='Untracked past5 frame-strongest peak max/sum, including identity-switching/background/noise peaks',
        truth_boundary='hist/ambient/bias/public sensor/query/sceneIDs/cache mapping only; no label/category/authored boxes',
        backend='CPU TASK_NOT_GPU_SUITABLE small greedy associations; reuse CPU raw extraction following prior measured CPU faster and CUDA count mismatch',
        new_sampling=0, model_forward=0, training=0, voxel_reconstruction=0,
        source_sha256=C.sha(__file__)))
    phase='input'
    try:
        bias=np.load(C.BIAS).astype(np.float32)
        input_hashes={}
        for split in ('cal','validation'):
            directory=C.SOURCE/'data'/split
            with np.load(directory/'geometry.npz',allow_pickle=False) as a:
                sensor,query,ids,uids=(a[k] for k in ('sensor','public_query','scene_ids','scene_uids'))
            with np.load(directory/'physics.npz',allow_pickle=False) as a:
                ambient=a['ambient']
                np.testing.assert_array_equal(a['sensor'],sensor)
                np.testing.assert_array_equal(a['public_query'],query)
            geometry=native_geometry(sensor,query)
            # Gate parity with the old fixed public geometry is an integration check.
            existing=C.public_geometry(sensor,query)
            np.testing.assert_array_equal(geometry['membership'],existing['membership'][:,:,1,-1])
            hist=np.load(directory/'hist.npy',mmap_mode='r',allow_pickle=False)
            assert hist.shape==(384,4,16,8,8,16)
            batches=[]
            for begin in range(0,384,16):
                if time.monotonic()-began>=300:
                    raise TimeoutError('CPU cap300s reached')
                phase=f'{split} batch{begin}'
                raw=np.array(hist[begin:begin+16],copy=True)
                normalized=C.normalize(raw,ambient,bias).reshape(-1,16,8,8,16)
                candidates=current_candidates(normalized,geometry)
                tracks=extract_tracks(candidates,geometry)
                batches.append({**tracks,**{f'candidate_{k}':v for k,v in candidates.items()}})
            merged={k:np.concatenate([b[k] for b in batches]).reshape(384,4,*batches[0][k].shape[1:]) for k in batches[0]}
            with np.load(directory/'rows.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_id'],np.repeat(ids,4*13))
                np.testing.assert_array_equal(a['replica'],np.tile(np.repeat(np.arange(4),13),384))
                np.testing.assert_array_equal(a['frame'],np.tile(C.FRAMES,384*4))
            np.savez_compressed(output/f'{split}_tracks.npz',**merged,names=np.array(NAMES),
                scene_ids=ids,scene_uids=uids,frames=C.FRAMES,queries=np.array(['HEAD','BODY']),
                replica=np.arange(4),cache_row_index=np.arange(19968).reshape(384,4,13))
            np.savez_compressed(output/f'{split}_public_geometry.npz',**geometry,sensor=sensor,public_query=query)
            # No-future invariance against the actual stored sequence at f7 and f11.
            prefix=[]
            for i,k,through in ((0,0,7),(383,3,11)):
                z=C.normalize(np.array(hist[i,k],copy=True),ambient,bias)[None]
                altered=z.copy(); altered[:,through+1:]=-10
                result=extract_tracks(current_candidates(altered,geometry),geometry)
                for key,values in result.items():
                    np.testing.assert_array_equal(values[0,:through-2],merged[key][i,k,:through-2])
                prefix.append(dict(scene=int(ids[i]),replica=k,through_frame=through,exact=True))
            C.save(output/f'{split}_summary.json',dict(shape=list(merged['features'].shape),prefix=prefix,
                track_hit_histogram=np.bincount(merged['features'][...,NAMES.index('hits')].astype(int).ravel(),minlength=6).tolist(),
                gapped_current_matches=int((merged['previous_missing_frames']>0).sum()),
                selected_current_rank_histogram=np.bincount(merged['features'][...,NAMES.index('current_rank')].astype(int).ravel(),minlength=9).tolist(),
                valid_candidates=int(merged['candidate_valid'].sum()),query_slots=39936))
            input_hashes[split]={str(p.relative_to(C.SOURCE)):C.sha(p) for p in
                               (directory/'hist.npy',directory/'geometry.npz',directory/'physics.npz',directory/'rows.npz')}
        C.save(output/'schema.json',dict(names=NAMES,axes=['scene384','replica4','frame13','query2','feature35'],
            input_sha256=input_hashes,bias_sha256=C.sha(C.BIAS),source_sha256=C.sha(__file__),
            missing='valid false and NaN for absent track or no measured residual; never free/negative',
            residual='Mean over window observations whose match residual is defined, including residual to preceding observation before window; last separate',
            depth_change='First observed query-depth minus current query-depth; compensated_depth_change transforms first peak to current query before subtracting',
            path='Window arrays left-aligned to actual begin=max(f3,current-4); absent observation NaN and nativeindex-1; all path XYZ transformed to current query',
            extent='Current selected native-bin nine-node expanded-corridor supporting-node XYZ bounds; not object extent or sensor coverage',
            validity_limits='Hard center-radius association can switch identity; background/noise persist; native bins and clipped support do not prove target identity'))
        C.save(output/'result.json',dict(status='COMPLETE',CPU_seconds=time.monotonic()-began,GPU_seconds=0,
            feature_count=len(NAMES),query_slots=79872,physical_output=str(output),
            limitation='Consumed ideal simulated Development. Associations are observed-peak hypotheses, not object attribution.'))
    except BaseException as error:
        C.save(output/f'failure_{time.time_ns()}.json',dict(error=repr(error),phase=phase,traceback=traceback.format_exc(),seconds=time.monotonic()-began))
        raise
    finally:
        C.save(output/'execution_receipt.json',dict(seconds=time.monotonic()-began,GPU_seconds=0,
            command=[sys.executable,*sys.argv],source_sha256=C.sha(__file__),
            outputs_sha256={p.name:C.sha(p) for p in output.iterdir() if p.is_file() and p.name!='execution_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
