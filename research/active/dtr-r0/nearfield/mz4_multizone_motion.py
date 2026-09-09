"""Fixed finite-family multizone motion experiment; no learning or real sensor claim."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

from body_query_collection_labels import read, sha, write
from contact_retina_spec import BODY_BOXES
from tof_jitter_geometry import rotation, trajectory

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from research_backend import BackendCandidate, Workload, select_backend, torch_observation


def scene_catalog(source):
    old = np.load(source)
    centers = np.array([(y, z) for y in range(-9, 10) for z in range(-9, 10)])
    pairs = np.array([(i, j) for i in range(len(centers))
                      for j in range(i + 1, len(centers))
                      if (np.abs(centers[i] - centers[j]) >= 2).any()])
    assert np.array_equal(centers, old['centers'])
    assert np.array_equal(pairs, old['pairs'])
    lo = 2.5 * np.tan(np.deg2rad(centers - 1))
    hi = 2.5 * np.tan(np.deg2rad(centers + 1))
    lo[:, 1] += 1.7
    hi[:, 1] += 1.7
    memberships = []
    for low, high in BODY_BOXES:
        hit = ((lo[:, 0] <= high[1]) & (hi[:, 0] >= low[1])
               & (lo[:, 1] <= high[2]) & (hi[:, 1] >= low[2]))
        memberships.append(np.concatenate(([False], hit, hit[pairs].any(1))))
    labels = np.stack(memberships, 1)
    assert np.array_equal(labels[:, 1], old['scene_head'])
    selected = old['selected'].copy()
    assert len(selected) == len(np.unique(selected)) == 500
    return centers, pairs, labels, selected


def choose_device(output):
    # Actual inference operation and dimensions; only the candidate block is smaller.
    rng = np.random.default_rng(71)
    v = rng.integers(-1, 2, (500, 1600)).astype(np.float32)
    h = rng.integers(0, 2, (1600, 1024)).astype(np.float32)
    candidates = []
    for device in ('cpu', 'cuda'):
        a, b = torch.as_tensor(v, device=device), torch.as_tensor(h, device=device)
        candidates.append(BackendCandidate(
            device, device, lambda a=a, b=b: a @ b,
            lambda result: torch_observation(output=result),
            torch.cuda.synchronize if device == 'cuda' else lambda: None))
    record = select_backend(Workload.BATCH_TENSOR, cpu=candidates[0], gpu=candidates[1],
                            warmups=1, repeats=3, record_path=output / 'backend.json',
                            capabilities={'torch': torch.__version__, 'cuda': torch.version.cuda})
    return record['selected_device_type']


def geometry(centers, pairs, device, yaw_bias=0.):
    """Zone solid-angle coverage of whole patches, with no center-ray surrogate."""
    yy, xx = torch.meshgrid(torch.arange(360, device=device, dtype=torch.float64),
                            torch.arange(640, device=device, dtype=torch.float64), indexing='ij')
    f = 320 / np.tan(np.deg2rad(50))
    rays = torch.stack((torch.ones_like(xx), (xx - 319.5) / f, -(yy - 179.5) / f), -1)
    angles = torch.rad2deg(torch.atan(rays[..., 1:]))
    crop = (angles.abs() <= 10).all(-1)
    c = torch.as_tensor(centers, device=device)
    masks = ((angles[crop][None] >= c[:, None] - 1)
             & (angles[crop][None] < c[:, None] + 1)).all(-1).double()
    norm = rays.norm(dim=-1)
    weight = norm.pow(-3)
    assert float((2.5 * norm[crop]).max()) < 2.6
    coverage, zone_maps, denominators, bounds = [], [], [], []
    for pose in trajectory():
        r = rotation(pose['pitch_deg'], pose['yaw_deg'] + yaw_bias)
        slope = np.tan(np.deg2rad(22.5))
        corners = np.array([[1., y, z] for y in (-slope, slope)
                            for z in (-slope, slope)]) @ r.T
        assert (corners[:, 0] > 0).all()
        assert (np.abs(corners[:, 1] / corners[:, 0]) < 320 / f).all()
        assert (np.abs(corners[:, 2] / corners[:, 0]) < 180 / f).all()
        s = rays @ torch.as_tensor(r, device=device)
        a = torch.rad2deg(torch.atan(s[..., 1:] / s[..., :1]))
        inside = (s[..., 0] > 0) & (a >= -22.5).all(-1) & (a < 22.5).all(-1)
        bins = torch.floor((a + 22.5) / 5.625).long().clamp(0, 7)
        zones = bins[..., 1] * 8 + bins[..., 0]
        zone_maps.append(torch.where(inside, zones, -1).cpu().numpy())
        denominator = torch.stack([weight[inside & (zones == zone)].sum() for zone in range(64)])
        assert (denominator > 0).all()
        weights = torch.zeros((64, int(crop.sum())), dtype=torch.float64, device=device)
        weights[zones[crop], torch.arange(int(crop.sum()), device=device)] = weight[crop]
        # Entire foreground support stays within every rotated sensor footprint.
        assert inside[crop].all()
        single = (weights @ masks.T) / denominator[:, None]
        coverage.append(single)
        denominators.append(denominator.cpu().numpy())
        bounds.append([float((3.25 * norm[inside]).min()), float((3.25 * norm[inside]).max())])
        assert bounds[-1][0] >= 3.25 and bounds[-1][1] < 4.
    single = torch.stack(coverage)
    pt = torch.as_tensor(pairs, device=device)
    all_coverage = torch.cat((torch.zeros((25, 64, 1), device=device, dtype=torch.float64),
                             single, single[..., pt].sum(-1)), -1)
    codes = all_coverage >= .02
    margin = float((all_coverage - .02).abs().min())
    # Independent scalar audit must preserve classification away from roundoff.
    assert margin > 1e-10
    return codes, dict(single_coverage=single.cpu().numpy(), zone_maps=np.stack(zone_maps),
                        denominators=np.stack(denominators), distance_bounds=np.array(bounds),
                        max_coverage=float(all_coverage.max()), threshold_margin=margin)


def infer(template, observed, valid, labels, device):
    """All-scene compatibility; labels only query the resulting feasible geometry."""
    h = torch.as_tensor(template.reshape(-1, template.shape[-1]), device=device).float()
    obs = torch.as_tensor(observed.reshape(len(observed), -1), device=device).float()
    v = torch.as_tensor(valid.reshape(len(valid), -1), device=device).float()
    mismatches = (v * (1 - 2 * obs)) @ h + (v * obs).sum(1, keepdim=True)
    assert (mismatches >= 0).all() and torch.equal(mismatches, mismatches.round())
    compatible = mismatches == 0
    count = compatible.sum(1)
    positives = compatible.float() @ torch.as_tensor(labels, device=device).float()
    states = torch.full((len(obs), 2), 2, device=device, dtype=torch.int8)
    states[(count[:, None] > 0) & (positives == 0)] = 0
    states[(count[:, None] > 0) & (positives == count[:, None])] = 1
    return states.cpu().numpy(), count.cpu().numpy(), compatible


def summary(states, counts, truth):
    answer = dict(exact_known=int(((states != 2).all(1) & (states == truth).all(1)).sum()),
                  cases=len(truth), any_unknown=int((states == 2).any(1).sum()),
                  empty=int((counts == 0).sum()), median_feasible=float(np.median(counts)))
    for i, name in enumerate(('BODY', 'HEAD')):
        t, s = truth[:, i], states[:, i]
        answer[name] = dict(positive_total=int(t.sum()), resolved_positive=int(((s == 1) & t).sum()),
                            false_positive=int(((s == 1) & ~t).sum()),
                            false_negative=int(((s == 0) & t).sum()), unknown=int((s == 2).sum()))
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to((ROOT / 'artifacts.local').resolve()) and not out.exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    start = time.perf_counter()
    protocol = Path(__file__).with_name('MZ4_MOTION_PROTOCOL_20260910.md')
    source = ROOT / 'artifacts.local/work/tof-motion-scan-20260910/run-v1/observations.npz'
    receipt = ROOT / 'artifacts.local/work/tof-motion-scan-20260910/run-v1/result.json'
    assert sha(source) == read(receipt)['observations_sha256']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in
              (protocol, Path(__file__), source, receipt,
               Path(__file__).with_name('tof_jitter_geometry.py'),
               Path(__file__).with_name('contact_retina_spec.py'))}
    write(out / 'launch.json', dict(inputs=hashes, phase='EXPLORE', training_steps=0))
    centers, pairs, labels, selected = scene_catalog(source)
    device = choose_device(out)
    code, geo = geometry(centers, pairs, device)
    biased, bias_geo = geometry(centers, pairs, device, .5)
    codes, bias_codes = code.cpu().numpy(), biased.cpu().numpy()
    del code, biased
    rng = np.random.default_rng(43)
    noise = rng.uniform(-.05, .05, (500, 25, 64))
    valid = rng.random((500, 25, 64)) >= .2
    truth = labels[selected]
    predictions, records, outputs, preserved = {}, {}, {}, {}
    for law in ('nearest_supported', 'background_only'):
        candidate = codes if law == 'nearest_supported' else np.zeros_like(codes)
        stationary = np.broadcast_to(candidate[:1], candidate.shape)
        moving_obs = np.moveaxis(candidate[..., selected], -1, 0)
        static_obs = np.moveaxis(stationary[..., selected], -1, 0)
        for obs in (moving_obs, static_obs):
            assert np.array_equal((np.where(obs, 2.55, 3.625) + noise) < 2.925, obs)
        methods = dict(single=(candidate[:1], static_obs[:, :1], valid[:, :1]),
                       stationary=(stationary, static_obs, valid),
                       moving_joint=(candidate, moving_obs, valid))
        if law == 'nearest_supported':
            methods.update(unaligned=(stationary, moving_obs, valid),
                           biased_yaw=(bias_codes, moving_obs, valid))
        for method, (template, obs, keep) in methods.items():
            states, counts, compat = infer(template, obs, keep, labels, device)
            included = compat[torch.arange(500, device=device),
                              torch.as_tensor(selected, device=device)].cpu().numpy()
            if method in ('single', 'stationary', 'moving_joint'):
                assert included.all()
                assert ((states == 2) | (states == truth)).all()
            key = law + '/' + method
            predictions[key] = states
            records[key] = counts
            preserved[key] = included
            outputs[key] = summary(states, counts, truth)
            print(key, outputs[key], flush=True)
            del compat
        if law == 'nearest_supported':
            frame_states, frame_counts = [], []
            for step in range(25):
                s, n, compatible = infer(candidate[step:step + 1], moving_obs[:, step:step + 1],
                                          valid[:, step:step + 1], labels, device)
                assert compatible[torch.arange(500, device=device),
                                  torch.as_tensor(selected, device=device)].all()
                frame_states.append(s)
                frame_counts.append(n)
            frame_states = np.stack(frame_states, 1)
            frame_counts = np.stack(frame_counts, 1)
            yes, no = (frame_states == 1).any(1), (frame_states == 0).any(1)
            assert not (yes & no).any()
            state = np.where(yes, 1, np.where(no, 0, 2)).astype(np.int8)
            key = law + '/moving_framewise'
            predictions[key], records[key] = state, frame_counts.min(1)
            preserved[key] = np.ones(500, bool)
            outputs[key] = summary(state, frame_counts.min(1), truth)
            outputs[key]['feasible_count_scope'] = 'minimum per-frame count; not a joint feasible set'
            print(key, outputs[key], flush=True)
    joint = predictions['nearest_supported/moving_joint']
    changes = {}
    for base in ('single', 'stationary', 'moving_framewise'):
        other = predictions['nearest_supported/' + base]
        changes[base] = dict(new_resolved_queries=int(((other == 2) & (joint != 2)).sum()),
                             lost_resolved_queries=int(((other != 2) & (joint == 2)).sum()),
                             newly_both_known=int(((other == 2).any(1) & (joint != 2).all(1)).sum()))
    torch.cuda.synchronize()
    np.savez_compressed(out / 'observations.npz', codes=codes, biased_codes=bias_codes,
                        centers=centers, pairs=pairs, labels=labels, selected=selected,
                        noise=noise, valid=valid, frame_states=frame_states, frame_counts=frame_counts,
                        **{'states_' + k.replace('/', '__'): v for k, v in predictions.items()},
                        **{'counts_' + k.replace('/', '__'): v for k, v in records.items()},
                        **{'included_' + k.replace('/', '__'): v for k, v in preserved.items()})
    np.savez_compressed(out / 'geometry.npz', **{k: v for k, v in geo.items()},
                        biased_single_coverage=bias_geo['single_coverage'],
                        biased_zone_maps=bias_geo['zone_maps'],
                        biased_denominators=bias_geo['denominators'])
    gain = all(changes[b]['new_resolved_queries'] > 0 for b in ('stationary', 'moving_framewise'))
    write(out / 'result.json', dict(status='IDEAL_INFORMATION_GAIN' if gain else 'NO_ADDED_JOINT_INFORMATION',
          results=outputs, moving_joint_changes=changes, input_hashes=hashes,
          artifact_hashes={n: sha(out / n) for n in ('observations.npz', 'geometry.npz', 'backend.json')},
          scene_hypotheses=len(labels), selected_cases=len(selected), training_steps=0,
          selected_device=device, seconds=time.perf_counter() - start,
          foreground_coverage_max=geo['max_coverage'], threshold_margin=geo['threshold_margin'],
          scope='Finite matched family; no independent hardware, RGB fusion, walking or safety result.'))
    print('DONE', out, time.perf_counter() - start, flush=True)


if __name__ == '__main__':
    main()
