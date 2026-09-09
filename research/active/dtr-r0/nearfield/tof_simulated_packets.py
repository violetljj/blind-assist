"""HYPOTHETICAL single-return laws on native depth; not a VL53L1X emulator.

Measurement functions accept scene depth only. Target masks, labels, identities,
reflectance and model predictions are neither inputs nor return-law authorities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from body_query_collection_labels import read, sha, write
from body_query_tof_coverage import footprints

LAWS = ('nearest_supported', 'max_solid_angle_area', 'farthest_supported')
FOVS = (15, 27)
MIN_RANGE, MAX_RANGE, BIN_WIDTH = .1, 4., .1
MIN_WEIGHT_FRACTION = .02
SOURCE = 'SIMULATED_HYPOTHESIS'


def disturbances(frames, seed=17):
    """Independent noise/dropout RNG streams, shared across every law and FoV."""
    noise_seed, dropout_seed = np.random.SeedSequence(seed).spawn(2)
    return (np.random.default_rng(noise_seed).uniform(-.05, .05, frames),
            np.random.default_rng(dropout_seed).random(frames) < .20)


def range_histogram(native, footprint):
    """Solid-angle bin mass divided by FULL footprint; UNKNOWN gets no mass.

    39 bins cover [0.1,4.0]m; interior boundaries enter the right bin, exactly
    4.0 enters the final bin. Use deterministic reductions, not CUDA atomics.
    """
    if native.ndim != 3 or tuple(native.shape[1:]) != (360, 640):
        raise ValueError('Expected Bx360x640 axial native depth')
    mask, weights, norms = footprint
    scene = native[:, mask].to(torch.float64)
    radial = scene * norms.to(torch.float64)
    valid = torch.isfinite(scene) & (scene > 0) & (scene < 100)
    valid &= (radial >= MIN_RANGE) & (radial <= MAX_RANGE)
    edges = torch.linspace(MIN_RANGE, MAX_RANGE, 40, dtype=torch.float64, device=native.device)
    bins = torch.bucketize(radial.contiguous(), edges[1:-1], right=True)
    weights = weights.to(torch.float64)
    masses = torch.stack([((valid & (bins == b)) * weights).sum(1)
                          for b in range(39)], dim=1) / weights.sum()
    return masses


def measure(native, footprint):
    """Return ideal radial bin centers; no source metadata accepted here."""
    masses = range_histogram(native, footprint)
    supported = masses >= MIN_WEIGHT_FRACTION
    any_supported = supported.any(1)
    nearest = supported.to(torch.int32).argmax(1)
    farthest = 38 - supported.flip(1).to(torch.int32).argmax(1)
    # Tie policy: first (nearest) maximum bin, including exact mass ties.
    largest = masses.masked_fill(~supported, -1).argmax(1)
    output = {}
    for law, indices in zip(LAWS, (nearest, largest, farthest)):
        values = MIN_RANGE + (indices.to(torch.float64) + .5) * BIN_WIDTH
        output[law] = (values.masked_fill(~any_supported, float('nan')), any_supported)
    return output


def packets(sample_ids, ideal_values, supported, noise, dropout, diagonal_fov):
    """Keep model-visible fields minimal. A law is identified by its file name."""
    if not (len(sample_ids) == len(ideal_values) == len(supported) == len(noise) == len(dropout)):
        raise ValueError('Packet metadata lengths differ')
    rows = []
    for sid, distance, known, perturbation, missing in zip(sample_ids, ideal_values, supported, noise, dropout):
        valid = bool(known and not missing)
        value = float(distance + perturbation) if valid else None
        if valid and (not np.isfinite(value) or not MIN_RANGE-1e-9 <= value <= MAX_RANGE+1e-9):
            raise ValueError('Noisy bin center outside declared range')
        rows.append(dict(sample_id=sid, range_m=value, valid=valid,
                         diagonal_fov_deg=diagonal_fov, range_uncertainty_m=.1,
                         delta_ms=0, source=SOURCE))
    return rows


def generate(artifacts, output, batch_size=8):
    artifacts, output = Path(artifacts).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh output required')
    output.relative_to(artifacts)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for source packet generation')
    if not 1 <= batch_size <= 32:
        raise ValueError('Batch size must be1..32')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    jobs = [('original_fixture', 'body-query-background-only-20260909', 'admission-visible-head-v2'),
            ('simplified_fixture', 'body-query-fresh-size-20260909', 'admission-v1')]
    records, cohort_bindings = [], {}
    for cohort, task, admission_name in jobs:
        run = artifacts/'work'/task/'full-capture-v1'
        admission = run/admission_name
        receipt = read(admission/'admission.json')
        rows_path = admission/'frame_metadata.json'
        if sha(rows_path) != receipt['frame_metadata_sha256']:
            raise ValueError('Source metadata hash mismatch')
        cohort_bindings[cohort] = dict(admission_sha256=sha(admission/'admission.json'),
                                       metadata_sha256=sha(rows_path), source_status=receipt['status'])
        for row in read(rows_path):
            index = row['capture_index']
            path = run/row['region_id']/'capture/evaluator/native'/f'{index:04d}.npy'
            sid = hashlib.sha256(f"tof-hypothesis-v1:{cohort}:{row['region_id']}:{index}:{row['native_sha256']}".encode()).hexdigest()[:24]
            records.append(dict(sample_id=sid, cohort=cohort, sample_index=row['sample_index'],
                capture_index=index, region_id=row['region_id'], admitted=bool(row['admitted']),
                native_path=str(path), native_sha256=row['native_sha256']))
    if len(records) != 600 or len({r['sample_id'] for r in records}) != 600:
        raise ValueError('Expected600 unique source frames')
    noise, dropout = disturbances(len(records))
    footprint = footprints('cuda')
    result = {(fov, law): [] for fov in FOVS for law in LAWS}
    sidecar = []
    started = time.perf_counter()
    for begin in range(0, len(records), batch_size):
        sub = records[begin:begin+batch_size]
        frames = []
        for record in sub:
            path = Path(record['native_path'])
            if sha(path) != record['native_sha256']:
                raise ValueError('Native source hash mismatch')
            frame = np.load(path, allow_pickle=False)
            if frame.shape != (360, 640) or frame.dtype != np.float32:
                raise ValueError('Expected640x360float32 native axial depth')
            frames.append(frame)
        native = torch.from_numpy(np.stack(frames)).cuda()
        missing_by_fov = {}
        for fov in FOVS:
            laws = measure(native, footprint[fov])
            for law, (value, supported) in laws.items():
                result[fov, law].extend(packets([r['sample_id'] for r in sub], value.cpu().numpy(),
                    supported.cpu().numpy(), noise[begin:begin+len(sub)], dropout[begin:begin+len(sub)], fov))
                missing_by_fov[fov] = (~supported).cpu().tolist()
        for j, record in enumerate(sub):
            sidecar.append(dict(record, synthetic_noise_m=float(noise[begin+j]),
                synthetic_dropout=bool(dropout[begin+j]),
                no_supported_histogram_bin={str(fov):missing_by_fov[fov][j] for fov in FOVS}))
    torch.cuda.synchronize()
    output.mkdir(parents=True)
    bindings = {}
    for (fov, law), values in result.items():
        path = output/f'fov{fov}-{law}.jsonl'
        path.write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in values), encoding='utf-8')
        bindings[path.name] = sha(path)
    write(output/'evaluator-sidecar.json', sidecar)
    write(output/'receipt.json', dict(status='PASS', frames=600, packet_files=bindings,
        evaluator_sidecar_sha256=sha(output/'evaluator-sidecar.json'), source_sha256=sha(__file__),
        dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in ('body_query_tof_coverage.py','body_query_collection_labels.py')},
        cohort_bindings=cohort_bindings, backend='CUDA', device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-started, training_steps=0, model_inference_frames=0,
        law_parameters=dict(radial_min_m=.1, radial_max_m=4., bin_width_m=.1, minimum_full_footprint_fraction=.02,
            seed=17, common_uniform_noise_m=[-.05,.05], common_dropout_probability=.2, uncertainty_m=.1),
        scope='IDEAL SIMULATED HYPOTHESES ONLY; no real VL53L1X return, validity, reflectance or timing claim',
        limits='Invalid means no supported ideal histogram bin or synthetic dropout. Unknown native pixels contribute no mass; not observed free space or measured sensor invalidity. Same-frame synthetic timestamps, no trajectory.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    generate(args.artifacts, args.output, args.batch_size)
