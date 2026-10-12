"""Frozen M3 and old local fusion on each new observation and query branch."""
import json
import time

import numpy as np

import cnh_counterfactual_common_dev as C
import cnh_aligned_boundary_dev as B
import cnh_bar_local_readout_dev as L


def rotation(degrees):
    angle = np.deg2rad(degrees)
    co, si = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[co, 0, si], [0, 1, 0], [-si, 0, co]]
    return matrix


def load_observations(split):
    folder = C.OUT/'data'/split
    with np.load(folder/'physics.npz') as archive:
        return {key: archive[key] for key in ('hist', 'ambient', 'sensor', 'public_query')}


def run():
    output = C.OUT/'baselines'
    output.mkdir(exist_ok=True)
    if (output/'receipt.json').exists():
        receipt = C.read(output/'receipt.json')
        for name, digest in receipt['outputs_sha256'].items():
            if C.sha(output/name) != digest:
                raise ValueError('Completed baseline output changed')
        print('Completed frozen baselines retained', flush=True)
        return
    stage = C.Stage('frozen_baselines')
    engine = None
    try:
        A, _, _, normal = B.imports()
        A.OUT = C.OUT/'baseline_runtime'
        engine = A.Engine()
        qs, locations = L.patches()
        records, parities = [], []
        for split in ('cal', 'validation'):
            observation = load_observations(split)
            hist, ambient = observation['hist'], observation['ambient']
            nscene, replicas = hist.shape[:2]
            if hist.shape[2:] != (16, 8, 8, 16):
                raise ValueError('Unexpected coarse history layout')
            if ambient.shape == (16, 8, 8):
                ambient = np.broadcast_to(ambient, (nscene, replicas, 16, 8, 8))
            elif ambient.shape == (nscene, 16, 8, 8):
                ambient = np.broadcast_to(ambient[:, None], (nscene, replicas, 16, 8, 8))
            elif ambient.shape != (nscene, replicas, 16, 8, 8):
                raise ValueError('Unexpected ambient layout')
            z = normal(hist.reshape(-1, 16, 8, 8, 16), ambient.reshape(-1, 16, 8, 8))
            sensor, query = observation['sensor'], observation['public_query']
            if sensor.shape != (16, 4, 4) or query.shape != sensor.shape:
                raise ValueError('Expected common fixed pose schedule')
            for yaw in (0., 3.):
                name = f'{split}_'+('ideal' if yaw == 0 else 'yaw_plus3')
                destination = output/f'{name}.npz'
                if destination.exists():
                    # A prior interrupted stage keeps completed branches; same plan/data.
                    record = C.read(output/f'{name}.json')
                    if record['sha256'] != C.sha(destination):
                        raise ValueError('Saved baseline branch changed')
                    records.append(record)
                    continue
                stage.check()
                goal = rotation(yaw)[None]@query
                m3 = np.empty((len(z), 13, 2), np.float32)
                local = np.empty_like(m3, dtype=np.float64)
                for frame_index, frame in enumerate(range(3, 16)):
                    begin = max(0, frame-7)
                    transforms = (goal[frame]@np.linalg.inv(sensor[frame]))[None]@sensor[begin:frame+1]
                    projection = L.Projection(engine, transforms)
                    norms = []
                    for patch in qs:
                        mapped = patch@projection.p
                        variance = np.asarray(mapped.multiply(mapped).sum(1)).ravel()
                        norms.append(np.where(variance > 0, np.sqrt(variance), np.inf))
                    if frame in (3, 13):
                        actual = projection(z[:1, begin:frame+1])
                        reference = engine.project(z[:1, begin:frame+1], transforms[None])
                        np.testing.assert_array_equal(actual.cpu().numpy(), reference.cpu().numpy())
                        parities.append(dict(branch=name, frame=frame, max_abs=0.))
                    for offset in range(0, len(z), 32):
                        stage.check()
                        feature = projection(z[offset:offset+32, begin:frame+1]).cpu().numpy().astype(np.float16)
                        m3[offset:offset+32, frame_index] = engine.predict(feature)
                        local[offset:offset+32, frame_index] = L.scan(feature, qs, norms, locations)[0]
                    del projection
                    print('BASELINE', name, frame_index+1, '/13', round(time.monotonic()-stage.start, 2), flush=True)
                np.savez_compressed(destination, m3_raw=m3.reshape(nscene, replicas, 13, 2),
                                    local_raw=local.reshape(nscene, replicas, 13, 2),
                                    estimated_query=goal, sensor=sensor)
                record = dict(branch=name, sha256=C.sha(destination), scenes=nscene, replicas=replicas,
                              photons_sha256=C.sha(C.OUT/'data'/split/'physics.npz'),
                              plan_sha256=C.sha(C.OUT/'PLAN.json'))
                C.save(output/f'{name}.json', record)
                records.append(record)
            del z, hist, ambient, observation
        receipt = dict(status='COMPLETE', branches=records, projection_parity=parities,
                       backend='CUDA frozen 5-model M3, shared FP32 projection->FP16; CPU sparse local scan',
                       device=engine.torch.cuda.get_device_name(),
                       outputs_sha256={r['branch']+'.npz': r['sha256'] for r in records},
                       thresholds=dict(M3=.8557642486787612, old_fusion_M3=.9404184587540165,
                                       old_fusion_local=4.625390338985158))
        C.save(output/'receipt.json', receipt)
        stage.finish('COMPLETE', branches=len(records), source_sha256=C.sha(__file__))
    except BaseException as error:
        stage.finish('FAILED', error=repr(error), source_sha256=C.sha(__file__))
        raise
    finally:
        if engine is not None:
            engine.nets = []
            engine.projector = None
            engine.torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
