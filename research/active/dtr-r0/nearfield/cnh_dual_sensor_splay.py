"""CPU geometry scan on the retained 48 synthetic no-cue head streams.

The head frame excludes the fixed sensor pitch: H=Ry(yaw). Each sensor is
H @ Ry(+/-splay) @ Rx(-10), matching the retained additive-yaw dual geometry.
This is finite-ray coverage only, without returns, occlusion or alarm inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import cnh_coverage_cue_geometry as G

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/cnh-coverage-cue-corrected-20261005'
OUT = ROOT / 'artifacts.local/work/cnh-dual-sensor-alarm-20261005/splay'
ANGLES = (10., 15., 20., 22.5)
HEIGHTS = (-.045, .265, .54)
RANGES = (.97, 1.13, 1.29)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Existing evidence must not be overwritten: {path}'
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def visible(yaw, points, splay):
    world = np.broadcast_to(points, (len(yaw), 3, len(points), 3)).copy()
    world[..., 2] = np.asarray(RANGES)[None, :, None]
    result = np.zeros(world.shape[:-1], dtype=bool)
    for angle in ((0.,) if splay == 0 else (-splay, splay)):
        rotation = G.rotation_yaw_pitch(yaw+angle, -10.)
        local = np.einsum('tqci,tij->tqcj', world, rotation)
        result |= G.visible_spheres(local, radius=.025)
    return result


def counts(yaw, points, splay):
    exposure = visible(yaw, points, splay)
    result = np.zeros((len(yaw), len(points)), dtype=np.int8)
    for q in range(3):
        result[q:] += exposure[:len(yaw)-q, q]
    return result


def run():
    tick = time.monotonic()
    assert not (OUT/'result.json').exists(), 'Result exists; inspect it instead of rerunning'
    plan = read(SOURCE/'PLAN.json')
    assert plan['sensor']['dt_s'] == .2
    points = np.array([[x, y, 0.] for x in plan['cells']['x_centers_m'] for y in HEIGHTS])
    centers = np.array([[0., y, 0.] for y in HEIGHTS])
    mask = (np.arange(600)>=50) & (np.arange(600)<590)
    paths = sorted((SOURCE/'E_grav/streams').glob('*.json'))
    assert len(paths) == 48
    rows = []
    parity = []
    for path in paths:
        source = read(path)
        ledger = path.with_suffix('.npz')
        assert sha(ledger) == source['ledger_sha256']
        with np.load(ledger) as z:
            yaw = z['true_yaw'][0].copy()
            np.testing.assert_array_equal(yaw, z['true_yaw'][1])
            retained_none = z['coverage_counts'][0].copy()
            retained_dual = z['coverage_counts'][1].copy()
            retained_zero = z['coverage_counts'][2].copy()
        single = counts(yaw, points, 0.)
        np.testing.assert_array_equal(single, retained_none)
        zero = counts(np.zeros(600), points, 0.)
        np.testing.assert_array_equal(zero, retained_zero)
        assert np.all(zero[mask] == 3)
        none_deficit = int((single[mask] < 3).sum())
        for angle in ANGLES:
            dual = counts(yaw, points, angle)
            if angle == 15:
                np.testing.assert_array_equal(dual, retained_dual)
                parity.append(dict(sigma=source['sigma'], tau=source['tau'], replica=source['replica'], differences=0))
            center = counts(yaw, centers, angle)
            rows.append(dict(sigma=source['sigma'], tau=source['tau'], replica=source['replica'],
                splay_deg=angle, cell_arrivals=int(mask.sum()*12), none_deficit_k3=none_deficit,
                dual_deficit_k3=int((dual[mask]<3).sum()),
                center_arrivals_per_height=int(mask.sum()),
                center_deficit_k3_by_height=(center[mask]<3).sum(0).tolist(),
                ledger=str(ledger.relative_to(ROOT)), ledger_sha256=source['ledger_sha256']))
        print('SPLAY STREAM', source['sigma'], source['tau'], source['replica'], flush=True)
    aggregates=[]
    for sigma in (5, 10, 15):
        for tau in (.5, 1., 2., 4.):
            for angle in ANGLES:
                rr=[r for r in rows if (r['sigma'],r['tau'],r['splay_deg'])==(sigma,tau,angle)]
                assert len(rr)==4
                den=sum(r['none_deficit_k3'] for r in rr)
                num=sum(r['dual_deficit_k3'] for r in rr)
                aggregates.append(dict(sigma=sigma,tau=tau,splay_deg=angle,K=4,
                    cell_arrivals=sum(r['cell_arrivals'] for r in rr),none_deficit_k3=den,
                    dual_deficit_k3=num,elimination_fraction=(den-num)/den,
                    residual_deficit_per_min=num/(4*1.8),
                    center_arrivals_per_height=4*540,
                    center_deficit_k3_by_height=np.sum([r['center_deficit_k3_by_height'] for r in rr],axis=0).tolist()))
    diagnostic=[]
    for angle in ANGLES:
        exposure=visible(np.array([0.]),centers,angle)[0]
        diagnostic.append(dict(splay_deg=angle,exposure_ranges_m=list(RANGES),
            visible_frame_count_by_height=exposure.sum(0).tolist(),
            each_center_k3=bool(np.all(exposure.sum(0)>=3))))
    mean={str(a):float(np.mean([r['elimination_fraction'] for r in aggregates if r['splay_deg']==a])) for a in ANGLES}
    baseline=mean['15.0']
    qualifying=[a for a in ANGLES if a!=15 and mean[str(a)]>=baseline+.05 and next(d for d in diagnostic if d['splay_deg']==a)['each_center_k3']]
    result=dict(status='COMPLETE',backend='TASK_NOT_GPU_SUITABLE; numpy CPU; no torch imports',
        scope='retained synthetic 48 streams; exact finite-ray geometric k3 coverage only',
        sensor_transform='H=old_sensor@Rx(+10)=Ry(yaw); sensor=H@Ry(+/-splay)@Rx(-10)',
        cells=points.tolist(),center_diagnostic_cells=centers.tolist(),height_order=list(HEIGHTS),
        window_m=[.9,1.4],radius_m=.025,dt_s=.2,evaluation_frames=[50,590],
        primary_splay_deg=15.,scan_degrees=list(ANGLES),streams=48,per_stream=rows,
        by_motion_cell=aggregates,front_facing_center_diagnostic=diagnostic,
        mean_elimination_fraction_over_12_motion_cells=mean,
        secondary_splay_qualifying_degrees=qualifying,
        secondary_rule='center k3 at yaw0 and unweighted12-cell mean elimination>=15deg+5pp',
        original_15deg_parity=dict(status='PASS',streams=48,coverage_count_differences=0,per_stream=parity),
        source_plan=str((SOURCE/'PLAN.json').relative_to(ROOT)),source_plan_sha256=sha(SOURCE/'PLAN.json'),
        source_sha256=sha(__file__),geometry_source_sha256=sha(G.__file__),seconds=time.monotonic()-tick)
    save(OUT/'result.json',result)
    lines=['| sigma | tau | ±10° | ±15° | ±20° | ±22.5° |','|---:|---:|---:|---:|---:|---:|']
    for sigma in (5,10,15):
        for tau in (.5,1.,2.,4.):
            rates=[next(r['elimination_fraction'] for r in aggregates if (r['sigma'],r['tau'],r['splay_deg'])==(sigma,tau,a)) for a in ANGLES]
            lines.append(f'| {sigma} | {tau:g} | '+ ' | '.join(f'{v*100:.2f}%' for v in rates)+' |')
    lines+=['','| splay | 12格均值 | 正视HEAD y=-.045 | 正视HEAD y=.265 | 正视BODY y=.54 |','|---:|---:|---:|---:|---:|']
    for d in diagnostic:
        a=d['splay_deg'];v=d['visible_frame_count_by_height']
        lines.append(f'| ±{a:g}° | {100*mean[str(a)]:.2f}% | {v[0]}/3 | {v[1]}/3 | {v[2]}/3 |')
    lines+=['',f'原±15°全部48流逐帧逐单元coverage_counts完全复现；扫描耗时{result["seconds"]:.2f}s。',
        f'次臂候选：{qualifying}；正视中心检查为独立3高度诊断，未混入12单元主分母。','']
    (OUT/'table.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],means=mean,secondary=qualifying,center=diagnostic),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.parse_args()
    run()
