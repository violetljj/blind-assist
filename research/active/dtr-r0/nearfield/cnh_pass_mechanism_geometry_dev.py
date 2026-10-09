"""Full oriented target faces and solid overlap in existing query fixtures.

CPU only. Stored public_query is sensor->query; world->query is Q inv(S).
Six complete faces are polygon-clipped, never replaced by a rotated AABB.
Surface intersection and solid overlap are separate (containment fixture).
Artificial +3 query geometry does not relabel original physical-body truth.
"""
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUT = ROOT/'artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009'
DEST = OUT/'geometry'
FACES = ((0,1,3,2), (4,6,7,5), (0,4,5,1), (2,3,7,6), (0,2,6,4), (1,5,7,3))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def literal(path, name):
    for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def corners(lo, hi):
    return np.asarray(list(itertools.product(*zip(lo, hi))), float)


def rotation(degrees):
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    r = np.eye(4); r[:3, :3] = [[c,0,s], [0,1,0], [-s,0,c]]
    return r


def clip(poly, axis, bound, lower):
    if not len(poly):
        return poly
    result = []
    previous = poly[-1]
    previous_inside = previous[axis] >= bound if lower else previous[axis] <= bound
    for current in poly:
        inside = current[axis] >= bound if lower else current[axis] <= bound
        if inside != previous_inside:
            fraction = (bound-previous[axis])/(current[axis]-previous[axis])
            point = previous+fraction*(current-previous)
            point[axis] = bound
            result.append(point)
        if inside:
            result.append(current.copy())
        previous, previous_inside = current, inside
    return np.asarray(result, float).reshape(-1,3)


def area(poly):
    if len(poly) < 3:
        return 0.
    centered = poly-poly.mean(0)
    return float(np.linalg.norm(np.cross(centered, np.roll(centered,-1,axis=0)).sum(0))*.5)


def clipped_faces(points, lo, hi):
    polys, areas = [], []
    for indices in FACES:
        poly = points[list(indices)].copy()
        for axis in range(3):
            poly = clip(poly, axis, lo[axis], True)
            poly = clip(poly, axis, hi[axis], False)
        polys.append(poly.tolist()); areas.append(area(poly))
    return polys, areas


def solid_overlap(box_lo, box_hi, transform, qlo, qhi, *, closed=False):
    """15-axis OBB/AABB SAT; open excludes zero-volume tangencies."""
    r = transform[:3,:3]
    tc = r@((box_lo+box_hi)/2)+transform[:3,3]
    th = (box_hi-box_lo)/2
    qc, qh = (qlo+qhi)/2, (qhi-qlo)/2
    basis = np.eye(3)
    axes = [*basis, *r.T, *[np.cross(a,b) for a in basis for b in r.T]]
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axis = axis/norm
        span = np.dot(np.abs(r.T@axis), th)+np.dot(np.abs(axis), qh)
        distance = abs(float(np.dot(axis, tc-qc)))
        if distance > span if closed else distance >= span:
            return False
    return True


def inspect(box, transform, query_box, eps, expanded=False):
    lo, hi = np.asarray(box['lo']), np.asarray(box['hi'])
    pts = corners(lo,hi)@transform[:3,:3].T+transform[:3,3]
    qlo, qhi = np.asarray(query_box[:3],float).copy(), np.asarray(query_box[3:],float).copy()
    if expanded:
        qlo[0], qhi[0] = -.4, .4
        qlo[1:] += eps; qhi[1:] -= eps
    else:
        qlo += eps; qhi -= eps
    polygons, areas = clipped_faces(pts,qlo,qhi)
    total_area = float(sum(area(pts[list(face)]) for face in FACES))
    qpoints = corners(qlo,qhi)
    world_qpoints = (qpoints-transform[:3,3])@transform[:3,:3]
    return dict(corners=pts, polygons=polygons, face_areas=areas,
        area=float(sum(areas)), full_target_area=total_area,
        surface_has_any=bool(sum(areas)>0),
        solid_interior_overlap=solid_overlap(lo,hi,transform,qlo,qhi),
        solid_closed_overlap=solid_overlap(lo,hi,transform,qlo,qhi,closed=True),
        target_entirely_in_query=bool(((pts>=qlo)&(pts<=qhi)).all()),
        query_entirely_in_target=bool(((world_qpoints>=lo)&(world_qpoints<=hi)).all()))


def fixtures(query, eps):
    cross = dict(lo=[.29,-.1,.7], hi=[.6,.1,.8])
    value = inspect(cross,np.eye(4),query,eps)
    np.testing.assert_allclose(value['area'], .026-.6*eps, atol=1e-12, rtol=0)
    assert value['solid_interior_overlap'] and value['surface_has_any']
    tangent = inspect(dict(lo=[.3,-.1,.7],hi=[.4,.1,.8]),np.eye(4),query,eps)
    assert not tangent['surface_has_any'] and not tangent['solid_interior_overlap']
    outside = inspect(dict(lo=[.5,-.1,.7],hi=[.6,.1,.8]),np.eye(4),query,eps)
    assert not outside['surface_has_any'] and not outside['solid_interior_overlap']
    contained = inspect(dict(lo=[-1.,-1.,0.],hi=[1.,1.,4.]),np.eye(4),query,eps)
    assert contained['query_entirely_in_target'] and contained['solid_interior_overlap']
    assert not contained['surface_has_any'] and contained['area'] == 0.
    tilted = rotation(45.); tilted[:3,3] = [1.3,.5,1.3]
    aabb_false = inspect(dict(lo=[-.7,-.3,-.025],hi=[.7,.3,.025]),
                         tilted, (0.,0.,0.,1.,1.,1.),0.)
    assert ((aabb_false['corners'].max(0)>0)&(aabb_false['corners'].min(0)<1)).all()
    assert not aabb_false['solid_interior_overlap'] and not aabb_false['surface_has_any']
    return dict(status='PASS', checks=['six-face analytical area', 'strict tangent excluded',
        'disjoint target', 'query-inside-target solid positive / target-face area zero',
        'rotated AABB overlap false positive rejected by polygon/SAT'])


def run():
    started = time.monotonic()
    DEST.mkdir(parents=True,exist_ok=True)
    receipts = OUT/'stage_receipts'; receipts.mkdir(exist_ok=True)
    receipt_path = receipts/f'geometry_{time.time_ns()}.json'
    plan = read(OUT/'PLAN.json'); cap = plan['allocations']['geometry_cpu']
    digest = sha(__file__)
    snapshot = DEST/f'source_snapshot_{digest[:12]}.py'
    if snapshot.exists():
        raise FileExistsError('Preserve previous geometry diagnostic')
    snapshot.write_bytes(Path(__file__).read_bytes())
    def check():
        if time.monotonic()-started > cap:
            raise TimeoutError('Geometry CPU allocation reached')
    try:
        model = ROOT/'research/active/dtr-r0/nearfield/cnh_boundary_token_model.py'
        data = ROOT/'research/active/dtr-r0/nearfield/cnh_counterfactual_data_dev.py'
        queries, eps = literal(model,'QUERY_BOXES'), literal(data,'EPS')
        assert eps == 1e-8 and queries[0][0] == queries[1][0] == -.3
        assert queries[0][3] == queries[1][3] == .3
        fixture = fixtures(queries[0],eps)
        rows_path = SOURCE/'scene_rows.json'
        rows = read(rows_path)['validation']
        folder = SOURCE/'data/validation'
        with np.load(folder/'geometry.npz',allow_pickle=False) as archive:
            sensor, query, category = (archive[k] for k in ('sensor','public_query','category'))
        w = query@np.linalg.inv(sensor)
        expected = np.repeat(np.eye(4)[None],16,axis=0); expected[:,:3,3] = -sensor[:,:3,3]
        np.testing.assert_allclose(w,expected,atol=1e-12,rtol=0)
        yaw = rotation(3.)@w
        assert np.allclose(yaw[:,0,1],0.,atol=1e-12)
        keep = [r for r in rows if r['placement'] == 'in1cm' and category[r['scene_id'],r['group']] == 'contact']
        assert len(keep) == 64 and sum(r['group']==0 for r in keep) == 32
        frames, summaries = [], []
        for row in keep:
            check()
            q = row['group']; height = ('HEAD','BODY')[q]
            for branch, transforms in (('ideal',w),('yaw_plus3',yaw)):
                current = []
                for f in plan['timely_frames']:
                    contact = inspect(row['target_box'],transforms[f],queries[q],eps)
                    passed = inspect(row['target_box'],transforms[f],queries[q],eps,True)
                    pts = contact['corners']; negative = row['side'] == -1
                    entry = dict(scene_id=row['scene_id'],height=height,side=('negative_x' if negative else 'positive_x'),
                        side_sign=row['side'], branch=branch,frame=f, background_id=row['background_id'],
                        background_family=row['background_family'],shape_family=row['shape_family'],rho=row['rho'],
                        target_id=row['target_id'],original_body_truth='contact',
                        target_front_z_distance_m=float(row['target_box']['lo'][2]-sensor[f,2,3]),
                        query_x_min_m=float(pts[:,0].min()),query_x_max_m=float(pts[:,0].max()),
                        query_x_near_center_m=float(pts[:,0].max() if negative else pts[:,0].min()),
                        query_x_far_center_m=float(pts[:,0].min() if negative else pts[:,0].max()),
                        query_y_min_m=float(pts[:,1].min()),query_y_max_m=float(pts[:,1].max()),
                        query_z_min_m=float(pts[:,2].min()),query_z_max_m=float(pts[:,2].max()),
                        contact_surface_area_m2=contact['area'],full_target_surface_area_m2=contact['full_target_area'],
                        contact_surface_area_fraction=contact['area']/contact['full_target_area'],
                        contact_surface_has_any=int(contact['surface_has_any']),
                        contact_solid_interior_overlap=int(contact['solid_interior_overlap']),
                        contact_solid_closed_overlap=int(contact['solid_closed_overlap']),
                        target_entirely_in_contact_query=int(contact['target_entirely_in_query']),
                        contact_query_entirely_in_target=int(contact['query_entirely_in_target']),
                        pass_expanded_surface_area_m2=passed['area'],pass_expanded_surface_has_any=int(passed['surface_has_any']),
                        pass_expanded_solid_interior_overlap=int(passed['solid_interior_overlap']),
                        pass_expanded_solid_closed_overlap=int(passed['solid_closed_overlap']),
                        corners_query_json=json.dumps(pts.tolist(),separators=(',',':')),
                        contact_face_areas_m2_json=json.dumps(contact['face_areas'],separators=(',',':')),
                        contact_clipped_face_polygons_json=json.dumps(contact['polygons'],separators=(',',':')),
                        pass_expanded_face_areas_m2_json=json.dumps(passed['face_areas'],separators=(',',':')))
                    frames.append(entry); current.append(entry)
                    if branch == 'ideal':
                        assert contact['surface_has_any'] and contact['solid_interior_overlap']
                summaries.append(dict(scene_id=row['scene_id'],height=height,side=current[0]['side'],branch=branch,
                    timely_frame_denominator=11,contact_surface_frames=sum(e['contact_surface_has_any'] for e in current),
                    contact_solid_frames=sum(e['contact_solid_interior_overlap'] for e in current),
                    pass_expanded_surface_frames=sum(e['pass_expanded_surface_has_any'] for e in current),
                    contact_surface_any_timely=int(any(e['contact_surface_has_any'] for e in current)),
                    contact_solid_any_timely=int(any(e['contact_solid_interior_overlap'] for e in current)),
                    contact_surface_area_min_m2=min(e['contact_surface_area_m2'] for e in current),
                    contact_surface_area_max_m2=max(e['contact_surface_area_m2'] for e in current),
                    first_contact_surface_frame=next((e['frame'] for e in current if e['contact_surface_has_any']),None),
                    last_contact_surface_frame=next((e['frame'] for e in reversed(current) if e['contact_surface_has_any']),None),
                    near_center_x_f10_m=next(e['query_x_near_center_m'] for e in current if e['frame']==10)))
        assert len(frames)==1408 and len(summaries)==128
        for name, records in (('per_frame_geometry.csv',frames),('scene_geometry_summary.csv',summaries)):
            with (DEST/name).open('x',encoding='utf8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        result = dict(status='PASS',source_sha256=digest,fixtures=fixture,physical_scenes=64,
            per_height_scenes=32,frame_rows=len(frames),scene_branch_rows=len(summaries),
            query_boxes=queries,epsilon_m=eps,face_order=['xlow','xhigh','ylow','yhigh','zlow','zhigh'],
            surface_definition='Area of original six target faces clipped to exact query; includes all faces, not only visible faces. Not solid-intersection boundary area or volume.',
            solid_definition='15-axis OBB/AABB separating-axis test; interior means positive-volume overlap, closed includes touching. Query shrink uses EPS as evaluator contact/pass policy.',
            containment='Query-inside-target fixture has positive solid overlap and zero target-face surface area; never infer empty solid solely from zero face area.',
            metadata=dict(stored_public_query='sensor->query',world_to_query='public_query[f]@inv(sensor[f])',
                yaw_query='Ry(+3)@world_to_query',shared_transform_for_HEAD_BODY=True,
                shared_x_bounds=[-.3,.3],query_x_y_mixing_max_abs=float(np.max(np.abs(yaw[:,0,1]))),
                interpretation='These static fixtures have no height-specific x transform; body/head query y intervals differ. This does not validate physical pose, time or external hardware contracts.'),
            summaries=summaries,inputs_sha256={str(rows_path):sha(rows_path),str(folder/'geometry.npz'):sha(folder/'geometry.npz'),
                str(model):sha(model),str(data):sha(data)},
            limits='Artificial offset query only. Original physical-body contact truth remains unchanged even if target leaves changed query; no verdict on supervision value or model causality. No new photons/inference/GPU.',
            history_scope='Per-frame current world->query target geometry only; no claim that all past observations lose target evidence or leave sensor FOV.')
        save(DEST/'geometry_summary.json',result)
        check()
        save(receipt_path,dict(stage='geometry',status='PASS',seconds=time.monotonic()-started,cpu_only=True,
            source_sha256=digest,plan_sha256=sha(OUT/'PLAN.json'),outputs=str(DEST)))
        print(json.dumps(dict(status='PASS',seconds=time.monotonic()-started,frame_rows=len(frames))))
    except BaseException as exc:
        save(receipt_path,dict(stage='geometry',status='FAILED',seconds=time.monotonic()-started,cpu_only=True,
            source_sha256=digest,error=repr(exc)))
        raise


if __name__=='__main__':
    run()
