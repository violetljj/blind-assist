"""Measured three-view vertical-line geometry; NumPy only, no depth model.

CLI: python measured_pole_geometry.py --input measurement.json --out NEW_DIR
Missing/invalid measurements retain a report and snapshot but exit with code 2.
Other complete diagnostics exit 0; callers must inspect the result status.
Views 0 and 1 reconstruct; view 2 only validates. K and pixels must refer to
the same undistorted upright image. Camera axes are OpenCV x-right/y-down/
z-forward in the physical camera basis. Each view explicitly maps normalized
image rays to that basis using image_ray_to_camera (orthogonal, det +/-1).
This map must fix the physical +Z optical axis; only image-XY basis changes
belong here. It must not turn a physically rear-facing ray into a front ray.
World is right handed, with an explicit unit upward direction; the example
uses X-right/Y-up/Z-backward, and its reference camera looks toward world -Z.

The returned anchor is the intersection of the infinite vertical pole axis
with dot(world_vertical, X)=0. It is not a visible foot or contact point.
Successful third-view consistency never sets verified_distance to true.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = 'measured-pole-geometry-v1'
NUMERICAL_CONDITION_LIMIT = 1e8


def validate_document(doc):
    missing, invalid = [], []
    def value(parent, key, path):
        v = parent.get(key) if isinstance(parent, dict) else None
        if v is None:
            missing.append(path)
        return v
    def text(parent, key, path, choices=None):
        v = value(parent, key, path)
        if v is not None and (not isinstance(v, str) or not v.strip() or (choices and v not in choices)):
            invalid.append(path)
        return v
    def number(parent, key, path, strictly_positive=False):
        v = value(parent, key, path)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or
                              not math.isfinite(v) or v < 0 or (strictly_positive and v == 0)):
            invalid.append(path)
        return v
    def array(parent, key, path, shape):
        v = value(parent, key, path)
        if v is None:
            return None
        try:
            a = np.asarray(v, float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError
            return a
        except (TypeError, ValueError):
            invalid.append(path)
            return None
    if not isinstance(doc, dict):
        return [], ['document must be an object']
    text(doc, 'schema_version', 'schema_version', (VERSION,))
    text(doc, 'evidence_role', 'evidence_role', ('MEASURED_INPUT', 'SYNTHETIC_EXAMPLE'))
    frame = doc.get('coordinate_frame', {})
    text(frame, 'world_handedness', 'coordinate_frame.world_handedness', ('right',))
    text(frame, 'camera_axes', 'coordinate_frame.camera_axes', ('opencv_x_right_y_down_z_forward',))
    text(frame, 'units', 'coordinate_frame.units', ('metre',))
    text(frame, 'world_origin_note', 'coordinate_frame.world_origin_note')
    g = array(frame, 'world_vertical_unit', 'coordinate_frame.world_vertical_unit', (3,))
    if g is not None and not np.isclose(np.linalg.norm(g), 1., atol=1e-6):
        invalid.append('coordinate_frame.world_vertical_unit must have unit length')
    measurement = doc.get('measurement', {})
    for key in ('measurement_id', 'measured_by', 'note'):
        text(measurement, key, f'measurement.{key}')
    for key in ('position_error_scenario_m', 'rotation_error_scenario_deg', 'endpoint_error_scenario_px'):
        number(measurement, key, f'measurement.{key}')
    criteria = doc.get('criteria', {})
    for key in ('max_vertical_line_residual_px', 'max_third_view_residual_px'):
        number(criteria, key, f'criteria.{key}')
    views = value(doc, 'views', 'views')
    if not isinstance(views, list) or len(views) != 3:
        if views is not None:
            invalid.append('views must contain exactly 2 FIT views followed by 1 VALIDATION view')
        return sorted(set(missing)), sorted(set(invalid))
    ids, correspondences = [], []
    for i, view in enumerate(views):
        prefix = f'views[{i}]'
        ids.append(text(view, 'view_id', prefix + '.view_id'))
        text(view, 'role', prefix + '.role', ('FIT',) if i < 2 else ('VALIDATION',))
        text(view, 'measurement_note', prefix + '.measurement_note')
        im = view.get('image', {}) if isinstance(view, dict) else {}
        for key in ('path', 'calibration_ref', 'undistortion_note'):
            text(im, key, prefix + '.image.' + key)
        text(im, 'pixel_domain', prefix + '.image.pixel_domain',
             ('undistorted_upright', 'atom_undistorted_upright_transpose', 'synthetic_pinhole'))
        width = number(im, 'width', prefix + '.image.width', True)
        height = number(im, 'height', prefix + '.image.height', True)
        for key, v in (('width', width), ('height', height)):
            if isinstance(v, (int, float)) and math.isfinite(v) and v != int(v):
                invalid.append(prefix + '.image.' + key + ' must be an integer')
        K = array(view, 'K', prefix + '.K', (3, 3))
        R = array(view, 'camera_to_world_R', prefix + '.camera_to_world_R', (3, 3))
        Q = array(view, 'image_ray_to_camera', prefix + '.image_ray_to_camera', (3, 3))
        array(view, 'camera_center_m', prefix + '.camera_center_m', (3,))
        ends = array(view, 'line_endpoints_px', prefix + '.line_endpoints_px', (2, 2))
        if K is not None and (abs(np.linalg.det(K)) < 1e-12 or K[0, 0] <= 0 or K[1, 1] <= 0 or
                              not np.allclose(K[2], [0, 0, 1], atol=1e-9)):
            invalid.append(prefix + '.K must be a nonsingular pinhole calibration with positive focal lengths')
        if R is not None and (not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or
                              not np.isclose(np.linalg.det(R), 1., atol=1e-5)):
            invalid.append(prefix + '.camera_to_world_R must be a proper rotation, not a reflection')
        if Q is not None and (not np.allclose(Q.T @ Q, np.eye(3), atol=1e-5) or
                              not np.isclose(abs(np.linalg.det(Q)), 1., atol=1e-5)):
            invalid.append(prefix + '.image_ray_to_camera must be orthogonal with determinant +1 or -1')
        if Q is not None and not np.allclose(Q[:, 2], [0., 0., 1.], atol=1e-5):
            invalid.append(prefix + '.image_ray_to_camera must preserve the physical +Z optical axis (Q e3 = e3)')
        if ends is not None:
            if np.linalg.norm(ends[1] - ends[0]) < 1e-6:
                invalid.append(prefix + '.line_endpoints_px has zero length')
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                if np.any(ends < 0) or np.any(ends >= [width, height]):
                    invalid.append(prefix + '.line_endpoints_px is outside the declared image')
        correspondence = view.get('correspondence', {}) if isinstance(view, dict) else {}
        correspondences.append(tuple(text(correspondence, key, prefix + '.correspondence.' + key)
                                     for key in ('physical_line_id', 'edge_definition')))
    if all(isinstance(x, str) for x in ids) and len(set(ids)) != 3:
        invalid.append('views.view_id must be unique')
    if all(all(isinstance(v, str) for v in c) for c in correspondences) and len(set(correspondences)) != 1:
        invalid.append('All three views must declare the same physical line and physical edge/centerline')
    return sorted(set(missing)), sorted(set(invalid))


def homogeneous(xy):
    return np.column_stack((np.asarray(xy, float), np.ones(len(xy))))


def prepared(view, g):
    K, R = np.asarray(view['K'], float), np.asarray(view['camera_to_world_R'], float)
    Q = np.asarray(view['image_ray_to_camera'], float)
    C, ends = np.asarray(view['camera_center_m'], float), np.asarray(view['line_endpoints_px'], float)
    pixel_points = homogeneous(ends)
    directions = np.linalg.solve(K, pixel_points.T).T @ Q.T @ R.T
    midpoint = directions.mean(axis=0)
    normal = np.cross(g, midpoint)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    # The admitted vertical line passes through the observed image midpoint.
    # Its endpoint residual records any departure from the known gravity direction.
    expected_line = np.linalg.solve(K.T, Q.T @ R.T @ normal)
    denom = np.linalg.norm(expected_line[:2])
    if denom < 1e-12:
        return None
    expected_line /= denom
    residual = np.abs(pixel_points @ expected_line)
    return dict(K=K, R=R, C=C, ends=ends, directions=directions, plane_normal=normal,
                vertical_residual_px=float(residual.max()))


def project_axis(anchor, g, view):
    K, R, C = (np.asarray(view[k], float) for k in ('K', 'camera_to_world_R', 'camera_center_m'))
    Q = np.asarray(view['image_ray_to_camera'], float)
    image_from_world = Q.T @ R.T
    n = np.cross(image_from_world @ (anchor - C), image_from_world @ g)
    line = np.linalg.solve(K.T, n)
    norm = np.linalg.norm(line[:2])
    if norm < 1e-12:
        return dict(status='PROJECTION_AT_VANISHING_POINT', line_abc=None)
    line /= norm
    points = homogeneous(view['line_endpoints_px'])
    distances = np.abs(points @ line)
    directions = np.linalg.solve(K, points.T).T @ Q.T @ R.T
    camera_z, miss = [], []
    for d in directions:
        solution, _, _, _ = np.linalg.lstsq(np.column_stack((d, -g)), anchor - C, rcond=None)
        camera_z.append(float(solution[0]))
        miss.append(float(np.linalg.norm(C + solution[0] * d - (anchor + solution[1] * g))))
    return dict(status='PROJECTED', line_abc=line.tolist(), endpoint_residual_px=distances.tolist(),
                max_endpoint_residual_px=float(distances.max()), rms_endpoint_residual_px=float(np.sqrt(np.mean(distances**2))),
                endpoint_image_ray_z_m=camera_z, endpoint_ray_to_axis_miss_m=miss,
                all_observed_endpoints_in_front=all(x > 1e-9 for x in camera_z))


def reconstruct_pair(views, g, max_vertical_residual_px, enforce_vertical=True):
    result = dict(status='UNKNOWN', anchor_world_m=None, verified_distance=False,
                  fit_view_ids=[v['view_id'] for v in views], fit_projections=[])
    centers = [np.asarray(v['camera_center_m'], float) for v in views]
    baseline = centers[1] - centers[0]
    length = float(np.linalg.norm(baseline))
    horizontal = float(np.linalg.norm(baseline - g * (g @ baseline)))
    result.update(baseline_m=length, horizontal_baseline_m=horizontal)
    if length <= 1e-9:
        result['status'] = 'ZERO_BASELINE_OR_PURE_ROTATION'
        return result
    if horizontal <= 1e-9:
        result['status'] = 'NO_HORIZONTAL_BASELINE'
        return result
    p = [prepared(v, g) for v in views]
    if any(x is None for x in p):
        result['status'] = 'LINE_AT_VERTICAL_VANISHING_POINT'
        return result
    residuals = [x['vertical_residual_px'] for x in p]
    result['input_vertical_residual_px'] = residuals
    incompatible = any(x > max_vertical_residual_px + 1e-9 for x in residuals)
    if enforce_vertical and incompatible:
        result['status'] = 'NON_VERTICAL_FIT_LINE'
        return result
    A = np.vstack((p[0]['plane_normal'], p[1]['plane_normal'], g))
    condition = float(np.linalg.cond(A))
    result['plane_system_condition'] = condition if np.isfinite(condition) else None
    if not np.isfinite(condition) or condition > NUMERICAL_CONDITION_LIMIT:
        result['status'] = 'DEGENERATE_LINE_PLANES'
        return result
    b = [p[0]['plane_normal'] @ p[0]['C'], p[1]['plane_normal'] @ p[1]['C'], 0.]
    anchor = np.linalg.solve(A, b)
    projections = [project_axis(anchor, g, view) for view in views]
    result.update(anchor_world_m=anchor.tolist(), fit_projections=projections,
                  vertical_residual_conflict=incompatible,
                  horizontal_distance_to_axis_m=[float(np.linalg.norm((anchor - C) - g * (g @ (anchor - C)))) for C in centers])
    if any(r['status'] != 'PROJECTED' or not r['all_observed_endpoints_in_front'] for r in projections):
        result['status'] = 'BEHIND_CAMERA_OR_UNPROJECTABLE'
    else:
        result['status'] = 'RECONSTRUCTED'
    if np.allclose(g, [0, 1, 0], atol=1e-9):
        result['world_x_z_m_at_y0'] = anchor[[0, 2]].tolist()
    return result


def axis_rotation(axis, angle):
    v = np.eye(3)[axis]
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def scenario_envelope(fit_views, g, measurement, threshold):
    variants = [('nominal', copy.deepcopy(fit_views))]
    pos, rotation, pixel = (measurement[k] for k in ('position_error_scenario_m', 'rotation_error_scenario_deg', 'endpoint_error_scenario_px'))
    for i in range(2):
        for sign in (-1, 1):
            for axis in range(3):
                if pos:
                    v = copy.deepcopy(fit_views)
                    v[i]['camera_center_m'][axis] += sign * pos
                    variants.append((f'view{i}_center_axis{axis}_{sign:+d}', v))
                if rotation:
                    v = copy.deepcopy(fit_views)
                    v[i]['camera_to_world_R'] = (axis_rotation(axis, math.radians(sign * rotation)) @ np.asarray(v[i]['camera_to_world_R'])).tolist()
                    variants.append((f'view{i}_rotation_world_axis{axis}_{sign:+d}', v))
            if pixel:
                for endpoint in range(2):
                    for axis in range(2):
                        v = copy.deepcopy(fit_views)
                        v[i]['line_endpoints_px'][endpoint][axis] += sign * pixel
                        variants.append((f'view{i}_endpoint{endpoint}_axis{axis}_{sign:+d}', v))
                v = copy.deepcopy(fit_views); e = np.asarray(v[i]['line_endpoints_px'], float)
                delta = e[1] - e[0]; normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
                v[i]['line_endpoints_px'] = (e + sign * pixel * normal).tolist()
                variants.append((f'view{i}_rigid_line_normal_{sign:+d}', v))
    rows = []
    for name, views in variants:
        r = reconstruct_pair(views, g, threshold, enforce_vertical=False)
        rows.append(dict(scenario=name, status=r['status'], anchor_world_m=r['anchor_world_m'],
                         vertical_residual_conflict=r.get('vertical_residual_conflict'),
                         horizontal_distance_to_axis_m=r.get('horizontal_distance_to_axis_m')))
    front = [r for r in rows if r['status'] == 'RECONSTRUCTED']
    values = np.asarray([r['anchor_world_m'] for r in front])
    return dict(name='SAMPLED_SCENARIO_ENVELOPE', fit_views_only=True, third_view_used=False,
                scenario_count=len(rows), finite_front_scenarios=len(front),
                anchor_world_coordinate_minmax_m=[values.min(axis=0).tolist(), values.max(axis=0).tolist()] if len(values) else None,
                scenarios=rows,
                coverage='Nominal plus one fit-view position axis, one world rotation axis, one endpoint coordinate, or one rigid image-line normal shift at a time.',
                omitted='No joint multi-axis or multi-view error corners, no K/distortion error, no gravity-vector error, no pixel correspondence truth, no moving-pole model; not a guaranteed bound or confidence interval.')


def analyse(doc):
    missing, invalid = validate_document(doc)
    report = dict(schema_version=VERSION, evidence_role=doc.get('evidence_role') if isinstance(doc, dict) else None,
                  status='MISSING_MEASUREMENT' if missing else 'INVALID_MEASUREMENT' if invalid else 'UNKNOWN',
                  missing_fields=missing, invalid_fields=invalid, verified_distance=False,
                  measurement_provenance_verified=False, accuracy_claim=None, safety_decision=None,
                  reconstruction=None, validation=None, scenario_envelope=None)
    if missing or invalid:
        return report
    views = doc['views']; g = np.asarray(doc['coordinate_frame']['world_vertical_unit'], float)
    criteria = doc['criteria']
    fitted = reconstruct_pair(views[:2], g, criteria['max_vertical_line_residual_px'])
    report.update(status=fitted['status'], reconstruction=fitted,
                  method='Intersect two gravity-aligned ray planes through observed line midpoints; reject incompatible vertical residual before reconstruction.',
                  anchor_definition='Pole-axis point satisfying dot(world_vertical_unit, anchor_world_m)=0; not an observed foot.',
                  units='metre', measurement_id=doc['measurement']['measurement_id'],
                  camera_height_used=False, world_vertical_unit=g.tolist())
    if fitted['status'] != 'RECONSTRUCTED':
        return report
    anchor = np.asarray(fitted['anchor_world_m'])
    held = project_axis(anchor, g, views[2]); prepared_held = prepared(views[2], g)
    vertical_residual = prepared_held['vertical_residual_px'] if prepared_held else None
    projection_ok = held['status'] == 'PROJECTED' and held['all_observed_endpoints_in_front']
    residual_ok = projection_ok and held['max_endpoint_residual_px'] <= criteria['max_third_view_residual_px'] + 1e-9
    vertical_ok = vertical_residual is not None and vertical_residual <= criteria['max_vertical_line_residual_px'] + 1e-9
    passed = bool(residual_ok and vertical_ok)
    report['validation'] = dict(view_id=views[2]['view_id'], role='HELD_OUT_PROJECTION_ONLY',
                                status='CONSISTENT_WITH_DECLARED_TOLERANCES' if passed else 'VALIDATION_CONFLICT',
                                contributed_to_reconstruction=False, contributed_to_scenario_envelope=False,
                                input_vertical_residual_px=vertical_residual, projected_axis=held,
                                declared_max_residual_px=criteria['max_third_view_residual_px'],
                                note='Projection consistency is not metric accuracy or verified physical correspondence.')
    report['status'] = 'RECONSTRUCTED_VALIDATION_CONSISTENT' if passed else 'RECONSTRUCTED_VALIDATION_CONFLICT'
    report['scenario_envelope'] = scenario_envelope(views[:2], g, doc['measurement'], criteria['max_vertical_line_residual_px'])
    return report


def synthetic_example():
    K = np.array([[400., 0., 240.], [0., 400., 320.], [0., 0., 1.]])
    R = np.diag([1., -1., -1.])  # Proper rotation: reference camera looks along world -Z.
    pole = np.array([[.3, .3, -2.5], [.3, 1.4, -2.5]])
    views = []
    for view_id, x, role in [('baseline_0cm', 0., 'FIT'), ('baseline_20cm', .2, 'FIT'), ('validation_10cm', .1, 'VALIDATION')]:
        C = np.array([x, 1.5, 0.]); camera = (pole - C) @ R
        pixels = camera @ K.T; pixels = pixels[:, :2] / pixels[:, 2, None]
        views.append(dict(view_id=view_id, role=role,
                          image=dict(path=f'synthetic://{view_id}', width=480, height=640, pixel_domain='synthetic_pinhole',
                                     calibration_ref='synthetic-pinhole-400px', undistortion_note='Synthetic exact pinhole, no lens distortion'),
                          K=K.tolist(), camera_to_world_R=R.tolist(), camera_center_m=C.tolist(),
                          image_ray_to_camera=np.eye(3).tolist(),
                          line_endpoints_px=pixels.tolist(), correspondence=dict(physical_line_id='synthetic-pole-1',edge_definition='same exact centerline'),
                          measurement_note='Synthetic generated camera and line; not device evidence'))
    return dict(schema_version=VERSION, evidence_role='SYNTHETIC_EXAMPLE',
                coordinate_frame=dict(world_handedness='right',world_vertical_unit=[0.,1.,0.],
                                      world_origin_note='Synthetic floor origin; camera at y=1.5 m and facing world -Z',
                                      units='metre',camera_axes='opencv_x_right_y_down_z_forward'),
                measurement=dict(measurement_id='synthetic-three-view-pole',measured_by='synthetic_generator',
                                 note='Known geometry example only; nonzero error scenarios are illustrative perturbations, not measured uncertainty.',
                                 position_error_scenario_m=.003,rotation_error_scenario_deg=.25,endpoint_error_scenario_px=1.),
                criteria=dict(max_vertical_line_residual_px=1.,max_third_view_residual_px=2.),views=views,
                synthetic_truth=dict(pole_axis_anchor_world_m=[.3,0.,-2.5],role='Example/test generator only; analyse does not read this field'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='New output directory; existing directories are rejected')
    args = parser.parse_args(); start = time.perf_counter(); cpu = time.process_time()
    try:
        input_bytes = args.input.read_bytes(); doc = json.loads(input_bytes.decode('utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.out.exists():
        parser.error(f'Output already exists; preserve it and choose a new --out: {args.out}')
    result = analyse(doc)
    result.update(input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                  input_path=str(args.input.resolve()),tool_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  generated_utc=datetime.now(timezone.utc).isoformat(),process_cpu_s=time.process_time()-cpu,
                  wall_s=time.perf_counter()-start)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'input_snapshot.json').write_bytes(input_bytes)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(status=result['status'],verified_distance=False,output=str((args.out/'result.json').resolve()),
                          missing_fields=result['missing_fields'],invalid_fields=result['invalid_fields'])))
    return 2 if result['status'] in ('MISSING_MEASUREMENT', 'INVALID_MEASUREMENT') else 0


if __name__ == '__main__':
    raise SystemExit(main())
