"""Pre-pixel S10 temporal-binding design; no capture, model, or final access.

Uses the existing piecewise trajectory and polygon contact functions. Numeric
bounding boxes are the consumed ep_10 source-native walker.0023 and advertisement
measurements; they are analytic assumptions, not a new source admission receipt.
"""
import math
import dtr_carla_c2_rich_scene as c2
from materialize_dtr_final_reckoning_source_probe import trajectory

TARGET_HALF = 0.18767888844013214
WEARER_RADIUS = 0.45
HORIZON_S = 3.0
FULL_WINDOW = (1.8, 3.8)


def design():
    return {
        'wearer': trajectory(0., 0., [(0., .2, 0.)]),
        'target': trajectory(3., 0., [(0., -2.3, 0.), (1., 2.7, 0.),
            (2.4, .2, 0.), (4.5, -1.8, 0.), (6.3, .2, 0.)]),
        # At1.8s wearer x=.36, target x=2.86, camera x=.44;
        # midpoint x=1.65. Relative-to-wearer values are2.5/.08/1.29.
        # Panel enters from y=8 in two samples and departs after full window.
        'shell': dict(trajectory(1.04, 8., [(0., .2, 0.), (1.6, 1.45, -40.),
            (1.8, 1.45, 0.), (2.4, .2, 0.), (3.9, .2, 40.),
            (4.1, .2, 0.)]), yaw_offset_degrees=90.),
        'shell_collision_relevant': False,
        'shell_collisions_enabled': False,
        'full_window_s': list(FULL_WINDOW),
        'reference_shell': trajectory(20., 15., [(0., 0., 0.)]),
        'camera_local': {'x_m': .08, 'z_m': 1.45, 'pitch_degrees': -5.,
                         'horizontal_fov_degrees': 90., 'width': 1280, 'height': 720},
        'authority': 'PREPIXEL_ANALYTIC_DESIGN_ONLY_NATIVE_RASTER_AND_WITNESS_REQUIRED',
    }


def contact_at(value, t):
    x, y = c2.trajectory_position(value['target'], t)
    h = TARGET_HALF
    polygon = [[x-h,y-h], [x+h,y-h], [x+h,y+h], [x-h,y+h]]
    return c2.contact_union(c2.trajectory_position(value['wearer'], t), {'target': polygon},
                            wearer_radius_m=WEARER_RADIUS)[0]


def runs(times, mask):
    groups = []
    for t, flag in zip(times, mask):
        if flag:
            if not groups or abs(groups[-1][-1] + .1 - t) > 1e-7:
                groups.append([])
            groups[-1].append(t)
    return [{'start_s': g[0], 'end_inclusive_s': g[-1], 'frames': len(g),
             'sample_cell_duration_s': round(len(g)*.1, 6)} for g in groups]


def analytic_receipt():
    value = design()
    times = [i/10 for i in range(91)]
    # Same discrete10Hz+3s convention as captured native future truth. Since
    # relative trajectories remain stationary after6.3s, support beyond9s is defined.
    current = [contact_at(value,t) for t in times]
    future = [any(contact_at(value, t+j/10) for j in range(31)) for t in times]
    hidden = [FULL_WINDOW[0] <= t <= FULL_WINDOW[1] for t in times]
    negative_hidden = [h and not f for h,f in zip(hidden,future)]
    # At least some target box lies in the unobscured frustum. Full-body
    # containment is deliberately not required at physical contact distance.
    vertical_half = math.atan(720/1280)
    top = math.tan(math.radians(-5)+vertical_half)
    bottom = math.tan(math.radians(-5)-vertical_half)
    visible = []
    for t in times:
        x,_ = c2.trajectory_position(value['target'],t)
        wearer_x,_ = c2.trajectory_position(value['wearer'],t)
        depth = x-wearer_x-.08
        target_bottom, target_top = -.13, 1.73
        visible.append(depth > TARGET_HALF and target_top >= 1.45+depth*bottom
                       and target_bottom <= 1.45+depth*top)
    return {'authority': value['authority'], 'design': value,
        'current_contact_runs': runs(times,current),
        'future_contact_runs': runs(times,future),
        'known_negative_in_full_window_runs': runs(times,negative_hidden),
        'planned_full_disappearance_frames': sum(hidden),
        'analytic_frustum_intersection_samples': sum(visible),
        'pre_window_frustum_samples': sum(v and t < FULL_WINDOW[0] for t,v in zip(times,visible)),
        'post_exit_frustum_samples': sum(v and t >= 4.1 for t,v in zip(times,visible)),
        'limitations': ['BBox/frustum overlap does not establish native silhouette visibility.',
            'A screen-facing panel yaw90 is a new pre-pixel design, not inherited raster success.',
            'Native instance+witness must verify every responsible actor, plan validity, zero pixels, '
            'reference pixels, and >=0.8s negative interval inside disappearance.',
            'No probe pixels become FIT_ONLY or FINAL; no model evidence is produced.']}


if __name__ == '__main__':
    import json
    print(json.dumps(analytic_receipt(), indent=2))
