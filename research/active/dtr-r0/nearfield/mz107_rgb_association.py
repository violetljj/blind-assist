"""Observable-only single RGB / ToF / Radar / IMU spatial association.

No simulator imports or evaluator reads. Pixel proposals are deliberately a
controlled-contrast baseline, not a general learned obstacle detector.
"""
import math
import cv2
import numpy as np


def proposals(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = gray.shape
    return [[int(x), int(y), int(x + bw), int(y + bh)]
            for x, y, bw, bh, area in stats[1:n]
            if area >= 12 and bw < .8*w and bh < .95*h]


def ray(az, el, pitch, yaw):
    a, e, p, y = map(math.radians, (az, el, pitch, yaw))
    # Source ToF zones use independent pinhole horizontal/vertical angles.
    f, right, up = 1., math.tan(a), math.tan(e)
    norm = math.sqrt(f*f+right*right+up*up)
    f, right, up = f/norm, right/norm, up/norm
    f, up = math.cos(p)*f-math.sin(p)*up, math.sin(p)*f+math.cos(p)*up
    return np.array([math.cos(y)*f-math.sin(y)*right,
                     math.sin(y)*f+math.cos(y)*right, up])


def pixel_ray(u, v, intr, pitch, yaw):
    a = math.degrees(math.atan((u-intr['cx'])/intr['fx']))
    e = math.degrees(math.atan((intr['cy']-v)/intr['fy']))
    return ray(a, e, pitch, yaw)


def point_inside(p):
    return .2 <= p[0] <= 3.6 and abs(p[1]) <= .3 and .4 <= p[2] <= 2.05


def extent_inside(box, distance, intr, pitch, yaw, height):
    corners = [pixel_ray(u, v, intr, pitch, yaw) for u in (box[0], box[2])
               for v in (box[1], box[3])]
    points = np.array([d*distance/math.hypot(d[0], d[1])+[0, 0, height] for d in corners])
    lo, hi = points.min(axis=0), points.max(axis=0)
    return bool(hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and lo[1] <= .3
                and hi[2] >= .4 and lo[2] <= 2.05)


def predict_frame(row, image, yaw):
    """Missing/conflicting RGB is fallback, never clearance or a ToF veto."""
    intr = row['rgb_intrinsics']; pitch = row['camera_pitch_deg']
    height = row['camera_in_body_m'][2]
    boxes = proposals(image) if image is not None else []
    tof = []; tof_positive = False
    if row['tof_packet_received']:
        for k, (r, status, a, e) in enumerate(zip(row['tof64_range_m'], row['tof64_status'],
                       row['tof64_theta_deg'], row['tof64_phi_deg'])):
            if status != 5 or r is None or not math.isfinite(r) or r <= 0: continue
            direction = ray(a, e, pitch, yaw)
            point = r*direction+[0, 0, height]
            tof_positive |= point_inside(point)
            ar, er = map(math.radians, (a, e))
            u = intr['cx']+intr['fx']*math.tan(ar)
            v = intr['cy']-intr['fy']*math.tan(er)
            tof.append(dict(zone=k, u=u, v=v, horizontal_range=r*math.hypot(direction[0], direction[1])))
    baseline = candidate = bool(tof_positive)
    associations = []; radar_count = 0
    if row['radar_packet_received']:
        for k, (r, a, valid) in enumerate(zip(row['radar_range_m'], row['radar_angle'], row['radar_valid'])):
            if not valid or r is None or a is None or not math.isfinite(r+a) or r <= 0: continue
            radar_count += 1
            angle = math.radians(a+yaw)
            current = .2 <= r*math.cos(angle) <= 3.6 and abs(r*math.sin(angle)) <= .3
            baseline |= current
            eligible = []
            for j, box in enumerate(boxes):
                angles = [math.degrees(math.atan((u-intr['cx'])/intr['fx'])) for u in (box[0],box[2])]
                if a < min(angles)-12 or a > max(angles)+12: continue
                inside = [t for t in tof if box[0] <= t['u'] <= box[2] and box[1] <= t['v'] <= box[3]]
                agreeing = [t for t in inside if abs(t['horizontal_range']-r) <= .35]
                # Any competing range in the same visual region prevents merging objects.
                if agreeing and len(agreeing) == len(inside): eligible.append((j, agreeing))
            if len(eligible) == 1:
                j, agreeing = eligible[0]
                refined = extent_inside(boxes[j], r, intr, pitch, yaw, height)
                candidate |= refined
                associations.append(dict(slot=k, state='ASSOCIATED', proposal=j,
                    tof_zones=[t['zone'] for t in agreeing], baseline_support=bool(current),
                    refined_support=refined))
            else:
                candidate |= current
                associations.append(dict(slot=k, state='UNKNOWN', eligible=len(eligible),
                    baseline_support=bool(current), refined_support=bool(current)))
    return dict(baseline=bool(baseline), candidate=bool(candidate), tof_support=bool(tof_positive),
                # No support is explicitly UNKNOWN; this prototype has no CLEAR state.
                baseline_state='ALERT' if baseline else 'UNKNOWN',
                candidate_state='ALERT' if candidate else 'UNKNOWN',
                proposals=boxes, associations=associations, valid_tof=len(tof), valid_radar=radar_count,
                integrated_yaw_deg=yaw)


def predict(rows, image_loader):
    output = []; last_episode = None; yaw = 0.; imu_available = True
    for row in rows:
        if row['episode_id'] != last_episode: yaw = 0.; imu_available = True
        last_episode = row['episode_id']
        if row['imu_valid']: yaw += row['delta_yaw']
        else: imu_available = False
        if not imu_available:
            raise ValueError('Missing IMU requires explicit uncertainty handling, not true-pose fallback')
        output.append(predict_frame(row, image_loader(row), yaw))
    return output
