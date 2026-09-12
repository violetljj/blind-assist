"""MZ108 observable image regions and reciprocal Radar association.

No simulator imports or evaluator reads. Pixel proposals are deliberately a
controlled-contrast baseline, not a general learned obstacle detector.
"""
import math
import cv2
import numpy as np


def otsu_proposals(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = gray.shape
    return [[int(x), int(y), int(x + bw), int(y + bh)]
            for x, y, bw, bh, area in stats[1:n]
            if area >= 12 and bw < .8*w and bh < .95*h]


def proposals(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    detector = cv2.MSER_create(5, 12, int(h*w*.5), .25, .2)
    _,regions = detector.detectRegions(gray)
    boxes = otsu_proposals(image)+[[int(x),int(y),int(x+bw),int(y+bh)]
             for x,y,bw,bh in regions if bw < .8*w and bh < .95*h]
    boxes.sort(key=lambda b: (-(b[2]-b[0])*(b[3]-b[1]), b))
    retained = []
    for box in boxes:
        area = (box[2]-box[0])*(box[3]-box[1])
        def contained(outer):
            overlap = max(0,min(box[2],outer[2])-max(box[0],outer[0]))*max(0,min(box[3],outer[3])-max(box[1],outer[1]))
            return overlap >= .95*area
        if area and not any(contained(o) for o in retained): retained.append(box)
    return retained


from mz107_rgb_association import ray, pixel_ray, point_inside, extent_inside


def predict_frame(row, image, yaw, use_regions=True, competitive=True):
    """Missing/conflicting RGB is fallback, never clearance or a ToF veto."""
    intr = row['rgb_intrinsics']; pitch = row['camera_pitch_deg']
    height = row['camera_in_body_m'][2]
    boxes = (proposals(image) if use_regions else otsu_proposals(image)) if image is not None else []
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
    associations = []; radar_count = 0; candidates = []
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
            candidates.append((k, r, current, eligible))
    # Reciprocal uniqueness is evaluated over ALL current returns, not greedily.
    counts = [sum(any(j == q for q,_ in eligible) for _,_,_,eligible in candidates)
              for j in range(len(boxes))]
    for k,r,current,eligible in candidates:
        unique = len(eligible) == 1
        competitors = counts[eligible[0][0]] if unique else 0
        if unique and (not competitive or competitors == 1):
            j, agreeing = eligible[0]
            refined = extent_inside(boxes[j], r, intr, pitch, yaw, height)
            candidate |= refined
            associations.append(dict(slot=k, state='ASSOCIATED', proposal=j,
                tof_zones=[t['zone'] for t in agreeing], baseline_support=bool(current),
                refined_support=refined, competing_returns=competitors))
        else:
            candidate |= current
            associations.append(dict(slot=k, state='UNKNOWN', eligible=len(eligible),
                competing_returns=competitors, baseline_support=bool(current),refined_support=bool(current)))
    return dict(baseline=bool(baseline), candidate=bool(candidate), tof_support=bool(tof_positive),
                # No support is explicitly UNKNOWN; this prototype has no CLEAR state.
                baseline_state='ALERT' if baseline else 'UNKNOWN',
                candidate_state='ALERT' if candidate else 'UNKNOWN',
                proposals=boxes, associations=associations, valid_tof=len(tof), valid_radar=radar_count,
                integrated_yaw_deg=yaw)


def predict(rows, image_loader, use_regions=True, competitive=True):
    output = []; last_episode = None; yaw = 0.; imu_available = True
    for row in rows:
        if row['episode_id'] != last_episode: yaw = 0.; imu_available = True
        last_episode = row['episode_id']
        if row['imu_valid']: yaw += row['delta_yaw']
        else: imu_available = False
        if not imu_available:
            raise ValueError('Missing IMU requires explicit uncertainty handling, not true-pose fallback')
        output.append(predict_frame(row, image_loader(row), yaw, use_regions, competitive))
    return output
