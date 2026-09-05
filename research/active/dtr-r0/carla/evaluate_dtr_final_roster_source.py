"""R1 source-only semantics over verified native raw shard records.

Callers verify shard inventories/hashes before passing source/instance/witness
rows. This module never reads detector ledgers, predictions, model inputs or
fit/final score files. PASS is source semantics, not method effectiveness.
"""
from __future__ import annotations
import hashlib
import json
import math

SCHEMA = "dtr-final-roster-source-evaluation-v1"
PASS = "SOURCE_GATE_MET"
FAIL = "NOT_EVALUABLE"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def number(value):
    require(type(value) in (float, int) and math.isfinite(value), "nonfinite_or_non_numeric")
    return float(value)


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x,y) for x,y in zip(a,b))
    if type(a) in (float,int) and type(b) in (float,int):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-4
    return type(a) is type(b) and a == b


def runs(indices):
    result = []
    for i in indices:
        if not result or result[-1][-1] + 1 != i:
            result.append([])
        result[-1].append(i)
    return result


def xy(transform):
    return number(transform["x"]), number(transform["y"])


def distance(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def pixels(row, target):
    value = row["instance_visibility"][target]["pixels"]
    require(type(value) is int and value >= 0, "invalid_instance_pixel_count")
    return value


def _aligned(source, instance, witness, dt):
    require(len(source) >= 3 and len(source) == len(instance) == len(witness), "missing_or_unaligned_source_instance_witness")
    episode = source[0]["episode_id"]
    for i, triplet in enumerate(zip(source,instance,witness)):
        left = triplet[0]
        for row in triplet:
            require(row["episode_id"] == episode and row["sample_index"] == i, "frame_identity")
            require(abs(number(row["time_s"])-i*dt) < 1e-8, "frame_time")
            require(left["actors"].keys() == row["actors"].keys(), "independent_actor_roster_mismatch")
            for actor in left["actors"]:
                for key in ("transform", "bounding_box", "local_position", "command_velocity", "kind"):
                    require(close(left["actors"][actor][key],row["actors"][actor][key]), "independent_actor_geometry_mismatch:"+actor+":"+key)
            for key in ("wearer_transform", "truth", "plan_receipt_sha256", "layout_receipt_sha256"):
                require(close(left[key],row[key]), "independent_replay_mismatch:"+key)
        require(close(left["camera_transform"],triplet[1]["camera_transform"]), "source_instance_camera_mismatch")
    return episode


def _future(source, horizon):
    times = [number(r["time_s"]) for r in source]
    actual = []
    for row in source:
        truth = row["truth"]
        require(type(truth["current_contact"]) is bool, "current_contact_not_boolean")
        require(bool(truth["responsible_assets"]) == truth["current_contact"], "responsible_contact_mismatch")
        actual.append(truth["current_contact"])
    future = []
    for i,row in enumerate(source):
        positive = any(actual[j] for j in range(i,len(source)) if times[j]-times[i] <= horizon+1e-8)
        recorded = row["truth"]["future_contact_within_horizon"]
        require(type(recorded) is bool and recorded == positive, "native_future_contact_mismatch")
        # A no-contact truncated future is UNKNOWN, never a known-negative gap.
        future.append(True if positive else False if times[-1]-times[i] >= horizon-1e-8 else None)
    return times, actual, future


def _headings(points):
    values = []
    for a,b in zip(points,points[1:]):
        if distance(a,b) > 1e-8:
            angle = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
            if values:
                angle = values[-1] + (angle-values[-1]+180)%360-180
            values.append(angle)
    return values


def _receipt(annex, rows):
    receipt = annex["issued_plan_receipt"]
    body = {k:v for k,v in receipt.items() if k != "receipt_sha256"}
    calculated = hashlib.sha256((json.dumps(body,sort_keys=True,separators=(",", ":"),ensure_ascii=False)).encode()).hexdigest()
    require(calculated.lower() == receipt["receipt_sha256"].lower(), "issued_plan_receipt_hash")
    require(all(r["plan_receipt_sha256"] == receipt["receipt_sha256"] for r in rows), "issued_plan_receipt_binding")
    require(number(receipt["issued_at_s"]) <= rows[0]["time_s"] and number(receipt["expires_at_s"]) >= rows[-1]["time_s"], "issued_plan_receipt_expired")
    require(receipt["coordinate_frame"] == "LAYOUT_FORWARD_RIGHT", "issued_plan_coordinate_frame")
    return receipt


def _evaluate(protocol, stratum_id, source, instance, witness, annex, reference):
    require(protocol["roster_id"] == "DTR_FINAL_RECKONING_ROSTER_R1", "not_frozen_R1_roster")
    design = protocol["source_design"]
    dt, horizon = number(design["sample_seconds"]), number(design["route_horizon_seconds"])
    require(dt == 0.1 and horizon == 3.0, "R1_sampling_or_horizon_changed")
    stratum = next(s for s in design["strata"] if s["stratum_id"] == stratum_id)
    episode = _aligned(source,instance,witness,dt)
    target = annex["target_asset"]
    times, actual, future = _future(source,horizon)
    counts = [pixels(r,target) for r in instance]
    positions = [xy(r["actors"][target]["transform"]) for r in source]
    ego = [xy(r["wearer_transform"]) for r in source]
    allowed = set(annex.get("expected_responsible_assets", [target]))
    require(all(set(r["truth"]["responsible_assets"]) <= allowed for r in source), "unexpected_responsible_hazard")
    outcome = stratum["expected_outcome"]
    require(any(actual) if outcome != "SAFE" else not any(actual), "expected_source_outcome")
    metrics = {"episode_id": episode, "known_future_frames": sum(x is not None for x in future),
               "right_censored_future_frames": sum(x is None for x in future)}
    selected=[i for i,t in enumerate(times) if number(annex.get("score_start_s",0.))-1e-8<=t<=number(annex.get("score_end_s",times[-1]))+1e-8]
    require(bool(selected), "empty_declared_score_window")
    metrics["score_window_s"]=[times[selected[0]],times[selected[-1]]]
    metrics["source_authority"]=annex.get("source_authority","VERIFIED_NATIVE_SOURCE_INSTANCE_WITNESS")
    code = stratum_id[:3]
    if code == "S01":
        minimum_pixels=math.ceil(number(annex.get("minimum_trackable_pixel_fraction",.0002))*design["resolution"][0]*design["resolution"][1])
        fraction = sum(counts[i]>=minimum_pixels for i in selected)/len(selected)
        velocities = [((b[0]-a[0])/dt,(b[1]-a[1])/dt) for a,b in zip(positions[selected[0]:selected[-1]+1],positions[selected[0]+1:selected[-1]+1])]
        accelerations = [distance(a,b)/dt for a,b in zip(velocities,velocities[1:])]
        maximum = max(accelerations)
        require(fraction >= .90 and maximum < .20, "clean_visibility_or_acceleration")
        metrics.update(visible_fraction=fraction, minimum_trackable_pixels=minimum_pixels, maximum_target_acceleration_mps2=maximum,
                       trackability_authority="RAW_INSTANCE_VISIBILITY_ONLY_NOT_DETECTOR_TRACKING")
    elif code in ("S02","S03"):
        windows = annex["removal_windows"]
        require(sorted(map(len,windows)) == ([1] if code == "S02" else [2,3,6]), "frozen_removal_lengths")
        used = set()
        for window in windows:
            require(all(type(i) is int for i in window) and window == list(range(window[0],window[-1]+1)), "removal_not_contiguous_indices")
            require(window[0]>0 and window[-1]+1<len(source) and not used.intersection(window), "removal_boundary_or_overlap")
            used.update(window)
            require(all(future[i] is True for i in window), "removal_outside_positive_route_contact")
            require(counts[window[0]-1]>0 and counts[window[-1]+1]>0, "adjacent_raw_observation_invalid")
        metrics.update(removal_windows=windows, credential_scope="SOURCE_ELIGIBILITY_ONLY_NO_MODEL_CREDENTIAL_CLAIM")
    elif code == "S04":
        contact = actual.index(True)
        lateral = []
        for i in range(min(contact,len(source)-1)):
            vx=(positions[i+1][0]-positions[i][0])/dt;vy=(positions[i+1][1]-positions[i][1])/dt
            yaw=math.radians(number(source[i]["wearer_transform"]["yaw"]))
            forward=vx*math.cos(yaw)+vy*math.sin(yaw)
            right=-vx*math.sin(yaw)+vy*math.cos(yaw)
            lateral.append(abs(right)>abs(forward))
        require(bool(lateral) and sum(lateral)>len(lateral)/2, "not_predominantly_lateral_before_contact")
        metrics["lateral_dominant_precontact_intervals"] = sum(lateral)
    elif code == "S05":
        nearest = min(number(r["truth"]["minimum_distance_m"]) for r in source)
        separation=[distance(a,b) for a,b in zip(positions,ego)]
        closing=[(a-b)/dt for a,b in zip(separation,separation[1:])]
        require(nearest<1.30 and any(v<0 for v in closing), "not_near_route_receding_transition")
        metrics.update(minimum_native_route_clearance_m=nearest, minimum_closing_speed_mps=min(closing))
    elif code == "S06":
        secondary=annex["secondary_asset"]
        require(secondary != target, "same_actor_twice")
        require(all(r["actors"][target]["kind"] == r["actors"][secondary]["kind"] for r in source), "objects_not_same_class")
        second=[pixels(r,secondary) for r in instance]
        overlaps=[]
        for i,r in enumerate(instance):
            a=r["instance_visibility"][target]["bbox_uv_normalized"]
            b=r["instance_visibility"][secondary]["bbox_uv_normalized"]
            if a is not None and b is not None and min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1]):
                overlaps.append(i)
        qualified=[w for w in runs(overlaps) if w[0]>0 and w[-1]+1<len(source)
                   and all(counts[i]>0 and second[i]>0 for i in (w[0]-1,w[-1]+1))]
        require(bool(qualified), "no_shared_visual_overlap_with_independent_context")
        metrics.update(overlap_runs=qualified, mechanism_scope="RAW_INSTANCE_BBOX_OVERLAP_NOT_DETECTOR_ID_SWITCH")
    elif code == "S07":
        receipt=_receipt(annex,source)
        first_contact=times[actual.index(True)]
        waypoints=receipt["time_parameterized_waypoints"]
        points=[]
        for a,b in zip(waypoints,waypoints[1:]):
            start,end=number(a["time_s"]),number(b["time_s"])
            require(end>start,"issued_plan_nonincreasing_time")
            if start>=first_contact:break
            if not points:points.append((number(a["forward_m"]),number(a["right_m"])))
            u=min(1.,(first_contact-start)/(end-start))
            points.append((number(a["forward_m"])+u*(number(b["forward_m"])-number(a["forward_m"])),
                           number(a["right_m"])+u*(number(b["right_m"])-number(a["right_m"]))))
        headings=_headings(points)
        require(len(headings)>=2 and max(headings)-min(headings)>=25, "issued_route_heading_change_below_25")
        metrics["issued_heading_change_degrees"]=max(headings)-min(headings)
    elif code == "S08":
        displacement=max(distance(a,b) for a in positions for b in positions)
        yaws=[number(r["wearer_transform"]["yaw"]) for r in source]
        unwrapped=[yaws[0]]
        for a,b in zip(yaws,yaws[1:]):unwrapped.append(unwrapped[-1]+(b-a+180)%360-180)
        rotation=max(unwrapped)-min(unwrapped)
        require(displacement<.05 and rotation>=30, "static_displacement_or_yaw_rotation")
        metrics.update(target_world_diameter_m=displacement,wearer_yaw_change_degrees=rotation)
    elif code == "S09":
        require(reference is not None and len(reference)==len(source), "missing_unoccluded_reference")
        refcounts=[]
        for a,b in zip(instance,reference):
            require(close(a["time_s"],b["time_s"]) and close(a["wearer_transform"],b["wearer_transform"]) and close(a["camera_transform"],b["camera_transform"]), "reference_pose_or_time_mismatch")
            for key in ("transform","bounding_box","local_position","command_velocity"):
                require(close(a["actors"][target][key],b["actors"][target][key]),"reference_target_geometry_mismatch")
            refcounts.append(pixels(b,target))
        require(all(counts[i]>0 for i in selected), "partial_has_full_disappearance")
        qualifying=runs([i for i,(a,b) in enumerate(zip(counts,refcounts)) if i in selected and b>0 and .05<=a/b<=.45])
        require(any(len(w)>=6 for w in qualifying), "partial_fraction_run_below_six")
        components=[r["instance_visibility"][target]["connected_components"] for r in instance]
        require(all(type(v) is int and v>=0 for v in components), "invalid_raw_mask_component_count")
        non_singletons=[r["instance_visibility"][target]["non_singleton_components"] for r in instance]
        fragmented=runs([i for w in qualifying for i in w if components[i]>=2 and non_singletons[i]>=2])
        require(any(len(w)>=6 for w in fragmented), "disconnected_surface_mechanism_not_established")
        metrics.update(partial_runs=qualifying, disconnected_surface_runs=fragmented,
                       component_scope="4_CONNECTED_RASTER_COMPONENTS_SINGLE_PIXELS_EXCLUDED_NOT_3D_SURFACE_IDENTITY")
    elif code == "S10":
        positives=runs([i for i,v in enumerate(future) if i in selected and v is True])
        require(len(positives)==2,"not_two_future_contact_windows")
        gap=list(range(positives[0][-1]+1,positives[1][0]))
        require(gap and all(future[i] is False for i in gap) and times[gap[-1]]-times[gap[0]]>=.80-1e-8,"known_negative_gap_below_0p80s")
        require(reference is not None and len(reference)==len(source), "missing_unoccluded_reference")
        for a,b in zip(instance,reference):
            require(close(a["time_s"],b["time_s"]) and close(a["camera_transform"],b["camera_transform"]) and close(a["wearer_transform"],b["wearer_transform"]), "reference_pose_or_time_mismatch")
            for key in ("transform","bounding_box","local_position","command_velocity"):
                require(close(a["actors"][target][key],b["actors"][target][key]), "reference_target_geometry_mismatch")
        qualifying=[]
        for zero in runs([i for i,c in enumerate(counts) if c==0]):
            overlap=[i for i in zero if i in gap]
            after=list(range(zero[-1]+1,zero[-1]+9))
            if (len(zero)>=6 and overlap and times[overlap[-1]]-times[overlap[0]]>=.80-1e-8
                and zero[0]>0 and counts[zero[0]-1]>0 and after[-1]<len(source)
                and all(counts[i]>0 for i in after) and all(pixels(reference[i],target)>0 for i in zero)):
                qualifying.append(zero)
        require(bool(qualifying),"negative_gap_disappearance_overlap_or_reappearance_context")
        metrics.update(positive_windows=positives,negative_gap_s=[times[gap[0]],times[gap[-1]]],zero_pixel_runs=qualifying)
    else:
        raise ValueError("unknown_stratum")
    return metrics


def evaluate_episode(protocol, stratum_id, source_rows, instance_rows, witness_rows, annex, reference_rows=None):
    """annex keys: target_asset, optional secondary_asset, removal_windows (indices),
    issued_plan_receipt and expected_responsible_assets. All are evaluator-only.
    Missing/failed evidence is NOT_EVALUABLE, never SAFE or a model result.
    """
    try:
        metrics=_evaluate(protocol,stratum_id,source_rows,instance_rows,witness_rows,annex,reference_rows)
        return {"schema":SCHEMA,"stratum_id":stratum_id,"status":PASS,"metrics":metrics,
                "method_predictions_opened":False,"payload_integrity_scope":"CALLER_VERIFIED_SHARDS_REQUIRED"}
    except (ValueError,KeyError,TypeError,IndexError,StopIteration) as error:
        return {"schema":SCHEMA,"stratum_id":stratum_id,"status":FAIL,"reason":str(error),"method_predictions_opened":False}


def add_mask_components(instance_rows, shard_root, target):
    """Derive 4-connected target surfaces from inventory-verified native PNGs.

    Actor ID is evaluator-only. Returns copies; never alters captured records.
    Caller must have passed the shard through verified load_shard first.
    """
    import copy
    import numpy as np
    from PIL import Image
    output=copy.deepcopy(instance_rows)
    for row in output:
        with Image.open(shard_root / row["sensor_path"]) as image:
            rgb=np.asarray(image.convert("RGB"))
        ids=rgb[:,:,1].astype(np.uint32) | (rgb[:,:,2].astype(np.uint32)<<8)
        mask=ids==int(row["actors"][target]["carla_actor_id"])
        require(int(mask.sum())==pixels(row,target),"decoded_instance_pixel_count_mismatch")
        pending=set(map(int,np.flatnonzero(mask)))
        width=mask.shape[1]
        sizes=[]
        while pending:
            todo=[pending.pop()];size=0
            while todo:
                current=todo.pop();size+=1
                neighbors=[current-width,current+width]
                if current%width:neighbors.append(current-1)
                if current%width+1<width:neighbors.append(current+1)
                for neighbor in neighbors:
                    if neighbor in pending:
                        pending.remove(neighbor);todo.append(neighbor)
            sizes.append(size)
        count=len(sizes)
        row["instance_visibility"][target]["non_singleton_components"]=sum(size>1 for size in sizes)
        row["instance_visibility"][target]["component_sizes_pixels"]=sorted(sizes,reverse=True)
        row["instance_visibility"][target]["connected_components"]=int(count)
    return output


def evaluate_roster(protocol, episodes, annexes):
    """One seed group: maps indexed by frozen stratum ID, each episode bundle has
    source_rows/instance_rows/witness_rows and optional reference_rows.
    """
    expected=[s["stratum_id"] for s in protocol["source_design"]["strata"]]
    rows=[]
    for key in expected:
        bundle=episodes.get(key)
        if bundle is None or key not in annexes:
            rows.append({"stratum_id":key,"status":FAIL,"reason":"missing_stratum_source_or_annex"})
        else:
            rows.append(evaluate_episode(protocol,key,annex=annexes[key],**bundle))
    complete=len(expected)==10 and set(episodes)==set(expected) and all(r["status"]==PASS for r in rows)
    return {"schema":SCHEMA,"status":PASS if complete else FAIL,"strata":rows,
            "expected_strata":10,"method_predictions_opened":False,"full_roster_authorized":False,
            "detector_credential_gate":"PENDING_FROZEN_RAW_LEDGER_MEASURED_COLLISION_CREDENTIAL_NOT_SOURCE_GEOMETRY"}
