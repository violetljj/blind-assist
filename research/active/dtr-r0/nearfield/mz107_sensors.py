"""UE snapshot sensor generator; truth/provenance are returned separately."""
import math
import random
from mz99_angle_information_capture import measure_radar,retained_slots

def basis(c):
    p,y,r=(math.radians(c.get(k,0)) for k in ('pitch','yaw','roll'))
    cp,sp,cy,sy,cr,sr=math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    return ((cp*cy,cp*sy,sp),(sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp),(-cr*sp*cy-sr*sy,-cr*sp*sy+sr*cy,cr*cp))

def sensors(u,world,frame,actors,state):
    ep=frame['episode'];t=frame['time_s'];cam=frame['camera']
    if ep not in state:state[ep]=dict(rng=random.Random(frame['sensor_seed']),previous_yaw=0.)
    st=state[ep];rng=st['rng'];f,r,up=basis(cam)
    origin=u.Vector(*(cam[k]*100 for k in ('x','y','z')))
    def trace(end):
        hit=u.SystemLibrary.line_trace_single(world,origin,end,u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
        if not hit or not hit.to_tuple()[0]:return None
        fields=hit.to_tuple();return fields[5],fields[10]
    packet=rng.random()>=.1
    row=dict(id=frame['id'],episode_id=ep,time_s=t,camera_in_body_m=[0.,0.,1.7],camera_pitch_deg=-3.,
        tof_packet_received=packet,tof_range_m=[None]*8,tof_theta_deg=[-22.5+(k+.5)*45/8 for k in range(8)],
        tof_status=[255 if packet else 0]*8,tof_range_sigma_m=[.04]*8,
        tof64_range_m=[None]*64,tof64_status=[255 if packet else 0]*64,tof64_theta_deg=[],tof64_phi_deg=[],
        radar_packet_received=True,radar_range_m=[None]*4,radar_angle=[None]*4,radar_velocity=[None]*4,radar_valid=[False]*4,
        delta_yaw=0. if t==0 else cam['yaw']-st['previous_yaw']+.25*.2+rng.gauss(0,.08),delta_pitch=0.,imu_valid=True)
    st['previous_yaw']=cam['yaw'];native_tof=[]
    for iy in range(8):
        for ix in range(8):
            k=iy*8+ix;az=-22.5+(ix+.5)*45/8;el=22.5-(iy+.5)*45/8
            row['tof64_theta_deg'].append(az);row['tof64_phi_deg'].append(el)
            d=[f[j]+math.tan(math.radians(az))*r[j]+math.tan(math.radians(el))*up[j] for j in range(3)]
            norm=math.sqrt(sum(v*v for v in d));hit=trace(origin+u.Vector(*(v/norm*400 for v in d)))
            true_range=None
            if hit:
                p,_=hit;true_range=math.sqrt((p.x-origin.x)**2+(p.y-origin.y)**2+(p.z-origin.z)**2)/100
                pd=(.9 if true_range<=2 else .65 if true_range<=3 else .35)*.85
                if packet and rng.random()<pd:
                    row['tof64_range_m'][k]=max(.05,round((true_range+rng.gauss(0,.04))/.02)*.02);row['tof64_status'][k]=5
            native_tof.append(true_range)
    for ix in range(8):
        values=[row['tof64_range_m'][iy*8+ix] for iy in range(8) if row['tof64_status'][iy*8+ix]==5]
        if values:row['tof_range_m'][ix]=min(values);row['tof_status'][ix]=5
    radar=[];provenance=[];bounds=[]
    for j,(actor,obj) in enumerate(zip(actors,frame['objects'])):
        center,extent=actor.get_actor_bounds(False)
        bounds.append(dict(name=obj['name'],center_m=[center.x/100,center.y/100,center.z/100],extent_m=[extent.x/100,extent.y/100,extent.z/100]))
        dx=(center.x-origin.x)/100;dy=(center.y-origin.y)/100
        az=math.degrees(math.atan2(dy,dx))-cam['yaw'];hit=trace(center)
        if abs(az)>60 or not hit or hit[1]!=actor.static_mesh_component:continue
        p,_=hit;rr=math.hypot((p.x-origin.x)/100,(p.y-origin.y)/100)
        velocity=-frame['wearer_speed']*dx/max(math.hypot(dx,dy),1e-9)
        observed=measure_radar(rr,az,velocity,0.,rng)
        if observed:
            radar.append(observed);provenance.append(dict(kind='real_actor',actor_id=ep+'/'+obj['name'],exact_angle_deg=az,pre_noise_range_m=rr))
    ghost=frame.get('radar_ghost')
    if ghost:
        dx=ghost['z']-cam['x'];dy=ghost['x'];rr=math.hypot(dx,dy);az=math.degrees(math.atan2(dy,dx))-cam['yaw']
        if abs(az)<=60:
            observed=measure_radar(rr,az,-frame['wearer_speed']*dx/max(rr,1e-9),0.,rng)
            if observed:radar.append(observed);provenance.append(dict(kind='persistent_ghost',actor_id=None,exact_angle_deg=az,pre_noise_range_m=rr))
    if rng.random()<.15:
        radar.append((round(rng.uniform(.4,5)/.05)*.05,round(rng.uniform(-60,60)/10)*10,round(rng.uniform(-1,1)/.1)*.1))
        provenance.append(dict(kind='transient',actor_id=None))
    slots=[None]*4
    for k,((rr,az,v),prov) in enumerate(retained_slots(radar,provenance)):
        row['radar_range_m'][k]=rr;row['radar_angle'][k]=az;row['radar_velocity'][k]=v;row['radar_valid'][k]=True;slots[k]=prov
    evaluation=dict(id=frame['id'],episode_id=ep,time_s=t,family=frame['family'],camera=cam,body_origin_m=frame['body_origin_m'],
        native_bounds=bounds,tof_native_ranges_m=native_tof,authority='EVALUATOR_ONLY_ENGINE_BOUNDS_AND_GEOMETRY')
    provenance=dict(id=frame['id'],episode_id=ep,time_s=t,radar_slots=slots,authority='EVALUATOR_ONLY_HYPOTHETICAL_RADAR_ORIGIN')
    return row,evaluation,provenance
