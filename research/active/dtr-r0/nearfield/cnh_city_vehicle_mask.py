"""Consumed City diagnostic: zero only near vehicle transforms, preserve indices.

No RemoveInstance, mesh/material assignment, insertion, save or seed mutation.
Every untouched transform and instance metadata is checked before/after and the
original transforms are restored before the task-owned editor exits.
"""
import hashlib
import json
from cnh_city_nearfield_derived import camera_centres
from cnh_route_scene_probe import transformed_bounds, intersects_sphere, xyz
from cnh_route_derived_assets import _stable_state

POLICY='CITY_NEARFIELD_VEHICLE_ZERO_SCALE'


def selected_indices(bounds, centres):
    return [i for i,(lo,hi) in enumerate(bounds)
            if any(intersects_sphere(lo,hi,c,8.) for c in centres)]


def transform_state(t):
    return [*xyz(t.translation),*xyz(t.rotation),float(t.rotation.w),*xyz(t.scale3d)]


def metadata(component, instanced, native_reader=None):
    names=['override_materials','forced_lod_model','min_draw_distance','ld_max_draw_distance']
    if instanced:
        names+=['instance_start_cull_distance','instance_end_cull_distance']
    result={name:_stable_state(component.get_editor_property(name)) for name in names}
    result['mesh']=component.static_mesh.get_path_name()
    if instanced:
        if native_reader is None:raise ValueError('Native instance metadata reader required')
        native=json.loads(native_reader(component))
        if native.get('data_status')!='AVAILABLE':raise ValueError('Native instance metadata unavailable')
        if native['instance_count']!=component.get_instance_count():raise ValueError('Native instance count differs')
        result['instance_state']=native
        result['count']=component.get_instance_count()
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class Session:
    def __init__(self,u,api,receipt):
        self.u,self.api,self.receipt=u,api,receipt
        self.entries=[]
        self.native_reader=u.BlindAssistCaptureLibrary.get_ism_preservation_state

    def restore(self):
        errors=[];checks=[]
        for component,instanced,transforms,chosen,before in reversed(self.entries):
            try:
                for i in chosen:
                    if instanced:
                        if not component.update_instance_transform(i,transforms[i],True,True,True):
                            raise RuntimeError('Instance transform restoration failed')
                    else:component.set_world_transform(transforms[i],False,True)
                current=([component.get_instance_transform(i,True) for i in range(component.get_instance_count())]
                         if instanced else [component.get_world_transform()])
                # UE UpdateInstanceTransform converts world FTransform through
                # component-local matrix decomposition. Only the touched entries
                # may acquire this roundoff; untouched instances remain exact.
                maxima=[0.]*10
                for i,(actual,expected) in enumerate(zip(current,transforms)):
                    a,b=transform_state(actual),transform_state(expected)
                    delta=[abs(x-y) for x,y in zip(a,b)]
                    maxima=[max(x,y) for x,y in zip(maxima,delta)]
                    limits=[.001]*3+[1e-6]*7 if i in chosen else [0.]*10
                    if any(d>limit for d,limit in zip(delta,limits)):
                        raise RuntimeError('Restored transform exceeds engine roundtrip precision: '+repr(delta))
                checks.append(dict(component=component.get_path_name(),maximum_absolute_delta=maxima,
                    units='translation_cm,quaternion_xyzw,scale_xyz',touched_translation_tolerance_cm=.001,
                    touched_quaternion_scale_tolerance=1e-6,untouched_tolerance=0.))
                if metadata(component,instanced,self.native_reader)!=before:raise RuntimeError('Restored metadata differs')
            except Exception as exc:errors.append(str(exc))
        self.receipt['restoration']=dict(restored=not errors,errors=errors,checks=checks,
            scope='ENGINE_TRANSFORM_ROUNDTRIP_ONLY_NOT_DEPTH_GATE_TOLERANCE')
        if errors:raise RuntimeError(str(errors))
        self.entries.clear()


def apply(u,api,spec):
    if spec.get('map_asset')!='/Game/Map/Small_City_LVL' or spec.get('native_geometry_policy')!=POLICY:
        raise ValueError('Explicit consumed City vehicle diagnostic required')
    control=spec.get('city_derived_control',{})
    fresh_city1=(control.get('authority')=='FRESH_CITY1_SAME_SITE_DEVELOPMENT' and
                 control.get('physical_site_id')=='city-consumed-engineering-site-1' and
                 control.get('independent_site_count')==1 and control.get('new_layouts') is True and
                 spec.get('data_role')=='Development' and
                 {row['physical_site_id'] for row in spec['layouts']}=={'city-consumed-engineering-site-1'})
    if (control.get('benchmark_eligible') is not False or
            not (control.get('authority')=='CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC' or fresh_city1)):
        raise ValueError('Explicit consumed or fresh same-site City1 Development control required')
    centres=camera_centres(spec)
    receipt=dict(policy=POLICY,status='PREFLIGHT',radius_m=8.,camera_centres_m=centres,
        source_authority=control['authority'],independent_site_count=control.get('independent_site_count'),
        saved=False,geometry_thresholds='UNCHANGED_V1',near_vehicles_removed=True,
        instance_removal_method='ZERO_SCALE_KEEP_INDEX_AND_CUSTOM_DATA',components=[],hlod=[])
    session=Session(u,api,receipt);planned=[]
    for actor in api.get_all_level_actors():
        if 'HLOD' in actor.get_class().get_name():
            hidden=bool(actor.is_hidden()) and bool(actor.is_temporarily_hidden_in_editor())
            receipt['hlod'].append(dict(actor=actor.get_path_name(),hidden=hidden))
            if not hidden:raise ValueError('HLOD must already be excluded by full-detail source loader')
        for component in actor.get_components_by_class(u.StaticMeshComponent):
            mesh=component.static_mesh
            if mesh is None or not mesh.get_path_name().startswith('/Game/Vehicle/'):continue
            instanced=isinstance(component,u.InstancedStaticMeshComponent)
            transforms=([component.get_instance_transform(i,True) for i in range(component.get_instance_count())]
                        if instanced else [component.get_world_transform()])
            chosen=selected_indices([transformed_bounds(u,mesh,t) for t in transforms],centres)
            if chosen:planned.append((component,instanced,transforms,chosen,metadata(component,instanced,session.native_reader)))
    if not planned:raise ValueError('No loaded /Game/Vehicle instances intersect frozen nearfield')
    try:
        for component,instanced,transforms,chosen,before in planned:
            session.entries.append((component,instanced,transforms,chosen,before))
            for i in chosen:
                t=transforms[i]
                masked=u.Transform(location=t.translation,rotation=t.rotation.rotator(),scale=u.Vector(0,0,0))
                if instanced:
                    if not component.update_instance_transform(i,masked,True,True,True):raise RuntimeError('Zero-scale update failed')
                else:component.set_world_transform(masked,False,True)
            after=([component.get_instance_transform(i,True) for i in range(component.get_instance_count())]
                   if instanced else [component.get_world_transform()])
            if len(after)!=len(transforms):raise RuntimeError('Instance indexing changed')
            for i,t in enumerate(after):
                if i in chosen:
                    if xyz(t.scale3d)!=[0.,0.,0.]:raise RuntimeError('Vehicle scale was not zeroed')
                elif transform_state(t)!=transform_state(transforms[i]):raise RuntimeError('Far vehicle transform changed')
            if metadata(component,instanced,session.native_reader)!=before:raise RuntimeError('Custom data/material/seed/cull state changed')
            receipt['components'].append(dict(component=component.get_path_name(),mesh=before['mesh'],
                instanced=instanced,instances=len(transforms),removed_indices=chosen,far_instances=len(transforms)-len(chosen),
                mapping=[dict(original_component=component.get_path_name(),original_index=i,
                              current_component=component.get_path_name(),current_index=i,visibility='ZERO_SCALE') for i in chosen],
                instance_metadata_sha256=digest(before),metadata_unchanged=True,far_transforms_unchanged=True))
        receipt.update(status='APPLIED_REQUIRES_GEOMETRY_DIAGNOSTIC',removed_instances=sum(len(x[3]) for x in planned))
        return session
    except Exception:
        session.restore()
        raise
