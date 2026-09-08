"""Evaluator-only original-instance reconstruction and collision/raster checks.

Clones exist only after the scene RGB/depth capture. They never enter model RGB,
cast no shadows, have no collision, and are destroyed before another scene view.
"""
import array
import ast
import math
import re
import struct


def depth_values(path):
    with path.open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Expected native NPY')
        version=f.read(2)
        size=struct.unpack('<H' if version[0] == 1 else '<I', f.read(2 if version[0] == 1 else 4))[0]
        header=f.read(size).decode('ascii')
        metadata=ast.literal_eval(header.strip())
        if metadata.get('descr') != '<f4' or metadata.get('shape') != (360,640) or metadata.get('fortran_order'):
            raise ValueError('Unexpected native depth layout')
        values=array.array('f'); values.frombytes(f.read())
        if len(values) != 640*360:
            raise ValueError('Incomplete depth')
        return values


def ray_status(identity, collision_axial, rendered_axial):
    if identity and abs(collision_axial-rendered_axial) <= .03:
        return 'MATCH'
    if not identity and collision_axial < rendered_axial-.03:
        return 'OCCLUDED'
    return 'UNKNOWN'


def check_rays(u, world, source, instance_index, camera, depths):
    candidates=[i for i,d in enumerate(depths) if 0 < d < 15]
    chosen=sorted({candidates[round(j*(len(candidates)-1)/31)] for j in range(32)}) if candidates else []
    p,y=math.radians(camera['pitch']),math.radians(camera['yaw'])
    forward=(math.cos(p)*math.cos(y),math.cos(p)*math.sin(y),math.sin(p))
    right=(-math.sin(y),math.cos(y),0.)
    up=(-math.sin(p)*math.cos(y),-math.sin(p)*math.sin(y),math.cos(p))
    start=u.Vector(*(camera[k]*100 for k in ('x','y','z')))
    focal=320/math.tan(math.radians(50))
    rows=[]
    for i in chosen:
        rx,ry=(i%640-319.5)/focal,(i//640-179.5)/focal
        direction=[forward[k]+rx*right[k]-ry*up[k] for k in range(3)]
        end=start+u.Vector(*(v*(depths[i]+.1)*100 for v in direction))
        hit=u.SystemLibrary.line_trace_single(world,start,end,u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                                             True,[],u.DrawDebugTrace.NONE)
        row=dict(pixel=[i%640,i//640],isolated_axial_m=depths[i],status='UNKNOWN')
        if hit:
            h=hit.to_tuple()
            impact=h[5]; component=h[10]; item=h[13]
            axial=sum((getattr(impact,k)-getattr(start,k))*forward[j]/100 for j,k in enumerate(('x','y','z')))
            identity=(component == source and (instance_index is None or item == instance_index))
            row.update(collision_axial_m=axial, component_path=component.get_path_name() if component else None,
                       instance_index=item,absolute_error_m=abs(axial-depths[i]))
            row['status']=ray_status(identity,axial,depths[i])
        rows.append(row)
    matches=sum(r['status']=='MATCH' for r in rows)
    unknown=sum(r['status']=='UNKNOWN' for r in rows)
    return dict(status='PASS' if matches >= 3 and unknown == 0 else 'UNKNOWN',
                matched_rays=matches,unknown_rays=unknown,rows=rows,
                authority='Native Visibility-channel complex collision against original component/instance; clone collision disabled')


def export(u, api, world, isolated_actor, spec, case, index, out):
    results=[]
    capture=isolated_actor.capture_component2d
    pose=case['camera']
    isolated_actor.set_actor_location(u.Vector(*(pose[k]*100 for k in ('x','y','z'))),False,False)
    isolated_actor.set_actor_rotation(u.Rotator(pitch=pose['pitch'],yaw=pose['yaw'],roll=pose.get('roll',0)),False)
    capture.primitive_render_mode=u.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST
    for target in spec['native_targets']:
        tid=target['target_id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',tid):
            raise ValueError('Unsafe target id')
        source=u.find_object(None,target['component_path'])
        if not isinstance(source,u.StaticMeshComponent) or source.static_mesh.get_path_name()!=target['mesh_asset']:
            raise ValueError('Original target component/mesh is not loaded: '+tid)
        instance_index=target.get('instance_index')
        transform=source.get_instance_transform(instance_index,True) if instance_index is not None else source.get_world_transform()
        pos=transform.translation
        if max(abs(v-expected) for v,expected in zip((pos.x/100,pos.y/100,pos.z/100),target['position_m'])) > .0001:
            raise ValueError('Original target moved: '+tid)
        clone=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(0,0,100000))
        try:
            clone.set_actor_label('BA evaluator-only '+tid)
            mesh=clone.static_mesh_component
            mesh.set_static_mesh(source.static_mesh)
            for slot in range(source.get_num_materials()):
                mesh.set_material(slot,source.get_material(slot))
            clone.set_actor_transform(transform,False,False)
            clone.set_actor_enable_collision(False)
            mesh.set_editor_property('cast_shadow',False)
            capture.clear_show_only_components()
            capture.show_only_component(mesh)
            capture.capture_scene()
            path=out/f'evaluator/isolated/{tid}/{index:04d}.npy'
            path.parent.mkdir(parents=True,exist_ok=True)
            if not u.BlindAssistCaptureLibrary.export_depth_npy(world,capture.texture_target,str(path)):
                raise RuntimeError('Isolated native export failed')
            rays=check_rays(u,world,source,instance_index,pose,depth_values(path))
            results.append(dict(target_id=tid,sample_index=index,source_component=source.get_path_name(),
                source_mesh=source.static_mesh.get_path_name(),original_instance_index=instance_index,
                exact_original_transform=str(transform),isolated_depth=str(path.relative_to(out)),**rays))
        finally:
            capture.clear_show_only_components()
            if not api.destroy_actor(clone):
                raise RuntimeError('Evaluator-only clone cleanup failed')
    return results
