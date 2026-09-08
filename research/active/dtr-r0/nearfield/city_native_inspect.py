"""Read original nearby mesh instance transforms for evaluator target discovery."""
import math


def inventory(u, api, camera, radius_m):
    rows=[]
    for actor in api.get_all_level_actors():
        if 'HLOD' in actor.get_class().get_name():
            continue
        for c in actor.get_components_by_class(u.StaticMeshComponent):
            mesh=c.static_mesh
            if mesh is None:
                continue
            instanced=isinstance(c,u.InstancedStaticMeshComponent)
            count=c.get_instance_count() if instanced else 1
            for i in range(count):
                t=c.get_instance_transform(i, True) if instanced else c.get_world_transform()
                p=t.translation
                if math.hypot(p.x/100-camera['x'],p.y/100-camera['y']) > radius_m:
                    continue
                r=t.rotation.rotator(); scale=t.scale3d
                b=mesh.get_bounding_box()
                corners=[u.MathLibrary.transform_location(t,u.Vector(x,y,z))
                    for x in (b.min.x,b.max.x) for y in (b.min.y,b.max.y) for z in (b.min.z,b.max.z)]
                rows.append(dict(actor=actor.get_actor_label(), actor_path=actor.get_path_name(),
                    component_path=c.get_path_name(),instance_index=i if instanced else None,
                    mesh_asset=mesh.get_path_name(),
                    position_m=[p.x/100,p.y/100,p.z/100],
                    rotation_deg=dict(pitch=r.pitch,yaw=r.yaw,roll=r.roll),
                    scale=[scale.x,scale.y,scale.z],
                    bounds_min_m=[min(getattr(v,k) for v in corners)/100 for k in ('x','y','z')],
                    bounds_max_m=[max(getattr(v,k) for v in corners)/100 for k in ('x','y','z')],
                    collision_profile=str(c.get_collision_profile_name())))
    return dict(schema='native-city-mesh-inventory-v1',rows=rows,
        scope='Original instance transforms and bounds for discovery; bounds are not obstacle labels')
