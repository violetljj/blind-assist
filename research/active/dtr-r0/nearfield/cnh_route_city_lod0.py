"""Task-owned City component LOD0/fallback control; never save or alter mesh assets."""
from __future__ import annotations

POLICY='CITY_COMPONENT_LOD0_FALLBACK_CONTROL'


def apply(u,api):
    variable='r.Nanite.ProxyRenderMode'
    previous=int(u.SystemLibrary.get_console_variable_int_value(variable))
    world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
    u.SystemLibrary.execute_console_command(world,variable+' 0')
    if u.SystemLibrary.get_console_variable_int_value(variable)!=0:
        raise RuntimeError('Fallback rendering remains disabled')
    rows=[]
    for actor in api.get_all_level_actors():
        for component in actor.get_components_by_class(u.StaticMeshComponent):
            mesh=component.static_mesh
            if mesh is None:continue
            before=dict(disallow_nanite=bool(component.get_editor_property('disallow_nanite')),
                forced_lod_model=int(component.get_editor_property('forced_lod_model')))
            # Editor post-change notifications rerun Blueprint construction and
            # invalidate ISM references. Change only this transient component;
            # the runtime LOD setter below rebuilds its render state.
            component.set_editor_property('disallow_nanite',True,notify_mode=u.PropertyAccessChangeNotifyMode.NEVER)
            if before['forced_lod_model']==1:component.set_forced_lod_model(0)
            component.set_forced_lod_model(1) # Engine: 1 means LOD0, 0 means automatic.
            after=dict(disallow_nanite=bool(component.get_editor_property('disallow_nanite')),
                forced_lod_model=int(component.get_editor_property('forced_lod_model')))
            if after!={'disallow_nanite':True,'forced_lod_model':1}:
                raise RuntimeError('Component LOD control not applied: '+component.get_path_name())
            rows.append(dict(component=component.get_path_name(),mesh=mesh.get_path_name(),before=before,after=after))
    if not rows:raise RuntimeError('No static components found for City LOD control')
    return dict(policy=POLICY,status='APPLIED_NOT_GEOMETRY_VERIFIED',components=rows,
        proxy_render_mode_before=previous,proxy_render_mode_after=0,
        asset_mutations=False,map_saved=False,
        scope='All capture modalities share task-owned component LOD0 fallback; original captures unchanged',
        release='Exit task-owned editor without saving')
