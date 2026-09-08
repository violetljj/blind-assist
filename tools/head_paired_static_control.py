"""Process-local background WPO control for HEAD-P1 mechanical source probes.

Called after map streaming and before creating controlled fixtures. No assets are
saved. All changes disappear with the task-owned editor process.
"""


def freeze_background_wpo(unreal, actor_api):
    rows=[]
    for actor in actor_api.get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            before=bool(component.get_editor_property('evaluate_world_position_offset'))
            if not before:
                continue
            component.set_editor_property('evaluate_world_position_offset',False)
            after=bool(component.get_editor_property('evaluate_world_position_offset'))
            if after:
                raise RuntimeError('Could not disable background WPO: '+component.get_path_name())
            rows.append(dict(component=component.get_path_name(),before=before,after=after))
    return dict(policy='BACKGROUND_WPO_DISABLED_IN_MEMORY_NO_ASSET_SAVE',changed=rows,
                controlled_fixture_policy='created_after_control_and_unmodified')
