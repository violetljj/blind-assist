"""Component-only controls must leave source mesh objects untouched."""
import unittest
from types import SimpleNamespace as NS
from cnh_route_city_lod0 import apply


class CityLod0Tests(unittest.TestCase):
    def test_component_controls_no_mesh_mutation(self):
        mesh=NS(get_path_name=lambda:'/Mesh')
        state={'disallow_nanite':False,'forced_lod_model':0}
        component=NS(static_mesh=mesh,get_editor_property=lambda key:state[key],
            set_editor_property=lambda key,value,**kwargs:state.__setitem__(key,value),
            set_forced_lod_model=lambda value:state.__setitem__('forced_lod_model',value),
            get_path_name=lambda:'/Actor/Component')
        api=NS(get_all_level_actors=lambda:[NS(get_components_by_class=lambda cls:[component])])
        cvar=[2]
        u=NS(StaticMeshComponent=object,PropertyAccessChangeNotifyMode=NS(NEVER=0),UnrealEditorSubsystem=object,
            get_editor_subsystem=lambda cls:NS(get_editor_world=lambda:object()),
            SystemLibrary=NS(get_console_variable_int_value=lambda key:cvar[0],
                execute_console_command=lambda world,command:cvar.__setitem__(0,0)))
        result=apply(u,api)
        self.assertEqual(state,dict(disallow_nanite=True,forced_lod_model=1))
        self.assertIs(component.static_mesh,mesh)
        self.assertFalse(result['asset_mutations'])
        self.assertEqual(result['components'][0]['before']['forced_lod_model'],0)
        self.assertEqual(result['proxy_render_mode_before'],2)
        self.assertEqual(result['proxy_render_mode_after'],0)


if __name__=='__main__':unittest.main()
