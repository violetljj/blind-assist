"""Native SceneDepth parity only; never remove geometry from physical clearance."""


class DepthSurfaceFilter:
    def __init__(self):
        self.excluded=[];self.masked=[]

    def __call__(self,instance,section):
        slot=section.get('material_slot');materials=instance.get('materials',[])
        if type(slot) is not int or not 0<=slot<len(materials):
            raise ValueError('Authenticated section material slot required')
        material=materials[slot];effective=material.get('effective_render_material',{})
        mode=effective.get('blend_mode')
        if type(mode) is not int or effective.get('data_status')!='AVAILABLE':
            raise ValueError('Active material blend mode required')
        row=dict(component=instance['component_path'],instance=instance.get('instance_index'),
            mesh=instance['mesh']['asset_path'],section=section['section'],material_slot=slot,
            material=material['asset_path'],blend_mode=mode,is_masked=effective.get('is_masked'))
        # UE5.8 EBlendMode: 2 Translucent, 7 TranslucentColoredTransmittance.
        # IsMasked can be true for translucent *shadow* masking; it does not
        # turn the surface into an opaque SceneDepth writer.
        if mode in (2,7):
            self.excluded.append(row);return False
        if mode==1:
            self.masked.append(row);return True # Alpha not rasterized; disclose.
        if mode!=0:
            raise ValueError('Unreviewed effective blend mode: '+str(mode))
        return True

    def receipt(self):
        return dict(policy='NATIVE_SCENE_DEPTH_EFFECTIVE_BLEND_PARITY',excluded_translucent_sections=self.excluded,
            retained_masked_sections_with_unevaluated_alpha=self.masked,
            physical_clearance_geometry_changed=False,
            limitations='Depth-buffer parity only; masked alpha and single-sided culling remain unmodelled. Transparent physical obstacles are not certified observable.')
