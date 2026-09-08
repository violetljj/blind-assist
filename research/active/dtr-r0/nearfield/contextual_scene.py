"""Attached street fixtures: scene ownership precedes route/view sampling."""
import copy
import math

M = '/Game/Building/SF/B/Material/'
MATERIALS = dict(metal=M+'MI_Bldg_PaintedMetal_Grey', blue=M+'MI_Bldg_PaintedMetal_Blue',
    black=M+'MI_Bldg_PaintedMetal_Black', stone=M+'MI_Bldg_Block_Limestone_Grey',
    brass=M+'MI_Bldg_Metal_Brass', wood=M+'MI_Bldg_Wood_Cedar')


def scene(family, height=1.65, tilt=30., thickness=.06):
    # Verified Street200V7 frontage plane is approximately world Y=15 m.
    # All fixtures are grounded or mechanically attached; no isolated hazard actors.
    x, floor = -76., .2
    objects = []
    def box(name, xyz, size, material='metal', parent='ground', roll=0., target=False):
        obj = dict(name=name, center_m=[x+xyz[0], xyz[1], floor+xyz[2]], size_m=list(size),
                   material_asset=MATERIALS[material], support_parent=parent, target_part=target)
        if roll:
            obj['rotation_deg'] = dict(pitch=0.,yaw=0.,roll=roll)
        objects.append(obj)
    if family in ('crossbar','oblique_rod'):
        route_y = 13.5
        halfspan = 1. if family == 'crossbar' else .7
        for side in (-1,1):
            y=route_y+side*halfspan
            box(f'foot_{side}', (0,y,.045), (.75,.30,.09), 'black')
            box(f'upright_{side}', (0,y,1.65), (.07,.07,3.3), parent=f'foot_{side}')
            # Visible height-adjustment sleeve anchors the controlled cross member.
            z=height if family=='crossbar' else height-side*halfspan*math.tan(math.radians(tilt))
            box(f'clamp_{side}', (0,y,z), (.13,.13,.12), 'brass', f'upright_{side}')
        roll=0. if family=='crossbar' else tilt
        length=2*halfspan/math.cos(math.radians(roll))
        box('adjustable_cross_member', (0,route_y,height), (thickness,length,thickness),
            'metal','clamp_-1',roll,target=True)
        # The fixture sits in a supported maintenance access bay, with side rails.
        for side in (-1,1):
            box(f'rear_foot_{side}', (1.5,route_y+side*halfspan,.045), (.5,.3,.09),'black')
            box(f'rear_post_{side}', (1.5,route_y+side*halfspan,.65), (.07,.07,1.3),parent=f'rear_foot_{side}')
            box(f'side_rail_{side}', (.75,route_y+side*halfspan,1.1), (1.5,.06,.06),parent=f'upright_{side}')
        context='Height-adjustable maintenance access frame on grounded feet, outside supports leave an aisle'
    elif family=='cabinet':
        route_y=13.6
        # Wall-mounted service cabinet: back touches facade, open door stays hinged.
        box('cabinet_back', (.55,14.94,height), (1.14,.08,.64),'metal','existing_facade')
        for dx in (0.,1.1):
            box(f'cabinet_side_{dx}', (dx,14.72,height), (.055,.48,.64),'metal','cabinet_back')
        for dz in (-.3,.3):
            box(f'cabinet_edge_{dz}', (.55,14.72,height+dz), (1.14,.48,.045),'metal','cabinet_back')
        box('cabinet_shelf', (.55,14.73,height-.12), (1.05,.4,.025),'black','cabinet_back')
        box('hinge', (0,14.47,height), (.07,.07,.58),'brass','cabinet_side_0.0')
        box('open_door', (0,13.93,height), (.045,1.10,.56),'blue','hinge',target=True)
        box('door_handle', (-.05,13.49,height), (.08,.035,.16),'brass','open_door')
        context='Service cabinet mounted against the building facade; door rotates about its side hinge into edge walking space'
    elif family=='hanging_sign':
        route_y=13.5
        box('facade_mount', (0,14.95,2.9), (.22,.12,.7),'black','existing_facade')
        box('shop_bracket', (0,14.0,3.18), (.08,2.,.08),'black','facade_mount')
        for side in (-1,1):
            y=route_y+side*.43
            top=height+.19
            box(f'suspension_{side}', (0,y,(top+3.18)/2), (.018,.018,3.18-top),
                'black','shop_bracket')
        box('hanging_panel', (0,route_y,height), (.07,.95,.38),'blue','suspension_-1',target=True)
        for dz in (-.20,.20):
            box(f'panel_trim_{dz}', (0,route_y,height+dz), (.08,1.,.03),'brass','hanging_panel')
        context='Projecting shop sign with wall bracket and two side suspension rods; supports remain outside head sweep'
    elif family=='window':
        route_y=13.3
        # A shallow facade bay has a real opening; the existing wall forms its recessed back.
        for dx in (-.7,.7):
            box(f'bay_side_{dx}', (dx,14.45,1.45), (.2,1.05,2.9),'stone','ground')
        box('bay_sill_wall', (0,14.0,.70), (1.4,.18,1.4),'stone','ground')
        box('bay_lintel', (0,14.0,2.7), (1.4,.18,.4),'stone','bay_side_-0.7')
        box('bay_roof', (0,14.45,2.95), (1.6,1.1,.1),'stone','bay_side_-0.7')
        for dx in (-.59,.59):
            box(f'window_jamb_{dx}', (dx,13.88,1.99), (.06,.08,1.08),'wood','bay_side_'+str(-.7 if dx<0 else .7))
        for dz in (1.46,2.52):
            box(f'window_sill_{dz}', (0,13.88,dz), (1.24,.12,.06),'wood','window_jamb_-0.59')
        for dz in (1.50,2.48):
            box(f'open_sash_rail_{dz}', (-.59,13.33,dz), (.05,1.12,.05),'wood','window_jamb_-0.59',target=True)
        for y in (12.79,13.86):
            box(f'open_sash_stile_{y}', (-.59,y,1.99), (.05,.05,1.03),'wood','open_sash_rail_1.5',target=True)
        context='Outward-opening casement remains hinged to an actual recessed frontage bay, with sill, jambs and lintel'
    else:
        raise ValueError(family)
    if any(min(o['size_m'])<=0 for o in objects):
        raise ValueError('Invalid fixture geometry')
    names={o['name'] for o in objects}
    if any(o['support_parent'] not in names|{'ground','existing_facade'} for o in objects):
        raise ValueError('Detached scene component')
    return dict(family=family, site_id='street200_frontage_x-76_y15', objects=objects,
                route_y_m=route_y, floor_z_m=floor, target_x_m=x-(.59 if family=='window' else 0), context=context)


def preview(template):
    result=copy.deepcopy(template)
    for key in ('worlds','suite_contract','provenance'):
        result.pop(key,None)
    cases=[]
    for family in ('crossbar','cabinet','oblique_rod','hanging_sign','window'):
        world=scene(family, height=1.78 if family=='cabinet' else 1.65)
        for view in ('context','walking'):
            camera=dict(x=world['target_x_m']-(3.4 if view=='context' else 1.5),
                y=world['route_y_m']-(1.2 if view=='context' else 0),z=1.9,
                pitch=-3.,yaw=24. if view=='context' else 0.,roll=0.)
            cases.append(dict(name=family+'_'+view, family=family, camera=camera,
                floor_z_m=.2, floor_check=False, objects=world['objects'],
                scene_context=world['context'], group_id=family, variant_id=view))
    result.update(schema='city-contextual-headspace-preview-v1', cases=cases,
        export_dependencies=False, settling_ticks=32, settling_interval_s=0., exposure_ev100=12.,
        purpose='SUPPORTED_FIXTURE_PLACEMENT_REVIEW_BEFORE_FACTORIAL_CAPTURE',
        scope='Synthetic supported assemblies in an existing street; no training or realism certification')
    return result


def collection(template):
    from contextual_sampling import conditions
    from contextual_geometry import target_contact, intrusion_metrics
    result=preview(template)
    cases=[]
    for condition in conditions('street200_frontage_x-76_y15'):
        family=condition['family']
        height=condition['height_m']
        if family=='cabinet' and condition['desired_relation']=='HEAD_ONLY' and not condition['boundary']:
            height=1.78
        if family=='oblique_rod' and condition['subset']=='core' and condition['desired_relation']=='HEAD_ONLY':
            height=1.70
        # A thick panel's edge, not its centre, should approach a body boundary.
        if condition['boundary'] and family in ('cabinet','hanging_sign'):
            half=.28 if family=='cabinet' else .215
            height+=half
        world=scene(family,height,condition['tilt_degrees'],condition['thickness_m'])
        route_y=world['route_y_m']-condition['lateral_offset_m']+condition['route_lateral_delta_m']
        # Fixtures are unchanged across near/far and lateral views: move the wearer.
        target_front=min(o['center_m'][0]-o['size_m'][0]/2 for o in world['objects'] if o['target_part'])
        camera=dict(x=target_front-.18-condition['distance_m'],y=route_y,z=1.9,
                    pitch=condition['camera_pitch_deg'],yaw=0.,roll=0.)
        wearer=dict(x=camera['x'],y=route_y,z=.2,yaw=0.,pitch=0.,roll=0.)
        objects=copy.deepcopy(world['objects'])
        if condition['occluder']:
            # Grounded notice stand beside the route; its panel occludes an edge
            # of the fixture, rather than an unexplained floating screen.
            ox,oy=target_front-.25,route_y+.4
            for name,xyz,size,parent in (
                ('notice_foot',[ox,oy,.23],[.22,.18,.06],'ground'),
                ('notice_post',[ox,oy,1.05],[.025,.025,1.64],'notice_foot'),
                ('notice_panel',[ox,oy,1.75],[.035,.2,.7],'notice_post')):
                objects.append(dict(name=name,center_m=xyz,size_m=size,material_asset=MATERIALS['metal'],
                                    support_parent=parent,target_part=False))
        contact=target_contact(objects,wearer)
        cases.append(dict(name=condition['condition_id'],condition=condition,
            effective_target_height_m=height,group_id=condition['counterfactual_parent_id'],
            variant_id=condition['condition_id'],camera=camera,wearer=wearer,
            floor_z_m=.2,floor_check=False,objects=objects,scene_context=world['context'],
            geometric_contact=contact,
            geometric_intrusion=intrusion_metrics(objects,wearer),
            sun_intensity_scale=.15 if condition['lighting_profile']=='darker' else 1.,
            skylight_intensity_scale=.35 if condition['lighting_profile']=='darker' else 1.))
    result.update(schema='city-contextual-headspace-128-v1',cases=cases,
        purpose='96_CORE_PLUS_32_HARD_ATTACHED_FIXTURE_DEVELOPMENT_NO_MODEL_SCORING',
        suite_contract=dict(core=96,hard=32,source_worlds=1,body_boxes_version='contact_retina_spec',
            boundary='ORTHOGONAL_EDGE_PERTURBATION_NOT_FIFTH_CLASS',
            authority='Actual assembly cuboid sweep plus separate visible native depth; intent is not truth',
            intrusion='No volume fraction risk target: thin hazards must not be downweighted by volume',
            split='Keep the entire source site together for any prospective source-disjoint split'))
    return result
