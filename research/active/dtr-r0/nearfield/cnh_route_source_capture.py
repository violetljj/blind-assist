"""Two fixed source layouts, native seven-pass endpoints; never benchmark admission.

Loaded-native clearance is checked before insertion. No simulation, map saves,
automatic resampling, or inferred 10 Hz timestamps are used.
"""
from __future__ import annotations
import json
import hashlib
import math
import os
import platform
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))


def prevent_reentry(callback):
    """UE asset compilation can pump Slate again during an active callback."""
    busy=False
    def guarded(*args,**kwargs):
        nonlocal busy
        if busy:
            return
        busy=True
        try:
            return callback(*args,**kwargs)
        finally:
            busy=False
    return guarded


def probe_radius_m(layout):
    """Enclose every frozen camera's 8m geometry neighborhood at probe origin."""
    origin=layout['camera']
    variants=[layout]+[dict(layout,**candidate) for candidate in layout.get('candidates',[])]
    poses=[variant['camera'] for variant in variants]
    poses += [pose for variant in variants for clip in variant['clips'] for pose in clip['poses']]
    displacement=max(math.sqrt(sum((float(p[k])-float(origin[k]))**2 for k in ('x','y','z'))) for p in poses)
    if not math.isfinite(displacement):
        raise ValueError('Nonfinite probe coverage displacement')
    return 8.+displacement


def verify_alley_static_probe(receipt):
    """Read-only compiled capability gate; never rewrite source material graphs."""
    if not receipt.get('instances'):
        raise ValueError('Alley probe has no native instances')
    for instance in receipt['instances']:
        if not instance.get('materials'):
            raise ValueError('Alley native instance has no material capability evidence')
        for material in instance['materials']:
            effective = material.get('effective_render_material', {})
            if (effective.get('data_status') != 'AVAILABLE' or
                    effective.get('capability') != 'NO_COMPILED_MATERIAL_DEFORMATION'):
                raise ValueError('Alley native material deformation unknown or present: '+str(material))


def choose_layout_rgb_exposure(pixels, width, height, initial_ev100):
    """One settled fixed-EV probe meters the shadowed nearfield."""
    if len(pixels) != width * height or width < 8 or height < 8:
        raise ValueError('RGB exposure probe dimensions differ')
    # Keep sky and the darkest ground edge from steering wall/obstacle exposure.
    def levels(x0,x1,y0,y1):
        result=[]
        for y in range(int(height*y0),int(height*y1),4):
            for x in range(int(width*x0),int(width*x1),4):
                pixel=pixels[y*width+x]
                channels=[float(getattr(pixel,k)) for k in ('r','g','b')]
                # UE ReadRenderTargetRaw returns byte-valued FLinearColor for RGBA8.
                if not all(math.isfinite(v) and 0. <= v <= 255. for v in channels):
                    raise ValueError('RGB exposure probe is not finite LDR byte data')
                result.append(sum(v*w for v,w in zip(channels,(.2126,.7152,.0722)))/255.)
        result.sort()
        return result
    centre=levels(.15,.85,.30,.85)
    near=levels(.421875,.671875,.50,.9583333333)
    quantile=lambda values,fraction:values[int(fraction*(len(values)-1))]
    near_p30=quantile(near,.30)
    if not .03 <= near_p30 <= .75 or quantile(centre,.99)>.98:
        raise ValueError('RGB exposure probe shadow is quantized black or highlight is clipped')
    target=.15
    desired_adjustment=math.log2(target/near_p30)
    if abs(desired_adjustment)>4.:
        raise ValueError('RGB exposure probe needs more than four stops')
    # Higher EV darkens; quantize the EV change for reproducible replay.
    adjustment=round(desired_adjustment*4)/4
    fixed_ev100=float(initial_ev100)-adjustment
    return dict(policy='ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2',initial_ev100=float(initial_ev100),
        initial_bias_ev=0.,fixed_ev100=fixed_ev100,ev_adjustment=adjustment,
        fixed_ev_window=[fixed_ev100-.01,fixed_ev100+.01],
        probe_luma_p60=quantile(centre,.60),probe_luma_p10=quantile(centre,.10),
        probe_luma_p90=quantile(centre,.90),probe_luma_p99=quantile(centre,.99),
        probe_nearfield_p30=near_p30,probe_nearfield_p60=quantile(near,.60),
        target_nearfield_p30=target,sampled_pixels=dict(centre=len(centre),nearfield=len(near)),
        probe_channel_scale='RGBA8_BYTE_0_TO_255',
        roi_fraction=dict(centre=dict(x=[.15,.85],y=[.30,.85]),
                          nearfield=dict(x=[.421875,.671875],y=[.50,.9583333333])),
        selection='initial_ev_minus_nearest_quarter_stop_log2_target_over_nearfield_p30',
        max_adjustment_stops=4.,
        authority='ONE_SETTLED_LDR_RGB_PROBE_PER_LAYOUT_NO_DEPTH_OR_GEOMETRY_CHANGE')


def validate_insertions(spec):
    from cnh_route_source_compare_adapter import SCOPE, DEVELOPMENT_SCOPE, ALLEY_SCOPE, is_development, validated_pose
    if spec.get('transport_policy') not in (None,'NATIVE_SEVEN_ASYNC_V1'):
        raise ValueError('Unknown transport policy')
    if spec.get('transport_policy') and not is_development(spec):
        raise ValueError('New async transport is Development-only until parity checked')
    if spec.get('native_geometry_policy') not in (None,'CITY_COMPONENT_LOD0_FALLBACK_CONTROL',
            'CITY_NEARFIELD_DERIVED_LOD0_MATERIALS_UNCHANGED','CITY_NEARFIELD_VEHICLE_ZERO_SCALE',
            'CITY_NEAR_INSTANCE_DERIVED_LOD0_SUBSTITUTION'):
        raise ValueError('Unknown native geometry intervention')
    if spec.get('native_geometry_policy') and spec.get('map_asset')!='/Game/Map/Small_City_LVL':
        raise ValueError('LOD0 diagnostic control is City-only')
    if spec.get('native_geometry_policy') in ('CITY_NEARFIELD_DERIVED_LOD0_MATERIALS_UNCHANGED',
            'CITY_NEARFIELD_VEHICLE_ZERO_SCALE','CITY_NEAR_INSTANCE_DERIVED_LOD0_SUBSTITUTION'):
        control=spec.get('city_derived_control',{})
        fresh_city1=(spec.get('native_geometry_policy')=='CITY_NEARFIELD_VEHICLE_ZERO_SCALE' and
                     spec.get('data_role')=='Development' and
                     control.get('authority')=='FRESH_CITY1_SAME_SITE_DEVELOPMENT' and
                     control.get('physical_site_id')=='city-consumed-engineering-site-1' and
                     control.get('independent_site_count')==1 and control.get('new_layouts') is True)
        if (is_development(spec) or len(spec['layouts'])!=2 or
                not (control.get('authority')=='CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC' or fresh_city1) or
                control.get('benchmark_eligible') is not False or
                any('candidates' in layout for layout in spec['layouts'])):
            raise ValueError('City derived control requires bounded frozen layouts without candidate search')
    if spec.get('scene_layer') not in (SCOPE, DEVELOPMENT_SCOPE, ALLEY_SCOPE) or spec.get('scene_layer') != spec.get('scope', SCOPE) or spec.get('benchmark_eligible') is not False:
        raise ValueError('Explicit source engineering scope required')
    if spec.get('native_material_policy') not in (None, 'STREET_TRANSIENT_ZERO_WPO_PDO', 'ALLEY_FROZEN_STATIC_COMPILED'):
        raise ValueError('Unsupported native material policy')
    if spec.get('native_material_policy') == 'STREET_TRANSIENT_ZERO_WPO_PDO' and spec.get('map_asset') != '/Game/BAResearchSlice/Street200V7':
        raise ValueError('Native material intervention is Street-only')
    if spec.get('rgb_exposure_policy') not in (None,'ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2'):
        raise ValueError('Unknown RGB exposure policy')
    if spec.get('rgb_exposure_policy') and spec['scene_layer'] != ALLEY_SCOPE:
        raise ValueError('Layout RGB exposure probe is alley Development only')
    if spec.get('rgb_exposure_policy') and spec.get('rgb_probe_ev100')!=-2.:
        raise ValueError('Alley RGB exposure requires declared fixed -2 EV100 probe')
    if spec.get('capture_mode') not in (None,'ALLEY_RGB_ONLY_REPLAY_V1','ALLEY_RGB_EXPOSURE_DIAGNOSTIC_V1',
                                      'ALLEY_MFPD_DEPTH_DIAGNOSTIC_V1'):
        raise ValueError('Unknown capture mode')
    if spec.get('capture_mode') and (spec['scene_layer']!=ALLEY_SCOPE or
            len(spec.get('layouts',[]))!=1 or not spec.get('rgb_replay_source')):
        raise ValueError('RGB-only replay requires one alley layout, exposure probe and old source')
    if spec.get('capture_mode')=='ALLEY_RGB_ONLY_REPLAY_V1' and (
            spec.get('rgb_exposure_policy')!='ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2' or
            spec.get('rgb_insert_material_policy')!='ALLEY_DERIVED_MFPD_OFF_V1'):
        raise ValueError('RGB-only replay requires frozen shadow exposure and derived MFPD-off material')
    if spec.get('rgb_insert_material_policy') not in (None,'ALLEY_DERIVED_MFPD_OFF_V1'):
        raise ValueError('Unknown derived RGB insert material policy')
    if spec.get('rgb_insert_material_policy') and spec.get('capture_mode') not in (
            'ALLEY_RGB_ONLY_REPLAY_V1','ALLEY_MFPD_DEPTH_DIAGNOSTIC_V1'):
        raise ValueError('Derived MFPD-off material only belongs to Development RGB/depth comparison')
    assets=spec.get('assets',[])
    if len(assets)!=2 or {a['id'] for a in assets}!={1,254}:
        raise ValueError('Two actual assets with IDs 1 and 254 required')
    if any(not a.get('mesh_asset','').startswith('/Game/') for a in assets):
        raise ValueError('Actual project assets required')
    ev=spec.get('exposure_ev100')
    if isinstance(ev,bool) or not isinstance(ev,(float,int)) or not math.isfinite(ev):
        raise ValueError('Explicit finite source exposure_ev100 required')
    validation_layouts=[]
    for base in spec['layouts']:
        candidates=base.get('candidates')
        if candidates is not None:
            if not isinstance(candidates,list) or not 1<=len(candidates)<=16:
                raise ValueError('One to sixteen prespecified candidates required')
            for candidate in candidates:
                if not isinstance(candidate,dict) or 'camera' not in candidate or 'clips' not in candidate:
                    raise ValueError('Each candidate must freeze camera and all clip insertions')
                if any(key in candidate for key in ('layout_id','physical_site_id','candidates')):
                    raise ValueError('Candidate cannot change layout or site identity')
                validation_layouts.append(dict(base,**candidate))
        else:
            validation_layouts.append(base)
    for layout in validation_layouts:
        if len(layout.get('clips',[]))!=4 or {c['id'] for c in layout['clips']}!={'centre','boundary','outside','removed'}:
            raise ValueError('Four named fixed clips required per layout')
        for clip in layout['clips']:
            count = 40 if is_development(spec) else 2
            if len(clip.get('poses',[]))!=count or clip.get('trajectory_model')!='piecewise_linear_fixed_orientation':
                raise ValueError(f'{count} fixed poses per clip required')
            poses = [validated_pose(p) for p in clip['poses']]
            if any(any(p[k] != poses[0][k] for k in ('pitch','yaw','roll')) for p in poses):
                raise ValueError('Fixed orientation required throughout a clip')
            objects=clip.get('insertions',[])
            if len(objects)!=2 or {a['id'] for a in objects}!={1,254}:
                raise ValueError('Both inserted identities must be explicit in each clip')
            for obj in objects:
                for key in ('center_m','scale'):
                    values=obj[key]
                    if len(values)!=3 or not all(math.isfinite(float(v)) for v in values):
                        raise ValueError('Finite explicit insertion centre/scale required')
                if any(float(v)<=0 for v in obj['scale']):
                    raise ValueError('Strictly positive insertion scale required')
                if set(obj['rotation_deg'])!={'pitch','yaw','roll'} or not all(math.isfinite(float(v)) for v in obj['rotation_deg'].values()):
                    raise ValueError('Explicit finite insertion rotation required')
                if type(obj.get('hidden')) is not bool:
                    raise ValueError('Explicit insertion hidden flag required')
            flags={a['id']:a['hidden'] for a in objects}
            if flags!={1:clip['id']=='removed',254:False}:
                raise ValueError('ID1 target removed only in removed clip; ID254 distractor always present')


def engine():
    import unreal as u
    from cnh_route_capture import camera_record, write_json, write_npy
    from cnh_route_derived_assets import derive
    from cnh_route_source_compare_adapter import load_source, place_captures, exclude_hlod_from_captures, verify_source_unchanged
    from cnh_route_scene_probe import probe
    from cnh_route_source_clearance import audit
    from ue_capture_readiness import CaptureReadiness
    from ue_pair_export import PairExporter
    from ue_rgb_export import RgbExporter
    out=Path(os.environ['BA_CNH_SOURCE_OUTPUT'])
    spec=json.loads(Path(os.environ['BA_CNH_SOURCE_SPEC']).read_text(encoding='utf-8-sig'))
    validate_insertions(spec)
    exposure_diagnostic=spec.get('capture_mode')=='ALLEY_RGB_EXPOSURE_DIAGNOSTIC_V1'
    material_depth_diagnostic=spec.get('capture_mode')=='ALLEY_MFPD_DEPTH_DIAGNOSTIC_V1'
    rgb_only=spec.get('capture_mode') in ('ALLEY_RGB_ONLY_REPLAY_V1','ALLEY_RGB_EXPOSURE_DIAGNOSTIC_V1')
    if rgb_only and out.resolve().is_relative_to(Path(spec['rgb_replay_source']['capture_root']).resolve()):
        raise ValueError('RGB replay output cannot overwrite historical capture')
    from cnh_route_source_compare_adapter import is_development
    world,source=load_source(u,spec)
    fast=spec.get('render_recipe')=='STATIC_SPATIAL_V1'
    if fast:
        if not is_development(spec):
            raise ValueError('Spatial recipe is separately declared Development only')
        commands=['r.AntiAliasingMethod 0','r.TemporalAA.Upsampling 0',
            'r.Lumen.ScreenProbeGather.Temporal 0','r.Lumen.Reflections.Temporal 0',
            'r.LumenScene.Radiosity.Temporal 0','r.Shadow.Denoiser 0',
            'r.AmbientOcclusion.Denoiser 0','r.Streaming.PoolSize 2500']
        for command in commands:u.SystemLibrary.execute_console_command(world,command)
        source['render_recipe']=dict(name='STATIC_SPATIAL_V1',commands=commands,
            authority='NEW_DEVELOPMENT_RECIPE_REQUIRES_SAMPLE_PARITY_NOT_INHERITED_RGB_EQUIVALENCE')
    source['capture_debug_flags']={'Navigation':False,'ZoneGraph':False}
    write_json(out/'source-receipt.json',source)
    api=u.get_editor_subsystem(u.EditorActorSubsystem)
    rig=dict(width=640,height=360,hfov_deg=100.,baseline_m=.06)
    actors=[]; captures={}; targets={}; derived={}; frames=[]; probes=[]; layout_exposures={}
    state=dict(stage='PROBE_PREPARE',probe_index=0,index=0,warm=0,finished=False,last_layout=None)
    handle=[None];started=time.monotonic()
    native_before=[None]
    city_derived_session=[None]
    readiness=CaptureReadiness(u,timeout=300)
    async_attributes=spec.get('transport_policy')=='NATIVE_SEVEN_ASYNC_V1'
    if rgb_only:
        pairs=RgbExporter(u,'native_async',limit=4)
    elif async_attributes:
        from ue_attribute_export import AttributePairExporter
        pairs=AttributePairExporter(u,'native_async',limit=4,probe_indices=(0,39,40,79,80,119,120,159))
    else:
        pairs=PairExporter(u,'native_async',limit=4)
    development = is_development(spec)
    alley_rgb_probe=spec.get('rgb_exposure_policy')=='ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2'
    expected_frames=1 if material_depth_diagnostic else sum(len(c['poses']) for l in spec['layouts'] for c in l['clips'])
    report=dict(status='RUNNING',scope=spec['scope'],benchmark_eligible=False,expected_frames=expected_frames,
        temporal_authority=('NOMINAL_10HZ_POSES_STATIC_WORLD_NOT_REALTIME_OR_DYNAMIC_VIDEO' if development else 'TWO_FIXED_SETTLED_ENDPOINTS_PER_CLIP_NOT_SIMULATED_VIDEO'),
        instance_method='PER_INSERTED_ID_ISOLATED_NATIVE_DEPTH_AGREEMENT',
        engine_version=u.SystemLibrary.get_engine_version(),source_unchanged=False)
    cases=[(layout,clip,i,camera) for layout in spec['layouts'] for clip in layout['clips'] for i,camera in enumerate(clip['poses'])]

    def destroy_capture(name):
        actor=captures.pop(name)
        u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
        if not api.destroy_actor(actor):
            raise RuntimeError('Task capture actor release failed')
        actors.remove(actor)

    def native_fingerprint():
        owned={a.get_path_name() for a in actors}
        rows=[];errors=[]
        for actor in sorted(api.get_all_level_actors(),key=lambda a:a.get_path_name()):
            if actor.get_path_name() in owned:
                continue
            try:
                transform=actor.get_actor_transform()
                row=dict(actor=actor.get_path_name(),
                    translation_cm=[float(getattr(transform.translation,k)) for k in ('x','y','z')],
                    rotation_xyzw=[float(getattr(transform.rotation,k)) for k in ('x','y','z','w')],
                    scale=[float(getattr(transform.scale3d,k)) for k in ('x','y','z')],components=[])
                for component in sorted(actor.get_components_by_class(u.PrimitiveComponent),key=lambda c:c.get_path_name()):
                    # Hidden editor sprites resize when their icon texture streams.
                    # Preserve their visibility state, but do not treat icon bounds
                    # as movement of geometry visible in the capture.
                    hidden=component.get_editor_property('hidden_in_game')
                    visible=component.get_editor_property('visible')
                    if hidden is True or visible is False:
                        row['components'].append(dict(path=component.get_path_name(),
                            excluded_reason='HIDDEN_OR_INVISIBLE_IN_FIXED_SETTLED_CAPTURE',
                            hidden_in_game=hidden,visible=visible))
                        continue
                    origin,extent,radius=u.SystemLibrary.get_component_bounds(component)
                    row['components'].append(dict(path=component.get_path_name(),
                        world_bounds_origin_cm=[float(getattr(origin,k)) for k in ('x','y','z')],
                        world_bounds_extent_cm=[float(getattr(extent,k)) for k in ('x','y','z')],radius_cm=float(radius)))
                rows.append(row)
            except Exception as exc:
                errors.append(dict(actor=actor.get_path_name(),error=str(exc)))
        encoded=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return dict(sha256=hashlib.sha256(encoded).hexdigest(),actors=rows,errors=errors,
            scope='LOADED_NATIVE_ACTOR_TRANSFORMS_VISIBLE_PRIMITIVE_BOUNDS_AND_VISIBILITY_NOT_VERTEX_ANIMATION')

    def finish(error=None,status=None):
        if state['finished']:
            return
        state['finished']=True
        cleanup=[]
        if native_before[0] is not None:
            try:
                after=native_fingerprint()
                native_unchanged=not native_before[0]['errors'] and not after['errors'] and native_before[0]['sha256']==after['sha256']
                write_json(out/'native-state-after.json',after)
                report['native_transform_bounds_unchanged']=native_unchanged
                if not native_unchanged:
                    error=(error or '')+'Native transform/bounds fingerprint changed or incomplete'
            except Exception:
                error=(error or '')+traceback.format_exc()
        try:
            if rgb_only:
                report['rgb_export']=pairs.finish()
            else:
                report['pair_export']=pairs.finish()
        except Exception:
            error=(error or '')+traceback.format_exc()
        if city_derived_session[0] is not None:
            try:
                city_derived_session[0].restore()
            except Exception:
                error=(error or '')+traceback.format_exc()
            write_json(out/'native-geometry-intervention.json',city_derived_session[0].receipt)
        for name in list(captures):
            try:
                destroy_capture(name)
            except Exception as exc:
                cleanup.append(str(exc))
        for actor in reversed(actors):
            try:
                if not api.destroy_actor(actor):
                    cleanup.append('Task actor destroy failed')
            except Exception as exc:
                cleanup.append(str(exc))
        integrity=verify_source_unchanged(source)
        write_json(out/'map-integrity.json',integrity)
        release=dict(released=not cleanup,errors=cleanup)
        write_json(out/'actor-release.json',release)
        unchanged=integrity['map_unchanged'] and all(r['source_configuration_unchanged'] for _,r in derived.values())
        report.update(status='FAIL' if error or cleanup or not unchanged else (status or 'PASS_NATIVE_TRANSPORT'),
            actor_release=release,frames=len(frames),wall_s=time.monotonic()-started,source_unchanged=unchanged)
        if error:
            report['error']=error
        write_json(out/'raw-manifest.json',dict(rig=rig,frames=frames,scope=spec['scope'],data_role='Development',benchmark_eligible=False,
            derived_assets={str(k):r for k,(_,r) in derived.items()},
            temporal_authority=report['temporal_authority'],rgb_exposure_by_layout=layout_exposures))
        report['rgb_exposure_by_layout']=layout_exposures
        write_json(out/'engine-receipt.json',report)
        if handle[0] is not None:
            u.unregister_slate_post_tick_callback(handle[0])
        u.SystemLibrary.quit_editor()

    def capture(name,source_type,fmt):
        actor=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,100000))
        actors.append(actor);captures[name]=actor
        c=actor.capture_component2d
        c.capture_every_frame=False;c.capture_on_movement=False;c.always_persist_rendering_state=True
        c.fov_angle=100.;c.capture_source=source_type
        c.texture_target=u.RenderingLibrary.create_render_target2d(world,640,360,fmt)
        if source_type==u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
            c.texture_target.target_gamma=2.2
            c.set_editor_property('show_flag_settings',[
                u.EngineShowFlagsSetting(show_flag_name='TemporalAA',enabled=not fast),
                u.EngineShowFlagsSetting(show_flag_name='AntiAliasing',enabled=not fast),
                *((u.EngineShowFlagsSetting(show_flag_name='EyeAdaptation',enabled=True),)
                  if alley_rgb_probe else ())])
            volumes=[a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)]
            if volumes:
                c.post_process_settings=max(volumes,key=lambda a:a.priority).settings
            pp=c.post_process_settings
            initial_rgb_ev=float(spec['rgb_probe_ev100']) if alley_rgb_probe else float(spec['exposure_ev100'])
            for key,value in [('motion_blur_amount',0.),('film_grain_intensity',0.),('vignette_intensity',0.),
                ('lens_flare_intensity',0.),('bloom_intensity',0.),('lumen_scene_lighting_quality',2.),
                ('lumen_final_gather_quality',2.),('auto_exposure_min_brightness',initial_rgb_ev-.01 if alley_rgb_probe else initial_rgb_ev),
                ('auto_exposure_max_brightness',initial_rgb_ev+.01 if alley_rgb_probe else initial_rgb_ev),
                *((('auto_exposure_bias',0.),) if alley_rgb_probe else ())]:
                pp.set_editor_property('override_'+key,True);pp.set_editor_property(key,value)
            c.post_process_settings=pp;c.post_process_blend_weight=1.
        else:
            c.texture_target.target_gamma=1.
        settings=[x for x in c.get_editor_property('show_flag_settings')
                  if str(x.show_flag_name) not in ('Navigation','ZoneGraph')]
        settings += [u.EngineShowFlagsSetting(show_flag_name=name,enabled=False)
                     for name in ('Navigation','ZoneGraph')]
        c.set_editor_property('show_flag_settings',settings)
        exclude_hlod_from_captures(u,{name:actor})
        return actor

    def setup_insertions():
        for asset in spec['assets']:
            mesh,receipt=derive(u,asset['mesh_asset'],'/Game/CNHSourceInsertion_'+str(os.getpid()),
                disable_mfpd=spec.get('rgb_insert_material_policy')=='ALLEY_DERIVED_MFPD_OFF_V1')
            derived[asset['id']]=(mesh,receipt)
            write_json(out/f'derived-{asset["id"]}.json',receipt)
            actor=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(0,0,100000));actors.append(actor)
            component=actor.static_mesh_component
            component.set_static_mesh(mesh);component.set_collision_profile_name('NoCollision')
            component.set_editor_property('forced_lod_model',1)
            targets[asset['id']]=actor
        for side in ('left','right'):
            capture('rgb_'+side,u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            if not rgb_only:
                capture('depth_'+side,u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
        if not rgb_only:
            capture('normal_left',u.SceneCaptureSource.SCS_NORMAL,u.TextureRenderTargetFormat.RTF_RGBA32F)
            capture('albedo_left',u.SceneCaptureSource.SCS_BASE_COLOR,u.TextureRenderTargetFormat.RTF_RGBA32F)
            for identifier in (1,254):
                actor=capture('isolated_'+str(identifier),u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
                actor.capture_component2d.primitive_render_mode=u.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST
                actor.capture_component2d.show_only_component(targets[identifier].static_mesh_component)

    def position_and_geometry(obj):
        identifier=obj['id'];actor=targets[identifier];mesh,receipt=derived[identifier]
        actor.set_actor_hidden_in_game(obj['hidden'])
        actor.set_actor_scale3d(u.Vector(*obj['scale']))
        actor.set_actor_rotation(u.Rotator(**obj['rotation_deg']),False)
        actor.set_actor_location(u.Vector(0,0,0),False,True)
        box=mesh.get_bounding_box()
        local_center=u.Vector(*[(getattr(box.min,k)+getattr(box.max,k))/2 for k in ('x','y','z')])
        offset=u.MathLibrary.transform_location(actor.get_actor_transform(),local_center)
        actor.set_actor_location(u.Vector(*[obj['center_m'][i]*100-getattr(offset,k) for i,k in enumerate(('x','y','z'))]),False,True)
        transform=actor.get_actor_transform();sections=[]
        for section in range(mesh.get_num_sections(0)):
            vertices,triangles,normals,uv,tangents=u.ProceduralMeshLibrary.get_section_from_static_mesh(mesh,0,section)
            if not vertices or not triangles:
                raise ValueError('Empty inserted render geometry')
            sections.append(dict(vertices_m=[[v.x/100,v.y/100,v.z/100] for v in vertices],triangles=list(triangles)))
        return dict(inserted_id=identifier,hidden=obj['hidden'],source_asset=receipt['source_mesh'],mesh=mesh.get_path_name(),sections=sections,
            actual_translation_m=[getattr(actor.get_actor_location(),k)/100 for k in ('x','y','z')],
            actual_rotation_quaternion=[float(getattr(transform.rotation,k)) for k in ('x','y','z','w')],
            actual_scale=[float(getattr(actor.get_actor_scale3d(),k)) for k in ('x','y','z')],
            desired_bounds_center_m=obj['center_m'],derived_nanite_enabled=False,material_wpo='EXPLICIT_ZERO')

    def prepare_frame():
        state['frame_started']=time.monotonic()
        layout,clip,index,camera=cases[state['index']]
        folder=out/'frames'/f'frame-{state["index"]:04d}';folder.mkdir(parents=True,exist_ok=False)
        geometry=[position_and_geometry(obj) for obj in clip['insertions']]
        write_json(folder/'inserted-geometry.json',dict(instances=geometry))
        write_json(folder/'camera.json',camera_record(camera,rig))
        place_captures(u,captures,camera)
        state['folder']=folder
        state['layout_changed']=state['last_layout']!=layout['layout_id']
        if not fast or state['layout_changed']:
            readiness.begin(world,captures['rgb_left'].capture_component2d)
        else:
            counters=readiness._snapshot()
            if counters['asset_registry_loading'] or any(counters[k] for k in readiness.COUNTERS) or not counters['streaming_update_completed']:
                readiness.begin(world,captures['rgb_left'].capture_component2d)
        state['prepare_s']=time.monotonic()-state['frame_started']

    def set_rgb_fixed_ev(value):
        for side in ('left','right'):
            component=captures['rgb_'+side].capture_component2d
            settings=component.post_process_settings
            for key,setting in (('auto_exposure_min_brightness',float(value)-.01),
                                ('auto_exposure_max_brightness',float(value)+.01),
                                ('auto_exposure_bias',0.)):
                settings.set_editor_property('override_'+key,True)
                settings.set_editor_property(key,setting)
            component.post_process_settings=settings

    def configure_rgb_diagnostic(arm):
        for side in ('left','right'):
            component=captures['rgb_'+side].capture_component2d
            settings=component.post_process_settings
            for key,value in (('auto_exposure_min_brightness',arm['min_ev100']),
                              ('auto_exposure_max_brightness',arm['max_ev100']),
                              ('auto_exposure_bias',arm['bias_ev'])):
                settings.set_editor_property('override_'+key,True)
                settings.set_editor_property(key,float(value))
            if arm['lumen']:
                settings.set_editor_property('override_dynamic_global_illumination_method',True)
                settings.set_editor_property('dynamic_global_illumination_method',u.DynamicGlobalIlluminationMethod.LUMEN)
                settings.set_editor_property('override_reflection_method',True)
                settings.set_editor_property('reflection_method',u.ReflectionMethod.LUMEN)
                flags=[flag for flag in component.get_editor_property('show_flag_settings')
                       if str(flag.show_flag_name) not in ('GlobalIllumination','SkyLighting','LumenGlobalIllumination')]
                flags.extend(u.EngineShowFlagsSetting(show_flag_name=name,enabled=True)
                             for name in ('GlobalIllumination','SkyLighting','LumenGlobalIllumination'))
                component.set_editor_property('show_flag_settings',flags)
            component.post_process_settings=settings

    def prepare_exposure_probe():
        layout,clip,index,camera=cases[state['index']]
        if index != 0 or clip['id'] != 'centre':
            raise ValueError('Layout RGB probe must precede its first collection frame')
        for obj in clip['insertions']:
            position_and_geometry(obj)
        set_rgb_fixed_ev(spec['rgb_probe_ev100'])
        place_captures(u,captures,camera)
        readiness.begin(world,captures['rgb_left'].capture_component2d)
        state.update(stage='EXPOSURE_PROBE_READY',warm=0)

    def raw(name,path):
        c=captures[name].capture_component2d;c.capture_scene()
        values=u.RenderingLibrary.read_render_target_raw(world,c.texture_target,normalize=False)
        if len(values)!=640*360:
            raise ValueError('Native attribute dimensions differ')
        write_npy(path,(v for p in values for v in (p.r,p.g,p.b)),(360,640,3))

    @prevent_reentry
    def tick(delta):
        try:
            if state['finished'] or not pairs.ready():
                return
            stage=state['stage']
            if stage=='PROBE_PREPARE':
                if state['probe_index']==0 and spec.get('native_geometry_policy'):
                    if spec['native_geometry_policy']=='CITY_NEARFIELD_DERIVED_LOD0_MATERIALS_UNCHANGED':
                        from cnh_city_nearfield_derived import apply as apply_derived, NearFarPreflightError
                        try:
                            city_derived_session[0]=apply_derived(u,api,spec)
                        except NearFarPreflightError as exc:
                            source['native_geometry_intervention']=exc.receipt
                            write_json(out/'native-geometry-intervention.json',exc.receipt)
                            write_json(out/'source-receipt.json',source)
                            report['control_preflight']=exc.receipt
                            finish(status='CONTROL_PREFLIGHT_REJECTED');return
                        source['native_geometry_intervention']=city_derived_session[0].receipt
                    elif spec['native_geometry_policy']=='CITY_NEARFIELD_VEHICLE_ZERO_SCALE':
                        from cnh_city_vehicle_mask import apply as apply_vehicle_mask
                        city_derived_session[0]=apply_vehicle_mask(u,api,spec)
                        source['native_geometry_intervention']=city_derived_session[0].receipt
                    elif spec['native_geometry_policy']=='CITY_NEAR_INSTANCE_DERIVED_LOD0_SUBSTITUTION':
                        from cnh_city_instance_substitution import apply as apply_substitution
                        city_derived_session[0]=apply_substitution(u,api,spec)
                        source['native_geometry_intervention']=city_derived_session[0].receipt
                    else:
                        from cnh_route_city_lod0 import apply as apply_lod0
                        source['native_geometry_intervention']=apply_lod0(u,api)
                    write_json(out/'native-geometry-intervention.json',source['native_geometry_intervention'])
                    write_json(out/'source-receipt.json',source)
                if state['probe_index']==0 and spec.get('native_material_policy') == 'STREET_TRANSIENT_ZERO_WPO_PDO':
                    from cnh_route_street_static_background import apply
                    source['native_material_intervention']=apply(u,api)
                    write_json(out/'native-material-intervention.json',source['native_material_intervention'])
                    write_json(out/'source-receipt.json',source)
                layout=spec['layouts'][state['probe_index']]
                actor=capture('probe_left',u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
                place_captures(u,captures,layout['camera'])
                readiness.begin(world,actor.capture_component2d)
                state.update(stage='PROBE_READY',warm=0);return
            if stage=='PROBE_READY':
                if not readiness.poll():
                    return
                captures['probe_left'].capture_component2d.capture_scene()
                state['warm']+=1
                if state['warm']<32:
                    return
                layout=spec['layouts'][state['probe_index']]
                if state['probe_index']==0 and spec.get('native_material_policy') == 'STREET_TRANSIENT_ZERO_WPO_PDO':
                    from cnh_route_street_static_background import verify
                    verify(u,source['native_material_intervention'])
                    write_json(out/'native-material-intervention.json',source['native_material_intervention'])
                    write_json(out/'source-receipt.json',source)
                destroy_capture('probe_left')
                receipt=probe(u,api,layout['camera'],probe_radius_m(layout),out/'evaluator'/f'layout-{state["probe_index"]:02d}')
                if spec.get('native_material_policy') == 'ALLEY_FROZEN_STATIC_COMPILED':
                    verify_alley_static_probe(receipt)
                probes.append(receipt)
                state['probe_index']+=1
                if state['probe_index']<len(spec['layouts']):
                    state['stage']='PROBE_PREPARE';return
                native_before[0]=native_fingerprint()
                write_json(out/'native-state-before.json',native_before[0])
                combined=dict(instances=[row for receipt in probes for row in receipt['instances']])
                banks=[[dict(layout,**candidate) for candidate in layout.get('candidates',[{}])] for layout in spec['layouts']]
                selected=[None]*len(banks);selected_indices=[None]*len(banks);attempts=[]
                for candidate_index in range(max(len(bank) for bank in banks)):
                    current=[selected[i] if selected[i] is not None else banks[i][min(candidate_index,len(banks[i])-1)] for i in range(len(banks))]
                    clearance=audit(u,api,current,combined,source)
                    examined=[]
                    for i,candidate in enumerate(clearance['candidates']):
                        if selected[i] is not None or candidate_index>=len(banks[i]):
                            continue
                        examined.append(dict(layout_id=candidate['layout_id'],candidate_index=candidate_index,
                            status=candidate['status'],required_missing=candidate.get('required_missing',[]),
                            conflicts=candidate.get('result',{}).get('conflicts',[])))
                        if candidate.get('status')=='PASS_LOADED_WORLD_ONLY':
                            selected[i]=current[i];selected_indices[i]=candidate_index
                    attempts.append(dict(candidate_index=candidate_index,examined=examined))
                    if all(item is not None for item in selected):
                        break
                selection=dict(policy='FIRST_PASS_FROM_AT_MOST_16_FROZEN_GEOMETRY_CANDIDATES_PER_LAYOUT',
                    selected_indices=selected_indices,attempts=attempts,
                    candidate_counts=[len(bank) for bank in banks],selected_layouts=selected,
                    counts_tested=[sum(any(r['layout_id']==layout['layout_id'] for r in a['examined']) for a in attempts) for layout in spec['layouts']],
                    collected_frames_before_selection=0,source_readiness_views_rendered=True,benchmark_eligible=False)
                write_json(out/'candidate-selection.json',selection)
                write_json(out/'clearance.json',clearance)
                if any(item is None for item in selected):
                    report['prefilter_rejections']=dict(reason='NO_PASS_WITHIN_FROZEN_CANDIDATES',selection=selection)
                    if not development or not any(item is not None for item in selected):
                        finish(status='PREFILTER_REJECTED');return
                selected=[item for item in selected if item is not None]
                # Freeze selected layouts before spawning objects or collecting any frame.
                cases[:]=[(layout,clip,i,camera) for layout in selected for clip in layout['clips'] for i,camera in enumerate(clip['poses'])]
                if material_depth_diagnostic:
                    cases[:]=cases[:1]
                setup_insertions();state['stage']='PREPARE';return
            if stage=='PREPARE':
                if alley_rgb_probe and cases[state['index']][0]['layout_id'] not in layout_exposures:
                    prepare_exposure_probe();return
                prepare_frame();state.update(stage='READY',warm=0);return
            if stage=='EXPOSURE_PROBE_READY':
                if not readiness.poll():
                    return
                rgb=captures['rgb_left'].capture_component2d
                rgb.capture_scene();state['warm']+=1
                if state['warm']<32:
                    return
                values=u.RenderingLibrary.read_render_target_raw(world,rgb.texture_target,normalize=False)
                layout,clip,index,camera=cases[state['index']]
                decision=choose_layout_rgb_exposure(values,640,360,spec['rgb_probe_ev100'])
                decision.update(layout_id=layout['layout_id'],physical_site_id=layout['physical_site_id'],
                    probe_clip_id=clip['id'],probe_pose_index=index,probe_camera=camera,
                    sky_and_indirect_illumination='EXISTING_MAP_LIGHTS_UNCHANGED')
                layout_exposures[layout['layout_id']]=decision
                write_json(out/'rgb-exposure'/f'{layout["layout_id"]}.json',decision)
                if exposure_diagnostic:
                    pairs.export(world,rgb.texture_target,out/'rgb-exposure-diagnostic'/'baseline.png',0)
                    state['diagnostic_arms']=[
                        dict(name='bias_only',min_ev100=float(spec['rgb_probe_ev100'])-.01,
                             max_ev100=float(spec['rgb_probe_ev100'])+.01,bias_ev=decision['ev_adjustment'],lumen=False),
                        dict(name='fixed_ev_only',min_ev100=decision['fixed_ev100']-.01,
                             max_ev100=decision['fixed_ev100']+.01,bias_ev=0.,lumen=False),
                        dict(name='narrow_ev_minus1',min_ev100=-1.01,max_ev100=-.99,bias_ev=0.,lumen=False),
                        dict(name='narrow_ev_minus1_5',min_ev100=-1.51,max_ev100=-1.49,bias_ev=0.,lumen=False),
                        dict(name='narrow_ev_minus2',min_ev100=-2.01,max_ev100=-1.99,bias_ev=0.,lumen=False),
                        dict(name='narrow_ev_minus2_5',min_ev100=-2.51,max_ev100=-2.49,bias_ev=0.,lumen=False),
                        dict(name='auto_no_lumen',min_ev100=-10.,max_ev100=20.,bias_ev=-1.5,lumen=False),
                        dict(name='fixed_ev_lumen',min_ev100=decision['fixed_ev100']-.01,
                             max_ev100=decision['fixed_ev100']+.01,bias_ev=0.,lumen=True),
                        dict(name='narrow_ev_minus1_5_lumen',min_ev100=-1.51,max_ev100=-1.49,bias_ev=0.,lumen=True),
                        dict(name='narrow_ev_minus2_lumen',min_ev100=-2.01,max_ev100=-1.99,bias_ev=0.,lumen=True),
                        dict(name='auto_lumen',min_ev100=-10.,max_ev100=20.,bias_ev=-1.5,lumen=True)]
                    state['diagnostic_results']=[dict(name='baseline',probe_luma_p60=decision['probe_luma_p60'])]
                    state['diagnostic_index']=0;state['warm']=0
                    configure_rgb_diagnostic(state['diagnostic_arms'][0])
                    readiness.begin(world,rgb)
                    state['stage']='EXPOSURE_DIAG_READY';return
                set_rgb_fixed_ev(decision['fixed_ev100'])
                state['stage']='PREPARE';return
            if stage=='EXPOSURE_DIAG_READY':
                if not readiness.poll():
                    return
                rgb=captures['rgb_left'].capture_component2d
                rgb.capture_scene();state['warm']+=1
                if state['warm']<32:
                    return
                arm=state['diagnostic_arms'][state['diagnostic_index']]
                pixels=u.RenderingLibrary.read_render_target_raw(world,rgb.texture_target,normalize=False)
                try:
                    measurement=choose_layout_rgb_exposure(pixels,640,360,spec['rgb_probe_ev100'])
                    observed=dict(probe_luma_p60=measurement['probe_luma_p60'])
                except ValueError as exc:
                    observed=dict(probe_measurement_error=str(exc))
                state['diagnostic_results'].append(dict(arm,**observed))
                pairs.export(world,rgb.texture_target,out/'rgb-exposure-diagnostic'/(arm['name']+'.png'),
                    state['diagnostic_index']+1)
                state['diagnostic_index']+=1
                if state['diagnostic_index']==len(state['diagnostic_arms']):
                    report['rgb_exposure_diagnostic']=state['diagnostic_results']
                    finish(status='PASS_RGB_EXPOSURE_DIAGNOSTIC');return
                state['warm']=0
                configure_rgb_diagnostic(state['diagnostic_arms'][state['diagnostic_index']])
                readiness.begin(world,rgb)
                return
            if stage=='READY':
                if not readiness.poll():
                    return
                state['stage']='SETTLE'
            for side in ('left','right'):
                captures['rgb_'+side].capture_component2d.capture_scene()
            state['warm']+=1
            if state['warm']<(32 if not fast or state['layout_changed'] else 1):
                return
            folder=state['folder']
            export_started=time.monotonic()
            if rgb_only:
                for side in ('left','right'):
                    path=folder/(side+'.png')
                    pairs.export(world,captures['rgb_'+side].capture_component2d.texture_target,
                        path,state['index']*2+int(side=='right'))
            else:
                for side in ('left','right'):
                    rgb=captures['rgb_'+side].capture_component2d;depth=captures['depth_'+side].capture_component2d
                    depth.capture_scene()
                    pairs.export(world,rgb.texture_target,depth.texture_target,folder/(side+'.png'),
                        folder/('depth_'+side+'.transport.npy'),state['index']*2+int(side=='right'))
            if rgb_only:
                pass
            elif async_attributes:
                names=('normal_left','albedo_left','isolated_1','isolated_254')
                components=[captures[name].capture_component2d for name in names]
                for component in components:component.capture_scene()
                paths=[folder/(name+'.transport.npy') for name in ('normal_left','albedo_left','isolated_depth_1','isolated_depth_254')]
                pairs.export_attributes(world,[c.texture_target for c in components],paths,[False,False,True,True],state['index'])
            else:
                raw('normal_left',folder/'normal_left.transport.npy');raw('albedo_left',folder/'albedo_left.transport.npy')
                for identifier in (1,254):
                    component=captures['isolated_'+str(identifier)].capture_component2d;component.capture_scene()
                    if not u.BlindAssistCaptureLibrary.export_depth_npy(world,component.texture_target,str(folder/f'isolated_depth_{identifier}.transport.npy')):
                        raise RuntimeError('Per-ID native depth export failed')
            layout,clip,index,camera=cases[state['index']]
            frames.append(dict(id=folder.name,folder=folder.relative_to(out).as_posix(),layout_id=layout['layout_id'],
                physical_site_id=layout['physical_site_id'],clip_id=clip['id'],pose_index=index,
                environment_category=layout.get('environment_category'),data_role='Development',
                nominal_time_s=round(index*.1,6) if development else None,
                machine_id=platform.node(),render_recipe=spec.get('render_recipe','FROZEN_STATIC_MATERIAL_ORIGINAL_RENDER'),
                capture_mode=spec.get('capture_mode','NATIVE_SEVEN_PASS'),
                rgb_exposure=layout_exposures.get(layout['layout_id'],dict(policy='SOURCE_FIXED_EV100_ONLY',
                    initial_ev100=float(spec['exposure_ev100']))),
                timing=dict(prepare_s=state['prepare_s'],settle_and_wait_s=export_started-state['frame_started']-state['prepare_s'],
                    submit_and_attribute_readback_s=time.monotonic()-export_started,total_s=time.monotonic()-state['frame_started']),
                asset_ids=[1,254],target_hidden=clip['id']=='removed',readiness=readiness.receipt()))
            state['last_layout']=layout['layout_id']
            if index==len(clip['poses'])-1 and clip['id']=='removed':
                write_json(out/'batches'/f'{layout["layout_id"]}.json',dict(status='RAW_LAYOUT_COMPLETE_REQUIRES_FINALIZE',
                    layout_id=layout['layout_id'],frames=[r for r in frames if r['layout_id']==layout['layout_id']],machine_id=platform.node()))
            state.update(index=state['index']+1,stage='PREPARE')
            write_json(out/'progress.json',dict(frames=len(frames),expected_frames=len(cases),requested_frames=expected_frames,wall_s=time.monotonic()-started))
            if len(frames)==len(cases):
                finish(status='PASS_MFPD_DEPTH_DIAGNOSTIC' if material_depth_diagnostic else None)
        except Exception:
            finish(traceback.format_exc())

    try:
        handle[0]=u.register_slate_post_tick_callback(tick)
    except Exception:
        finish(traceback.format_exc())


if __name__=='__main__':
    try:
        engine()
    except Exception:
        out=Path(os.environ['BA_CNH_SOURCE_OUTPUT'])
        (out/'startup-failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        import unreal
        unreal.SystemLibrary.quit_editor()
