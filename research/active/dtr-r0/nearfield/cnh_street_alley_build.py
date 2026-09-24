"""Native authoring support for three bounded alley demonstration maps.

No benchmark capture or test split assignment. Source assets are never edited.
Clutter admission uses measured component bounds, not proposed anchor points.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def closure_roots_for_site(site, source_assets, materials):
    """Map and this site's visible/planned assets, excluding other split pools.

    Insert meshes are capture candidates, not actors in the saved authoring map.
    Including them here makes the closure useful for pre-capture review while
    retaining the distinction from what the saved map actually renders.
    """
    keys={row['asset'] for row in site['rows']}
    keys.update('insert__'+key for key in site['insert_assets'])
    roots={site['map_asset']}
    roots.update(source_assets[key]['source'].split('.')[0] for key in keys)
    roots.update(materials[site[field]].get_path_name().split('.')[0]
                 for field in ('wall_material','floor_material'))
    return roots


def aabb_gap(a,b):
    """Euclidean separation of closed 3D AABBs; intersection has zero gap."""
    if any(len(x)!=3 for x in (*a,*b)):
        raise ValueError('Three-dimensional min/max bounds required')
    if any(not math.isfinite(v) for point in (*a,*b) for v in point):
        raise ValueError('Finite bounds required')
    if any(lo>hi for box in (a,b) for lo,hi in zip(*box)):
        raise ValueError('Reversed AABB')
    return math.sqrt(sum(max(a[0][i]-b[1][i],b[0][i]-a[1][i],0.)**2 for i in range(3)))


def measured_bounds(u,component):
    origin,extent,_=u.SystemLibrary.get_component_bounds(component)
    return [[(float(getattr(origin,k))+sign*float(getattr(extent,k)))/100 for k in ('x','y','z')]
            for sign in (-1,1)]


class AssetBank:
    def __init__(self,u,asset_root):
        self.u=u;self.root=asset_root;self.meshes={};self.derived={};self.bounds={};self.receipts={};self.files={}
        if not asset_root.startswith('/Game/BAResearchAlley/') or '..' in asset_root:
            raise ValueError('Dedicated /Game/BAResearchAlley asset namespace required')
        if u.EditorAssetLibrary.does_directory_exist(asset_root):
            raise FileExistsError(asset_root)
        self.editor=u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
        self.tools=u.AssetToolsHelpers.get_asset_tools()

    def remember_file(self,path):
        package=path.split('.')[0]
        if package.startswith('/Game/'):
            file=Path(self.u.Paths.project_dir())/'Content'/(package[6:]+'.uasset')
        elif package.startswith('/Engine/'):
            file=Path(self.u.Paths.engine_content_dir())/(package[8:]+'.uasset')
        else:
            raise ValueError('Unresolved source content mount: '+path)
        self.files[str(file)]=sha(file)

    def static_material(self,material):
        if material is None:raise ValueError('Missing material interface')
        result=json.loads(self.u.BlindAssistCaptureLibrary.get_material_geometry_capability(material))
        if result.get('capability')!='NO_COMPILED_MATERIAL_DEFORMATION':
            raise ValueError('Static material not verified; asset rejected, not hidden: '+str(result))
        self.remember_file(material.get_path_name())
        return result

    def load(self,key,path):
        if key in self.meshes:return
        u=self.u;mesh=u.load_asset(path)
        if not isinstance(mesh,u.StaticMesh):raise ValueError('StaticMesh unavailable: '+path)
        self.remember_file(path)
        materials=[self.static_material(slot.get_editor_property('material_interface'))
                   for slot in mesh.get_editor_property('static_materials')]
        box=mesh.get_bounding_box()
        self.bounds[key]=[[float(getattr(point,k))/100 for k in ('x','y','z')] for point in (box.min,box.max)]
        self.meshes[key]=mesh
        self.receipts[key]=dict(source=path,bounds_m=self.bounds[key],materials=materials)

    def clone(self,key):
        if key in self.derived:return self.derived[key]
        source=self.meshes[key];u=self.u
        name='SM_'+hashlib.sha256(key.encode()).hexdigest()[:12]
        mesh=self.tools.duplicate_asset(name,self.root,source)
        if mesh is None:raise RuntimeError('Task mesh duplication failed')
        settings=self.editor.get_nanite_settings(mesh);settings.set_editor_property('enabled',False)
        self.editor.set_nanite_settings(mesh,settings,True)
        if self.editor.get_nanite_settings(mesh).get_editor_property('enabled'):raise RuntimeError('Nanite disable failed')
        refs=lambda obj:[s.get_editor_property('material_interface') for s in obj.get_editor_property('static_materials')]
        if refs(mesh)!=refs(source):raise RuntimeError('Source material references changed')
        if not u.EditorAssetLibrary.save_loaded_asset(mesh,only_if_is_dirty=False):raise RuntimeError('Derived mesh save failed')
        self.derived[key]=mesh;self.receipts[key]['derived']=mesh.get_path_name()
        return mesh

    def verify(self):
        changed=[path for path,digest in self.files.items() if sha(path)!=digest]
        if changed:raise RuntimeError('Source asset bytes changed: '+repr(changed))
        return dict(unchanged=True,files=self.files)


def spawn_row(u,api,bank,row):
    actor=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*(v*100 for v in row['location_m'])),u.Rotator(yaw=row.get('yaw',0.)))
    if actor is None:raise RuntimeError('Actor spawn failed')
    actor.set_actor_label('CNH_Alley_'+row['name'])
    component=actor.static_mesh_component;component.set_static_mesh(bank.clone(row['asset']))
    actor.set_actor_scale3d(u.Vector(*row.get('scale',[1.,1.,1.])))
    component.set_forced_lod_model(1);component.set_collision_profile_name('BlockAll')
    return actor,component


def validate(config):
    from cnh_street_alley_plan import validate as planner_validate
    return planner_validate(config)


def transformed_row_bounds(row,bounds):
    angle=math.radians(row.get('yaw',0.));c,s=math.cos(angle),math.sin(angle)
    scale=row.get('scale',[1.,1.,1.]);t=row['location_m']
    points=[]
    for point in itertools.product(*zip(*bounds)):
        x,y,z=[point[i]*scale[i] for i in range(3)]
        points.append([c*x-s*y+t[0],s*x+c*y+t[1],z+t[2]])
    return [[fn(p[i] for p in points) for i in range(3)] for fn in (min,max)]


def clearance_record(row,bounds,site):
    gaps=[dict(protected=p['name'],gap_m=aabb_gap(bounds,[p['low'],p['high']])) for p in site['protected_volumes']]
    minimum=min((g['gap_m'] for g in gaps),default=0.)
    passed=bool(gaps) and minimum+1e-5>=site['clearance_margin_m']
    return dict(name=row['name'],role=row['role'],bounds_m=bounds,
        required_margin_m=site['clearance_margin_m'],minimum_gap_m=minimum,passed=passed,gaps=gaps)


def dependency_closure(u, roots):
    registry=u.AssetRegistryHelpers.get_asset_registry()
    options=u.AssetRegistryDependencyOptions(include_soft_package_references=True,include_hard_package_references=True,
        include_searchable_names=False,include_soft_management_references=False,include_hard_management_references=False)
    pending=list(roots);seen=set();packages=[];files={};external=[]
    while pending:
        package=str(pending.pop()).split('.')[0]
        if package in seen:continue
        seen.add(package)
        if package.startswith('/Script/'):
            external.append(package);continue
        if package.startswith('/Game/'):base=Path(u.Paths.project_content_dir())/package[6:]
        elif package.startswith('/Engine/'):base=Path(u.Paths.engine_content_dir())/package[8:]
        else:raise ValueError('Unresolved dependency mount: '+package)
        found=[]
        for suffix in ('.uasset','.umap','.uexp','.ubulk'):
            path=Path(str(base)+suffix)
            if path.exists():
                key=str(path.resolve());files[key]=dict(path=key,sha256=sha(path),bytes=path.stat().st_size,package=package);found.append(key)
        if not found:raise FileNotFoundError('Missing dependency package files: '+package)
        deps=[str(x) for x in registry.get_dependencies(package,options)]
        pending.extend(deps);packages.append(dict(package=package,files=found,dependencies=deps))
    return dict(status='COMPLETE_HARD_AND_SOFT_PACKAGE_CLOSURE',roots=sorted(roots),packages=packages,files=list(files.values()),script_packages=sorted(external))


def prepare(u,config,out):
    from cnh_street_alley_plan import generate,check_registry,register_saved_site
    validate(config);out=Path(out)
    catalog=json.loads(Path(__file__).with_name('cnh_street_alley_assets.json').read_text())
    # This loading bank has no mutation until clone(); plans are all checked first.
    loading=AssetBank(u,config['namespace']+'/PreflightUnused_Assets')
    needed={'wall','paving'} | {key for s in config['sites'] for key in s['clutter_assets']}
    for key in sorted(needed):loading.load(key,catalog['meshes'][key]['asset_path'])
    insert_keys={key for site in config['sites'] for key in site.get('insert_assets',[])}
    for key in sorted(insert_keys):
        loading.load('insert__'+key,catalog['insert_assets'][key]['asset_path'])
        for material in loading.receipts['insert__'+key]['materials']:
            if material.get('blend_mode')!=0:raise ValueError('Controlled insert must be opaque: '+key)
    materials={};material_receipts={}
    selected_materials={s[field] for s in config['sites'] for field in ('wall_material','floor_material')}
    for key in sorted(selected_materials):
        entry=catalog['materials'][key]
        material=u.load_asset(entry['asset_path'])
        capability=loading.static_material(material)
        if capability.get('blend_mode')!=0:raise ValueError('Authoring override material must be opaque: '+key)
        materials[key]=material;material_receipts[key]=capability
    write(out/'loaded-assets.json',dict(bounds=loading.bounds,assets=loading.receipts,materials=material_receipts))
    generated=generate(config,loading.bounds)
    write(out/'family-receipt.json',dict(status='NATIVE_STATIC_FAMILY_CAPABILITY_VERIFIED_NOT_CAPTURE_ADMISSION',
        family_assignment=generated['family_assignment'],
        sites=[{k:s[k] for k in ('site_id','proposed_split','clutter_assets','insert_assets','physical_site_id','revision_of_map_asset')} for s in generated['sites']],
        assets={key:dict(loading.receipts[key],family_id=(catalog['insert_assets'][key.removeprefix('insert__')]['family_id'] if key.startswith('insert__') else catalog['meshes'][key].get('family_id'))) for key in sorted(loading.receipts)},
        source_hashes=loading.files))
    sites=generated['sites']
    registry_path=Path(u.Paths.project_dir())/'Saved/CNHAlley/site-registry.json'
    if registry_path.exists():check_registry(sites,json.loads(registry_path.read_text()))
    generated['persistent_site_registry']=str(registry_path)
    if not sites or len({s['map_asset'] for s in sites})!=len(sites):raise ValueError('Distinct maps required')
    preflight=[]
    for site in sites:
        if not site['map_asset'].startswith(config['namespace']+'/') or '..' in site['map_asset']:
            raise ValueError('Map outside authoring namespace')
        if site['asset_root']!=site['map_asset']+'_Assets':raise ValueError('Site-local asset namespace required')
        if u.EditorAssetLibrary.does_asset_exist(site['map_asset']) or u.EditorAssetLibrary.does_directory_exist(site['asset_root']):
            raise FileExistsError('No overwrite: '+site['map_asset'])
        if site['clearance_margin_m']!=.15:raise ValueError('Fixed 15cm margin required')
        checks=[clearance_record(row,transformed_row_bounds(row,loading.bounds[row['asset']]),site)
                for row in site['rows'] if row['role'] in ('wall','clutter')]
        preflight.append(dict(site_id=site['site_id'],checks=checks,passed=all(x['passed'] for x in checks)))
    write(out/'all-sites-preflight.json',dict(sites=preflight,assets=loading.receipts,materials=material_receipts,
        authority='NATIVE_ASSET_BOUNDS_AUTHORING_PREFLIGHT_NOT_SCENE_OR_DATA_ADMISSION'))
    if not all(x['passed'] for x in preflight):raise ValueError('Planned wall/clutter intersects protected query margin; no maps created')
    write(out/'generated-plan.json',generated)
    preserved={}
    content=Path(u.Paths.project_dir())/'Content'
    for relative in ('BAResearchSlice/Street200V7.umap',*(f'BAResearchExpansion/BrickServiceA_V{i}.umap' for i in (1,2,3))):
        path=content/relative
        if path.exists():preserved[str(path)]=sha(path)

    def build_map(site):
        api=u.get_editor_subsystem(u.EditorActorSubsystem)
        levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
        editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
        receipt=dict(status='BUILDING',site_id=site['site_id'],map_asset=site['map_asset'],views=site['views'],
            proposed_split=site.get('proposed_split'),formal_split_authority='NONE_DEVELOPMENT_AUTHORING_DEMO',
            protected_volumes=site['protected_volumes'],actors=[],clearance_checks=[],source_assets=loading.receipts)
        path=out/(site['site_id']+'-geometry-receipt.json')
        try:
            if not levels.new_level(site['map_asset']):raise RuntimeError('New level creation failed')
            world=editor.get_editor_world()
            if world.get_path_name().split('.')[0]!=site['map_asset']:raise RuntimeError('Wrong active map')
            bank=AssetBank(u,site['asset_root']);bank.meshes=dict(loading.meshes);bank.bounds=dict(loading.bounds)
            bank.receipts=json.loads(json.dumps(loading.receipts));bank.files=dict(loading.files)
            neutral=bank.tools.create_asset('M_NeutralGround',site['asset_root'],u.Material,u.MaterialFactoryNew())
            if neutral is None:raise RuntimeError('Neutral ground creation failed')
            ml=u.MaterialEditingLibrary
            color=ml.create_material_expression(neutral,u.MaterialExpressionConstant3Vector)
            color.set_editor_property('constant',u.LinearColor(.16,.17,.18,1.))
            rough=ml.create_material_expression(neutral,u.MaterialExpressionConstant);rough.set_editor_property('r',.95)
            ml.connect_material_property(color,'',u.MaterialProperty.MP_BASE_COLOR)
            ml.connect_material_property(rough,'',u.MaterialProperty.MP_ROUGHNESS)
            ml.recompile_material(neutral)
            if not u.EditorAssetLibrary.save_loaded_asset(neutral,only_if_is_dirty=False):raise RuntimeError('Neutral ground save failed')
            for row in site['rows']:
                actor,component=spawn_row(u,api,bank,row)
                material=neutral if row.get('material')=='__neutral__' else materials.get(row.get('material'))
                if row.get('material') is not None and material is None:raise ValueError('Unknown material alias')
                if material is not None:
                    for index in range(component.get_num_materials()):component.set_material(index,material)
                actual=measured_bounds(u,component)
                receipt['actors'].append(dict(name=row['name'],asset=row['asset'],role=row['role'],
                    material=row.get('material'),path=actor.get_path_name(),actual_component_bounds_m=actual))
                if row['role'] in ('wall','clutter'):
                    check=clearance_record(row,actual,site);receipt['clearance_checks'].append(check)
                    if not check['passed']:raise ValueError('Measured wall/clutter margin failed; no actor hidden: '+row['name'])
            sun=api.spawn_actor_from_class(u.DirectionalLight,u.Vector(0,0,1000),u.Rotator(pitch=-35,yaw=-25))
            sun.light_component.set_editor_property('intensity',5.)
            sun.light_component.set_editor_property('atmosphere_sun_light',True)
            api.spawn_actor_from_class(u.SkyAtmosphere,u.Vector(0,0,0))
            sky=api.spawn_actor_from_class(u.SkyLight,u.Vector(0,0,500));sky.light_component.set_editor_property('real_time_capture',True)
            if not u.EditorLoadingAndSavingUtils.save_map(world,site['map_asset']):raise RuntimeError('New map save failed')
            receipt.update(status='SAVED_AUTHORING_GEOMETRY_REQUIRES_VISUAL_REVIEW',source_integrity=bank.verify(),
                derived_assets=bank.receipts)
            map_file=content/(site['map_asset'].removeprefix('/Game/')+'.umap')
            roots=closure_roots_for_site(site,loading.receipts,materials)
            u.AssetRegistryHelpers.get_asset_registry().scan_modified_asset_files([str(map_file)])
            closure=dependency_closure(u,roots)
            closure_path=out/(site['site_id']+'-dependency-closure.json')
            write(closure_path,closure);receipt['dependency_closure']=dict(path=str(closure_path),sha256=sha(closure_path),status=closure['status'])
            register_saved_site(registry_path,site,receipt,sha(map_file))
            receipt['persistent_site_registry']=str(registry_path)
        except Exception:
            import traceback
            receipt.update(status='FAILED_AUTHORING_RETAIN_NEW_PARTIAL_MAP_NO_AUTO_DELETE',error=traceback.format_exc())
            raise
        finally:
            receipt['preserved_maps_unchanged']=all(sha(p)==digest for p,digest in preserved.items())
            receipt['preserved_map_sha256']=preserved
            write(path,receipt)
            if not receipt['preserved_maps_unchanged']:raise RuntimeError('Preserved map bytes changed')
        return receipt
    return sites,build_map


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).parent))
    import unreal
    from cnh_street_alley_preview import run
    run(unreal)
