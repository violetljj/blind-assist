"""Frozen authored-map validation, CPU only."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import cnh_route_source_compare_adapter as adapter
from cnh_route_source_capture import choose_layout_rgb_exposure, validate_insertions, verify_alley_static_probe
from cnh_route_source_launch import verify_layout_exposure_receipts

class AlleyTests(unittest.TestCase):
    def test_manifest_and_site_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'map.umap';path.write_bytes(b'map')
            digest=adapter.hashlib.sha256(b'map').hexdigest()
            manifest=dict(map_asset='/Game/BAResearchAlley/R3Map',physical_site_id='siteA',proposed_split='test',
                files=[dict(path=str(path),sha256=digest)],family_receipt=dict(path=str(path),sha256=digest))
            spec=dict(scope=adapter.ALLEY_SCOPE,benchmark_eligible=False,map_asset=manifest['map_asset'],
                map_file=str(path),map_sha256=digest,alley_manifest=manifest,data_role='Development',
                native_material_policy='ALLEY_FROZEN_STATIC_COMPILED',nominal_sample_interval_s=.1,
                render_recipe='STATIC_SPATIAL_V1',layouts=[dict(layout_id='a',physical_site_id='siteA',
                camera=dict(x=0,y=0,z=1.6,pitch=0,yaw=0,roll=0))])
            adapter.validate_spec(spec)
            for key,value in [('data_role','test'),('native_material_policy','STREET_TRANSIENT_ZERO_WPO_PDO'),('map_asset','/Game/Other')]:
                with self.assertRaises(ValueError):adapter.validate_spec(dict(spec,**{key:value}))
            bad=copy.deepcopy(spec);bad['layouts'][0]['physical_site_id']='other'
            with self.assertRaises(ValueError):adapter.validate_spec(bad)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):adapter.validate_spec(spec)

    def test_compiled_material_unknown_rejected(self):
        good=dict(effective_render_material=dict(data_status='AVAILABLE',capability='NO_COMPILED_MATERIAL_DEFORMATION'))
        verify_alley_static_probe(dict(instances=[dict(materials=[good])]))
        for material in ({},dict(effective_render_material=dict(data_status='UNKNOWN'))):
            with self.assertRaises(ValueError):verify_alley_static_probe(dict(instances=[dict(materials=[material])]))

class CollectionPrepareTests(unittest.TestCase):
    @staticmethod
    def assets():
        site=dict(site_id='train_a',proposed_split='train',map_asset='/Game/BAResearchAlley/a',
            insert_assets=['bollard','road_sign'])
        a=dict(source='/Game/bollard',bounds_m=[[-.3,-.4,-.01],[.3,.4,1.]])
        b=dict(source='/Game/sign',bounds_m=[[-.3,-.4,-.01],[.3,.4,1.]])
        catalog=dict(insert_assets={key:dict(asset_path=source,role='CONTROLLED_INSERT',
            proposed_split='train',family_id=key,source_family_root='/Game/'+key)
            for key,source in [('bollard',a['source']),('road_sign',b['source'])]})
        return site,[a,b],catalog

    def test_fixed_budget_and_reserved_test(self):
        from cnh_alley_collection_prepare import make_spec
        site,assets,catalog=self.assets()
        manifest=dict(map_file='unused',map_sha256='hash',physical_site_id='site')
        spec=make_spec({},site,manifest,assets,catalog)
        self.assertEqual(spec['limits'],dict(frames=160,layouts=1,timeout_s=600))
        clips=spec['layouts'][0]['clips']
        self.assertEqual(sum(len(c['poses']) for c in clips),160)
        self.assertEqual(clips[0]['poses'][-1]['x'],5.9)
        self.assertAlmostEqual(clips[2]['insertions'][0]['scale'][1]*.8,.12)
        self.assertTrue(clips[3]['insertions'][0]['hidden'])
        self.assertEqual(spec['rgb_exposure_policy'],'ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2')
        self.assertEqual(spec['rgb_probe_ev100'],-2.)
        with self.assertRaises(ValueError):make_spec({},dict(site,proposed_split='test'),manifest,assets,catalog)

    def test_rejects_undeclared_or_cross_partition_insertions(self):
        from cnh_alley_collection_prepare import make_spec
        site,assets,catalog=self.assets()
        manifest=dict(map_file='unused',map_sha256='hash',physical_site_id='site')
        with self.assertRaises(ValueError):make_spec({},site,manifest,assets[::-1],catalog)
        bad=copy.deepcopy(catalog);bad['insert_assets']['road_sign']['proposed_split']='dev'
        with self.assertRaises(ValueError):make_spec({},site,manifest,assets,bad)
        with self.assertRaises(ValueError):make_spec({},dict(site,insert_assets=['bollard','bollard']),manifest,assets,catalog)

    def test_exposure_policy_is_alley_only_and_does_not_change_geometry(self):
        from cnh_alley_collection_prepare import make_spec
        site,assets,catalog=self.assets()
        manifest=dict(map_file='unused',map_sha256='hash',physical_site_id='site')
        spec=make_spec(dict(exposure_ev100=13.2),site,manifest,assets,catalog)
        before=copy.deepcopy(spec['layouts'])
        validate_insertions(spec)
        self.assertEqual(spec['layouts'],before)
        with self.assertRaises(ValueError):
            validate_insertions(dict(spec,rgb_exposure_policy='UNKNOWN'))
        with self.assertRaises(ValueError):
            validate_insertions(dict(spec,scene_layer=adapter.DEVELOPMENT_SCOPE,scope=adapter.DEVELOPMENT_SCOPE))

    def test_single_probe_selects_quantized_fixed_ev_from_surface_roi(self):
        dark=SimpleNamespace(r=255/16,g=255/16,b=255/16)
        bright=SimpleNamespace(r=255.,g=255.,b=255.)
        pixels=[bright if y<12 or y>=34 else dark for y in range(40) for x in range(40)]
        decision=choose_layout_rgb_exposure(pixels,40,40,-2.)
        self.assertEqual(decision['fixed_ev100'],-3.25)
        self.assertEqual(decision['ev_adjustment'],1.25)
        self.assertAlmostEqual(decision['probe_luma_p60'],1/16)
        self.assertEqual(decision['initial_ev100'],-2.)
        self.assertGreater(decision['sampled_pixels']['nearfield'],0)
        with self.assertRaises(ValueError):
            choose_layout_rgb_exposure([SimpleNamespace(r=0.,g=0.,b=0.)]*1600,40,40,13.2)
        with self.assertRaises(ValueError):
            choose_layout_rgb_exposure(pixels[:-1],40,40,-2.)

    def test_sunlit_wall_cannot_dark_shift_shadowed_nearfield(self):
        wall=SimpleNamespace(r=204.,g=204.,b=204.)
        floor=SimpleNamespace(r=38.25,g=38.25,b=38.25)
        pixels=[floor if 16<=x<27 and 20<=y<38 else wall
                for y in range(40) for x in range(40)]
        decision=choose_layout_rgb_exposure(pixels,40,40,-2.)
        self.assertEqual(decision['fixed_ev100'],-2.)
        self.assertAlmostEqual(decision['probe_nearfield_p30'],.15)
        self.assertGreater(decision['probe_luma_p60'],.5)

    def test_every_frame_must_reference_its_layout_probe(self):
        spec=dict(rgb_exposure_policy='ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2')
        decision=dict(policy=spec['rgb_exposure_policy'],probe_pose_index=0,
            probe_clip_id='centre',physical_site_id='site',fixed_ev100=-2.)
        engine=dict(rgb_exposure_by_layout=dict(layout=decision))
        manifest=dict(rgb_exposure_by_layout=dict(layout=decision),frames=[dict(
            layout_id='layout',physical_site_id='site',rgb_exposure=decision) for _ in range(160)])
        self.assertEqual(verify_layout_exposure_receipts(spec,engine,manifest),engine['rgb_exposure_by_layout'])
        bad=copy.deepcopy(manifest);bad['frames'][80]['rgb_exposure']['fixed_ev100']=-1.
        with self.assertRaises(ValueError):verify_layout_exposure_receipts(spec,engine,bad)
