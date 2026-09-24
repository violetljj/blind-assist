"""Frozen authored-map validation, CPU only."""
import copy
import tempfile
import unittest
from pathlib import Path
import cnh_route_source_compare_adapter as adapter
from cnh_route_source_capture import verify_alley_static_probe

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
    def test_fixed_budget_and_reserved_test(self):
        from cnh_alley_collection_prepare import make_spec
        site=dict(site_id='train_a',proposed_split='train',map_asset='/Game/BAResearchAlley/a')
        manifest=dict(map_file='unused',map_sha256='hash',physical_site_id='site')
        asset=dict(source='/Game/source',bounds_m=[[-.3,-.4,-.01],[.3,.4,1.]])
        spec=make_spec({},site,manifest,[asset,asset])
        self.assertEqual(spec['limits'],dict(frames=160,layouts=1,timeout_s=600))
        clips=spec['layouts'][0]['clips']
        self.assertEqual(sum(len(c['poses']) for c in clips),160)
        self.assertEqual(clips[0]['poses'][-1]['x'],5.9)
        self.assertAlmostEqual(clips[2]['insertions'][0]['scale'][1]*.8,.12)
        self.assertTrue(clips[3]['insertions'][0]['hidden'])
        with self.assertRaises(ValueError):make_spec({},dict(site,proposed_split='test'),manifest,[asset,asset])
