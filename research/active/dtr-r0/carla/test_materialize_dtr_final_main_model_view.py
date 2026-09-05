"""Synthetic C2 fixture only: no real source/model/evaluator access."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import materialize_dtr_final_main_model_view as view
from test_dtr_carla_c4_join import C4JoinFixture


class MainModelViewTest(unittest.TestCase):
    def fixture(self,root):
        fixture=C4JoinFixture(root)
        layout=next(iter(fixture.scenes))
        template=fixture.scenes[layout]['episodes'][0]
        episodes=[]
        for i in range(1,13):
            episode=deepcopy(template)
            episode['episode_id']=f'ep_{i:02d}'
            episodes.append((layout,episode))
        source=root/'source-model'
        fixture._model_root(source,episodes)
        return source

    def test_main_only_resealed_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=self.fixture(root)
            source_hash=view.adapter.sha256_file(source/'manifest.json')
            original_open=Path.open
            def no_auxiliary(path,*args,**kwargs):
                if 'ep_11' in path.parts or 'ep_12' in path.parts or path.name in ('ep_11.json','ep_12.json'):
                    raise AssertionError('auxiliary input opened')
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',no_auxiliary):
                receipt=view.materialize(source,root/'main-model',hardlink=True)
            contract=view.adapter.load_model_contract(root/'main-model')
            self.assertEqual(tuple(e.episode_id for e in contract.episodes),view.MAIN_IDS)
            self.assertEqual(receipt['frames'],10)
            self.assertEqual(source_hash,view.adapter.sha256_file(source/'manifest.json'))
            self.assertFalse((root/'main-model/episodes/ep_11').exists())
            self.assertFalse((root/'main-model/plans/ep_12.json').exists())
            self.assertNotEqual(receipt['source_manifest_sha256'],receipt['derived_manifest_sha256'])
            self.assertTrue((source/'episodes/ep_01/rgb/000000.png').samefile(root/'main-model/episodes/ep_01/rgb/000000.png'))
            with self.assertRaises(FileExistsError):
                view.materialize(source,root/'main-model')

    def test_root_escape_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=self.fixture(root)
            manifest=view.read(source/'manifest.json')
            manifest['model_contract']['path']='../source-model/model_contract.json'
            (source/'manifest.json').unlink()
            view.write(source/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'noncanonical_root_reference'):
                view.materialize(source,root/'main-model')
            self.assertFalse((root/'main-model').exists())


if __name__=='__main__':
    unittest.main()
