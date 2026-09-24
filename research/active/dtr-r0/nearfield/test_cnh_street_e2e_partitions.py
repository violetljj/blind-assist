import unittest
from unittest.mock import patch
from cnh_street_e2e_partitions import guard_rows,planned_split

class PartitionTests(unittest.TestCase):
    def test_alley_physical_sites_cannot_cross(self):
        rows=[dict(layout_id=l,physical_site_id='same',data_role='Development',environment_category='alley') for l in ('a','b')]
        plan=dict(schema='cnh-development-merge-partitions-v1',layouts={l:dict(partition=p,physical_site_id='same') for l,p in [('a','train'),('b','dev')]})
        with self.assertRaises(ValueError):planned_split(rows,plan)
        rows[1]['physical_site_id']='other';plan['layouts']['b']['physical_site_id']='other'
        self.assertEqual(planned_split(rows,plan)[0].tolist(),[True,False])

    def test_test_authority_cannot_be_relabelled(self):
        row=dict(layout_id='a',physical_site_id='p',data_role='Development',proposed_split='test')
        with self.assertRaises(ValueError):guard_rows([row],dict(layouts={'a':dict(partition='train',physical_site_id='p')}))

    def test_loader_rejects_before_opening_targets_or_receipt(self):
        from cnh_street_e2e_train import load_inputs
        import json
        with patch('pathlib.Path.read_text',return_value=json.dumps(dict(frames=[dict(data_role='Development',split='test')]))), patch('numpy.load') as loader:
            with self.assertRaises(ValueError):load_inputs(['never-read'])
            loader.assert_not_called()
