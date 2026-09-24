import json
from pathlib import Path
import unittest
from cnh_street_expansion_build import design,rotated_bounds,validate


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(Path(__file__).with_name('cnh_street_expansion_a.json').read_text())
        self.wall=([-1.12953,-5.50019,-.02216],[-.00360,.02221,3.50158])
        self.paving=([0,0,0],[1,1,.2])

    def test_native_wall_rows_leave_exact_three_metre_gap(self):
        plan=design(self.spec,self.wall,self.paving)
        walls=[r for r in plan['actors'] if r['asset']=='wall']
        self.assertEqual(len(walls),16)
        for row in walls:
            low,high=rotated_bounds(*self.wall,row['yaw'])
            low=[x+y for x,y in zip(low,row['location_m'])]
            high=[x+y for x,y in zip(high,row['location_m'])]
            self.assertEqual(row['scale'],[1,1,1])
            if 'north' in row['name']:self.assertAlmostEqual(low[1],1.5)
            else:self.assertAlmostEqual(high[1],-1.5)

    def test_ground_area_has_no_overlap_and_flat_top(self):
        plan=design(self.spec,self.wall,self.paving)
        tiles=[r for r in plan['actors'] if r['asset']=='paving']
        area=sum(r['scale'][0]*r['scale'][1] for r in tiles)
        self.assertAlmostEqual(area,72+3*plan['alley_length_m']+144)
        for row in tiles:
            self.assertAlmostEqual(row['location_m'][2]+.2,0)
            self.assertGreater(row['scale'][0],0)
            self.assertLessEqual(row['scale'][0],1)

    def test_destination_and_dimensions_cannot_silently_expand(self):
        self.spec['map_asset']=self.spec['preserved_source_map']
        with self.assertRaises(ValueError):validate(self.spec)

    def test_wrong_actual_asset_bounds_rejected(self):
        with self.assertRaises(ValueError):design(self.spec,([-1,-2,0],[0,0,1]),self.paving)

    def test_v2_native_width_retained_ground_and_plaza_enclosure(self):
        self.spec.update(map_asset='/Game/BAResearchExpansion/BrickServiceA_V2',
            asset_root='/Game/BAResearchExpansion/BrickServiceA_V2_Assets')
        plan=design(self.spec,([-.4,-3.,0.],[0.,0.,3.5]),self.paving)
        self.assertEqual(plan['alley_length_m'],24.)
        self.assertTrue(any(x['name']=='surrounding_ground' for x in plan['actors']))
        self.assertEqual(len([x for x in plan['actors'] if x['name'].startswith('plaza_boundary_')]),12)
        self.assertTrue(all(x['scale']==[1,1,1] for x in plan['actors'] if x['asset']=='wall'))


if __name__=='__main__':unittest.main()
