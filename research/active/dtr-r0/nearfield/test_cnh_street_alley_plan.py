import copy
import unittest
from cnh_street_alley_plan import generate,box_gap,network,union_cells,boundary,check_registry


def config(topology='straight',width=1.8):
    return dict(schema='cnh-street-alley-generator-v1',namespace='/Game/BAResearchAlley',clearance_margin_m=.15,
        sites=[dict(site_id='Alley_Unit',topology=topology,width_m=width,length_m=24.,branch_length_m=16.,
            seed=13,clutter_count=8,clutter_assets=['trashcan','breaker_box','pipe','pallet'],
            wall_material='brick',floor_material='paving',proposed_split='train')])


BOUNDS={'wall':[[-.3,0.,0.],[0.,6.,3.5]],'paving':[[0.,0.,-.2],[1.,1.,0.]],
        'trashcan':[[0,0,0],[.5,.7,1.1]],'breaker_box':[[0,0,0],[.2,.5,.8]],
        'pipe':[[0,0,0],[1.5,.1,.1]],'pallet':[[0,0,0],[.8,1.1,.2]]}


class PlanTest(unittest.TestCase):
    def test_three_topologies_real_bounds_all_outside_margin(self):
        for topology in ('straight','L','T'):
            for width in (1.5,2.6,4.):
                site=generate(config(topology,width),BOUNDS)['sites'][0]
                props=[r for r in site['rows'] if r['role']=='clutter']
                self.assertEqual(len(props),8)
                self.assertGreaterEqual(len(site['clutter_counts']),3)
                for row in site['rows']:
                    if row['role']=='support':continue
                    self.assertGreaterEqual(min(box_gap(row['planned_bounds_m'],[p['low'],p['high']]) for p in site['protected_volumes']),.15-1e-6)

    def test_seed_replays_and_changes_dressing_not_partition(self):
        a=config('T');one=generate(a,BOUNDS)
        self.assertEqual(one,generate(a,BOUNDS))
        b=copy.deepcopy(a);b['sites'][0]['seed']+=1
        two=generate(b,BOUNDS)
        self.assertNotEqual(one['sites'][0]['rows'],two['sites'][0]['rows'])
        self.assertEqual(one['sites'][0]['geometry_signature'],two['sites'][0]['geometry_signature'])
        self.assertEqual(two['sites'][0]['proposed_split'],'train')

    def test_duplicate_geometry_different_identity_or_split_rejected(self):
        a=config();a['sites'].append(dict(a['sites'][0],site_id='Alley_Other',proposed_split='test',seed=14))
        with self.assertRaisesRegex(ValueError,'Duplicate physical geometry'):generate(a,BOUNDS)

    def test_wide_intruder_cannot_be_silently_omitted(self):
        a=config();bad=copy.deepcopy(BOUNDS);bad['trashcan']=[[0,0,0],[50,50,50]]
        with self.assertRaises(ValueError):generate(a,bad)

    def test_registry_blocks_cross_run_relabeling_same_physical_geometry(self):
        a=generate(config(),BOUNDS)['sites'][0]
        bconfig=config();bconfig['sites'][0].update(site_id='Alley_FreshName',proposed_split='test',seed=10)
        b=generate(bconfig,BOUNDS)['sites'][0]
        registry=dict(schema='cnh-street-alley-site-registry-v1',sites=[a])
        with self.assertRaisesRegex(ValueError,'geometry cannot acquire'):check_registry([b],registry)

    def test_uniform_scale_controls_real_prop_height(self):
        a=config();a['sites'][0]['clutter_scales']={'trashcan':.3}
        site=generate(a,BOUNDS)['sites'][0]
        for row in site['rows']:
            if row['asset']=='trashcan':
                self.assertAlmostEqual(row['planned_bounds_m'][1][2]-row['planned_bounds_m'][0][2],
                                       .3*(BOUNDS['trashcan'][1][2]-BOUNDS['trashcan'][0][2]))
        a['sites'][0]['clutter_scales']['trashcan']=float('nan')
        with self.assertRaises(ValueError):generate(a,BOUNDS)


if __name__=='__main__':unittest.main()
