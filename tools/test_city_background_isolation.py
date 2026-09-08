import unittest
import numpy as np

from city_background_isolation import audit, frustum_candidates


class BackgroundIsolationTests(unittest.TestCase):
    def test_far_building_and_intersecting_large_bounds_retained(self):
        bounds = np.array([[[10000,-1,-1],[10001,1,1]],
                           [[-200,-2,-2],[2,2,2]], [[-20,-1,-1],[-10,1,1]],
                           [[1,100,0],[2,101,1]]])
        self.assertEqual(frustum_candidates(bounds, dict(x=0,y=0,z=0,yaw=0)).tolist(),
                         [True, True, False, False])

    def test_invalid_bounds_cannot_disappear(self):
        bounds = [[[float('nan'),0,0],[1,1,1]], [[5,0,0],[1,1,1]]]
        self.assertEqual(frustum_candidates(bounds, dict(x=0,y=0,z=0,yaw=0)).tolist(), [True, True])

    def test_topdown_ue_pitch(self):
        bounds = [[[-1,-1,-10],[1,1,-9]], [[-1,-1,9],[1,1,10]]]
        self.assertEqual(frustum_candidates(bounds, dict(x=0,y=0,z=0,yaw=0,pitch=-90)).tolist(), [True, False])

    def test_guid_alias_does_not_merge_and_hlod_blocks_admission(self):
        rows = [dict(actor_package=f'/Map/{i}', guid="<Struct Guid addr>",
                     bounds_m=dict(min=[10,-1,-1], max=[11,1,1]),
                     native_class=cls, label='BLDG')
                for i, cls in enumerate(['/Script/Engine.Actor', '/Script/Engine.WorldPartitionHLOD',
                                         '/Script/Engine.DirectionalLight'])]
        views = [dict(view_id=s, split=s, camera=dict(x=0,y=0,z=0,yaw=0)) for s in ['train','test']]
        result = audit(dict(map_asset='/Map', rows=rows), views)
        self.assertEqual(result['views'][0]['candidate_count'], 2)
        self.assertEqual(result['cross_split_overlaps'][0]['direct_actor_count'], 1)
        self.assertEqual(result['cross_split_overlaps'][0]['shared_hlod_count'], 1)
        self.assertEqual(result['visible_background_isolation'], 'UNVERIFIED')


if __name__ == '__main__':
    unittest.main()
