import copy
import unittest
from unittest.mock import patch
from cnh_route_source_compare_adapter import validate_spec, DEVELOPMENT_SCOPE
from cnh_route_source_capture import validate_insertions


class DevelopmentContractTest(unittest.TestCase):
    def spec(self):
        pose=dict(x=0.,y=0.,z=1.6,pitch=0.,yaw=0.,roll=0.)
        objects=[dict(id=i,center_m=[4.,0.,1.],scale=[1.,1.,1.],rotation_deg=dict(pitch=0.,yaw=0.,roll=0.),hidden=False) for i in (1,254)]
        clips=[]
        for name in ('centre','boundary','outside','removed'):
            obs=copy.deepcopy(objects);obs[0]['hidden']=name=='removed'
            clips.append(dict(id=name,trajectory_model='piecewise_linear_fixed_orientation',poses=[dict(pose,x=i*.1) for i in range(40)],insertions=obs))
        return dict(scope=DEVELOPMENT_SCOPE,scene_layer=DEVELOPMENT_SCOPE,benchmark_eligible=False,data_role='Development',
            map_asset='/Game/BAResearchSlice/Street200V7',map_file='unused',map_sha256=__import__('hashlib').sha256(b'map').hexdigest(),
            nominal_sample_interval_s=.1,native_material_policy='STREET_TRANSIENT_ZERO_WPO_PDO',exposure_ev100=13.2,
            assets=[dict(id=i,mesh_asset='/Game/mesh') for i in (1,254)],
            layouts=[dict(layout_id=str(i),physical_site_id='Street200V7-single-street-block',environment_category='sidewalk',camera=pose,clips=copy.deepcopy(clips)) for i in range(2)])

    @patch('pathlib.Path.read_bytes',return_value=b'map')
    def test_development_accepts_shared_block_and_40_poses(self,_):
        spec=self.spec();validate_spec(spec);validate_insertions(spec)
        spec['data_role']='test'
        with self.assertRaises(ValueError):validate_spec(spec)

    def test_wrong_pose_count_and_orientation_rejected(self):
        spec=self.spec();spec['layouts'][0]['clips'][0]['poses'].pop()
        with self.assertRaises(ValueError):validate_insertions(spec)
        spec=self.spec();spec['layouts'][0]['clips'][0]['poses'][1]['yaw']=5.
        with self.assertRaises(ValueError):validate_insertions(spec)


if __name__=='__main__':unittest.main()
