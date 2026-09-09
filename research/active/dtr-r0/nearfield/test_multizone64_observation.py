"""CUDA artificial MZ0 geometry only: no source scenes or model outcomes."""
import math
import unittest

import numpy as np
import torch

from multizone64_observation import observe, geometry, native_events, center_events


class MultizoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')
        torch.set_num_threads(1)
        cls.g = geometry('cuda')

    def blank(self):
        return torch.full((1,360,640),torch.nan,device='cuda',dtype=torch.float64)

    def test_crop_partition_rays_and_radial_plane(self):
        g = self.g
        yy,xx=np.meshgrid(np.arange(360),np.arange(640),indexing='ij')
        f=320/math.tan(math.radians(50))
        right,up=(xx-319.5)/f,-(yy-179.5)/f
        expected=(np.abs(np.rad2deg(np.arctan(right)))<=22.5)&(np.abs(np.rad2deg(np.arctan(up)))<=22.5)
        np.testing.assert_array_equal(g['crop'].cpu(),expected)
        self.assertEqual(int((g['zone_ids']>=0).sum()),int(expected.sum()))
        self.assertEqual(len(torch.unique(g['zone_ids'][g['crop']])),64)
        torch.testing.assert_close(g['center_rays'].norm(dim=1),torch.ones(64,device='cuda',dtype=torch.float64))
        depth=torch.full((1,360,640),2.,device='cuda',dtype=torch.float64)
        packet=observe(depth,8,'nearest')
        for zone in range(64):
            mask=g['zone_ids'].cpu().numpy()==zone
            radial=2*np.sqrt(1+right[mask]**2+up[mask]**2)
            self.assertAlmostEqual(packet['range_m'][0,zone,0].item(),radial.min(),places=12)
        self.assertTrue(packet['valid'].all().item())
        self.assertGreater(packet['range_m'].min().item(),2.)

    def test_thin_near_far_and_one_bin_mean(self):
        depth=self.blank()
        coordinates=(self.g['zone_ids']==27).nonzero()[:12]
        # Three near pixels survive multi_surface support despite dominant far.
        radial=torch.tensor([1.21,1.24,1.28]+[2.51,2.54,2.58]*3,device='cuda',dtype=torch.float64)
        yy,xx=coordinates.T
        depth[0,yy,xx]=radial/self.g['radial_factor'][yy,xx]
        near=observe(depth,8,'nearest');median=observe(depth,8,'median');multi=observe(depth,8,'multi_surface')
        self.assertAlmostEqual(near['range_m'][0,27,0].item(),1.21)
        self.assertAlmostEqual(median['range_m'][0,27,0].item(),(2.51+2.54)/2)
        np.testing.assert_allclose(multi['range_m'][0,27].cpu(),[np.mean([1.21,1.24,1.28]),np.mean([2.51,2.54,2.58])],atol=1e-12)
        self.assertEqual(multi['valid'].sum().item(),2)
        depth[0,yy[3:],xx[3:]]=torch.nan
        single=observe(depth,8,'multi_surface')
        self.assertTrue(single['valid'][0,27,0].item())
        self.assertFalse(single['valid'][0,27,1].item())
        depth[0,yy[2],xx[2]]=torch.nan
        self.assertFalse(observe(depth,8,'multi_surface')['valid'].any().item())

    def test_native_pixel_coordinates_crop_and_unknown(self):
        depth=self.blank()
        #Three points at x1m within HEAD, three at x2m within HEAD_FAR.
        for row,col,x in ((179,319,1.),(179,320,1.),(180,319,1.),
                          (180,320,2.),(179,321,2.),(180,321,2.)):
            depth[0,row,col]=x
        native=native_events(depth)
        self.assertEqual(native['counts'].tolist(),[[0,0,3,3]])
        self.assertEqual(native['events'].tolist(),[[False,False,True,True]])
        # A shallow off-crop ray still lies within HEAD corridor.
        depth[0,179,450]=.2
        self.assertEqual(native_events(depth,False)['counts'][0,2].item(),4)
        self.assertEqual(native_events(depth,True)['counts'][0,2].item(),3)
        invalid=torch.stack([torch.full((360,640),v,device='cuda') for v in (torch.nan,0.,-1.,torch.inf,100.)])
        for mode in ('nearest','median','multi_surface'):
            packet=observe(invalid,1,mode)
            self.assertFalse(packet['valid'].any().item())
            self.assertTrue(torch.isnan(packet['range_m']).all().item())
            self.assertFalse(center_events(packet)['observation_valid'].any().item())
        self.assertFalse(native_events(invalid)['observation_valid'].any().item())

    def test_center_events_single_zone_and_boundaries(self):
        depths=self.blank()
        depths[0,179,319]=1.
        packet=observe(depths,1,'nearest')
        result=center_events(packet)
        self.assertEqual(result['counts'].tolist(),[[0,0,1,0]])
        self.assertTrue(result['events'][0,2].item())
        self.assertFalse(native_events(depths)['events'].any().item())
        # Geometric HEAD near/far transition belongs only to FAR; outer far closed.
        packet['range_m']=torch.tensor([[[1.63,3.13]]],device='cuda',dtype=torch.float64)
        packet['valid']=torch.ones((1,1,2),device='cuda',dtype=torch.bool)
        self.assertEqual(center_events(packet)['counts'].tolist(),[[0,0,0,2]])


if __name__=='__main__':
    unittest.main()
