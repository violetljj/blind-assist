"""Focused geometric, causal and signed-data contracts; no dataset extraction."""
import unittest

import numpy as np

import cnh_raw_radial_features_dev as F


def poses():
    return np.tile(np.eye(4),(16,1,1))


def yaw(degrees):
    c,s = np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


class RadialFeatureTest(unittest.TestCase):
    def test_identity_and_layout_signed(self):
        # Distinct zone/bin/time offsets expose sector-axis transpositions.
        zone = np.arange(64).reshape(8,8)
        z = -1000 + np.arange(16)[:,None,None,None]*2 + zone[None,:,:,None]*3 + np.arange(16)
        a = np.broadcast_to(zone,(16,8,8)).copy()
        result = F.extract_features(z,a,poses())
        self.assertEqual(result['current'].shape,(13,136))
        self.assertEqual(result['support'].shape,(13,24))
        self.assertEqual(result['temporal_delta'].shape,(13,384))
        expected = z[3].reshape(64,16)[F.SECTORS].mean(1).reshape(128)
        np.testing.assert_allclose(result['current'][0,:128],expected)
        self.assertTrue((result['current'][:,:128]<0).all())
        np.testing.assert_allclose(result['current'][0,128:],
                                   np.log1p(a[3].ravel())[F.SECTORS].mean(1),rtol=1e-6)
        delta = result['temporal_delta'].reshape(13,3,8,16)
        support = result['support'].reshape(13,3,8)
        for f,t in enumerate(F.FRAMES):
            for j,lag in enumerate(F.LAGS):
                if t >= lag:
                    np.testing.assert_allclose(delta[f,j],2*lag,atol=1e-5)
                    np.testing.assert_array_equal(support[f,j],1)
                else:
                    self.assertTrue(np.isnan(delta[f,j]).all())
                    np.testing.assert_array_equal(support[f,j],0)

    def test_small_yaw_backward_direction_analytic(self):
        # Past radiance linear in grid column, so bilinear sampling is exact.
        z = np.broadcast_to(np.arange(8)[None,None,:,None],(16,8,8,16)).copy()
        p = poses(); p[3,:3,:3] = yaw(3)
        result = F.extract_features(z,np.zeros((16,8,8)),p)
        actual = result['temporal_delta'][0,:128].reshape(8,16)
        # Independent scalar pinhole equations: positive current yaw samples
        # farther RIGHT in past image. Vertical coordinates also change.
        expected = []
        c,s = np.cos(np.deg2rad(3)),np.sin(np.deg2rad(3))
        for sector in F.SECTORS:
            values = []
            for index in sector:
                row,col = divmod(int(index),8)
                x,y = F.CENTERS[col],F.CENTERS[row]
                depth = c-s*x
                pc = ((c*x+s)/depth + F.EDGE)*8/(2*F.EDGE)-.5
                pr = (y/depth + F.EDGE)*8/(2*F.EDGE)-.5
                if 0<=pc<=7 and 0<=pr<=7:
                    values.append(col-pc)
            expected.append(np.mean(values) if values else np.nan)
        np.testing.assert_allclose(actual,np.repeat(np.array(expected)[:,None],16,axis=1),rtol=1e-6)
        self.assertTrue((actual[np.isfinite(actual)] < 0).all())
        # Rotate both endpoints equally: relative rotation is identity.
        p[2,:3,:3] = yaw(3)
        both = F.extract_features(z,np.zeros((16,8,8)),p)
        np.testing.assert_allclose(both['temporal_delta'][0,:128],0,atol=1e-6)

    def test_translation_invariance(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=(16,8,8,16)); a = np.ones((16,8,8)); p = poses()
        p[:,:3,3] = rng.normal(size=(16,3))*1e6
        p[4,:3,3] = np.nan  # Ignored, including validation.
        first,second = F.extract_features(z,a,poses()),F.extract_features(z,a,p)
        for key in first:
            np.testing.assert_array_equal(first[key],second[key])

    def test_future_mutation_causality(self):
        rng = np.random.default_rng(7)
        z = rng.normal(size=(16,8,8,16)); a = rng.random((16,8,8)); p = poses()
        before = F.extract_features(z,a,p)
        z[9:] *= 100; a[9:] += 50; p[9:,:3,:3] = yaw(22)
        after = F.extract_features(z,a,p)
        for key in before:
            np.testing.assert_array_equal(before[key][:6],after[key][:6])

    def test_missing_fov_behind_camera_and_negative_past(self):
        z = np.ones((16,8,8,16)); a = np.zeros((16,8,8)); p = poses()
        for angle in (70,180):
            p[3,:3,:3] = yaw(angle)
            result = F.extract_features(z,a,p)
            np.testing.assert_array_equal(result['support'][0],0)
            self.assertTrue(np.isnan(result['temporal_delta'][0]).all())
            self.assertTrue(np.isfinite(result['current']).all())


if __name__ == '__main__':
    unittest.main()
