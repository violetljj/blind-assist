"""Independent synthetic checks of the real sensor-ray query contract."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rgb_body_query_3rscan import (
    FREE_RAY, IGNORE, POSITIVE, UNKNOWN, color_coordinates, optical_z,
    parse_info, sample_prediction, selected_groups, sensor_labels,
)


class RealSensorQueryTests(unittest.TestCase):
    def test_sensor_shift_optical_z_and_extrinsics_contract(self):
        z = optical_z(np.array([[0, 2000]], np.uint16), 1000)
        self.assertTrue(np.isnan(z[0, 0]))
        self.assertEqual(z[0, 1], 2.)
        # Pixel u=1, fx=1 is ray (1,0,1): axial Z=2 gives X=2.
        box = dict(name='off_axis', low=[1.9, -.1, 1.9], high=[2.1, .1, 2.1])
        label, counts = sensor_labels(z, np.eye(3), box, min_support=1)
        self.assertEqual(label[0, 1], POSITIVE)
        self.assertEqual(counts['positive_pixels'], 1)
        radial_mistake = np.array([[np.nan, 2./np.sqrt(2)]])
        self.assertEqual(sensor_labels(radial_mistake, np.eye(3), box, min_support=1)[0][0, 1], UNKNOWN)
        with self.assertRaises(ValueError):
            optical_z(np.array([[2000]]), 0)
        eye = ' '.join(map(str, np.eye(4).reshape(-1)))
        fields = dict(m_calibrationColorExtrinsic=eye, m_calibrationDepthExtrinsic=eye,
                      m_calibrationColorIntrinsic=eye, m_calibrationDepthIntrinsic=eye,
                      m_depthShift='1000', m_colorHeight='3', m_colorWidth='4',
                      m_depthHeight='2', m_depthWidth='2')
        payload = '\n'.join(f'{k} = {v}' for k, v in fields.items())
        self.assertEqual(parse_info(payload)['shift'], 1000)
        shifted = np.eye(4); shifted[0, 3] = .1
        fields['m_calibrationDepthExtrinsic'] = ' '.join(map(str, shifted.reshape(-1)))
        with self.assertRaises(ValueError):
            parse_info('\n'.join(f'{k} = {v}' for k, v in fields.items()))

    def test_distinct_intrinsics_project_instead_of_resize(self):
        kd = np.array([[2., 0, 0], [0, 4., 0], [0, 0, 1.]])
        kc = np.array([[6., 0, 1], [0, 8., 2], [0, 0, 1.]])
        mx, my = color_coordinates(kd, kc, (2, 2))
        np.testing.assert_allclose(mx, [[1., 4.], [1., 4.]])
        np.testing.assert_allclose(my, [[2., 2.], [4., 4.]])
        # Both maps share a tiny 2x2 depth shape; RGB indices deliberately differ.
        prediction = np.arange(42, dtype=np.float32).reshape(6, 7)+1
        sampled = sample_prediction(prediction, mx, my)
        np.testing.assert_allclose(sampled, [[16., 19.], [30., 33.]])
        self.assertNotEqual(sampled[0, 0], prediction[0, 0])

    def test_first_return_missing_occlusion_and_free_ray(self):
        k = np.array([[100., 0, 0], [0, 100., 0], [0, 0, 1.]])
        q = dict(name='ray_interval', low=[-1., -1., 1.], high=[1., 1., 3.])
        z = np.array([[.5, 2., 4., 0., np.nan]])
        label, counts = sensor_labels(z, k, q, min_support=1)
        np.testing.assert_array_equal(label, [[UNKNOWN, POSITIVE, FREE_RAY, UNKNOWN, UNKNOWN]])
        self.assertEqual(counts['state'], 'POSITIVE')
        self.assertEqual(counts['unknown_pixels'], 3)
        # Valid depth outside RGB overlap is still unobserved for the RGB task.
        hidden, hc = sensor_labels(np.array([[4.]]), np.eye(3), q,
                                   observed=np.array([[False]]), min_support=1)
        self.assertEqual(hidden[0, 0], UNKNOWN)
        self.assertEqual(hc['state'], 'UNKNOWN')
        free, fc = sensor_labels(np.array([[4.]]), np.eye(3), q, min_support=1)
        self.assertEqual(free[0, 0], FREE_RAY)
        self.assertEqual(fc['state'], 'FREE_ON_SAMPLED_RAYS')

    def test_closed_xyz_bounds_and_out_of_box_rays(self):
        q = dict(name='closed_z', low=[-1., -1., 1.], high=[1., 1., 3.])
        k = np.array([[100., 0, 0], [0, 100., 0], [0, 0, 1.]])
        z = np.array([[1., 3., 1.-1e-6, 3.+1e-6]])
        np.testing.assert_array_equal(sensor_labels(z, k, q, min_support=1)[0],
                                      [[POSITIVE, POSITIVE, UNKNOWN, FREE_RAY]])
        kx = np.array([[10., 0, 1.], [0, 10., 0], [0, 0, 1.]])
        qx = dict(name='closed_xyz', low=[-.3, 0., 3.], high=[.3, 0., 3.])
        label, c = sensor_labels(np.full((1, 4), 3.), kx, qx, min_support=1)
        np.testing.assert_array_equal(label, [[POSITIVE, POSITIVE, POSITIVE, IGNORE]])
        self.assertEqual(c['domain_pixels'], 3)

    def test_environment_selection_does_not_split_rescans(self):
        # One physical environment owns a reference and two rescans. All are
        # cached, but only one representative can enter the grouped interface.
        check_root = Path(__file__).resolve().parents[4]/'artifacts.local/work/rgb-body-query-cross-session-dev-20261009/check'
        check_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=check_root) as temp:
            root = Path(temp)
            meta = []
            for i in range(12):
                env = f'environment-{i:02d}'
                meta.append(dict(type='train', reference=env,
                                 scans=[dict(reference=f'{env}-rescan-{j}') for j in range(2)]))
                for scan in [env]+[s['reference'] for s in meta[-1]['scans']]:
                    (root/scan).mkdir()
                    (root/scan/'sequence.zip').write_bytes(b'fixture existence marker')
            # Official non-train groups must not leak into the selection.
            meta.append(dict(type='test', reference='protected-test', scans=[]))
            (root/'protected-test').mkdir()
            (root/'protected-test/sequence.zip').write_bytes(b'fixture')
            (root/'3RScan.json').write_text(json.dumps(meta), encoding='utf-8')
            groups = selected_groups(root)
            self.assertEqual(len(groups), 12)
            self.assertEqual(len({g['environment'] for g in groups}), 12)
            self.assertTrue(all(g['scan'] == g['environment'] for g in groups))
            splits = {split: {g['environment'] for g in groups if g['split'] == split}
                      for split in ('train', 'cal', 'validation')}
            self.assertEqual([len(splits[s]) for s in splits], [7, 2, 3])
            self.assertTrue(splits['train'].isdisjoint(splits['validation']))
            self.assertTrue(splits['train'].isdisjoint(splits['cal']))
            self.assertTrue(splits['cal'].isdisjoint(splits['validation']))
            self.assertNotIn('protected-test', {g['environment'] for g in groups})
            # If a reference is absent, representative rescan retains group ID.
            (root/'environment-00/sequence.zip').unlink()
            fallback = next(g for g in selected_groups(root) if g['environment'] == 'environment-00')
            self.assertEqual(fallback['scan'], 'environment-00-rescan-0')

    def test_prediction_bilinear_sampling_is_read_only_and_invalid_safe(self):
        prediction = np.array([[1., 3.], [5., 7.]], np.float32)
        original = prediction.copy()
        prediction.flags.writeable = False
        mx = np.array([[.5, -1., 0.]], np.float32)
        my = np.array([[.5, 0., 0.]], np.float32)
        out = sample_prediction(prediction, mx, my)
        self.assertEqual(out[0, 0], 4.)
        self.assertTrue(np.isnan(out[0, 1]))
        self.assertEqual(out[0, 2], 1.)
        np.testing.assert_array_equal(prediction, original)
        partial = np.array([[1., np.nan], [5., 7.]], np.float32)
        self.assertTrue(np.isnan(sample_prediction(partial, mx[:, :1], my[:, :1])[0, 0]))


if __name__ == '__main__':
    unittest.main()
