"""Focused synthetic checks for measured pole geometry, not device accuracy."""
import copy
import unittest

import numpy as np

from measured_pole_geometry import analyse, axis_rotation, synthetic_example


def project_pole(doc, points):
    for view in doc['views']:
        K, R, Q, C = (np.asarray(view[k], float) for k in
                       ('K', 'camera_to_world_R', 'image_ray_to_camera', 'camera_center_m'))
        q = (np.asarray(points) - C) @ R @ Q
        uv = q @ K.T
        view['line_endpoints_px'] = (uv[:, :2] / uv[:, 2, None]).tolist()


class MeasuredPoleGeometryTests(unittest.TestCase):
    def test_known_geometry_and_no_input_mutation(self):
        doc = synthetic_example(); original = copy.deepcopy(doc)
        result = analyse(doc)
        self.assertEqual(result['status'], 'RECONSTRUCTED_VALIDATION_CONSISTENT')
        np.testing.assert_allclose(result['reconstruction']['anchor_world_m'], [.3, 0., -2.5], atol=1e-10)
        self.assertLess(result['validation']['projected_axis']['max_endpoint_residual_px'], 1e-9)
        self.assertFalse(result['verified_distance'])
        self.assertEqual(result['scenario_envelope']['scenario_count'], 45)
        self.assertFalse(result['scenario_envelope']['third_view_used'])
        self.assertEqual(doc, original)

    def test_held_out_wrong_correspondence_cannot_move_fit_or_envelope(self):
        good = synthetic_example(); bad = copy.deepcopy(good)
        for p in bad['views'][2]['line_endpoints_px']:
            p[0] += 20.
        a, b = analyse(good), analyse(bad)
        self.assertEqual(b['status'], 'RECONSTRUCTED_VALIDATION_CONFLICT')
        self.assertEqual(a['reconstruction'], b['reconstruction'])
        self.assertEqual(a['scenario_envelope'], b['scenario_envelope'])
        self.assertAlmostEqual(b['validation']['projected_axis']['max_endpoint_residual_px'], 20.)

    def test_zero_baseline_and_pure_rotation_are_unobservable(self):
        for yaw in (0., .1):
            doc = synthetic_example()
            doc['views'][1]['camera_center_m'] = doc['views'][0]['camera_center_m'].copy()
            R = np.asarray(doc['views'][0]['camera_to_world_R'])
            doc['views'][1]['camera_to_world_R'] = (axis_rotation(1, yaw) @ R).tolist()
            project_pole(doc, [[.3, .3, -2.5], [.3, 1.4, -2.5]])
            self.assertEqual(analyse(doc)['status'], 'ZERO_BASELINE_OR_PURE_ROTATION')

    def test_forward_motion_center_pole_is_degenerate(self):
        doc = synthetic_example()
        for view, z in zip(doc['views'], [0., -.2, -.1]):
            view['camera_center_m'] = [0., 1.5, z]
        project_pole(doc, [[0., .3, -2.5], [0., 1.4, -2.5]])
        self.assertEqual(analyse(doc)['status'], 'DEGENERATE_LINE_PLANES')

    def test_non_vertical_fit_line_is_not_reconstructed(self):
        doc = synthetic_example(); doc['views'][0]['line_endpoints_px'][0][0] += 40.
        result = analyse(doc)
        self.assertEqual(result['status'], 'NON_VERTICAL_FIT_LINE')
        self.assertIsNone(result['reconstruction']['anchor_world_m'])

    def test_behind_camera_is_not_a_front_distance(self):
        doc = synthetic_example()
        project_pole(doc, [[.3, .3, 2.5], [.3, 1.4, 2.5]])
        self.assertEqual(analyse(doc)['status'], 'BEHIND_CAMERA_OR_UNPROJECTABLE')

    def test_missing_measurement_and_reflected_physical_rotation_stop(self):
        doc = synthetic_example(); doc['views'][0]['image_ray_to_camera'] = None
        result = analyse(doc)
        self.assertEqual(result['status'], 'MISSING_MEASUREMENT')
        self.assertIn('views[0].image_ray_to_camera', result['missing_fields'])
        self.assertIsNone(result['reconstruction'])
        doc = synthetic_example(); doc['views'][0]['camera_to_world_R'] = np.diag([1., -1., 1.]).tolist()
        self.assertEqual(analyse(doc)['status'], 'INVALID_MEASUREMENT')
        doc = synthetic_example(); doc['coordinate_frame']['units'] = 'centimetre'
        self.assertEqual(analyse(doc)['status'], 'INVALID_MEASUREMENT')

    def test_reflected_image_ray_basis_preserves_world_reconstruction(self):
        doc = synthetic_example(); S = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for view in doc['views']:
            view['image_ray_to_camera'] = S.tolist()
            view['K'] = (S @ np.asarray(view['K']) @ S).tolist()
            view['line_endpoints_px'] = np.asarray(view['line_endpoints_px'])[:, ::-1].tolist()
            view['image']['width'], view['image']['height'] = view['image']['height'], view['image']['width']
        result = analyse(doc)
        self.assertEqual(result['status'], 'RECONSTRUCTED_VALIDATION_CONSISTENT')
        np.testing.assert_allclose(result['reconstruction']['anchor_world_m'], [.3, 0., -2.5], atol=1e-10)

    def test_image_ray_mapping_cannot_flip_physical_optical_axis(self):
        doc = synthetic_example()
        for view in doc['views']:
            view['image_ray_to_camera'] = np.diag([1., 1., -1.]).tolist()
        project_pole(doc, [[.3, .3, 2.5], [.3, 1.4, 2.5]])
        result = analyse(doc)
        self.assertEqual(result['status'], 'INVALID_MEASUREMENT')
        self.assertIsNone(result['reconstruction'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
