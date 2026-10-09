"""Core-trace evidence checks; fixtures remain in canonical ignored payloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import rgb_body_query_thin_labels as thin
from rgb_body_query_geometry import camera_queries
from rgb_body_query_reference_eval import interval_labels, MIN_SUPPORT, UNKNOWN


class ThinLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[4]
        base = repo/'artifacts.local/work/rgb-body-query-label-completion-dev-20261009/test-fixtures'
        base.mkdir(parents=True, exist_ok=True)
        cls.fixture = Path(tempfile.mkdtemp(prefix='thin-label-', dir=base))

    def test_core_negatives_require_sixteen_nonempty_reachable_rays(self):
        for n, state in [(0, 'UNKNOWN'), (15, 'UNKNOWN'), (16, 'NEGATIVE_CORE_INTRUSION')]:
            z = np.full((1, max(n, 1)), 4.)
            mask = np.zeros(z.shape, bool)
            mask[0, :n] = True
            _, record = interval_labels(z, z, mask, np.zeros_like(z), np.zeros_like(z), camera_queries()[0])
            self.assertEqual(thin.core_state(record), state)

    def test_inherited_disagreement_missing_and_boundary_crossing_stay_unknown(self):
        q = dict(name='narrow', low=[-.1, -.1, 1.], high=[.1, .1, 1.2])
        cres = np.full((1, MIN_SUPPORT), 1.1)
        for other in (np.nan, 2., 1.3):
            label, record = interval_labels(cres, np.full_like(cres, other), np.ones(cres.shape, bool), np.zeros_like(cres), np.zeros_like(cres), q)
            self.assertTrue(np.all(label == UNKNOWN))
            self.assertEqual(thin.core_state(record), 'UNKNOWN')

    def test_original_closed_boundary_support_is_inherited(self):
        z = np.full((1, MIN_SUPPORT), 3.)
        for query in (camera_queries()[0], camera_queries()[3]):
            _, record = interval_labels(z, z, np.ones(z.shape, bool), np.zeros_like(z), np.zeros_like(z), query)
            self.assertEqual(thin.core_state(record), 'POSITIVE_CORE_INTRUSION')

    def test_raster_rejects_coordinates_from_outside_the_source_image(self):
        trace = dict(points_native=[[1., 1.], [1., 5.]], width_native_px=1)
        raster = thin.raster_trace(trace, (8, 8))
        self.assertEqual(int(raster.sum()), 5)
        for points in ([[1., 1.], [8., 5.]], [[1., 1.], [np.nan, 5.]]):
            with self.assertRaises(ValueError):
                thin.raster_trace(dict(trace, points_native=points), (8, 8))

    def test_invalid_prediction_pixels_and_their_interpolated_neighbors_stay_invalid(self):
        path = self.fixture/'invalid-prediction.npz'
        depth = np.full((2, 2), 4., np.float32)
        valid = np.array([[False, True], [True, True]])
        np.savez(path, depth=depth, valid_mask=valid)
        scaled, kept = thin.scaled_depth(path, (4, 4))
        self.assertFalse(kept[0, 0])
        self.assertFalse(kept[1, 1])
        self.assertTrue(kept[-1, -1])
        self.assertTrue(np.isnan(scaled[~kept]).all())
        self.assertEqual(float(scaled[-1, -1]), 4.)
        np.savez(path, depth=np.array([[np.nan, 2.], [-1., 2.]], np.float32))
        scaled, kept = thin.scaled_depth(path, (2, 2))
        np.testing.assert_array_equal(kept, [[False, True], [False, True]])
        self.assertTrue(np.isnan(scaled[~kept]).all())

    def cached_source_fixture(self, ambiguous=False):
        root = self.fixture/('ambiguous-cache' if ambiguous else 'cache')
        old = root/'artifacts.local/work/sanpo-coverage-20260927'
        seq = root/'artifacts.local/work/rgb-body-query-dev-20261009/sequence'
        old.mkdir(parents=True, exist_ok=True)
        seq.mkdir(parents=True, exist_ok=True)
        row = dict(session='old-session', camera='camera_chest', frame=162)
        base = 'old-session/camera_chest/left'
        entries = [dict(path=base+'/'+kind+'/000162.float16.gz', sha256=kind) for kind in ('depth_maps', 'zed_depth_maps')]
        if ambiguous:
            entries.append(dict(path=base+'/depth_maps/000162.other.gz', sha256='extra'))
        (old/'acquisition_receipt.json').write_text(json.dumps(dict(files=entries)), 'utf-8')
        (seq/'planned_files.json').write_text(json.dumps(dict(session='seq-session', camera='camera_chest', frames=[], references=[dict(path='description.json', sha256='description')])), 'utf-8')
        (seq/'acquisition_receipt.json').write_text(json.dumps(dict(verified_files=[])), 'utf-8')
        return root, row

    def test_cached_multidot_depth_suffix_preserves_receipt_identity(self):
        root, row = self.cached_source_fixture()
        with patch.object(thin, 'source_frames', return_value=[row]):
            source = thin.sources(root)['old-session', 'camera_chest', 162]
        self.assertEqual(source['cres_path'].name, '000162.float16.gz')
        self.assertEqual(source['zed_path'].name, '000162.float16.gz')
        self.assertEqual(source['cres_sha256'], 'depth_maps')
        self.assertEqual(source['zed_sha256'], 'zed_depth_maps')
        self.assertTrue(source['cres_path'].is_relative_to(root/'artifacts.local/datasets'))

    def test_ambiguous_cached_depth_identity_is_rejected(self):
        root, row = self.cached_source_fixture(ambiguous=True)
        with patch.object(thin, 'source_frames', return_value=[row]), self.assertRaisesRegex(ValueError, 'Missing unique cached cres'):
            thin.sources(root)

    def test_changed_rgb_source_is_rejected_before_reference_or_prediction_use(self):
        source_path = self.fixture/'changed-source.png'
        source_path.write_bytes(b'fixture: not the original RGB')
        key = ('fixture-session', 'camera_chest', 1)
        trace = dict(session=key[0], camera=key[1], frame=key[2], name='core', annotation_provenance='agent_visual_review')
        annotations = self.fixture/'identity-annotations.json'
        annotations.write_text(json.dumps(dict(traces=[trace])), 'utf-8')
        source = dict(rgb_path=source_path, rgb_sha256='frozen-different-hash')
        with patch.object(thin, 'sources', return_value={key: source}), patch.object(thin, 'predictions', return_value={}), patch.object(thin, 'load_depth') as loader:
            with self.assertRaisesRegex(ValueError, 'Changed source rgb'):
                thin.run(self.fixture, [annotations], self.fixture/'identity-result', budget_s=10)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
