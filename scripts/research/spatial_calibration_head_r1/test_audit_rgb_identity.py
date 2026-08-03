#!/usr/bin/env python3

import unittest

import numpy as np

from audit_rgb_identity import candidate_edges, phash64, variants


class RGBIdentityAuditTest(unittest.TestCase):
    def test_variants_and_phash_are_deterministic(self) -> None:
        image = np.arange(64 * 96, dtype=np.uint8).reshape(64, 96)
        generated = variants(image)
        self.assertEqual(set(generated), {"original", "horizontal_mirror", "center_crop_90", "center_crop_90_horizontal_mirror", "center_crop_80", "center_crop_80_horizontal_mirror"})
        self.assertEqual(phash64(image), phash64(image.copy()))

    def test_cross_parent_near_hash_is_candidate(self) -> None:
        frames = [
            {"visit_id": "a", "payload_sha256": "a", "phash_variants": {"original": "0000000000000000"}},
            {"visit_id": "b", "payload_sha256": "b", "phash_variants": {"original": "0000000000000003"}},
            {"visit_id": "a", "payload_sha256": "c", "phash_variants": {"original": "0000000000000001"}},
        ]
        edges = candidate_edges(frames)
        self.assertEqual([(row["left_index"], row["right_index"]) for row in edges], [(0, 1), (1, 2)])


if __name__ == "__main__":
    unittest.main()
