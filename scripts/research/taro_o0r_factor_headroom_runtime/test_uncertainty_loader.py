from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_factor_headroom_runtime.uncertainty_loader import (
    UncertaintyArtifactError,
    load_factory_bound_uncertainty_model,
)
from scripts.research.taro_o0r_source_adapter_runtime import test_source_adapter as fixtures
from scripts.research.taro_o0r_truth_materializer_runtime.materializer import (
    canonical_sha256,
    seal_record,
    uncertainty_model_artifact,
    uncertainty_model_receipt,
)


class UncertaintyLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = fixtures.fitted_model(1)
        cls.artifact = uncertainty_model_artifact(cls.original)
        cls.receipt = uncertainty_model_receipt(cls.original)

    def test_roundtrip_restores_factory_bound_immutable_model(self) -> None:
        # Canonical JSON sorts object keys, so exercise the persisted receipt's
        # actual order instead of relying on the in-memory construction order.
        persisted_receipt = json.loads(json.dumps(self.receipt, sort_keys=True))
        loaded = load_factory_bound_uncertainty_model(
            self.artifact,
            persisted_receipt,
            expected_artifact_canonical_sha256=canonical_sha256(self.artifact),
            expected_model_sha256=self.original.content_sha256,
        )
        loaded._assert_integrity()
        self.assertEqual(loaded.content_sha256, self.original.content_sha256)
        self.assertTrue(all(not cell.values.flags.writeable for cell in loaded.cells))
        self.assertEqual(loaded.resolve(2, 1.0, "scale_log_abs_residual")["model_sha256"], self.original.content_sha256)

    def test_model_swap_and_resealed_cell_tamper_fail_closed(self) -> None:
        with self.assertRaises(UncertaintyArtifactError) as swap:
            load_factory_bound_uncertainty_model(
                self.artifact,
                self.receipt,
                expected_artifact_canonical_sha256=canonical_sha256(self.artifact),
                expected_model_sha256="0" * 64,
            )
        self.assertEqual(swap.exception.code, "UNCERTAINTY_MODEL_BINDING_MISMATCH")

        tampered = copy.deepcopy(self.artifact)
        tampered["cells"][0]["values"][0] += 1.0
        tampered.pop("content_sha256")
        tampered = seal_record(tampered)
        with self.assertRaises(UncertaintyArtifactError) as cell:
            load_factory_bound_uncertainty_model(
                tampered,
                self.receipt,
                expected_artifact_canonical_sha256=canonical_sha256(tampered),
                expected_model_sha256=self.original.content_sha256,
            )
        self.assertEqual(cell.exception.code, "UNCERTAINTY_MODEL_CELL_HASH_MISMATCH")


if __name__ == "__main__":
    unittest.main()
