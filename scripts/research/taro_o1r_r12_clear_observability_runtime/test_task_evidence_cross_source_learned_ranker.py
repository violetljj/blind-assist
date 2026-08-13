import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as subject


class CrossSourceLearnedRankerTest(unittest.TestCase):
    def test_model_family_is_frozen(self) -> None:
        self.assertEqual((12013, 12031, 12047), subject.SEEDS)
        self.assertEqual((32, 16), subject.HIDDEN_WIDTHS)
        self.assertEqual(0.75, subject.RESIDUAL_SCALE)
        self.assertEqual(300, subject.EPOCHS)

    def test_reference_blocks_are_reference_local(self) -> None:
        class Record:
            def __init__(self, reference_id: str, value: float):
                self.reference_id = reference_id
                self.features = np.full(4, value, dtype=np.float64)

        records = [Record("a", 1.0), Record("a", 3.0), Record("b", 100.0), Record("b", 104.0)]
        _raw, z, unit = subject._reference_blocks(records)  # type: ignore[arg-type]
        np.testing.assert_allclose(z[:, 0], [-1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(unit[:, 0], [0.0, 1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
