#!/usr/bin/env python3

import unittest

import numpy as np


class DepthartQairtRewriteTest(unittest.TestCase):
    def test_supported_einsums_are_batched_matmul(self) -> None:
        rng = np.random.default_rng(7)
        x0 = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
        w0 = rng.standard_normal((3, 6, 4), dtype=np.float32)
        np.testing.assert_allclose(
            np.einsum("bkdl,kcd->bkcl", x0, w0),
            np.matmul(w0, x0),
            rtol=1e-6,
            atol=1e-6,
        )
        x1 = rng.standard_normal((2, 3, 7, 5), dtype=np.float32)
        w1 = rng.standard_normal((3, 4, 7), dtype=np.float32)
        np.testing.assert_allclose(
            np.einsum("bkrl,kdr->bkdl", x1, w1),
            np.matmul(w1, x1),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
