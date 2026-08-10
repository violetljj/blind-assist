#!/usr/bin/env python3

import unittest

from bind_ag_st_source_native_boundary_rgb import _sha256_bytes


class SourceNativeBoundaryRgbBindingTest(unittest.TestCase):
    def test_byte_receipt_is_stable(self) -> None:
        self.assertEqual(
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
            _sha256_bytes(b"abc"),
        )


if __name__ == "__main__":
    unittest.main()
