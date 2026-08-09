import types
import unittest
from unittest.mock import patch

from scripts.research.assistive_geometry.depthart_training_scan import install_depthart_training_scan


class DepthArtTrainingScanTest(unittest.TestCase):
    def test_installs_eager_autograd_path(self) -> None:
        eager = object()
        package_ops = types.SimpleNamespace(_cuda_impl=eager, _C=object(), __file__="ops.py")
        package_cross = types.SimpleNamespace(selective_scan=None, cross_selective_scan=object())
        tvimblock = types.SimpleNamespace(cross_selective_scan=object())
        previous = tvimblock.cross_selective_scan

        def fake_import(name: str):
            return package_ops if name.endswith(".ops") else package_cross

        with patch("importlib.import_module", side_effect=fake_import):
            returned, metadata = install_depthart_training_scan(tvimblock)
        self.assertIs(previous, returned)
        self.assertIs(eager, package_cross.selective_scan)
        self.assertIs(package_cross.cross_selective_scan, tvimblock.cross_selective_scan)
        self.assertFalse(metadata["outer_torch_library_operator_used_for_training"])

    def test_missing_cuda_backend_fails_closed(self) -> None:
        package_ops = types.SimpleNamespace(_cuda_impl=object(), _C=None, __file__="ops.py")
        package_cross = types.SimpleNamespace(selective_scan=None, cross_selective_scan=object())
        tvimblock = types.SimpleNamespace(cross_selective_scan=object())
        with patch("importlib.import_module", side_effect=(package_ops, package_cross)):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                install_depthart_training_scan(tvimblock)


if __name__ == "__main__":
    unittest.main()
