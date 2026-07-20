import importlib.util
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_bonn_rgbd_dynamic_source.py")
SPEC = importlib.util.spec_from_file_location("bonn_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BonnRgbdDynamicAuditTest(unittest.TestCase):
    def test_index_rejects_non_monotonic_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.txt"
            path.write_text("# header\n2.0 item\n1.0 item\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                MODULE._index(path, 2)

    def test_nearest_uses_closest_timestamp(self):
        rows = [(1.0, ("a",)), (1.1, ("b",))]
        self.assertEqual("b", MODULE._nearest(rows, 1.08)[1][0])


if __name__ == "__main__":
    unittest.main()
