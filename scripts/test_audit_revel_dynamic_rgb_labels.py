import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("audit_revel_dynamic_rgb_labels.py")
SPEC = importlib.util.spec_from_file_location("revel_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RevelDynamicRgbLabelsTest(unittest.TestCase):
    def test_parse_empty_and_yolo_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.txt"
            path.write_text("\n1 0.5 0.6 0.2 0.3\n", encoding="utf-8")
            self.assertEqual(MODULE._parse_label(path), [(1, 0.5, 0.6, 0.2, 0.3)])

    def test_parse_rejects_wrong_field_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.txt"; path.write_text("1 0.5 0.6\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE._parse_label(path)

    def test_layout_requires_expected_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                MODULE._layout(Path(directory))


if __name__ == "__main__":
    unittest.main()
