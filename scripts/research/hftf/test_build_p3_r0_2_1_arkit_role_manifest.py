from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_p3_r0_2_1_arkit_role_manifest.py")
SPEC = importlib.util.spec_from_file_location("build_role_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildRoleManifestTest(unittest.TestCase):
    def test_assigns_exact_roles_across_two_sources(self) -> None:
        protocol = {"parent_roles": {"train": ["t"], "validation": ["v"]}}
        result = MODULE.build(protocol, {"videos": [{"visit_id": "t", "video_id": "1"}]}, {"videos": [{"visit_id": "v", "video_id": "2"}]}, "A" * 64)
        self.assertEqual(["train", "validation"], [video["role"] for video in result["videos"]])
        self.assertFalse(result["labels_opened"])

    def test_missing_parent_fails_closed(self) -> None:
        protocol = {"parent_roles": {"train": ["missing"], "validation": []}}
        with self.assertRaisesRegex(ValueError, "requested parent missing"):
            MODULE.build(protocol, {"videos": []}, {"videos": []}, "A" * 64)

    def test_role_overlap_fails_closed(self) -> None:
        protocol = {"parent_roles": {"train": ["p"], "validation": ["p"]}}
        with self.assertRaisesRegex(ValueError, "train/validation overlap"):
            MODULE.build(protocol, {"videos": [{"visit_id": "p"}]}, {"videos": []}, "A" * 64)


if __name__ == "__main__":
    unittest.main()
