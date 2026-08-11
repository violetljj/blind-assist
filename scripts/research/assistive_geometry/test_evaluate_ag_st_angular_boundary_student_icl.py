#!/usr/bin/env python3

import unittest

from evaluate_ag_st_angular_boundary_student_icl import checkpoint_parent_sources


class AngularBoundaryStudentIclTest(unittest.TestCase):
    def test_checkpoint_source_audit_rejects_hidden_icl_use(self) -> None:
        clean = {"fit": [("arkitscenes", "a")], "selection": [("tum_rgbd", "b")], "canary": []}
        contaminated = {**clean, "canary": [("icl_exact", "c")]}
        self.assertNotIn("icl_exact", checkpoint_parent_sources(clean))
        self.assertIn("icl_exact", checkpoint_parent_sources(contaminated))


if __name__ == "__main__":
    unittest.main()
