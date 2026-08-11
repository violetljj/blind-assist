#!/usr/bin/env python3

import unittest

from evaluate_ag_st_angular_boundary_student_bonn import checkpoint_parent_sources


class AngularBoundaryStudentBonnTest(unittest.TestCase):
    def test_checkpoint_sources_are_extracted_across_all_roles(self) -> None:
        split = {
            "fit": [("arkitscenes", "a")],
            "selection": [("tum_rgbd", "b")],
            "canary": [("arkitscenes", "c")],
        }
        self.assertEqual({"arkitscenes", "tum_rgbd"}, checkpoint_parent_sources(split))


if __name__ == "__main__":
    unittest.main()
