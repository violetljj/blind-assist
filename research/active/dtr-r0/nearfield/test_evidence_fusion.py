import unittest
import numpy as np
from evidence_fusion import BranchEvidence,fuse


def branch(points=(),unknown=()):
    states=np.full((3,3),"NO_NEAR_OBSERVED",dtype="<U20")
    for p in unknown: states[p]="UNKNOWN"
    for p in points: states[p]="OBSTACLE"
    return BranchEvidence(states=="OBSTACLE",states)


class FusionTest(unittest.TestCase):
    def test_missing_ground_keeps_direction_but_not_verified_height(self):
        result=fuse(branch([(1,1),(1,2)]),None)
        center=result["directions"][1]
        self.assertEqual(center["range_state"],"NEAR")
        self.assertEqual(center["height_status"],"UNKNOWN")
        self.assertEqual(center["ground_height_bands"],[])
        self.assertEqual(np.asarray(result["diagnostic_alerts"]).sum(),2)

    def test_no_evidence_is_not_far_or_clear(self):
        r=fuse(branch(),None)
        self.assertEqual({d["range_state"] for d in r["directions"]},{"UNKNOWN"})

    def test_union_preserves_complementary_sources_without_mutation(self):
        a=branch([(0,1),(1,1)]);b=branch([(1,1),(2,2)])
        r=fuse(a,b)
        self.assertEqual(np.asarray(r["diagnostic_alerts"]).sum(),3)
        self.assertEqual(r["provenance"][0][1],"RAW_ONLY")
        self.assertEqual(r["provenance"][1][1],"BOTH")
        self.assertEqual(r["provenance"][2][2],"GROUND_ONLY")
        self.assertEqual(a.alerts.sum(),2)

    def test_height_difference_is_not_physical_conflict(self):
        r=fuse(branch([(1,1)]),branch([(1,0)]))
        self.assertEqual(r["height_set_disagreement"],["center"])
        self.assertTrue(r["physical_height_conflict"].startswith("NOT_EVALUABLE"))
        self.assertEqual(r["directions"][1]["ground_height_bands"],["low"])
        self.assertTrue(r["directions"][1]["unassigned_raw_height_evidence"])

    def test_conditional_is_distinct_when_valid_ground_misses_raw(self):
        a,b=branch([(1,1)]),branch()
        self.assertTrue(np.asarray(fuse(a,b)["diagnostic_alerts"]).any())
        self.assertFalse(np.asarray(fuse(a,b,mode="conditional")["diagnostic_alerts"]).any())

    def test_unknown_preserved_and_inconsistent_branch_rejected(self):
        r=fuse(branch(unknown=[(0,0)]),branch())
        self.assertEqual(r["diagnostic_states"][0][0],"UNKNOWN")
        with self.assertRaises(ValueError): BranchEvidence(np.ones((3,3)),np.full((3,3),"UNKNOWN"))
        with self.assertRaises(ValueError): fuse(branch(),None,mode="average")


if __name__=="__main__":unittest.main()
