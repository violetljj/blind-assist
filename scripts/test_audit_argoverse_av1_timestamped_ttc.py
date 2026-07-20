import importlib.util
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_argoverse_av1_timestamped_ttc.py")
SPEC = importlib.util.spec_from_file_location("argoverse_ttc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ArgoverseTimestampedTtcAuditTest(unittest.TestCase):
    def test_timestamped_oncoming_target_has_second_scale_ttc(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required by Argoverse TTC audit")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture.csv").write_text(
                "TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME\n"
                "0.0,00000000-0000-0000-0000-000000000000,AV,0,0,PIT\n"
                "0.0,target,VEHICLE,5,0,PIT\n"
                "1.0,00000000-0000-0000-0000-000000000000,AV,1,0,PIT\n"
                "1.0,target,VEHICLE,3,0,PIT\n", encoding="utf-8")
            report = MODULE.audit(root, horizon_seconds=3.0, vehicle_radius_m=2.0)
            self.assertEqual(1, report["front_facing_track_pair_count"])
            self.assertEqual(1.0, report["timestamp"]["median_period_seconds"])
            self.assertEqual(1, report["kinematics"]["ttc_collision_candidate_count_within_horizon"])
            self.assertTrue(report["source_ttc_seconds_available"])
            self.assertFalse(report["ustrf_motion_input_admitted"])


if __name__ == "__main__":
    unittest.main()
