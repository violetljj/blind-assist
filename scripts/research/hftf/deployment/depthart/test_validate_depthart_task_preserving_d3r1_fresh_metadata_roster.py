import copy
import hashlib
import unittest

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d3r1_fresh_metadata_roster import (
    ROLE,
    ROSTER_SCHEMA,
)
from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d3r1_fresh_metadata_roster import (
    assert_exact_roster,
)


def fixture() -> dict:
    return {
        "schema": ROSTER_SCHEMA,
        "status": "D3R1_FRESH_METADATA_POOL_127_LOCKED_MEDIA_UNOPENED",
        "source": {
            "repository_commit": "7283761bf26c27570ec59a5dc0f8686fbff07726",
            "metadata_bytes": 127263,
            "metadata_sha256": "06A0686866F186764ED0B92DE1A943529CEBD78B4AF5B671907C40BB2DCD13E1",
            "official_rows": 5047,
        },
        "selection": {
            "training_row_count": 4498,
            "eligible_row_count": 3724,
            "eligible_unique_visit_count": 1233,
            "eligible_unique_session_count": 3724,
            "pool": [
                {
                    "pool_order": index + 1,
                    "visit_id": f"visit-{index + 1}",
                    "video_id": f"video-{index + 1}",
                    "fold": "Training",
                    "role": ROLE,
                    "selection_sha256": hashlib.sha256(
                        f"visit-{index + 1}:video-{index + 1}".encode("ascii")
                    ).hexdigest().upper(),
                }
                for index in range(127)
            ]
        },
        "invariants": {
            "pool_count": 127,
            "unique_parent_count": 127,
            "unique_session_count": 127,
            "workspace_excluded_identity_count": 490,
            "concurrent_excluded_identity_count": 64,
            "effective_excluded_identity_count": 554,
            "selection_overlap_with_workspace_snapshot": 0,
            "selection_overlap_with_concurrent_identity_firewalls": 0,
            "selection_overlap_with_d3_predecessor_pool": 0,
            "media_head_requests": 0,
            "media_body_bytes_read": 0,
            "truth_read": False,
            "model_outputs_read": False,
            "training": False,
            "source_scope_registered": False,
            "download_authorized": False,
        },
    }


class D3R1FreshMetadataRosterValidatorTest(unittest.TestCase):
    def test_exact_fixture_passes(self) -> None:
        value = fixture()
        assert_exact_roster(value, copy.deepcopy(value))

    def test_mutated_order_fails(self) -> None:
        value = fixture()
        mutated = copy.deepcopy(value)
        mutated["selection"]["pool"][0]["pool_order"] = 2
        with self.assertRaisesRegex(ValueError, "independent replay"):
            assert_exact_roster(mutated, value)

    def test_predecessor_overlap_fails(self) -> None:
        value = fixture()
        value["invariants"]["selection_overlap_with_d3_predecessor_pool"] = 1
        with self.assertRaisesRegex(ValueError, "identity firewall"):
            assert_exact_roster(value, copy.deepcopy(value))


if __name__ == "__main__":
    unittest.main()
