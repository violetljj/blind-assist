import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_stage_c_d5_s0b_p0c_tartanground_provider import (
    build_resolution,
    canonical_bytes,
    resolve_url,
    sha256_bytes,
)


class TartanGroundProviderResolutionTest(unittest.TestCase):
    def test_resolve_url_pins_revision_and_preserves_path_structure(self):
        self.assertEqual(
            resolve_url("abc123", "Env Name/Data_diff/P1000/metadata.zip"),
            "https://huggingface.co/datasets/theairlabcmu/TartanGround/"
            "resolve/abc123/Env%20Name/Data_diff/P1000/metadata.zip",
        )

    def test_complete_catalog_resolves_without_burning_source(self):
        catalog = {
            "catalog_observation": {
                "parents": [
                    {
                        "environment": "Env",
                        "parent_id": "Env/Data_diff/P1000",
                        "robot_version": "diff",
                        "trajectory_id": "P1000",
                        "archive_paths": {
                            "metadata.zip": "Env/Data_diff/P1000/metadata.zip"
                        },
                    }
                ]
            }
        }
        provider = {
            "sha": "abc123",
            "private": False,
            "gated": False,
            "disabled": False,
            "lastModified": "2026-08-02T00:00:00Z",
            "siblings": [{"rfilename": "Env/Data_diff/P1000/metadata.zip"}],
        }

        mapping, result = build_resolution(catalog, provider)

        self.assertEqual(
            result["status"], "PROVIDER_AND_EXACT_ARCHIVE_MAPPING_RESOLVED"
        )
        self.assertTrue(
            result["catalog"]["all_archive_paths_present_at_revision"]
        )
        self.assertTrue(
            result["development_policy"]["repairable_after_engineering_failure"]
        )
        self.assertFalse(result["development_policy"]["source_burned_by_resolution"])
        self.assertEqual(
            result["archive_url_map"]["sha256"],
            sha256_bytes(canonical_bytes(mapping)),
        )

    def test_missing_provider_path_is_a_repairable_mapping_mismatch(self):
        catalog = {
            "catalog_observation": {
                "parents": [
                    {
                        "environment": "Env",
                        "parent_id": "Env/Data_diff/P1000",
                        "robot_version": "diff",
                        "trajectory_id": "P1000",
                        "archive_paths": {
                            "metadata.zip": "Env/Data_diff/P1000/metadata.zip"
                        },
                    }
                ]
            }
        }
        provider = {"sha": "abc123", "siblings": []}

        _, result = build_resolution(catalog, provider)

        self.assertEqual(result["status"], "PROVIDER_CATALOG_PATH_MISMATCH")
        self.assertEqual(
            result["catalog"]["missing_archive_paths"],
            ["Env/Data_diff/P1000/metadata.zip"],
        )
        self.assertFalse(
            result["claim_boundary"][
                "provider_source_and_exact_archive_mapping_established"
            ]
        )


if __name__ == "__main__":
    unittest.main()
