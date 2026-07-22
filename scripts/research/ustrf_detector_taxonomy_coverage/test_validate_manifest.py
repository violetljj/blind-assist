from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from validate_manifest import validate


CONFIG = Path("configs/ustrf_detector_taxonomy_coverage_v1.json")


class ValidateManifestTest(unittest.TestCase):
    def write_mutation(self, mutate) -> Path:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutate(payload)
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False)
        with handle:
            json.dump(payload, handle)
        return Path(handle.name)

    def test_live_manifest_passes(self) -> None:
        self.assertTrue(validate(CONFIG)["manifest_valid"])

    def test_rejects_threshold_drift(self) -> None:
        path = self.write_mutation(lambda row: row["detector"].update(confidence_threshold=0.34))
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "threshold/NMS drift"):
            validate(path)

    def test_rejects_person_mapping_shift(self) -> None:
        path = self.write_mutation(lambda row: row["detector"].update(person_class_index=1))
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "person index/label"):
            validate(path)

    def test_rejects_reopened_h2(self) -> None:
        path = self.write_mutation(lambda row: row["stage_locks"].update(H2_temporal_depth_reopen=True))
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "downstream authority"):
            validate(path)

    def test_rejects_result_leakage(self) -> None:
        path = self.write_mutation(lambda row: row.update(result={"passed": True}))
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "result leakage"):
            validate(path)


if __name__ == "__main__":
    unittest.main()
