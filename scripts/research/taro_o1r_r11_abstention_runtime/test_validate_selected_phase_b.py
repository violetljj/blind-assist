from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

from scripts.research.taro_o1r_r11_abstention_runtime import validate_selected_phase_b as validator


def row(index: int, state: str, prefix: str) -> dict:
    return {"grid_index": index, "query_id": f"{prefix}-{index}", "state": state}


def fixture() -> tuple[list[tuple[str, str]], list[dict], list[dict], list[dict]]:
    identities = [(f"p{i:02d}", f"v{i:02d}") for i in range(24)]
    baselines, candidates, labels = [], [], []
    for index, identity in enumerate(identities[:12]):
        for repeat in range(2):
            common = {"parent_id": identity[0], "video_id": identity[1], "physical_frame_id": f"{identity[1]}:o{repeat}"}
            values = [row(i, "OCCUPIED_OBSERVED", f"o{index}-{repeat}") for i in range(9)]
            baselines.append({**common, "query_results": copy.deepcopy(values)})
            candidates.append({**common, "query_results": copy.deepcopy(values)})
            labels.append({**common, "query_labels": copy.deepcopy(values)})
    for index, identity in enumerate(identities[12:]):
        common = {"parent_id": identity[0], "video_id": identity[1], "physical_frame_id": f"{identity[1]}:c"}
        truth = [row(i, "CLEAR_OBSERVED" if i < 2 else "UNKNOWN", f"c{index}") for i in range(9)]
        prediction = [row(i, "UNKNOWN", f"c{index}") for i in range(9)]
        baselines.append({**common, "query_results": copy.deepcopy(prediction)})
        candidates.append({**common, "query_results": copy.deepcopy(prediction)})
        labels.append({**common, "query_labels": truth})
    return identities, baselines, candidates, labels


class IndependentR11PhaseBValidatorTests(unittest.TestCase):
    def test_independent_reducer_passes_exact_dual_class_fixture(self) -> None:
        result = validator._reduce(*fixture())
        self.assertEqual(result["terminal"], validator.PASS_TERMINAL)
        self.assertTrue(result["passed"])
        self.assertEqual(result["evaluability"]["definite_occupied_query_count"], 216)
        self.assertEqual(result["evaluability"]["physical_frames_with_definite_clear"], 12)

    def test_independent_reducer_rejects_resealed_metric_direction(self) -> None:
        identities, baselines, candidates, labels = fixture()
        for candidate in candidates[:3]:
            candidate["query_results"][0]["state"] = "UNKNOWN"
        result = validator._reduce(identities, baselines, candidates, labels)
        self.assertEqual(result["terminal"], validator.FAIL_TERMINAL)
        self.assertFalse(result["gates"]["micro_occupied_recall_loss_vs_r7"]["passed"])

    def test_validator_imports_neither_producer_nor_metric_module(self) -> None:
        tree = ast.parse(Path(validator.__file__).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        joined = "\n".join(imports)
        self.assertNotIn("run_selected_phase_b", joined)
        self.assertNotIn("phase_b_metrics", joined)


if __name__ == "__main__":
    unittest.main()
