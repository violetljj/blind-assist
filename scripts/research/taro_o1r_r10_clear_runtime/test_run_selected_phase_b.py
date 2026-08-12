from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from scripts.research.taro_o1r_r10_clear_runtime import phase_b_metrics
from scripts.research.taro_o1r_r10_clear_runtime import run_selected_phase_b as runner


def selection(frame_counts: list[int]) -> dict:
    identities = [[f"p{index}", f"v{index}"] for index in range(8)]
    return {
        "content_sha256": "A" * 64,
        "parent_scores_sha256": "B" * 64,
        "selected_parent_identities": identities,
        "selected_parent_scores": [
            {"parent_id": parent, "video_id": video, "frame_count": count}
            for (parent, video), count in zip(identities, frame_counts, strict=True)
        ],
    }


class SelectedPhaseBTests(unittest.TestCase):
    def test_selected_cohort_frame_count_is_dynamic(self) -> None:
        frozen = selection([1, 2, 3, 4, 5, 6, 7, 8])
        cohort = runner.derive_selected_cohort(frozen)
        self.assertEqual(cohort["parent_count"], 8)
        self.assertEqual(cohort["physical_frame_count"], 36)
        self.assertEqual(cohort["query_count"], 324)
        self.assertEqual(cohort["selection_sha256"], "A" * 64)

    def test_only_sealed_top8_frames_enter_read_plan(self) -> None:
        frozen = selection([1] * 8)
        all_frames = [
            SimpleNamespace(parent_id=f"p{index}", video_id=f"v{index}")
            for index in range(12)
        ]
        selected = runner.select_frames(all_frames, frozen["selected_parent_identities"])
        self.assertEqual(len(selected), 8)
        self.assertEqual({(row.parent_id, row.video_id) for row in selected}, {tuple(row) for row in frozen["selected_parent_identities"]})

    def test_faro_receipts_prove_selected_only_and_highres_only(self) -> None:
        frozen = selection([1] * 8)
        frames = [
            SimpleNamespace(parent_id=parent, video_id=video)
            for parent, video in frozen["selected_parent_identities"]
        ]
        per_parent = Counter((row.parent_id, row.video_id) for row in frames)
        summary = runner.validate_faro_read_counts(
            {"highres_depth": 8},
            per_parent,
            frames,
            frozen["selected_parent_identities"],
        )
        self.assertEqual(summary["selected_highres_depth_reads"], 8)
        self.assertEqual(summary["unselected_highres_depth_reads"], 0)
        self.assertEqual(summary["only_payload_role_read"], "highres_depth")

        with self.assertRaises(runner.SelectedPhaseBError):
            runner.validate_faro_read_counts(
                {"highres_depth": 8, "color": 1},
                per_parent,
                frames,
                frozen["selected_parent_identities"],
            )

    def test_duplicate_or_misaligned_selection_is_rejected(self) -> None:
        duplicate = selection([1] * 8)
        duplicate["selected_parent_identities"][-1] = duplicate["selected_parent_identities"][0]
        with self.assertRaises(runner.SelectedPhaseBError):
            runner.derive_selected_cohort(duplicate)

        misaligned = selection([1] * 8)
        misaligned["selected_parent_scores"][0]["video_id"] = "different"
        with self.assertRaises(runner.SelectedPhaseBError):
            runner.derive_selected_cohort(misaligned)

    def test_metrics_are_frozen_and_unknown_is_never_negative(self) -> None:
        self.assertIs(runner.phase_b_metrics.summarize, phase_b_metrics.summarize)
        self.assertFalse(runner.phase_b_metrics.EXPECTED_GATES["unknown_is_negative"])
        self.assertFalse(runner.top8.validate_protocol(
            runner._read_json(runner._repo_path(runner.EXPECTED_BINDINGS["R10_PROTOCOL"]))
        )["phase_firewall"]["unknown_is_negative"])

    def test_runner_does_not_depend_on_r8_or_fixed_selected_frame_count(self) -> None:
        source = runner.Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("taro_o1r_r8_clear_runtime", source)
        self.assertNotRegex(source, r"(?m)^SELECTED_FRAME_COUNT\s*=")


if __name__ == "__main__":
    unittest.main()
