#!/usr/bin/env python3
"""Pure tests for the frozen prompt-free semantic probe."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_silver_prompt_free_semantic_probe as subject


def row(group: str, *, confidence: float, area: float, overlap: float) -> dict[str, float | str]:
    return {
        "semantic_group": group,
        "confidence": confidence,
        "area": area,
        "bottom": 0.9,
        "corridor_overlap": overlap,
        "threat": confidence * area * overlap,
    }


class PublicSilverPromptFreeSemanticProbeTest(unittest.TestCase):
    def test_preregistered_vocabulary_is_exact(self) -> None:
        self.assertEqual("surface_material", subject.semantic_group("sand box"))
        self.assertEqual("barrier_structure", subject.semantic_group("barrier"))
        self.assertIsNone(subject.semantic_group("sandwich"))
        self.assertIsNone(subject.semantic_group("person"))

    def test_frame_vector_responds_to_surface_corridor_detection(self) -> None:
        clear = subject.semantic_frame_vector([])
        obstacle = subject.semantic_frame_vector([
            row("surface_material", confidence=0.8, area=0.2, overlap=0.9)
        ])
        self.assertEqual((14,), clear.shape)
        self.assertGreater(obstacle[0], clear[0])
        self.assertGreater(obstacle[4], clear[4])

    def test_episode_vector_preserves_terminal_and_slope(self) -> None:
        frames = [
            [],
            [row("surface_material", confidence=0.5, area=0.1, overlap=0.7)],
            [row("surface_material", confidence=0.9, area=0.3, overlap=0.9)],
        ]
        vector = subject.semantic_episode_vector(frames)
        self.assertEqual((70,), vector.shape)
        dimension = 14
        terminal = vector[2 * dimension:3 * dimension]
        self.assertTrue(np.allclose(subject.semantic_frame_vector(frames[-1]), terminal))
        self.assertGreater(vector[-dimension], 0.0)


if __name__ == "__main__":
    unittest.main()
