from __future__ import annotations

import unittest

import torch

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    typed_semantic_referent_graph_v3 as v3,
)


class TypedSemanticReferentGraphV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.target, self.frame, _ = v3.generate_synthetic_frame(302)

    def test_full_graph_contains_all_three_node_types_and_spatial_edges(self) -> None:
        graph = v3.build_typed_graph(self.target, self.frame, include_spatial_relations=True)
        self.assertEqual(set(graph.node_types.tolist()), set(v3.NODE_TYPES.values()))
        relation_names = {v3.RELATIONS[index] for index in graph.edge_types.tolist()}
        self.assertIn("TARGET_TO_OCR", relation_names)
        self.assertTrue(any(name.startswith("OCR_TO_CANDIDATE_") for name in relation_names))
        self.assertTrue(any(name.startswith("CANDIDATE_TO_OCR_") for name in relation_names))

    def test_no_relation_ablation_removes_ocr_candidate_and_ocr_spatial_edges(self) -> None:
        graph = v3.build_typed_graph(self.target, self.frame, include_spatial_relations=False)
        relation_names = {v3.RELATIONS[index] for index in graph.edge_types.tolist()}
        self.assertIn("TARGET_TO_OCR", relation_names)
        self.assertFalse(any(name.startswith("OCR_TO_CANDIDATE_") for name in relation_names))
        self.assertNotIn("OCR_SAME_LINE", relation_names)

    def test_three_heads_have_expected_shapes(self) -> None:
        graph = v3.build_typed_graph(self.target, self.frame, include_spatial_relations=True)
        batch = v3.GraphBatch.from_records([graph], torch.device("cpu"))
        model = v3.TypedSemanticReferentGraphNet(hidden=16, layers=2)
        outputs = model(batch)
        self.assertEqual(tuple(outputs["identity_logits"].shape), (3,))
        self.assertEqual(tuple(outputs["reliability"].shape), (1,))
        self.assertEqual(tuple(outputs["none_logits"].shape), (1,))

    def test_graph_output_is_candidate_permutation_equivariant(self) -> None:
        torch.manual_seed(302)
        model = v3.TypedSemanticReferentGraphNet(hidden=16, layers=2).eval()
        scores, _ = v3.predict_scores(
            model, self.target, self.frame, include_relations=True, device=torch.device("cpu")
        )
        permuted = v3.Frame(
            self.frame.episode_id,
            self.frame.frame_index,
            self.frame.viewpoint,
            tuple(reversed(self.frame.candidates)),
            self.frame.tokens,
            self.frame.blur,
            self.frame.perspective,
            self.frame.truth,
            self.frame.expected_state,
            self.frame.note,
        )
        permuted_scores, _ = v3.predict_scores(
            model, self.target, permuted, include_relations=True, device=torch.device("cpu")
        )
        for candidate_id in scores:
            self.assertAlmostEqual(scores[candidate_id]["score"], permuted_scores[candidate_id]["score"], places=6)


if __name__ == "__main__":
    unittest.main()
