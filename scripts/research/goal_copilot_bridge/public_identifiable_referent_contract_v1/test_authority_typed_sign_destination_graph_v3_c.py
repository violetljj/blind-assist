from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from .authority_typed_sign_destination_graph_v3_c import (
    RELATION_INDEX,
    SignCarrier,
    score_candidates,
    token_roles,
)
from .semantic_anchor_graph_and_belief_v2 import Box, Candidate, Frame, TargetGraph, Token


class _FixedRelationModel(nn.Module):
    def __init__(self, relation: str, decisive_logit: float = 8.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.relation = relation
        self.decisive_logit = decisive_logit

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = features.new_full((features.shape[0], 5), -8.0)
        logits[:, RELATION_INDEX[self.relation]] = 8.0
        return {"relation_logits": logits, "decisive_logit": features.new_full((features.shape[0],), self.decisive_logit)}


class AuthorityTypedSignDestinationGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = TargetGraph(("LAB", "406"))
        self.candidates = (
            Candidate("A", Box(0.0, 0.3, 0.3, 0.95)),
            Candidate("B", Box(0.35, 0.3, 0.65, 0.95)),
            Candidate("C", Box(0.7, 0.3, 1.0, 0.95)),
        )
        self.image = np.full((240, 320, 3), 220, dtype=np.uint8)
        self.carriers = (SignCarrier("sign", Box(0.38, 0.18, 0.62, 0.30)),)

    def _frame(self, tokens: tuple[Token, ...], candidates=None) -> Frame:
        return Frame("e", 0, "native", candidates or self.candidates, tokens, 0.0, 0.0, "B", "TARGET", "test")

    def test_numeric_identity_token_is_decisive(self) -> None:
        roles = token_roles(self.target)
        self.assertEqual(("406",), roles["decisive"])
        self.assertEqual(("LAB",), roles["generic"])

    def test_missing_decisive_token_cannot_create_identity_score(self) -> None:
        tokens = (Token("LAB", Box(0.42, 0.21, 0.50, 0.25), 0.99),)
        scores, _ = score_candidates(
            _FixedRelationModel("LABELS"), self.target, self._frame(tokens), self.image,
            self.carriers, (), authority_typing=True, device=torch.device("cpu"),
        )
        self.assertTrue(all(value["score"] < 0.10 for value in scores.values()))

    def test_lists_relation_has_no_identity_authority(self) -> None:
        tokens = (
            Token("LAB", Box(0.41, 0.21, 0.48, 0.25), 0.99),
            Token("406", Box(0.49, 0.21, 0.56, 0.25), 0.99),
        )
        full, _ = score_candidates(
            _FixedRelationModel("LISTS"), self.target, self._frame(tokens), self.image,
            self.carriers, (), authority_typing=True, device=torch.device("cpu"),
        )
        ablation, _ = score_candidates(
            _FixedRelationModel("LISTS"), self.target, self._frame(tokens), self.image,
            self.carriers, (), authority_typing=False, device=torch.device("cpu"),
        )
        self.assertLess(max(value["score"] for value in full.values()), 0.10)
        self.assertGreater(max(value["score"] for value in ablation.values()), 0.90)

    def test_candidate_scores_are_permutation_invariant(self) -> None:
        tokens = (
            Token("LAB", Box(0.41, 0.21, 0.48, 0.25), 0.99),
            Token("406", Box(0.49, 0.21, 0.56, 0.25), 0.99),
        )
        model = _FixedRelationModel("LABELS")
        original, _ = score_candidates(model, self.target, self._frame(tokens), self.image, self.carriers, (), authority_typing=True, device=torch.device("cpu"))
        reversed_scores, _ = score_candidates(model, self.target, self._frame(tokens, tuple(reversed(self.candidates))), self.image, self.carriers, (), authority_typing=True, device=torch.device("cpu"))
        self.assertEqual(set(original), set(reversed_scores))
        for key in original:
            self.assertAlmostEqual(original[key]["score"], reversed_scores[key]["score"], places=7)


if __name__ == "__main__":
    unittest.main()
