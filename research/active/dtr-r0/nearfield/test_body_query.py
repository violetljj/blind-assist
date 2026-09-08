"""Checks for native-label semantics and the count bottleneck."""
import itertools
import unittest
import torch
from body_query_model import near_from_counts,fixed_projection,query_boxes


class CountTest(unittest.TestCase):
    def test_all_discrete_combinations_preserve_three_pixel_rule(self):
        values=torch.tensor(list(itertools.product(range(4),repeat=6)))
        both=values[:,None].repeat(1,2,1).reshape(-1,12)
        logits=torch.full((len(values),12,4),-40.)
        logits.scatter_(-1,both[:,:,None],40.)
        prediction=near_from_counts(logits).sigmoid()>=.5
        self.assertTrue(torch.equal(prediction,(values.sum(1)>=3)[:,None].repeat(1,2)))

    def test_probability_matches_exhaustive_independent_count_sum(self):
        torch.manual_seed(11)
        logits=torch.randn(1,12,4,dtype=torch.float64,requires_grad=True)
        p=logits.softmax(-1).reshape(2,6,4)
        truth=[]
        for h in range(2):
            total=0.
            for counts in itertools.product(range(4),repeat=6):
                if sum(counts)>=3:
                    probability=1.
                    for i,c in enumerate(counts):probability=probability*p[h,i,c]
                    total=total+probability
            truth.append(total)
        actual=near_from_counts(logits).sigmoid()[0]
        self.assertTrue(torch.allclose(actual,torch.stack(truth),atol=1e-10))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_boxes_tile_original_band_and_projection_has_support(self):
        boxes=query_boxes()
        self.assertEqual(len(boxes),12)
        for h in range(2):
            for r in range(2):
                row=boxes[h*6+r*3:h*6+r*3+3]
                self.assertEqual(row[0][1][1],row[1][0][1])
                self.assertEqual(row[1][1][1],row[2][0][1])
        grid,valid,xyz=fixed_projection()
        self.assertEqual(tuple(grid.shape),(12,27,2))
        self.assertTrue(valid.any(1).all())
        self.assertTrue(torch.isfinite(xyz).all())


if __name__=='__main__':unittest.main()
