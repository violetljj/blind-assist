import numpy as np

from mz81_conditional_feature_probe import average_precision, select_threshold


def test_threshold_respects_budget_and_maximizes_true_bits():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([True, False, True, False])
    assert select_threshold(scores, truth, 0) == {'threshold': 0.9, 'TP': 1, 'FP': 0}
    assert select_threshold(scores, truth, 1) == {'threshold': 0.7, 'TP': 2, 'FP': 1}


def test_average_precision_perfect_ordering():
    assert average_precision(np.array([4., 3., 2., 1.]),
                             np.array([True, True, False, False])) == 1.0
