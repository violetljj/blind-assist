"""Descriptive coverage at matched numbers of incorrect confirmations."""

from collections import Counter


FALSE_CONFIRM_BUDGETS = (1, 3, 5, 10)


def retrospective_coverage(labels, scores):
    """Allow all score ties together and retain the best coverage within each budget."""
    total_at_score = Counter()
    correct_at_score = Counter()
    for label, score in zip(labels, scores, strict=True):
        total_at_score[score] += 1
        correct_at_score[score] += int(label)

    frontier = [(0, 0)]
    correct = wrong = 0
    for score in sorted(total_at_score, reverse=True):
        correct += correct_at_score[score]
        wrong += total_at_score[score] - correct_at_score[score]
        frontier.append((wrong, correct))
    return {
        str(budget): max(good for bad, good in frontier if bad <= budget)
        for budget in FALSE_CONFIRM_BUDGETS
    }
