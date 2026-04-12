"""Cash Flow grader with partial credit."""

import numpy as np


def grade(overdraft_avoided: float, max_score: float) -> float:
    """
    Grade cash flow management performance.
    
    Args:
        overdraft_avoided: The task score (1.0 = fully avoided, 0.0-0.4 = overdraft with partial credit).
        max_score: Maximum possible score (typically 1.0).
    
    Returns:
        Float score in [0.001, 0.999].
    """
    if max_score <= 0:
        return 0.5
    score = overdraft_avoided / max_score
    return round(float(np.clip(score, 0.001, 0.999)), 4)


class CashFlowGrader:
    def __call__(self, overdraft_avoided: float, max_score: float) -> float:
        return grade(overdraft_avoided, max_score)
