"""Cash flow management grader.

Score based on whether the agent successfully prevented overdraft
by maintaining sufficient balance before large payments.
"""

import numpy as np


def grade(overdraft_avoided: float, max_score: float) -> float:
    """Grade cash flow management performance.

    Args:
        overdraft_avoided: 1.0 if overdraft was avoided, 0.0 if not.
        max_score: Maximum possible score (1.0).

    Returns:
        Score in [0.0, 1.0].
    """
    if max_score <= 0:
        return 0.5
    score = overdraft_avoided / max_score
    return round(float(np.clip(score, 0.001, 0.999)), 4)


class CashFlowGrader:
    """Callable grader class for cash flow management."""

    def __call__(self, overdraft_avoided: float, max_score: float) -> float:
        return grade(overdraft_avoided, max_score)
