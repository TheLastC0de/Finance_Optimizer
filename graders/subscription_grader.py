"""Subscription audit grader.

Score based on fraction of unnecessary subscriptions cancelled.
"""

import numpy as np


def grade(cancelled_unnecessary: int, total_unnecessary: int) -> float:
    """Grade subscription audit performance.

    Args:
        cancelled_unnecessary: Number of unnecessary subscriptions cancelled.
        total_unnecessary: Total number of unnecessary subscriptions.

    Returns:
        Score in [0.0, 1.0].
    """
    if total_unnecessary == 0:
        return 0.999
    score = cancelled_unnecessary / total_unnecessary
    return round(float(np.clip(score, 0.001, 0.999)), 4)


class SubscriptionGrader:
    """Callable grader class for subscription audit."""

    def __call__(self, cancelled_unnecessary: int, total_unnecessary: int) -> float:
        return grade(cancelled_unnecessary, total_unnecessary)
