"""Subscription Audit grader."""

import numpy as np

def grade(cancelled_unnecessary: int, total_unnecessary: int) -> float:
    if total_unnecessary == 0:
        return 0.999
    score = cancelled_unnecessary / total_unnecessary
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class SubscriptionGrader:
    def __call__(self, cancelled_unnecessary: int, total_unnecessary: int) -> float:
        return grade(cancelled_unnecessary, total_unnecessary)
