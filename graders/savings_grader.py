"""Savings Builder grader."""

import numpy as np

def grade(final_checking: float, target_checking: float, original_excess: float) -> float:
    if original_excess <= 0:
        return 0.999
    if final_checking < target_checking:
        return 0.001
    amount_moved = (original_excess + target_checking) - final_checking
    score = amount_moved / original_excess
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class SavingsGrader:
    def __call__(self, final_checking: float, target_checking: float, original_excess: float) -> float:
        return grade(final_checking, target_checking, original_excess)
