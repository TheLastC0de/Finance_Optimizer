"""Cash Flow grader."""

import numpy as np

def grade(overdraft_avoided: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.5
    score = overdraft_avoided / max_score
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class CashFlowGrader:
    def __call__(self, overdraft_avoided: float, max_score: float) -> float:
        return grade(overdraft_avoided, max_score)
