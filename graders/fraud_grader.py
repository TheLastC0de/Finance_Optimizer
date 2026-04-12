"""Fraud Categorization grader."""

import numpy as np

def grade(fraud_identified: bool) -> float:
    score = 1.0 if fraud_identified else 0.0
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class FraudGrader:
    def __call__(self, fraud_identified: bool) -> float:
        return grade(fraud_identified)
