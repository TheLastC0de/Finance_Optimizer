"""Duplicate Charge Alert grader."""

import numpy as np

def grade(correct_alert_set: bool) -> float:
    score = 1.0 if correct_alert_set else 0.0
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class DuplicateGrader:
    def __call__(self, correct_alert_set: bool) -> float:
        return grade(correct_alert_set)
