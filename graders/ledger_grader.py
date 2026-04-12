"""Ledger Cleanup grader."""

import numpy as np

def grade(correct_categories: int, total_transactions: int) -> float:
    if total_transactions == 0:
        return 0.001
    score = correct_categories / total_transactions
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class LedgerGrader:
    def __call__(self, correct_categories: int, total_transactions: int) -> float:
        return grade(correct_categories, total_transactions)
