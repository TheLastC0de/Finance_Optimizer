"""Ledger cleanup grader.

Score based on fraction of transactions correctly categorized.
"""

import numpy as np


def grade(correct_categories: int, total_transactions: int) -> float:
    """Grade ledger cleanup performance.

    Args:
        correct_categories: Number of correctly categorized transactions.
        total_transactions: Total number of transactions to categorize.

    Returns:
        Score in [0.0, 1.0].
    """
    if total_transactions == 0:
        return 0.001
    score = correct_categories / total_transactions
    return round(float(np.clip(score, 0.001, 0.999)), 4)


class LedgerGrader:
    """Callable grader class for ledger cleanup."""

    def __call__(self, correct_categories: int, total_transactions: int) -> float:
        return grade(correct_categories, total_transactions)
