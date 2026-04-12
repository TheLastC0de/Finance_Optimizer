"""Global Financial Health grader."""

import numpy as np

def grade(net_worth_growth: float, debt_reduction: float, budget_adherence: float) -> float:
    """
    Compute a composite financial health score.
    
    Args:
        net_worth_growth: Ratio of final net worth to initial (capped at 1.2).
        debt_reduction: Percentage of debt paid off (0.0 to 1.0).
        budget_adherence: Ratio of successful actions to total steps.
    
    Returns:
        Float score in [0.001, 0.999].
    """
    # Weighted average: 40% Growth, 40% Debt, 20% Adherence
    score = (0.4 * min(net_worth_growth, 1.2) / 1.2) + (0.4 * debt_reduction) + (0.2 * budget_adherence)
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class HealthScoreGrader:
    def __call__(self, net_worth_growth: float, debt_reduction: float, budget_adherence: float) -> float:
        return grade(net_worth_growth, debt_reduction, budget_adherence)
