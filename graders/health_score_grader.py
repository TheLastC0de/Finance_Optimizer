"""Global Financial Health grader including Credit Score."""

import numpy as np

def grade(net_worth_growth: float, debt_reduction: float, credit_score: float) -> float:
    """
    Compute a composite financial health score.
    
    Args:
        net_worth_growth: Ratio of final net worth to initial (capped at 1.2).
        debt_reduction: Percentage of debt paid off (0.0 to 1.0).
        credit_score: FICO score (300-850).
    
    Returns:
        Float score in [0.001, 1.0].
    """
    # Normalize credit score to [0, 1]
    cs_norm = (credit_score - 300) / (850 - 300)
    
    # Weighted average: 30% Growth, 30% Debt, 40% Credit Rating (very important in real world)
    score = (0.3 * min(net_worth_growth, 1.2) / 1.2) + (0.3 * debt_reduction) + (0.4 * cs_norm)
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class HealthScoreGrader:
    def __call__(self, net_worth_growth: float, debt_reduction: float, credit_score: float) -> float:
        return grade(net_worth_growth, debt_reduction, credit_score)
