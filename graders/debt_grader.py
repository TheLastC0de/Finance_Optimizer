"""Strategic Debt grader with buffer constraints."""

import numpy as np

def grade(debt_paid_pct: float, final_checking: float, target_buffer: float = 500.0) -> float:
    """
    Computes a debt repayment score with constraints.
    
    Args:
        debt_paid_pct: Ratio of debt paid off (0.0 to 1.0).
        final_checking: The balance in checking at end of task.
        target_buffer: Minimum buffer that should have been maintained.
        
    Returns:
        Float score in [0.001, 1.0].
    """
    # Base score is percentage of debt paid
    score = debt_paid_pct
    
    # Violation penalty: If checking is below buffer, slash score significantly.
    # This ensures "perfect" only comes from smart constraint balancing.
    if final_checking < target_buffer:
        # Penalty follows a gradient but caps the score low
        penalty_ratio = final_checking / target_buffer
        score = min(score, 0.4) * penalty_ratio
        
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class DebtGrader:
    def __call__(self, debt_paid_pct: float, final_checking: float) -> float:
        return grade(debt_paid_pct, final_checking)
