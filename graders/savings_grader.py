"""Savings Builder grader with efficiency bonus."""

import numpy as np


def grade(final_checking: float, target_checking: float, original_excess: float, steps_used: int = 1) -> float:
    """
    Grade savings builder performance.
    
    Awards score based on how much excess was moved to savings.
    Bonus multiplier for efficiency (fewer steps = higher score).
    
    Args:
        final_checking: The agent's final checking balance.
        target_checking: The minimum checking balance target (e.g. 500.0).
        original_excess: The amount of excess that should be moved (checking - target at start).
        steps_used: Number of steps the agent took (for efficiency bonus).
    
    Returns:
        Float score in [0.001, 0.999].
    """
    if original_excess <= 0:
        return 0.999
    if final_checking < target_checking:
        # Transferred too much — penalize proportionally
        overshoot = target_checking - final_checking
        penalty = min(overshoot / original_excess, 1.0)
        return round(float(np.clip(0.5 * (1.0 - penalty), 0.001, 0.999)), 4)
    
    amount_moved = (original_excess + target_checking) - final_checking
    base_score = amount_moved / original_excess
    
    # Efficiency bonus: perfect score in 1-2 steps gets a small boost
    if steps_used <= 2 and base_score >= 0.95:
        base_score = min(base_score * 1.05, 1.0)
    
    return round(float(np.clip(base_score, 0.001, 0.999)), 4)


class SavingsGrader:
    def __call__(self, final_checking: float, target_checking: float, original_excess: float, steps_used: int = 1) -> float:
        return grade(final_checking, target_checking, original_excess, steps_used)
