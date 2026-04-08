import numpy as np

def grade(improvement: float, max_improvement: float) -> float:
    if max_improvement <= 0:
        return 0.5
    score = improvement / max_improvement
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class CashFlowGrader:
    def __call__(self, improvement: float = 0.0, max_improvement: float = 500.0, *args, **kwargs) -> float:
        return grade(improvement, max_improvement)
