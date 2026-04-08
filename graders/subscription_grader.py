import numpy as np

def grade(cancelled: int, total: int) -> float:
    if total == 0:
        return 0.999
    score = cancelled / total
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class SubscriptionGrader:
    def __call__(self, cancelled: int = 0, total: int = 2, *args, **kwargs) -> float:
        return grade(cancelled, total)
