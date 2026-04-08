import numpy as np

def grade(correct: int, total: int) -> float:
    if total == 0:
        return 0.001
    score = correct / total
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class LedgerGrader:
    def __call__(self, correct: int = 0, total: int = 50, *args, **kwargs) -> float:
        return grade(correct, total)
