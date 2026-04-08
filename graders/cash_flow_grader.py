import numpy as np

def grade(net_balance_improvement: float, max_possible_improvement: float) -> float:
    if max_possible_improvement <= 0:
        return 0.5
    score = net_balance_improvement / max_possible_improvement
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class CashFlowGrader:
    def __call__(self, net_balance_improvement: float, max_possible_improvement: float) -> float:
        return grade(net_balance_improvement, max_possible_improvement)
