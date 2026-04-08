import numpy as np

def grade_ledger(correct_categories: int, total_transactions: int) -> float:
    if total_transactions == 0:
        return 0.001
    score = correct_categories / total_transactions
    return round(float(np.clip(score, 0.001, 0.999)), 4)

def grade_subscription(cancelled_unnecessary: int, total_unnecessary: int) -> float:
    if total_unnecessary == 0:
        return 0.999
    score = cancelled_unnecessary / total_unnecessary
    return round(float(np.clip(score, 0.001, 0.999)), 4)

def grade_cash_flow(net_balance_improvement: float, max_possible_improvement: float) -> float:
    if max_possible_improvement <= 0:
        return 0.5
    score = net_balance_improvement / max_possible_improvement
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class LedgerGrader:
    def __call__(self, correct: int = 0, total: int = 50, *args, **kwargs) -> float:
        return grade_ledger(correct, total)

class SubscriptionGrader:
    def __call__(self, cancelled: int = 0, total: int = 2, *args, **kwargs) -> float:
        return grade_subscription(cancelled, total)

class CashFlowGrader:
    def __call__(self, improvement: float = 0.0, max_improvement: float = 500.0, *args, **kwargs) -> float:
        return grade_cash_flow(improvement, max_improvement)
