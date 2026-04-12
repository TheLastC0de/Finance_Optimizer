"""All Finance Optimizer Tasks Graders.

Consolidated module containing graders for all 6 finance optimization tasks.
"""

import numpy as np

# --- Ledger Cleanup Grader ---
def grade_ledger(correct_categories: int, total_transactions: int) -> float:
    if total_transactions == 0:
        return 0.001
    score = correct_categories / total_transactions
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class LedgerGrader:
    def __call__(self, correct_categories: int, total_transactions: int) -> float:
        return grade_ledger(correct_categories, total_transactions)

# --- Subscription Audit Grader ---
def grade_subscription(cancelled_unnecessary: int, total_unnecessary: int) -> float:
    if total_unnecessary == 0:
        return 0.999
    score = cancelled_unnecessary / total_unnecessary
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class SubscriptionGrader:
    def __call__(self, cancelled_unnecessary: int, total_unnecessary: int) -> float:
        return grade_subscription(cancelled_unnecessary, total_unnecessary)

# --- Cash Flow Management Grader ---
def grade_cash_flow(overdraft_avoided: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.5
    score = overdraft_avoided / max_score
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class CashFlowGrader:
    def __call__(self, overdraft_avoided: float, max_score: float) -> float:
        return grade_cash_flow(overdraft_avoided, max_score)

# --- Fraud Categorization Grader ---
def grade_fraud(fraud_identified: bool) -> float:
    score = 1.0 if fraud_identified else 0.0
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class FraudGrader:
    def __call__(self, fraud_identified: bool) -> float:
        return grade_fraud(fraud_identified)

# --- Savings Builder Grader ---
def grade_savings(final_checking: float, target_checking: float, original_excess: float) -> float:
    if original_excess <= 0:
        return 0.999
    if final_checking < target_checking:
        return 0.001
    amount_moved = (original_excess + target_checking) - final_checking
    score = amount_moved / original_excess
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class SavingsGrader:
    def __call__(self, final_checking: float, target_checking: float, original_excess: float) -> float:
        return grade_savings(final_checking, target_checking, original_excess)

# --- Duplicate Charge Alert Grader ---
def grade_duplicate(correct_alert_set: bool) -> float:
    score = 1.0 if correct_alert_set else 0.0
    return round(float(np.clip(score, 0.001, 0.999)), 4)

class DuplicateGrader:
    def __call__(self, correct_alert_set: bool) -> float:
        return grade_duplicate(correct_alert_set)
