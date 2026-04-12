from typing import List, Dict, Any
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from finance_optimizer.models import FinanceOptimizerAction, FinanceOptimizerObservation

import numpy as np
from finance_optimizer.server.grader import LedgerGrader, SubscriptionGrader, CashFlowGrader

# Initialize singletons for environment loop
ledger_grader_inst = LedgerGrader()
subscription_grader_inst = SubscriptionGrader()
cash_flow_grader_inst = CashFlowGrader()

class FinanceOptimizerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS = [
        {
            "task_id": "ledger_cleanup",
            "name": "Ledger Cleanup",
            "difficulty": "easy",
            "description": "Correctly categorize 50 raw transactions.",
            "data_corpus": [],
            "aliases": ["task_easy", "categorize_transactions"]
        },
        {
            "task_id": "subscription_audit",
            "name": "Subscription Audit",
            "difficulty": "medium",
            "description": "Identify and cancel duplicate or unused subscriptions.",
            "data_corpus": [],
            "aliases": ["task_medium", "cancel_subscriptions"]
        },
        {
            "task_id": "cash_flow",
            "name": "Cash Flow Management",
            "difficulty": "hard",
            "description": "Prevent overdraft by transferring funds before a large payment.",
            "data_corpus": [],
            "aliases": ["task_hard", "prevent_overdraft"]
        }
    ]

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.ledger = []
        self.subscriptions = []
        self.checking_balance = 1200.0
        self.savings_balance = 1000.0
        self.days_passed = 0
        self.task_scores = {
            "ledger_cleanup": 0.0,
            "subscription_audit": 0.0,
            "cash_flow": 0.0
        }

    def reset(self, seed: int | None = None, task_id: str | None = None) -> FinanceOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._state.task_id = task_id or "ledger_cleanup"
        self.task_scores = {k: 0.0 for k in self.task_scores}
        
        self.ledger = []
        for i in range(25):
            self.ledger.append({"id": f"tx_{i}", "vendor": "UBER *TRIP", "amount": -15.0, "category": "Uncategorized"})
            self.ledger.append({"id": f"tx_{i+25}", "vendor": "SAFEWAY #33", "amount": -45.0, "category": "Uncategorized"})
            
        self.subscriptions = [
            {"vendor_name": "Netflix_Primary", "cost": 15.99, "type": "streaming", "duplicate": False},
            {"vendor_name": "Netflix_Duplicate", "cost": 15.99, "type": "streaming", "duplicate": True},
            {"vendor_name": "Gym", "cost": 50.0, "type": "gym", "last_visit_days_ago": 90},
            {"vendor_name": "Rent", "cost": 1500.0, "type": "housing", "due_in_days": 7}
        ]
        
        self.checking_balance = 1200.0
        self.savings_balance = 1000.0
        self.days_passed = 0
        
        return self._get_obs(reward=0.0)

    def _get_obs(self, reward: float = 0.0, done: bool = False):
        metadata = {
            "tasks": self.TASKS,
            "task_scores": self.task_scores,
            "step": self._state.step_count
        }
        return FinanceOptimizerObservation(
            ledger=self.ledger,
            subscriptions=self.subscriptions,
            checking_balance=self.checking_balance,
            savings_balance=self.savings_balance,
            done=done,
            reward=reward,
            metadata=metadata
        )

    def step(self, action: FinanceOptimizerAction) -> FinanceOptimizerObservation:
        self._state.step_count += 1
        reward = 0.0
        
        if action.action_type == "CategorizeTransaction":
            for tx in self.ledger:
                if tx["id"] == action.tx_id:
                    if tx["vendor"] == "UBER *TRIP" and action.category == "Transportation":
                        if tx["category"] != "Transportation":
                            tx["category"] = "Transportation"
                            reward += 0.1
                            self.task_scores["ledger_cleanup"] = min(1.0, self.task_scores["ledger_cleanup"] + 0.02)
                    elif tx["vendor"] == "SAFEWAY #33" and action.category == "Groceries":
                        if tx["category"] != "Groceries":
                            tx["category"] = "Groceries"
                            reward += 0.1
                            self.task_scores["ledger_cleanup"] = min(1.0, self.task_scores["ledger_cleanup"] + 0.02)
                            
        elif action.action_type == "CancelSubscription":
            new_subs = []
            for sub in self.subscriptions:
                if sub["vendor_name"] == action.vendor_name:
                    if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90:
                        reward += 0.5
                        self.task_scores["subscription_audit"] = min(1.0, self.task_scores["subscription_audit"] + 0.5)
                    else:
                        new_subs.append(sub)  # Keep valid subscriptions
                else:
                    new_subs.append(sub)
            self.subscriptions = new_subs
            
        elif action.action_type == "TransferFunds":
            amt = action.amount or 0.0
            if action.from_account == "Savings" and action.to_account == "Checking":
                if self.savings_balance >= amt > 0:
                    self.savings_balance -= amt
                    self.checking_balance += amt

        self.days_passed += 1
        done = False
        
        # Grader for Task 3: Cash Flow
        for sub in self.subscriptions:
            if sub.get("due_in_days") is not None:
                if sub["due_in_days"] == self.days_passed:
                    if self.checking_balance >= sub["cost"]:
                        self.checking_balance -= sub["cost"]
                        self.task_scores["cash_flow"] = 1.0
                    else:
                        self.checking_balance -= sub["cost"]
                        self.checking_balance -= 35.0  # overdraft
                        reward -= 2.0
                        self.task_scores["cash_flow"] = 0.0
                        done = True
                        
        if self._state.step_count >= 100:
            done = True
            
        # Clip and round reward like the sample repo
        reward = round(float(np.clip(reward, -1.0, 1.0)), 4)
        
        obs = self._get_obs(reward=reward, done=done)
        
        if done:
            obs.final_score = self._compute_final_score()
            
        return obs

    def _compute_final_score(self) -> float:
        task_id = self._state.task_id if hasattr(self._state, "task_id") else "ledger_cleanup"
        
        if task_id == "ledger_cleanup":
            correct = sum(1 for tx in self.ledger if tx.get("category") in ["Transportation", "Groceries"])
            return ledger_grader_inst(correct, 50)
            
        elif task_id == "subscription_audit":
            unnecessary_remaining = sum(1 for sub in self.subscriptions if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90)
            cancelled = 2 - unnecessary_remaining
            return subscription_grader_inst(cancelled, 2)
            
        elif task_id == "cash_flow":
            improvement = self.checking_balance - 1200.0
            return cash_flow_grader_inst(improvement, 500.0)
            
        return 0.001

    @property
    def state(self) -> State:
        return self._state

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        return self.TASKS
