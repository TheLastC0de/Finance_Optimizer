from typing import List, Dict, Any
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import FinanceOptimizerAction, FinanceOptimizerObservation

import numpy as np
from graders.ledger_grader import LedgerGrader
from graders.subscription_grader import SubscriptionGrader
from graders.cash_flow_grader import CashFlowGrader
from graders.fraud_grader import FraudGrader
from graders.savings_grader import SavingsGrader
from graders.duplicate_grader import DuplicateGrader

# Initialize singletons for environment loop
ledger_grader_inst = LedgerGrader()
subscription_grader_inst = SubscriptionGrader()
cash_flow_grader_inst = CashFlowGrader()
fraud_grader_inst = FraudGrader()
savings_grader_inst = SavingsGrader()
duplicate_grader_inst = DuplicateGrader()

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
        },
        {
            "task_id": "fraud_categorization",
            "name": "Fraud Categorization",
            "difficulty": "medium",
            "description": "Identify and categorize a highly anomalous transaction as Fraud.",
            "data_corpus": [],
            "aliases": ["task_fraud", "flag_fraud"]
        },
        {
            "task_id": "savings_builder",
            "name": "Savings Builder",
            "difficulty": "hard",
            "description": "Move idle cash to savings, maintaining exactly a minimum balance in checking.",
            "data_corpus": [],
            "aliases": ["task_savings", "build_savings"]
        },
        {
            "task_id": "duplicate_charge_alert",
            "name": "Duplicate Charge Alert",
            "difficulty": "hard",
            "description": "Identify a duplicated charge in the ledger and alert the system with its transaction ID.",
            "data_corpus": [],
            "aliases": ["task_duplicate", "alert_duplicate"]
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
            "cash_flow": 0.0,
            "fraud_categorization": 0.0,
            "savings_builder": 0.0,
            "duplicate_charge_alert": 0.0
        }
        self.original_excess = 0.0

    def reset(self, seed: int | None = None, task_id: str | None = None, **kwargs) -> FinanceOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._state.task_id = task_id or "ledger_cleanup"
        self.task_scores = {k: 0.0 for k in self.task_scores}
        
        rng = np.random.RandomState(seed if seed is not None else int(uuid4().int % 2**32))
        
        self.ledger = []
        vendors_transport = ["UBER *TRIP", "LYFT *RIDE", "BART *TRANSIT", "LIME *SCOOTER"]
        vendors_groceries = ["SAFEWAY #33", "WHOLEFOODS", "TRADER JOE", "TARGET *GROC"]
        
        # Base realistic ledger
        for i in range(50):
            if rng.rand() > 0.5:
                vendor = rng.choice(vendors_transport)
                amount = -round(float(rng.uniform(5.0, 35.0)), 2)
            else:
                vendor = rng.choice(vendors_groceries)
                amount = -round(float(rng.uniform(20.0, 150.0)), 2)
            self.ledger.append({"id": f"tx_{i}", "vendor": vendor, "amount": amount, "category": "Uncategorized"})
            
        self.subscriptions = [
            {"vendor_name": "Netflix_Primary", "cost": 15.99, "type": "streaming", "duplicate": False},
            {"vendor_name": "Netflix_Duplicate", "cost": 15.99, "type": "streaming", "duplicate": True},
            {"vendor_name": "Gym", "cost": float(rng.choice([50.0, 75.0, 100.0])), "type": "gym", "last_visit_days_ago": int(rng.uniform(60, 120))},
            {"vendor_name": "Rent", "cost": float(rng.choice([1500.0, 1800.0, 2000.0])), "type": "housing", "due_in_days": int(rng.uniform(3, 10))}
        ]
        
        self.checking_balance = round(float(rng.uniform(800.0, 1500.0)), 2)
        self.savings_balance = round(float(rng.uniform(500.0, 3000.0)), 2)
        self.days_passed = 0
        self.original_excess = 0.0
        self.is_done = False

        # Inject task-specific state invariants
        if self._state.task_id == "fraud_categorization":
            self.ledger.append({"id": "tx_fraud_99", "vendor": "UNKNOWN INTL *RUSSIA", "amount": -round(float(rng.uniform(3000.0, 8000.0)), 2), "category": "Uncategorized"})
            
        elif self._state.task_id == "savings_builder":
            self.checking_balance = round(float(rng.uniform(2000.0, 4000.0)), 2)
            self.savings_balance = 0.0
            self.original_excess = self.checking_balance - 500.0

        elif self._state.task_id == "duplicate_charge_alert":
            dup_vendor = rng.choice(["AMAZON.COM", "APPLE.COM", "STEAM GAMES"])
            dup_amount = -round(float(rng.uniform(50.0, 200.0)), 2)
            self.ledger.append({"id": "tx_dup_orig", "vendor": dup_vendor, "amount": dup_amount, "category": "Uncategorized"})
            self.ledger.append({"id": "tx_dup_copy", "vendor": dup_vendor, "amount": dup_amount, "category": "Uncategorized"})
            
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
        done = False
        task_id = getattr(self._state, "task_id", "ledger_cleanup")
        
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
                    elif tx["vendor"] == "UNKNOWN INTL *RUSSIA" and action.category == "Fraud":
                        if tx["category"] != "Fraud":
                            tx["category"] = "Fraud"
                            reward += 1.0
                            self.task_scores["fraud_categorization"] = 1.0
                            if task_id == "fraud_categorization":
                                done = True

            # Check if all transactions categorized
            if task_id == "ledger_cleanup":
                uncategorized = sum(1 for tx in self.ledger if tx["category"] == "Uncategorized")
                if uncategorized == 0:
                    done = True
                            
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
            # Check if all unnecessary subs cancelled
            if task_id == "subscription_audit":
                unnecessary = sum(1 for s in self.subscriptions if s.get("duplicate") or s.get("last_visit_days_ago", 0) >= 90)
                if unnecessary == 0:
                    done = True
            
        elif action.action_type == "TransferFunds":
            amt = action.amount or 0.0
            if action.from_account == "Savings" and action.to_account == "Checking":
                if self.savings_balance >= amt > 0:
                    self.savings_balance -= amt
                    self.checking_balance += amt
            elif action.from_account == "Checking" and action.to_account == "Savings":
                if self.checking_balance >= amt > 0:
                    self.checking_balance -= amt
                    self.savings_balance += amt
                    if task_id == "savings_builder":
                        reward += 0.5

        elif action.action_type == "SetAlert":
            if action.text == "done":
                done = True
            elif action.text == "tx_dup_copy" and task_id == "duplicate_charge_alert":
                reward += 1.0
                self.task_scores["duplicate_charge_alert"] = 1.0
                done = True

        self.days_passed += 1
        
        # Cash flow: auto-deduct rent on due day (only for cash_flow task)
        if task_id == "cash_flow":
            for sub in self.subscriptions:
                if sub.get("due_in_days") is not None:
                    if sub["due_in_days"] == self.days_passed:
                        if self.checking_balance >= sub["cost"]:
                            self.checking_balance -= sub["cost"]
                            self.task_scores["cash_flow"] = 1.0
                            done = True
                        else:
                            self.checking_balance -= sub["cost"]
                            self.checking_balance -= 35.0  # overdraft fee
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
            self.is_done = True
            
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
            # Score based on whether overdraft was avoided (task_score=1.0) or not (0.0)
            return cash_flow_grader_inst(self.task_scores["cash_flow"], 1.0)
            
        elif task_id == "fraud_categorization":
            fraud_identified = self.task_scores["fraud_categorization"] == 1.0
            return fraud_grader_inst(fraud_identified)

        elif task_id == "savings_builder":
            return savings_grader_inst(self.checking_balance, 500.0, self.original_excess)

        elif task_id == "duplicate_charge_alert":
            alert_set = self.task_scores["duplicate_charge_alert"] == 1.0
            return duplicate_grader_inst(alert_set)
            
        return 0.001

    @property
    def state(self) -> State:
        return self._state

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        return self.TASKS
