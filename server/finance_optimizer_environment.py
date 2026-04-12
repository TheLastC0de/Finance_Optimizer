from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime, timedelta
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

# ─── Vendor → Category mapping (canonical source of truth) ───
VENDOR_CATEGORIES: Dict[str, str] = {
    # Transportation
    "UBER *TRIP": "Transportation",
    "LYFT *RIDE": "Transportation",
    "BART *TRANSIT": "Transportation",
    "LIME *SCOOTER": "Transportation",
    # Groceries
    "SAFEWAY #33": "Groceries",
    "WHOLEFOODS": "Groceries",
    "TRADER JOE": "Groceries",
    "TARGET *GROC": "Groceries",
    # Dining
    "DOORDASH": "Dining",
    "GRUBHUB": "Dining",
    "STARBUCKS #12": "Dining",
    "CHIPOTLE #09": "Dining",
    # Entertainment
    "AMC THEATERS": "Entertainment",
    "STEAM GAMES": "Entertainment",
    "SPOTIFY PREMIUM": "Entertainment",
    "TICKETMASTER": "Entertainment",
    # Utilities
    "PG&E ELECTRIC": "Utilities",
    "AT&T WIRELESS": "Utilities",
    "COMCAST CABLE": "Utilities",
    "WATER DEPT": "Utilities",
}

# Group vendors by category for data generation
VENDORS_BY_CATEGORY: Dict[str, List[str]] = {}
for _vendor, _cat in VENDOR_CATEGORIES.items():
    VENDORS_BY_CATEGORY.setdefault(_cat, []).append(_vendor)

# Subscription pool for randomized generation
SUBSCRIPTION_POOL = [
    {"vendor_name": "Netflix", "cost": 15.99, "type": "streaming"},
    {"vendor_name": "Spotify", "cost": 9.99, "type": "streaming"},
    {"vendor_name": "HBO Max", "cost": 15.99, "type": "streaming"},
    {"vendor_name": "Disney+", "cost": 7.99, "type": "streaming"},
    {"vendor_name": "YouTube Premium", "cost": 13.99, "type": "streaming"},
    {"vendor_name": "Planet Fitness", "cost": 25.0, "type": "gym"},
    {"vendor_name": "Equinox", "cost": 200.0, "type": "gym"},
    {"vendor_name": "24 Hour Fitness", "cost": 50.0, "type": "gym"},
    {"vendor_name": "Adobe Creative", "cost": 54.99, "type": "software"},
    {"vendor_name": "Dropbox Plus", "cost": 11.99, "type": "software"},
    {"vendor_name": "AWS", "cost": 29.99, "type": "cloud"},
    {"vendor_name": "iCloud+", "cost": 2.99, "type": "cloud"},
]


class FinanceOptimizerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS = [
        {
            "task_id": "ledger_cleanup",
            "name": "Ledger Cleanup",
            "difficulty": "easy",
            "description": "Correctly categorize 50 raw transactions across 5 spending categories.",
            "data_corpus": [],
            "aliases": ["task_easy", "categorize_transactions"]
        },
        {
            "task_id": "subscription_audit",
            "name": "Subscription Audit",
            "difficulty": "medium",
            "description": "Identify and cancel duplicate or unused subscriptions from a realistic portfolio.",
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
        self.ledger: List[Dict[str, Any]] = []
        self.subscriptions: List[Dict[str, Any]] = []
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
        self.is_done = False
        self._num_unnecessary_subs = 0

    def reset(self, seed: int | None = None, task_id: str | None = None, **kwargs) -> FinanceOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._state.task_id = task_id or "ledger_cleanup"
        self.task_scores = {k: 0.0 for k in self.task_scores}
        
        rng = np.random.RandomState(seed if seed is not None else int(uuid4().int % 2**32))
        base_date = datetime(2026, 3, 1)
        
        # ── Generate diverse ledger ──
        self.ledger = []
        all_categories = list(VENDORS_BY_CATEGORY.keys())
        
        for i in range(50):
            cat = rng.choice(all_categories)
            vendor = rng.choice(VENDORS_BY_CATEGORY[cat])
            
            # Realistic amount ranges per category
            if cat == "Transportation":
                amount = -round(float(rng.uniform(5.0, 45.0)), 2)
            elif cat == "Groceries":
                amount = -round(float(rng.uniform(15.0, 180.0)), 2)
            elif cat == "Dining":
                amount = -round(float(rng.uniform(8.0, 75.0)), 2)
            elif cat == "Entertainment":
                amount = -round(float(rng.uniform(10.0, 120.0)), 2)
            else:  # Utilities
                amount = -round(float(rng.uniform(30.0, 250.0)), 2)
            
            tx_date = base_date + timedelta(days=int(rng.uniform(0, 60)))
            
            self.ledger.append({
                "id": f"tx_{i}",
                "vendor": vendor,
                "amount": amount,
                "category": "Uncategorized",
                "date": tx_date.strftime("%Y-%m-%d"),
            })
            
        # ── Generate randomized subscriptions ──
        pool_indices = list(range(len(SUBSCRIPTION_POOL)))
        rng.shuffle(pool_indices)
        num_subs = int(rng.randint(5, 8))
        selected = pool_indices[:num_subs]
        
        self.subscriptions = []
        self._num_unnecessary_subs = 0
        
        for idx in selected:
            base_sub = SUBSCRIPTION_POOL[idx].copy()
            base_sub["duplicate"] = False
            base_sub["last_visit_days_ago"] = 0
            
            # ~30% chance of being a duplicate
            if rng.rand() < 0.3:
                base_sub["duplicate"] = True
                self._num_unnecessary_subs += 1
            # ~25% chance of being unused (gym/software types)
            elif base_sub["type"] in ("gym", "software") and rng.rand() < 0.5:
                base_sub["last_visit_days_ago"] = int(rng.uniform(90, 180))
                self._num_unnecessary_subs += 1
                
            self.subscriptions.append(base_sub)
        
        # Always add rent (never cancellable)
        rent_cost = float(rng.choice([1500.0, 1800.0, 2000.0, 2200.0]))
        self.subscriptions.append({
            "vendor_name": "Rent",
            "cost": rent_cost,
            "type": "housing",
            "due_in_days": int(rng.uniform(3, 10)),
            "duplicate": False,
            "last_visit_days_ago": 0,
        })
        
        # Ensure at least 1 unnecessary sub exists for subscription_audit
        if self._num_unnecessary_subs == 0:
            # Force a random non-rent sub to be duplicate
            non_rent = [s for s in self.subscriptions if s["type"] != "housing"]
            if non_rent:
                target = rng.choice(non_rent)
                target["duplicate"] = True
                self._num_unnecessary_subs = 1
        
        self.checking_balance = round(float(rng.uniform(800.0, 1800.0)), 2)
        self.savings_balance = round(float(rng.uniform(500.0, 3000.0)), 2)
        self.days_passed = 0
        self.original_excess = 0.0
        self.is_done = False

        # ── Task-specific state invariants ──
        if self._state.task_id == "fraud_categorization":
            fraud_vendors = ["UNKNOWN INTL *RUSSIA", "WIRE *OFFSHORE", "CRYPTO *ANON"]
            self.ledger.append({
                "id": "tx_fraud_99",
                "vendor": rng.choice(fraud_vendors),
                "amount": -round(float(rng.uniform(3000.0, 8000.0)), 2),
                "category": "Uncategorized",
                "date": (base_date + timedelta(days=int(rng.uniform(0, 60)))).strftime("%Y-%m-%d"),
            })
            
        elif self._state.task_id == "savings_builder":
            self.checking_balance = round(float(rng.uniform(2000.0, 4000.0)), 2)
            self.savings_balance = 0.0
            self.original_excess = self.checking_balance - 500.0

        elif self._state.task_id == "duplicate_charge_alert":
            dup_vendors = ["AMAZON.COM", "APPLE.COM", "STEAM GAMES", "BEST BUY"]
            dup_vendor = rng.choice(dup_vendors)
            dup_amount = -round(float(rng.uniform(50.0, 300.0)), 2)
            dup_date = (base_date + timedelta(days=int(rng.uniform(0, 60)))).strftime("%Y-%m-%d")
            self.ledger.append({"id": "tx_dup_orig", "vendor": dup_vendor, "amount": dup_amount, "category": "Uncategorized", "date": dup_date})
            self.ledger.append({"id": "tx_dup_copy", "vendor": dup_vendor, "amount": dup_amount, "category": "Uncategorized", "date": dup_date})
            
        return self._get_obs(reward=0.0)

    def _get_obs(self, reward: float = 0.0, done: bool = False):
        metadata = {
            "tasks": self.TASKS,
            "task_scores": self.task_scores,
            "step": self._state.step_count,
            "vendor_categories": VENDOR_CATEGORIES,  # expose mapping to agents
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
                    expected_category = VENDOR_CATEGORIES.get(tx["vendor"])
                    
                    # Fraud vendors get special handling
                    if tx["vendor"] in ("UNKNOWN INTL *RUSSIA", "WIRE *OFFSHORE", "CRYPTO *ANON"):
                        if action.category == "Fraud" and tx["category"] != "Fraud":
                            tx["category"] = "Fraud"
                            reward += 1.0
                            self.task_scores["fraud_categorization"] = 1.0
                            if task_id == "fraud_categorization":
                                done = True
                    # Standard vendor categorization
                    elif expected_category and action.category == expected_category:
                        if tx["category"] != expected_category:
                            tx["category"] = expected_category
                            reward += 0.1
                            self.task_scores["ledger_cleanup"] = min(
                                1.0, self.task_scores["ledger_cleanup"] + 0.02
                            )
                    # Penalty for wrong categorization
                    elif action.category and action.category != expected_category:
                        reward -= 0.05
                    break

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
                        self.task_scores["subscription_audit"] = min(
                            1.0, self.task_scores["subscription_audit"] + (1.0 / max(self._num_unnecessary_subs, 1))
                        )
                    else:
                        reward -= 0.3  # penalty for cancelling a valid subscription
                        new_subs.append(sub)
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
                    if task_id == "cash_flow":
                        reward += 0.3
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
        
        # Cash flow: auto-deduct rent on due day
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
                            # Partial credit: how close were we?
                            shortfall = sub["cost"] - (self.checking_balance + sub["cost"] + 35.0)
                            partial = max(0.0, 1.0 - abs(shortfall) / sub["cost"])
                            self.task_scores["cash_flow"] = round(partial * 0.4, 4)  # max 0.4 for overdraft
                            reward -= 2.0
                            done = True
                        
        if self._state.step_count >= 100:
            done = True
            
        # Clip and round reward
        reward = round(float(np.clip(reward, -1.0, 1.0)), 4)
        
        obs = self._get_obs(reward=reward, done=done)
        
        if done:
            obs.final_score = self._compute_final_score()
            self.is_done = True
            
        return obs

    def _compute_final_score(self) -> float:
        task_id = self._state.task_id if hasattr(self._state, "task_id") else "ledger_cleanup"
        
        if task_id == "ledger_cleanup":
            correct = sum(1 for tx in self.ledger if tx.get("category") in VENDOR_CATEGORIES.values())
            return ledger_grader_inst(correct, 50)
            
        elif task_id == "subscription_audit":
            unnecessary_remaining = sum(1 for sub in self.subscriptions if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90)
            cancelled = self._num_unnecessary_subs - unnecessary_remaining
            return subscription_grader_inst(cancelled, self._num_unnecessary_subs)
            
        elif task_id == "cash_flow":
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
