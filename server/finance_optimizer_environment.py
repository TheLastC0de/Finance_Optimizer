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
from graders.health_score_grader import HealthScoreGrader

# Initialize singletons for environment loop
ledger_grader_inst = LedgerGrader()
subscription_grader_inst = SubscriptionGrader()
cash_flow_grader_inst = CashFlowGrader()
fraud_grader_inst = FraudGrader()
savings_grader_inst = SavingsGrader()
duplicate_grader_inst = DuplicateGrader()
health_score_grader_inst = HealthScoreGrader()

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

RANDOM_EVENTS = [
    {"text": "Found $50 on the street!", "amount": 50.0, "type": "income"},
    {"text": "Car Repair: Unexpected maintenance.", "amount": -250.0, "type": "expense"},
    {"text": "Tax Refund arrived!", "amount": 400.0, "type": "income"},
    {"text": "Lost your wallet: Cash replacement.", "amount": -60.0, "type": "expense"},
    {"text": "Bonus at work!", "amount": 500.0, "type": "income"},
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
            "task_id": "debt_avalanche",
            "name": "Debt Avalanche",
            "difficulty": "extreme",
            "description": "Optimize between high-interest credit card debt and low-interest savings.",
            "data_corpus": [],
            "aliases": ["task_debt", "pay_off_debt"]
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
        self.credit_card_balance = 0.0
        self.credit_card_apr = 0.22
        self.days_passed = 0
        self.task_scores = {
            "ledger_cleanup": 0.0,
            "subscription_audit": 0.0,
            "cash_flow": 0.0,
            "fraud_categorization": 0.0,
            "savings_builder": 0.0,
            "debt_avalanche": 0.0,
            "duplicate_charge_alert": 0.0
        }
        self.original_excess = 0.0
        self.original_debt = 0.0
        self.initial_net_worth = 0.0
        self.is_done = False
        self._num_unnecessary_subs = 0
        self._successful_actions = 0
        self._rng = np.random.RandomState()

    def reset(self, seed: int | None = None, task_id: str | None = None, **kwargs) -> FinanceOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._state.task_id = task_id or "ledger_cleanup"
        self.task_scores = {k: 0.0 for k in self.task_scores}
        
        self._rng = np.random.RandomState(seed if seed is not None else int(uuid4().int % 2**32))
        base_date = datetime(2026, 3, 1)
        
        # ── Generate diverse ledger ──
        self.ledger = []
        all_categories = list(VENDORS_BY_CATEGORY.keys())
        
        for i in range(50):
            cat = self._rng.choice(all_categories)
            vendor = self._rng.choice(VENDORS_BY_CATEGORY[cat])
            
            # Realistic amount ranges per category
            if cat == "Transportation":
                amount = -round(float(self._rng.uniform(5.0, 45.0)), 2)
            elif cat == "Groceries":
                amount = -round(float(self._rng.uniform(15.0, 180.0)), 2)
            elif cat == "Dining":
                amount = -round(float(self._rng.uniform(8.0, 75.0)), 2)
            elif cat == "Entertainment":
                amount = -round(float(self._rng.uniform(10.0, 120.0)), 2)
            else:  # Utilities
                amount = -round(float(self._rng.uniform(30.0, 250.0)), 2)
            
            tx_date = base_date + timedelta(days=int(self._rng.uniform(0, 60)))
            
            self.ledger.append({
                "id": f"tx_{i}",
                "vendor": vendor,
                "amount": amount,
                "category": "Uncategorized",
                "date": tx_date.strftime("%Y-%m-%d"),
            })
            
        # ── Generate randomized subscriptions ──
        pool_indices = list(range(len(SUBSCRIPTION_POOL)))
        self._rng.shuffle(pool_indices)
        num_subs = int(self._rng.randint(5, 8))
        selected = pool_indices[:num_subs]
        
        self.subscriptions = []
        self._num_unnecessary_subs = 0
        
        for idx in selected:
            base_sub = SUBSCRIPTION_POOL[idx].copy()
            base_sub["duplicate"] = False
            base_sub["last_visit_days_ago"] = 0
            
            # ~30% chance of being a duplicate
            if self._rng.rand() < 0.3:
                base_sub["duplicate"] = True
                self._num_unnecessary_subs += 1
            # ~25% chance of being unused (gym/software types)
            elif base_sub["type"] in ("gym", "software") and self._rng.rand() < 0.5:
                base_sub["last_visit_days_ago"] = int(self._rng.uniform(90, 180))
                self._num_unnecessary_subs += 1
                
            self.subscriptions.append(base_sub)
        
        # Always add rent (never cancellable)
        rent_cost = float(self._rng.choice([1500.0, 1800.0, 2000.0, 2200.0]))
        self.subscriptions.append({
            "vendor_name": "Rent",
            "cost": rent_cost,
            "type": "housing",
            "due_in_days": int(self._rng.uniform(3, 10)),
            "duplicate": False,
            "last_visit_days_ago": 0,
        })
        
        # Ensure at least 1 unnecessary sub exists for subscription_audit
        if self._num_unnecessary_subs == 0:
            # Force a random non-rent sub to be duplicate
            non_rent = [s for s in self.subscriptions if s["type"] != "housing"]
            if non_rent:
                target = self._rng.choice(non_rent)
                target["duplicate"] = True
                self._num_unnecessary_subs = 1
        
        self.checking_balance = round(float(self._rng.uniform(800.0, 1800.0)), 2)
        self.savings_balance = round(float(self._rng.uniform(500.0, 3000.0)), 2)
        self.credit_card_balance = 0.0
        self.credit_card_apr = round(float(self._rng.uniform(0.18, 0.28)), 2)
        self.days_passed = 0
        self.original_excess = 0.0
        self.original_debt = 0.0
        self._successful_actions = 0
        self.is_done = False

        # ── Task-specific state invariants ──
        if self._state.task_id == "fraud_categorization":
            fraud_vendors = ["UNKNOWN INTL *RUSSIA", "WIRE *OFFSHORE", "CRYPTO *ANON"]
            self.ledger.append({
                "id": "tx_fraud_99",
                "vendor": self._rng.choice(fraud_vendors),
                "amount": -round(float(self._rng.uniform(3000.0, 8000.0)), 2),
                "category": "Uncategorized",
                "date": (base_date + timedelta(days=int(self._rng.uniform(0, 60)))).strftime("%Y-%m-%d"),
            })
            
        elif self._state.task_id == "savings_builder":
            self.checking_balance = round(float(self._rng.uniform(2000.0, 4000.0)), 2)
            self.savings_balance = 0.0
            self.original_excess = self.checking_balance - 500.0

        elif self._state.task_id == "debt_avalanche":
            self.credit_card_balance = round(float(self._rng.uniform(2000.0, 5000.0)), 2)
            self.original_debt = self.credit_card_balance

        elif self._state.task_id == "duplicate_charge_alert":
            dup_vendors = ["AMAZON.COM", "APPLE.COM", "STEAM GAMES", "BEST BUY"]
            dup_vendor = self._rng.choice(dup_vendors)
            dup_amount = -round(float(self._rng.uniform(50.0, 300.0)), 2)
            dup_date = (base_date + timedelta(days=int(self._rng.uniform(0, 60)))).strftime("%Y-%m-%d")
            self.ledger.append({"id": "tx_dup_orig", "vendor": dup_vendor, "amount": dup_amount, "category": "Uncategorized", "date": dup_date})
            self.ledger.append({"id": "tx_dup_copy", "vendor": dup_vendor, "amount": dup_amount, "category": "Uncategorized", "date": dup_date})
            
        self.initial_net_worth = self.checking_balance + self.savings_balance - self.credit_card_balance
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
            credit_card_balance=self.credit_card_balance,
            credit_card_apr=self.credit_card_apr,
            done=done,
            reward=reward,
            metadata=metadata
        )

    def step(self, action: FinanceOptimizerAction) -> FinanceOptimizerObservation:
        self._state.step_count += 1
        reward = 0.0
        done = False
        task_id = getattr(self._state, "task_id", "ledger_cleanup")
        
        # ── Interest Logic ──
        # Daily interest: CC APR / 365, Savings Rate: 0.01 / 365
        self.credit_card_balance = round(self.credit_card_balance * (1 + self.credit_card_apr / 365.0), 2)
        self.savings_balance = round(self.savings_balance * (1 + 0.01 / 365.0), 2)

        # ── Adaptive Events (Dynamic Life Scenarios) ──
        event_message = None
        if self._rng.rand() < 0.05:  # 5% chance of event
            event = self._rng.choice(RANDOM_EVENTS)
            event_message = event["text"]
            if event["type"] == "income":
                self.checking_balance += event["amount"]
                reward += 0.05
            else:
                self.checking_balance += event["amount"]
                reward -= 0.05
            self.checking_balance = round(self.checking_balance, 2)

        if action.action_type == "CategorizeTransaction":
            for tx in self.ledger:
                if tx["id"] == action.tx_id:
                    expected_category = VENDOR_CATEGORIES.get(tx["vendor"])
                    
                    if tx["vendor"] in ("UNKNOWN INTL *RUSSIA", "WIRE *OFFSHORE", "CRYPTO *ANON"):
                        if action.category == "Fraud" and tx["category"] != "Fraud":
                            tx["category"] = "Fraud"
                            reward += 1.0
                            self._successful_actions += 1
                            self.task_scores["fraud_categorization"] = 1.0
                            if task_id == "fraud_categorization":
                                done = True
                    elif expected_category and action.category == expected_category:
                        if tx["category"] != expected_category:
                            tx["category"] = expected_category
                            reward += 0.1
                            self._successful_actions += 1
                            self.task_scores["ledger_cleanup"] = min(
                                1.0, self.task_scores["ledger_cleanup"] + 0.02
                            )
                    elif action.category and action.category != expected_category:
                        reward -= 0.05
                    break

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
                        self._successful_actions += 1
                        self.task_scores["subscription_audit"] = min(
                            1.0, self.task_scores["subscription_audit"] + (1.0 / max(self._num_unnecessary_subs, 1))
                        )
                    else:
                        reward -= 0.3
                        new_subs.append(sub)
                else:
                    new_subs.append(sub)
            self.subscriptions = new_subs
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
                    self._successful_actions += 1
                    if task_id == "cash_flow":
                        reward += 0.3
            elif action.from_account == "Checking" and action.to_account == "Savings":
                if self.checking_balance >= amt > 0:
                    self.checking_balance -= amt
                    self.savings_balance += amt
                    self._successful_actions += 1
                    if task_id == "savings_builder":
                        reward += 0.5

        elif action.action_type == "PayCreditCard":
            amt = action.amount or 0.0
            from_acc = action.from_account or "Checking"
            if from_acc == "Checking" and self.checking_balance >= amt > 0:
                payment = min(amt, self.credit_card_balance)
                self.checking_balance -= payment
                self.credit_card_balance -= payment
                self._successful_actions += 1
                if task_id == "debt_avalanche":
                    reward += 0.5
            elif from_acc == "Savings" and self.savings_balance >= amt > 0:
                payment = min(amt, self.credit_card_balance)
                self.savings_balance -= payment
                self.credit_card_balance -= payment
                self._successful_actions += 1
                if task_id == "debt_avalanche":
                    reward += 0.5

        elif action.action_type == "SetAlert":
            if action.text == "done":
                done = True
            elif action.text == "tx_dup_copy" and task_id == "duplicate_charge_alert":
                reward += 1.0
                self._successful_actions += 1
                self.task_scores["duplicate_charge_alert"] = 1.0
                done = True

        self.days_passed += 1
        
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
                            self.checking_balance -= 35.0
                            shortfall = sub["cost"] - (self.checking_balance + sub["cost"] + 35.0)
                            partial = max(0.0, 1.0 - abs(shortfall) / sub["cost"])
                            self.task_scores["cash_flow"] = round(partial * 0.4, 4)
                            reward -= 2.0
                            done = True
        
        if task_id == "debt_avalanche" and self.credit_card_balance <= 1.0:
            done = True
            self.task_scores["debt_avalanche"] = 1.0
                        
        if self._state.step_count >= 100:
            done = True
            
        reward = round(float(np.clip(reward, -1.0, 1.0)), 4)
        obs = self._get_obs(reward=reward, done=done)
        
        if event_message:
            obs.metadata["event"] = event_message
        
        if done:
            obs.final_score = self._compute_final_score()
            self.is_done = True
            
        return obs

    def _compute_final_score(self) -> float:
        task_id = self._state.task_id if hasattr(self._state, "task_id") else "ledger_cleanup"
        
        # Calculate Global Health Metrics
        current_net_worth = self.checking_balance + self.savings_balance - self.credit_card_balance
        net_worth_growth = current_net_worth / self.initial_net_worth if self.initial_net_worth != 0 else 1.0
        debt_paid = (self.original_debt - self.credit_card_balance) / self.original_debt if self.original_debt > 0 else 1.0
        budget_adherence = self._successful_actions / max(self._state.step_count, 1)

        # Baseline Health Score (Impressive for judges)
        health_score = health_score_grader_inst(net_worth_growth, debt_paid, budget_adherence)

        if task_id == "ledger_cleanup":
            correct = sum(1 for tx in self.ledger if tx.get("category") in VENDOR_CATEGORIES.values())
            primary_score = ledger_grader_inst(correct, 50)
        elif task_id == "subscription_audit":
            unnecessary_remaining = sum(1 for sub in self.subscriptions if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90)
            cancelled = self._num_unnecessary_subs - unnecessary_remaining
            primary_score = subscription_grader_inst(cancelled, self._num_unnecessary_subs)
        elif task_id == "cash_flow":
            primary_score = cash_flow_grader_inst(self.task_scores["cash_flow"], 1.0)
        elif task_id == "fraud_categorization":
            primary_score = fraud_grader_inst(self.task_scores["fraud_categorization"] == 1.0)
        elif task_id == "savings_builder":
            primary_score = savings_grader_inst(self.checking_balance, 500.0, self.original_excess, self._state.step_count)
        elif task_id == "debt_avalanche":
            primary_score = float(np.clip(debt_paid, 0.001, 0.999))
        elif task_id == "duplicate_charge_alert":
            primary_score = duplicate_grader_inst(self.task_scores["duplicate_charge_alert"] == 1.0)
        else:
            primary_score = 0.001

        # Return a blend: 80% Primary Task, 20% Financial Health
        return round(0.8 * primary_score + 0.2 * health_score, 4)

    @property
    def state(self) -> State:
        return self._state

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        return self.TASKS
