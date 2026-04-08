from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from finance_optimizer.models import FinanceOptimizerAction, FinanceOptimizerObservation
except ModuleNotFoundError:
    from models import FinanceOptimizerAction, FinanceOptimizerObservation

class FinanceOptimizerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.ledger = []
        self.subscriptions = []
        self.checking_balance = 1200.0
        self.savings_balance = 1000.0
        self.days_passed = 0

    def reset(self) -> FinanceOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
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
        return FinanceOptimizerObservation(
            ledger=self.ledger,
            subscriptions=self.subscriptions,
            checking_balance=self.checking_balance,
            savings_balance=self.savings_balance,
            done=done,
            reward=reward
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
                    elif tx["vendor"] == "SAFEWAY #33" and action.category == "Groceries":
                        if tx["category"] != "Groceries":
                            tx["category"] = "Groceries"
                            reward += 0.1
                            
        elif action.action_type == "CancelSubscription":
            new_subs = []
            for sub in self.subscriptions:
                if sub["vendor_name"] == action.vendor_name:
                    if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90:
                        reward += 0.5
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
        
        for sub in self.subscriptions:
            if sub.get("due_in_days") is not None:
                if sub["due_in_days"] == self.days_passed:
                    if self.checking_balance >= sub["cost"]:
                        self.checking_balance -= sub["cost"]
                    else:
                        self.checking_balance -= sub["cost"]
                        self.checking_balance -= 35.0  # overdraft
                        reward -= 2.0
                        done = True
                        
        if self._state.step_count >= 50:
            done = True
            
        return self._get_obs(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state
