"""Full integration test: run baseline solver over WebSocket for ALL 6 tasks."""
import asyncio
import json
from client import FinanceOptimizerEnv
from models import FinanceOptimizerAction

# Canonical vendor mapping (same as environment)
VENDOR_CATEGORIES = {
    "UBER *TRIP": "Transportation", "LYFT *RIDE": "Transportation",
    "BART *TRANSIT": "Transportation", "LIME *SCOOTER": "Transportation",
    "SAFEWAY #33": "Groceries", "WHOLEFOODS": "Groceries",
    "TRADER JOE": "Groceries", "TARGET *GROC": "Groceries",
    "DOORDASH": "Dining", "GRUBHUB": "Dining",
    "STARBUCKS #12": "Dining", "CHIPOTLE #09": "Dining",
    "AMC THEATERS": "Entertainment", "STEAM GAMES": "Entertainment",
    "SPOTIFY PREMIUM": "Entertainment", "TICKETMASTER": "Entertainment",
    "PG&E ELECTRIC": "Utilities", "AT&T WIRELESS": "Utilities",
    "COMCAST CABLE": "Utilities", "WATER DEPT": "Utilities",
}
FRAUD_VENDORS = {"UNKNOWN INTL *RUSSIA", "WIRE *OFFSHORE", "CRYPTO *ANON"}

async def run_all_tasks():
    env = FinanceOptimizerEnv(base_url="http://localhost:8000")
    await env.connect()
    
    all_tasks = [
        "ledger_cleanup", "subscription_audit", "cash_flow",
        "fraud_categorization", "savings_builder", "duplicate_charge_alert"
    ]
    
    for task_id in all_tasks:
        result = await env.reset(seed=42, task_id=task_id)
        obs = result.observation
        steps = 0
        total_reward = 0.0
        
        while not obs.done and steps < 80:
            steps += 1
            action_dict = {}
            
            if task_id == "ledger_cleanup":
                target = next((tx for tx in obs.ledger if tx["category"] == "Uncategorized"), None)
                if target:
                    cat = VENDOR_CATEGORIES.get(target["vendor"], "Other")
                    action_dict = {"action_type": "CategorizeTransaction", "tx_id": target["id"], "category": cat}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}
                    
            elif task_id == "subscription_audit":
                target = next((s for s in obs.subscriptions if s.get("duplicate") or s.get("last_visit_days_ago", 0) >= 90), None)
                if target:
                    action_dict = {"action_type": "CancelSubscription", "vendor_name": target["vendor_name"]}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}
                    
            elif task_id == "cash_flow":
                if obs.checking_balance < 2500 and obs.savings_balance > 0:
                    action_dict = {"action_type": "TransferFunds", "from_account": "Savings", "to_account": "Checking", "amount": min(obs.savings_balance, 1000.0)}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "wait"}
                    
            elif task_id == "fraud_categorization":
                target = next((tx for tx in obs.ledger if tx["vendor"] in FRAUD_VENDORS and tx["category"] != "Fraud"), None)
                if target:
                    action_dict = {"action_type": "CategorizeTransaction", "tx_id": target["id"], "category": "Fraud"}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}
                    
            elif task_id == "savings_builder":
                if obs.checking_balance > 500:
                    action_dict = {"action_type": "TransferFunds", "from_account": "Checking", "to_account": "Savings", "amount": obs.checking_balance - 500}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}
                    
            elif task_id == "duplicate_charge_alert":
                action_dict = {"action_type": "SetAlert", "text": "tx_dup_copy"}
            
            action = FinanceOptimizerAction(**action_dict)
            result = await env.step(action)
            obs = result.observation
            total_reward += float(obs.reward or 0)
        
        score = obs.final_score if obs.final_score is not None else 0.0
        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"  [{status}] {task_id}: score={score:.4f}  steps={steps}  total_reward={total_reward:.2f}")
    
    await env.close()
    print("\nWebSocket integration test complete.")

asyncio.run(run_all_tasks())
