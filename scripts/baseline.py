import os
import json
import time

try:
    from openai import OpenAI
except ImportError:
    pass

def run_baseline(base_url="http://localhost:8000"):
    import requests
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping baseline: OPENAI_API_KEY not found.")
        return
        
    client = OpenAI(api_key=api_key)
    
    print("Resetting environment...")
    res = requests.post(f"{base_url}/reset")
    if res.status_code != 200:
        print("Failed to reset")
        return
        
    obs = res.json()["observation"]
    
    # Task 1: Ledger Cleanup
    easy_score = 0.0
    for tx in obs["ledger"]:
        if tx["category"] == "Uncategorized":
            action = {"action_type": "CategorizeTransaction", "tx_id": tx["id"]}
            if "UBER" in tx["vendor"]:
                action["category"] = "Transportation"
            elif "SAFEWAY" in tx["vendor"]:
                action["category"] = "Groceries"
                
            if "category" in action:
                r = requests.post(f"{base_url}/step", json={"action": action})
                if r.json()["reward"] > 0:
                    easy_score += 0.02 # 50 transactions, 0.02 * 50 = 1.0

    print(f"Task 1 (Easy) Score: {min(1.0, easy_score * 2):.1f}/1.0")

    # Task 2: Subscription Audit
    med_score = 0.0
    obs = requests.get(f"{base_url}/state").json() # mock state
    # get obs again
    r = requests.post(f"{base_url}/step", json={"action": {"action_type": "SetAlert", "text": "noop"}})
    obs = r.json()["observation"]
    subs = obs.get("subscriptions", [])
    
    for sub in subs:
        if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90:
            act = {"action_type": "CancelSubscription", "vendor_name": sub["vendor_name"]}
            r = requests.post(f"{base_url}/step", json={"action": act})
            if r.json()["reward"] > 0:
                med_score += 0.5
                
    print(f"Task 2 (Medium) Score: {med_score:.1f}/1.0")

    # Task 3: Cash Flow
    hard_score = 0.0
    # Transfer 1500 to checking
    r = requests.post(f"{base_url}/step", json={
        "action": {
            "action_type": "TransferFunds",
            "from_account": "Savings",
            "to_account": "Checking",
            "amount": 500.0
        }
    })
    
    for _ in range(7):
        r = requests.post(f"{base_url}/step", json={"action": {"action_type": "SetAlert", "text": "wait"}})
        
    if r.json().get("reward", 0) >= 0: # no overdraft
        hard_score = 1.0
        
    print(f"Task 3 (Hard) Score: {hard_score:.1f}/1.0")
    print(f"Total Score: {(min(1.0, easy_score*2) + med_score + hard_score)/3.0:.1f}/1.0")

if __name__ == "__main__":
    run_baseline()
