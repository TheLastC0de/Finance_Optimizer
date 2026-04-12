---
title: Finance Optimizer Environment Server
emoji: 📻
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Finance Optimizer Environment

A real-world task simulation acting as a personal financial auditor to categorize spending, identify wasted money on forgotten subscriptions, and prevent overdrafts.

## Quick Start

You can use the environment through the HTTP API:

```python
import requests

# Rest
res = requests.post("http://localhost:8000/reset")
print(res.json())

# Step
action = {
    "action_type": "CategorizeTransaction",
    "tx_id": "tx_0",
    "category": "Transportation"
}
res = requests.post("http://localhost:8000/step", json={"action": action})
print(res.json())
```

## Environment Details

### Action
**FinanceOptimizerAction**: Polymorphic action type
- `action_type`: "CategorizeTransaction", "CancelSubscription", "TransferFunds", "SetAlert"
- `tx_id` (str, optional)
- `category` (str, optional)
- `vendor_name` (str, optional)
- `from_account` (str, optional)
- `to_account` (str, optional)
- `amount` (float, optional)
- `text` (str, optional)

### Observation
- `ledger`: List of recent transactions.
- `subscriptions`: List of active recurring subscriptions.
- `checking_balance`: Float.
- `savings_balance`: Float.

### Tasks & Grades
- **Task 1 (Easy): Ledger Cleanup**. Categorize 50 raw transactions correctly. Grade 0.0 - 1.0.
- **Task 2 (Medium): Subscription Audit**. Identify duplicate subscriptions and cancel them. Grade 0.0 - 1.0.
- **Task 3 (Hard): Cash Flow**. Simulate 7 days and transfer funds to prevent an overdraft. Grade 0.0 - 1.0.

### Baseline Scores
Running `python scripts/baseline.py` yields:
- Task 1: 1.0/1.0
- Task 2: 1.0/1.0
- Task 3: 1.0/1.0

## Deployment

Deploy via OpenEnv CLI:
```bash
openenv push --namespace my-org
```
