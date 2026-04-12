# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Finance Optimizer Environment.

This module creates an HTTP server that exposes the FinanceOptimizerEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from models import FinanceOptimizerAction, FinanceOptimizerObservation
from server.finance_optimizer_environment import FinanceOptimizerEnvironment


# Create the app with web interface and README integration
app = create_app(
    FinanceOptimizerEnvironment,
    FinanceOptimizerAction,
    FinanceOptimizerObservation,
    env_name="finance_optimizer",
    max_concurrent_envs=10,  # increase this number to allow more concurrent WebSocket sessions
)

from typing import Any, List

# Register tasks from the environment
TASK_REGISTRY = {
    task["task_id"]: task for task in FinanceOptimizerEnvironment.TASKS
}

@app.get("/tasks")
def list_tasks() -> dict[str, List[dict[str, Any]]]:
    tasks = []
    for task in TASK_REGISTRY.values():
        tasks.append({
            "id": task["task_id"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "name": task.get("name", task["task_id"]),
            "score_range": [0.0, 1.0],
            "data_corpus": task.get("data_corpus", []),
            "max_steps": 100,
            "action_schema": FinanceOptimizerAction.model_json_schema()
        })
    return {"tasks": tasks}

import threading

_env: FinanceOptimizerEnvironment | None = None
_env_lock = threading.Lock()

def _get_env() -> FinanceOptimizerEnvironment:
    global _env
    if _env is None:
        _env = FinanceOptimizerEnvironment()
    return _env

@app.get("/grader")
async def get_grader_score():
    with _env_lock:
        env = _get_env()
        score = env._compute_final_score() if getattr(env._state, "task_id", None) else 0.0
        task_id = getattr(env._state, "task_id", "ledger_cleanup")
        
        return {
            "task_id": task_id,
            "score": score,
            "done": getattr(env, "is_done", False),
        }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/baseline")
async def run_baseline():
    """Run baseline heuristic agent against all tasks and return scores."""
    task_ids = [t["task_id"] for t in FinanceOptimizerEnvironment.TASKS]
    results = []
    
    for task_id in task_ids:
        with _env_lock:
            env = _get_env()
            obs = env.reset(seed=42, task_id=task_id)

        score = 0.0
        steps = 0

        while not obs.done:
            steps += 1
            action_dict = {}

            if task_id == "ledger_cleanup":
                target_tx = next(
                    (tx for tx in obs.ledger if tx["category"] == "Uncategorized"),
                    None,
                )
                if target_tx:
                    action_dict = {
                        "action_type": "CategorizeTransaction",
                        "tx_id": target_tx["id"],
                    }
                    if "UBER" in target_tx["vendor"]:
                        action_dict["category"] = "Transportation"
                    else:
                        action_dict["category"] = "Groceries"
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}

            elif task_id == "subscription_audit":
                target_sub = next(
                    (
                        sub
                        for sub in obs.subscriptions
                        if sub.get("duplicate")
                        or sub.get("last_visit_days_ago", 0) >= 90
                    ),
                    None,
                )
                if target_sub:
                    action_dict = {
                        "action_type": "CancelSubscription",
                        "vendor_name": target_sub["vendor_name"],
                    }
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}

            elif task_id == "cash_flow":
                if obs.checking_balance < 1500 and obs.savings_balance > 0:
                    action_dict = {
                        "action_type": "TransferFunds",
                        "from_account": "Savings",
                        "to_account": "Checking",
                        "amount": 500.0,
                    }
                else:
                    action_dict = {"action_type": "SetAlert", "text": "wait"}

            elif task_id == "fraud_categorization":
                target_fraud = next(
                    (tx for tx in obs.ledger if tx["vendor"] == "UNKNOWN INTL *RUSSIA" and tx["category"] != "Fraud"),
                    None,
                )
                if target_fraud:
                    action_dict = {
                        "action_type": "CategorizeTransaction",
                        "tx_id": target_fraud["id"],
                        "category": "Fraud",
                    }
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}

            elif task_id == "savings_builder":
                if obs.checking_balance > 500:
                    excess = obs.checking_balance - 500
                    action_dict = {
                        "action_type": "TransferFunds",
                        "from_account": "Checking",
                        "to_account": "Savings",
                        "amount": excess,
                    }
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}

            elif task_id == "duplicate_charge_alert":
                action_dict = {"action_type": "SetAlert", "text": "tx_dup_copy"}

            action = FinanceOptimizerAction(**action_dict)

            with _env_lock:
                obs = env.step(action)

            if obs.done and obs.final_score is not None:
                score = obs.final_score

        results.append(
            {
                "task_id": task_id,
                "score": round(score, 4),
                "resolved": score >= 0.5,
                "steps": steps,
            }
        )

    total = sum(r["score"] for r in results)
    resolved_count = sum(1 for r in results if r["resolved"])
    return {
        "model": "heuristic",
        "results": results,
        "total_score": round(total, 3),
        "average_score": round(total / len(results), 3) if results else 0.0,
        "resolved": f"{resolved_count}/{len(results)}",
    }



def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m finance_optimizer.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn finance_optimizer.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
