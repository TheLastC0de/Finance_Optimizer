import asyncio
import os
import sys
from typing import Any, List, Optional

from client import FinanceOptimizerEnv
from models import FinanceOptimizerAction

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
TASK_NAME = os.getenv("TASK_NAME", "")
ALL_TASKS = ["ledger_cleanup", "subscription_audit", "cash_flow"]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={float(reward):.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={float(score):.2f} rewards={rewards_str}", flush=True)

async def run_task(task_name: str, env: FinanceOptimizerEnv, seed: int) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_name, env="finance_optimizer", model="rule-based")
    
    try:
        result = await env.reset(seed=seed, task_id=task_name)
        obs = result.observation
        
        while not obs.done:
            steps_taken += 1
            action_dict = {}
            
            if task_name == "ledger_cleanup":
                target_tx = next((tx for tx in obs.ledger if tx["category"] == "Uncategorized"), None)
                if target_tx:
                    action_dict = {"action_type": "CategorizeTransaction", "tx_id": target_tx["id"]}
                    if "UBER" in target_tx["vendor"]:
                        action_dict["category"] = "Transportation"
                    else:
                        action_dict["category"] = "Groceries"
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}
                    
            elif task_name == "subscription_audit":
                target_sub = next((sub for sub in obs.subscriptions if sub.get("duplicate") or sub.get("last_visit_days_ago", 0) >= 90), None)
                if target_sub:
                    action_dict = {"action_type": "CancelSubscription", "vendor_name": target_sub["vendor_name"]}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "done"}
                    
            elif task_name == "cash_flow":
                if obs.checking_balance < 1500 and obs.savings_balance > 0:
                    action_dict = {"action_type": "TransferFunds", "from_account": "Savings", "to_account": "Checking", "amount": 500.0}
                else:
                    action_dict = {"action_type": "SetAlert", "text": "wait"}
            
            action = FinanceOptimizerAction(**action_dict)
            
            action_str_repr = f"{action.action_type}"
            
            try:
                result = await env.step(action)
            except Exception as exc:
                log_step(steps_taken, action_str_repr, 0.0, True, str(exc))
                break
                
            obs = result.observation
            reward = float(obs.reward) if obs.reward is not None else 0.0
            rewards.append(reward)
            
            final = getattr(obs, "final_score", None)
            if obs.done and final is not None:
                score = float(final)
                
            log_step(steps_taken, action_str_repr, reward, obs.done, None)
            
            if steps_taken > 100:
                break
                
        if score > 0.0:
            score = round(min(max(score, 0.01), 0.99), 2)
        success = score >= 0.5
        
    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    tasks_to_run = [TASK_NAME] if TASK_NAME else ALL_TASKS
    
    env = FinanceOptimizerEnv(base_url=ENV_URL)
    
    try:
        await env.connect()
        for i, task_name in enumerate(tasks_to_run):
            await run_task(task_name, env, seed=42 + i)
    finally:
        try:
            await env.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
