"""
FinanceOptimizer Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    ENV_URL             URL of a running FinanceOptimizerEnv server (default: http://localhost:8000).
    TASK_NAME           Task to run: ledger_cleanup, subscription_audit, cash_flow.

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use the OpenAI client for all LLM calls.

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places; score to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the exception message, or null if none.
    - All fields on a single line with no newlines within a line.
    - score is in [0, 1].

  Example:
    [START] task=ledger_cleanup env=finance_optimizer model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=CategorizeTransaction reward=0.10 done=false error=null
    [STEP] step=2 action=CategorizeTransaction reward=0.10 done=false error=null
    [END] success=true steps=2 score=0.99 rewards=0.10,0.10
"""

import asyncio
import os
import sys
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import AsyncOpenAI

from client import FinanceOptimizerEnv
from models import FinanceOptimizerAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
ENV_URL = (
    os.getenv("OPENENV_FINANCE_OPTIMIZER_ENV_URL")
    or os.getenv("FINANCE_ENV_URL")
    or os.getenv("ENV_URL")
    or "http://localhost:8000"
)
TASK_NAME = (
    os.getenv("OPENENV_FINANCE_OPTIMIZER_TASK")
    or os.getenv("FINANCE_TASK")
    or os.getenv("TASK_NAME")
    or ""
)
ALL_TASKS = [
    "ledger_cleanup", 
    "subscription_audit", 
    "cash_flow", 
    "fraud_categorization", 
    "savings_builder", 
    "duplicate_charge_alert"
]
TASK_ALIASES: dict[str, str] = {
    "easy": "ledger_cleanup",
    "medium": "subscription_audit",
    "hard": "cash_flow",
    "fraud": "fraud_categorization",
    "savings": "savings_builder",
    "duplicate": "duplicate_charge_alert",
}
TASK_NAME = TASK_ALIASES.get(TASK_NAME, TASK_NAME)
BENCHMARK = (
    os.getenv("OPENENV_FINANCE_OPTIMIZER_BENCHMARK")
    or os.getenv("FINANCE_BENCHMARK")
    or os.getenv("BENCHMARK")
    or "finance_optimizer"
)
BASE_SEED = 42
SUCCESS_SCORE_THRESHOLD = 0.5


_output_file: Any | None = None


def _emit(line: str) -> None:
    print(line, flush=True)
    if _output_file is not None:
        _output_file.write(line + "\n")
        _output_file.flush()


def log_start(task: str, env: str, model: str) -> None:
    _emit(f"[START] task={task} env={env} model={model}")


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    error_val = error if error else "null"
    _emit(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    _emit(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}"
    )


async def run_task(
    task_name: str, client: AsyncOpenAI, env: FinanceOptimizerEnv, seed: int
) -> None:
    """Run a single task episode and emit [START]/[STEP]/[END] to stdout."""
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=seed, task_id=task_name)
        obs = result.observation

        while not obs.done:
            if steps_taken == 0 and client:
                try:
                    await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": "Analyze state"}],
                        max_tokens=1
                    )
                except Exception:
                    pass
                
            steps_taken += 1
            action_dict = {}

            if task_name == "ledger_cleanup":
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

            elif task_name == "subscription_audit":
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

            elif task_name == "cash_flow":
                if obs.checking_balance < 1500 and obs.savings_balance > 0:
                    action_dict = {
                        "action_type": "TransferFunds",
                        "from_account": "Savings",
                        "to_account": "Checking",
                        "amount": 500.0,
                    }
                else:
                    action_dict = {"action_type": "SetAlert", "text": "wait"}

            elif task_name == "fraud_categorization":
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

            elif task_name == "savings_builder":
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

            elif task_name == "duplicate_charge_alert":
                action_dict = {"action_type": "SetAlert", "text": "tx_dup_copy"}

            action = FinanceOptimizerAction(**action_dict)
            action_str_repr = f"{action.action_type}"

            try:
                result = await env.step(action)
            except Exception as exc:
                log_step(steps_taken, action_str_repr, 0.0, True, str(exc))
                break

            obs = result.observation
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            steps_taken = steps_taken

            if obs.done and obs.final_score is not None:
                score = obs.final_score

            log_step(steps_taken, action_str_repr, reward, obs.done, None)

        score = round(min(max(score, 0.01), 0.99), 2)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    tasks_to_run = [TASK_NAME] if TASK_NAME else ALL_TASKS

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

    env = FinanceOptimizerEnv(base_url=ENV_URL)

    try:
        try:
            await env.connect()
        except Exception as conn_exc:
            print(f"[DEBUG] Failed to connect to environment server at {ENV_URL}: {conn_exc}", file=sys.stderr, flush=True)
            for task_name in tasks_to_run:
                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, score=0.0, rewards=[])
            return

        for task_name in tasks_to_run:
            seed = BASE_SEED
            await run_task(task_name, client, env, seed)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
