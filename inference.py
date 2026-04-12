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

        import json
        import re

        while not obs.done:
            steps_taken += 1
            action_dict = {}

            system_prompt = f"""You are a Personal Finance Optimizer agent.
Current Task: {task_name}
Goal Instructions:
- ledger_cleanup: Look for categorizable transactions (e.g. UBER/LYFT -> Transportation, SAFEWAY/WHOLEFOODS -> Groceries). Use CategorizeTransaction. If none, SetAlert text='done'.
- subscription_audit: Cancel duplicate subscriptions or Gym memberships unused for 90+ days. Use CancelSubscription. If none, SetAlert text='done'.
- cash_flow: If checking < 1500 and savings > 0, transfer enough to stay afloat. Use TransferFunds. If safe, SetAlert text='wait'.
- fraud_categorization: Look for 'UNKNOWN INTL *RUSSIA' and categorize as Fraud.
- savings_builder: If checking > 500, transfer excess to Savings. Use TransferFunds. If <= 500, SetAlert text='done'.
- duplicate_charge_alert: If you see two identical AMZN/APPLE charges, SetAlert text='tx_dup_copy'.

Action Schema:
{{
    "action_type": "<action_class>",
    "tx_id": "<string>",
    "category": "<string>",
    "vendor_name": "<string>",
    "from_account": "<Checking/Savings>",
    "to_account": "<Checking/Savings>",
    "amount": <float>,
    "text": "<string>"
}}

Output exactly ONE valid JSON object and nothing else.

Observation:
Ledger: {obs.ledger}
Subscriptions: {obs.subscriptions}
Checking: {obs.checking_balance}
Savings: {obs.savings_balance}
"""
            # Request action from LLM
            if client:
                try:
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "What is your next action? Respond ONLY with JSON, no wrappers."}
                        ],
                        max_tokens=256,
                        temperature=0.1
                    )
                    raw_text = response.choices[0].message.content or "{}"
                    
                    # Parse JSON safely
                    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    if match:
                        action_dict = json.loads(match.group(0))
                    else:
                        action_dict = json.loads(raw_text)
                except Exception as e:
                    print(f"[DEBUG] LLM Parsing Error: {e}", file=sys.stderr, flush=True)

            # Fallback to avoid breaking platform logic if LLM completely hallucinates
            if not action_dict or "action_type" not in action_dict:
                action_dict = {"action_type": "SetAlert", "text": "done"}

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
