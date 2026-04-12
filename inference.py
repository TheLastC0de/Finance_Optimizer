"""
FinanceOptimizer Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    ENV_URL             URL of a running FinanceOptimizerEnv server (default: http://localhost:8000).
    TASK_NAME           Task to run: ledger_cleanup, subscription_audit, cash_flow, etc.

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
import json
import os
import re
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
MAX_CONVERSATION_HISTORY = 6  # Keep last N exchanges to avoid context overflow


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


# ─── Task-Specific Prompt Templates ───

TASK_PROMPTS = {
    "ledger_cleanup": """You are a financial transaction categorizer. Your job is to assign the correct spending category to uncategorized bank transactions.

VENDOR → CATEGORY MAPPING (use this exactly):
  Transportation: UBER *TRIP, LYFT *RIDE, BART *TRANSIT, LIME *SCOOTER
  Groceries: SAFEWAY #33, WHOLEFOODS, TRADER JOE, TARGET *GROC
  Dining: DOORDASH, GRUBHUB, STARBUCKS #12, CHIPOTLE #09
  Entertainment: AMC THEATERS, STEAM GAMES, SPOTIFY PREMIUM, TICKETMASTER
  Utilities: PG&E ELECTRIC, AT&T WIRELESS, COMCAST CABLE, WATER DEPT

RULES:
- Pick ONE uncategorized transaction and categorize it using CategorizeTransaction.
- Match the vendor name EXACTLY to the mapping above.
- When all transactions are categorized, use SetAlert with text="done".

Example:
{"action_type": "CategorizeTransaction", "tx_id": "tx_5", "category": "Dining"}""",

    "subscription_audit": """You are a subscription auditor. Your job is to identify wasteful subscriptions and cancel them.

RULES:
- Cancel subscriptions that are marked as "duplicate": true.
- Cancel subscriptions where "last_visit_days_ago" >= 90 (unused for 3+ months).
- Do NOT cancel subscriptions that are valid and in use (especially "Rent").
- Use CancelSubscription with the exact vendor_name.
- When all wasteful subscriptions are cancelled, use SetAlert with text="done".

Example:
{"action_type": "CancelSubscription", "vendor_name": "Netflix"}""",

    "cash_flow": """You are a cash flow manager. Your job is to prevent an overdraft by ensuring sufficient funds in Checking before rent is due.

RULES:
- Look at the Rent subscription's "due_in_days" field and its "cost".
- If checking_balance < rent cost, transfer funds from Savings to Checking.
- Transfer enough to cover rent with a safety margin.
- Use TransferFunds with from_account="Savings", to_account="Checking".
- If checking_balance is already sufficient, use SetAlert text="wait" to advance the day.

Example:
{"action_type": "TransferFunds", "from_account": "Savings", "to_account": "Checking", "amount": 1000.0}""",

    "fraud_categorization": """You are a fraud detection specialist. Your job is to identify suspicious transactions and flag them as Fraud.

RULES:
- Look for transactions with suspicious vendors like "UNKNOWN INTL *RUSSIA", "WIRE *OFFSHORE", or "CRYPTO *ANON".
- These have abnormally large negative amounts (typically -$3000 to -$8000).
- Categorize the fraudulent transaction using CategorizeTransaction with category="Fraud".
- Once flagged, use SetAlert text="done".

Example:
{"action_type": "CategorizeTransaction", "tx_id": "tx_fraud_99", "category": "Fraud"}""",

    "savings_builder": """You are a savings optimizer. Your job is to transfer excess funds from Checking to Savings while maintaining a minimum checking balance of $500.

RULES:
- Calculate: excess = checking_balance - 500.0
- If excess > 0, transfer it all to Savings in one action.
- Use TransferFunds with from_account="Checking", to_account="Savings".
- After transferring, use SetAlert text="done".

Example:
{"action_type": "TransferFunds", "from_account": "Checking", "to_account": "Savings", "amount": 1500.0}""",

    "debt_avalanche": """You are a debt optimization specialist. Your job is to pay off high-interest credit card debt using available funds while maintaining a safety net.

RULES:
- Credit card debt costs ~22% APR, while Savings only earns 1%.
- Priority: Pay off Credit Card as fast as possible.
- Minimum: Keep at least $500 in Checking for daily expenses.
- Use PayCreditCard with from_account ("Checking" or "Savings") and the amount.
- When debt is $0, use SetAlert text="done".

Example:
{"action_type": "PayCreditCard", "from_account": "Checking", "amount": 1000.0}""",

    "duplicate_charge_alert": """You are a charge auditor. Your job is to identify a duplicate charge in the ledger and alert the system.

RULES:
- Look for two transactions with the SAME vendor AND the SAME amount.
- The duplicate's ID will typically be "tx_dup_copy".
- Use SetAlert with text="tx_dup_copy" to flag it.
- Only one alert is needed.

Example:
{"action_type": "SetAlert", "text": "tx_dup_copy"}""",
}


def _build_observation_context(obs: Any, task_name: str) -> str:
    """Build a compact observation string, showing only relevant data for the task."""
    
    if task_name == "ledger_cleanup":
        # Only show uncategorized transactions
        uncategorized = [tx for tx in obs.ledger if tx.get("category") == "Uncategorized"]
        remaining = len(uncategorized)
        # Show first 10 to avoid token overload
        sample = uncategorized[:10]
        return (
            f"Uncategorized transactions remaining: {remaining}\n"
            f"Next batch:\n{json.dumps(sample, indent=2)}"
        )
    
    elif task_name == "subscription_audit":
        return (
            f"Subscriptions:\n{json.dumps(obs.subscriptions, indent=2)}\n"
            f"Checking: ${obs.checking_balance:.2f}"
        )
    
    elif task_name == "cash_flow":
        rent_sub = next((s for s in obs.subscriptions if s.get("type") == "housing"), None)
        return (
            f"Checking: ${obs.checking_balance:.2f}\n"
            f"Savings: ${obs.savings_balance:.2f}\n"
            f"Rent: {json.dumps(rent_sub)}"
        )
    
    elif task_name == "fraud_categorization":
        # Show transactions with abnormally large amounts
        suspicious = [tx for tx in obs.ledger if tx.get("amount", 0) < -2000]
        return (
            f"Suspicious transactions (amount < -$2000):\n"
            f"{json.dumps(suspicious, indent=2)}\n"
            f"Total transactions: {len(obs.ledger)}"
        )
    
    elif task_name == "savings_builder":
        return (
            f"Checking: ${obs.checking_balance:.2f}\n"
            f"Savings: ${obs.savings_balance:.2f}\n"
            f"Target minimum checking: $500.00\n"
            f"Excess available: ${max(0, obs.checking_balance - 500):.2f}"
        )
    
    elif task_name == "debt_avalanche":
        return (
            f"Checking: ${obs.checking_balance:.2f}\n"
            f"Savings: ${obs.savings_balance:.2f}\n"
            f"Credit Card Balance: ${obs.credit_card_balance:.2f}\n"
            f"Credit Card APR: {obs.credit_card_apr:.0%}\n"
            f"Target: Pay off all debt while keeping $500 in Checking."
        )

    elif task_name == "duplicate_charge_alert":
        # Group by (vendor, amount) to find duplicates
        from collections import Counter
        counts = Counter((tx["vendor"], tx["amount"]) for tx in obs.ledger)
        dupes = {k: v for k, v in counts.items() if v > 1}
        dupe_txs = [tx for tx in obs.ledger if (tx["vendor"], tx["amount"]) in dupes]
        return (
            f"Potential duplicate charges:\n{json.dumps(dupe_txs, indent=2)}"
        )
    
    return f"Ledger: {len(obs.ledger)} transactions\nChecking: ${obs.checking_balance:.2f}\nSavings: ${obs.savings_balance:.2f}"


async def run_task(
    task_name: str, client: AsyncOpenAI, env: FinanceOptimizerEnv, seed: int
) -> None:
    """Run a single task episode with LLM-driven decision making."""
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=seed, task_id=task_name)
        obs = result.observation

        # Maintain conversation history for multi-turn reasoning
        system_prompt = TASK_PROMPTS.get(task_name, "You are a financial agent. Output valid JSON actions.")
        conversation_history: list[dict[str, str]] = []

        while not obs.done:
            steps_taken += 1
            action_dict = {}

            # Build compact observation
            obs_context = _build_observation_context(obs, task_name)
            
            user_msg = f"Step {steps_taken}. Current state:\n{obs_context}\n\nRespond with exactly ONE JSON action object."
            
            if client:
                try:
                    # Build messages: system + trimmed history + current
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # Add recent conversation history (trimmed to avoid overflow)
                    if conversation_history:
                        messages.extend(conversation_history[-MAX_CONVERSATION_HISTORY:])
                    
                    messages.append({"role": "user", "content": user_msg})
                    
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=256,
                        temperature=0.05,
                    )
                    raw_text = response.choices[0].message.content or "{}"
                    
                    # Parse JSON from LLM response
                    match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
                    if match:
                        action_dict = json.loads(match.group(0))
                    else:
                        action_dict = json.loads(raw_text)
                    
                    # Record this exchange in conversation history
                    conversation_history.append({"role": "user", "content": user_msg})
                    conversation_history.append({"role": "assistant", "content": raw_text})
                    
                except Exception as e:
                    print(f"[DEBUG] LLM error at step {steps_taken}: {e}", file=sys.stderr, flush=True)

            # Fallback if LLM failed
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

            if obs.done and obs.final_score is not None:
                score = obs.final_score

            log_step(steps_taken, action_str_repr, reward, obs.done, None)
            
            # Safety: break if stuck in a loop
            if steps_taken >= 80:
                break

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
