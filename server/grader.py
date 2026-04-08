from typing import Any

def grade(action_dict: dict[str, Any], task_id: str, temperature: float = 0.0, seed: int = 42) -> float:
    """Evaluate an action against a given task ID."""
    try:
        from finance_optimizer.server.tasks import TASK_REGISTRY
    except ImportError:
        from server.tasks import TASK_REGISTRY

    if task_id not in TASK_REGISTRY:
        return 0.0
        
    score = 0.0
    action_type = action_dict.get("action_type", "")
    
    if task_id == "ledger_cleanup" and action_type == "CategorizeTransaction":
        score = 1.0
    elif task_id == "subscription_audit" and action_type == "CancelSubscription":
        score = 1.0
    elif task_id == "cash_flow" and action_type == "TransferFunds":
        score = 1.0
        
    return score
