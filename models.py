from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel

class TaskInfo(BaseModel):
    task_id: str
    difficulty: str
    description: str
    action_schema: Dict[str, Any]

class FinanceOptimizerAction(Action):
    action_type: Literal["CategorizeTransaction", "CancelSubscription", "TransferFunds", "SetAlert"] = Field(
        ..., description="Type of action to perform"
    )
    tx_id: Optional[str] = Field(None, description="Transaction ID for CategorizeTransaction")
    category: Optional[str] = Field(None, description="Category for CategorizeTransaction")
    vendor_name: Optional[str] = Field(None, description="Vendor name for CancelSubscription")
    from_account: Optional[str] = Field(None, description="Source account for TransferFunds")
    to_account: Optional[str] = Field(None, description="Destination account for TransferFunds")
    amount: Optional[float] = Field(None, description="Amount to transfer for TransferFunds")
    text: Optional[str] = Field(None, description="Text for SetAlert")
    model_config = {"extra": "allow"}


class FinanceOptimizerObservation(Observation):
    ledger: List[Dict[str, Any]] = Field(default_factory=list, description="Last 60 days of transactions")
    subscriptions: List[Dict[str, Any]] = Field(default_factory=list, description="Active recurring subscriptions")
    checking_balance: float = Field(default=0.0, description="Checking account balance")
    savings_balance: float = Field(default=0.0, description="Savings account balance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task progress and scores")
    model_config = {"extra": "allow"}
