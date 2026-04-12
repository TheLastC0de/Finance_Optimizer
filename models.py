"""
Data models for the Finance Optimizer Environment.

The finance optimizer environment helps users categorize bank ledgers,
audit subscriptions, and manage cash flow through three progressive tasks.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FinanceOptimizerAction(Action):
    """Action for the Finance Optimizer environment.

    Attributes:
        action_type: Type of action to perform.
        tx_id: Transaction ID for CategorizeTransaction.
        category: Category for CategorizeTransaction.
        vendor_name: Vendor name for CancelSubscription.
        from_account: Source account for TransferFunds.
        to_account: Destination account for TransferFunds.
        amount: Amount to transfer for TransferFunds.
        text: Text for SetAlert.
    """

    action_type: Literal["CategorizeTransaction", "CancelSubscription", "TransferFunds", "PayCreditCard", "SetAlert"] = Field(
        ..., description="Type of action to perform"
    )
    tx_id: Optional[str] = Field(None, description="Transaction ID for CategorizeTransaction")
    category: Optional[str] = Field(None, description="Category for CategorizeTransaction")
    vendor_name: Optional[str] = Field(None, description="Vendor name for CancelSubscription")
    from_account: Optional[str] = Field(None, description="Source account for TransferFunds or PayCreditCard")
    to_account: Optional[str] = Field(None, description="Destination account for TransferFunds")
    amount: Optional[float] = Field(None, description="Amount to transfer or pay")
    text: Optional[str] = Field(None, description="Text for SetAlert")


class FinanceOptimizerObservation(Observation):
    """Observation from the Finance Optimizer environment.

    Contains account state, transaction history, and episode metrics.

    Attributes:
        ledger: Last 60 days of transactions.
        subscriptions: Active recurring subscriptions.
        checking_balance: Checking account balance.
        savings_balance: Savings account balance.
        credit_card_balance: Current balance on high-interest credit card.
        credit_card_apr: Annual Percentage Rate for credit card debt.
        credit_score: FICO-style credit score (300-850).
        metadata: Task progress and scores.
        final_score: Final grader score in [0, 1]; only set at episode end.
    """

    ledger: List[Dict[str, Any]] = Field(default_factory=list, description="Last 60 days of transactions")
    subscriptions: List[Dict[str, Any]] = Field(default_factory=list, description="Active recurring subscriptions")
    checking_balance: float = Field(default=0.0, description="Checking account balance")
    savings_balance: float = Field(default=0.0, description="Savings account balance")
    credit_card_balance: float = Field(default=0.0, description="Credit card balance")
    credit_card_apr: float = Field(default=0.22, description="Credit card APR (e.g. 0.22 for 22%)")
    credit_score: int = Field(default=700, description="FICO-style credit score (300-850)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task progress and scores")
    final_score: Optional[float] = None
