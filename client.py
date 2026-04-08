# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Finance Optimizer Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FinanceOptimizerAction, FinanceOptimizerObservation


class FinanceOptimizerEnv(
    EnvClient[FinanceOptimizerAction, FinanceOptimizerObservation, State]
):
    """
    Client for the Finance Optimizer Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with FinanceOptimizerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(FinanceOptimizerAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = FinanceOptimizerEnv.from_docker_image("finance_optimizer-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(FinanceOptimizerAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FinanceOptimizerAction) -> Dict:
        """
        Convert FinanceOptimizerAction to JSON payload for step message.

        Args:
            action: FinanceOptimizerAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FinanceOptimizerObservation]:
        """
        Parse server response into StepResult[FinanceOptimizerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with FinanceOptimizerObservation
        """
        obs_data = payload.get("observation", {})
        observation = FinanceOptimizerObservation(
            ledger=obs_data.get("ledger", []),
            subscriptions=obs_data.get("subscriptions", []),
            checking_balance=obs_data.get("checking_balance", 0.0),
            savings_balance=obs_data.get("savings_balance", 0.0),
            metadata=obs_data.get("metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
