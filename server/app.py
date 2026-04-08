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

try:
    from finance_optimizer.models import FinanceOptimizerAction, FinanceOptimizerObservation, TaskInfo
    from finance_optimizer.server.finance_optimizer_environment import FinanceOptimizerEnvironment
except ModuleNotFoundError:
    from models import FinanceOptimizerAction, FinanceOptimizerObservation, TaskInfo
    from server.finance_optimizer_environment import FinanceOptimizerEnvironment


# Create the app with web interface and README integration
app = create_app(
    FinanceOptimizerEnvironment,
    FinanceOptimizerAction,
    FinanceOptimizerObservation,
    env_name="finance_optimizer",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

from typing import Any, List
from fastapi import HTTPException

try:
    from finance_optimizer.server.tasks import TASK_REGISTRY
    from finance_optimizer.server.grader import grade
except ModuleNotFoundError:
    from server.tasks import TASK_REGISTRY
    from server.grader import grade

@app.get("/tasks")
def list_tasks() -> List[TaskInfo]:
    return [
        TaskInfo(
            task_id=task["id"],
            difficulty=task["difficulty"],
            description=task["description"],
            action_schema=FinanceOptimizerAction.model_json_schema()
        )
        for task in TASK_REGISTRY.values()
    ]

@app.post("/grader")
def get_grader_score(task_id: str, action: FinanceOptimizerAction) -> dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    
    score = grade(action.model_dump(), task_id)
        
    return {
        "task_id": task_id,
        "score": score,
        "passed": 1 if score > 0.5 else 0,
        "total": 1,
        "metric": "finance_optimizer_alignment",
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
