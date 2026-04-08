import pytest
from models import FinanceOptimizerAction
from server.finance_optimizer_environment import FinanceOptimizerEnvironment

@pytest.fixture
def env():
    return FinanceOptimizerEnvironment()

@pytest.mark.parametrize("task_id", ["ledger_cleanup", "subscription_audit", "cash_flow"])
def test_reset_returns_valid_observation(env, task_id):
    obs = env.reset(seed=42, task_id=task_id)
    assert obs.metadata.get("step") == 0
    assert len(obs.ledger) > 0
    assert not obs.done

@pytest.mark.parametrize("task_id", ["ledger_cleanup", "subscription_audit", "cash_flow"])
def test_five_steps(env, task_id):
    obs = env.reset(seed=42, task_id=task_id)
    for _ in range(5):
        action = FinanceOptimizerAction(action_type="SetAlert", text="noop")
        obs = env.step(action)
        assert obs.metadata.get("step") > 0
        assert obs.done is False
