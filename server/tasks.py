from finance_optimizer.server.finance_optimizer_environment import FinanceOptimizerEnvironment

TASK_REGISTRY = {
    task["task_id"]: task for task in FinanceOptimizerEnvironment.TASKS
}
