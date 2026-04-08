from finance_optimizer.server.finance_optimizer_environment import FinanceOptimizerEnvironment

TASK_REGISTRY = {
    task["id"]: task for task in FinanceOptimizerEnvironment.TASKS
}
