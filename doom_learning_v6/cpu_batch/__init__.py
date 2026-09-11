"""Shared-graph CPU execution for independent v6 neural trajectories."""

__all__=['CpuBatchLane','MultiTrajectoryCpuExecutor','SharedCpuGraph']


def __getattr__(name):
    if name not in __all__:raise AttributeError(name)
    from . import backend
    return getattr(backend,name)
