"""Explicit execution backends for the v6 neural propagation step."""


class BackendError(RuntimeError):
    """A selected neural backend could not execute the requested interval."""


class CpuBackend:
    """Adapter preserving the reviewed native C++ kernel as the reference."""

    name='cpu'

    def __init__(self,brain):self.brain=brain
    def advance(self,steps):return self.brain._advance_cpu(steps)
    def sync_for_checkpoint(self):return None
    def restore_from_host(self):return None
    def metadata(self):return {'name':self.name,'build':self.brain.build}
    def close(self):return None


def create_backend(name,brain):
    if name=='cpu':return CpuBackend(brain)
    if name=='metal':
        from .metal.backend import MetalBackend
        return MetalBackend(brain)
    raise ValueError(f'Unknown neural backend: {name}')
