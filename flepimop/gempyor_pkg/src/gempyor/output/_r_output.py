__all__: tuple[str, ...] = ()

from ._base import OutputABC


class ROutput(OutputABC):
    def get_chains(self):
        return super().get_chains()
