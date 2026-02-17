"""TANGO adapter (facility implementation)."""

from accel_agent.adapters.base import ControlSystemAdapter, ReadResult

# TODO: Implement with PyTango for production.


class TangoAdapter(ControlSystemAdapter):
    """Placeholder for TANGO adapter implementation."""

    async def get(self, device_name: str) -> ReadResult:
        raise NotImplementedError("TANGO adapter not implemented")

    async def set(self, device_name: str, value) -> None:
        raise NotImplementedError("TANGO adapter not implemented")

    async def search_devices(
        self, pattern: str, subsystem: str | None = None
    ) -> list[dict]:
        raise NotImplementedError("TANGO adapter not implemented")

    async def read_log(
        self,
        source: str,
        time_range: tuple,
        filters: dict | None = None,
    ) -> list[dict]:
        raise NotImplementedError("TANGO adapter not implemented")
