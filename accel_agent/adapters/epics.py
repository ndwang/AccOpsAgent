"""EPICS / Channel Access adapter (facility implementation)."""

from accel_agent.adapters.base import ControlSystemAdapter, ReadResult

# TODO: Implement with pyepics, pcaspy, or pvAccess for production.


class EpicsAdapter(ControlSystemAdapter):
    """Placeholder for EPICS adapter implementation."""

    async def get(self, device_name: str) -> ReadResult:
        raise NotImplementedError("EPICS adapter not implemented")

    async def set(self, device_name: str, value) -> None:
        raise NotImplementedError("EPICS adapter not implemented")

    async def search_devices(
        self, pattern: str, subsystem: str | None = None
    ) -> list[dict]:
        raise NotImplementedError("EPICS adapter not implemented")

    async def read_log(
        self,
        source: str,
        time_range: tuple,
        filters: dict | None = None,
    ) -> list[dict]:
        raise NotImplementedError("EPICS adapter not implemented")
