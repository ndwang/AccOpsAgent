"""Mock adapter for development and testing."""

from datetime import datetime

from accel_agent.adapters.base import ControlSystemAdapter, ReadResult


class SimulatorAdapter(ControlSystemAdapter):
    def __init__(self):
        self._state: dict = {}

    async def get(self, device_name: str) -> ReadResult:
        return ReadResult(
            value=self._state.get(device_name, 0.0),
            units="mm",
            timestamp=datetime.now(),
            alarm_status="OK",
            tolerance=0.01,
        )

    async def set(self, device_name: str, value) -> None:
        self._state[device_name] = value

    async def search_devices(
        self, pattern: str, subsystem: str | None = None
    ) -> list[dict]:
        import re
        regex = re.compile(pattern.replace("*", ".*"), re.IGNORECASE)
        return [
            {"name": k, "description": "simulated"}
            for k in self._state
            if regex.match(k)
        ]

    async def read_log(
        self,
        source: str,
        time_range: tuple,
        filters: dict | None = None,
    ) -> list[dict]:
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "message": "Simulated log entry",
            }
        ]
