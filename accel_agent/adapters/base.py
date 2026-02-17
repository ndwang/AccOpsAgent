"""Control system adapter abstract interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReadResult:
    value: float | str | list
    units: str
    timestamp: datetime
    alarm_status: str  # "OK", "MINOR", "MAJOR", "INVALID"
    tolerance: float = 0.0


class ControlSystemAdapter(ABC):
    """
    Abstract interface to the accelerator control system.
    Each facility implements this for their system (EPICS, TANGO, etc.).
    """

    @abstractmethod
    async def get(self, device_name: str) -> ReadResult:
        ...

    @abstractmethod
    async def set(self, device_name: str, value) -> None:
        ...

    @abstractmethod
    async def search_devices(
        self, pattern: str, subsystem: str | None = None
    ) -> list[dict]:
        ...

    @abstractmethod
    async def read_log(
        self,
        source: str,
        time_range: tuple,
        filters: dict | None = None,
    ) -> list[dict]:
        ...
