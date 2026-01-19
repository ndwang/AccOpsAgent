"""Data models for diagnostics and parameters."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DiagnosticStatus(str, Enum):
    """Status of a diagnostic measurement."""

    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    UNKNOWN = "unknown"
    ERROR = "error"


class DiagnosticSnapshot(BaseModel):
    """Snapshot of a diagnostic measurement at a specific time.

    Attributes:
        timestamp: ISO format timestamp when measurement was taken
        diagnostic_name: Name of the diagnostic
        value: Measured value
        unit: Unit of measurement
        status: Status of the measurement (normal/warning/alarm)
        message: Optional message providing context
        metadata: Additional metadata about the measurement
    """

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    diagnostic_name: str
    value: float
    unit: str
    status: DiagnosticStatus = DiagnosticStatus.NORMAL
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if diagnostic is in a healthy state."""
        return self.status in (DiagnosticStatus.NORMAL, DiagnosticStatus.WARNING)

    def is_alarm(self) -> bool:
        """Check if diagnostic is in alarm state."""
        return self.status == DiagnosticStatus.ALARM


class ParameterValue(BaseModel):
    """Current value of a controllable parameter.

    Attributes:
        timestamp: ISO format timestamp when value was read
        parameter_name: Name of the parameter (knob)
        value: Current value
        unit: Unit of the value
        setpoint: Desired setpoint (may differ from actual value)
        metadata: Additional metadata
    """

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    parameter_name: str
    value: float
    unit: str
    setpoint: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_at_setpoint(self, tolerance: float = 1e-6) -> bool:
        """Check if current value matches setpoint within tolerance."""
        if self.setpoint is None:
            return True
        return abs(self.value - self.setpoint) <= tolerance


class ActionResult(BaseModel):
    """Result of executing a control action.

    Attributes:
        success: Whether the action succeeded
        action_type: Type of action performed
        timestamp: When the action was executed
        parameters: Parameters of the action
        message: Result message
        error: Error message if action failed
        diagnostics_snapshot: Diagnostics after action
    """

    success: bool
    action_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any]
    message: str
    error: Optional[str] = None
    diagnostics_snapshot: Optional[list[DiagnosticSnapshot]] = None
